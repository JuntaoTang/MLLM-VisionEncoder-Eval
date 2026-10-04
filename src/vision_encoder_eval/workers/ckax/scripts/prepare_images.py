# -*- coding: utf-8 -*-
"""
Prepare the calibration image set (diverse sources)
=============================================================
Extract diverse images from multiple sources on the server:
1. LMUData TSV files (MMBench, ScienceQA, MME, VizWiz, etc.)
2. OCR-VQA HuggingFace dataset (book covers)
3. Standalone image directories (COCO, ImageNet, etc.)

Sampling is stratified across benchmarks for visual diversity, so the set
mixes natural images, charts, scientific diagrams, documents and real-world
photos instead of one homogeneous source.

Usage:
    python scripts/prepare_images.py --source diverse --num_images 2000
    python scripts/prepare_images.py --source lmu --num_images 1000 --per_benchmark 200
    python scripts/prepare_images.py --source ocr-vqa --num_images 1000
    python scripts/prepare_images.py --check
"""
import os
import sys
import json
import argparse
import base64
import io
import random
import csv
import sys as _sys
try:
    csv.field_size_limit(_sys.maxsize)   # TSV base64 images can exceed the
except OverflowError:                    # default 128 KB limit; on Windows
    csv.field_size_limit(2 ** 31 - 1)    # C long is 32-bit
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vision_encoder_eval.workers.ckax.config import PATHS


# ============================================================
# Source Discovery
# ============================================================
def _fallback_dirs():
    """Optional extra image directories, OFF by default.

    The package points at nothing outside its own folder: the optional probes
    be hard-coded to a particular server layout (~/data/coco, /data/imagenet,
    ...).  Set LOCAL_IMAGE_FALLBACK to a path-separated list to re-enable them.
    `run.sh images` never needs them (it passes --source diverse)."""
    return [d for d in os.environ.get("LOCAL_IMAGE_FALLBACK", "").split(os.pathsep)
            if d]


def check_available_sources():
    """Check what image sources are available on the server."""
    print("=" * 60)
    print("  Checking available image sources...")
    print("=" * 60)

    sources = {}

    # 1. Optional extra image directories (off by default; see _fallback_dirs)
    common_dirs = _fallback_dirs()
    for d in common_dirs:
        if os.path.isdir(d):
            n_files = len([f for f in os.listdir(d) if f.endswith(('.jpg', '.png', '.jpeg'))])
            sources[d] = {"type": "directory", "count": n_files}
            print(f"  [FOUND] {d} ({n_files} images)")

    # 2. Check OCR-VQA dataset
    ocr_vqa_path = PATHS["ocr_vqa_cache"]
    if os.path.isdir(ocr_vqa_path):
        total_size = sum(
            f.stat().st_size for f in Path(ocr_vqa_path).rglob("*") if f.is_file()
        )
        sources["ocr-vqa"] = {
            "type": "huggingface",
            "path": ocr_vqa_path,
            "size_gb": total_size / 1e9,
        }
        print(f"  [FOUND] OCR-VQA: {ocr_vqa_path} ({total_size/1e9:.1f} GB)")

    # 3. Check LMUData
    lmu_path = PATHS["lmu_data"]
    if os.path.isdir(lmu_path):
        tsv_files = sorted(Path(lmu_path).rglob("*.tsv"))
        sources["lmu"] = {
            "type": "tsv",
            "path": lmu_path,
            "tsv_count": len(tsv_files),
            "tsv_files": [str(f) for f in tsv_files],
        }
        print(f"  [FOUND] LMUData: {lmu_path} ({len(tsv_files)} TSV files)")
        for tsv in tsv_files:
            size_mb = tsv.stat().st_size / 1e6
            print(f"    - {tsv.name} ({size_mb:.1f} MB)")

    if not sources:
        print("  [WARN] No image sources found!")

    print("=" * 60)
    return sources


# ============================================================
# OCR-VQA Extraction
# ============================================================
def extract_from_ocr_vqa(num_images, output_dir, seed=42):
    """Extract images from OCR-VQA dataset (book covers).

    Loads from LOCAL cache only -- server has no internet.
    Cache location: ~/.cache/huggingface/datasets/howard-hou___ocr-vqa/
    """
    from PIL import Image

    print(f"\n[OCR-VQA] Extracting {num_images} book cover images (local cache)...")

    # Force offline mode: server has NO internet access
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    dataset = None

    # Strategy 1: load_dataset with offline env vars (uses local Arrow cache)
    try:
        from datasets import load_dataset
        dataset = load_dataset("howard-hou/ocr-vqa", split="train", streaming=True)
        print("  Loaded via HuggingFace datasets (offline/streaming)")
    except Exception as e1:
        print(f"  [WARN] Streaming load failed: {e1}")
        # Strategy 2: non-streaming from local cache
        try:
            dataset = load_dataset("howard-hou/ocr-vqa", split="train")
            print("  Loaded via HuggingFace datasets (offline/full)")
        except Exception as e2:
            print(f"  [WARN] Full load failed: {e2}")

    # Strategy 3: read Arrow files directly from cache directory
    if dataset is None:
        ocr_cache = PATHS["ocr_vqa_cache"]
        arrow_files = sorted(Path(ocr_cache).rglob("*.arrow")) if os.path.isdir(ocr_cache) else []
        if arrow_files:
            try:
                import pyarrow.ipc as ipc
                print(f"  Loading {len(arrow_files)} Arrow files directly from cache...")
                # Build a simple iterator over Arrow record batches
                class ArrowDatasetIterator:
                    def __init__(self, arrow_files):
                        self.arrow_files = arrow_files
                    def __iter__(self):
                        for af in self.arrow_files:
                            try:
                                reader = ipc.open_stream(str(af))
                                for batch in reader:
                                    d = batch.to_pydict()
                                    n = len(list(d.values())[0]) if d else 0
                                    for i in range(n):
                                        yield {k: v[i] for k, v in d.items()}
                            except Exception:
                                try:
                                    table = ipc.open_file(str(af)).read_all()
                                    d = table.to_pydict()
                                    n = len(list(d.values())[0]) if d else 0
                                    for i in range(n):
                                        yield {k: v[i] for k, v in d.items()}
                                except Exception:
                                    continue
                dataset = ArrowDatasetIterator(arrow_files)
                print(f"  Loaded via direct Arrow file reading")
            except ImportError:
                print(f"  [ERROR] pyarrow not available for direct cache reading")

    if dataset is None:
        print(f"  [ERROR] Cannot load OCR-VQA from local cache!")
        print(f"  Cache path: {PATHS['ocr_vqa_cache']}")
        return []

    os.makedirs(output_dir, exist_ok=True)
    saved = []

    for i, sample in enumerate(tqdm(dataset, desc="  OCR-VQA", unit="img", total=num_images)):
        if len(saved) >= num_images:
            break
        if "image" in sample and sample["image"] is not None:
            img = sample["image"]
            if isinstance(img, Image.Image):
                img_path = os.path.join(output_dir, f"ocr_vqa_{i:06d}.jpg")
                img.convert("RGB").save(img_path, quality=95)
                saved.append({
                    "path": img_path,
                    "source": "ocr-vqa",
                    "index": i,
                })
                if len(saved) % 200 == 0:
                    print(f"    Saved {len(saved)}/{num_images}...")

    print(f"  [OCR-VQA] Done: {len(saved)} images")
    return saved


# ============================================================
# LMUData Extraction (Stratified)
# ============================================================
def _find_image_column(fieldnames):
    """Find the image data column in TSV fieldnames."""
    candidates = ["image", "img", "image_base64", "image_path", "img_path"]
    for col in candidates:
        if col in fieldnames:
            return col
    return None


def _decode_image(img_data, lmu_dir):
    """Decode image from base64 string or file path. Returns PIL Image or None."""
    from PIL import Image

    if not img_data or (isinstance(img_data, str) and len(img_data) < 10):
        return None

    # Try base64 decode (most common in VLMEvalKit TSV files)
    if isinstance(img_data, str) and len(img_data) > 100:
        try:
            img_bytes = base64.b64decode(img_data)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            if img.width >= 32 and img.height >= 32:
                return img
        except Exception:
            pass

    # Try as file path (relative to LMUData dir, then to the image root)
    if isinstance(img_data, str) and len(img_data) < 500:
        img_root = os.environ.get("IMG_DIR", os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "images_diverse"))
        for candidate in [img_data, os.path.join(lmu_dir, img_data),
                          os.path.join(img_root, img_data)]:
            if os.path.isfile(candidate):
                try:
                    img = Image.open(candidate).convert("RGB")
                    if img.width >= 32 and img.height >= 32:
                        return img
                except Exception:
                    pass

    return None



def scan_lmu():
    """
    Quick scan of LMUData: count available images per benchmark.
    Only reads TSV structure, does NOT decode or save any images.
    """
    lmu_path = PATHS["lmu_data"]
    print(f"\n{'='*64}")
    print(f"  LMUData Image Scan (read-only, no extraction)")
    print(f"  Path: {lmu_path}")
    print(f"{'='*64}")

    if not os.path.isdir(lmu_path):
        print(f"  [ERROR] LMUData directory not found: {lmu_path}")
        return {}

    tsv_files = sorted(Path(lmu_path).rglob("*.tsv"))
    if not tsv_files:
        print(f"  [ERROR] No TSV files found")
        return {}

    results = {}
    total_images = 0

    print(f"\n  {'Benchmark':<30} {'Rows':>8} {'WithImg':>8} {'Size(MB)':>10}")
    print(f"  {'-'*60}")

    for tsv_file in tsv_files:
        bench_name = tsv_file.stem
        size_mb = tsv_file.stat().st_size / 1e6
        total_rows = 0
        valid_images = 0

        try:
            with open(tsv_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t")
                if reader.fieldnames is None:
                    print(f"  {bench_name:<30} {'?':>8} {'?':>8} {size_mb:>10.1f}  [EMPTY]")
                    continue

                img_col = _find_image_column(reader.fieldnames)
                if img_col is None:
                    print(f"  {bench_name:<30} {'?':>8} {'0':>8} {size_mb:>10.1f}  [no image col]")
                    continue

                for row in reader:
                    total_rows += 1
                    img_data = row.get(img_col, "")
                    if img_data and (len(str(img_data)) > 50
                                 or (5 < len(str(img_data)) < 500
                                     and ("/" in str(img_data)
                                          or "." in str(img_data)))):
                        valid_images += 1

            results[bench_name] = {
                "total_rows": total_rows,
                "valid_images": valid_images,
                "size_mb": size_mb,
                "img_col": img_col,
            }
            total_images += valid_images
            print(f"  {bench_name:<30} {total_rows:>8} {valid_images:>8} {size_mb:>10.1f}")

        except Exception as e:
            print(f"  {bench_name:<30} {'ERR':>8} {'ERR':>8} {size_mb:>10.1f}  [{e}]")

    print(f"  {'-'*60}")
    print(f"  {'TOTAL':<30} {'':>8} {total_images:>8}")

    # Also check OCR-VQA
    ocr_path = PATHS["ocr_vqa_cache"]
    if os.path.isdir(ocr_path):
        arrow_files = list(Path(ocr_path).rglob("*.arrow"))
        print(f"\n  [OCR-VQA] Cache found: {ocr_path}")
        print(f"    Arrow files: {len(arrow_files)}")
        print(f"    (OCR-VQA has ~400K+ images in full dataset)")

    print(f"\n  Recommendation:")
    if total_images > 0:
        print(f"    LMUData total available: {total_images} images")
        print(f"    Use --all to extract everything, or --num_images N to limit")
        print(f"    Feature extraction time estimate: ~{total_images * 37 * 0.03 / 60:.0f} min")
        print(f"      (37 tokenizers x {total_images} images x ~0.03s/image)")
    print(f"{'='*64}")

    return results


def extract_from_lmu(num_images, output_dir, seed=42, per_benchmark=None):
    """
    Extract images from LMUData TSV files with stratified sampling.

    Each TSV file represents a different benchmark (MMBench, ScienceQA, etc.).
    Benchmarks are sampled equally so that the set spans natural images,
    charts, scientific diagrams, documents, etc.
    """
    from PIL import Image

    lmu_path = PATHS["lmu_data"]
    print(f"\n[LMUData] Stratified extraction from {lmu_path}")

    # Find all TSV files
    tsv_files = sorted(Path(lmu_path).rglob("*.tsv"))
    if not tsv_files:
        print(f"  [ERROR] No TSV files found in {lmu_path}")
        return []

    print(f"  Found {len(tsv_files)} TSV files:")
    for tsv in tsv_files:
        print(f"    - {tsv.name} ({tsv.stat().st_size / 1e6:.1f} MB)")

    # Allocate images per benchmark (stratified / equal)
    n_benchmarks = len(tsv_files)
    extract_all = (num_images <= 0)  # 0 or negative means "extract everything"

    if extract_all:
        # Extract all available images from every benchmark
        allocation = {tsv: 999999 for tsv in tsv_files}
        print(f"\n  Mode: EXTRACT ALL available images from {n_benchmarks} benchmarks")
    else:
        if per_benchmark is None:
            per_benchmark = max(1, num_images // n_benchmarks)

        allocation = {}
        remaining = num_images
        for i, tsv in enumerate(tsv_files):
            if i == n_benchmarks - 1:
                allocation[tsv] = remaining  # last benchmark gets the remainder
            else:
                allocation[tsv] = min(per_benchmark, remaining)
                remaining -= allocation[tsv]

    print(f"\n  Allocation ({num_images} total, ~{per_benchmark} per benchmark):")
    for tsv, count in allocation.items():
        print(f"    {tsv.name}: {count} images")

    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)
    saved = []

    for tsv_file, target_count in allocation.items():
        if target_count <= 0:
            continue

        bench_name = tsv_file.stem
        print(f"\n  Processing: {bench_name} (target: {target_count})")

        bench_saved = 0

        try:
            # --- Pass 1: scan for valid image rows ---
            valid_row_indices = []
            total_rows = 0

            with open(tsv_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t")
                if reader.fieldnames is None:
                    print(f"    [SKIP] Empty file")
                    continue

                img_col = _find_image_column(reader.fieldnames)
                if img_col is None:
                    print(f"    [SKIP] No image column. Columns: {list(reader.fieldnames)[:8]}")
                    continue

                for row_idx, row in enumerate(tqdm(reader, desc=f"    Scanning {bench_name}", unit="row", leave=False)):
                    total_rows += 1
                    img_data = row.get(img_col, "")
                    if img_data and (len(str(img_data)) > 50
                                 or (5 < len(str(img_data)) < 500
                                     and ("/" in str(img_data)
                                          or "." in str(img_data)))):
                        valid_row_indices.append(row_idx)

            print(f"    Rows: {total_rows}, with image data: {len(valid_row_indices)}, col: '{img_col}'")

            if not valid_row_indices:
                print(f"    [SKIP] No valid images found")
                continue

            # Stratified sample from valid rows
            if len(valid_row_indices) > target_count:
                selected_rows = set(random.sample(valid_row_indices, target_count))
            else:
                selected_rows = set(valid_row_indices)
                if len(valid_row_indices) < target_count:
                    print(f"    [INFO] Only {len(valid_row_indices)} images available (target: {target_count})")

            # --- Pass 2: extract selected images ---
            with open(tsv_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t")

                extract_pbar = tqdm(total=len(selected_rows), desc=f"    Extracting {bench_name}", unit="img", leave=False)
                for row_idx, row in enumerate(reader):
                    if row_idx not in selected_rows:
                        continue

                    img_data = row.get(img_col, "")
                    img = _decode_image(img_data, lmu_path)
                    if img is None:
                        continue

                    img_name = f"lmu_{bench_name}_{row_idx:06d}.jpg"
                    img_path = os.path.join(output_dir, img_name)
                    img.save(img_path, quality=95)

                    saved.append({
                        "path": img_path,
                        "source": f"lmu/{bench_name}",
                        "benchmark": bench_name,
                        "index": row_idx,
                    })
                    bench_saved += 1
                    extract_pbar.update(1)
                extract_pbar.close()

        except Exception as e:
            print(f"    [ERROR] {e}")
            continue

        print(f"    Saved: {bench_saved}/{target_count}")

    print(f"\n  [LMUData] Total: {len(saved)} images from {n_benchmarks} benchmarks")
    return saved


# ============================================================
# Diverse Extraction (Multi-Source)
# ============================================================
def extract_diverse(total_images, output_dir, seed=42, ocr_ratio=0.3, per_benchmark=None):
    """
    Extract diverse images from all available sources.

    Combines LMUData (diverse benchmarks: natural images, charts, science
    diagrams, documents) with OCR-VQA (book covers) for maximum visual
    diversity in probe training.
    """
    print(f"\n{'='*60}")
    print(f"  Diverse Image Extraction")
    print(f"  Target: {total_images} images | OCR ratio: {ocr_ratio}")
    print(f"{'='*60}")

    all_saved = []

    # Calculate split
    n_ocr = int(total_images * ocr_ratio)
    n_lmu = total_images - n_ocr

    # 1. LMUData (more diverse, extract first)
    lmu_path = PATHS["lmu_data"]
    if os.path.isdir(lmu_path) and n_lmu > 0:
        lmu_saved = extract_from_lmu(n_lmu, output_dir, seed, per_benchmark)
        all_saved.extend(lmu_saved)
        # If LMUData gave fewer than expected, give the surplus to OCR-VQA
        shortfall = n_lmu - len(lmu_saved)
        if shortfall > 0:
            n_ocr += shortfall
            print(f"  [INFO] LMUData shortfall: {shortfall}, adding to OCR-VQA quota")

    # 2. OCR-VQA (book covers)
    ocr_vqa_path = PATHS["ocr_vqa_cache"]
    if os.path.isdir(ocr_vqa_path) and n_ocr > 0:
        ocr_saved = extract_from_ocr_vqa(n_ocr, output_dir, seed)
        all_saved.extend(ocr_saved)

    # 3. Fallback: standalone image directories
    if len(all_saved) < total_images:
        common_dirs = _fallback_dirs()
        for d in common_dirs:
            if not os.path.isdir(d) or len(all_saved) >= total_images:
                continue
            import shutil
            imgs = sorted([
                f for f in os.listdir(d)
                if f.lower().endswith(('.jpg', '.png', '.jpeg'))
            ])
            needed = total_images - len(all_saved)
            random.seed(seed)
            selected = random.sample(imgs, min(needed, len(imgs)))
            dir_name = os.path.basename(d)
            for img_name in selected:
                src = os.path.join(d, img_name)
                dst = os.path.join(output_dir, f"dir_{dir_name}_{img_name}")
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                all_saved.append({"path": dst, "source": f"dir/{dir_name}"})
            print(f"  [DIR] Copied {len(selected)} images from {d}")

    # Save metadata
    save_metadata(output_dir, all_saved)

    # Summary
    source_counts = {}
    for rec in all_saved:
        src = rec.get("source", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1

    print(f"\n{'='*60}")
    print(f"  Diverse extraction complete: {len(all_saved)} images")
    print(f"  Source breakdown:")
    for src, count in sorted(source_counts.items()):
        print(f"    {src}: {count}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")

    return all_saved


# ============================================================
# Metadata
# ============================================================
def save_metadata(output_dir, image_records):
    """Save image source metadata as JSON for reproducibility."""
    os.makedirs(output_dir, exist_ok=True)

    metadata = {
        "total_images": len(image_records),
        "sources": {},
        "images": [],
    }

    for rec in image_records:
        src = rec.get("source", "unknown")
        metadata["sources"][src] = metadata["sources"].get(src, 0) + 1
        metadata["images"].append({
            "filename": os.path.basename(rec["path"]),
            "source": src,
            "benchmark": rec.get("benchmark", ""),
        })

    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved: {meta_path}")

    # Compatibility with run.sh auto-detection
    source_path = os.path.join(output_dir, "image_source.txt")
    with open(source_path, "w") as f:
        f.write(output_dir)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Prepare diverse images for feature extraction"
    )
    parser.add_argument(
        "--source",
        choices=["auto", "ocr-vqa", "lmu", "diverse", "dir"],
        default="auto",
        help="Image source: auto, ocr-vqa, lmu, diverse (all sources), dir",
    )
    parser.add_argument("--image_dir", type=str, default=None,
                        help="Input image directory (for --source dir)")
    parser.add_argument("--num_images", type=int, default=1000,
                        help="Total number of images to extract")
    parser.add_argument("--per_benchmark", type=int, default=None,
                        help="Images per LMUData benchmark (default: auto-divide)")
    parser.add_argument("--ocr_ratio", type=float, default=0.3,
                        help="Fraction from OCR-VQA in diverse mode (default: 0.3)")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check", action="store_true",
                        help="Just check what sources are available")
    parser.add_argument("--scan", action="store_true",
                        help="Scan LMUData: count available images per benchmark (no extraction)")
    parser.add_argument("--all", action="store_true", dest="extract_all",
                        help="Extract ALL available images (ignore --num_images)")
    args = parser.parse_args()

    if args.check:
        check_available_sources()
        return

    if args.scan:
        scan_lmu()
        return

    # --all means extract everything
    if args.extract_all:
        args.num_images = 0  # 0 signals "no limit" to extraction functions

    output_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "images"
    )

    # --- Source: existing directory ---
    if args.source == "dir" and args.image_dir:
        if os.path.isdir(args.image_dir):
            imgs = [
                f for f in os.listdir(args.image_dir)
                if f.lower().endswith(('.jpg', '.png', '.jpeg'))
            ]
            print(f"  Image directory: {args.image_dir} ({len(imgs)} images)")
            os.makedirs(output_dir, exist_ok=True)
            save_metadata(output_dir, [
                {"path": os.path.join(args.image_dir, f), "source": "dir"}
                for f in imgs
            ])
        else:
            print(f"  [ERROR] Directory not found: {args.image_dir}")
        return

    # --- Source: auto-detect ---
    if args.source == "auto":
        sources = check_available_sources()

        available = []
        if "lmu" in sources:
            available.append("lmu")
        if "ocr-vqa" in sources:
            available.append("ocr-vqa")

        if len(available) >= 2:
            args.source = "diverse"
            print(f"\n  Auto-selected: diverse (found: {', '.join(available)})")
        elif "lmu" in sources:
            args.source = "lmu"
        elif "ocr-vqa" in sources:
            args.source = "ocr-vqa"
        else:
            for path, info in sources.items():
                if info["type"] == "directory" and info.get("count", 0) >= args.num_images:
                    print(f"\n  Using existing directory: {path}")
                    os.makedirs(output_dir, exist_ok=True)
                    save_metadata(output_dir, [])
                    return
            print("\n  [ERROR] No image source available!")
            sys.exit(1)

    # --- Extract ---
    if args.source == "diverse":
        extract_diverse(
            args.num_images, output_dir, args.seed,
            args.ocr_ratio, args.per_benchmark,
        )
    elif args.source == "ocr-vqa":
        saved = extract_from_ocr_vqa(args.num_images, output_dir, args.seed)
        save_metadata(output_dir, saved)
    elif args.source == "lmu":
        saved = extract_from_lmu(
            args.num_images, output_dir, args.seed, args.per_benchmark
        )
        save_metadata(output_dir, saved)

    print(f"\n  Done! Images in: {output_dir}")
    print(f"  Next: IMG_DIR={output_dir} bash run.sh features")


if __name__ == "__main__":
    main()
