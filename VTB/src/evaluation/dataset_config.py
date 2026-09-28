"""Parse eval dataset specs and prepare subsampled LMUData caches."""

from __future__ import annotations

import csv
import os
import shutil
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from src.evaluation.lmu_data import (
    parse_image_path_field,
    resolve_image_path,
    serialize_image_path_field,
)

VTB_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVAL_DATASETS_YAML = os.path.join(VTB_ROOT, "configs", "eval_datasets.yaml")


def _configure_csv_limits() -> None:
    """LMUData TSVs (MMMU/MMBench) embed large base64 image fields."""
    max_int = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_int)
            break
        except OverflowError:
            max_int = int(max_int / 10)


_configure_csv_limits()


@dataclass(frozen=True)
class DatasetEvalSpec:
    name: str
    max_samples: int | None = None
    # When set with max_samples, draw a seeded random subset (not TSV head).
    sample_seed: int | None = None


@lru_cache(maxsize=1)
def load_eval_dataset_catalog() -> dict[str, str]:
    """Return dataset name -> eval type (mcq/vqa/yes_no/caption) from eval_datasets.yaml."""
    if not os.path.isfile(EVAL_DATASETS_YAML):
        return {}
    with open(EVAL_DATASETS_YAML, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return {
        str(entry["name"]): str(entry["type"])
        for entry in data.get("available", [])
        if entry.get("name") and entry.get("type")
    }


def eval_dataset_type(dataset_name: str) -> str | None:
    return load_eval_dataset_catalog().get(dataset_name)


def discover_local_datasets(lmudata_dir: str) -> list[str]:
    root = Path(lmudata_dir)
    if not root.is_dir():
        return []
    return sorted(p.stem for p in root.glob("*.tsv"))


def parse_eval_dataset_specs(eval_cfg: dict[str, Any]) -> list[DatasetEvalSpec]:
    """Parse eval.datasets from runtime config.

    Supported forms:
      datasets: [MMMU_DEV_VAL, VQAv2_VAL]
      datasets:
        MMMU_DEV_VAL: 100          # shorthand max_samples
        VQAv2_VAL:
          max_samples: 50
      max_samples: 200             # global default when per-dataset unset
      sample_seed: 42              # seeded random subsample (ignored for tsv_head)
      sampling: vlmevalkit|tsv_head
    """
    raw = eval_cfg.get("datasets")
    default_max = eval_cfg.get("max_samples")
    sampling = str(eval_cfg.get("sampling") or "vlmevalkit").strip().lower()
    seed_raw = eval_cfg.get("sample_seed", 42)
    default_seed = None if sampling == "tsv_head" else int(seed_raw)

    if raw is None:
        return [DatasetEvalSpec("VQAv2_VAL", default_max, default_seed)]

    if isinstance(raw, list):
        return [DatasetEvalSpec(str(name), default_max, default_seed) for name in raw]

    if isinstance(raw, dict):
        specs: list[DatasetEvalSpec] = []
        for name, opts in raw.items():
            max_samples = default_max
            sample_seed = default_seed
            if isinstance(opts, bool) and not opts:
                continue
            if isinstance(opts, int):
                max_samples = opts
            elif isinstance(opts, dict):
                max_samples = opts.get("max_samples", default_max)
                if "sample_seed" in opts:
                    sample_seed = int(opts["sample_seed"])
            specs.append(DatasetEvalSpec(str(name), max_samples, sample_seed))
        if not specs:
            raise ValueError("eval.datasets dict is empty")
        return specs

    raise ValueError(f"Unsupported eval.datasets type: {type(raw)!r}")


def validate_dataset_specs(specs: list[DatasetEvalSpec], lmudata_dir: str) -> None:
    available = set(discover_local_datasets(lmudata_dir))
    missing = [spec.name for spec in specs if spec.name not in available]
    if missing:
        hint = ", ".join(sorted(available)) or "(none found)"
        raise FileNotFoundError(
            f"LMUData TSV not found for: {', '.join(missing)} under {lmudata_dir}. "
            f"Available: {hint}"
        )


def _fix_tsv_image_paths(tsv_path: str, source_dir: str, dataset_name: str) -> None:
    """Rewrite stale absolute image_path entries to the active LMUData tree.

    MMMU (and similar) store image_path as a stringified list, e.g. ``['1_1.jpg']``.
    Those must be parsed before joining under ``images/<dataset>/`` — otherwise the
    cell becomes a broken path like ``.../MMMU_DEV_VAL/['1_1.jpg']``.
    """
    image_dir = os.path.join(source_dir, "images", dataset_name)
    with open(tsv_path, newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin, delimiter="\t")
        if not reader.fieldnames:
            return
        rows = list(reader)

    changed = False
    for row in rows:
        raw = row.get("image_path") or ""
        if not raw:
            continue
        parts = parse_image_path_field(raw)
        if not parts:
            continue
        # Keep short relative names as relative list/scalar so VLMEval dump_image
        # still pairs them with embedded base64 ``image`` cells.
        if all(not os.path.isabs(p) and "/" not in p and "\\" not in p for p in parts):
            fixed = serialize_image_path_field(parts, force_list=len(parts) != 1 or str(raw).strip().startswith("["))
            if fixed != raw.strip().strip('"'):
                row["image_path"] = fixed
                changed = True
            continue

        resolved = [resolve_image_path(p, image_dir) for p in parts]
        fixed = serialize_image_path_field(
            resolved, force_list=len(resolved) != 1 or str(raw).strip().startswith("[")
        )
        if fixed != raw.strip().strip('"'):
            row["image_path"] = fixed
            changed = True

    if not changed:
        return

    with open(tsv_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(
            fout, fieldnames=reader.fieldnames, delimiter="\t", quoting=csv.QUOTE_MINIMAL
        )
        writer.writeheader()
        writer.writerows(rows)


def _resolve_shared_image_cells(
    rows: list[dict[str, str]],
    image_map: dict[str, str] | None = None,
) -> None:
    """Inline VLMEvalKit shared-image refs so subsampled TSVs stay self-contained.

    Official TSVs may store a short index in ``image`` that points at another row's
    base64. Random subsample drops those targets and triggers
    ``assert idx in image_map`` inside ImageBaseDataset.
    """
    if not rows or "image" not in rows[0]:
        return
    index_key = "index" if "index" in rows[0] else ("Index" if "Index" in rows[0] else None)
    if index_key is None:
        return
    if image_map is None:
        image_map = {str(r.get(index_key, "")): str(r.get("image") or "") for r in rows}
    for row in rows:
        img = str(row.get("image") or "")
        seen: set[str] = set()
        while img and len(img) <= 64 and img not in seen:
            seen.add(img)
            nxt = image_map.get(img)
            if not nxt or nxt == img:
                break
            img = nxt
        row["image"] = img


def _subsample_tsv(
    src_tsv: str,
    dst_tsv: str,
    max_samples: int,
    sample_seed: int | None = None,
) -> int:
    """Take up to max_samples rows. With sample_seed, shuffle then take; else head.

    MME is paired by image_path (2 questions/image); sampling preserves pairs.
    """
    import random
    from collections import defaultdict

    with open(src_tsv, newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError(f"Empty or invalid TSV: {src_tsv}")
        fieldnames = list(reader.fieldnames)

        # Load full table when ``image`` may use index→base64 sharing, or for MME pairs.
        needs_full = "image" in fieldnames or (
            "image_path" in fieldnames and "category" in fieldnames
        )
        is_mme_like = (
            "image_path" in fieldnames
            and "category" in fieldnames
            and "question" in fieldnames
            and "answer" in fieldnames
        )
        if sample_seed is None and not needs_full:
            rows: list[dict[str, str]] = []
            for row in reader:
                rows.append(row)
                if len(rows) >= max_samples:
                    break
        else:
            all_rows = list(reader)
            index_key = "index" if "index" in fieldnames else ("Index" if "Index" in fieldnames else None)
            image_map = None
            if "image" in fieldnames and index_key is not None:
                image_map = {
                    str(r.get(index_key, "")): str(r.get("image") or "") for r in all_rows
                }

            if is_mme_like and sample_seed is not None and len(all_rows) > max_samples:
                # Sample whole image_path groups so yes/no pairs stay intact.
                groups: dict[str, list[dict[str, str]]] = defaultdict(list)
                for r in all_rows:
                    groups[str(r.get("image_path") or "")].append(r)
                keys = list(groups.keys())
                rng = random.Random(int(sample_seed))
                rng.shuffle(keys)
                rows = []
                for k in keys:
                    grp = groups[k]
                    if len(rows) + len(grp) > max_samples and rows:
                        break
                    rows.extend(grp)
                rows.sort(key=lambda r: str(r.get("index", r.get("Index", ""))))
            elif sample_seed is None:
                rows = all_rows[:max_samples]
            else:
                rng = random.Random(int(sample_seed))
                if len(all_rows) <= max_samples:
                    rows = all_rows
                else:
                    rows = rng.sample(all_rows, max_samples)
                    rows.sort(key=lambda r: str(r.get("index", r.get("Index", ""))))
            if image_map is not None:
                _resolve_shared_image_cells(rows, image_map=image_map)

    from src.evaluation.mcq_utils import expand_options_columns_on_rows

    fieldnames = expand_options_columns_on_rows(rows, fieldnames)

    with open(dst_tsv, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(
            fout, fieldnames=fieldnames, delimiter="\t", quoting=csv.QUOTE_MINIMAL
        )
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _expand_mcq_options_tsv(tsv_path: str) -> None:
    """Rewrite TSV in place so ``options`` becomes A/B/C/... columns when missing."""
    from src.evaluation.mcq_utils import expand_options_columns_on_rows

    with open(tsv_path, newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin, delimiter="\t")
        if not reader.fieldnames:
            return
        fieldnames = list(reader.fieldnames)
        if "options" not in fieldnames:
            return
        if any(letter in fieldnames for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
            return
        rows = list(reader)

    new_fields = expand_options_columns_on_rows(rows, fieldnames)
    if new_fields == fieldnames:
        return
    with open(tsv_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(
            fout, fieldnames=new_fields, delimiter="\t", quoting=csv.QUOTE_MINIMAL
        )
        writer.writeheader()
        writer.writerows(rows)


# VLMEvalKit dataset ids whose on-disk LMUData TSV may differ only by case
# (e.g. GQA_TestDev_Balanced vs GQA_TESTDEV_BALANCED).
_LMUDATA_TSV_ALIASES: dict[str, tuple[str, ...]] = {
    "GQA_TestDev_Balanced": ("GQA_TESTDEV_BALANCED",),
    "GQA_TestDev_All": ("GQA_TESTDEV_ALL",),
    "GQA_Test_Balanced": ("GQA_TEST_BALANCED",),
    "GQA_Test_All": ("GQA_TEST_ALL",),
    "GQA_Val_Balanced": ("GQA_VAL_BALANCED",),
    "GQA_Val_All": ("GQA_VAL_ALL",),
}


def resolve_lmudata_tsv(source_dir: str, dataset_name: str) -> str:
    """Resolve ``{dataset}.tsv`` under LMUData, with case / alias fallback."""
    exact = os.path.join(source_dir, f"{dataset_name}.tsv")
    if os.path.isfile(exact):
        return exact
    for alt in _LMUDATA_TSV_ALIASES.get(dataset_name, ()):
        path = os.path.join(source_dir, f"{alt}.tsv")
        if os.path.isfile(path):
            return path
    if os.path.isdir(source_dir):
        want = f"{dataset_name}.tsv".lower()
        for name in os.listdir(source_dir):
            if name.lower() == want and os.path.isfile(os.path.join(source_dir, name)):
                return os.path.join(source_dir, name)
    raise FileNotFoundError(
        f"No such file or directory: '{exact}' "
        f"(also tried aliases {_LMUDATA_TSV_ALIASES.get(dataset_name, ())})"
    )


def ensure_lmudata_dataset_alias(source_dir: str, dataset_name: str) -> str:
    """Ensure ``source_dir/{dataset}.tsv`` exists (symlink to alias if needed)."""
    exact = os.path.join(source_dir, f"{dataset_name}.tsv")
    if os.path.isfile(exact):
        return exact
    resolved = resolve_lmudata_tsv(source_dir, dataset_name)
    if os.path.abspath(resolved) == os.path.abspath(exact):
        return exact
    try:
        if os.path.lexists(exact):
            os.remove(exact)
        os.symlink(os.path.abspath(resolved), exact)
    except OSError:
        # Another worker may create the alias concurrently; re-check.
        if not os.path.isfile(exact):
            raise
    return exact


def prepare_dataset_tsv(
    source_dir: str,
    cache_dir: str,
    spec: DatasetEvalSpec,
) -> tuple[str, int | None]:
    """Materialize one dataset TSV (optionally subsampled) under cache_dir."""
    os.makedirs(cache_dir, exist_ok=True)
    src_tsv = resolve_lmudata_tsv(source_dir, spec.name)
    # Keep the canonical VLMEvalKit filename in the cache/view.
    dst_tsv = os.path.join(cache_dir, f"{spec.name}.tsv")
    ensure_lmudata_dataset_alias(source_dir, spec.name)

    if spec.max_samples is None:
        if os.path.abspath(src_tsv) != os.path.abspath(dst_tsv):
            shutil.copy2(src_tsv, dst_tsv)
        _fix_tsv_image_paths(dst_tsv, source_dir, spec.name)
        _expand_mcq_options_tsv(dst_tsv)
        return dst_tsv, None

    count = _subsample_tsv(
        src_tsv, dst_tsv, int(spec.max_samples), sample_seed=spec.sample_seed
    )
    _fix_tsv_image_paths(dst_tsv, source_dir, spec.name)
    _expand_mcq_options_tsv(dst_tsv)
    return dst_tsv, count


def ensure_lmudata_image_paths(source_dir: str, dataset_names: list[str]) -> None:
    """Rewrite stale image_path columns in local LMUData TSVs."""
    for name in dataset_names:
        try:
            tsv_path = ensure_lmudata_dataset_alias(source_dir, name)
        except FileNotFoundError:
            continue
        _fix_tsv_image_paths(tsv_path, source_dir, name)


def ensure_lmudata_images_link(source_dir: str, cache_dir: str) -> None:
    src_images = os.path.join(source_dir, "images")
    dst_images = os.path.join(cache_dir, "images")
    if os.path.lexists(dst_images):
        return
    if not os.path.isdir(src_images):
        raise FileNotFoundError(f"LMUData images directory not found: {src_images}")
    try:
        os.symlink(os.path.abspath(src_images), dst_images)
    except FileExistsError:
        return


def format_dataset_plan(specs: list[DatasetEvalSpec]) -> str:
    parts = []
    for spec in specs:
        if spec.max_samples is None:
            parts.append(f"{spec.name}(all)")
        elif spec.sample_seed is not None:
            parts.append(f"{spec.name}(n={spec.max_samples},seed={spec.sample_seed})")
        else:
            parts.append(f"{spec.name}(n={spec.max_samples},head)")
    return ", ".join(parts)
