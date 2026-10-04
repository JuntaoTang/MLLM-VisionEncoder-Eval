"""GPU KNN benchmark for all 70 exported 200-shot feature matrices."""

from vision_encoder_eval.core.runtime import asset_path
import csv
import gc
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np

BASE = os.environ.get(
    "FEATURE_EXPORT_ROOT",
    str(Path(__file__).resolve().parents[1] / "feature_exports"),
)
MANIFEST = os.path.join(BASE, "export_manifest.json")
OUT = asset_path('runtime', 'knn_cache/all_exported_200shot_gpu')
JSON_OUT = os.path.join(OUT, "all_models_5_10_20_45_95_195shot.json")
CSV_OUT = os.path.join(OUT, "all_models_5_10_20_45_95_195shot.csv")
MD_OUT = os.path.join(OUT, "all_models_5_10_20_45_95_195shot.md")
IJ_PATH = os.environ.get(
    "IJEPA_HELPER",
    str(Path(__file__).resolve().parent / "models" / "ijepa.py"),
)

# Exact per-image forward FLOPs used by the original family-specific scripts.
# Keys are the tokenizer ids from export_manifest.json.
SPECS = {
    "clip_openai__l14": ("OpenAI-CLIP-L14-224", 162025537536),
    "dino_vitb16": ("DINO-ViT-B16", 35126120448),
    "dino_vitb8": ("DINO-ViT-B8", 156295139328),
    "dino_vits16": ("DINO-ViT-S16", 9196996608),
    "dino_vits8": ("DINO-ViT-S8", 44810717184),
    "dinov2_base": ("DINOv2-Base", 303147436032),
    "dinov2_giant": ("DINOv2-Giant", 3566685917184),
    "dinov2_large": ("DINOv2-Large", 1013607653376),
    "dinov2_small": ("DINOv2-Small", 93393478656),
    "dinov3_vitl16": ("DINOv3-ViT-L16-224", 123107377152),
    "toklip_l_384": ("TokLIP-L-384", 515850633216),
    "toklip_s_256": ("TokLIP-S-256", 219074789376),
    "uniar_bsq": ("UniAR-BSQ", 251336693271),
    "unitok_attn": ("UniTok-Attn", 195070066688),
    "vilau_256": ("VILA-U-256", 161490141184),
    "eupe_convnext_b": ("EUPE-ConvNeXt-B", 40107638784),
    "eupe_vit_b": ("EUPE-ViT-B", 46393233408),
    "eupe_vit_s": ("EUPE-ViT-S", 12282513408),
    "eupe_vit_t": ("EUPE-ViT-T", 3412730880),
    "ijepa_vith14": ("I-JEPA-ViT-H-14", 166622658560),
    "mc1_b16_224_2.5b": ("MetaCLIP1-B16-224-2.5B", 35126906880),
    "mc1_b16_224_400m": ("MetaCLIP1-B16-224-400M", 35126906880),
    "mc1_b32_224_2.5b": ("MetaCLIP1-B32-224-2.5B", 8817623040),
    "mc1_b32_224_400m": ("MetaCLIP1-B32-224-400M", 8817623040),
    "mc1_g14_224_2.5b": ("MetaCLIP1-G14-224-2.5B", 967496032256),
    "mc1_h14_224_2.5b": ("MetaCLIP1-H14-224-2.5B", 334590279680),
    "mc1_h14_224_v1.2": ("MetaCLIP1-H14-224-v1.2", 334590279680),
    "mc1_l14_224_2.5b": ("MetaCLIP1-L14-224-2.5B", 162025537536),
    "mc1_l14_224_400m": ("MetaCLIP1-L14-224-400M", 162025537536),
    "mc2_b16_224": ("MetaCLIP2-B16-224", 35126906880),
    "mc2_b16_384": ("MetaCLIP2-B16-384", 110967951360),
    "mc2_b32_224": ("MetaCLIP2-B32-224", 8817623040),
    "mc2_b32_224_mt5": ("MetaCLIP2-B32-224-mT5", 8817623040),
    "mc2_b32_384": ("MetaCLIP2-B32-384", 26086379520),
    "mc2_g14_224": ("MetaCLIP2-G14-224", 967496032256),
    "mc2_g14_378": ("MetaCLIP2-G14-378", 2858452253696),
    "mc2_h14_378": ("MetaCLIP2-H14-378", 1006962882560),
    "mc2_l14_224": ("MetaCLIP2-L14-224", 162025537536),
    "mc2_m16_224": ("MetaCLIP2-M16-224", 123108425728),
    "mc2_m16_224_mt5": ("MetaCLIP2-M16-224-mT5", 123108425728),
    "mc2_m16_384": ("MetaCLIP2-M16-384", 382131601408),
    "mc2_s16_224": ("MetaCLIP2-S16-224", 9197291520),
    "mc2_s16_224_mt5": ("MetaCLIP2-S16-224-mT5", 9197291520),
    "mc2_s16_384": ("MetaCLIP2-S16-384", 30980229120),
    "pe_core_b16_224": ("PE-Core-B16-224", 35127693312),
    "pe_core_g14_448": ("PE-Core-G14-448", 4113241780224),
    "pe_lang_l14_448": ("PE-Lang-L14-448", 723595132928),
    "pixio_vitb16": ("PixIO-ViT-B16", 17858691072),
    "pixio_vith16": ("PixIO-ViT-H16", 122824949760),
    "pixio_vitl16": ("PixIO-ViT-L16", 60508078080),
    "raev2_dinov3l_k7": ("RAEv2-DINOv3-L-K7", 158105042948),
    "siglip2_b16_224": ("SigLIP2-B16-224", 35127300096),
    "siglip2_b16_256": ("SigLIP2-B16-256", 46394413056),
    "siglip2_b16_384": ("SigLIP2-B16-384", 110968344576),
    "siglip2_b16_512": ("SigLIP2-B16-512", 214055424000),
    "siglip2_b32_256": ("SigLIP2-B32-256", 11500425216),
    "siglip2_g16_256": ("SigLIP2-gopt-16-256", 598926409728),
    "siglip2_g16_384": ("SigLIP2-gopt-16-384", 1390045544448),
    "siglip2_l16_256": ("SigLIP2-L16-256", 162120433664),
    "siglip2_l16_384": ("SigLIP2-L16-384", 382132649984),
    "siglip2_l16_512": ("SigLIP2-L16-512", 723972620288),
    "siglip2_sm14_224": ("SigLIP2-SO400M-14-224", 219857241600),
    "siglip2_sm14_384": ("SigLIP2-SO400M-14-378", 667454432256),
    "siglip2_sm16_256": ("SigLIP2-SO400M-16-256", 219963409920),
    "siglip2_sm16_384": ("SigLIP2-SO400M-16-384", 516818880000),
    "siglip2_sm16_512": ("SigLIP2-SO400M-16-512", 975223604736),
    "webssl_dino1b_full2b_224": ("WebSSL-DINO-1B-224", 728132468736),
    "webssl_mae1b_full2b_224": ("WebSSL-MAE-1B-224", 582584624640),
    "webssl_mae300m_full2b_224": ("WebSSL-MAE-300M-224", 119304279040),
    "webssl_mae3b_full2b_224": ("WebSSL-MAE-3B-224", 1514407885824),
}


def load_ijepa_module():
    spec = importlib.util.spec_from_file_location("shared_knn", IJ_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_outputs(rows, metadata):
    os.makedirs(OUT, exist_ok=True)
    payload = {"metadata": metadata, "results": rows}
    tmp = JSON_OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, JSON_OUT)

    fields = ["Model", "Shot", "Train", "Top1", "TFLOPs", "Tokenizer", "FeatureDim", "SearchSeconds"]
    with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    with open(MD_OUT, "w", encoding="utf-8") as f:
        f.write("| Model | Shot | Train | Top1 | TFLOPs |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(f"| {r['Model']} | {r['Shot']} | {r['Train']} | {r['Top1']} | {r['TFLOPs']:.6f} |\n")


def main():
    os.makedirs(OUT, exist_ok=True)
    shared = load_ijepa_module()
    features0, labels, source_indices, _, _ = shared.load_exported_arrays()
    del features0
    protocol = shared.build_protocol(labels, source_indices)
    query_rows = protocol["query_indices"]

    with open(MANIFEST, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    entries = sorted(manifest["entries"], key=lambda x: x["rank"])
    tokens = {e["tokenizer"] for e in entries}
    if tokens != set(SPECS):
        raise RuntimeError(f"FLOPs map mismatch: missing={tokens-set(SPECS)}, extra={set(SPECS)-tokens}")

    rows = []
    start_all = time.perf_counter()
    metadata = {
        "protocol": "seed42, 195 train pool + fixed 5 query/class, nested shots",
        "shots": list(shared.TRAIN_SHOTS),
        "k": 20,
        "temperature": shared.TEMPERATURE,
        "normalization": "L2",
        "faiss": "GPU IndexFlatIP float32 cuda:0",
        "feature_root": BASE,
        "model_count": len(entries),
    }

    for pos, entry in enumerate(entries, 1):
        token = entry["tokenizer"]
        display, per_image_flops = SPECS[token]
        path = os.path.join(BASE, entry["feature_file"])
        feats = np.load(path, mmap_mode="r")
        if feats.shape != (200000, entry["feature_dim"]):
            raise ValueError(f"{token}: unexpected shape {feats.shape}")

        model_start = time.perf_counter()
        query = shared.normalized_rows(feats, query_rows)
        query_labels = np.asarray(labels[query_rows], dtype=np.int64)
        print(f"[{pos:02d}/{len(entries)}] {display} dim={feats.shape[1]}", flush=True)

        for shot in shared.TRAIN_SHOTS:
            db_rows = protocol["train_indices_by_shot"][str(shot)]
            db = shared.normalized_rows(feats, db_rows)
            db_labels = np.asarray(labels[db_rows], dtype=np.int64)
            result, search_sec, _ = shared.exact_knn(
                db, db_labels, query, query_labels, use_gpu=True, gpu_device=0
            )
            train = len(db_rows)
            search_flops = 2 * len(query_rows) * train * feats.shape[1]
            total_tflops = ((train + len(query_rows)) * per_image_flops + search_flops) / 1e12
            rows.append({
                "Model": display,
                "Shot": shot,
                "Train": train,
                "Top1": f"{result['top20']['accuracy']:.2f}%",
                "TFLOPs": total_tflops,
                "Tokenizer": token,
                "FeatureDim": int(feats.shape[1]),
                "SearchSeconds": search_sec,
            })
            print(f"  shot={shot:3d} top1={result['top20']['accuracy']:6.2f}% TFLOPs={total_tflops:.6f}", flush=True)
            del db, db_labels
            gc.collect()

        print(f"  model_seconds={time.perf_counter()-model_start:.3f}", flush=True)
        metadata["elapsed_seconds"] = time.perf_counter() - start_all
        metadata["completed_models"] = pos
        write_outputs(rows, metadata)
        del query, query_labels, feats
        gc.collect()

    metadata["elapsed_seconds"] = time.perf_counter() - start_all
    metadata["completed_models"] = len(entries)
    write_outputs(rows, metadata)
    print(f"DONE models={len(entries)} rows={len(rows)} seconds={metadata['elapsed_seconds']:.3f}")
    print(JSON_OUT)
    print(CSV_OUT)
    print(MD_OUT)


if __name__ == "__main__":
    main()


