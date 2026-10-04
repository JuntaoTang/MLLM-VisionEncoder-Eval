"""I-JEPA KNN evaluation on the exported ImageNet 200-shot feature set.

The split and KNN protocol intentionally match the existing benchmark:
- random.Random(42), shuffled independently per class
- one nested train pool and a fixed 5-query/class split
- L2-normalized float32 features
- exact FAISS IndexFlatIP search (cosine after normalization)
- temperature-weighted voting
"""

from vision_encoder_eval.core.runtime import asset_path
import argparse
import gc
import json
import os
import random
import time

import faiss
import numpy as np


# =============================================================================
# Data and output paths
# =============================================================================
DATA_ROOT = asset_path('runtime', 'vae_linear_probing_200shot_features_fp32')
FEATURE_FILE = os.path.join(DATA_ROOT, "features", "020_ijepa_vith14.npy")
LABEL_FILE = os.path.join(DATA_ROOT, "labels.npy")
SOURCE_INDEX_FILE = os.path.join(DATA_ROOT, "source_indices.npy")
EXPORT_MANIFEST = os.path.join(DATA_ROOT, "export_manifest.json")

RESULT_ROOT = asset_path('runtime', 'ijepa_knn/vae_linear_probing_200shot')
PROTOCOL_JSON = os.path.join(RESULT_ROOT, "ijepa_195pool_5query_seed42_protocol.json")
SUMMARY_JSON = os.path.join(RESULT_ROOT, "ijepa_model_x_shot_summary.json")

MODEL_NAME = "I-JEPA-ViT-H-14"
FEATURE_REPRESENTATION = (
    "spatial mean of the normalized RAEv2 I-JEPA-H/14 K=1 tokenizer latent"
)


# =============================================================================
# Fixed protocol settings
# =============================================================================
NUM_CLASSES = 1000
AVAILABLE_PER_CLASS = 200
TRAIN_POOL_PER_CLASS = 195
QUERY_PER_CLASS = 5
TRAIN_SHOTS = (5, 10, 20, 45, 95, 195)
K_VALUES = (1, 5, 10, 20)
TEMPERATURE = 0.07
SEED = 42

# Architecture values retained from the original ijepa.py for analytical FLOPs.
IMAGE_SIZE = 224
PATCH_SIZE = 14
EMBED_DIM = 1280
DEPTH = 32
MLP_RATIO = 4.0


def load_exported_arrays():
    """Open the shared exported arrays without loading the 1 GB feature file eagerly."""
    for path in (FEATURE_FILE, LABEL_FILE, SOURCE_INDEX_FILE, EXPORT_MANIFEST):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    features = np.load(FEATURE_FILE, mmap_mode="r")
    labels = np.load(LABEL_FILE, mmap_mode="r")
    source_indices = np.load(SOURCE_INDEX_FILE, mmap_mode="r")

    expected_rows = NUM_CLASSES * AVAILABLE_PER_CLASS
    if features.shape != (expected_rows, EMBED_DIM):
        raise ValueError(
            f"feature shape {features.shape} != {(expected_rows, EMBED_DIM)}"
        )
    if labels.shape != (expected_rows,):
        raise ValueError(f"label shape {labels.shape} != {(expected_rows,)}")
    if source_indices.shape != (expected_rows,):
        raise ValueError(
            f"source index shape {source_indices.shape} != {(expected_rows,)}"
        )
    if features.dtype != np.float32:
        raise ValueError(f"features must be float32, got {features.dtype}")

    counts = np.bincount(np.asarray(labels), minlength=NUM_CLASSES)
    if counts.shape[0] != NUM_CLASSES or not np.all(counts == AVAILABLE_PER_CLASS):
        raise ValueError(
            "labels must contain exactly 200 examples for each of 1000 classes"
        )
    if not np.all(source_indices[1:] > source_indices[:-1]):
        raise ValueError("source_indices.npy must be strictly increasing")

    with open(EXPORT_MANIFEST, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    matches = [
        x for x in manifest.get("entries", [])
        if x.get("feature_file") == "features/020_ijepa_vith14.npy"
    ]
    if len(matches) != 1:
        raise ValueError("I-JEPA entry is missing or duplicated in export_manifest.json")

    return features, labels, source_indices, manifest, matches[0]


def build_protocol(labels, source_indices):
    """Reproduce the old seeded, nested-shot protocol on 200 examples/class."""
    rng = random.Random(SEED)
    train_pool_by_class = {}
    query_by_class = {}
    train_source_indices_by_class = {}
    query_source_indices_by_class = {}
    train_indices_by_shot = {str(shot): [] for shot in TRAIN_SHOTS}
    query_indices = []

    for class_id in range(NUM_CLASSES):
        rows = np.flatnonzero(labels == class_id).tolist()
        if len(rows) != AVAILABLE_PER_CLASS:
            raise ValueError(
                f"class {class_id}: {len(rows)} rows, expected {AVAILABLE_PER_CLASS}"
            )
        rng.shuffle(rows)
        pool = rows[:TRAIN_POOL_PER_CLASS]
        query = rows[TRAIN_POOL_PER_CLASS:]

        train_pool_by_class[str(class_id)] = pool
        query_by_class[str(class_id)] = query
        train_source_indices_by_class[str(class_id)] = (
            np.asarray(source_indices[pool], dtype=np.int64).tolist()
        )
        query_source_indices_by_class[str(class_id)] = (
            np.asarray(source_indices[query], dtype=np.int64).tolist()
        )
        query_indices.extend(query)
        for shot in TRAIN_SHOTS:
            train_indices_by_shot[str(shot)].extend(pool[:shot])

    protocol = {
        "seed": SEED,
        "dataset": DATA_ROOT,
        "feature_file": FEATURE_FILE,
        "labels_file": LABEL_FILE,
        "source_indices_file": SOURCE_INDEX_FILE,
        "num_classes": NUM_CLASSES,
        "available_per_class": AVAILABLE_PER_CLASS,
        "train_pool_per_class": TRAIN_POOL_PER_CLASS,
        "query_per_class": QUERY_PER_CLASS,
        "train_shots": list(TRAIN_SHOTS),
        "num_train_pool": NUM_CLASSES * TRAIN_POOL_PER_CLASS,
        "num_query": NUM_CLASSES * QUERY_PER_CLASS,
        "train_pool_indices": [
            row
            for class_id in range(NUM_CLASSES)
            for row in train_pool_by_class[str(class_id)]
        ],
        "query_indices": query_indices,
        "train_indices_by_shot": train_indices_by_shot,
        "train_pool_by_class": train_pool_by_class,
        "query_by_class": query_by_class,
        "train_source_indices_by_class": train_source_indices_by_class,
        "query_source_indices_by_class": query_source_indices_by_class,
        "split_definition": (
            "For each class, shuffle its 200 exported rows with one shared "
            "random.Random(42) stream; first 195 are the nested train pool and "
            "last 5 are the fixed query set."
        ),
    }

    train_set = set(protocol["train_pool_indices"])
    query_set = set(query_indices)
    if train_set & query_set:
        raise RuntimeError("train/query overlap")
    if len(train_set) != NUM_CLASSES * TRAIN_POOL_PER_CLASS:
        raise RuntimeError("unexpected number of unique training rows")
    if len(query_set) != NUM_CLASSES * QUERY_PER_CLASS:
        raise RuntimeError("unexpected number of unique query rows")

    return protocol


def save_protocol(protocol):
    os.makedirs(RESULT_ROOT, exist_ok=True)
    with open(PROTOCOL_JSON, "w", encoding="utf-8") as f:
        json.dump(protocol, f, indent=2)


def normalized_rows(features, rows):
    """Materialize selected rows as contiguous float32 and L2-normalize them."""
    array = np.ascontiguousarray(features[np.asarray(rows, dtype=np.int64)])
    if array.dtype != np.float32:
        array = array.astype(np.float32)
    faiss.normalize_L2(array)
    norms = np.linalg.norm(array, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise RuntimeError(
            f"L2 normalization failed: min={norms.min()}, max={norms.max()}"
        )
    return array


def exact_knn(
    db_features,
    db_labels,
    query_features,
    query_labels,
    use_gpu=True,
    gpu_device=0,
):
    """Run exact cosine KNN on GPU by default, preserving float32 semantics."""
    cpu_index = faiss.IndexFlatIP(db_features.shape[1])
    gpu_resources = None

    if use_gpu:
        gpu_count = faiss.get_num_gpus()
        if gpu_count <= gpu_device:
            raise RuntimeError(
                f"requested GPU {gpu_device}, but FAISS sees {gpu_count} GPU(s)"
            )
        # Keep the database vectors in float32, exactly as in the CPU protocol.
        gpu_resources = faiss.StandardGpuResources()
        gpu_resources.setTempMemory(512 * 1024 * 1024)
        index = faiss.index_cpu_to_gpu(gpu_resources, gpu_device, cpu_index)
        backend = f"FAISS GPU IndexFlatIP (cuda:{gpu_device}, float32)"
    else:
        index = cpu_index
        backend = "FAISS CPU IndexFlatIP"

    index.add(db_features)

    start = time.perf_counter()
    similarities, neighbours = index.search(query_features, max(K_VALUES))
    search_seconds = time.perf_counter() - start

    results = {}
    for k in K_VALUES:
        labels = db_labels[neighbours[:, :k]]
        scores = similarities[:, :k]
        weights = np.exp(
            (scores - scores.max(axis=1, keepdims=True)) / TEMPERATURE
        )
        votes = np.zeros((len(query_features), NUM_CLASSES), dtype=np.float32)
        query_rows = np.arange(len(query_features))[:, None]
        np.add.at(
            votes,
            (np.broadcast_to(query_rows, labels.shape), labels),
            weights,
        )
        predictions = votes.argmax(axis=1)
        correct = int(np.count_nonzero(predictions == query_labels))
        results[f"top{k}"] = {
            "accuracy": 100.0 * correct / len(query_labels),
            "correct": correct,
            "total": int(len(query_labels)),
        }
    # gpu_resources must stay alive until search and result transfer finish.
    return results, search_seconds, backend


def encoder_flops_per_image():
    num_patches = (IMAGE_SIZE // PATCH_SIZE) ** 2
    d = EMBED_DIM
    mlp_dim = int(d * MLP_RATIO)
    patch = num_patches * 3 * PATCH_SIZE * PATCH_SIZE * d
    block = (
        3 * num_patches * d * d
        + 2 * num_patches * num_patches * d
        + num_patches * d * d
        + 2 * num_patches * d * mlp_dim
    )
    return patch + DEPTH * block


def run(validate_only=False, use_gpu=True, gpu_device=0):
    features, labels, source_indices, manifest, entry = load_exported_arrays()
    protocol = build_protocol(labels, source_indices)
    save_protocol(protocol)

    print("=" * 100)
    print("I-JEPA exported 200-shot feature KNN")
    print("=" * 100)
    print("feature file :", FEATURE_FILE)
    print("feature shape:", features.shape)
    print("labels       :", LABEL_FILE)
    print("source index :", SOURCE_INDEX_FILE)
    print("shots        :", TRAIN_SHOTS)
    print("query/class  :", QUERY_PER_CLASS)
    print("seed         :", SEED)
    print("protocol     :", PROTOCOL_JSON)
    print("L2 normalize : True")
    backend = (
        f"GPU IndexFlatIP (cuda:{gpu_device}, float32)"
        if use_gpu else "CPU IndexFlatIP"
    )
    print("FAISS index  :", backend)
    print("temperature  :", TEMPERATURE)

    if validate_only:
        print("Validation complete; benchmark was not run.")
        return {}

    query_rows = protocol["query_indices"]
    query_features = normalized_rows(features, query_rows)
    query_labels = np.asarray(labels[query_rows], dtype=np.int64)

    all_results = {}
    per_image_flops = encoder_flops_per_image()

    for shot in TRAIN_SHOTS:
        print("\n" + "=" * 100)
        print(f"{shot}-shot/class ({NUM_CLASSES * shot} database, "
              f"{len(query_rows)} query)")
        print("=" * 100)

        db_rows = protocol["train_indices_by_shot"][str(shot)]
        db_features = normalized_rows(features, db_rows)
        db_labels = np.asarray(labels[db_rows], dtype=np.int64)

        knn_results, search_seconds, faiss_backend = exact_knn(
            db_features,
            db_labels,
            query_features,
            query_labels,
            use_gpu=use_gpu,
            gpu_device=gpu_device,
        )
        for k in K_VALUES:
            print(f"K={k:2d}: {knn_results[f'top{k}']['accuracy']:.4f}%")

        num_database = len(db_rows)
        num_query = len(query_rows)
        feature_flops = int((num_database + num_query) * per_image_flops)
        search_flops = int(
            2 * num_query * num_database * features.shape[1]
        )

        record = {
            "model": MODEL_NAME,
            "feature_representation": FEATURE_REPRESENTATION,
            "feature_file": FEATURE_FILE,
            "labels_file": LABEL_FILE,
            "source_indices_file": SOURCE_INDEX_FILE,
            "protocol_file": PROTOCOL_JSON,
            "seed": SEED,
            "shot": shot,
            "database_size": num_database,
            "query_size": num_query,
            "query_per_class": QUERY_PER_CLASS,
            "feature_dim": int(features.shape[1]),
            "feature_dtype": str(features.dtype),
            "l2_normalized": True,
            "knn_index": faiss_backend,
            "gpu_enabled": use_gpu,
            "gpu_device": gpu_device if use_gpu else None,
            "knn_similarity": "cosine similarity",
            "temperature": TEMPERATURE,
            "k_values": list(K_VALUES),
            "knn_results": knn_results,
            "search_seconds": search_seconds,
            "encoder_flops_per_image": per_image_flops,
            "feature_extraction_flops": feature_flops,
            "exact_knn_search_flops": search_flops,
            "knn_total_flops": feature_flops + search_flops,
        }
        all_results[str(shot)] = record

        result_path = os.path.join(RESULT_ROOT, f"ijepa_{shot}shot.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        del db_features, db_labels
        gc.collect()

    summary = {
        "model": MODEL_NAME,
        "data_manifest": EXPORT_MANIFEST,
        "manifest_format": manifest.get("format_version"),
        "manifest_entry": entry,
        "results": all_results,
    }
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSummary saved to:", SUMMARY_JSON)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate arrays and write the protocol without running FAISS KNN",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="use CPU IndexFlatIP instead of the default FAISS GPU index",
    )
    parser.add_argument(
        "--gpu-device",
        type=int,
        default=0,
        help="CUDA device used by FAISS GPU (default: 0)",
    )
    args = parser.parse_args()
    run(
        validate_only=args.validate_only,
        use_gpu=not args.cpu,
        gpu_device=args.gpu_device,
    )


if __name__ == "__main__":
    main()


