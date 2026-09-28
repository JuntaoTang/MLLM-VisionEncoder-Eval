"""Shared fixed-protocol post-processing for the KNN benchmarks.

The model files retain their original model/transform/extraction code.  They extract
the ordered 195-image pool and the fixed queries once; this module slices the
cached feature matrix and performs the six exact searches.
"""
import glob
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROTOCOL_PATH = os.environ.get(
    "KNN_PROTOCOL",
    os.path.join(HERE, "..", "protocols", "imagenet_train_195shot_5query_seed42.json"),
)
SHOTS = (5, 10, 20, 45, 95, 195)
NUM_CLASSES = 1000
POOL_PER_CLASS = 195
QUERY_PER_CLASS = 5
K = 20


def load_fixed_protocol(path=PROTOCOL_PATH, shots=SHOTS, pool_per_class=POOL_PER_CLASS,
                        query_per_class=QUERY_PER_CLASS, num_classes=NUM_CLASSES):
    with open(path, "r", encoding="utf-8") as f:
        p = json.load(f)
    expected = {"seed": 42, "train_pool_per_class": pool_per_class,
                "query_per_class": query_per_class,
                "num_train_pool": num_classes * pool_per_class,
                "num_query": num_classes * query_per_class}
    for key, value in expected.items():
        if p.get(key) != value:
            raise ValueError(f"protocol {key}: {p.get(key)!r} != {value!r}")
    if p.get("train_shots") != list(shots):
        raise ValueError(f"protocol train_shots must be {list(shots)}")
    if len(p["train_pool_indices"]) != num_classes * pool_per_class or len(p["query_indices"]) != num_classes * query_per_class:
        raise ValueError("protocol index counts do not match configured pool/query sizes")
    if set(p["train_pool_indices"]) & set(p["query_indices"]):
        raise ValueError("database pool and query overlap")
    return p


def indices_for_shot(protocol, shot):
    """Take the first ``shot`` entries in every class's ordered 195-item pool."""
    pool = np.asarray(protocol["train_pool_indices"], dtype=np.int64)
    return pool.reshape(NUM_CLASSES, POOL_PER_CLASS)[:, :shot].reshape(-1).tolist()


def _feature_pairs(cache_root, database_size=195000, query_size=5000):
    """Yield 195k database + 5k query feature/label pairs.

    Labels are optional because some migrated scripts only save features.  The
    fixed protocol stores features ordered by class: 195 database images per class
    and 5 query images per class, so labels can be synthesized safely.
    """
    files = glob.glob(os.path.join(cache_root, "**", "*.npy"), recursive=True)

    def arr_len(path):
        try:
            return int(np.load(path, mmap_mode="r").shape[0])
        except Exception:
            return -1

    def is_label(path):
        return "label" in os.path.basename(path).lower()

    db_words = ("database", "db_", "train", "protocol_database")
    q_words = ("query", "val", "protocol_query")

    for directory in sorted({os.path.dirname(f) for f in files}):
        in_dir = [f for f in files if os.path.dirname(f) == directory]
        db_feats = [f for f in in_dir if not is_label(f) and any(w in os.path.basename(f).lower() for w in db_words) and arr_len(f) == database_size]
        q_feats = [f for f in in_dir if not is_label(f) and any(w in os.path.basename(f).lower() for w in q_words) and arr_len(f) == query_size]
        db_labs = [f for f in in_dir if is_label(f) and any(w in os.path.basename(f).lower() for w in db_words) and arr_len(f) == database_size]
        q_labs = [f for f in in_dir if is_label(f) and any(w in os.path.basename(f).lower() for w in q_words) and arr_len(f) == query_size]
        if db_feats and q_feats:
            yield directory, sorted(db_feats)[0], (sorted(db_labs)[0] if db_labs else None), sorted(q_feats)[0], (sorted(q_labs)[0] if q_labs else None)


def _find_feature_flops(result_root, model_key):
    keys = {"feature_flops_per_image", "flops_per_image", "feature_extraction_flops_per_image", "model_flops_per_image", "model_gflops_per_image", "encoder_flops_per_image"}
    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k in keys and isinstance(v, (int, float)):
                    val = float(v)
                    return val * 1e9 if k == "model_gflops_per_image" else val
                if k in {"feature_extraction_flops", "total_feature_flops"} and isinstance(v, (int, float)):
                    return float(v) / 50000.0
                got = walk(v)
                if got is not None: return got
        elif isinstance(x, list):
            for v in x:
                got = walk(v)
                if got is not None: return got
    paths = glob.glob(os.path.join(result_root, "**", "*.json"), recursive=True)
    paths.sort(key=lambda p: (model_key.lower() not in p.lower(), p))
    for path in paths:
        try:
            got = walk(json.load(open(path, encoding="utf-8")))
            if got is not None: return got
        except Exception:
            pass
    raise RuntimeError(f"cannot find per-image feature FLOPs for {model_key}")


def _exact_top20(db, db_labels, query, query_labels, temperature=0.07):
    import faiss
    db = np.ascontiguousarray(db.astype(np.float32, copy=False))
    query = np.ascontiguousarray(query.astype(np.float32, copy=False))
    faiss.normalize_L2(db); faiss.normalize_L2(query)
    index = faiss.IndexFlatIP(db.shape[1])
    index.add(db)
    start = time.perf_counter()
    scores, neighbours = index.search(query, K)
    search_sec = time.perf_counter() - start
    labels = db_labels[neighbours]
    weights = np.exp((scores - scores.max(1, keepdims=True)) / temperature)
    votes = np.zeros((len(query), NUM_CLASSES), dtype=np.float32)
    rows = np.arange(len(query))[:, None]
    np.add.at(votes, (np.broadcast_to(rows, labels.shape), labels), weights)
    predictions = votes.argmax(1)
    return float((predictions == query_labels).mean() * 100.0), search_sec


def run_multishot_from_cache(cache_root, result_root, benchmark_name, model_name=None,
                             protocol_path=PROTOCOL_PATH, shots=SHOTS,
                             pool_per_class=POOL_PER_CLASS,
                             query_per_class=QUERY_PER_CLASS,
                             num_classes=NUM_CLASSES):
    shots = tuple(shots)
    database_size = num_classes * pool_per_class
    query_size = num_classes * query_per_class
    protocol = load_fixed_protocol(
        protocol_path, shots, pool_per_class, query_per_class, num_classes
    )
    os.makedirs(result_root, exist_ok=True)
    summary = {}
    pairs = list(_feature_pairs(cache_root, database_size, query_size))
    if not pairs:
        raise RuntimeError(f"no complete feature cache found below {cache_root}")
    for directory, dbp, dblp, qp, qlp in pairs:
        model = model_name or os.path.basename(directory) or benchmark_name
        db_all = np.load(dbp)
        labels_all = np.load(dblp) if dblp else np.repeat(np.arange(num_classes, dtype=np.int64), pool_per_class)
        query = np.load(qp)
        query_labels = np.load(qlp) if qlp else np.repeat(np.arange(num_classes, dtype=np.int64), query_per_class)
        if len(db_all) != database_size or len(query) != query_size:
            continue
        per_image = _find_feature_flops(result_root, model)
        model_results = {}
        db_grid = np.arange(database_size).reshape(num_classes, pool_per_class)
        for shot in shots:
            take = db_grid[:, :shot].reshape(-1)
            db, db_labels = db_all[take], labels_all[take]
            accuracy, seconds = _exact_top20(db, db_labels, query, query_labels)
            feature_flops = int((len(db) + len(query)) * per_image)
            exact_search_flops = int(2 * len(query) * len(db) * db.shape[1])
            total_flops = feature_flops + exact_search_flops
            record = {
                "model": model, "shot": shot, "k": K,
                "protocol_file": protocol_path, "seed": 42,
                "database_size": int(len(db)), "query_size": query_size,
                "knn_top20_accuracy": accuracy,
                "feature_extraction_flops": feature_flops,
                "exact_knn_search_flops": exact_search_flops,
                "knn_total_flops": total_flops,
                "tflops": total_flops / 1e12,
                "train_size": int(len(db)),
                "feature_cache_database_size": database_size,
                "feature_cache_reused": True, "search_seconds": seconds,
            }
            model_results[str(shot)] = record
            with open(os.path.join(result_root, f"{model}_{shot}shot.json"), "w") as f:
                json.dump(record, f, indent=2)
        summary[model] = model_results
    out = os.path.join(result_root, f"{benchmark_name}_model_x_shot_summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print("\nMODEL x SHOT SUMMARY")
    print(f"{'model':<32} {'shot':>4} {'top20':>10} {'TFLOPs':>16}")
    for model, rows in summary.items():
        for shot in shots:
            r = rows[str(shot)]
            print(f"{model:<32} {shot:>4} {r['knn_top20_accuracy']:>9.4f}% {r['tflops']:>16.6f}")
    return summary
