import os
import json
import time
import random
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets

import open_clip
import faiss


# ============================================================
# 1. Global settings
# ============================================================

IMAGENET_VAL = "/workspace/root/val"

# IMPORTANT:
# EXACT SAME protocol used by the previous MetaCLIP experiments.
PROTOCOL_PATH = (
    "/cache/metaclip_knn/"
    "val_45shot_5query_seed42_protocol.json"
)

CACHE_ROOT = "/cache/siglip2_multi_knn"

SEED = 42

BATCH_SIZE = 128
NUM_WORKERS = 8

K_VALUES = [1, 5, 10, 20]
MAX_K = max(K_VALUES)

TEMPERATURE = 0.07

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# FLOPs convention:
# 1 MAC = 2 FLOPs
FLOPS_PER_MAC = 2


# ============================================================
# 2. Models
# ============================================================

MODELS = [

    # --------------------------------------------------------
    # SO400M
    # --------------------------------------------------------

    {
        "name": "SigLIP2-SO400M-14-224",
        "arch": "ViT-SO400M-14-SigLIP2",
        "path":
            "/cache/models/model/hub/"
            "models--timm--ViT-SO400M-14-SigLIP2",
        "input_size": 224,
        "patch": 14,
        "depth": 27,
        "width": 1152,
        "heads": 16,
        "mlp_dim": 4304,
    },

    {
        "name": "SigLIP2-SO400M-14-378",
        "arch": "ViT-SO400M-14-SigLIP2-378",
        "path":
            "/cache/models/model/hub/"
            "models--timm--ViT-SO400M-14-SigLIP2-378",
        "input_size": 378,
        "patch": 14,
        "depth": 27,
        "width": 1152,
        "heads": 16,
        "mlp_dim": 4304,
    },

    {
        "name": "SigLIP2-SO400M-16-256",
        "arch": "ViT-SO400M-16-SigLIP2-256",
        "path":
            "/cache/models/model/hub/"
            "models--timm--ViT-SO400M-16-SigLIP2-256",
        "input_size": 256,
        "patch": 16,
        "depth": 27,
        "width": 1152,
        "heads": 16,
        "mlp_dim": 4304,
    },

    {
        "name": "SigLIP2-SO400M-16-384",
        "arch": "ViT-SO400M-16-SigLIP2-384",
        "path":
            "/cache/models/model/hub/"
            "models--timm--ViT-SO400M-16-SigLIP2-384",
        "input_size": 384,
        "patch": 16,
        "depth": 27,
        "width": 1152,
        "heads": 16,
        "mlp_dim": 4304,
    },

    {
        "name": "SigLIP2-SO400M-16-512",
        "arch": "ViT-SO400M-16-SigLIP2-512",
        "path":
            "/cache/models/model/hub/"
            "models--timm--ViT-SO400M-16-SigLIP2-512",
        "input_size": 512,
        "patch": 16,
        "depth": 27,
        "width": 1152,
        "heads": 16,
        "mlp_dim": 4304,
    },

    # --------------------------------------------------------
    # GOPT
    # --------------------------------------------------------

    {
        "name": "SigLIP2-gopt-16-256",
        "arch": "ViT-gopt-16-SigLIP2-256",
        "path":
            "/cache/models/model/hub/"
            "models--timm--ViT-gopt-16-SigLIP2-256",
        "input_size": 256,
        "patch": 16,
        "depth": 40,
        "width": 1536,
        "heads": 16,
        "mlp_dim": 6144,
    },

    {
        "name": "SigLIP2-gopt-16-384",
        "arch": "ViT-gopt-16-SigLIP2-384",
        "path":
            "/cache/models/model/hub/"
            "models--timm--ViT-gopt-16-SigLIP2-384",
        "input_size": 384,
        "patch": 16,
        "depth": 40,
        "width": 1536,
        "heads": 16,
        "mlp_dim": 6144,
    },

    # --------------------------------------------------------
    # Base
    # --------------------------------------------------------

    {
        "name": "SigLIP2-B32-256",
        "arch": "ViT-B-32-SigLIP2-256",
        "path":
            "/workspace/.cache/huggingface/hub/"
            "models--timm--ViT-B-32-SigLIP2-256",
        "input_size": 256,
        "patch": 32,
        "depth": 12,
        "width": 768,
        "heads": 12,
        "mlp_dim": 3072,
    },

    {
        "name": "SigLIP2-B16-224",
        "arch": "ViT-B-16-SigLIP2",
        "path":
            "/workspace/.cache/huggingface/hub/"
            "models--timm--ViT-B-16-SigLIP2",
        "input_size": 224,
        "patch": 16,
        "depth": 12,
        "width": 768,
        "heads": 12,
        "mlp_dim": 3072,
    },

    {
        "name": "SigLIP2-B16-256",
        "arch": "ViT-B-16-SigLIP2-256",
        "path":
            "/workspace/.cache/huggingface/hub/"
            "models--timm--ViT-B-16-SigLIP2-256",
        "input_size": 256,
        "patch": 16,
        "depth": 12,
        "width": 768,
        "heads": 12,
        "mlp_dim": 3072,
    },

    {
        "name": "SigLIP2-B16-384",
        "arch": "ViT-B-16-SigLIP2-384",
        "path":
            "/workspace/.cache/huggingface/hub/"
            "models--timm--ViT-B-16-SigLIP2-384",
        "input_size": 384,
        "patch": 16,
        "depth": 12,
        "width": 768,
        "heads": 12,
        "mlp_dim": 3072,
    },

    {
        "name": "SigLIP2-B16-512",
        "arch": "ViT-B-16-SigLIP2-512",
        "path":
            "/workspace/.cache/huggingface/hub/"
            "models--timm--ViT-B-16-SigLIP2-512",
        "input_size": 512,
        "patch": 16,
        "depth": 12,
        "width": 768,
        "heads": 12,
        "mlp_dim": 3072,
    },

    # --------------------------------------------------------
    # Large
    # --------------------------------------------------------

    {
        "name": "SigLIP2-L16-256",
        "arch": "ViT-L-16-SigLIP2-256",
        "path":
            "/workspace/.cache/huggingface/hub/"
            "models--timm--ViT-L-16-SigLIP2-256",
        "input_size": 256,
        "patch": 16,
        "depth": 24,
        "width": 1024,
        "heads": 16,
        "mlp_dim": 4096,
    },

    {
        "name": "SigLIP2-L16-384",
        "arch": "ViT-L-16-SigLIP2-384",
        "path":
            "/workspace/.cache/huggingface/hub/"
            "models--timm--ViT-L-16-SigLIP2-384",
        "input_size": 384,
        "patch": 16,
        "depth": 24,
        "width": 1024,
        "heads": 16,
        "mlp_dim": 4096,
    },

    {
        "name": "SigLIP2-L16-512",
        "arch": "ViT-L-16-SigLIP2-512",
        "path":
            "/workspace/.cache/huggingface/hub/"
            "models--timm--ViT-L-16-SigLIP2-512",
        "input_size": 512,
        "patch": 16,
        "depth": 24,
        "width": 1024,
        "heads": 16,
        "mlp_dim": 4096,
    },
]


# ============================================================
# 3. Seed
# ============================================================

def seed_everything(seed):

    random.seed(seed)
    np.random.seed(seed)

# ============================================================
# 3. Unified 45-shot protocol
# ============================================================
IMAGENET_TRAIN = "/cache/imagenet-1k/standard/train"
PROTOCOL_PATH = "/cache/metaclip_knn/val_45shot_5query_seed42_protocol.json"
CACHE_ROOT = "/cache/siglip2_multi_knn_45shot"
SEED = 42
TRAIN_SHOTS = [5, 10, 20, 45]
TRAIN_POOL_PER_CLASS = 45
QUERY_PER_CLASS = 5
K = 20
TEMPERATURE = 0.07
BATCH_SIZE = 128
NUM_WORKERS = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FLOPS_PER_MAC = 2
os.makedirs(CACHE_ROOT, exist_ok=True)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(SEED)


def separator(char="-", n=110):
    print(char * n)


def find_checkpoint(model_dir):
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Directory not found: {model_dir}")
    candidates = []
    priority = {
        "open_clip_model.safetensors": 0,
        "model.safetensors": 1,
    }
    for root, _, files in os.walk(model_dir):
        for fn in files:
            path = os.path.join(root, fn)
            if fn in priority:
                candidates.append((priority[fn], path))
            elif fn.endswith((".safetensors", ".pth", ".pt", ".bin")):
                candidates.append((2 if fn.endswith(".safetensors") else 3, path))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found in {model_dir}")
    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[0][1]


def build_or_load_protocol():
    os.makedirs(os.path.dirname(PROTOCOL_PATH), exist_ok=True)
    if os.path.isfile(PROTOCOL_PATH):
        with open(PROTOCOL_PATH) as f:
            p = json.load(f)
        required = {
            "seed": SEED,
            "train_pool_per_class": TRAIN_POOL_PER_CLASS,
            "query_per_class": QUERY_PER_CLASS,
            "train_shots": TRAIN_SHOTS,
        }
        for k, v in required.items():
            if p.get(k) != v:
                raise RuntimeError(f"Protocol field {k} mismatch: {p.get(k)} != {v}")
        print(f"Loaded fixed protocol: {PROTOCOL_PATH}")
        return p

    ds = datasets.ImageFolder(IMAGENET_TRAIN, transform=None)
    if len(ds.classes) != 1000:
        raise RuntimeError(f"Expected 1000 classes, got {len(ds.classes)}")
    by_class = {c: [] for c in range(1000)}
    for idx, (_, label) in enumerate(ds.samples):
        by_class[label].append(idx)
    rng = random.Random(SEED)
    train_pool_by_class, query_by_class = {}, {}
    train_indices_by_shot = {str(s): [] for s in TRAIN_SHOTS}
    query_indices = []
    for c in range(1000):
        ids = by_class[c].copy()
        need = TRAIN_POOL_PER_CLASS + QUERY_PER_CLASS
        if len(ids) < need:
            raise RuntimeError(f"Class {c} has {len(ids)} images; need {need}")
        rng.shuffle(ids)
        pool = ids[:TRAIN_POOL_PER_CLASS]
        query = ids[TRAIN_POOL_PER_CLASS:need]
        train_pool_by_class[str(c)] = pool
        query_by_class[str(c)] = query
        query_indices.extend(query)
        for shot in TRAIN_SHOTS:
            train_indices_by_shot[str(shot)].extend(pool[:shot])
    p = {
        "seed": SEED,
        "dataset": IMAGENET_TRAIN,
        "num_classes": 1000,
        "train_pool_per_class": TRAIN_POOL_PER_CLASS,
        "query_per_class": QUERY_PER_CLASS,
        "train_shots": TRAIN_SHOTS,
        "train_pool_size": 1000 * TRAIN_POOL_PER_CLASS,
        "query_size": 1000 * QUERY_PER_CLASS,
        "train_pool_by_class": train_pool_by_class,
        "query_by_class": query_by_class,
        "train_indices_by_shot": train_indices_by_shot,
        "query_indices": query_indices,
    }
    with open(PROTOCOL_PATH, "w") as f:
        json.dump(p, f, indent=2)
    print(f"Created protocol: {PROTOCOL_PATH}")
    return p


class IndexedImageNet(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, i):
        idx = int(self.indices[i])
        image, label = self.dataset[idx]
        return image, label, idx


def calculate_vit_flops(image_size, patch_size, depth, width, heads, mlp_dim, embed_dim):
    if image_size % patch_size != 0:
        raise ValueError(f"Image size {image_size} is not divisible by patch size {patch_size}")
    n = (image_size // patch_size) ** 2 + 1
    head_dim = width // heads
    flops = FLOPS_PER_MAC * (image_size // patch_size) ** 2 * (3 * patch_size * patch_size) * width
    for _ in range(depth):
        flops += FLOPS_PER_MAC * n * width * (3 * width)
        flops += FLOPS_PER_MAC * heads * n * n * head_dim * 2
        flops += FLOPS_PER_MAC * n * width * width
        flops += FLOPS_PER_MAC * n * width * mlp_dim
        flops += FLOPS_PER_MAC * n * mlp_dim * width
    flops += FLOPS_PER_MAC * width * embed_dim
    return int(flops)


def extract_features(model, loader, split_name):
    fs, ys, ids = [], [], []
    total = len(loader.dataset)
    start = time.time()
    for step, (images, labels, indices) in enumerate(loader):
        images = images.to(DEVICE, non_blocking=True)
        with torch.inference_mode():
            feat = model.encode_image(images)
            feat = F.normalize(feat.float(), dim=-1)
        fs.append(feat.cpu().numpy().astype(np.float32))
        ys.append(labels.numpy().astype(np.int64))
        ids.append(indices.numpy().astype(np.int64))
        if step == 0 or (step + 1) % 20 == 0 or (step + 1) == len(loader):
            done = min((step + 1) * BATCH_SIZE, total)
            print(f"\r{split_name}: {done}/{total} ({100*done/total:.2f}%)", end="")
    print()
    return np.concatenate(fs), np.concatenate(ys), np.concatenate(ids)


def run_knn(train_features, train_labels, query_features, query_labels, k=20):
    train_features = np.ascontiguousarray(train_features, dtype=np.float32)
    query_features = np.ascontiguousarray(query_features, dtype=np.float32)
    index = faiss.IndexFlatIP(train_features.shape[1])
    index.add(train_features)
    sims, neigh = index.search(query_features, k)
    preds = np.empty(len(query_labels), dtype=np.int64)
    for i in range(len(query_labels)):
        labels = train_labels[neigh[i]]
        weights = np.exp((sims[i] - sims[i].max()) / TEMPERATURE)
        scores = {}
        for lab, w in zip(labels, weights):
            lab = int(lab)
            scores[lab] = scores.get(lab, 0.0) + float(w)
        preds[i] = max(scores.items(), key=lambda x: x[1])[0]
    acc = float((preds == query_labels).mean())
    same = float((train_labels[neigh] == query_labels[:, None]).mean())
    return acc, same


def load_or_extract(model, dataset, indices, loader_name, cache_dir, feature_dim=None):
    safe = loader_name.replace("/", "_")
    fp = os.path.join(cache_dir, safe + "_features.npy")
    lp = os.path.join(cache_dir, safe + "_labels.npy")
    ip = os.path.join(cache_dir, safe + "_indices.npy")
    expected = np.asarray(indices, dtype=np.int64)
    if all(os.path.isfile(x) for x in [fp, lp, ip]):
        f, y, ids = np.load(fp, mmap_mode="r"), np.load(lp), np.load(ip)
        if np.array_equal(ids, expected):
            print(f"Loading cache: {fp}")
            return f, y, ids
        print("Cache index mismatch; extracting again.")
    ds = IndexedImageNet(dataset, expected)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                        pin_memory=True, persistent_workers=NUM_WORKERS > 0)
    f, y, ids = extract_features(model, loader, loader_name)
    np.save(fp, f)
    np.save(lp, y)
    np.save(ip, ids)
    return f, y, ids


def calculate_total_flops(per_image, n_train, n_query, dim):
    feature = per_image * (n_train + n_query)
    search = FLOPS_PER_MAC * n_train * n_query * dim
    voting = n_query * K * 5
    total = feature + search + voting
    return {"feature_flops": int(feature), "search_flops": int(search), "voting_flops": int(voting),
            "total_flops": int(total), "total_tflops": total / 1e12, "total_pflops": total / 1e15}


# ============================================================
# Main benchmark
# ============================================================
print("=" * 100)
print("SigLIP2 Multi-Resolution KNN Benchmark | 5/10/20/45-shot")
print("=" * 100)
print(f"Device: {DEVICE}")
print(f"Train root: {IMAGENET_TRAIN}")
print(f"Shots: {TRAIN_SHOTS}; query/class: {QUERY_PER_CLASS}; K: {K}")

if not os.path.isdir(IMAGENET_TRAIN):
    raise FileNotFoundError(IMAGENET_TRAIN)

protocol = build_or_load_protocol()
base_dataset = datasets.ImageFolder(IMAGENET_TRAIN, transform=None)
all_results = []

for model_cfg in MODELS:
    name = model_cfg["name"]
    model = None
    try:
        print("\n" + "=" * 100)
        print(name)
        print("=" * 100)
        checkpoint = find_checkpoint(model_cfg["path"])
        print("Checkpoint:", checkpoint)
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_cfg["arch"], pretrained=checkpoint, device=DEVICE)
        model.eval()
        with torch.inference_mode():
            dummy = torch.zeros(1, 3, model_cfg["input_size"], model_cfg["input_size"], device=DEVICE)
            feature_dim = int(model.encode_image(dummy).shape[-1])
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Feature dim: {feature_dim}; params: {num_params/1e6:.2f}M")
        base_dataset.transform = preprocess
        model_cache = os.path.join(CACHE_ROOT, name)
        os.makedirs(model_cache, exist_ok=True)
        query_indices = np.asarray(protocol["query_indices"], dtype=np.int64)
        query_features, query_labels, query_cached_ids = load_or_extract(
            model, base_dataset, query_indices, "query_45pool5query_seed42", model_cache)
        if not np.array_equal(query_cached_ids, query_indices):
            raise RuntimeError("Query cache verification failed")
        for shot in TRAIN_SHOTS:
            train_indices = np.asarray(protocol["train_indices_by_shot"][str(shot)], dtype=np.int64)
            train_features, train_labels, train_cached_ids = load_or_extract(
                model, base_dataset, train_indices, f"train_{shot}shot", model_cache)
            if not np.array_equal(train_cached_ids, train_indices):
                raise RuntimeError(f"Train cache verification failed for {shot}-shot")
            overlap = np.intersect1d(train_cached_ids, query_cached_ids)
            if len(overlap):
                raise RuntimeError(f"Train/query overlap: {len(overlap)}")
            acc, same = run_knn(train_features, train_labels, query_features, query_labels, K)
            per_image = calculate_vit_flops(model_cfg["input_size"], model_cfg["patch"], model_cfg["depth"],
                                            model_cfg["width"], model_cfg["heads"], model_cfg["mlp_dim"], feature_dim)
            flops = calculate_total_flops(per_image, len(train_features), len(query_features), feature_dim)
            result = {
                "model": name, "architecture": model_cfg["arch"], "checkpoint": checkpoint,
                "input_size": model_cfg["input_size"], "feature_dim": feature_dim, "params": num_params,
                "train_shots": shot, "database_size": len(train_features), "query_size": len(query_features),
                "K": K, "temperature": TEMPERATURE, "accuracy": acc * 100.0,
                "same_class_neighbor_rate": same, "feature_flops_per_image": per_image,
                "feature_gflops_per_image": per_image / 1e9, **flops,
                "protocol_path": PROTOCOL_PATH, "seed": SEED, "normalization": "L2",
                "faiss": "IndexFlatIP", "train_indices_cache": f"train_{shot}shot",
                "query_indices_cache": "query_45pool5query_seed42",
            }
            all_results.append(result)
            shot_result = os.path.join(model_cache, f"train_{shot}shot_results.json")
            with open(shot_result, "w") as f:
                json.dump(result, f, indent=2)
            print(f"{name} | {shot}-shot | K={K} | Top-1={acc*100:.4f}% | total={flops['total_pflops']:.6f} PFLOPs")
        del model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"[ERROR] {name}: {repr(e)}")
        import traceback
        traceback.print_exc()
        if model is not None:
            del model
        torch.cuda.empty_cache()

all_results_path = os.path.join(CACHE_ROOT, "all_siglip2_knn_results_5_10_20_45shot.json")
with open(all_results_path, "w") as f:
    json.dump({"protocol": protocol, "results": all_results}, f, indent=2)

print("\n" + "=" * 120)
print("FINAL RESULTS")
print("=" * 120)
print(f"{'Model':<38}{'Shot':>6}{'Train':>8}{'Top1':>10}{'GFLOPs/img':>14}{'PFLOPs':>14}")
for r in all_results:
    print(f"{r['model']:<38}{r['train_shots']:>6}{r['database_size']:>8}{r['accuracy']:>9.3f}%"
          f"{r['feature_gflops_per_image']:>14.3f}{r['total_pflops']:>14.6f}")
print("Saved:", all_results_path)

