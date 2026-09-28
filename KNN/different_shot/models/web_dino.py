import os
import json
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F
import faiss

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, Dinov2Model


# ============================================================
# 1. Configuration
# ============================================================

MODEL_DIR = "/cache/models/model/webssl-dino1b-full2b-224"

IMAGENET_VAL = "/workspace/root/val"

PROTOCOL_PATH = "/cache/metaclip_knn/val_45shot_5query_seed42_protocol.json"

CACHE_ROOT = "/cache/webssl_dino1b_knn_45pool"
FEATURE_ROOT = os.path.join(CACHE_ROOT, "features")
RESULT_ROOT = os.path.join(CACHE_ROOT, "results")

MODEL_NAME = "WebSSL-DINO-1B"

BATCH_SIZE = 128
NUM_WORKERS = 8

K_VALUES = [1, 5, 10, 20]
TEMPERATURE = 0.07

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(FEATURE_ROOT, exist_ok=True)
os.makedirs(RESULT_ROOT, exist_ok=True)


# ============================================================
# 2. Print configuration
# ============================================================

print("=" * 100)
print("WebSSL-DINO-1B ImageNet 5-shot KNN")
print("=" * 100)

print(f"Model       : {MODEL_NAME}")
print(f"Model dir   : {MODEL_DIR}")
print(f"ImageNet    : {IMAGENET_VAL}")
print(f"Protocol    : {PROTOCOL_PATH}")
print(f"Feature dir : {FEATURE_ROOT}")
print(f"Result dir  : {RESULT_ROOT}")
print(f"Device      : {DEVICE}")
print(f"Batch size  : {BATCH_SIZE}")
print(f"Workers     : {NUM_WORKERS}")
print(f"K values    : {K_VALUES}")
print(f"Temperature : {TEMPERATURE}")
print("=" * 100)


# ============================================================
# 3. Load exact protocol
# ============================================================

print("\n[1/8] Loading exact 5-shot protocol...")

with open(PROTOCOL_PATH, "r") as f:
    protocol = json.load(f)

    # Canonical fixed 45-pool/5-query protocol plus legacy aliases used below.
    protocol.setdefault("train_indices", protocol["train_pool_indices"])
    protocol.setdefault("database_indices", protocol["train_pool_indices"])
    protocol.setdefault("db_indices", protocol["train_pool_indices"])
    protocol.setdefault("num_train", 45000)
    protocol.setdefault("train_per_class", 45)
    protocol.setdefault("images_per_class", 50)

required_keys = [
    "seed",
    "dataset",
    "dataset_path",
    "num_classes",
    "images_per_class",
    "train_per_class",
    "query_per_class",
    "num_train",
    "num_query",
    "train_indices",
    "query_indices",
]

for key in required_keys:
    if key not in protocol:
        raise KeyError(
            f"Protocol file missing required key: {key}"
        )

print(f"Seed              : {protocol['seed']}")
print(f"Dataset           : {protocol['dataset']}")
print(f"Num classes       : {protocol['num_classes']}")
print(f"Images/class      : {protocol['images_per_class']}")
print(f"Train/class       : {protocol['train_per_class']}")
print(f"Query/class       : {protocol['query_per_class']}")
print(f"Num train         : {protocol['num_train']}")
print(f"Num query         : {protocol['num_query']}")

if protocol["num_train"] != 45000:
    raise ValueError(
        f"Expected 45000 database images, got {protocol['num_train']}"
    )

if protocol["num_query"] != 5000:
    raise ValueError(
        f"Expected 5000 query images, got {protocol['num_query']}"
    )

if protocol["train_per_class"] != 45:
    raise ValueError(
        f"Expected 45 train images/class, got {protocol['train_per_class']}"
    )

if protocol["query_per_class"] != 5:
    raise ValueError(
        f"Expected 5 query images/class, got {protocol['query_per_class']}"
    )

train_indices = np.asarray(
    protocol["train_indices"],
    dtype=np.int64
)

query_indices = np.asarray(
    protocol["query_indices"],
    dtype=np.int64
)

if len(np.intersect1d(train_indices, query_indices)) != 0:
    raise RuntimeError(
        "Train/query protocol has overlapping indices!"
    )

print("Protocol verified: 5000 database + 5000 query.")


# ============================================================
# 4. ImageNet dataset
# ============================================================

print("\n[2/8] Building ImageNet validation dataset...")


class ImageNetFolderDataset(Dataset):

    def __init__(self, root, processor):
        self.root = root
        self.processor = processor

        self.samples = []

        classes = sorted(
            [
                d for d in os.listdir(root)
                if os.path.isdir(os.path.join(root, d))
            ]
        )

        self.class_to_idx = {
            cls_name: idx
            for idx, cls_name in enumerate(classes)
        }

        for cls_name in classes:
            cls_dir = os.path.join(root, cls_name)
            label = self.class_to_idx[cls_name]

            for fname in sorted(os.listdir(cls_dir)):
                path = os.path.join(cls_dir, fname)

                if not os.path.isfile(path):
                    continue

                lower = fname.lower()

                if lower.endswith(
                    (".jpg", ".jpeg", ".png", ".bmp", ".webp")
                ):
                    self.samples.append(
                        (path, label)
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        path, label = self.samples[index]

        image = Image.open(path).convert("RGB")

        inputs = self.processor(
            images=image,
            return_tensors="pt"
        )

        pixel_values = inputs["pixel_values"].squeeze(0)

        return pixel_values, label


# ============================================================
# 5. Load official processor + model
# ============================================================

print("\n[3/8] Loading Web-DINO processor and model...")

processor = AutoImageProcessor.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)

model = Dinov2Model.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
    attn_implementation="sdpa"
)

model = model.to(DEVICE)
model.eval()


# ============================================================
# 6. Print model architecture information
# ============================================================

config = model.config

hidden_size = int(config.hidden_size)
num_layers = int(config.num_hidden_layers)
num_heads = int(config.num_attention_heads)

patch_size = int(config.patch_size)

image_size = config.image_size

if isinstance(image_size, (list, tuple)):
    image_size = int(image_size[0])
else:
    image_size = int(image_size)

print("\nModel architecture:")
print(f"  hidden size       : {hidden_size}")
print(f"  layers            : {num_layers}")
print(f"  attention heads   : {num_heads}")
print(f"  patch size        : {patch_size}")
print(f"  image size        : {image_size}")
print(f"  parameters        : {sum(p.numel() for p in model.parameters()):,}")

if hidden_size != 1536:
    raise RuntimeError(
        f"Expected Web-DINO-1B hidden size 1536, got {hidden_size}"
    )

if num_layers != 40:
    raise RuntimeError(
        f"Expected 40 layers, got {num_layers}"
    )

if num_heads != 24:
    raise RuntimeError(
        f"Expected 24 attention heads, got {num_heads}"
    )

if patch_size != 14:
    raise RuntimeError(
        f"Expected patch size 14, got {patch_size}"
    )

if image_size != 224:
    raise RuntimeError(
        f"Expected image size 224, got {image_size}"
    )


# ============================================================
# 7. Build dataset
# ============================================================

dataset = ImageNetFolderDataset(
    IMAGENET_VAL,
    processor
)

print(f"\nFull ImageNet val images: {len(dataset):,}")

if len(dataset) != 50000:
    print(
        f"WARNING: expected 50,000 ImageNet validation images, "
        f"found {len(dataset):,}"
    )


# ============================================================
# 8. Feature extraction
# ============================================================

def extract_features(indices, split_name):

    cache_path = os.path.join(
        FEATURE_ROOT,
        f"{MODEL_NAME}_{split_name}_features.npy"
    )

    labels_path = os.path.join(
        FEATURE_ROOT,
        f"{MODEL_NAME}_{split_name}_labels.npy"
    )

    # --------------------------------------------------------
    # Reuse cache
    # --------------------------------------------------------

    if os.path.exists(cache_path) and os.path.exists(labels_path):

        print(
            f"\nFound cached {split_name} features:"
        )
        print(f"  {cache_path}")

        features = np.load(cache_path)
        labels = np.load(labels_path)

        print(f"  feature shape: {features.shape}")
        print(f"  label shape  : {labels.shape}")

        return features.astype(np.float32), labels.astype(np.int64)

    # --------------------------------------------------------
    # Subset dataset
    # --------------------------------------------------------

    class IndexedDataset(Dataset):

        def __init__(self, base_dataset, indices):
            self.base_dataset = base_dataset
            self.indices = indices

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            real_idx = int(self.indices[idx])
            return self.base_dataset[real_idx]

    subset = IndexedDataset(
        dataset,
        indices
    )

    loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0)
    )

    features_list = []
    labels_list = []

    print(
        f"\nExtracting {split_name} features..."
    )

    start_time = time.time()

    with torch.no_grad():

        for step, (images, labels) in enumerate(loader):

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            outputs = model(
                pixel_values=images
            )

            # ------------------------------------------------
            # Official DINO feature:
            # CLS token
            # ------------------------------------------------

            features = outputs.last_hidden_state[:, 0, :]

            # ------------------------------------------------
            # L2 normalize
            # ------------------------------------------------

            features = F.normalize(
                features.float(),
                p=2,
                dim=1
            )

            features_list.append(
                features.cpu().numpy()
            )

            labels_list.append(
                labels.numpy()
            )

            if (
                step == 0
                or (step + 1) % 20 == 0
                or (step + 1) == len(loader)
            ):

                processed = min(
                    (step + 1) * BATCH_SIZE,
                    len(subset)
                )

                elapsed = time.time() - start_time

                print(
                    f"  {processed:5d}/{len(subset):5d} "
                    f"({processed / len(subset) * 100:6.2f}%) "
                    f"| {elapsed:.1f}s"
                )

    features = np.concatenate(
        features_list,
        axis=0
    ).astype(np.float32)

    labels = np.concatenate(
        labels_list,
        axis=0
    ).astype(np.int64)

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    assert features.shape[0] == len(indices)
    assert features.shape[1] == hidden_size
    assert labels.shape[0] == len(indices)

    norms = np.linalg.norm(
        features,
        axis=1
    )

    print(
        f"\n{split_name} feature statistics:"
    )
    print(
        f"  shape : {features.shape}"
    )
    print(
        f"  min   : {features.min():.6f}"
    )
    print(
        f"  max   : {features.max():.6f}"
    )
    print(
        f"  mean  : {features.mean():.6f}"
    )
    print(
        f"  std   : {features.std():.6f}"
    )
    print(
        f"  norm  : mean={norms.mean():.6f}, "
        f"std={norms.std():.6f}, "
        f"min={norms.min():.6f}, "
        f"max={norms.max():.6f}"
    )

    np.save(
        cache_path,
        features
    )

    np.save(
        labels_path,
        labels
    )

    print(
        f"Saved features to:\n  {cache_path}"
    )

    return features, labels


# ============================================================
# 9. Extract database/query features
# ============================================================

print("\n[4/8] Feature extraction")

train_features, train_labels = extract_features(
    train_indices,
    "database"
)

query_features, query_labels = extract_features(
    query_indices,
    "query"
)


# ============================================================
# 10. Build FAISS cosine-similarity index
# ============================================================

print("\n[5/8] Building FAISS IndexFlatIP...")

feature_dim = train_features.shape[1]

index = faiss.IndexFlatIP(feature_dim)

index.add(
    train_features
)

print(
    f"FAISS database size : {index.ntotal:,}"
)

print(
    f"Feature dimension   : {feature_dim}"
)


# ============================================================
# 11. KNN inference
# ============================================================

print("\n[6/8] Running KNN...")


def temperature_weighted_vote(
    similarities,
    neighbor_labels,
    temperature=0.07
):

    # similarities: [B, K]
    # labels      : [B, K]

    similarities = similarities.astype(
        np.float32
    )

    neighbor_labels = neighbor_labels.astype(
        np.int64
    )

    # Numerical stability
    max_sim = similarities.max(
        axis=1,
        keepdims=True
    )

    weights = np.exp(
        (similarities - max_sim)
        / temperature
    )

    predictions = []

    for i in range(
        similarities.shape[0]
    ):

        labels_i = neighbor_labels[i]
        weights_i = weights[i]

        unique_labels = np.unique(
            labels_i
        )

        scores = {}

        for label in unique_labels:

            mask = (
                labels_i == label
            )

            scores[int(label)] = float(
                weights_i[mask].sum()
            )

        prediction = max(
            scores,
            key=scores.get
        )

        predictions.append(
            prediction
        )

    return np.asarray(
        predictions,
        dtype=np.int64
    )


results = {}

start_knn = time.time()

for k in K_VALUES:

    print(
        f"\nRunning K={k}..."
    )

    similarities, neighbor_indices = index.search(
        query_features,
        k
    )

    neighbor_labels = train_labels[
        neighbor_indices
    ]

    predictions = temperature_weighted_vote(
        similarities,
        neighbor_labels,
        temperature=TEMPERATURE
    )

    top1 = (
        predictions == query_labels
    ).mean() * 100.0

    results[f"top1_k{k}"] = float(
        top1
    )

    print(
        f"  Top-1 @ K={k}: {top1:.4f}%"
    )


knn_time = time.time() - start_knn

print(
    f"\nKNN elapsed time: {knn_time:.2f}s"
)


# ============================================================
# 12. FLOPs calculation
# ============================================================

print("\n[7/8] Calculating FLOPs...")


def compute_webdino_flops():

    # --------------------------------------------------------
    # Architecture
    # --------------------------------------------------------

    D = hidden_size
    L = num_layers
    H = 4096

    P = patch_size
    image = image_size

    num_patches = (
        image // P
    ) ** 2

    # + CLS token
    N = num_patches + 1

    # --------------------------------------------------------
    # Patch embedding
    #
    # Conv2D equivalent:
    # Npatch * (3 * P * P) * D
    #
    # multiply + add = 2 FLOPs
    # --------------------------------------------------------

    patch_flops = (
        2
        * num_patches
        * (3 * P * P)
        * D
    )

    # --------------------------------------------------------
    # Attention
    #
    # QKV:
    # 3 * N * D * D
    #
    # QK^T:
    # N * N * D
    #
    # Attention @ V:
    # N * N * D
    #
    # Output projection:
    # N * D * D
    #
    # Every matmul = 2 FLOPs/MAC
    # --------------------------------------------------------

    qkv_flops = (
        2
        * 3
        * N
        * D
        * D
    )

    qk_flops = (
        2
        * N
        * N
        * D
    )

    av_flops = (
        2
        * N
        * N
        * D
    )

    proj_flops = (
        2
        * N
        * D
        * D
    )

    attention_flops = (
        qkv_flops
        + qk_flops
        + av_flops
        + proj_flops
    )

    # --------------------------------------------------------
    # DINOv2 SwiGLU FFN
    #
    # Web-DINO 1B:
    # D = 1536
    # SwiGLU hidden = 4096
    #
    # Three input projections:
    #   D -> H
    #   D -> H
    #   D -> H
    #
    # One output projection:
    #   H -> D
    #
    # total = 4 * D * H
    # --------------------------------------------------------

    mlp_flops = (
        2
        * N
        * D
        * H
        * 4
    )

    block_flops = (
        attention_flops
        + mlp_flops
    )

    encoder_flops_per_image = (
        patch_flops
        + L * block_flops
    )

    # --------------------------------------------------------
    # Our KNN protocol encodes:
    #
    # 5000 database + 5000 query
    # = 10000 images
    # --------------------------------------------------------

    num_encoded_images = (
        protocol["num_train"]
        + protocol["num_query"]
    )

    encoder_total_flops = (
        encoder_flops_per_image
        * num_encoded_images
    )

    # --------------------------------------------------------
    # Exact cosine / inner-product similarity
    #
    # 5000 query x 5000 database
    #
    # Dot product of D dimensions:
    # 2 * D FLOPs
    # --------------------------------------------------------

    num_train = protocol["num_train"]
    num_query = protocol["num_query"]

    knn_similarity_flops = (
        2
        * num_train
        * num_query
        * feature_dim
    )

    total_flops = (
        encoder_total_flops
        + knn_similarity_flops
    )

    return {
        "image_size": image,
        "patch_size": P,
        "num_patches": num_patches,
        "num_tokens": N,
        "hidden_size": D,
        "num_layers": L,
        "num_heads": num_heads,
        "swiglu_hidden_size": H,

        "patch_embedding_flops": int(
            patch_flops
        ),

        "attention_flops_per_layer": int(
            attention_flops
        ),

        "swiglu_flops_per_layer": int(
            mlp_flops
        ),

        "transformer_block_flops": int(
            block_flops
        ),

        "encoder_flops_per_image": int(
            encoder_flops_per_image
        ),

        "encoder_flops_per_image_gflops":
            encoder_flops_per_image / 1e9,

        "num_encoded_images":
            num_encoded_images,

        "encoder_total_flops":
            int(encoder_total_flops),

        "encoder_total_tflops":
            encoder_total_flops / 1e12,

        "knn_similarity_flops":
            int(knn_similarity_flops),

        "knn_similarity_tflops":
            knn_similarity_flops / 1e12,

        "knn_total_flops":
            int(total_flops),

        "knn_total_tflops":
            total_flops / 1e12,

        "knn_total_pflops":
            total_flops / 1e15,
    }


flops = compute_webdino_flops()


# ============================================================
# 13. Save results
# ============================================================

result = {

    "model": MODEL_NAME,

    "model_dir": MODEL_DIR,

    "dataset": {
        "name": "ImageNet-1K",
        "split": "val",
        "path": IMAGENET_VAL,
    },

    "protocol": {
        "protocol_file": PROTOCOL_PATH,
        "seed": protocol["seed"],
        "num_classes": protocol["num_classes"],
        "train_per_class":
            protocol["train_per_class"],
        "query_per_class":
            protocol["query_per_class"],
        "num_train":
            protocol["num_train"],
        "num_query":
            protocol["num_query"],
    },

    "feature": {
        "type": "CLS",
        "dimension": int(feature_dim),
        "normalization": "L2",
        "dtype": "float32",
    },

    "knn": {
        "index": "FAISS IndexFlatIP",
        "similarity": "cosine",
        "temperature": TEMPERATURE,
        "K": K_VALUES,
        "results": results,
    },

    "flops": flops,

    "runtime": {
        "knn_seconds": knn_time,
    },
}


result_path = os.path.join(
    RESULT_ROOT,
    f"{MODEL_NAME}_5shot_seed42_results.json"
)

with open(
    result_path,
    "w"
) as f:
    json.dump(
        result,
        f,
        indent=2
    )


# ============================================================
# 14. Final output
# ============================================================

print("\n" + "=" * 100)
print("FINAL RESULTS")
print("=" * 100)

print(
    f"Model: {MODEL_NAME}"
)

print(
    f"Feature: CLS ({feature_dim}-D), L2 normalized"
)

print(
    f"Database: {protocol['num_train']}"
)

print(
    f"Query: {protocol['num_query']}"
)

print("\nKNN Accuracy:")

for k in K_VALUES:

    print(
        f"  K={k:2d}: "
        f"{results[f'top1_k{k}']:.4f}%"
    )


print("\nFLOPs:")

print(
    f"  Encoder FLOPs/image : "
    f"{flops['encoder_flops_per_image_gflops']:.6f} GFLOPs"
)

print(
    f"  Encoder total       : "
    f"{flops['encoder_total_tflops']:.6f} TFLOPs"
)

print(
    f"  KNN similarity       : "
    f"{flops['knn_similarity_tflops']:.6f} TFLOPs"
)

print(
    f"  KNN TOTAL            : "
    f"{flops['knn_total_tflops']:.6f} TFLOPs"
)

print(
    f"  KNN TOTAL            : "
    f"{flops['knn_total_pflops']:.6f} PFLOPs"
)

print("\nSaved result:")
print(
    f"  {result_path}"
)

print("=" * 100)

# Four exact searches reuse the single 45k database feature cache above.
from multishot_protocol import run_multishot_from_cache
MULTISHOT_RESULTS = run_multishot_from_cache(CACHE_ROOT, RESULT_ROOT, "web_dino")

