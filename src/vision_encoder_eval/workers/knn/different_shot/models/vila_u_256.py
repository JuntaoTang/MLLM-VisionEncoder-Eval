
from vision_encoder_eval.core.runtime import asset_path
import os
import sys
import json
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ============================================================
# 0. Environment
# ============================================================

REPO_DIR = asset_path('model_assets', 'repositories/vila-u')
MODEL_DIR = asset_path('model_assets', 'model/vilau_256/vision_tower')

IMAGENET_ROOT = "/workspace/root/val"

PROTOCOL_PATH = asset_path('runtime', 'metaclip_knn/val_45shot_5query_seed42_protocol.json')

CACHE_ROOT = asset_path('runtime', 'vilau_knn_cache/val_5shot_seed42_45pool')

RESULT_PATH = os.path.join(
    CACHE_ROOT,
    "knn_result.json"
)

os.makedirs(CACHE_ROOT, exist_ok=True)

# Make sure VILA-U custom modules can be imported.
os.chdir(REPO_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

# Compatibility for newer transformers where no_init_weights moved/was removed.
try:
    import transformers.modeling_utils as _tf_modeling_utils
    if not hasattr(_tf_modeling_utils, "no_init_weights"):
        from contextlib import contextmanager
        @contextmanager
        def no_init_weights(_enable=True):
            yield
        _tf_modeling_utils.no_init_weights = no_init_weights
except Exception:
    pass

# OpenCV is used only by optional video helpers; this KNN run uses PIL images.
try:
    import cv2 as _cv2
except (ImportError, OSError):
    import types
    sys.modules.pop("cv2", None)
    sys.modules["cv2"] = types.ModuleType("cv2")


# ============================================================
# 1. Imports
# ============================================================

import numpy as np
import torch
import torch.nn.functional as F

import faiss

from PIL import Image
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModel,
    PreTrainedModel,
)

# Compatibility with Transformers 5 for this Transformers 4-era custom model.
if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
    PreTrainedModel.all_tied_weights_keys = {}

# IMPORTANT:
# This registers rqvaesigliptransformer_model with Transformers.
from vila_u.model.multimodal_encoder.rqvaesigliptransformer import (
    modeling_rqvaesigliptransformer
)


# ============================================================
# 2. Settings
# ============================================================

BATCH_SIZE = 128
NUM_WORKERS = 8

K_VALUES = [1, 5, 10, 20]
MAX_K = max(K_VALUES)

TEMPERATURE = 0.07

STORE_DTYPE = np.float16

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# For reproducibility
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 3. Utility
# ============================================================

def human_flops(flops):
    if flops >= 1e18:
        return f"{flops / 1e18:.4f} EFLOPs"
    elif flops >= 1e15:
        return f"{flops / 1e15:.4f} PFLOPs"
    elif flops >= 1e12:
        return f"{flops / 1e12:.4f} TFLOPs"
    elif flops >= 1e9:
        return f"{flops / 1e9:.4f} GFLOPs"
    elif flops >= 1e6:
        return f"{flops / 1e6:.4f} MFLOPs"
    else:
        return f"{flops:.0f} FLOPs"


def print_section(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


# ============================================================
# 4. ImageNet Dataset
# ============================================================

class ImageNetDataset(Dataset):

    def __init__(
        self,
        root,
        indices,
        processor,
    ):
        self.root = root
        self.indices = np.asarray(indices, dtype=np.int64)
        self.processor = processor

        self.classes = sorted(
            [
                d for d in os.listdir(root)
                if os.path.isdir(os.path.join(root, d))
            ]
        )

        self.class_to_idx = {
            cls_name: i
            for i, cls_name in enumerate(self.classes)
        }

        self.samples = []

        for cls_name in self.classes:

            class_dir = os.path.join(root, cls_name)

            files = sorted(
                [
                    os.path.join(class_dir, f)
                    for f in os.listdir(class_dir)
                    if f.lower().endswith(
                        (".jpg", ".jpeg", ".png", ".webp")
                    )
                ]
            )

            label = self.class_to_idx[cls_name]

            for path in files:
                self.samples.append((path, label))

        # Important:
        # Protocol indices refer to the deterministic sorted ImageNet
        # validation dataset ordering.
        self.samples = [
            self.samples[i]
            for i in self.indices
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        path, label = self.samples[idx]

        image = Image.open(path).convert("RGB")

        # AutoImageProcessor handles:
        # resize -> center crop -> rescale -> normalize
        inputs = self.processor(
            images=image,
            return_tensors="pt"
        )

        pixel_values = inputs["pixel_values"].squeeze(0)

        return pixel_values, label


# ============================================================
# 5. Load protocol
# ============================================================

print_section("VILA-U KNN | Configuration")

print(f"Repository : {REPO_DIR}")
print(f"Model      : {MODEL_DIR}")
print(f"ImageNet   : {IMAGENET_ROOT}")
print(f"Protocol   : {PROTOCOL_PATH}")
print(f"Cache      : {CACHE_ROOT}")
print(f"Device     : {DEVICE}")

if not os.path.exists(PROTOCOL_PATH):
    raise FileNotFoundError(
        f"Protocol file not found:\n{PROTOCOL_PATH}"
    )

with open(PROTOCOL_PATH, "r") as f:
    protocol = json.load(f)

    # Canonical fixed 45-pool/5-query protocol plus legacy aliases used below.
    protocol.setdefault("train_indices", protocol["train_pool_indices"])
    protocol.setdefault("database_indices", protocol["train_pool_indices"])
    protocol.setdefault("db_indices", protocol["train_pool_indices"])
    protocol.setdefault("num_train", 45000)
    protocol.setdefault("train_per_class", 45)
    protocol.setdefault("images_per_class", 50)

train_indices = np.asarray(
    protocol["train_indices"],
    dtype=np.int64
)

query_indices = np.asarray(
    protocol["query_indices"],
    dtype=np.int64
)

print(f"\nProtocol keys:")
print(list(protocol.keys()))

print(f"\nDatabase images : {len(train_indices)}")
print(f"Query images    : {len(query_indices)}")

print(f"Train index min/max : "
      f"{train_indices.min()} / {train_indices.max()}")

print(f"Query index min/max  : "
      f"{query_indices.min()} / {query_indices.max()}")


# ============================================================
# 6. Check overlap
# ============================================================

overlap = np.intersect1d(
    train_indices,
    query_indices
)

if len(overlap) != 0:
    raise RuntimeError(
        f"Protocol error: DB/query overlap = {len(overlap)}"
    )

print(f"DB / Query overlap : {len(overlap)}")


# ============================================================
# 7. Load processor
# ============================================================

print_section("Loading Image Processor")

processor = AutoImageProcessor.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
)

print(processor)


# ============================================================
# 8. Load model
# ============================================================

print_section("Loading VILA-U Vision Model")

config = AutoConfig.from_pretrained(
    MODEL_DIR,
    trust_remote_code=True,
    local_files_only=True,
)

print(config)

# The outer checkpoint already contains all SigLIP weights. Point its nested
# architecture lookup at the verified local config to keep loading offline.
_local_siglip_config = asset_path('model_assets', 'model/vilau_256/siglip-large-patch16-256-config')
if isinstance(config.rqvaesiglip, dict):
    config.rqvaesiglip["pretrained_model"] = _local_siglip_config
else:
    config.rqvaesiglip.pretrained_model = _local_siglip_config

model = AutoModel.from_pretrained(
    MODEL_DIR,
    config=config,
    trust_remote_code=True,
    local_files_only=True,
)

model = model.to(DEVICE)
model.eval()

print("\nModel loaded successfully.")

# ------------------------------------------------------------
# The outer RQVAESIGLIPTransformer does NOT expose:
#
#     model(pixel_values=...)
#
# Therefore we explicitly use:
#
# model.rqvaesiglip.siglip_model.vision_model
#
# and take:
#
#     outputs.pooler_output
#
# Shape = [B, 1024]
# ------------------------------------------------------------

vision_model = (
    model
    .rqvaesiglip
    .siglip_model
    .vision_model
)

print("\nVision model:")
print(vision_model)


# ============================================================
# 9. Verify feature extraction
# ============================================================

print_section("Testing Feature Extraction")

test_image = os.path.join(
    IMAGENET_ROOT,
    "n01440764",
    "ILSVRC2012_val_00000293.JPEG"
)

if not os.path.exists(test_image):

    # fallback to first protocol image
    print(
        "Default test image not found, "
        "using first ImageNet image."
    )

    first_idx = int(train_indices[0])

    # Reconstruct deterministic sample list
    all_samples = []

    classes = sorted(
        [
            d for d in os.listdir(IMAGENET_ROOT)
            if os.path.isdir(
                os.path.join(IMAGENET_ROOT, d)
            )
        ]
    )

    for cls_name in classes:

        class_dir = os.path.join(
            IMAGENET_ROOT,
            cls_name
        )

        files = sorted(
            [
                os.path.join(class_dir, f)
                for f in os.listdir(class_dir)
                if f.lower().endswith(
                    (".jpg", ".jpeg", ".png", ".webp")
                )
            ]
        )

        for path in files:
            all_samples.append(path)

    test_image = all_samples[first_idx]

image = Image.open(test_image).convert("RGB")

inputs = processor(
    images=image,
    return_tensors="pt"
)

pixel_values = inputs["pixel_values"].to(DEVICE)

with torch.inference_mode():

    test_output = vision_model(
        pixel_values=pixel_values,
        return_dict=True,
    )

test_features = test_output.pooler_output

print(f"Test image       : {test_image}")
print(f"Pixel values     : {pixel_values.shape}")
print(f"Pooler output    : {test_features.shape}")
print(f"Feature dtype    : {test_features.dtype}")

if test_features.ndim != 2:
    raise RuntimeError(
        f"Expected [B,D] feature, got {test_features.shape}"
    )

FEATURE_DIM = test_features.shape[-1]

print(f"Feature dimension: {FEATURE_DIM}")

if FEATURE_DIM != 1024:
    raise RuntimeError(
        f"Expected VILA-U SigLIP feature dimension 1024, "
        f"got {FEATURE_DIM}"
    )

print("\nFeature extraction path verified:")
print(
    "Image -> VILA-U SigLIP Vision Transformer "
    "-> pooler_output [B, 1024]"
)


# ============================================================
# 10. Cache paths
# ============================================================

TRAIN_FEATURES_PATH = os.path.join(
    CACHE_ROOT,
    "train_features.npy"
)

TRAIN_LABELS_PATH = os.path.join(
    CACHE_ROOT,
    "train_labels.npy"
)

TRAIN_INDICES_PATH = os.path.join(
    CACHE_ROOT,
    "train_indices.npy"
)

QUERY_FEATURES_PATH = os.path.join(
    CACHE_ROOT,
    "query_features.npy"
)

QUERY_LABELS_PATH = os.path.join(
    CACHE_ROOT,
    "query_labels.npy"
)

QUERY_INDICES_PATH = os.path.join(
    CACHE_ROOT,
    "query_indices.npy"
)


# ============================================================
# 11. Determine cache validity
# ============================================================

def cache_valid():

    required = [
        TRAIN_FEATURES_PATH,
        TRAIN_LABELS_PATH,
        TRAIN_INDICES_PATH,
        QUERY_FEATURES_PATH,
        QUERY_LABELS_PATH,
        QUERY_INDICES_PATH,
    ]

    if not all(os.path.exists(p) for p in required):
        return False

    try:

        cached_train_indices = np.load(
            TRAIN_INDICES_PATH
        )

        cached_query_indices = np.load(
            QUERY_INDICES_PATH
        )

        if not np.array_equal(
            cached_train_indices,
            train_indices
        ):
            return False

        if not np.array_equal(
            cached_query_indices,
            query_indices
        ):
            return False

        train_features = np.load(
            TRAIN_FEATURES_PATH,
            mmap_mode="r"
        )

        query_features = np.load(
            QUERY_FEATURES_PATH,
            mmap_mode="r"
        )

        if train_features.shape != (
            len(train_indices),
            FEATURE_DIM
        ):
            return False

        if query_features.shape != (
            len(query_indices),
            FEATURE_DIM
        ):
            return False

        return True

    except Exception:
        return False


# ============================================================
# 12. Extract features
# ============================================================

def extract_features(
    dataset,
    name,
):

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
    )

    all_features = []
    all_labels = []

    total = len(dataset)

    start_time = time.perf_counter()

    with torch.inference_mode():

        for batch_idx, (pixel_values, labels) in enumerate(loader):

            pixel_values = pixel_values.to(
                DEVICE,
                non_blocking=True
            )

            outputs = vision_model(
                pixel_values=pixel_values,
                return_dict=True,
            )

            features = outputs.pooler_output

            # Unified KNN protocol:
            # L2-normalized features.
            features = F.normalize(
                features,
                dim=-1
            )

            features = features.detach().cpu().numpy()

            all_features.append(
                features.astype(
                    STORE_DTYPE,
                    copy=False
                )
            )

            all_labels.append(
                labels.numpy().astype(
                    np.int64,
                    copy=False
                )
            )

            processed = min(
                (batch_idx + 1) * BATCH_SIZE,
                total
            )

            if (
                batch_idx == 0
                or
                (batch_idx + 1) % 20 == 0
                or
                processed == total
            ):

                elapsed = (
                    time.perf_counter()
                    - start_time
                )

                rate = processed / max(elapsed, 1e-9)

                print(
                    f"[{name}] "
                    f"{processed:6d}/{total:6d} "
                    f"({processed / total * 100:6.2f}%) | "
                    f"{rate:8.2f} img/s | "
                    f"{elapsed / 60:8.2f} min"
                )

    features = np.concatenate(
        all_features,
        axis=0
    )

    labels = np.concatenate(
        all_labels,
        axis=0
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    print(
        f"\n{name} extraction finished:"
    )

    print(
        f"  Features : {features.shape}"
    )

    print(
        f"  Labels   : {labels.shape}"
    )

    print(
        f"  Time     : {elapsed / 60:.2f} min"
    )

    print(
        f"  Speed    : "
        f"{total / max(elapsed, 1e-9):.2f} img/s"
    )

    return features, labels, elapsed


# ============================================================
# 13. Build datasets / load cache
# ============================================================

if cache_valid():

    print_section("Loading Cached Features")

    train_features = np.load(
        TRAIN_FEATURES_PATH
    )

    train_labels = np.load(
        TRAIN_LABELS_PATH
    )

    query_features = np.load(
        QUERY_FEATURES_PATH
    )

    query_labels = np.load(
        QUERY_LABELS_PATH
    )

    print(
        f"Train features : {train_features.shape}"
    )

    print(
        f"Query features : {query_features.shape}"
    )

    feature_extraction_time = 0.0

    print(
        "\nCache is valid for the current protocol."
    )

else:

    print_section(
        "Cache Missing / Invalid -> Extracting Features"
    )

    train_dataset = ImageNetDataset(
        IMAGENET_ROOT,
        train_indices,
        processor,
    )

    query_dataset = ImageNetDataset(
        IMAGENET_ROOT,
        query_indices,
        processor,
    )

    print(
        f"Train dataset : {len(train_dataset)}"
    )

    print(
        f"Query dataset : {len(query_dataset)}"
    )

    train_features, train_labels, train_time = (
        extract_features(
            train_dataset,
            "DATABASE"
        )
    )

    query_features, query_labels, query_time = (
        extract_features(
            query_dataset,
            "QUERY"
        )
    )

    feature_extraction_time = (
        train_time + query_time
    )

    # Save protocol-specific cache
    np.save(
        TRAIN_FEATURES_PATH,
        train_features
    )

    np.save(
        TRAIN_LABELS_PATH,
        train_labels
    )

    np.save(
        TRAIN_INDICES_PATH,
        train_indices
    )

    np.save(
        QUERY_FEATURES_PATH,
        query_features
    )

    np.save(
        QUERY_LABELS_PATH,
        query_labels
    )

    np.save(
        QUERY_INDICES_PATH,
        query_indices
    )

    print(
        "\nFeatures saved to:"
    )

    print(
        f"  {CACHE_ROOT}"
    )


# ============================================================
# 14. Basic feature checks
# ============================================================

print_section("Feature Validation")

print(
    f"Train feature shape : {train_features.shape}"
)

print(
    f"Query feature shape : {query_features.shape}"
)

print(
    f"Train dtype         : {train_features.dtype}"
)

print(
    f"Query dtype         : {query_features.dtype}"
)

if train_features.shape != (
    len(train_indices),
    FEATURE_DIM
):
    raise RuntimeError(
        "Train feature shape mismatch."
    )

if query_features.shape != (
    len(query_indices),
    FEATURE_DIM
):
    raise RuntimeError(
        "Query feature shape mismatch."
    )

# Make sure FAISS gets float32.
train_features_f32 = np.ascontiguousarray(
    train_features.astype(
        np.float32,
        copy=False
    )
)

query_features_f32 = np.ascontiguousarray(
    query_features.astype(
        np.float32,
        copy=False
    )
)

# Check normalization.
train_norms = np.linalg.norm(
    train_features_f32,
    axis=1
)

query_norms = np.linalg.norm(
    query_features_f32,
    axis=1
)

print(
    f"Train norm mean : {train_norms.mean():.6f}"
)

print(
    f"Query norm mean : {query_norms.mean():.6f}"
)


# ============================================================
# 15. FAISS KNN
# ============================================================

print_section("FAISS KNN")

NUM_DATABASE = len(train_features_f32)
NUM_QUERY = len(query_features_f32)
D = FEATURE_DIM

print(
    f"Database size : {NUM_DATABASE}"
)

print(
    f"Query size    : {NUM_QUERY}"
)

print(
    f"Feature dim   : {D}"
)

print(
    f"K values      : {K_VALUES}"
)

print(
    f"Max K         : {MAX_K}"
)

print(
    f"Temperature   : {TEMPERATURE}"
)

print(
    "\nBuilding IndexFlatIP..."
)

index = faiss.IndexFlatIP(D)

index.add(train_features_f32)

print(
    f"FAISS ntotal : {index.ntotal}"
)


# ------------------------------------------------------------
# IMPORTANT:
# Search ONLY ONCE at MAX_K.
# Results are reused for K=1,5,10,20.
# ------------------------------------------------------------

print(
    f"\nSearching once with K={MAX_K}..."
)

if torch.cuda.is_available():
    torch.cuda.synchronize()

knn_start = time.perf_counter()

Dists, Neighbors = index.search(
    query_features_f32,
    MAX_K
)

if torch.cuda.is_available():
    torch.cuda.synchronize()

knn_search_time = (
    time.perf_counter()
    - knn_start
)

print(
    f"FAISS search time : "
    f"{knn_search_time:.4f} sec"
)

print(
    f"Distance shape    : {Dists.shape}"
)

print(
    f"Neighbor shape    : {Neighbors.shape}"
)


# ============================================================
# 16. Similarity-weighted KNN
# ============================================================

def weighted_knn_accuracy(
    distances,
    neighbors,
    train_labels,
    query_labels,
    k,
    temperature=0.07,
):

    distances_k = distances[:, :k]
    neighbors_k = neighbors[:, :k]

    neighbor_labels = train_labels[
        neighbors_k
    ]

    # Cosine similarity because features are L2-normalized
    similarities = distances_k

    # Temperature-scaled soft voting.
    weights = np.exp(
        similarities / temperature
    )

    num_classes = int(
        max(
            train_labels.max(),
            query_labels.max()
        ) + 1
    )

    predictions = np.empty(
        len(query_labels),
        dtype=np.int64
    )

    # Vectorized class voting would require a
    # large [N,K,C] tensor. Since K<=20 and N=5000,
    # this simple loop is inexpensive and memory-safe.

    for i in range(len(query_labels)):

        scores = np.bincount(
            neighbor_labels[i],
            weights=weights[i],
            minlength=num_classes
        )

        predictions[i] = np.argmax(scores)

    accuracy = (
        predictions == query_labels
    ).mean()

    return accuracy, predictions


# ============================================================
# 17. Evaluate K
# ============================================================

results = {}

for k in K_VALUES:

    start = time.perf_counter()

    accuracy, predictions = (
        weighted_knn_accuracy(
            Dists,
            Neighbors,
            train_labels,
            query_labels,
            k,
            TEMPERATURE,
        )
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    results[f"top1_k{k}"] = float(
        accuracy * 100
    )

    print(
        f"K={k:2d} | "
        f"Top-1 = {accuracy * 100:8.4f}% | "
        f"Voting time = {elapsed:.4f}s"
    )


# ============================================================
# 18. FLOPs
# ============================================================

print_section("FLOPs")

# ------------------------------------------------------------
# VILA-U internal visual encoder:
#
# SigLIP-L/16 @ 256
#
# image = 256
# patch = 16
# patches = 16*16 = 256
# sequence length = 256
#
# NOTE:
# SigLIP vision transformer has no CLS token.
# It uses a MultiheadAttention pooling head.
# ------------------------------------------------------------

IMAGE_SIZE = 256
PATCH_SIZE = 16

WIDTH = 1024
LAYERS = 24
HEADS = 16
MLP_DIM = 4096

NUM_PATCHES = (
    IMAGE_SIZE // PATCH_SIZE
) ** 2

SEQ_LEN = NUM_PATCHES

HEAD_DIM = WIDTH // HEADS

print(
    f"Image size       : {IMAGE_SIZE}"
)

print(
    f"Patch size       : {PATCH_SIZE}"
)

print(
    f"Num patches      : {NUM_PATCHES}"
)

print(
    f"Sequence length  : {SEQ_LEN}"
)

print(
    f"Hidden dim       : {WIDTH}"
)

print(
    f"Layers           : {LAYERS}"
)

print(
    f"Heads            : {HEADS}"
)

print(
    f"MLP dim          : {MLP_DIM}"
)


# ------------------------------------------------------------
# Patch embedding
#
# Conv2d:
# output spatial = 16*16
# kernel parameters = 3*16*16
#
# FLOPs = 2 * output_elements * kernel_size * out_channels
# ------------------------------------------------------------

patch_flops_per_image = (
    2
    * NUM_PATCHES
    * (3 * PATCH_SIZE * PATCH_SIZE)
    * WIDTH
)


# ------------------------------------------------------------
# Transformer block
#
# QKV:
# 2 * S * W * (3W)
#
# QK^T:
# 2 * H * S^2 * head_dim
#
# Attn @ V:
# 2 * H * S^2 * head_dim
#
# Output projection:
# 2 * S * W * W
#
# MLP:
# 2 * S * W * MLP
# +
# 2 * S * MLP * W
# ------------------------------------------------------------

qkv_flops = (
    2
    * SEQ_LEN
    * WIDTH
    * (3 * WIDTH)
)

qk_flops = (
    2
    * HEADS
    * SEQ_LEN
    * SEQ_LEN
    * HEAD_DIM
)

av_flops = (
    2
    * HEADS
    * SEQ_LEN
    * SEQ_LEN
    * HEAD_DIM
)

out_proj_flops = (
    2
    * SEQ_LEN
    * WIDTH
    * WIDTH
)

mlp_flops = (
    2
    * SEQ_LEN
    * WIDTH
    * MLP_DIM
    +
    2
    * SEQ_LEN
    * MLP_DIM
    * WIDTH
)

block_flops = (
    qkv_flops
    + qk_flops
    + av_flops
    + out_proj_flops
    + mlp_flops
)

transformer_flops_per_image = (
    LAYERS * block_flops
)


# ------------------------------------------------------------
# SigLIP pooling head
#
# Actual model:
#
# SiglipMultiheadAttentionPoolingHead
#
# It performs attention pooling over the 256 visual tokens.
#
# For FLOPs accounting we include the major MHA projections
# and attention matrix operations, plus its MLP.
#
# ------------------------------------------------------------

pool_qkv_flops = (
    2
    * 1
    * WIDTH
    * (3 * WIDTH)
)

pool_qk_flops = (
    2
    * HEADS
    * 1
    * SEQ_LEN
    * HEAD_DIM
)

pool_av_flops = (
    2
    * HEADS
    * 1
    * SEQ_LEN
    * HEAD_DIM
)

pool_out_proj_flops = (
    2
    * WIDTH
    * WIDTH
)

pool_mlp_flops = (
    2
    * WIDTH
    * MLP_DIM
    +
    2
    * MLP_DIM
    * WIDTH
)

pooling_head_flops = (
    pool_qkv_flops
    + pool_qk_flops
    + pool_av_flops
    + pool_out_proj_flops
    + pool_mlp_flops
)


# ------------------------------------------------------------
# Per-image feature FLOPs
# ------------------------------------------------------------

feature_flops_per_image = (
    patch_flops_per_image
    + transformer_flops_per_image
    + pooling_head_flops
)

TOTAL_FEATURE_IMAGES = (
    NUM_DATABASE + NUM_QUERY
)

feature_extraction_flops = (
    TOTAL_FEATURE_IMAGES
    * feature_flops_per_image
)


# ------------------------------------------------------------
# KNN search FLOPs
#
# IndexFlatIP:
#
# N_query * N_database * D
#
# Each dot-product multiply-add = 2 FLOPs.
# ------------------------------------------------------------

knn_search_flops = (
    NUM_QUERY
    * NUM_DATABASE
    * D
    * 2
)


# ------------------------------------------------------------
# Voting FLOPs
#
# Unified approximate convention:
#
# N_query * K * 5
# ------------------------------------------------------------

voting_flops = sum(
    NUM_QUERY * k * 5
    for k in K_VALUES
)

# Total:
total_flops = (
    feature_extraction_flops
    + knn_search_flops
    + voting_flops
)


print(
    f"\nPatch embedding FLOPs/image : "
    f"{human_flops(patch_flops_per_image)}"
)

print(
    f"Transformer FLOPs/image     : "
    f"{human_flops(transformer_flops_per_image)}"
)

print(
    f"Pooling head FLOPs/image    : "
    f"{human_flops(pooling_head_flops)}"
)

print(
    f"Feature FLOPs/image         : "
    f"{human_flops(feature_flops_per_image)}"
)

print(
    f"\nTotal feature images        : "
    f"{TOTAL_FEATURE_IMAGES}"
)

print(
    f"Feature extraction FLOPs    : "
    f"{human_flops(feature_extraction_flops)}"
)

print(
    f"KNN search FLOPs            : "
    f"{human_flops(knn_search_flops)}"
)

print(
    f"KNN voting FLOPs            : "
    f"{human_flops(voting_flops)}"
)

print(
    f"\nKNN TOTAL FLOPs             : "
    f"{human_flops(total_flops)}"
)


# ============================================================
# 19. Theoretical FLOPs sanity check
# ============================================================

print("\nFLOPs breakdown:")

print(
    f"  Feature extraction : "
    f"{feature_extraction_flops / total_flops * 100:.2f}%"
)

print(
    f"  FAISS search       : "
    f"{knn_search_flops / total_flops * 100:.2f}%"
)

print(
    f"  Voting             : "
    f"{voting_flops / total_flops * 100:.2f}%"
)


# ============================================================
# 20. Result summary
# ============================================================

print_section("FINAL RESULT")

print(
    f"Model      : VILA-U internal SigLIP Vision"
)

print(
    f"Feature    : pooler_output"
)

print(
    f"Feature dim: {FEATURE_DIM}"
)

print(
    f"Protocol   : ImageNet val 5-shot/class"
)

print(
    f"Database   : {NUM_DATABASE}"
)

print(
    f"Query      : {NUM_QUERY}"
)

for k in K_VALUES:

    print(
        f"Top-1 @ K={k:2d} : "
        f"{results[f'top1_k{k}']:.4f}%"
    )

print(
    f"\nFeature FLOPs : "
    f"{human_flops(feature_extraction_flops)}"
)

print(
    f"Search FLOPs  : "
    f"{human_flops(knn_search_flops)}"
)

print(
    f"Voting FLOPs  : "
    f"{human_flops(voting_flops)}"
)

print(
    f"TOTAL FLOPs   : "
    f"{human_flops(total_flops)}"
)

print(
    f"\nFAISS search time : "
    f"{knn_search_time:.4f} sec"
)


# ============================================================
# 21. Save results
# ============================================================

result = {

    "model": "VILA-U",
    "model_dir": MODEL_DIR,

    "feature_extraction": {
        "path": (
            "model.rqvaesiglip."
            "siglip_model."
            "vision_model"
        ),
        "feature": "pooler_output",
        "feature_dim": int(FEATURE_DIM),
        "normalization": "L2",
    },

    "protocol": {
        "protocol_path": PROTOCOL_PATH,
        "dataset": "ImageNet-1K validation",
        "database_size": int(NUM_DATABASE),
        "query_size": int(NUM_QUERY),
        "shots_per_class": 5,
        "seed": SEED,
        "db_query_overlap": int(len(overlap)),
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "k_values": K_VALUES,
        "temperature": TEMPERATURE,
        "metric": "cosine_similarity",
        "faiss_index": "IndexFlatIP",
        "weighted_voting": True,
    },

    "accuracy": results,

    "flops": {
        "image_size": IMAGE_SIZE,
        "patch_size": PATCH_SIZE,
        "num_patches": NUM_PATCHES,
        "hidden_dim": WIDTH,
        "layers": LAYERS,
        "heads": HEADS,
        "mlp_dim": MLP_DIM,

        "feature_flops_per_image": int(
            feature_flops_per_image
        ),

        "feature_extraction_flops": int(
            feature_extraction_flops
        ),

        "knn_search_flops": int(
            knn_search_flops
        ),

        "voting_flops": int(
            voting_flops
        ),

        "total_flops": int(
            total_flops
        ),
    },

    "runtime": {
        "feature_extraction_seconds": float(
            feature_extraction_time
        ),
        "faiss_search_seconds": float(
            knn_search_time
        ),
    },

    "cache": CACHE_ROOT,
}


with open(
    RESULT_PATH,
    "w"
) as f:

    json.dump(
        result,
        f,
        indent=2
    )


print(
    f"\nResult saved to:"
)

print(
    RESULT_PATH
)

print("\nDONE.")

# Four exact searches reuse the single 45k database feature cache above.
from vision_encoder_eval.workers.knn.different_shot.multishot_protocol import run_multishot_from_cache
MULTISHOT_RESULTS = run_multishot_from_cache(CACHE_ROOT, CACHE_ROOT, "vila_u_256", model_name="VILA-U-256")

