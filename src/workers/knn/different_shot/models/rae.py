
from vision_encoder_eval.core.runtime import asset_path
import os
import sys
import json
import time
import random
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

import faiss


# ============================================================
# 1. Configuration
# ============================================================

MODEL_DIR = asset_path('model_assets', 'model/raev2_dinov3l_k7')

IMAGENET_TRAIN_DIR = (
    asset_path('imagenet', 'standard/train')
)

# ------------------------------------------------------------
# Fixed unified KNN protocol
# ------------------------------------------------------------

PROTOCOL_FILE = (
    asset_path('runtime', 'knn_protocols/imagenet_train_195shot_5query_seed42_protocol.json')
)

SEED = 42

SHOT_VALUES = (5, 10, 20, 45, 95, 195)
TRAIN_POOL_PER_CLASS = 195
QUERY_PER_CLASS = 5
NUM_CLASSES = 1000
NUM_DATABASE = NUM_CLASSES * TRAIN_POOL_PER_CLASS
NUM_QUERY = NUM_CLASSES * QUERY_PER_CLASS

# ------------------------------------------------------------
# Image preprocessing
# ------------------------------------------------------------

IMG_SIZE = 256
RESIZE_SIZE = int(IMG_SIZE * 1.125)  # 288

# IMPORTANT:
# RAEv2 encode() expects RGB float tensor in [0, 1].
# Therefore DO NOT use ImageNet normalization.
#
# This is model-specific preprocessing and is intentionally
# kept consistent with the supplied RAEv2 reference code.
# ------------------------------------------------------------

transform = transforms.Compose([
    transforms.Resize(
        RESIZE_SIZE,
        interpolation=transforms.InterpolationMode.BICUBIC,
    ),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
])

# ------------------------------------------------------------
# DataLoader
# ------------------------------------------------------------

BATCH_SIZE = 256
NUM_WORKERS = 8

# ------------------------------------------------------------
# KNN
# ------------------------------------------------------------

K_VALUES = [1, 5, 10, 20]

TEMPERATURE = 0.07

# ------------------------------------------------------------
# Device
# ------------------------------------------------------------

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# ------------------------------------------------------------
# Feature
# ------------------------------------------------------------

FEATURE_DIM = 1024

# ------------------------------------------------------------
# Cache
# ------------------------------------------------------------

CACHE_DIR = (
    asset_path('runtime', 'knn_cache/raev2_dinov3l_k7_train195pool5query_seed42')
)

os.makedirs(
    CACHE_DIR,
    exist_ok=True
)

DATABASE_FEATURES = os.path.join(
    CACHE_DIR,
    "database_features.npy"
)

DATABASE_LABELS = os.path.join(
    CACHE_DIR,
    "database_labels.npy"
)

QUERY_FEATURES = os.path.join(
    CACHE_DIR,
    "query_features.npy"
)

QUERY_LABELS = os.path.join(
    CACHE_DIR,
    "query_labels.npy"
)

FLOPS_FILE = os.path.join(
    CACHE_DIR,
    "flops.json"
)

RESULT_FILE = os.path.join(
    CACHE_DIR,
    "results.json"
)


# ============================================================
# 2. Reproducibility
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 3. Indexed Dataset
# ============================================================

class IndexedSubset(Dataset):

    def __init__(
        self,
        dataset,
        indices,
    ):
        self.dataset = dataset
        self.indices = np.asarray(
            indices,
            dtype=np.int64,
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):

        real_index = int(
            self.indices[index]
        )

        return self.dataset[real_index]


# ============================================================
# 4. Load fixed protocol
# ============================================================

def load_protocol():

    print()
    print("=" * 100)
    print("LOADING FIXED KNN PROTOCOL")
    print("=" * 100)

    print(
        f"Protocol file : {PROTOCOL_FILE}"
    )

    if not os.path.exists(PROTOCOL_FILE):
        print("Protocol missing; creating deterministic ImageNet-train 195-pool/5-query protocol.")
        source = datasets.ImageFolder(IMAGENET_TRAIN_DIR)
        by_class = {class_id: [] for class_id in range(NUM_CLASSES)}
        for index, (_, label) in enumerate(source.samples):
            by_class[int(label)].append(index)
        rng = random.Random(SEED)
        train_pool_indices = []
        query_indices = []
        for class_id in range(NUM_CLASSES):
            indices = by_class[class_id][:]
            if len(indices) < TRAIN_POOL_PER_CLASS + QUERY_PER_CLASS:
                raise RuntimeError(
                    f"Class {class_id} has {len(indices)} images; "
                    f"need {TRAIN_POOL_PER_CLASS + QUERY_PER_CLASS}."
                )
            rng.shuffle(indices)
            train_pool_indices.extend(indices[:TRAIN_POOL_PER_CLASS])
            query_indices.extend(indices[TRAIN_POOL_PER_CLASS:TRAIN_POOL_PER_CLASS + QUERY_PER_CLASS])
        protocol = {
            "dataset": "ImageNet-1K train",
            "dataset_path": IMAGENET_TRAIN_DIR,
            "seed": SEED,
            "num_classes": NUM_CLASSES,
            "train_pool_per_class": TRAIN_POOL_PER_CLASS,
            "query_per_class": QUERY_PER_CLASS,
            "num_train_pool": len(train_pool_indices),
            "num_query": len(query_indices),
            "train_shots": list(SHOT_VALUES),
            "train_pool_indices": train_pool_indices,
            "query_indices": query_indices,
        }
        os.makedirs(os.path.dirname(PROTOCOL_FILE), exist_ok=True)
        with open(PROTOCOL_FILE, "w", encoding="utf-8") as f:
            json.dump(protocol, f)
        print(f"Protocol created: {PROTOCOL_FILE}")

    with open(
        PROTOCOL_FILE,
        "r",
    ) as f:

        protocol = json.load(f)


        # Canonical fixed 45-pool/5-query protocol plus legacy aliases used below.

        protocol.setdefault("train_indices", protocol["train_pool_indices"])

        protocol.setdefault("database_indices", protocol["train_pool_indices"])

        protocol.setdefault("db_indices", protocol["train_pool_indices"])

        protocol.setdefault("num_train", NUM_DATABASE)

        protocol.setdefault("train_per_class", TRAIN_POOL_PER_CLASS)

        protocol.setdefault("images_per_class", TRAIN_POOL_PER_CLASS + QUERY_PER_CLASS)

    # --------------------------------------------------------
    # Support common key names
    # --------------------------------------------------------

    database_indices = None
    query_indices = None

    for key in [
        "train_indices",
        "train_pool_indices",
        "database_indices",
        "db_indices",
    ]:
        if key in protocol:
            database_indices = protocol[key]
            break

    for key in [
        "query_indices",
        "val_indices",
        "test_indices",
    ]:
        if key in protocol:
            query_indices = protocol[key]
            break

    if database_indices is None:
        raise KeyError(
            "Cannot find database indices in protocol."
        )

    if query_indices is None:
        raise KeyError(
            "Cannot find query indices in protocol."
        )

    database_indices = np.asarray(
        database_indices,
        dtype=np.int64,
    )

    query_indices = np.asarray(
        query_indices,
        dtype=np.int64,
    )

    # --------------------------------------------------------
    # Check sizes
    # --------------------------------------------------------

    if len(database_indices) != NUM_DATABASE:
        raise RuntimeError(
            f"Expected {NUM_DATABASE} database images, "
            f"got {len(database_indices)}"
        )

    if len(query_indices) != NUM_QUERY:
        raise RuntimeError(
            f"Expected {NUM_QUERY} query images, "
            f"got {len(query_indices)}"
        )

    # --------------------------------------------------------
    # Check overlap
    # --------------------------------------------------------

    overlap = np.intersect1d(
        database_indices,
        query_indices,
    )

    if len(overlap) != 0:

        raise RuntimeError(
            f"Database/query overlap detected: "
            f"{len(overlap)} images"
        )

    print(
        f"Database images : "
        f"{len(database_indices):,}"
    )

    print(
        f"Query images    : "
        f"{len(query_indices):,}"
    )

    print(
        f"Overlap         : "
        f"{len(overlap):,}"
    )

    print(
        f"Seed            : "
        f"{SEED}"
    )

    print("=" * 100)

    return (
        database_indices,
        query_indices,
    )


# ============================================================
# 5. Load RAEv2
# ============================================================

def load_model():

    print()
    print("=" * 100)
    print("LOADING RAEv2 DINOv3-L K7")
    print("=" * 100)

    print(
        f"Model directory:\n"
        f"{MODEL_DIR}"
    )

    # ========================================================
    # Transformers 5.x compatibility
    # ========================================================

    if not hasattr(
        nn.Module,
        "all_tied_weights_keys"
    ):

        nn.Module.all_tied_weights_keys = {}

    from transformers import AutoModel

    # ========================================================
    # Official/local RAEv2 loading
    # ========================================================

    model = AutoModel.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        local_files_only=True,
    )

    # ========================================================
    # RAEv2 / Transformers DINOv3 compatibility
    #
    # RAEv2 expects:
    #
    #     backbone.layer
    #
    # Current Transformers:
    #
    #     backbone.model.layer
    # ========================================================

    backbone = model.encoder.backbone

    if (
        not hasattr(
            backbone,
            "layer",
        )
        and hasattr(
            backbone,
            "model",
        )
        and hasattr(
            backbone.model,
            "layer",
        )
    ):

        backbone.layer = (
            backbone.model.layer
        )

        print(
            "[Compatibility] "
            "Mapped backbone.model.layer "
            "-> backbone.layer"
        )

    # ========================================================
    # Sanity checks
    # ========================================================

    assert hasattr(
        backbone,
        "layer",
    ), (
        "ERROR: DINOv3 backbone.layer "
        "not found."
    )

    assert len(
        backbone.layer
    ) == 24, (
        "ERROR: Expected 24 DINOv3 layers, "
        f"got {len(backbone.layer)}"
    )

    # ========================================================
    # RAEv2 K=7
    # ========================================================

    expected_layers = [
        11,
        13,
        15,
        17,
        19,
        21,
        23,
    ]

    actual_layers = (
        model.config.encoder_layer_indices
    )

    assert (
        list(actual_layers)
        == expected_layers
    ), (
        "Unexpected RAEv2 layer configuration: "
        f"{actual_layers}"
    )

    # ========================================================
    # Device
    # ========================================================

    model = model.to(
        DEVICE
    )

    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    # ========================================================
    # Print information
    # ========================================================

    print()
    print("Model class:")
    print(type(model))

    print()
    print("Encoder layer indices:")
    print(actual_layers)

    print()
    print("Hidden size:")
    print(model.config.hidden_size)

    print()
    print("Image size:")
    print(model.config.image_size)

    print()
    print("Patch size:")
    print(model.config.patch_size)

    print()
    print("Device:")
    print(DEVICE)

    print()
    print("Feature:")
    print("RAEv2 latent [1024,16,16]")
    print("Global Average Pooling -> 1024-d")
    print("L2 normalization -> cosine similarity")

    print()
    print("Model loaded successfully.")

    print("=" * 100)

    return model


# ============================================================
# 6. Feature extraction
# ============================================================

@torch.no_grad()
def extract_features(
    model,
    dataset,
    desc,
):

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(
            NUM_WORKERS > 0
        ),
    )

    features = []
    labels = []

    total_images = 0
    total_batches = len(loader)

    start_time = time.time()

    print()
    print("=" * 100)
    print(desc)
    print("=" * 100)

    print(
        f"Total images : {len(dataset):,}"
    )

    print(
        f"Total batches: {total_batches:,}"
    )

    print()

    for batch_idx, batch in enumerate(
        loader
    ):

        images, batch_labels = batch

        images = images.to(
            DEVICE,
            non_blocking=True,
        )

        # ====================================================
        # RAEv2 encode
        #
        # [B,3,256,256]
        #       ↓
        # [B,1024,16,16]
        # ====================================================

        z = model.encode(
            images
        )

        if not torch.is_tensor(z):

            raise RuntimeError(
                "RAEv2 encode() returned "
                f"unexpected type: {type(z)}"
            )

        if z.ndim != 4:

            raise RuntimeError(
                "Unexpected latent shape: "
                f"{z.shape}"
            )

        if z.shape[1:] != (
            1024,
            16,
            16,
        ):

            raise RuntimeError(
                "Unexpected RAEv2 latent shape: "
                f"{z.shape}"
            )

        # ====================================================
        # GAP
        #
        # [B,1024,16,16]
        #       ↓
        # [B,1024]
        # ====================================================

        feat = z.mean(
            dim=(2, 3)
        )

        # ====================================================
        # L2 normalization
        #
        # Unified KNN protocol
        # ====================================================

        feat = torch.nn.functional.normalize(
            feat.float(),
            p=2,
            dim=1,
        )

        features.append(
            feat.cpu().numpy()
        )

        labels.append(
            batch_labels.numpy()
        )

        total_images += len(
            batch_labels
        )

        # ====================================================
        # Progress
        # ====================================================

        if (
            batch_idx == 0
            or (batch_idx + 1) % 10 == 0
            or batch_idx + 1 == total_batches
        ):

            elapsed = (
                time.time()
                - start_time
            )

            speed = (
                total_images / elapsed
                if elapsed > 0
                else 0
            )

            remaining = max(
                len(dataset)
                - total_images,
                0,
            )

            eta_seconds = (
                remaining / speed
                if speed > 0
                else 0
            )

            percent = (
                total_images
                / len(dataset)
                * 100
            )

            print(
                f"[{batch_idx + 1:,}/"
                f"{total_batches:,}] "
                f"{percent:6.2f}% | "
                f"{total_images:,}/"
                f"{len(dataset):,} | "
                f"{speed:.1f} img/s | "
                f"ETA "
                f"{eta_seconds / 60:.1f} min",
                flush=True,
            )

    features = np.concatenate(
        features,
        axis=0,
    )

    labels = np.concatenate(
        labels,
        axis=0,
    )

    # ========================================================
    # Sanity
    # ========================================================

    if len(features) != len(dataset):

        raise RuntimeError(
            "Feature count mismatch: "
            f"{len(features)} vs "
            f"{len(dataset)}"
        )

    print()
    print(
        "Feature extraction complete."
    )

    print(
        "Features:",
        features.shape,
    )

    print(
        "Labels:",
        labels.shape,
    )

    print(
        "dtype:",
        features.dtype,
    )

    print(
        "Feature memory:",
        f"{features.nbytes / (1024 ** 2):.2f} MB",
    )

    return (
        features,
        labels,
    )


# ============================================================
# 7. Save / load cache
# ============================================================

def save_cache(
    features,
    labels,
    feature_path,
    label_path,
):

    np.save(
        feature_path,
        features,
    )

    np.save(
        label_path,
        labels,
    )

    print()
    print("Saved cache:")
    print(
        f"  {feature_path}"
    )
    print(
        f"  {label_path}"
    )


def load_cache(
    feature_path,
    label_path,
):

    features = np.load(
        feature_path
    )

    labels = np.load(
        label_path
    )

    print(
        f"Loaded features: "
        f"{features.shape}"
    )

    print(
        f"Loaded labels: "
        f"{labels.shape}"
    )

    return (
        features,
        labels,
    )


# ============================================================
# 8. FLOPs profiling
# ============================================================

@torch.no_grad()
def profile_model_flops(
    model,
    sample_images,
):

    print()
    print("=" * 100)
    print("PROFILING RAEv2 FLOPs")
    print("=" * 100)

    print(
        f"Profile batch size: "
        f"{sample_images.shape[0]}"
    )

    print(
        f"Input shape: "
        f"{tuple(sample_images.shape)}"
    )

    if DEVICE != "cuda":

        print(
            "[WARNING] CUDA not available. "
            "FLOPs profiling will use CPU."
        )

    # Warmup
    for _ in range(2):

        _ = model.encode(
            sample_images
        )

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    # --------------------------------------------------------
    # torch profiler
    # --------------------------------------------------------

    from torch.profiler import (
        profile,
        ProfilerActivity,
    )

    activities = [
        ProfilerActivity.CPU
    ]

    if DEVICE == "cuda":
        activities.append(
            ProfilerActivity.CUDA
        )

    with profile(
        activities=activities,
        record_shapes=False,
        profile_memory=False,
        with_flops=True,
    ) as prof:

        _ = model.encode(
            sample_images
        )

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    total_flops = 0

    for event in prof.key_averages():

        flops = getattr(
            event,
            "flops",
            0,
        )

        if flops is not None:
            total_flops += flops

    batch_size = sample_images.shape[0]

    flops_per_image = (
        total_flops
        / batch_size
    )

    print()
    print(
        f"Profiler FLOPs / batch : "
        f"{total_flops:,.0f}"
    )

    print(
        f"Profiler FLOPs / image : "
        f"{flops_per_image:,.0f}"
    )

    print(
        f"FLOPs / image          : "
        f"{flops_per_image / 1e12:.6f} TFLOPs"
    )

    print("=" * 100)

    return {
        "profile_batch_size": int(
            batch_size
        ),
        "model_flops_per_batch": float(
            total_flops
        ),
        "model_flops_per_image": float(
            flops_per_image
        ),
    }


# ============================================================
# 9. Unified KNN FLOPs
# ============================================================

def calculate_total_flops(
    model_flops_per_image,
    database_size,
    query_size,
    feature_dim,
):

    # --------------------------------------------------------
    # Feature extraction
    #
    # Every database and query image is encoded.
    # --------------------------------------------------------

    num_images = (
        database_size
        + query_size
    )

    feature_extraction_flops = (
        num_images
        * model_flops_per_image
    )

    # --------------------------------------------------------
    # Exact exhaustive similarity search
    #
    # For inner product:
    #
    # 1 multiplication + 1 addition
    # per dimension
    #
    # => approximately 2 * D FLOPs
    # --------------------------------------------------------

    knn_search_flops = (
        2
        * query_size
        * database_size
        * feature_dim
    )

    total_flops = (
        feature_extraction_flops
        + knn_search_flops
    )

    return {
        "database_size": int(
            database_size
        ),
        "query_size": int(
            query_size
        ),
        "feature_dim": int(
            feature_dim
        ),
        "feature_extraction_flops": float(
            feature_extraction_flops
        ),
        "knn_search_flops": float(
            knn_search_flops
        ),
        "total_flops": float(
            total_flops
        ),
        "feature_extraction_pflops": (
            feature_extraction_flops
            / 1e15
        ),
        "knn_search_pflops": (
            knn_search_flops
            / 1e15
        ),
        "total_pflops": (
            total_flops
            / 1e15
        ),
    }


# ============================================================
# 10. Unified KNN evaluation
# ============================================================

def evaluate_knn(
    database_features,
    database_labels,
    query_features,
    query_labels,
):

    print()
    print("=" * 100)
    print("UNIFIED KNN EVALUATION")
    print("=" * 100)

    print(
        f"Database : "
        f"{database_features.shape}"
    )

    print(
        f"Query    : "
        f"{query_features.shape}"
    )

    print(
        f"K values : "
        f"{K_VALUES}"
    )

    print(
        "Metric   : "
        "cosine similarity "
        "(L2-normalized inner product)"
    )

    print(
        "Weight   : "
        "exp((sim-max_sim)/temperature)"
    )

    print(
        f"Temperature: "
        f"{TEMPERATURE}"
    )

    # ========================================================
    # FAISS
    # ========================================================

    database_features = np.ascontiguousarray(
        database_features.astype(
            np.float32
        )
    )

    query_features = np.ascontiguousarray(
        query_features.astype(
            np.float32
        )
    )

    index = faiss.IndexFlatIP(
        database_features.shape[1]
    )

    index.add(
        database_features
    )

    max_k = max(
        K_VALUES
    )

    print()
    print(
        f"FAISS index size: "
        f"{index.ntotal:,}"
    )

    print(
        f"Searching top-{max_k}..."
    )

    search_start = time.time()

    similarities, indices = index.search(
        query_features,
        max_k,
    )

    search_time = (
        time.time()
        - search_start
    )

    neighbor_labels = (
        database_labels[indices]
    )

    print(
        f"Search time: "
        f"{search_time:.3f} sec"
    )

    # ========================================================
    # Evaluate all K
    # ========================================================

    results = {}

    for k in K_VALUES:

        sims = similarities[:, :k]

        labels = (
            neighbor_labels[:, :k]
        )

        correct_top1 = 0
        correct_top5 = 0
        correct_top10 = 0
        correct_top20 = 0

        # ----------------------------------------------------
        # Similarity weighted voting
        # ----------------------------------------------------

        for i in range(
            len(query_labels)
        ):

            sample_sims = sims[i]

            sample_labels = labels[i]

            # Numerical stabilization
            max_sim = np.max(
                sample_sims
            )

            weights = np.exp(
                (
                    sample_sims
                    - max_sim
                )
                / TEMPERATURE
            )

            class_scores = {}

            for j in range(k):

                label = int(
                    sample_labels[j]
                )

                weight = float(
                    weights[j]
                )

                class_scores[label] = (
                    class_scores.get(
                        label,
                        0.0,
                    )
                    + weight
                )

            ranked_classes = sorted(
                class_scores.items(),
                key=lambda x: x[1],
                reverse=True,
            )

            ranked_labels = [
                label
                for label, _ in ranked_classes
            ]

            target = int(
                query_labels[i]
            )

            if (
                target
                in ranked_labels[:1]
            ):
                correct_top1 += 1

            if (
                target
                in ranked_labels[:5]
            ):
                correct_top5 += 1

            if (
                target
                in ranked_labels[:10]
            ):
                correct_top10 += 1

            if (
                target
                in ranked_labels[:20]
            ):
                correct_top20 += 1

        n = len(
            query_labels
        )

        results[str(k)] = {
            "top1": (
                correct_top1
                / n
                * 100.0
            ),
            "top5": (
                correct_top5
                / n
                * 100.0
            ),
            "top10": (
                correct_top10
                / n
                * 100.0
            ),
            "top20": (
                correct_top20
                / n
                * 100.0
            ),
        }

    # ========================================================
    # Print results
    # ========================================================

    print()
    print("=" * 100)
    print("FINAL KNN RESULTS")
    print("=" * 100)

    print(
        "Representation : "
        "RAEv2 latent + GAP + L2"
    )

    print(
        f"Dimension      : "
        f"{database_features.shape[1]}"
    )

    print(
        f"Database       : "
        f"{len(database_labels):,}"
    )

    print(
        f"Query          : "
        f"{len(query_labels):,}"
    )

    print(
        "Metric         : "
        "cosine / IndexFlatIP"
    )

    print(
        "Weight         : "
        f"exp((sim-max_sim)/{TEMPERATURE})"
    )

    print()

    for k in K_VALUES:

        r = results[str(k)]

        print(
            f"K={k:2d} | "
            f"Top-1 {r['top1']:8.4f}% | "
            f"Top-5 {r['top5']:8.4f}% | "
            f"Top-10 {r['top10']:8.4f}% | "
            f"Top-20 {r['top20']:8.4f}%"
        )

    print("=" * 100)

    return results


# ============================================================
# 11. Main
# ============================================================

def _single_pool_main():

    print()
    print("=" * 100)
    print("RAEv2 DINOv3-L K7 - IMAGENET-TRAIN 195-POOL/5-QUERY KNN")
    print("=" * 100)

    print()
    print("Model:")
    print(
        MODEL_DIR
    )

    print()
    print("ImageNet:")
    print(
        IMAGENET_TRAIN_DIR
    )

    print()
    print("Protocol:")
    print(
        PROTOCOL_FILE
    )

    print()
    print("Protocol settings:")
    print(
        f"  Database pool  : {NUM_DATABASE:,}"
    )
    print(
        "  Query          : 5,000"
    )
    print(
        "  Overlap        : 0"
    )
    print(
        "  Seed           : 42"
    )

    print()
    print("Preprocessing:")
    print(
        "  Resize         : 288"
    )
    print(
        "  Center crop    : 256"
    )
    print(
        "  Normalization  : none (RAEv2 requires [0,1])"
    )

    print()
    print("Representation:")
    print(
        "  RAEv2 encode()"
    )
    print(
        "  [B,1024,16,16]"
    )
    print(
        "  GAP -> 1024-d"
    )
    print(
        "  L2 normalization"
    )

    print()
    print("KNN:")
    print(
        "  FAISS IndexFlatIP"
    )
    print(
        "  Exact exhaustive search"
    )
    print(
        "  Temperature = 0.07"
    )
    print(
        "  K = [1,5,10,20]"
    )

    print("=" * 100)

    # ========================================================
    # Load protocol
    # ========================================================

    (
        database_indices,
        query_indices,
    ) = load_protocol()

    # ========================================================
    # Load dataset
    # ========================================================

    print()
    print("=" * 100)
    print("LOADING IMAGENET")
    print("=" * 100)

    dataset = datasets.ImageFolder(
        IMAGENET_TRAIN_DIR,
        transform=transform,
    )

    print(
        f"Images : "
        f"{len(dataset):,}"
    )

    print(
        f"Classes: "
        f"{len(dataset.classes):,}"
    )

    # --------------------------------------------------------
    # Protocol index sanity check
    # --------------------------------------------------------

    max_index = max(
        database_indices.max(),
        query_indices.max(),
    )

    if max_index >= len(dataset):

        raise RuntimeError(
            "Protocol index exceeds ImageNet dataset size: "
            f"max index={max_index}, "
            f"dataset size={len(dataset)}"
        )

    # ========================================================
    # Build subsets
    # ========================================================

    database_dataset = IndexedSubset(
        dataset,
        database_indices,
    )

    query_dataset = IndexedSubset(
        dataset,
        query_indices,
    )

    print()
    print(
        f"Database dataset: "
        f"{len(database_dataset):,}"
    )

    print(
        f"Query dataset   : "
        f"{len(query_dataset):,}"
    )

    # ========================================================
    # Load model
    # ========================================================

    model = load_model()

    # ========================================================
    # FLOPs profiling
    # ========================================================

    print()
    print("=" * 100)
    print("FLOPs PROFILING")
    print("=" * 100)

    profile_loader = DataLoader(
        database_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    sample_images, _ = next(
        iter(profile_loader)
    )

    sample_images = sample_images.to(
        DEVICE
    )

    flops_info = profile_model_flops(
        model,
        sample_images,
    )

    # ========================================================
    # FLOPs calculation
    # ========================================================

    flops_total = calculate_total_flops(
        model_flops_per_image=(
            flops_info[
                "model_flops_per_image"
            ]
        ),
        database_size=NUM_DATABASE,
        query_size=NUM_QUERY,
        feature_dim=FEATURE_DIM,
    )

    print()
    print("=" * 100)
    print("FLOPs BREAKDOWN")
    print("=" * 100)

    print(
        f"Model FLOPs / image : "
        f"{flops_info['model_flops_per_image'] / 1e12:.6f} TFLOPs"
    )

    print(
        f"Feature extraction  : "
        f"{flops_total['feature_extraction_pflops']:.6f} PFLOPs"
    )

    print(
        f"KNN search          : "
        f"{flops_total['knn_search_pflops']:.6f} PFLOPs"
    )

    print(
        f"TOTAL KNN FLOPs     : "
        f"{flops_total['total_pflops']:.6f} PFLOPs"
    )

    print("=" * 100)

    # ========================================================
    # Save FLOPs
    # ========================================================

    with open(
        FLOPS_FILE,
        "w",
    ) as f:

        json.dump(
            flops_total,
            f,
            indent=2,
        )

    # ========================================================
    # Database features
    # ========================================================

    if (
        os.path.exists(
            DATABASE_FEATURES
        )
        and os.path.exists(
            DATABASE_LABELS
        )
    ):

        print()
        print("=" * 100)
        print("DATABASE CACHE FOUND")
        print("=" * 100)

        (
            database_features,
            database_labels,
        ) = load_cache(
            DATABASE_FEATURES,
            DATABASE_LABELS,
        )

    else:

        (
            database_features,
            database_labels,
        ) = extract_features(
            model,
            database_dataset,
            "DATABASE FEATURE EXTRACTION",
        )

        save_cache(
            database_features,
            database_labels,
            DATABASE_FEATURES,
            DATABASE_LABELS,
        )

    # ========================================================
    # Query features
    # ========================================================

    if (
        os.path.exists(
            QUERY_FEATURES
        )
        and os.path.exists(
            QUERY_LABELS
        )
    ):

        print()
        print("=" * 100)
        print("QUERY CACHE FOUND")
        print("=" * 100)

        (
            query_features,
            query_labels,
        ) = load_cache(
            QUERY_FEATURES,
            QUERY_LABELS,
        )

    else:

        (
            query_features,
            query_labels,
        ) = extract_features(
            model,
            query_dataset,
            "QUERY FEATURE EXTRACTION",
        )

        save_cache(
            query_features,
            query_labels,
            QUERY_FEATURES,
            QUERY_LABELS,
        )

    # ========================================================
    # Cache sanity
    # ========================================================

    assert database_features.shape == (
        NUM_DATABASE,
        FEATURE_DIM,
    )

    assert query_features.shape == (
        NUM_QUERY,
        FEATURE_DIM,
    )

    assert len(
        database_labels
    ) == NUM_DATABASE

    assert len(
        query_labels
    ) == NUM_QUERY

    # ========================================================
    # KNN
    # ========================================================

    results = evaluate_knn(
        database_features,
        database_labels,
        query_features,
        query_labels,
    )

    # ========================================================
    # Final output
    # ========================================================

    output = {
        "model": "RAEv2-DINOv3-L-K7",
        "model_dir": MODEL_DIR,
        "protocol": PROTOCOL_FILE,

        "database_size": NUM_DATABASE,
        "query_size": NUM_QUERY,
        "seed": SEED,

        "input_size": IMG_SIZE,
        "resize_size": RESIZE_SIZE,

        "representation": (
            "RAEv2 encode -> GAP -> L2"
        ),

        "feature_dim": FEATURE_DIM,

        "metric": (
            "cosine similarity / FAISS IndexFlatIP"
        ),

        "temperature": TEMPERATURE,

        "k_values": K_VALUES,

        "results": results,

        "flops": flops_total,

        "model_flops_per_image": (
            flops_info[
                "model_flops_per_image"
            ]
        ),
    }

    with open(
        RESULT_FILE,
        "w",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
        )

    print()
    print("=" * 100)
    print("DONE")
    print("=" * 100)

    print()
    print("KNN results:")
    print()

    for k in K_VALUES:

        r = results[str(k)]

        print(
            f"K={k:2d} | "
            f"Top-1  = {r['top1']:.4f}% | "
            f"Top-5  = {r['top5']:.4f}% | "
            f"Top-10 = {r['top10']:.4f}% | "
            f"Top-20 = {r['top20']:.4f}%"
        )

    print()
    print(
        f"Model FLOPs/image : "
        f"{flops_info['model_flops_per_image'] / 1e12:.6f} TFLOPs"
    )

    print(
        f"Feature extraction: "
        f"{flops_total['feature_extraction_pflops']:.6f} PFLOPs"
    )

    print(
        f"KNN search        : "
        f"{flops_total['knn_search_pflops']:.6f} PFLOPs"
    )

    print(
        f"TOTAL KNN FLOPs    : "
        f"{flops_total['total_pflops']:.6f} PFLOPs"
    )

    print()
    print(
        f"Results saved to:\n"
        f"{RESULT_FILE}"
    )

    print(
        f"FLOPs saved to:\n"
        f"{FLOPS_FILE}"
    )

    print(
        f"Cache directory:\n"
        f"{CACHE_DIR}"
    )

    print("=" * 100)


# ============================================================
# Entry
# ============================================================

if False:  # entry point is defined below
    _single_pool_main()
def main():
    """Extract the 195k pool once, then evaluate 5/10/20/45/95/195-shot slices."""
    _single_pool_main()
    from vision_encoder_eval.workers.knn.different_shot.multishot_protocol import run_multishot_from_cache
    return run_multishot_from_cache(
        CACHE_DIR, CACHE_DIR, "rae", model_name="RAEv2-DINOv3-L-K7",
        protocol_path=PROTOCOL_FILE, shots=SHOT_VALUES,
        pool_per_class=TRAIN_POOL_PER_CLASS,
        query_per_class=QUERY_PER_CLASS, num_classes=NUM_CLASSES,
    )


if __name__ == "__main__":
    main()

