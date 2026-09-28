import os
import sys
import json
import time
import warnings
import gc

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import faiss


# ============================================================
# 1. Settings
# ============================================================

REPO_DIR = (
    "/cache/models/repositories/UniAR"
)

CKPT_PATH = (
    "/cache/models/model/uniar_bsq/"
    "bsq_encoder"
)

# ImageNet official validation set
IMAGENET_ROOT = (
    "/workspace/root/val"
)

# ============================================================
# IMPORTANT:
# Use exactly the same protocol as the SigLIP experiments.
# ============================================================

PROTOCOL_FILE = "/cache/metaclip_knn/val_45shot_5query_seed42_protocol.json"

CACHE_DIR = (
    "/cache/uniar_bsq_knn_cache"
)

os.makedirs(
    CACHE_DIR,
    exist_ok=True
)


# ============================================================
# KNN protocol
# ============================================================

SEED = 42

K_VALUES = [
    1,
    5,
    10,
    20,
]

TEMPERATURE = 0.07

BATCH_SIZE = 256

NUM_WORKERS = 8


# ============================================================
# UniAR image preprocessing
# ============================================================

IMG_SIZE = 256

RESIZE_SIZE = int(
    IMG_SIZE * 1.125
)

BSQ_FEATURE_LEVEL = 0


# ============================================================
# Feature storage
# ============================================================

STORE_DTYPE = np.float16


# ============================================================
# FAISS
# ============================================================

FAISS_USE_GPU = True


# ============================================================
# Device
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Cache files
# ============================================================

DATABASE_FEATURE_FILE = os.path.join(
    CACHE_DIR,
    "protocol_database_features_fp16.npy",
)

DATABASE_LABEL_FILE = os.path.join(
    CACHE_DIR,
    "protocol_database_labels.npy",
)

QUERY_FEATURE_FILE = os.path.join(
    CACHE_DIR,
    "protocol_query_features_fp16.npy",
)

QUERY_LABEL_FILE = os.path.join(
    CACHE_DIR,
    "protocol_query_labels.npy",
)

RESULT_FILE = os.path.join(
    CACHE_DIR,
    "protocol_knn_result.json",
)


# ============================================================
# 2. Import UniAR
# ============================================================

sys.path.insert(
    0,
    REPO_DIR
)

from uniar.vision_encoder.modeling_vision_encoder import (
    load_bsq_image_tokenizer_and_transform,
)


# ============================================================
# 3. Seed
# ============================================================

def seed_everything(
    seed=42
):

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed(
            seed
        )

        torch.cuda.manual_seed_all(
            seed
        )


seed_everything(
    SEED
)


# ============================================================
# 4. Transform
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        RESIZE_SIZE,
        interpolation=(
            transforms.InterpolationMode.BICUBIC
        ),
    ),

    transforms.CenterCrop(
        IMG_SIZE
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5],
    ),

])


# ============================================================
# 5. Safe ImageFolder
# ============================================================

class SafeImageFolder(
    datasets.ImageFolder
):

    def __init__(
        self,
        root,
        transform=None,
    ):

        print(
            "=" * 100
        )

        print(
            "Checking ImageNet images"
        )

        print(
            "=" * 100
        )

        super().__init__(
            root=root,
            transform=transform,
        )

        original_samples = (
            len(self.samples)
        )

        print(
            f"Original samples : "
            f"{original_samples:,}"
        )

        valid_samples = []

        bad_samples = []

        start_time = time.time()

        for idx, (
            path,
            target,
        ) in enumerate(
            self.samples
        ):

            try:

                with Image.open(
                    path
                ) as img:

                    img.verify()

                valid_samples.append(
                    (
                        path,
                        target
                    )
                )

            except Exception as e:

                bad_samples.append(
                    (
                        path,
                        target,
                        repr(e)
                    )
                )

            if (
                (idx + 1) % 10000
                == 0
            ):

                elapsed = (
                    time.time()
                    - start_time
                )

                rate = (
                    (idx + 1)
                    / max(
                        elapsed,
                        1e-6
                    )
                )

                print(
                    f"[CHECK] "
                    f"{idx + 1:,}/"
                    f"{original_samples:,} "
                    f"("
                    f"{100.0 * (idx + 1) / original_samples:.2f}%"
                    f") | "
                    f"{rate:.1f} img/s | "
                    f"bad="
                    f"{len(bad_samples):,}"
                )

        self.samples = (
            valid_samples
        )

        self.targets = [
            target
            for _, target
            in valid_samples
        ]

        print()

        print(
            "=" * 100
        )

        print(
            "Image integrity check finished"
        )

        print(
            "=" * 100
        )

        print(
            f"Original samples : "
            f"{original_samples:,}"
        )

        print(
            f"Valid samples    : "
            f"{len(valid_samples):,}"
        )

        print(
            f"Bad samples      : "
            f"{len(bad_samples):,}"
        )

        if bad_samples:

            bad_file = os.path.join(
                CACHE_DIR,
                "bad_images.txt",
            )

            with open(
                bad_file,
                "w",
            ) as f:

                for (
                    path,
                    target,
                    error,
                ) in bad_samples:

                    f.write(
                        f"{path}\t"
                        f"class={target}\t"
                        f"{error}\n"
                    )

            print(
                f"Bad image list saved to:"
            )

            print(
                bad_file
            )

        else:

            print(
                "No corrupted images found."
            )

        print(
            "=" * 100
        )


# ============================================================
# 6. Load fixed protocol
# ============================================================

def load_fixed_protocol():

    print()
    print(
        "=" * 100
    )

    print(
        "Loading fixed KNN protocol"
    )

    print(
        "=" * 100
    )

    print(
        "Protocol:"
    )

    print(
        PROTOCOL_FILE
    )

    if not os.path.exists(
        PROTOCOL_FILE
    ):

        raise FileNotFoundError(
            f"\nProtocol file does not exist:\n"
            f"{PROTOCOL_FILE}"
        )

    with open(
        PROTOCOL_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        protocol = json.load(
            f
        )

    print()

    print(
        "Protocol keys:"
    )

    print(
        list(
            protocol.keys()
        )
    )

    # --------------------------------------------------------
    # Database
    # --------------------------------------------------------

    if "train_pool_indices" in protocol:
        database_indices = protocol["train_pool_indices"]

    elif (
        "train_indices"
        in protocol
    ):

        database_indices = (
            protocol[
                "train_indices"
            ]
        )

    elif (
        "database_indices"
        in protocol
    ):

        database_indices = (
            protocol[
                "database_indices"
            ]
        )

    else:

        raise KeyError(
            "Protocol does not contain "
            "'train_indices' or "
            "'database_indices'."
        )

    # --------------------------------------------------------
    # Query
    # --------------------------------------------------------

    if (
        "query_indices"
        in protocol
    ):

        query_indices = (
            protocol[
                "query_indices"
            ]
        )

    elif (
        "val_indices"
        in protocol
    ):

        query_indices = (
            protocol[
                "val_indices"
            ]
        )

    else:

        raise KeyError(
            "Protocol does not contain "
            "'query_indices' or "
            "'val_indices'."
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

    print()

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

    return (
        database_indices,
        query_indices,
        protocol,
    )

# ============================================================
# 7. Runtime patch
# ============================================================

def patch_uniar_positional_embedding_dtype(
    model
):

    original_pos_embed = (
        model.fast_pos_embed_interpolate
    )

    if hasattr(
        model.patch_embed,
        "proj",
    ):

        target_dtype = (
            model.patch_embed
            .proj
            .weight
            .dtype
        )

    else:

        target_dtype = next(
            model.parameters()
        ).dtype

    print(
        "Target positional embedding dtype:",
        target_dtype,
    )

    def patched_pos_embed(
        grid_thw
    ):

        pos = (
            original_pos_embed(
                grid_thw
            )
        )

        pos = pos.to(
            dtype=target_dtype,
            device=grid_thw.device,
        )

        return pos

    model.fast_pos_embed_interpolate = (
        patched_pos_embed
    )

    return model


# ============================================================
# 8. Load ImageNet
# ============================================================

print()
print(
    "=" * 100
)

print(
    "Loading ImageNet"
)

print(
    "=" * 100
)

base_dataset = SafeImageFolder(
    IMAGENET_ROOT,
    transform=transform,
)

num_classes = len(
    base_dataset.classes
)

print()

print(
    f"Dataset size : "
    f"{len(base_dataset):,}"
)

print(
    f"Num classes  : "
    f"{num_classes:,}"
)


# ============================================================
# 9. Load fixed protocol
# ============================================================

(
    database_indices,
    query_indices,
    protocol,
) = load_fixed_protocol()


# ============================================================
# 10. Verify protocol compatibility
# ============================================================

print()
print(
    "=" * 100
)

print(
    "Checking protocol / dataset compatibility"
)

print(
    "=" * 100
)

max_database_idx = int(
    database_indices.max()
)

max_query_idx = int(
    query_indices.max()
)

max_protocol_idx = max(
    max_database_idx,
    max_query_idx,
)

print(
    f"Dataset length    : "
    f"{len(base_dataset):,}"
)

print(
    f"Max protocol index: "
    f"{max_protocol_idx:,}"
)

if (
    max_protocol_idx
    >= len(base_dataset)
):

    raise RuntimeError(
        "\nThe saved protocol is incompatible "
        "with the current ImageFolder ordering.\n"
        f"Dataset length = "
        f"{len(base_dataset):,}\n"
        f"Max protocol index = "
        f"{max_protocol_idx:,}\n"
    )

print(
    "[OK] Protocol indices are valid."
)


# ============================================================
# 11. Verify expected 5-shot protocol
# ============================================================

if (
    len(database_indices)
    != 5000
):

    print(
        "[WARNING] Database size is not 5000:"
    )

    print(
        len(database_indices)
    )

if (
    len(query_indices)
    != 5000
):

    print(
        "[WARNING] Query size is not 5000:"
    )

    print(
        len(query_indices)
    )

print()

print(
    "Expected aligned protocol:"
)

print(
    "Database = 5000"
)

print(
    "Query    = 5000"
)

print(
    "Seed     = 42"
)


# ============================================================
# 12. Build subsets
# ============================================================

database_dataset = Subset(
    base_dataset,
    database_indices,
)

query_dataset = Subset(
    base_dataset,
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


# ============================================================
# 13. DataLoaders
# ============================================================

database_loader = DataLoader(
    database_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=(
        NUM_WORKERS > 0
    ),
    drop_last=False,
)

query_loader = DataLoader(
    query_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=(
        NUM_WORKERS > 0
    ),
    drop_last=False,
)


# ============================================================
# 14. Load UniAR BSQ encoder
# ============================================================

print()
print(
    "=" * 100
)

print(
    "Loading UniAR BSQ encoder"
)

print(
    "=" * 100
)

print(
    f"Checkpoint: "
    f"{CKPT_PATH}"
)

print(
    f"Feature level: "
    f"{BSQ_FEATURE_LEVEL}"
)

# ------------------------------------------------------------
# Keep UniAR official loading method.
# ------------------------------------------------------------

model = (
    load_bsq_image_tokenizer_and_transform(
        CKPT_PATH,
        feature_level=BSQ_FEATURE_LEVEL,
        no_merger=True,
    )
)

model = model.to(
    DEVICE
)

model.eval()

model = (
    patch_uniar_positional_embedding_dtype(
        model
    )
)

print()

print(
    "Model dtype:",
    next(
        model.parameters()
    ).dtype,
)

if hasattr(
    model.patch_embed,
    "proj",
):

    print(
        "patch_embed.proj.weight:",
        model.patch_embed
        .proj
        .weight
        .dtype,
    )


# ============================================================
# 15. UniAR feature extraction
# ============================================================

@torch.no_grad()
def extract_features(
    loader,
    split_name,
):

    print()
    print(
        "=" * 100
    )

    print(
        f"Extracting {split_name} features"
    )

    print(
        "=" * 100
    )

    total = len(
        loader.dataset
    )

    all_features = []
    all_labels = []

    processed = 0

    start_time = time.time()

    for batch_idx, (
        images,
        labels,
    ) in enumerate(
        loader
    ):

        # ----------------------------------------------------
        # Image -> BF16
        # ----------------------------------------------------

        images = images.to(
            DEVICE,
            dtype=torch.bfloat16,
            non_blocking=True,
        )

        batch_size = (
            images.shape[0]
        )

        labels_np = (
            labels.numpy()
        )

        # ----------------------------------------------------
        # Image -> patches
        # ----------------------------------------------------

        (
            flatten_image,
            grid_thw,
        ) = model.convert_img_to_patch(
            images
        )

        # ----------------------------------------------------
        # UniAR BSQ
        # ----------------------------------------------------

        output = model(
            flatten_image,
            grid_thw=grid_thw,
            bsq_only=True,
            bsq_feature_level=(
                BSQ_FEATURE_LEVEL
            ),
            direct_indices=False,
        )

        features = output[0]

        # ----------------------------------------------------
        # [B*N, D]
        #
        # -> [B, N, D]
        #
        # -> mean pooling
        #
        # -> [B, D]
        # ----------------------------------------------------

        feature_dim = (
            features.shape[-1]
        )

        num_tokens = (
            features.shape[0]
            // batch_size
        )

        features = features.reshape(
            batch_size,
            num_tokens,
            feature_dim,
        )

        features = features.mean(
            dim=1
        )

        # ----------------------------------------------------
        # L2 normalization
        #
        # This is important:
        # aligned with SigLIP KNN protocol.
        # ----------------------------------------------------

        features = F.normalize(
            features.float(),
            p=2,
            dim=1,
        )

        # ----------------------------------------------------
        # CPU FP16 storage
        # ----------------------------------------------------

        features_np = (
            features
            .cpu()
            .numpy()
            .astype(
                STORE_DTYPE
            )
        )

        all_features.append(
            features_np
        )

        all_labels.append(
            labels_np
        )

        processed += batch_size

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        elapsed = (
            time.time()
            - start_time
        )

        speed = (
            processed
            / max(
                elapsed,
                1e-6
            )
        )

        remaining = (
            total
            - processed
        )

        eta = (
            remaining
            / max(
                speed,
                1e-6
            )
        )

        print(
            f"\r"
            f"[{split_name}] "
            f"{processed:,}/"
            f"{total:,} "
            f"("
            f"{100.0 * processed / total:.2f}%"
            f") | "
            f"{speed:.1f} img/s | "
            f"ETA "
            f"{eta / 60:.1f} min",
            end="",
            flush=True,
        )

    print()

    features = np.concatenate(
        all_features,
        axis=0,
    )

    labels = np.concatenate(
        all_labels,
        axis=0,
    )

    # --------------------------------------------------------
    # Final normalization in FP32
    # --------------------------------------------------------

    features = (
        features
        .astype(
            np.float32
        )
    )

    features = (
        features
        /
        np.maximum(
            np.linalg.norm(
                features,
                axis=1,
                keepdims=True,
            ),
            1e-12,
        )
    )

    features = (
        features
        .astype(
            STORE_DTYPE
        )
    )

    print()

    print(
        f"{split_name} feature shape:"
    )

    print(
        features.shape
    )

    print(
        f"{split_name} feature dtype:"
    )

    print(
        features.dtype
    )

    print(
        f"{split_name} labels shape:"
    )

    print(
        labels.shape
    )

    return (
        features,
        labels,
    )

# ============================================================
# 16. Feature cache helper
# ============================================================

def load_feature_cache(
    feature_file,
    label_file,
    expected_size,
    name,
):

    if (
        not os.path.exists(
            feature_file
        )
        or
        not os.path.exists(
            label_file
        )
    ):

        return None, None

    print()

    print(
        f"[CACHE] Loading {name}"
    )

    features = np.load(
        feature_file,
        mmap_mode="r",
    )

    labels = np.load(
        label_file
    )

    print(
        f"{name} features:"
        f" {features.shape}"
    )

    print(
        f"{name} labels:"
        f" {labels.shape}"
    )

    # --------------------------------------------------------
    # Check count
    # --------------------------------------------------------

    if (
        features.shape[0]
        != expected_size
    ):

        print(
            f"[CACHE] {name} feature count "
            f"mismatch."
        )

        return None, None

    if (
        labels.shape[0]
        != expected_size
    ):

        print(
            f"[CACHE] {name} label count "
            f"mismatch."
        )

        return None, None

    return (
        features,
        labels,
    )


# ============================================================
# 17. Database features
# ============================================================

database_features, database_labels = (
    load_feature_cache(
        DATABASE_FEATURE_FILE,
        DATABASE_LABEL_FILE,
        len(database_indices),
        "Database",
    )
)

if database_features is None:

    (
        database_features,
        database_labels,
    ) = extract_features(
        database_loader,
        "Database",
    )

    np.save(
        DATABASE_FEATURE_FILE,
        database_features,
    )

    np.save(
        DATABASE_LABEL_FILE,
        database_labels,
    )

    print()

    print(
        "Database features saved:"
    )

    print(
        DATABASE_FEATURE_FILE
    )

    print(
        DATABASE_LABEL_FILE
    )


# ============================================================
# 18. Query features
# ============================================================

query_features, query_labels = (
    load_feature_cache(
        QUERY_FEATURE_FILE,
        QUERY_LABEL_FILE,
        len(query_indices),
        "Query",
    )
)

if query_features is None:

    (
        query_features,
        query_labels,
    ) = extract_features(
        query_loader,
        "Query",
    )

    np.save(
        QUERY_FEATURE_FILE,
        query_features,
    )

    np.save(
        QUERY_LABEL_FILE,
        query_labels,
    )

    print()

    print(
        "Query features saved:"
    )

    print(
        QUERY_FEATURE_FILE
    )

    print(
        QUERY_LABEL_FILE
    )


# ============================================================
# 19. Feature sanity checks
# ============================================================

print()
print(
    "=" * 100
)

print(
    "Feature sanity checks"
)

print(
    "=" * 100
)

print(
    "Database:"
)

print(
    database_features.shape,
    database_features.dtype,
)

print()

print(
    "Query:"
)

print(
    query_features.shape,
    query_features.dtype,
)

assert (
    len(database_features)
    == len(database_indices)
)

assert (
    len(query_features)
    == len(query_indices)
)

assert (
    database_features.shape[1]
    == query_features.shape[1]
)

feature_dim = (
    database_features.shape[1]
)

print()

print(
    f"Feature dimension: "
    f"{feature_dim}"
)


# ============================================================
# 20. Check L2 normalization
# ============================================================

database_sample = np.asarray(
    database_features[
        :min(
            1000,
            len(database_features)
        )
    ],
    dtype=np.float32,
)

query_sample = np.asarray(
    query_features[
        :min(
            1000,
            len(query_features)
        )
    ],
    dtype=np.float32,
)

database_norms = np.linalg.norm(
    database_sample,
    axis=1,
)

query_norms = np.linalg.norm(
    query_sample,
    axis=1,
)

print()

print(
    "Database L2 norm:"
)

print(
    f"min  = "
    f"{database_norms.min():.6f}"
)

print(
    f"max  = "
    f"{database_norms.max():.6f}"
)

print(
    f"mean = "
    f"{database_norms.mean():.6f}"
)

print()

print(
    "Query L2 norm:"
)

print(
    f"min  = "
    f"{query_norms.min():.6f}"
)

print(
    f"max  = "
    f"{query_norms.max():.6f}"
)

print(
    f"mean = "
    f"{query_norms.mean():.6f}"
)

del database_sample
del query_sample

del database_norms
del query_norms

gc.collect()


# ============================================================
# 21. Build FAISS IndexFlatIP
# ============================================================

print()
print(
    "=" * 100
)

print(
    "Building FAISS IndexFlatIP"
)

print(
    "=" * 100
)

print(
    "Metric: Inner Product"
)

print(
    "Because features are L2-normalized:"
)

print(
    "Inner Product = Cosine Similarity"
)

dimension = (
    database_features.shape[1]
)

cpu_index = faiss.IndexFlatIP(
    dimension
)


# ============================================================
# 22. Move FAISS to GPU
# ============================================================

if (
    FAISS_USE_GPU
    and torch.cuda.is_available()
):

    print(
        "FAISS device: GPU"
    )

    faiss_resources = (
        faiss.StandardGpuResources()
    )

    index = faiss.index_cpu_to_gpu(
        faiss_resources,
        0,
        cpu_index,
    )

else:

    print(
        "FAISS device: CPU"
    )

    index = cpu_index


# ============================================================
# 23. Add database features
# ============================================================

print()
print(
    "=" * 100
)

print(
    "Adding database features"
)

print(
    "=" * 100
)

start_time = time.time()

database_features_fp32 = (
    np.asarray(
        database_features,
        dtype=np.float32,
    )
)

index.add(
    database_features_fp32
)

elapsed = (
    time.time()
    - start_time
)

print(
    f"Database vectors: "
    f"{index.ntotal:,}"
)

print(
    f"Add time: "
    f"{elapsed:.2f} sec"
)

del database_features_fp32

gc.collect()


# ============================================================
# 24. Similarity-weighted KNN
# ============================================================

def evaluate_similarity_weighted_knn(
    index,
    database_labels,
    query_features,
    query_labels,
    k_values,
    temperature,
):

    print()
    print(
        "=" * 100
    )

    print(
        "Similarity-weighted KNN"
    )

    print(
        "=" * 100
    )

    print(
        f"K values    : "
        f"{k_values}"
    )

    print(
        f"Temperature : "
        f"{temperature}"
    )

    print(
        "Metric      : "
        f"Inner Product / Cosine Similarity"
    )

    num_queries = (
        len(query_features)
    )

    max_k = max(
        k_values
    )

    # --------------------------------------------------------
    # Search in batches
    # --------------------------------------------------------

    SEARCH_BATCH_SIZE = 4096

    all_similarities = np.empty(
        (
            num_queries,
            max_k,
        ),
        dtype=np.float32,
    )

    all_neighbors = np.empty(
        (
            num_queries,
            max_k,
        ),
        dtype=np.int64,
    )

    start_time = time.time()

    for start in range(
        0,
        num_queries,
        SEARCH_BATCH_SIZE,
    ):

        end = min(
            start + SEARCH_BATCH_SIZE,
            num_queries,
        )

        queries = np.asarray(
            query_features[
                start:end
            ],
            dtype=np.float32,
        )

        similarities, neighbors = (
            index.search(
                queries,
                max_k,
            )
        )

        all_similarities[
            start:end
        ] = similarities

        all_neighbors[
            start:end
        ] = neighbors

        processed = end

        elapsed = (
            time.time()
            - start_time
        )

        speed = (
            processed
            / max(
                elapsed,
                1e-6
            )
        )

        eta = (
            num_queries
            - processed
        ) / max(
            speed,
            1e-6
        )

        print(
            f"\r"
            f"[FAISS] "
            f"{processed:,}/"
            f"{num_queries:,} "
            f"("
            f"{100.0 * processed / num_queries:.2f}%"
            f") | "
            f"{speed:.1f} query/s | "
            f"ETA "
            f"{eta / 60:.1f} min",
            end="",
            flush=True,
        )

    print()

    search_time = (
        time.time()
        - start_time
    )

    print()

    print(
        f"FAISS search time: "
        f"{search_time / 60:.2f} min"
    )

    # --------------------------------------------------------
    # Evaluate each K
    # --------------------------------------------------------

    results = {}

    for k in k_values:

        print()
        print(
            f"Evaluating K={k}"
        )

        top1_correct = 0

        top5_correct = 0

        top10_correct = 0

        top20_correct = 0

        for i in range(
            num_queries
        ):

            similarities_i = (
                all_similarities[
                    i,
                    :k
                ]
            )

            neighbors_i = (
                all_neighbors[
                    i,
                    :k
                ]
            )

            labels_i = (
                database_labels[
                    neighbors_i
                ]
            )

            target = int(
                query_labels[i]
            )

            # ------------------------------------------------
            # Stable softmax-like weighting
            #
            # weight =
            # exp((similarity - max_similarity) / T)
            # ------------------------------------------------

            weights = np.exp(
                (
                    similarities_i
                    - similarities_i.max()
                )
                / temperature
            )

            class_scores = np.bincount(
                labels_i,
                weights=weights,
                minlength=1000,
            )

            ranked_classes = (
                np.argsort(
                    -class_scores
                )
            )

            # ------------------------------------------------
            # Top-1
            # ------------------------------------------------

            if (
                target
                == ranked_classes[0]
            ):

                top1_correct += 1

            # ------------------------------------------------
            # Top-5
            # ------------------------------------------------

            if (
                target
                in ranked_classes[:5]
            ):

                top5_correct += 1

            # ------------------------------------------------
            # Top-10
            # ------------------------------------------------

            if (
                target
                in ranked_classes[:10]
            ):

                top10_correct += 1

            # ------------------------------------------------
            # Top-20
            # ------------------------------------------------

            if (
                target
                in ranked_classes[:20]
            ):

                top20_correct += 1

        results[
            f"knn_{k}"
        ] = {

            "top1": (
                top1_correct
                / num_queries
            ),

            "top5": (
                top5_correct
                / num_queries
            ),

            "top10": (
                top10_correct
                / num_queries
            ),

            "top20": (
                top20_correct
                / num_queries
            ),

        }

        print(
            f"K={k}: "
            f"Top-1="
            f"{results[f'knn_{k}']['top1'] * 100:.2f}% | "
            f"Top-5="
            f"{results[f'knn_{k}']['top5'] * 100:.2f}% | "
            f"Top-10="
            f"{results[f'knn_{k}']['top10'] * 100:.2f}% | "
            f"Top-20="
            f"{results[f'knn_{k}']['top20'] * 100:.2f}%"
        )

    return (
        results,
        search_time,
    )


# ============================================================
# 25. Run KNN
# ============================================================

(
    knn_results,
    search_time,
) = evaluate_similarity_weighted_knn(
    index=index,
    database_labels=database_labels,
    query_features=query_features,
    query_labels=query_labels,
    k_values=K_VALUES,
    temperature=TEMPERATURE,
)


# ============================================================
# 26. FLOPs estimation
# ============================================================

print()
print(
    "=" * 100
)

print(
    "KNN FLOPs estimation"
)

print(
    "=" * 100
)

num_database = (
    len(database_features)
)

num_query = (
    len(query_features)
)

# ------------------------------------------------------------
# Exact exhaustive IP search:
#
# Each query x database pair:
#
# dot product over D dimensions
#
# approximately:
#
# 2 * D FLOPs
#
# ------------------------------------------------------------

knn_search_flops = (
    2
    * num_query
    * num_database
    * feature_dim
)

knn_search_gflops = (
    knn_search_flops
    / 1e9
)

knn_search_pflops = (
    knn_search_flops
    / 1e15
)

print(
    f"Database images : "
    f"{num_database:,}"
)

print(
    f"Query images    : "
    f"{num_query:,}"
)

print(
    f"Feature dim     : "
    f"{feature_dim:,}"
)

print()

print(
    f"FAISS search GFLOPs: "
    f"{knn_search_gflops:.3f}"
)

print(
    f"FAISS search PFLOPs: "
    f"{knn_search_pflops:.6f}"
)


# ============================================================
# 27. Save result
# ============================================================

final_result = {

    "model": "UniAR-BSQ",

    "checkpoint": CKPT_PATH,

    "protocol_file": (
        PROTOCOL_FILE
    ),

    "seed": SEED,

    "database_size": (
        num_database
    ),

    "query_size": (
        num_query
    ),

    "feature_dim": (
        feature_dim
    ),

    "batch_size": (
        BATCH_SIZE
    ),

    "num_workers": (
        NUM_WORKERS
    ),

    "image_size": (
        IMG_SIZE
    ),

    "resize_size": (
        RESIZE_SIZE
    ),

    "feature_level": (
        BSQ_FEATURE_LEVEL
    ),

    "feature_pooling": (
        "mean over BSQ tokens"
    ),

    "feature_normalization": (
        "L2"
    ),

    "faiss_index": (
        "IndexFlatIP"
    ),

    "faiss_metric": (
        "inner_product"
    ),

    "similarity_weighted": True,

    "temperature": (
        TEMPERATURE
    ),

    "k_values": (
        K_VALUES
    ),

    "knn_results": (
        knn_results
    ),

    "faiss_search_time_sec": (
        search_time
    ),

    "knn_search_gflops": (
        knn_search_gflops
    ),

    "knn_search_pflops": (
        knn_search_pflops
    ),

}


with open(
    RESULT_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        final_result,
        f,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# 28. Final summary
# ============================================================

print()
print()
print(
    "=" * 100
)

print(
    "FINAL RESULT"
)

print(
    "=" * 100
)

print(
    f"Model         : UniAR-BSQ"
)

print(
    f"Database      : "
    f"{num_database:,}"
)

print(
    f"Query         : "
    f"{num_query:,}"
)

print(
    f"Feature dim   : "
    f"{feature_dim:,}"
)

print(
    f"Normalization : L2"
)

print(
    f"FAISS         : IndexFlatIP"
)

print(
    f"Temperature   : "
    f"{TEMPERATURE}"
)

print()

for k in K_VALUES:

    metrics = (
        knn_results[
            f"knn_{k}"
        ]
    )

    print(
        f"K={k:2d} | "
        f"Top-1  "
        f"{metrics['top1'] * 100:7.3f}% | "
        f"Top-5  "
        f"{metrics['top5'] * 100:7.3f}% | "
        f"Top-10 "
        f"{metrics['top10'] * 100:7.3f}% | "
        f"Top-20 "
        f"{metrics['top20'] * 100:7.3f}%"
    )

print()

print(
    f"KNN search: "
    f"{knn_search_gflops:.3f} GFLOPs"
)

print(
    f"KNN search: "
    f"{knn_search_pflops:.6f} PFLOPs"
)

print()

print(
    "Result saved:"
)

print(
    RESULT_FILE
)

print(
    "=" * 100
)

print(
    "Done."
)

# ============================================================
# 29. KNN TOTAL FLOPs
# ============================================================
#
# Total KNN FLOPs =
#
#   UniAR feature extraction FLOPs
#       +
#   Exact KNN similarity-search FLOPs
#
# Feature extraction:
#
#   (N_database + N_query)
#       ×
#   UniAR FLOPs / image
#
# KNN search:
#
#   2 × N_query × N_database × feature_dim
#
# ============================================================

print()
print()
print("=" * 100)
print("KNN TOTAL FLOPs")
print("=" * 100)


# ============================================================
# 29.1 Profile UniAR FLOPs / image
# ============================================================

print()
print("Profiling UniAR forward FLOPs...")
print("This uses the actual loaded UniAR model.")
print()


# ------------------------------------------------------------
# Take exactly one image
# ------------------------------------------------------------

profile_loader = DataLoader(
    query_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)

profile_images, _ = next(
    iter(profile_loader)
)


# ------------------------------------------------------------
# Move to the same device / dtype as normal inference
# ------------------------------------------------------------

profile_images = profile_images.to(
    DEVICE,
    dtype=torch.bfloat16,
    non_blocking=True,
)


# ============================================================
# 29.2 Define the exact UniAR forward
# ============================================================

@torch.no_grad()
def profile_uniar_forward(images):

    (
        flatten_image,
        grid_thw,
    ) = model.convert_img_to_patch(
        images
    )

    output = model(
        flatten_image,
        grid_thw=grid_thw,
        bsq_only=True,
        bsq_feature_level=BSQ_FEATURE_LEVEL,
        direct_indices=False,
    )

    features = output[0]

    batch_size = images.shape[0]

    feature_dim_local = (
        features.shape[-1]
    )

    num_tokens = (
        features.shape[0]
        // batch_size
    )

    features = features.reshape(
        batch_size,
        num_tokens,
        feature_dim_local,
    )

    features = features.mean(
        dim=1
    )

    features = F.normalize(
        features.float(),
        p=2,
        dim=1,
    )

    return features


# ============================================================
# 29.3 Warmup
# ============================================================

print("Warmup...")

with torch.no_grad():

    for _ in range(2):

        _ = profile_uniar_forward(
            profile_images
        )


if DEVICE.type == "cuda":

    torch.cuda.synchronize()


# ============================================================
# 29.4 PyTorch FLOPs profiler
# ============================================================

activities = [
    torch.profiler.ProfilerActivity.CPU
]

if DEVICE.type == "cuda":

    activities.append(
        torch.profiler.ProfilerActivity.CUDA
    )


print(
    "Running torch.profiler "
    "with_flops=True..."
)


if DEVICE.type == "cuda":

    torch.cuda.synchronize()


with torch.profiler.profile(
    activities=activities,
    record_shapes=False,
    profile_memory=False,
    with_stack=False,
    with_flops=True,
) as prof:

    with torch.no_grad():

        _ = profile_uniar_forward(
            profile_images
        )


if DEVICE.type == "cuda":

    torch.cuda.synchronize()


# ============================================================
# 29.5 Collect FLOPs
# ============================================================

profiled_flops = 0

for event in prof.key_averages():

    event_flops = getattr(
        event,
        "flops",
        0,
    )

    if event_flops is None:
        event_flops = 0

    profiled_flops += event_flops


profiled_flops = float(
    profiled_flops
)


# Since profiling batch size = 1:
# FLOPs / image = profiled FLOPs.

model_flops_per_image = (
    profiled_flops
)

model_gflops_per_image = (
    model_flops_per_image
    / 1e9
)


# ============================================================
# 29.6 Calculate feature extraction FLOPs
# ============================================================

num_database = len(
    database_features
)

num_query = len(
    query_features
)

total_images = (
    num_database
    + num_query
)


feature_extraction_flops = (
    total_images
    * model_flops_per_image
)

feature_extraction_gflops = (
    feature_extraction_flops
    / 1e9
)

feature_extraction_pflops = (
    feature_extraction_flops
    / 1e15
)


# ============================================================
# 29.7 Existing KNN search FLOPs
# ============================================================
#
# You already calculated:
#
#   2 × N_query × N_database × D
#
# above as knn_search_flops.
#
# We reuse that exact value.
# ============================================================

# Safety check: recalculate it here.

knn_search_flops = (
    2
    * num_query
    * num_database
    * feature_dim
)

knn_search_gflops = (
    knn_search_flops
    / 1e9
)

knn_search_pflops = (
    knn_search_flops
    / 1e15
)


# ============================================================
# 29.8 KNN TOTAL FLOPs
# ============================================================

knn_total_flops = (
    feature_extraction_flops
    +
    knn_search_flops
)

knn_total_gflops = (
    knn_total_flops
    / 1e9
)

knn_total_pflops = (
    knn_total_flops
    / 1e15
)


# ============================================================
# 29.9 Print complete FLOPs breakdown
# ============================================================

print()
print("=" * 100)
print("KNN FLOPs BREAKDOWN")
print("=" * 100)

print()

print(
    "----- UniAR Feature Extraction -----"
)

print(
    f"UniAR FLOPs / image : "
    f"{model_gflops_per_image:.6f} GFLOPs"
)

print(
    f"Database images     : "
    f"{num_database:,}"
)

print(
    f"Query images        : "
    f"{num_query:,}"
)

print(
    f"Total images        : "
    f"{total_images:,}"
)

print(
    f"Feature extraction  : "
    f"{feature_extraction_gflops:.6f} GFLOPs"
)

print(
    f"Feature extraction  : "
    f"{feature_extraction_pflops:.9f} PFLOPs"
)

print()

print(
    "----- Exact KNN Search -----"
)

print(
    f"Formula             : "
    f"2 × {num_query:,} × "
    f"{num_database:,} × "
    f"{feature_dim:,}"
)

print(
    f"KNN search          : "
    f"{knn_search_gflops:.6f} GFLOPs"
)

print(
    f"KNN search          : "
    f"{knn_search_pflops:.9f} PFLOPs"
)

print()

print(
    "----- KNN TOTAL -----"
)

print(
    f"Feature extraction  : "
    f"{feature_extraction_gflops:.6f} GFLOPs"
)

print(
    f"+ KNN search        : "
    f"{knn_search_gflops:.6f} GFLOPs"
)

print(
    "-" * 60
)

print(
    f"KNN TOTAL          : "
    f"{knn_total_gflops:.6f} GFLOPs"
)

print(
    f"KNN TOTAL          : "
    f"{knn_total_pflops:.9f} PFLOPs"
)

print("=" * 100)


# ============================================================
# 29.10 Save FLOPs into the existing JSON
# ============================================================

try:

    if os.path.exists(
        RESULT_FILE
    ):

        with open(
            RESULT_FILE,
            "r",
            encoding="utf-8",
        ) as f:

            saved_result = json.load(
                f
            )

    else:

        saved_result = {}


    saved_result[
        "flops"
    ] = {

        "definition": (
            "KNN total FLOPs = "
            "UniAR feature extraction FLOPs "
            "for database and query images "
            "+ exact exhaustive KNN similarity "
            "search FLOPs"
        ),

        "model_flops_method": (
            "torch.profiler with_flops=True"
        ),

        "model_gflops_per_image": (
            float(
                model_gflops_per_image
            )
        ),

        "database_images": (
            int(
                num_database
            )
        ),

        "query_images": (
            int(
                num_query
            )
        ),

        "total_images_encoded": (
            int(
                total_images
            )
        ),

        "feature_dimension": (
            int(
                feature_dim
            )
        ),

        "feature_extraction_flops": (
            float(
                feature_extraction_flops
            )
        ),

        "feature_extraction_gflops": (
            float(
                feature_extraction_gflops
            )
        ),

        "feature_extraction_pflops": (
            float(
                feature_extraction_pflops
            )
        ),

        "knn_search_flops": (
            int(
                knn_search_flops
            )
        ),

        "knn_search_gflops": (
            float(
                knn_search_gflops
            )
        ),

        "knn_search_pflops": (
            float(
                knn_search_pflops
            )
        ),

        "knn_total_flops": (
            float(
                knn_total_flops
            )
        ),

        "knn_total_gflops": (
            float(
                knn_total_gflops
            )
        ),

        "knn_total_pflops": (
            float(
                knn_total_pflops
            )
        ),
    }


    with open(
        RESULT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            saved_result,
            f,
            indent=2,
            ensure_ascii=False,
        )


    print()
    print(
        "FLOPs information added to:"
    )

    print(
        RESULT_FILE
    )

except Exception as e:

    print()
    print(
        "[WARNING] Failed to update "
        "result JSON:"
    )

    print(
        repr(e)
    )


# ============================================================
# 29.11 Final one-line result
# ============================================================

print()
print()
print("=" * 100)

print(
    "FINAL KNN TOTAL FLOPs:"
)

print(
    f"{knn_total_gflops:.6f} GFLOPs"
)

print(
    f"{knn_total_pflops:.9f} PFLOPs"
)

print("=" * 100)

# Four exact searches reuse the single 45k database feature cache above.
from multishot_protocol import run_multishot_from_cache
MULTISHOT_RESULTS = run_multishot_from_cache(CACHE_DIR, CACHE_DIR, "uniar_bsq")

