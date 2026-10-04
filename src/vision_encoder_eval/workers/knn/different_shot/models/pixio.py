
from vision_encoder_eval.core.runtime import asset_path
import os
import sys
import json
import time
import math
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import Dataset, DataLoader

import faiss

from transformers import AutoImageProcessor, AutoModel


# ============================================================
# 0. Settings
# ============================================================

# ------------------------------------------------------------
# Local Pixio models
# ------------------------------------------------------------

MODEL_DIRS = {
    "pixio-vitb16": asset_path('model_assets', 'model/pixio-vitb16'),
    "pixio-vith16": asset_path('model_assets', 'model/pixio-vith16'),
    "pixio-vitl16": asset_path('model_assets', 'model/pixio-vitl16'),
}

# ------------------------------------------------------------
# ImageNet
# ------------------------------------------------------------

IMAGENET_VAL_ROOT = "/workspace/root/val"

# Existing unified protocol
PROTOCOL_PATH = asset_path('runtime', 'metaclip_knn/val_45shot_5query_seed42_protocol.json')

# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

OUTPUT_ROOT = asset_path('runtime', 'pixio_knn')

FEATURE_ROOT = os.path.join(
    OUTPUT_ROOT,
    "features_45pool",
)

RESULT_ROOT = os.path.join(
    OUTPUT_ROOT,
    "results",
)

os.makedirs(FEATURE_ROOT, exist_ok=True)
os.makedirs(RESULT_ROOT, exist_ok=True)


# ------------------------------------------------------------
# KNN protocol
# ------------------------------------------------------------

NUM_CLASSES = 1000

TRAIN_PER_CLASS = 45
QUERY_PER_CLASS = 5

NUM_DATABASE = NUM_CLASSES * TRAIN_PER_CLASS
NUM_QUERY = NUM_CLASSES * QUERY_PER_CLASS

K_VALUES = [1, 5, 10, 20]

TEMPERATURE = 0.07

# ------------------------------------------------------------
# Runtime
# ------------------------------------------------------------

BATCH_SIZE = 128
NUM_WORKERS = 8

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

STORE_DTYPE = np.float32

# ------------------------------------------------------------
# Feature extraction
# ------------------------------------------------------------

# Pixio has 8 class tokens.
NUM_CLASS_TOKENS = 8

# We use the mean of the 8 class tokens.
POOL_CLASS_TOKENS = True

# L2 normalization before FAISS IP.
NORMALIZE_FEATURES = True

# ------------------------------------------------------------
# Cache behavior
# ------------------------------------------------------------

USE_FEATURE_CACHE = True

# If True, existing feature cache will be overwritten.
OVERWRITE_FEATURE_CACHE = False


# ============================================================
# 1. Dataset
# ============================================================

class ImageNetValDataset(Dataset):

    def __init__(
        self,
        root,
        processor,
    ):
        self.root = root
        self.processor = processor

        self.samples = []

        # ImageFolder-style ImageNet val directory:
        #
        # val/
        #   n01440764/
        #       xxx.JPEG
        #   n01443537/
        #       xxx.JPEG
        #
        class_dirs = sorted(
            [
                d
                for d in os.listdir(root)
                if os.path.isdir(os.path.join(root, d))
            ]
        )

        if len(class_dirs) != NUM_CLASSES:
            print(
                f"[Warning] Found {len(class_dirs)} class directories, "
                f"expected {NUM_CLASSES}"
            )

        self.class_to_idx = {
            cls_name: idx
            for idx, cls_name in enumerate(class_dirs)
        }

        for cls_name in class_dirs:

            cls_dir = os.path.join(root, cls_name)

            files = sorted(
                [
                    f
                    for f in os.listdir(cls_dir)
                    if f.lower().endswith(
                        (".jpg", ".jpeg", ".png", ".webp")
                    )
                ]
            )

            label = self.class_to_idx[cls_name]

            for fname in files:

                path = os.path.join(
                    cls_dir,
                    fname,
                )

                self.samples.append(
                    (
                        path,
                        label,
                    )
                )

        print(
            f"[Dataset] ImageNet val images: "
            f"{len(self.samples):,}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        path, label = self.samples[idx]

        image = Image.open(path).convert("RGB")

        encoded = self.processor(
            images=image,
            return_tensors="pt",
        )

        pixel_values = encoded["pixel_values"][0]

        return (
            pixel_values,
            label,
        )


# ============================================================
# 2. Collate
# ============================================================

def collate_fn(batch):

    pixel_values = torch.stack(
        [x[0] for x in batch],
        dim=0,
    )

    labels = torch.tensor(
        [x[1] for x in batch],
        dtype=torch.long,
    )

    return pixel_values, labels


# ============================================================
# 3. Protocol loading
# ============================================================

def load_protocol():

    if not os.path.isfile(PROTOCOL_PATH):

        raise FileNotFoundError(
            f"\nProtocol file not found:\n"
            f"{PROTOCOL_PATH}\n\n"
            f"Please make sure the existing unified "
            f"5-shot protocol is available."
        )

    with open(
        PROTOCOL_PATH,
        "r",
    ) as f:

        protocol = json.load(f)


        # Canonical fixed 45-pool/5-query protocol plus legacy aliases used below.

        protocol.setdefault("train_indices", protocol["train_pool_indices"])

        protocol.setdefault("database_indices", protocol["train_pool_indices"])

        protocol.setdefault("db_indices", protocol["train_pool_indices"])

        protocol.setdefault("num_train", 45000)

        protocol.setdefault("train_per_class", 45)

        protocol.setdefault("images_per_class", 50)

    print("\n" + "=" * 80)
    print("Loading existing KNN protocol")
    print("=" * 80)

    print(
        f"Protocol: {PROTOCOL_PATH}"
    )

    print(
        f"Protocol keys: {list(protocol.keys())}"
    )

    return protocol


# ============================================================
# 4. Extract indices from protocol
# ============================================================

def extract_protocol_indices(protocol):

    """
    Try common formats used by the previous protocol generator.

    Expected ultimately:
        database_indices: 5000 indices
        query_indices:    5000 indices

    """

    database_indices = None
    query_indices = None

    possible_database_keys = [
        "train_pool_indices",
        "database_indices",
        "db_indices",
        "train_indices",
        "gallery_indices",
        "reference_indices",
    ]

    possible_query_keys = [
        "query_indices",
        "test_indices",
        "val_indices",
        "query",
    ]

    for key in possible_database_keys:

        if key in protocol:

            database_indices = protocol[key]

            break

    for key in possible_query_keys:

        if key in protocol:

            query_indices = protocol[key]

            break

    # --------------------------------------------------------
    # Case: nested database/query
    # --------------------------------------------------------

    if database_indices is None:

        for parent_key in [
            "database",
            "db",
            "gallery",
            "train",
        ]:

            if parent_key in protocol:

                obj = protocol[parent_key]

                if isinstance(obj, dict):

                    for key in [
                        "indices",
                        "image_indices",
                        "samples",
                    ]:

                        if key in obj:

                            database_indices = obj[key]

                            break

                if database_indices is not None:
                    break

    if query_indices is None:

        for parent_key in [
            "query",
            "test",
            "val",
        ]:

            if parent_key in protocol:

                obj = protocol[parent_key]

                if isinstance(obj, dict):

                    for key in [
                        "indices",
                        "image_indices",
                        "samples",
                    ]:

                        if key in obj:

                            query_indices = obj[key]

                            break

                if query_indices is not None:
                    break

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if database_indices is None:
        raise RuntimeError(
            "Could not find database indices in protocol JSON.\n"
            f"Available keys: {list(protocol.keys())}"
        )

    if query_indices is None:
        raise RuntimeError(
            "Could not find query indices in protocol JSON.\n"
            f"Available keys: {list(protocol.keys())}"
        )

    database_indices = np.asarray(
        database_indices,
        dtype=np.int64,
    )

    query_indices = np.asarray(
        query_indices,
        dtype=np.int64,
    )

    print(
        f"Database indices: {len(database_indices):,}"
    )

    print(
        f"Query indices:    {len(query_indices):,}"
    )

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
        f"Database/query overlap: {len(overlap)}"
    )

    return (
        database_indices,
        query_indices,
    )


# ============================================================
# 5. Build subset dataset
# ============================================================

class IndexedDataset(Dataset):

    def __init__(
        self,
        base_dataset,
        indices,
    ):

        self.base_dataset = base_dataset
        self.indices = np.asarray(
            indices,
            dtype=np.int64,
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        return self.base_dataset[
            int(self.indices[idx])
        ]


# ============================================================
# 6. Feature extraction
# ============================================================

@torch.no_grad()
def extract_features(
    model,
    dataloader,
    device,
    split_name,
):

    model.eval()

    all_features = []
    all_labels = []

    total = len(dataloader.dataset)

    start_time = time.time()

    processed = 0

    for batch_idx, (pixel_values, labels) in enumerate(
        dataloader
    ):

        pixel_values = pixel_values.to(
            device,
            non_blocking=True,
        )

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        outputs = model(
            pixel_values=pixel_values,
        )

        # ----------------------------------------------------
        # Pixio:
        #
        # outputs.last_hidden_state
        #
        # [B, 8 + Npatch, D]
        #
        # First 8 tokens = class tokens
        # ----------------------------------------------------

        hidden = outputs.last_hidden_state

        if hidden.ndim != 3:

            raise RuntimeError(
                f"Unexpected hidden state shape: "
                f"{tuple(hidden.shape)}"
            )

        if hidden.shape[1] < NUM_CLASS_TOKENS:

            raise RuntimeError(
                f"Number of tokens ({hidden.shape[1]}) "
                f"is smaller than the expected "
                f"{NUM_CLASS_TOKENS} class tokens."
            )

        class_tokens = hidden[
            :,
            :NUM_CLASS_TOKENS,
            :,
        ]

        # ----------------------------------------------------
        # Pool 8 class tokens
        # ----------------------------------------------------

        if POOL_CLASS_TOKENS:

            features = class_tokens.mean(
                dim=1
            )

        else:

            features = class_tokens[:, 0]

        # ----------------------------------------------------
        # FP32
        # ----------------------------------------------------

        features = features.float()

        # ----------------------------------------------------
        # L2 normalization
        # ----------------------------------------------------

        if NORMALIZE_FEATURES:

            features = F.normalize(
                features,
                p=2,
                dim=-1,
            )

        all_features.append(
            features.cpu().numpy().astype(
                STORE_DTYPE
            )
        )

        all_labels.append(
            labels.numpy()
        )

        processed += pixel_values.shape[0]

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            batch_idx % 20 == 0
            or processed >= total
        ):

            elapsed = time.time() - start_time

            speed = (
                processed / elapsed
                if elapsed > 0
                else 0
            )

            eta = (
                (total - processed) / speed
                if speed > 0
                else 0
            )

            print(
                f"[{split_name}] "
                f"{processed:,}/{total:,} "
                f"({processed / total * 100:.2f}%) "
                f"| {speed:.1f} img/s "
                f"| ETA {eta / 60:.2f} min"
            )

    features = np.concatenate(
        all_features,
        axis=0,
    )

    labels = np.concatenate(
        all_labels,
        axis=0,
    )

    print(
        f"[{split_name}] Feature shape: "
        f"{features.shape}"
    )

    return features, labels


# ============================================================
# 7. FLOPs
# ============================================================

def get_config_value(
    config,
    names,
    default=None,
):

    for name in names:

        if hasattr(config, name):

            value = getattr(
                config,
                name,
            )

            if value is not None:

                return value

    return default


def calculate_pixio_encoder_flops(
    model,
    image_size=None,
):

    """
    Analytic ViT FLOPs.

    Convention:

    Patch embedding:
        2 * H * W * 3 * D

    Attention per layer:
        QKV
        attention score
        attention-value product
        output projection

    MLP per layer:
        2 * N * D * MLP
        2 * N * MLP * D

    We count multiply-add as 2 FLOPs.

    Token count:
        8 class tokens + patch tokens

    Note:
        This is an analytic estimate for the ViT encoder,
        not a profiler measurement.
    """

    config = model.config

    if image_size is None:

        image_size = get_config_value(
            config,
            [
                "image_size",
                "vision_config.image_size",
            ],
            default=256,
        )

    patch_size = get_config_value(
        config,
        [
            "patch_size",
        ],
        default=16,
    )

    hidden_size = get_config_value(
        config,
        [
            "hidden_size",
        ],
        default=None,
    )

    num_layers = get_config_value(
        config,
        [
            "num_hidden_layers",
            "depth",
        ],
        default=None,
    )

    num_heads = get_config_value(
        config,
        [
            "num_attention_heads",
            "num_heads",
        ],
        default=None,
    )

    intermediate_size = get_config_value(
        config,
        [
            "intermediate_size",
            "mlp_ratio",
        ],
        default=None,
    )

    if hidden_size is None:
        raise RuntimeError(
            "Could not determine hidden_size from Pixio config."
        )

    if num_layers is None:
        raise RuntimeError(
            "Could not determine depth from Pixio config."
        )

    if num_heads is None:
        raise RuntimeError(
            "Could not determine num_attention_heads from Pixio config."
        )

    if intermediate_size is None:

        intermediate_size = 4 * hidden_size

    # --------------------------------------------------------
    # Handle image_size potentially being list/tuple
    # --------------------------------------------------------

    if isinstance(
        image_size,
        (list, tuple),
    ):

        image_h = int(image_size[0])
        image_w = int(image_size[1])

    else:

        image_h = int(image_size)
        image_w = int(image_size)

    patch_size = int(patch_size)

    hidden_size = int(hidden_size)
    num_layers = int(num_layers)
    num_heads = int(num_heads)
    intermediate_size = int(intermediate_size)

    # --------------------------------------------------------
    # Number of patches
    # --------------------------------------------------------

    grid_h = image_h // patch_size
    grid_w = image_w // patch_size

    num_patches = grid_h * grid_w

    num_tokens = (
        NUM_CLASS_TOKENS
        + num_patches
    )

    # --------------------------------------------------------
    # Patch embedding
    #
    # Conv patch projection
    # equivalent to:
    #
    # 2 * Npatch * patch_area * 3 * D
    # --------------------------------------------------------

    patch_embed_flops = (
        2
        * num_patches
        * patch_size
        * patch_size
        * 3
        * hidden_size
    )

    # --------------------------------------------------------
    # Per Transformer block
    # --------------------------------------------------------

    # QKV:
    # 3 * linear D -> D
    qkv_flops = (
        2
        * num_tokens
        * hidden_size
        * (3 * hidden_size)
    )

    # Attention QK^T:
    #
    # num_heads *
    # N * N * head_dim
    #
    # multiply-add => *2
    #

    head_dim = hidden_size // num_heads

    attention_qk_flops = (
        2
        * num_heads
        * num_tokens
        * num_tokens
        * head_dim
    )

    # Attention * V

    attention_v_flops = (
        2
        * num_heads
        * num_tokens
        * num_tokens
        * head_dim
    )

    # Output projection

    proj_flops = (
        2
        * num_tokens
        * hidden_size
        * hidden_size
    )

    attention_flops = (
        qkv_flops
        + attention_qk_flops
        + attention_v_flops
        + proj_flops
    )

    # --------------------------------------------------------
    # MLP
    # --------------------------------------------------------

    mlp_flops = (
        2
        * num_tokens
        * hidden_size
        * intermediate_size
        +
        2
        * num_tokens
        * intermediate_size
        * hidden_size
    )

    # --------------------------------------------------------
    # Transformer blocks
    # --------------------------------------------------------

    transformer_flops = (
        num_layers
        * (
            attention_flops
            + mlp_flops
        )
    )

    # --------------------------------------------------------
    # Total
    # --------------------------------------------------------

    total_flops = (
        patch_embed_flops
        + transformer_flops
    )

    return {
        "image_height": image_h,
        "image_width": image_w,
        "patch_size": patch_size,
        "grid_h": grid_h,
        "grid_w": grid_w,
        "num_patches": num_patches,
        "num_tokens": num_tokens,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "intermediate_size": intermediate_size,
        "patch_embed_flops": patch_embed_flops,
        "attention_flops_per_layer": attention_flops,
        "mlp_flops_per_layer": mlp_flops,
        "encoder_flops_per_image": total_flops,
    }


# ============================================================
# 8. KNN FLOPs
# ============================================================

def calculate_knn_flops(
    encoder_flops_per_image,
    num_database,
    num_query,
    feature_dim,
):

    # --------------------------------------------------------
    # Feature extraction
    #
    # Both database and query images are encoded.
    # --------------------------------------------------------

    encoder_total_flops = (
        encoder_flops_per_image
        * (
            num_database
            + num_query
        )
    )

    # --------------------------------------------------------
    # Exact all-pairs similarity
    #
    # database:
    #     [Ndb, D]
    #
    # query:
    #     [Nq, D]
    #
    # similarity:
    #     [Nq, Ndb]
    #
    # Dot product:
    #     D multiplications + D additions
    #
    # convention = 2D FLOPs
    # --------------------------------------------------------

    similarity_flops = (
        2
        * num_query
        * num_database
        * feature_dim
    )

    total_flops = (
        encoder_total_flops
        + similarity_flops
    )

    return {
        "encoder_total_flops": encoder_total_flops,
        "knn_similarity_flops": similarity_flops,
        "knn_total_flops": total_flops,
    }


# ============================================================
# 9. Formatting
# ============================================================

def format_flops(flops):

    if flops >= 1e15:

        return f"{flops / 1e15:.4f} PFLOPs"

    if flops >= 1e12:

        return f"{flops / 1e12:.4f} TFLOPs"

    if flops >= 1e9:

        return f"{flops / 1e9:.4f} GFLOPs"

    if flops >= 1e6:

        return f"{flops / 1e6:.4f} MFLOPs"

    return f"{flops:.0f} FLOPs"


# ============================================================
# 10. Weighted KNN
# ============================================================

def run_knn(
    database_features,
    database_labels,
    query_features,
    query_labels,
):

    print("\n" + "=" * 80)
    print("Building FAISS IndexFlatIP")
    print("=" * 80)

    database_features = np.ascontiguousarray(
        database_features,
        dtype=np.float32,
    )

    query_features = np.ascontiguousarray(
        query_features,
        dtype=np.float32,
    )

    feature_dim = database_features.shape[1]

    print(
        f"Database features: {database_features.shape}"
    )

    print(
        f"Query features:    {query_features.shape}"
    )

    print(
        f"Feature dim:       {feature_dim}"
    )

    # --------------------------------------------------------
    # FAISS
    # --------------------------------------------------------

    index = faiss.IndexFlatIP(
        feature_dim
    )

    index.add(
        database_features
    )

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    max_k = max(K_VALUES)

    print(
        f"Searching top-{max_k}..."
    )

    search_start = time.time()

    similarities, neighbors = index.search(
        query_features,
        max_k,
    )

    search_time = time.time() - search_start

    print(
        f"FAISS search time: "
        f"{search_time:.2f} sec"
    )

    # --------------------------------------------------------
    # Weighted KNN
    # --------------------------------------------------------

    results = {}

    for k in K_VALUES:

        sims_k = similarities[:, :k]

        inds_k = neighbors[:, :k]

        labels_k = database_labels[
            inds_k
        ]

        # ----------------------------------------------------
        # exp(similarity / temperature)
        #
        # Subtract row max for numerical stability.
        # ----------------------------------------------------

        scaled = (
            sims_k
            / TEMPERATURE
        )

        scaled = (
            scaled
            - scaled.max(
                axis=1,
                keepdims=True,
            )
        )

        weights = np.exp(
            scaled
        )

        # ----------------------------------------------------
        # Weighted voting
        # ----------------------------------------------------

        predictions = np.zeros(
            len(query_labels),
            dtype=np.int64,
        )

        for i in range(
            len(query_labels)
        ):

            class_scores = {}

            for j in range(k):

                cls = int(
                    labels_k[i, j]
                )

                w = float(
                    weights[i, j]
                )

                class_scores[cls] = (
                    class_scores.get(
                        cls,
                        0.0,
                    )
                    + w
                )

            predictions[i] = max(
                class_scores,
                key=class_scores.get,
            )

        accuracy = (
            predictions
            == query_labels
        ).mean()

        results[str(k)] = {
            "accuracy": float(
                accuracy
            ),
            "correct": int(
                (
                    predictions
                    == query_labels
                ).sum()
            ),
            "total": int(
                len(query_labels)
            ),
        }

        print(
            f"K={k:2d} | "
            f"Accuracy={accuracy * 100:.4f}% | "
            f"Correct="
            f"{(predictions == query_labels).sum():,}/"
            f"{len(query_labels):,}"
        )

    return results


# ============================================================
# 11. Save / load feature cache
# ============================================================

def get_cache_paths(
    model_name,
):

    model_feature_dir = os.path.join(
        FEATURE_ROOT,
        model_name,
    )

    os.makedirs(
        model_feature_dir,
        exist_ok=True,
    )

    return {
        "database_features": os.path.join(
            model_feature_dir,
            "database_features.npy",
        ),
        "database_labels": os.path.join(
            model_feature_dir,
            "database_labels.npy",
        ),
        "query_features": os.path.join(
            model_feature_dir,
            "query_features.npy",
        ),
        "query_labels": os.path.join(
            model_feature_dir,
            "query_labels.npy",
        ),
    }


def cache_exists(
    paths,
):

    return all(
        os.path.isfile(p)
        for p in paths.values()
    )


def save_feature_cache(
    paths,
    database_features,
    database_labels,
    query_features,
    query_labels,
):

    print("\nSaving feature cache...")

    np.save(
        paths["database_features"],
        database_features,
    )

    np.save(
        paths["database_labels"],
        database_labels,
    )

    np.save(
        paths["query_features"],
        query_features,
    )

    np.save(
        paths["query_labels"],
        query_labels,
    )

    print(
        f"Saved to: "
        f"{os.path.dirname(paths['database_features'])}"
    )


def load_feature_cache(
    paths,
):

    print("\nLoading feature cache...")

    database_features = np.load(
        paths["database_features"],
        mmap_mode="r",
    )

    database_labels = np.load(
        paths["database_labels"],
    )

    query_features = np.load(
        paths["query_features"],
        mmap_mode="r",
    )

    query_labels = np.load(
        paths["query_labels"],
    )

    return (
        database_features,
        database_labels,
        query_features,
        query_labels,
    )


# ============================================================
# 12. Model information
# ============================================================

def print_model_config(
    model_name,
    model,
):

    print("\n" + "=" * 80)
    print(f"{model_name} CONFIG")
    print("=" * 80)

    config = model.config

    print(
        f"Model type: "
        f"{getattr(config, 'model_type', 'N/A')}"
    )

    print(
        f"Hidden size: "
        f"{get_config_value(config, ['hidden_size'], 'N/A')}"
    )

    print(
        f"Depth: "
        f"{get_config_value(config, ['num_hidden_layers', 'depth'], 'N/A')}"
    )

    print(
        f"Attention heads: "
        f"{get_config_value(config, ['num_attention_heads', 'num_heads'], 'N/A')}"
    )

    print(
        f"Intermediate size: "
        f"{get_config_value(config, ['intermediate_size'], 'N/A')}"
    )

    print(
        f"Patch size: "
        f"{get_config_value(config, ['patch_size'], 'N/A')}"
    )

    print(
        f"Image size: "
        f"{get_config_value(config, ['image_size'], 'N/A')}"
    )


# ============================================================
# 13. Run one model
# ============================================================

def run_model(
    model_name,
    model_dir,
    protocol,
    database_indices,
    query_indices,
):

    print("\n\n")
    print("#" * 100)
    print(f"# MODEL: {model_name}")
    print("#" * 100)

    # --------------------------------------------------------
    # Check local model
    # --------------------------------------------------------

    required_files = [
        "config.json",
        "preprocessor_config.json",
        "model.safetensors",
    ]

    for filename in required_files:

        path = os.path.join(
            model_dir,
            filename,
        )

        if not os.path.isfile(path):

            raise FileNotFoundError(
                f"\nMissing required Pixio file:\n"
                f"{path}\n"
            )

    print(
        f"Model directory: {model_dir}"
    )

    print(
        "Loading strictly from local files..."
    )

    # --------------------------------------------------------
    # Processor
    # --------------------------------------------------------

    processor = AutoImageProcessor.from_pretrained(
        model_dir,
        local_files_only=True,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = AutoModel.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype=(
            torch.float16
            if DEVICE.startswith("cuda")
            else torch.float32
        ),
    )

    model = model.to(DEVICE)

    model.eval()

    print_model_config(
        model_name,
        model,
    )

    # --------------------------------------------------------
    # FLOPs
    # --------------------------------------------------------

    flops_info = calculate_pixio_encoder_flops(
        model
    )

    print("\n" + "=" * 80)
    print("Encoder FLOPs")
    print("=" * 80)

    print(
        f"Input: "
        f"{flops_info['image_height']}x"
        f"{flops_info['image_width']}"
    )

    print(
        f"Patch: "
        f"{flops_info['patch_size']}x"
        f"{flops_info['patch_size']}"
    )

    print(
        f"Patch grid: "
        f"{flops_info['grid_h']}x"
        f"{flops_info['grid_w']}"
    )

    print(
        f"Patch tokens: "
        f"{flops_info['num_patches']}"
    )

    print(
        f"Class tokens: "
        f"{NUM_CLASS_TOKENS}"
    )

    print(
        f"Total tokens: "
        f"{flops_info['num_tokens']}"
    )

    print(
        f"Hidden size: "
        f"{flops_info['hidden_size']}"
    )

    print(
        f"Layers: "
        f"{flops_info['num_layers']}"
    )

    print(
        f"Attention heads: "
        f"{flops_info['num_heads']}"
    )

    print(
        f"Intermediate size: "
        f"{flops_info['intermediate_size']}"
    )

    print(
        f"Encoder FLOPs/image: "
        f"{format_flops(flops_info['encoder_flops_per_image'])}"
    )

    # --------------------------------------------------------
    # Feature cache
    # --------------------------------------------------------

    cache_paths = get_cache_paths(
        model_name
    )

    use_cache = (
        USE_FEATURE_CACHE
        and cache_exists(cache_paths)
        and not OVERWRITE_FEATURE_CACHE
    )

    if use_cache:

        print("\n" + "=" * 80)
        print("Feature cache found")
        print("=" * 80)

        (
            database_features,
            database_labels,
            query_features,
            query_labels,
        ) = load_feature_cache(
            cache_paths
        )

    else:

        # ----------------------------------------------------
        # Base dataset
        # ----------------------------------------------------

        base_dataset = ImageNetValDataset(
            root=IMAGENET_VAL_ROOT,
            processor=processor,
        )

        # ----------------------------------------------------
        # Database
        # ----------------------------------------------------

        database_dataset = IndexedDataset(
            base_dataset,
            database_indices,
        )

        # ----------------------------------------------------
        # Query
        # ----------------------------------------------------

        query_dataset = IndexedDataset(
            base_dataset,
            query_indices,
        )

        # ----------------------------------------------------
        # DataLoader
        # ----------------------------------------------------

        database_loader = DataLoader(
            database_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=(
                NUM_WORKERS > 0
            ),
            collate_fn=collate_fn,
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
            collate_fn=collate_fn,
        )

        # ----------------------------------------------------
        # Database features
        # ----------------------------------------------------

        print("\n" + "=" * 80)
        print("Extracting DATABASE features")
        print("=" * 80)

        database_features, database_labels = (
            extract_features(
                model=model,
                dataloader=database_loader,
                device=DEVICE,
                split_name="DATABASE",
            )
        )

        # ----------------------------------------------------
        # Query features
        # ----------------------------------------------------

        print("\n" + "=" * 80)
        print("Extracting QUERY features")
        print("=" * 80)

        query_features, query_labels = (
            extract_features(
                model=model,
                dataloader=query_loader,
                device=DEVICE,
                split_name="QUERY",
            )
        )

        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        assert (
            len(database_features)
            == NUM_DATABASE
        )

        assert (
            len(query_features)
            == NUM_QUERY
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        save_feature_cache(
            paths=cache_paths,
            database_features=database_features,
            database_labels=database_labels,
            query_features=query_features,
            query_labels=query_labels,
        )

    # --------------------------------------------------------
    # Feature information
    # --------------------------------------------------------

    feature_dim = database_features.shape[1]

    print("\n" + "=" * 80)
    print("Feature information")
    print("=" * 80)

    print(
        f"Database: "
        f"{database_features.shape}"
    )

    print(
        f"Query: "
        f"{query_features.shape}"
    )

    print(
        f"Feature dimension: "
        f"{feature_dim}"
    )

    # --------------------------------------------------------
    # KNN FLOPs
    # --------------------------------------------------------

    knn_flops = calculate_knn_flops(
        encoder_flops_per_image=(
            flops_info[
                "encoder_flops_per_image"
            ]
        ),
        num_database=NUM_DATABASE,
        num_query=NUM_QUERY,
        feature_dim=feature_dim,
    )

    print("\n" + "=" * 80)
    print("KNN FLOPs")
    print("=" * 80)

    print(
        f"Encoder FLOPs / image: "
        f"{format_flops(flops_info['encoder_flops_per_image'])}"
    )

    print(
        f"Images encoded: "
        f"{NUM_DATABASE + NUM_QUERY:,}"
    )

    print(
        f"Encoder total FLOPs: "
        f"{format_flops(knn_flops['encoder_total_flops'])}"
    )

    print(
        f"Exact similarity FLOPs: "
        f"{format_flops(knn_flops['knn_similarity_flops'])}"
    )

    print(
        f"KNN TOTAL FLOPs: "
        f"{format_flops(knn_flops['knn_total_flops'])}"
    )

    print(
        f"KNN TOTAL TFLOPs: "
        f"{knn_flops['knn_total_flops'] / 1e12:.6f}"
    )

    print(
        f"KNN TOTAL PFLOPs: "
        f"{knn_flops['knn_total_flops'] / 1e15:.6f}"
    )

    # --------------------------------------------------------
    # KNN
    # --------------------------------------------------------

    knn_start = time.time()

    results = run_knn(
        database_features=database_features,
        database_labels=database_labels,
        query_features=query_features,
        query_labels=query_labels,
    )

    knn_time = (
        time.time()
        - knn_start
    )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    result = {

        "model": model_name,

        "model_dir": model_dir,

        "protocol": {
            "protocol_path": PROTOCOL_PATH,
            "num_classes": NUM_CLASSES,
            "database_per_class": TRAIN_PER_CLASS,
            "query_per_class": QUERY_PER_CLASS,
            "num_database": NUM_DATABASE,
            "num_query": NUM_QUERY,
            "seed": 42,
        },

        "feature": {
            "type": "mean_of_8_class_tokens",
            "num_class_tokens": NUM_CLASS_TOKENS,
            "l2_normalized": NORMALIZE_FEATURES,
            "dtype": "float32",
            "dimension": int(feature_dim),
        },

        "knn": {
            "index": "FAISS_IndexFlatIP",
            "temperature": TEMPERATURE,
            "k_values": K_VALUES,
            "search_time_seconds": knn_time,
            "results": results,
        },

        "flops": {
            "image_height": flops_info[
                "image_height"
            ],
            "image_width": flops_info[
                "image_width"
            ],
            "patch_size": flops_info[
                "patch_size"
            ],
            "num_patches": flops_info[
                "num_patches"
            ],
            "num_tokens": flops_info[
                "num_tokens"
            ],
            "hidden_size": flops_info[
                "hidden_size"
            ],
            "num_layers": flops_info[
                "num_layers"
            ],
            "num_heads": flops_info[
                "num_heads"
            ],
            "intermediate_size": flops_info[
                "intermediate_size"
            ],
            "encoder_flops_per_image": (
                flops_info[
                    "encoder_flops_per_image"
                ]
            ),
            "encoder_total_flops": (
                knn_flops[
                    "encoder_total_flops"
                ]
            ),
            "knn_similarity_flops": (
                knn_flops[
                    "knn_similarity_flops"
                ]
            ),
            "knn_total_flops": (
                knn_flops[
                    "knn_total_flops"
                ]
            ),
            "knn_total_tflops": (
                knn_flops[
                    "knn_total_flops"
                ] / 1e12
            ),
            "knn_total_pflops": (
                knn_flops[
                    "knn_total_flops"
                ] / 1e15
            ),
        },
    }

    result_path = os.path.join(
        RESULT_ROOT,
        f"{model_name}.json",
    )

    with open(
        result_path,
        "w",
    ) as f:

        json.dump(
            result,
            f,
            indent=2,
        )

    print("\n" + "=" * 80)
    print(f"{model_name} FINAL RESULT")
    print("=" * 80)

    for k in K_VALUES:

        acc = results[str(k)][
            "accuracy"
        ]

        print(
            f"K={k:2d}: "
            f"{acc * 100:.4f}%"
        )

    print(
        f"\nFeature dimension: {feature_dim}"
    )

    print(
        f"Encoder: "
        f"{flops_info['encoder_flops_per_image'] / 1e9:.4f} GFLOPs/image"
    )

    print(
        f"KNN total: "
        f"{knn_flops['knn_total_flops'] / 1e15:.6f} PFLOPs"
    )

    print(
        f"Result saved: {result_path}"
    )

    # --------------------------------------------------------
    # Release GPU memory
    # --------------------------------------------------------

    del model
    del processor

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    return result


# ============================================================
# 14. Main
# ============================================================

def _single_pool_main():

    print("\n")
    print("#" * 100)
    print("# Pixio Unified ImageNet KNN Evaluation")
    print("#" * 100)

    print(
        f"Device: {DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        f"Workers: {NUM_WORKERS}"
    )

    print(
        f"Database: "
        f"{NUM_DATABASE:,}"
    )

    print(
        f"Query: "
        f"{NUM_QUERY:,}"
    )

    print(
        f"K values: {K_VALUES}"
    )

    print(
        f"Temperature: {TEMPERATURE}"
    )

    print(
        f"Feature pooling: "
        f"mean of {NUM_CLASS_TOKENS} class tokens"
    )

    print(
        f"Local-only loading: True"
    )

    # --------------------------------------------------------
    # Check ImageNet
    # --------------------------------------------------------

    if not os.path.isdir(
        IMAGENET_VAL_ROOT
    ):

        raise FileNotFoundError(
            f"ImageNet val directory not found:\n"
            f"{IMAGENET_VAL_ROOT}"
        )

    # --------------------------------------------------------
    # Load existing protocol
    # --------------------------------------------------------

    protocol = load_protocol()

    (
        database_indices,
        query_indices,
    ) = extract_protocol_indices(
        protocol
    )

    # --------------------------------------------------------
    # Run all Pixio models
    # --------------------------------------------------------

    all_results = {}

    total_start = time.time()

    for model_name, model_dir in MODEL_DIRS.items():

        result = run_model(
            model_name=model_name,
            model_dir=model_dir,
            protocol=protocol,
            database_indices=database_indices,
            query_indices=query_indices,
        )

        all_results[
            model_name
        ] = result

    total_time = (
        time.time()
        - total_start
    )

    # --------------------------------------------------------
    # Save combined result
    # --------------------------------------------------------

    combined_path = os.path.join(
        RESULT_ROOT,
        "pixio_all_results.json",
    )

    with open(
        combined_path,
        "w",
    ) as f:

        json.dump(
            all_results,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n\n")
    print("#" * 100)
    print("# FINAL SUMMARY")
    print("#" * 100)

    print(
        f"\n{'Model':<20}"
        f"{'K=1':>12}"
        f"{'K=5':>12}"
        f"{'K=10':>12}"
        f"{'K=20':>12}"
        f"{'Dim':>10}"
        f"{'Total PFLOPs':>18}"
    )

    print("-" * 100)

    for model_name, result in all_results.items():

        r = result["knn"]["results"]

        dim = result["feature"]["dimension"]

        pflops = result["flops"][
            "knn_total_pflops"
        ]

        print(
            f"{model_name:<20}"
            f"{r['1']['accuracy'] * 100:>11.4f}%"
            f"{r['5']['accuracy'] * 100:>11.4f}%"
            f"{r['10']['accuracy'] * 100:>11.4f}%"
            f"{r['20']['accuracy'] * 100:>11.4f}%"
            f"{dim:>10}"
            f"{pflops:>18.6f}"
        )

    print("-" * 100)

    print(
        f"\nTotal wall-clock time: "
        f"{total_time / 3600:.2f} hours"
    )

    print(
        f"Combined result:"
        f"\n{combined_path}"
    )

    print("\nDone.")


# ============================================================
# Entry
# ============================================================

if False:  # entry point is defined below
    _single_pool_main()
def main():
    """Extract the 45k pool once, then evaluate cached 5/10/20/45-shot slices."""
    _single_pool_main()
    from vision_encoder_eval.workers.knn.different_shot.multishot_protocol import run_multishot_from_cache
    return run_multishot_from_cache(FEATURE_ROOT, RESULT_ROOT, "pixio")


if __name__ == "__main__":
    main()

