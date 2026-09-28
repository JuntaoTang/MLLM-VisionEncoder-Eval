import os
import json
import time
import random
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F
import faiss

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from transformers import AutoImageProcessor, AutoModel


# ============================================================
# 1. SETTINGS
# ============================================================

SEED = 42

# ------------------------------------------------------------
# ImageNet validation set
# ------------------------------------------------------------
IMAGENET_ROOT = "/workspace/root/val"

# ------------------------------------------------------------
# Unified 5/10/20/45-shot protocol
# ------------------------------------------------------------
PROTOCOL_PATH = "/cache/metaclip_knn/val_45shot_5query_seed42_protocol.json"
TRAIN_SHOTS = [5, 10, 20, 45]
EXTRACTION_SHOTS = [45]
TRAIN_POOL_PER_CLASS = 45
QUERY_PER_CLASS = 5
K = 20

# ------------------------------------------------------------
# DINO models
# ------------------------------------------------------------
MODEL_PATHS = {
    "dino-vits16": "/cache/models/model/dino-vits16",
    "dino-vits8": "/cache/models/model/dino-vits8",
    "dino-vitb16": "/cache/models/model/dino-vitb16",
    "dino-vitb8": "/cache/models/model/dino-vitb8",
}

# ------------------------------------------------------------
# Cache
# ------------------------------------------------------------
CACHE_ROOT = "/cache/dino_knn_cache_45shot"
RESULT_ROOT = "/cache/dino_knn_results_45shot"

os.makedirs(CACHE_ROOT, exist_ok=True)
os.makedirs(RESULT_ROOT, exist_ok=True)

# ------------------------------------------------------------
# Runtime
# ------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 128
NUM_WORKERS = 8

# ------------------------------------------------------------
# Feature
# ------------------------------------------------------------
FEATURE_DTYPE = np.float32

# ------------------------------------------------------------
# KNN
# ------------------------------------------------------------
K_VALUES = [1, 5, 10, 20]
MAX_K = max(K_VALUES)

TEMPERATURE = 0.07


# ============================================================
# 2. SEED
# ============================================================

def seed_everything(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


seed_everything(SEED)


# ============================================================
# 3. DATASET
# ============================================================

class ImageNetSubset(Dataset):

    def __init__(
        self,
        root,
        indices,
        processor,
    ):

        self.dataset = datasets.ImageFolder(
            root=root
        )

        self.indices = list(indices)
        self.processor = processor

    def __len__(self):

        return len(self.indices)

    def __getitem__(self, idx):

        real_idx = self.indices[idx]

        path, label = self.dataset.samples[real_idx]

        image = Image.open(
            path
        ).convert("RGB")

        pixel_values = self.processor(
            images=image,
            return_tensors="pt",
        )["pixel_values"].squeeze(0)

        return pixel_values, label


# ============================================================
# 4. LOAD PROTOCOL
# ============================================================

def load_protocol():

    os.makedirs(
        os.path.dirname(PROTOCOL_PATH),
        exist_ok=True,
    )

    if os.path.exists(PROTOCOL_PATH):

        with open(PROTOCOL_PATH, "r") as f:
            protocol = json.load(f)

        # Canonical protocol compatibility: derive legacy per-shot fields from
        # the single ordered 45-image-per-class pool without changing the file.
        if "train_indices_by_shot" not in protocol:
            pool = np.asarray(protocol["train_pool_indices"], dtype=np.int64).reshape(1000, 45)
            protocol["train_indices_by_shot"] = {
                str(shot): pool[:, :shot].reshape(-1).tolist()
                for shot in (5, 10, 20, 45)
            }
        protocol.setdefault("train_pool_size", len(protocol["train_pool_indices"]))
        protocol.setdefault("query_size", len(protocol["query_indices"]))

        required = {
            "seed": SEED,
            "train_pool_per_class": TRAIN_POOL_PER_CLASS,
            "query_per_class": QUERY_PER_CLASS,
            "train_shots": TRAIN_SHOTS,
        }

        for key, expected in required.items():

            actual = protocol.get(key)

            if actual != expected:

                raise RuntimeError(
                    f"Protocol field {key} mismatch: "
                    f"{actual} != {expected}"
                )

        print("=" * 110)
        print("LOADING FIXED 5/10/20/45-SHOT KNN PROTOCOL")
        print("=" * 110)
        print(f"Protocol : {PROTOCOL_PATH}")
        print(f"Train root: {protocol.get('dataset', IMAGENET_ROOT)}")
        print(f"Shots    : {TRAIN_SHOTS}")
        print(f"Query    : {len(protocol['query_indices'])}")
        print()

        return protocol

    dataset = datasets.ImageFolder(
        root=IMAGENET_ROOT,
    )

    if len(dataset.classes) != 1000:

        raise RuntimeError(
            f"Expected 1000 classes, got {len(dataset.classes)}"
        )

    by_class = {c: [] for c in range(1000)}

    for idx, (_, label) in enumerate(dataset.samples):
        by_class[label].append(idx)

    rng = random.Random(SEED)

    train_pool_by_class = {}
    query_by_class = {}
    train_indices_by_shot = {str(s): [] for s in TRAIN_SHOTS}
    query_indices = []

    for class_id in range(1000):

        indices = by_class[class_id].copy()
        needed = TRAIN_POOL_PER_CLASS + QUERY_PER_CLASS

        if len(indices) < needed:

            raise RuntimeError(
                f"Class {class_id} has {len(indices)} images; "
                f"need {needed}"
            )

        rng.shuffle(indices)

        train_pool = indices[:TRAIN_POOL_PER_CLASS]
        query = indices[TRAIN_POOL_PER_CLASS:needed]

        train_pool_by_class[str(class_id)] = train_pool
        query_by_class[str(class_id)] = query
        query_indices.extend(query)

        for shot in TRAIN_SHOTS:
            train_indices_by_shot[str(shot)].extend(train_pool[:shot])

    protocol = {
        "seed": SEED,
        "dataset": IMAGENET_ROOT,
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
        json.dump(protocol, f, indent=2)

    print("=" * 110)
    print("CREATED FIXED 5/10/20/45-SHOT KNN PROTOCOL")
    print("=" * 110)
    print(f"Protocol : {PROTOCOL_PATH}")
    print(f"Train root: {IMAGENET_ROOT}")
    print(f"Shots    : {TRAIN_SHOTS}")
    print(f"Query    : {len(query_indices)}")
    print()

    return protocol


# ============================================================
# 5. DATALOADER
# ============================================================

def build_loader(
    indices,
    processor,
):

    dataset = ImageNetSubset(
        root=IMAGENET_ROOT,
        indices=indices,
        processor=processor,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
        drop_last=False,
    )

    return loader


# ============================================================
# 6. FEATURE EXTRACTION
# ============================================================

@torch.inference_mode()
def extract_features(
    model,
    loader,
    split_name,
):

    model.eval()

    features_list = []
    labels_list = []

    total_images = len(loader.dataset)

    processed = 0

    start_time = time.time()

    for batch_idx, (
        pixel_values,
        labels,
    ) in enumerate(loader):

        pixel_values = pixel_values.to(
            DEVICE,
            non_blocking=True,
        )

        # ----------------------------------------------------
        # DINO backbone forward
        # ----------------------------------------------------

        outputs = model(
            pixel_values=pixel_values
        )

        # ----------------------------------------------------
        # DINO feature:
        #
        # Last-layer CLS token
        #
        # [B, N, D]
        #        ↓
        # [B, D]
        # ----------------------------------------------------

        if hasattr(
            outputs,
            "last_hidden_state"
        ):

            features = outputs.last_hidden_state[:, 0, :]

        elif hasattr(
            outputs,
            "pooler_output"
        ):

            features = outputs.pooler_output

        else:

            raise RuntimeError(
                "Model output does not contain "
                "last_hidden_state or pooler_output."
            )

        # ----------------------------------------------------
        # FP32
        # ----------------------------------------------------

        features = features.float()

        # ----------------------------------------------------
        # L2 normalize
        # ----------------------------------------------------

        features = F.normalize(
            features,
            p=2,
            dim=-1,
        )

        features_list.append(
            features.cpu().numpy().astype(
                FEATURE_DTYPE
            )
        )

        labels_list.append(
            labels.numpy()
        )

        processed += pixel_values.shape[0]

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            batch_idx == 0
            or batch_idx % 10 == 0
            or processed == total_images
        ):

            elapsed = time.time() - start_time

            speed = (
                processed /
                max(elapsed, 1e-8)
            )

            print(
                f"[{split_name}] "
                f"{processed:5d}/{total_images:5d} | "
                f"{speed:8.2f} img/s | "
                f"{elapsed:8.1f}s"
            )

    features = np.concatenate(
        features_list,
        axis=0,
    )

    labels = np.concatenate(
        labels_list,
        axis=0,
    )

    elapsed = time.time() - start_time

    speed = (
        len(features) /
        max(elapsed, 1e-8)
    )

    # --------------------------------------------------------
    # Sanity check
    # --------------------------------------------------------

    finite = np.isfinite(
        features
    ).all()

    norms = np.linalg.norm(
        features,
        axis=1,
    )

    print()

    print(
        f"[{split_name}] Finished"
    )

    print(
        f"  Features : {features.shape}"
    )

    print(
        f"  Time     : {elapsed:.2f}s"
    )

    print(
        f"  Speed    : {speed:.2f} img/s"
    )

    print(
        f"  Finite   : {finite}"
    )

    print(
        f"  Norm mean: {norms.mean():.8f}"
    )

    print(
        f"  Norm std : {norms.std():.8e}"
    )

    return (
        features,
        labels,
        elapsed,
    )


# ============================================================
# 7. DINO ARCHITECTURE FLOPs
# ============================================================

def get_config_value(
    config,
    *names,
    default=None,
):

    for name in names:

        if hasattr(
            config,
            name,
        ):

            value = getattr(
                config,
                name,
            )

            if value is not None:
                return value

    return default


def calculate_vit_flops(
    model,
    processor,
):

    config = model.config

    # --------------------------------------------------------
    # Image size
    # --------------------------------------------------------

    image_size = get_config_value(
        config,
        "image_size",
        default=None,
    )

    if image_size is None:

        processor_size = getattr(
            processor,
            "size",
            None,
        )

        if isinstance(
            processor_size,
            dict,
        ):

            image_size = (
                processor_size.get(
                    "height"
                )
                or processor_size.get(
                    "shortest_edge"
                )
            )

        else:

            image_size = processor_size

    if image_size is None:

        image_size = 224

    if isinstance(
        image_size,
        (list, tuple),
    ):

        image_height = int(
            image_size[0]
        )

        image_width = int(
            image_size[1]
        )

    else:

        image_height = int(
            image_size
        )

        image_width = int(
            image_size
        )

    # --------------------------------------------------------
    # Patch size
    # --------------------------------------------------------

    patch_size = get_config_value(
        config,
        "patch_size",
        default=None,
    )

    if patch_size is None:

        raise ValueError(
            "Cannot determine patch_size."
        )

    patch_size = int(
        patch_size
    )

    # --------------------------------------------------------
    # Hidden dimension
    # --------------------------------------------------------

    hidden_size = get_config_value(
        config,
        "hidden_size",
        "embed_dim",
        default=None,
    )

    if hidden_size is None:

        raise ValueError(
            "Cannot determine hidden_size/embed_dim."
        )

    hidden_size = int(
        hidden_size
    )

    # --------------------------------------------------------
    # Depth
    # --------------------------------------------------------

    num_layers = get_config_value(
        config,
        "num_hidden_layers",
        "num_layers",
        "depth",
        default=None,
    )

    if num_layers is None:

        raise ValueError(
            "Cannot determine transformer depth."
        )

    num_layers = int(
        num_layers
    )

    # --------------------------------------------------------
    # Attention heads
    # --------------------------------------------------------

    num_heads = get_config_value(
        config,
        "num_attention_heads",
        "num_heads",
        default=None,
    )

    if num_heads is None:

        raise ValueError(
            "Cannot determine number of attention heads."
        )

    num_heads = int(
        num_heads
    )

    # --------------------------------------------------------
    # MLP dimension
    # --------------------------------------------------------

    mlp_dim = get_config_value(
        config,
        "intermediate_size",
        "mlp_dim",
        default=None,
    )

    if mlp_dim is None:

        mlp_ratio = get_config_value(
            config,
            "mlp_ratio",
            default=4.0,
        )

        mlp_dim = int(
            hidden_size * float(
                mlp_ratio
            )
        )

    mlp_dim = int(
        mlp_dim
    )

    # --------------------------------------------------------
    # Number of patches
    # --------------------------------------------------------

    grid_h = (
        image_height //
        patch_size
    )

    grid_w = (
        image_width //
        patch_size
    )

    num_patches = (
        grid_h * grid_w
    )

    # CLS token
    num_tokens = (
        num_patches + 1
    )

    # ========================================================
    # FLOPs
    #
    # Convention:
    #
    # 1 MAC = 2 FLOPs
    # ========================================================

    # --------------------------------------------------------
    # Patch embedding
    #
    # Conv2d:
    #
    # N_patches *
    # patch_size^2 *
    # 3 *
    # hidden_size
    #
    # MACs
    # --------------------------------------------------------

    patch_embed_macs = (
        num_patches
        * patch_size
        * patch_size
        * 3
        * hidden_size
    )

    # --------------------------------------------------------
    # QKV projections + output projection
    #
    # Q = N * D * D
    # K = N * D * D
    # V = N * D * D
    # O = N * D * D
    #
    # total = 4 * N * D^2
    # --------------------------------------------------------

    qkv_projection_macs = (
        4
        * num_tokens
        * hidden_size
        * hidden_size
    )

    # --------------------------------------------------------
    # Attention QK^T
    #
    # N^2 * D
    # --------------------------------------------------------

    attention_qk_macs = (
        num_tokens
        * num_tokens
        * hidden_size
    )

    # --------------------------------------------------------
    # Attention AV
    #
    # N^2 * D
    # --------------------------------------------------------

    attention_av_macs = (
        num_tokens
        * num_tokens
        * hidden_size
    )

    # --------------------------------------------------------
    # MLP
    #
    # D -> MLP -> D
    #
    # 2 * N * D * MLP
    # --------------------------------------------------------

    mlp_macs = (
        2
        * num_tokens
        * hidden_size
        * mlp_dim
    )

    # --------------------------------------------------------
    # One transformer block
    # --------------------------------------------------------

    block_macs = (
        qkv_projection_macs
        + attention_qk_macs
        + attention_av_macs
        + mlp_macs
    )

    # --------------------------------------------------------
    # All blocks
    # --------------------------------------------------------

    transformer_macs = (
        num_layers
        * block_macs
    )

    # --------------------------------------------------------
    # Total
    # --------------------------------------------------------

    total_macs = (
        patch_embed_macs
        + transformer_macs
    )

    total_flops = (
        total_macs * 2
    )

    return {

        "image_height":
            image_height,

        "image_width":
            image_width,

        "patch_size":
            patch_size,

        "grid_h":
            grid_h,

        "grid_w":
            grid_w,

        "num_patches":
            num_patches,

        "num_tokens":
            num_tokens,

        "hidden_size":
            hidden_size,

        "num_layers":
            num_layers,

        "num_heads":
            num_heads,

        "mlp_dim":
            mlp_dim,

        "patch_embed_macs":
            patch_embed_macs,

        "qkv_projection_macs_per_block":
            qkv_projection_macs,

        "attention_qk_macs_per_block":
            attention_qk_macs,

        "attention_av_macs_per_block":
            attention_av_macs,

        "mlp_macs_per_block":
            mlp_macs,

        "transformer_macs":
            transformer_macs,

        "total_macs_per_image":
            total_macs,

        "feature_flops_per_image":
            total_flops,
    }


# ============================================================
# 8. KNN FLOPs
# ============================================================

def calculate_knn_flops(
    num_database,
    num_query,
    feature_dim,
    max_k,
):

    # --------------------------------------------------------
    # FAISS IndexFlatIP
    #
    # query × database matrix multiplication
    #
    # Q * DB * D MACs
    #
    # 1 MAC = 2 FLOPs
    # --------------------------------------------------------

    similarity_flops = (
        num_query
        * num_database
        * feature_dim
        * 2
    )

    # --------------------------------------------------------
    # Voting
    #
    # Approximate accounting:
    #
    # subtraction
    # division
    # exp
    # multiplication
    # accumulation
    #
    # 5 FLOPs / neighbor
    #
    # Tiny compared with similarity computation,
    # but explicitly included.
    # --------------------------------------------------------

    voting_flops = (
        num_query
        * max_k
        * 5
    )

    total_knn_flops = (
        similarity_flops
        + voting_flops
    )

    return {

        "similarity_flops":
            similarity_flops,

        "voting_flops":
            voting_flops,

        "total_knn_flops":
            total_knn_flops,
    }


# ============================================================
# 9. KNN
# ============================================================

def weighted_knn(
    database_features,
    database_labels,
    query_features,
    query_labels,
):

    print()
    print("=" * 110)
    print("FAISS KNN")
    print("=" * 110)

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

    feature_dim = (
        database_features.shape[1]
    )

    # --------------------------------------------------------
    # Normalized feature + inner product
    #
    # = cosine similarity
    # --------------------------------------------------------

    index = faiss.IndexFlatIP(
        feature_dim
    )

    index.add(
        database_features
    )

    print(
        f"Database : {database_features.shape}"
    )

    print(
        f"Query    : {query_features.shape}"
    )

    print(
        f"Dim      : {feature_dim}"
    )

    print(
        f"Index    : IndexFlatIP"
    )

    print(
        f"Temperature : {TEMPERATURE}"
    )

    print()

    # --------------------------------------------------------
    # Search ONCE with K=20
    # --------------------------------------------------------

    start = time.time()

    similarities, indices = index.search(
        query_features,
        MAX_K,
    )

    search_time = (
        time.time() - start
    )

    print(
        f"FAISS search time: "
        f"{search_time:.4f}s"
    )

    results = {}

    # ========================================================
    # K values
    # ========================================================

    for k in K_VALUES:

        start = time.time()

        sims_k = similarities[
            :, :k
        ]

        inds_k = indices[
            :, :k
        ]

        labels_k = (
            database_labels[
                inds_k
            ]
        )

        # ----------------------------------------------------
        # Temperature weighted voting
        # ----------------------------------------------------

        scaled = (
            sims_k
            - sims_k.max(
                axis=1,
                keepdims=True,
            )
        ) / TEMPERATURE

        weights = np.exp(
            scaled
        )

        predictions = np.empty(
            len(query_labels),
            dtype=np.int64,
        )

        for i in range(
            len(query_labels)
        ):

            neighbor_labels = (
                labels_k[i]
            )

            neighbor_weights = (
                weights[i]
            )

            unique_labels, inverse = (
                np.unique(
                    neighbor_labels,
                    return_inverse=True,
                )
            )

            class_weights = np.bincount(
                inverse,
                weights=neighbor_weights,
            )

            predictions[i] = (
                unique_labels[
                    np.argmax(
                        class_weights
                    )
                ]
            )

        accuracy = (
            predictions == query_labels
        ).mean() * 100.0

        voting_time = (
            time.time() - start
        )

        results[k] = {

            "accuracy":
                float(accuracy),

            "voting_time_sec":
                float(voting_time),
        }

        print(
            f"K={k:2d} | "
            f"Top-1={accuracy:.4f}% | "
            f"Voting={voting_time:.4f}s"
        )

    return (
        results,
        search_time,
    )


# ============================================================
# 10. CACHE PATHS
# ============================================================

def get_cache_paths(
    model_name,
    train_shot,
):

    model_cache_dir = os.path.join(
        CACHE_ROOT,
        model_name,
    )

    os.makedirs(
        model_cache_dir,
        exist_ok=True,
    )

    return {

        "database_features":
            os.path.join(
                model_cache_dir,
                f"train_{train_shot}shot_features.npy",
            ),

        "database_labels":
            os.path.join(
                model_cache_dir,
                f"train_{train_shot}shot_labels.npy",
            ),

        "query_features":
            os.path.join(
                model_cache_dir,
                "query_features.npy",
            ),

        "query_labels":
            os.path.join(
                model_cache_dir,
                "query_labels.npy",
            ),
    }


# ============================================================
# 11. EVALUATE ONE MODEL
# ============================================================

def evaluate_model(
    model_name,
    model_path,
    database_indices,
    query_indices,
    train_shot,
):

    print()
    print()
    print("#" * 110)
    print(
        f"MODEL: {model_name} | {train_shot}-shot"
    )
    print("#" * 110)

    print(
        f"Model path : {model_path}"
    )

    print(
        f"Device     : {DEVICE}"
    )

    print(
        f"Precision  : FP32"
    )

    print()

    # ========================================================
    # Load processor
    # ========================================================

    processor = (
        AutoImageProcessor.from_pretrained(
            model_path,
            local_files_only=True,
        )
    )

    # ========================================================
    # Load model
    # ========================================================

    model = (
        AutoModel.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.float32,
        )
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    # ========================================================
    # Model info
    # ========================================================

    num_params = sum(
        p.numel()
        for p in model.parameters()
    )

    config = model.config

    hidden_size = get_config_value(
        config,
        "hidden_size",
        "embed_dim",
        default=None,
    )

    num_layers = get_config_value(
        config,
        "num_hidden_layers",
        "num_layers",
        "depth",
        default=None,
    )

    num_heads = get_config_value(
        config,
        "num_attention_heads",
        "num_heads",
        default=None,
    )

    patch_size = get_config_value(
        config,
        "patch_size",
        default=None,
    )

    print(
        f"Parameters : "
        f"{num_params / 1e6:.2f} M"
    )

    print(
        f"Hidden size: "
        f"{hidden_size}"
    )

    print(
        f"Layers     : "
        f"{num_layers}"
    )

    print(
        f"Heads      : "
        f"{num_heads}"
    )

    print(
        f"Patch size : "
        f"{patch_size}"
    )

    # ========================================================
    # FLOPs
    # ========================================================

    flops_info = (
        calculate_vit_flops(
            model,
            processor,
        )
    )

    feature_flops_per_image = (
        flops_info[
            "feature_flops_per_image"
        ]
    )

    num_database = (
        len(database_indices)
    )

    num_query = (
        len(query_indices)
    )

    total_images = (
        num_database
        + num_query
    )

    # --------------------------------------------------------
    # Feature extraction FLOPs
    # --------------------------------------------------------

    feature_extraction_flops = (
        total_images
        * feature_flops_per_image
    )

    # ========================================================
    # Print FLOPs architecture
    # ========================================================

    print()
    print("-" * 110)
    print("FEATURE EXTRACTION FLOPs")
    print("-" * 110)

    print(
        f"Image size       : "
        f"{flops_info['image_height']} x "
        f"{flops_info['image_width']}"
    )

    print(
        f"Patch size       : "
        f"{flops_info['patch_size']}"
    )

    print(
        f"Patch grid       : "
        f"{flops_info['grid_h']} x "
        f"{flops_info['grid_w']}"
    )

    print(
        f"Num patches      : "
        f"{flops_info['num_patches']}"
    )

    print(
        f"Num tokens       : "
        f"{flops_info['num_tokens']}"
    )

    print(
        f"Hidden size      : "
        f"{flops_info['hidden_size']}"
    )

    print(
        f"Transformer depth: "
        f"{flops_info['num_layers']}"
    )

    print(
        f"Attention heads  : "
        f"{flops_info['num_heads']}"
    )

    print(
        f"MLP dimension    : "
        f"{flops_info['mlp_dim']}"
    )

    print(
        f"Feature FLOPs/img: "
        f"{feature_flops_per_image / 1e9:.6f} GFLOPs"
    )

    print(
        f"DB feature FLOPs : "
        f"{num_database * feature_flops_per_image / 1e12:.6f} TFLOPs"
    )

    print(
        f"Query feature FLOPs: "
        f"{num_query * feature_flops_per_image / 1e12:.6f} TFLOPs"
    )

    print(
        f"Total feature FLOPs: "
        f"{feature_extraction_flops / 1e15:.9f} PFLOPs"
    )

    # ========================================================
    # KNN FLOPs
    # ========================================================

    # Feature dimension is determined after feature extraction.
    # For ViT CLS it is normally hidden_size.
    feature_dim = int(
        hidden_size
    )

    knn_flops = calculate_knn_flops(
        num_database=num_database,
        num_query=num_query,
        feature_dim=feature_dim,
        max_k=MAX_K,
    )

    similarity_flops = (
        knn_flops[
            "similarity_flops"
        ]
    )

    voting_flops = (
        knn_flops[
            "voting_flops"
        ]
    )

    total_knn_flops = (
        knn_flops[
            "total_knn_flops"
        ]
    )

    # --------------------------------------------------------
    # TOTAL
    # --------------------------------------------------------

    total_flops = (
        feature_extraction_flops
        + total_knn_flops
    )

    print()
    print("-" * 110)
    print("KNN FLOPs")
    print("-" * 110)

    print(
        f"KNN similarity FLOPs : "
        f"{similarity_flops:,}"
    )

    print(
        f"KNN similarity       : "
        f"{similarity_flops / 1e15:.9f} PFLOPs"
    )

    print(
        f"KNN voting FLOPs     : "
        f"{voting_flops:,}"
    )

    print(
        f"KNN total FLOPs      : "
        f"{total_knn_flops:,}"
    )

    print(
        f"KNN total            : "
        f"{total_knn_flops / 1e15:.9f} PFLOPs"
    )

    print()
    print("-" * 110)
    print("TOTAL KNN FLOPs")
    print("-" * 110)

    print(
        f"Feature extraction   : "
        f"{feature_extraction_flops / 1e15:.9f} PFLOPs"
    )

    print(
        f"KNN search + voting  : "
        f"{total_knn_flops / 1e15:.9f} PFLOPs"
    )

    print(
        f"TOTAL                : "
        f"{total_flops / 1e15:.9f} PFLOPs"
    )

    # ========================================================
    # Build dataloaders
    # ========================================================

    database_loader = build_loader(
        database_indices,
        processor,
    )

    query_loader = build_loader(
        query_indices,
        processor,
    )

    # ========================================================
    # Cache
    # ========================================================

    cache_paths = get_cache_paths(
        model_name,
        train_shot,
    )

    # ========================================================
    # Database features
    # ========================================================

    if (
        os.path.exists(
            cache_paths[
                "database_features"
            ]
        )
        and
        os.path.exists(
            cache_paths[
                "database_labels"
            ]
        )
    ):

        print()
        print(
            "Loading cached database features..."
        )

        database_features = np.load(
            cache_paths[
                "database_features"
            ]
        )

        database_labels = np.load(
            cache_paths[
                "database_labels"
            ]
        )

        database_extract_time = 0.0

    else:

        print()
        print("=" * 110)
        print("EXTRACT DATABASE FEATURES")
        print("=" * 110)

        (
            database_features,
            database_labels,
            database_extract_time,
        ) = extract_features(
            model,
            database_loader,
            "DATABASE",
        )

        np.save(
            cache_paths[
                "database_features"
            ],
            database_features,
        )

        np.save(
            cache_paths[
                "database_labels"
            ],
            database_labels,
        )

    # ========================================================
    # Query features
    # ========================================================

    if (
        os.path.exists(
            cache_paths[
                "query_features"
            ]
        )
        and
        os.path.exists(
            cache_paths[
                "query_labels"
            ]
        )
    ):

        print()
        print(
            "Loading cached query features..."
        )

        query_features = np.load(
            cache_paths[
                "query_features"
            ]
        )

        query_labels = np.load(
            cache_paths[
                "query_labels"
            ]
        )

        query_extract_time = 0.0

    else:

        print()
        print("=" * 110)
        print("EXTRACT QUERY FEATURES")
        print("=" * 110)

        (
            query_features,
            query_labels,
            query_extract_time,
        ) = extract_features(
            model,
            query_loader,
            "QUERY",
        )

        np.save(
            cache_paths[
                "query_features"
            ],
            query_features,
        )

        np.save(
            cache_paths[
                "query_labels"
            ],
            query_labels,
        )

    # ========================================================
    # Validate feature dimensions
    # ========================================================

    actual_feature_dim = (
        database_features.shape[1]
    )

    if actual_feature_dim != feature_dim:

        print()
        print(
            "WARNING:"
        )

        print(
            f"Config hidden size = "
            f"{feature_dim}"
        )

        print(
            f"Actual feature dim = "
            f"{actual_feature_dim}"
        )

        print(
            "Using actual feature dimension "
            "for KNN FLOPs."
        )

        feature_dim = (
            actual_feature_dim
        )

        # Recalculate KNN FLOPs
        knn_flops = calculate_knn_flops(
            num_database=num_database,
            num_query=num_query,
            feature_dim=feature_dim,
            max_k=MAX_K,
        )

        similarity_flops = (
            knn_flops[
                "similarity_flops"
            ]
        )

        voting_flops = (
            knn_flops[
                "voting_flops"
            ]
        )

        total_knn_flops = (
            knn_flops[
                "total_knn_flops"
            ]
        )

        total_flops = (
            feature_extraction_flops
            + total_knn_flops
        )

    # ========================================================
    # KNN
    # ========================================================

    knn_results, search_time = (
        weighted_knn(
            database_features=
                database_features,

            database_labels=
                database_labels,

            query_features=
                query_features,

            query_labels=
                query_labels,
        )
    )

    # ========================================================
    # Result JSON
    # ========================================================

    result = {

        "model":
            model_name,

        "model_path":
            model_path,

        "protocol": {

            "protocol_file":
                PROTOCOL_PATH,

            "seed":
                SEED,

            "database_size":
                num_database,

            "query_size":
                num_query,

            "database_per_class":
                int(train_shot),

            "train_shot":
                int(train_shot),

            "query_per_class":
                5,

            "database_query_overlap":
                0,
        },

        "runtime": {

            "device":
                DEVICE,

            "precision":
                "FP32",

            "batch_size":
                BATCH_SIZE,

            "num_workers":
                NUM_WORKERS,
        },

        "feature": {

            "type":
                "CLS",

            "source":
                "last_hidden_state[:, 0, :]",

            "l2_normalized":
                True,

            "dtype":
                "float32",

            "dimension":
                int(feature_dim),
        },

        "model_config": {

            "parameters":
                int(num_params),

            "parameters_million":
                float(
                    num_params / 1e6
                ),

            "hidden_size":
                int(
                    flops_info[
                        "hidden_size"
                    ]
                ),

            "num_layers":
                int(
                    flops_info[
                        "num_layers"
                    ]
                ),

            "num_heads":
                int(
                    flops_info[
                        "num_heads"
                    ]
                ),

            "patch_size":
                int(
                    flops_info[
                        "patch_size"
                    ]
                ),

            "image_height":
                int(
                    flops_info[
                        "image_height"
                    ]
                ),

            "image_width":
                int(
                    flops_info[
                        "image_width"
                    ]
                ),

            "num_patches":
                int(
                    flops_info[
                        "num_patches"
                    ]
                ),

            "num_tokens":
                int(
                    flops_info[
                        "num_tokens"
                    ]
                ),

            "mlp_dim":
                int(
                    flops_info[
                        "mlp_dim"
                    ]
                ),
        },

        "flops": {

            "convention":
                "1 MAC = 2 FLOPs",

            "feature_flops_per_image":
                int(
                    feature_flops_per_image
                ),

            "feature_gflops_per_image":
                float(
                    feature_flops_per_image
                    / 1e9
                ),

            "database_feature_flops":
                int(
                    num_database
                    * feature_flops_per_image
                ),

            "query_feature_flops":
                int(
                    num_query
                    * feature_flops_per_image
                ),

            "feature_extraction_flops":
                int(
                    feature_extraction_flops
                ),

            "feature_extraction_pflops":
                float(
                    feature_extraction_flops
                    / 1e15
                ),

            "knn_similarity_flops":
                int(
                    similarity_flops
                ),

            "knn_voting_flops":
                int(
                    voting_flops
                ),

            "knn_total_flops":
                int(
                    total_knn_flops
                ),

            "knn_total_pflops":
                float(
                    total_knn_flops
                    / 1e15
                ),

            "total_flops":
                int(
                    total_flops
                ),

            "total_pflops":
                float(
                    total_flops
                    / 1e15
                ),
        },

        "timing": {

            "database_feature_extraction_sec":
                float(
                    database_extract_time
                ),

            "query_feature_extraction_sec":
                float(
                    query_extract_time
                ),

            "faiss_search_sec":
                float(
                    search_time
                ),
        },

        "knn": {

            str(k): {

                "top1_accuracy":
                    float(
                        knn_results[
                            k
                        ][
                            "accuracy"
                        ]
                    ),

                "voting_time_sec":
                    float(
                        knn_results[
                            k
                        ][
                            "voting_time_sec"
                        ]
                    ),
            }

            for k in K_VALUES
        },
    }

    # ========================================================
    # Save
    # ========================================================

    result_path = os.path.join(
        RESULT_ROOT,
        f"{model_name}_{train_shot}shot_knn_result.json",
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

    # ========================================================
    # Final model summary
    # ========================================================

    print()
    print("#" * 110)
    print(
        f"FINAL RESULT: {model_name} | {train_shot}-shot"
    )
    print("#" * 110)

    print()

    for k in K_VALUES:

        print(
            f"K={k:2d}: "
            f"{knn_results[k]['accuracy']:.4f}%"
        )

    print()

    print(
        f"Feature extraction FLOPs : "
        f"{feature_extraction_flops / 1e15:.9f} PFLOPs"
    )

    print(
        f"KNN search + voting      : "
        f"{total_knn_flops / 1e15:.9f} PFLOPs"
    )

    print(
        f"KNN TOTAL FLOPs          : "
        f"{total_flops / 1e15:.9f} PFLOPs"
    )

    print()

    print(
        f"Result saved to:\n"
        f"{result_path}"
    )

    # ========================================================
    # Free GPU
    # ========================================================

    del model
    del processor
    del database_loader
    del query_loader

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

        torch.cuda.ipc_collect()

    return result


# ============================================================
# 12. MAIN
# ============================================================

def main():

    print("=" * 110)
    print(
        "DINO ImageNet 5/10/20/45-shot KNN Benchmark"
    )
    print("=" * 110)

    print()
    print(
        f"ImageNet       : {IMAGENET_ROOT}"
    )

    print(
        f"Protocol       : {PROTOCOL_PATH}"
    )

    print(
        f"Cache          : {CACHE_ROOT}"
    )

    print(
        f"Results        : {RESULT_ROOT}"
    )

    print(
        f"Device         : {DEVICE}"
    )

    print(
        f"Precision      : FP32"
    )

    print(
        f"Batch size     : {BATCH_SIZE}"
    )

    print(
        f"Workers        : {NUM_WORKERS}"
    )

    print(
        f"Feature        : CLS"
    )

    print(
        f"L2 normalize   : True"
    )

    print(
        f"K              : {K_VALUES}"
    )

    print(
        f"Temperature     : {TEMPERATURE}"
    )

    print()

    # ========================================================
    # Load protocol
    # ========================================================

    protocol = load_protocol()

    query_indices = protocol["query_indices"]

    # ========================================================
    # Run models
    # ========================================================

    all_results = {}

    for model_name, model_path in MODEL_PATHS.items():

        all_results[model_name] = {}

        for train_shot in EXTRACTION_SHOTS:

            database_indices = protocol["train_indices_by_shot"][str(train_shot)]

            result = evaluate_model(
                model_name=
                    model_name,

                model_path=
                    model_path,

                database_indices=
                    database_indices,

                query_indices=
                    query_indices,

                train_shot=
                    train_shot,
            )

            all_results[
                model_name
            ][
                str(train_shot)
            ] = result

    # ========================================================
    # Combined result
    # ========================================================

    combined_path = os.path.join(
        RESULT_ROOT,
        "dino_all_models_results_5_10_20_45shot.json",
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

    # ========================================================
    # Final table
    # ========================================================

    print()
    print()
    print("#" * 130)
    print(
        "DINO FINAL SUMMARY"
    )
    print("#" * 130)

    print(
        f"{'Model':<18}"
        f"{'Shot':>8}"
        f"{'K=1':>12}"
        f"{'K=5':>12}"
        f"{'K=10':>12}"
        f"{'K=20':>12}"
        f"{'Feature PFLOPs':>18}"
        f"{'KNN PFLOPs':>18}"
        f"{'TOTAL PFLOPs':>18}"
    )

    print(
        "-" * 130
    )

    for model_name, shot_results in all_results.items():

        for train_shot in EXTRACTION_SHOTS:

            result = shot_results[str(train_shot)]

            knn = result[
                "knn"
            ]

            flops = result[
                "flops"
            ]

            print(
                f"{model_name:<18}"
                f"{train_shot:>8}"
                f"{knn['1']['top1_accuracy']:>11.4f}%"
                f"{knn['5']['top1_accuracy']:>11.4f}%"
                f"{knn['10']['top1_accuracy']:>11.4f}%"
                f"{knn['20']['top1_accuracy']:>11.4f}%"
                f"{flops['feature_extraction_pflops']:>18.6f}"
                f"{flops['knn_total_pflops']:>18.9f}"
                f"{flops['total_pflops']:>18.6f}"
            )

    print(
        "-" * 130
    )

    print()
    print(
        f"Combined results:\n"
        f"{combined_path}"
    )


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    main()
    from multishot_protocol import run_multishot_from_cache
    run_multishot_from_cache(CACHE_ROOT, RESULT_ROOT, "dino")
