import os
import json
import time
import random
import warnings
from pathlib import Path

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
# 1. Global Settings
# ============================================================

SEED = 42

# ------------------------------------------------------------
# ImageNet
# ------------------------------------------------------------
IMAGENET_ROOT = "/workspace/root/val"

# Existing fixed protocol
PROTOCOL_PATH = "/cache/metaclip_knn/val_45shot_5query_seed42_protocol.json"

# ------------------------------------------------------------
# Models
# ------------------------------------------------------------
MODEL_PATHS = {
    "dinov2-small": "/cache/models/model/dinov2-small",
    "dinov2-base": "/cache/models/model/dinov2-base",
    "dinov2-large": "/cache/models/model/dinov2-large",
    "dinov2-giant": "/cache/models/model/dinov2-giant",
}

# ------------------------------------------------------------
# Cache / Results
# ------------------------------------------------------------
CACHE_ROOT = "/cache/dinov2_knn_cache_45pool"
RESULT_ROOT = "/cache/dinov2_knn_results_45pool"

os.makedirs(CACHE_ROOT, exist_ok=True)
os.makedirs(RESULT_ROOT, exist_ok=True)

# ------------------------------------------------------------
# Runtime
# ------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 128
NUM_WORKERS = 8

# Feature extraction
FEATURE_DTYPE = np.float32

# ------------------------------------------------------------
# KNN
# ------------------------------------------------------------
K_VALUES = [1, 5, 10, 20]
MAX_K = max(K_VALUES)

TEMPERATURE = 0.07

# ============================================================
# 2. Reproducibility
# ============================================================

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


seed_everything(SEED)


# ============================================================
# 3. Dataset
# ============================================================

class ImageDataset(Dataset):
    def __init__(self, root, indices, processor):
        self.dataset = datasets.ImageFolder(root=root)
        self.indices = indices
        self.processor = processor

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]

        path, label = self.dataset.samples[real_idx]

        image = Image.open(path).convert("RGB")

        pixel_values = self.processor(
            images=image,
            return_tensors="pt"
        )["pixel_values"].squeeze(0)

        return pixel_values, label


# ============================================================
# 4. Load Fixed 5-shot Protocol
# ============================================================

def load_protocol():

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

    print("=" * 100)
    print("Loading fixed KNN protocol")
    print("=" * 100)
    print(f"Protocol : {PROTOCOL_PATH}")

    # --------------------------------------------------------
    # Try several common field names
    # --------------------------------------------------------
    db_indices = None
    query_indices = None

    for key in [
        "train_pool_indices",
        "database_indices",
        "db_indices",
        "train_indices",
    ]:
        if key in protocol:
            db_indices = protocol[key]
            break

    for key in [
        "query_indices",
        "val_indices",
        "test_indices",
    ]:
        if key in protocol:
            query_indices = protocol[key]
            break

    if db_indices is None or query_indices is None:

        print("\nProtocol keys:")
        for k, v in protocol.items():
            if isinstance(v, (list, tuple)):
                print(f"  {k}: list[{len(v)}]")
            else:
                print(f"  {k}: {v}")

        raise KeyError(
            "Cannot find database/query indices in protocol JSON."
        )

    db_indices = [int(x) for x in db_indices]
    query_indices = [int(x) for x in query_indices]

    print(f"Database : {len(db_indices)}")
    print(f"Query    : {len(query_indices)}")
    print(f"Seed     : {SEED}")
    print()

    return db_indices, query_indices


# ============================================================
# 5. DataLoader
# ============================================================

def build_loader(root, indices, processor):

    dataset = ImageDataset(
        root=root,
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
# 6. Feature Extraction
# ============================================================

@torch.inference_mode()
def extract_features(model, loader, split_name):

    model.eval()

    all_features = []
    all_labels = []

    total = len(loader.dataset)

    start_time = time.time()
    processed = 0

    for batch_idx, (pixel_values, labels) in enumerate(loader):

        pixel_values = pixel_values.to(
            DEVICE,
            non_blocking=True
        )

        # ----------------------------------------------------
        # DINOv2 feature definition:
        #
        # outputs.last_hidden_state[:, 0, :]
        #
        # = CLS token
        # ----------------------------------------------------
        outputs = model(
            pixel_values=pixel_values
        )

        features = outputs.last_hidden_state[:, 0, :]

        # ----------------------------------------------------
        # Explicit FP32
        # ----------------------------------------------------
        features = features.float()

        # ----------------------------------------------------
        # L2 normalization
        # ----------------------------------------------------
        features = F.normalize(
            features,
            p=2,
            dim=-1
        )

        all_features.append(
            features.cpu().numpy().astype(FEATURE_DTYPE)
        )

        all_labels.append(
            labels.numpy()
        )

        processed += pixel_values.shape[0]

        if (
            batch_idx == 0
            or batch_idx % 10 == 0
            or processed == total
        ):
            elapsed = time.time() - start_time
            ips = processed / max(elapsed, 1e-8)

            print(
                f"[{split_name}] "
                f"{processed:5d}/{total:5d} | "
                f"{ips:8.2f} img/s | "
                f"{elapsed:8.1f}s"
            )

    features = np.concatenate(
        all_features,
        axis=0
    )

    labels = np.concatenate(
        all_labels,
        axis=0
    )

    elapsed = time.time() - start_time
    ips = len(features) / max(elapsed, 1e-8)

    print()
    print(
        f"[{split_name}] Finished: "
        f"{len(features)} images | "
        f"shape={features.shape} | "
        f"{ips:.2f} img/s | "
        f"{elapsed:.2f}s"
    )

    # --------------------------------------------------------
    # Sanity check
    # --------------------------------------------------------
    finite = np.isfinite(features).all()
    norms = np.linalg.norm(
        features,
        axis=1
    )

    print(
        f"[{split_name}] finite={finite} | "
        f"norm mean={norms.mean():.6f} | "
        f"norm std={norms.std():.6e}"
    )

    return features, labels, elapsed


# ============================================================
# 7. DINOv2 FLOPs Calculation
# ============================================================

def get_config_value(config, *names, default=None):

    for name in names:

        if hasattr(config, name):

            value = getattr(config, name)

            if value is not None:
                return value

    return default


def calculate_dinov2_flops(model, processor):

    config = model.config

    # --------------------------------------------------------
    # Image size
    # --------------------------------------------------------
    image_size = get_config_value(
        config,
        "image_size",
        default=None
    )

    if image_size is None:

        image_size = getattr(
            processor,
            "size",
            None
        )

        if isinstance(image_size, dict):

            image_size = (
                image_size.get("height")
                or image_size.get("shortest_edge")
            )

    if isinstance(image_size, (list, tuple)):
        image_height = image_size[0]
        image_width = image_size[1]
    else:
        image_height = int(image_size)
        image_width = int(image_size)

    # --------------------------------------------------------
    # Patch size
    # --------------------------------------------------------
    patch_size = get_config_value(
        config,
        "patch_size",
        default=14
    )

    patch_size = int(patch_size)

    # --------------------------------------------------------
    # Hidden dimension
    # --------------------------------------------------------
    hidden_size = get_config_value(
        config,
        "hidden_size",
        default=None
    )

    if hidden_size is None:
        raise ValueError(
            "Cannot determine hidden_size from DINOv2 config."
        )

    hidden_size = int(hidden_size)

    # --------------------------------------------------------
    # Transformer depth
    # --------------------------------------------------------
    num_layers = get_config_value(
        config,
        "num_hidden_layers",
        "num_layers",
        default=None
    )

    if num_layers is None:
        raise ValueError(
            "Cannot determine number of transformer layers."
        )

    num_layers = int(num_layers)

    # --------------------------------------------------------
    # MLP dimension
    # --------------------------------------------------------
    mlp_dim = get_config_value(
        config,
        "intermediate_size",
        "mlp_dim",
        default=4 * hidden_size
    )

    mlp_dim = int(mlp_dim)

    # --------------------------------------------------------
    # Number of patches
    # --------------------------------------------------------
    grid_h = image_height // patch_size
    grid_w = image_width // patch_size

    num_patches = grid_h * grid_w

    # + CLS token
    num_tokens = num_patches + 1

    # ========================================================
    # FLOPs
    #
    # Convention:
    # 1 MAC = 2 FLOPs
    # ========================================================

    # --------------------------------------------------------
    # 1. Patch Embedding
    #
    # Each patch:
    # patch_size * patch_size * 3
    # ->
    # hidden_size
    #
    # MAC:
    # num_patches * patch_area * 3 * hidden_size
    # --------------------------------------------------------
    patch_embed_macs = (
        num_patches
        * patch_size
        * patch_size
        * 3
        * hidden_size
    )

    # --------------------------------------------------------
    # 2. Transformer block
    #
    # QKV + output projection:
    #
    # 4 * N * D^2
    # --------------------------------------------------------
    projection_macs = (
        4
        * num_tokens
        * hidden_size
        * hidden_size
    )

    # --------------------------------------------------------
    # 3. Attention QK^T
    #
    # N^2 * D
    # --------------------------------------------------------
    attention_qk_macs = (
        num_tokens
        * num_tokens
        * hidden_size
    )

    # --------------------------------------------------------
    # 4. Attention AV
    #
    # N^2 * D
    # --------------------------------------------------------
    attention_av_macs = (
        num_tokens
        * num_tokens
        * hidden_size
    )

    # --------------------------------------------------------
    # 5. MLP
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
    # Per Transformer block
    # --------------------------------------------------------
    block_macs = (
        projection_macs
        + attention_qk_macs
        + attention_av_macs
        + mlp_macs
    )

    # --------------------------------------------------------
    # All Transformer blocks
    # --------------------------------------------------------
    transformer_macs = (
        num_layers
        * block_macs
    )

    # --------------------------------------------------------
    # Total MAC
    # --------------------------------------------------------
    total_macs_per_image = (
        patch_embed_macs
        + transformer_macs
    )

    # --------------------------------------------------------
    # MAC -> FLOPs
    # --------------------------------------------------------
    total_flops_per_image = (
        total_macs_per_image * 2
    )

    return {
        "image_height": image_height,
        "image_width": image_width,
        "patch_size": patch_size,
        "grid_h": grid_h,
        "grid_w": grid_w,
        "num_patches": num_patches,
        "num_tokens": num_tokens,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "mlp_dim": mlp_dim,

        "patch_embed_macs": patch_embed_macs,
        "projection_macs_per_block": projection_macs,
        "attention_qk_macs_per_block": attention_qk_macs,
        "attention_av_macs_per_block": attention_av_macs,
        "mlp_macs_per_block": mlp_macs,
        "transformer_macs": transformer_macs,

        "total_macs_per_image": total_macs_per_image,
        "feature_flops_per_image": total_flops_per_image,
    }


# ============================================================
# 8. KNN FLOPs
# ============================================================

def calculate_knn_flops(
    num_database,
    num_query,
    feature_dim,
    max_k
):

    # --------------------------------------------------------
    # FAISS IndexFlatIP:
    #
    # Every query compares against every database feature.
    #
    # q dot x:
    # D multiplications + D-1 additions
    #
    # Benchmark convention:
    # approximately D MACs = 2D FLOPs
    #
    # Total:
    # Q * DB * D * 2
    # --------------------------------------------------------
    similarity_flops = (
        num_query
        * num_database
        * feature_dim
        * 2
    )

    # --------------------------------------------------------
    # Top-k / voting FLOPs
    #
    # Similarity-weighted voting:
    #
    #   exp((sim-max_sim)/T)
    #   accumulate class weights
    #
    # We count:
    # - subtraction
    # - division
    # - exp approximately 1 FLOP
    # - multiply
    # - addition
    #
    # This is tiny compared with matrix multiplication,
    # but included because user requested total FLOPs.
    #
    # Approximate as 5 FLOPs / neighbor.
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
        "similarity_flops": similarity_flops,
        "voting_flops": voting_flops,
        "total_knn_flops": total_knn_flops,
    }


# ============================================================
# 9. Similarity-Weighted KNN
# ============================================================

def weighted_knn(
    database_features,
    database_labels,
    query_features,
    query_labels,
    k_values,
    temperature=0.07
):

    print()
    print("=" * 100)
    print("FAISS KNN")
    print("=" * 100)

    database_features = np.ascontiguousarray(
        database_features.astype(np.float32)
    )

    query_features = np.ascontiguousarray(
        query_features.astype(np.float32)
    )

    feature_dim = database_features.shape[1]

    print(f"Database features : {database_features.shape}")
    print(f"Query features    : {query_features.shape}")
    print(f"Feature dimension : {feature_dim}")
    print(f"Temperature       : {temperature}")
    print(f"K values          : {k_values}")
    print()

    # --------------------------------------------------------
    # Since features are L2 normalized:
    #
    # IndexFlatIP = cosine similarity
    # --------------------------------------------------------
    index = faiss.IndexFlatIP(feature_dim)

    index.add(database_features)

    print(
        f"FAISS index size  : {index.ntotal}"
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Search only once with max_k.
    #
    # Then reuse neighbors for K=1/5/10/20.
    #
    # This makes the FLOPs accounting consistent.
    # --------------------------------------------------------
    start = time.time()

    similarities, indices = index.search(
        query_features,
        max(k_values)
    )

    search_time = time.time() - start

    print(
        f"FAISS search time : {search_time:.4f}s"
    )

    results = {}

    for k in k_values:

        start = time.time()

        sims_k = similarities[:, :k]
        inds_k = indices[:, :k]

        labels_k = database_labels[inds_k]

        # ----------------------------------------------------
        # Numerical-stable temperature weighting
        # ----------------------------------------------------
        scaled = (
            sims_k
            - sims_k.max(axis=1, keepdims=True)
        ) / temperature

        weights = np.exp(scaled)

        predictions = np.empty(
            len(query_labels),
            dtype=np.int64
        )

        # ----------------------------------------------------
        # Weighted voting
        # ----------------------------------------------------
        for i in range(len(query_labels)):

            labels = labels_k[i]
            w = weights[i]

            unique_labels, inverse = np.unique(
                labels,
                return_inverse=True
            )

            class_weights = np.bincount(
                inverse,
                weights=w
            )

            predictions[i] = unique_labels[
                np.argmax(class_weights)
            ]

        accuracy = (
            predictions == query_labels
        ).mean() * 100.0

        elapsed = time.time() - start

        results[k] = {
            "accuracy": float(accuracy),
            "time_sec": float(elapsed),
        }

        print(
            f"K={k:2d} | "
            f"Top-1={accuracy:.4f}% | "
            f"voting_time={elapsed:.4f}s"
        )

    return results, search_time


# ============================================================
# 10. Cache
# ============================================================

def get_cache_paths(model_name):

    model_cache_dir = os.path.join(
        CACHE_ROOT,
        model_name
    )

    os.makedirs(
        model_cache_dir,
        exist_ok=True
    )

    return (
        os.path.join(
            model_cache_dir,
            "database_features.npy"
        ),
        os.path.join(
            model_cache_dir,
            "database_labels.npy"
        ),
        os.path.join(
            model_cache_dir,
            "query_features.npy"
        ),
        os.path.join(
            model_cache_dir,
            "query_labels.npy"
        ),
    )


# ============================================================
# 11. Main Model Evaluation
# ============================================================

def evaluate_model(
    model_name,
    model_path,
    db_indices,
    query_indices
):

    print()
    print("#" * 100)
    print(f"MODEL: {model_name}")
    print("#" * 100)

    print(f"Model path : {model_path}")
    print(f"Device     : {DEVICE}")
    print(f"Precision  : FP32")
    print()

    # --------------------------------------------------------
    # Load processor
    # --------------------------------------------------------
    processor = AutoImageProcessor.from_pretrained(
        model_path,
        local_files_only=True
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------
    model = AutoModel.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.float32
    )

    model = model.to(DEVICE)

    model.eval()

    # --------------------------------------------------------
    # Model information
    # --------------------------------------------------------
    num_params = sum(
        p.numel()
        for p in model.parameters()
    )

    config = model.config

    hidden_size = get_config_value(
        config,
        "hidden_size",
        default=None
    )

    num_layers = get_config_value(
        config,
        "num_hidden_layers",
        "num_layers",
        default=None
    )

    num_heads = get_config_value(
        config,
        "num_attention_heads",
        "num_heads",
        default=None
    )

    patch_size = get_config_value(
        config,
        "patch_size",
        default=None
    )

    print(
        f"Parameters : {num_params / 1e6:.2f} M"
    )

    print(
        f"Hidden size: {hidden_size}"
    )

    print(
        f"Layers     : {num_layers}"
    )

    print(
        f"Heads      : {num_heads}"
    )

    print(
        f"Patch size : {patch_size}"
    )

    # ========================================================
    # FLOPs
    # ========================================================

    flops_info = calculate_dinov2_flops(
        model,
        processor
    )

    feature_flops_per_image = (
        flops_info["feature_flops_per_image"]
    )

    num_database = len(db_indices)
    num_query = len(query_indices)

    num_total_images = (
        num_database
        + num_query
    )

    feature_extraction_flops = (
        num_total_images
        * feature_flops_per_image
    )

    # ========================================================
    # Print FLOPs architecture
    # ========================================================

    print()
    print("-" * 100)
    print("DINOv2 FLOPs")
    print("-" * 100)

    print(
        f"Image size          : "
        f"{flops_info['image_height']} x "
        f"{flops_info['image_width']}"
    )

    print(
        f"Patch size          : "
        f"{flops_info['patch_size']}"
    )

    print(
        f"Patch grid          : "
        f"{flops_info['grid_h']} x "
        f"{flops_info['grid_w']}"
    )

    print(
        f"Number of patches   : "
        f"{flops_info['num_patches']}"
    )

    print(
        f"Number of tokens    : "
        f"{flops_info['num_tokens']}"
    )

    print(
        f"Hidden size         : "
        f"{flops_info['hidden_size']}"
    )

    print(
        f"Transformer layers  : "
        f"{flops_info['num_layers']}"
    )

    print(
        f"MLP dimension      : "
        f"{flops_info['mlp_dim']}"
    )

    print(
        f"Feature FLOPs/img   : "
        f"{feature_flops_per_image / 1e9:.6f} GFLOPs"
    )

    print(
        f"DB feature FLOPs    : "
        f"{num_database * feature_flops_per_image / 1e12:.6f} TFLOPs"
    )

    print(
        f"Query feature FLOPs : "
        f"{num_query * feature_flops_per_image / 1e12:.6f} TFLOPs"
    )

    print(
        f"Total feature FLOPs : "
        f"{feature_extraction_flops / 1e15:.6f} PFLOPs"
    )

    # ========================================================
    # Dataset
    # ========================================================

    db_loader = build_loader(
        IMAGENET_ROOT,
        db_indices,
        processor
    )

    query_loader = build_loader(
        IMAGENET_ROOT,
        query_indices,
        processor
    )

    # ========================================================
    # Cache
    # ========================================================

    (
        db_feature_path,
        db_label_path,
        query_feature_path,
        query_label_path,
    ) = get_cache_paths(model_name)

    # ========================================================
    # Database features
    # ========================================================

    if (
        os.path.exists(db_feature_path)
        and os.path.exists(db_label_path)
    ):

        print()
        print(
            f"Loading cached database features..."
        )

        database_features = np.load(
            db_feature_path
        )

        database_labels = np.load(
            db_label_path
        )

        db_extract_time = 0.0

    else:

        print()
        print("=" * 100)
        print("Extracting database features")
        print("=" * 100)

        database_features, database_labels, db_extract_time = (
            extract_features(
                model,
                db_loader,
                "DATABASE"
            )
        )

        np.save(
            db_feature_path,
            database_features
        )

        np.save(
            db_label_path,
            database_labels
        )

        print(
            f"Saved: {db_feature_path}"
        )

    # ========================================================
    # Query features
    # ========================================================

    if (
        os.path.exists(query_feature_path)
        and os.path.exists(query_label_path)
    ):

        print()
        print(
            f"Loading cached query features..."
        )

        query_features = np.load(
            query_feature_path
        )

        query_labels = np.load(
            query_label_path
        )

        query_extract_time = 0.0

    else:

        print()
        print("=" * 100)
        print("Extracting query features")
        print("=" * 100)

        query_features, query_labels, query_extract_time = (
            extract_features(
                model,
                query_loader,
                "QUERY"
            )
        )

        np.save(
            query_feature_path,
            query_features
        )

        np.save(
            query_label_path,
            query_labels
        )

        print(
            f"Saved: {query_feature_path}"
        )

    # ========================================================
    # Sanity check
    # ========================================================

    assert database_features.shape[0] == num_database
    assert query_features.shape[0] == num_query

    feature_dim = database_features.shape[1]

    print()
    print(
        f"Database feature shape: "
        f"{database_features.shape}"
    )

    print(
        f"Query feature shape   : "
        f"{query_features.shape}"
    )

    # ========================================================
    # KNN FLOPs
    # ========================================================

    knn_flops = calculate_knn_flops(
        num_database=num_database,
        num_query=num_query,
        feature_dim=feature_dim,
        max_k=MAX_K
    )

    similarity_flops = knn_flops[
        "similarity_flops"
    ]

    voting_flops = knn_flops[
        "voting_flops"
    ]

    total_knn_flops = knn_flops[
        "total_knn_flops"
    ]

    # ========================================================
    # TOTAL FLOPs
    #
    # Feature Extraction
    #       +
    # KNN Similarity Search
    #       +
    # KNN Voting
    # ========================================================

    total_flops = (
        feature_extraction_flops
        + total_knn_flops
    )

    # ========================================================
    # Print FLOPs summary
    # ========================================================

    print()
    print("=" * 100)
    print("FLOPs SUMMARY")
    print("=" * 100)

    print(
        f"Feature extraction FLOPs : "
        f"{feature_extraction_flops:,}"
    )

    print(
        f"Feature extraction       : "
        f"{feature_extraction_flops / 1e15:.9f} PFLOPs"
    )

    print(
        f"KNN similarity FLOPs     : "
        f"{similarity_flops:,}"
    )

    print(
        f"KNN similarity           : "
        f"{similarity_flops / 1e15:.9f} PFLOPs"
    )

    print(
        f"KNN voting FLOPs         : "
        f"{voting_flops:,}"
    )

    print(
        f"KNN total FLOPs          : "
        f"{total_knn_flops:,}"
    )

    print(
        f"KNN total                : "
        f"{total_knn_flops / 1e15:.9f} PFLOPs"
    )

    print()
    print(
        f"TOTAL KNN FLOPs          : "
        f"{total_flops:,}"
    )

    print(
        f"TOTAL KNN                : "
        f"{total_flops / 1e15:.9f} PFLOPs"
    )

    # ========================================================
    # KNN
    # ========================================================

    knn_results, search_time = weighted_knn(
        database_features=database_features,
        database_labels=database_labels,
        query_features=query_features,
        query_labels=query_labels,
        k_values=K_VALUES,
        temperature=TEMPERATURE,
    )

    # ========================================================
    # Result object
    # ========================================================

    result = {
        "model": model_name,
        "model_path": model_path,

        "protocol": {
            "protocol_file": PROTOCOL_PATH,
            "seed": SEED,
            "database_size": num_database,
            "query_size": num_query,
            "database_per_class": 5,
            "query_per_class": 5,
            "database_query_overlap": 0,
        },

        "runtime": {
            "device": DEVICE,
            "precision": "FP32",
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
        },

        "model_config": {
            "parameters": num_params,
            "parameters_million": num_params / 1e6,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "patch_size": patch_size,
        },

        "feature": {
            "type": "CLS token",
            "source": "outputs.last_hidden_state[:, 0, :]",
            "l2_normalized": True,
            "dtype": "float32",
            "dimension": int(feature_dim),
        },

        "flops": {
            "convention": "1 MAC = 2 FLOPs",

            "feature_flops_per_image":
                int(feature_flops_per_image),

            "feature_gflops_per_image":
                float(feature_flops_per_image / 1e9),

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
                int(feature_extraction_flops),

            "feature_extraction_pflops":
                float(
                    feature_extraction_flops
                    / 1e15
                ),

            "knn_similarity_flops":
                int(similarity_flops),

            "knn_voting_flops":
                int(voting_flops),

            "knn_total_flops":
                int(total_knn_flops),

            "knn_total_pflops":
                float(
                    total_knn_flops
                    / 1e15
                ),

            "total_flops":
                int(total_flops),

            "total_pflops":
                float(
                    total_flops
                    / 1e15
                ),
        },

        "timing": {
            "database_feature_extraction_sec":
                float(db_extract_time),

            "query_feature_extraction_sec":
                float(query_extract_time),

            "faiss_search_sec":
                float(search_time),
        },

        "knn": {
            str(k): {
                "top1_accuracy":
                    knn_results[k]["accuracy"],
                "voting_time_sec":
                    knn_results[k]["time_sec"],
            }
            for k in K_VALUES
        },
    }

    # ========================================================
    # Save result
    # ========================================================

    result_path = os.path.join(
        RESULT_ROOT,
        f"{model_name}_knn_result.json"
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

    # ========================================================
    # Final summary
    # ========================================================

    print()
    print("#" * 100)
    print(f"FINAL RESULT: {model_name}")
    print("#" * 100)

    print()
    print("KNN Accuracy:")

    for k in K_VALUES:

        print(
            f"  K={k:2d}: "
            f"{knn_results[k]['accuracy']:.4f}%"
        )

    print()
    print("FLOPs:")

    print(
        f"  Feature extraction : "
        f"{feature_extraction_flops / 1e15:.6f} PFLOPs"
    )

    print(
        f"  KNN search         : "
        f"{total_knn_flops / 1e15:.9f} PFLOPs"
    )

    print(
        f"  TOTAL              : "
        f"{total_flops / 1e15:.6f} PFLOPs"
    )

    print()
    print(
        f"Result saved to:\n"
        f"{result_path}"
    )

    # ========================================================
    # Free GPU memory
    # ========================================================

    del model
    del processor

    del db_loader
    del query_loader

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    return result


# ============================================================
# 12. Main
# ============================================================

def _single_pool_main():

    print("=" * 100)
    print("DINOv2 ImageNet 5-shot / class KNN Benchmark")
    print("=" * 100)

    print()
    print(f"ImageNet       : {IMAGENET_ROOT}")
    print(f"Protocol       : {PROTOCOL_PATH}")
    print(f"Cache root     : {CACHE_ROOT}")
    print(f"Result root    : {RESULT_ROOT}")
    print(f"Device         : {DEVICE}")
    print(f"Batch size     : {BATCH_SIZE}")
    print(f"Workers        : {NUM_WORKERS}")
    print(f"Precision      : FP32")
    print(f"Feature        : CLS")
    print(f"K values       : {K_VALUES}")
    print(f"Temperature    : {TEMPERATURE}")
    print()

    # --------------------------------------------------------
    # Load fixed split
    # --------------------------------------------------------
    db_indices, query_indices = load_protocol()

    # --------------------------------------------------------
    # Validate split
    # --------------------------------------------------------
    db_set = set(db_indices)
    query_set = set(query_indices)

    overlap = db_set.intersection(
        query_set
    )

    print(
        f"DB/query overlap: {len(overlap)}"
    )

    if len(overlap) != 0:
        raise RuntimeError(
            "Database and query have overlap!"
        )

    # --------------------------------------------------------
    # Run all DINOv2 models
    # --------------------------------------------------------
    all_results = {}

    for model_name, model_path in MODEL_PATHS.items():

        result = evaluate_model(
            model_name=model_name,
            model_path=model_path,
            db_indices=db_indices,
            query_indices=query_indices,
        )

        all_results[model_name] = result

    # ========================================================
    # Save combined result
    # ========================================================

    combined_path = os.path.join(
        RESULT_ROOT,
        "dinov2_all_models_results.json"
    )

    with open(
        combined_path,
        "w"
    ) as f:
        json.dump(
            all_results,
            f,
            indent=2
        )

    # ========================================================
    # Final table
    # ========================================================

    print()
    print()
    print("#" * 120)
    print("DINOv2 FINAL SUMMARY")
    print("#" * 120)

    print(
        f"{'Model':<18}"
        f"{'K=1':>12}"
        f"{'K=5':>12}"
        f"{'K=10':>12}"
        f"{'K=20':>12}"
        f"{'Feature PFLOPs':>18}"
        f"{'KNN PFLOPs':>16}"
        f"{'TOTAL PFLOPs':>18}"
    )

    print("-" * 120)

    for model_name, result in all_results.items():

        knn = result["knn"]
        flops = result["flops"]

        print(
            f"{model_name:<18}"
            f"{knn['1']['top1_accuracy']:>11.4f}%"
            f"{knn['5']['top1_accuracy']:>11.4f}%"
            f"{knn['10']['top1_accuracy']:>11.4f}%"
            f"{knn['20']['top1_accuracy']:>11.4f}%"
            f"{flops['feature_extraction_pflops']:>18.6f}"
            f"{flops['knn_total_pflops']:>16.9f}"
            f"{flops['total_pflops']:>18.6f}"
        )

    print("-" * 120)

    print()
    print(
        f"Combined results saved to:\n"
        f"{combined_path}"
    )


# ============================================================
# Entry
# ============================================================

if False:  # entry point is defined below
    _single_pool_main()
def main():
    """Extract the 45k pool once, then evaluate cached 5/10/20/45-shot slices."""
    _single_pool_main()
    from multishot_protocol import run_multishot_from_cache
    return run_multishot_from_cache(CACHE_ROOT, RESULT_ROOT, "dinov2")


if __name__ == "__main__":
    main()

