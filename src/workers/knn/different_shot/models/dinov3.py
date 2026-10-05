
from vision_encoder_eval.core.runtime import asset_path
import os
import json
import time
import random
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets

from transformers import AutoImageProcessor, AutoModel

import faiss


# ============================================================
# 1. Configuration
# ============================================================

MODEL_NAME = "DINOv3-ViTL16-224"

MODEL_PATH = (
    asset_path('model_assets', 'model/dinov3-vitl16-pretrain-lvd1689m')
)

IMAGENET_VAL_DIR = "/workspace/root/val"

PROTOCOL_PATH = (
    asset_path('runtime', 'metaclip_knn/val_45shot_5query_seed42_protocol.json')
)

CACHE_ROOT = asset_path('runtime', 'dinov3_knn_pooler_45pool')

CACHE_DIR = os.path.join(
    CACHE_ROOT,
    MODEL_NAME,
)

RESULT_PATH = os.path.join(
    CACHE_DIR,
    "knn_results.json",
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 128
NUM_WORKERS = 8

K_VALUES = [1, 5, 10, 20]

TEMPERATURE = 0.07

SEED = 42

# FLOPs convention:
# 1 MAC = 2 FLOPs
FLOPS_PER_MAC = 2


# ============================================================
# 2. Reproducibility
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)

os.makedirs(CACHE_DIR, exist_ok=True)


# ============================================================
# 3. General utilities
# ============================================================

def check_finite_tensor(x, name):
    """
    Check whether NumPy array or PyTorch Tensor
    contains NaN or Inf.
    """

    if isinstance(x, np.ndarray):
        finite = np.isfinite(x).all()

        if not finite:
            num_nan = int(np.isnan(x).sum())
            num_inf = int(np.isinf(x).sum())

            raise RuntimeError(
                f"{name} contains invalid values: "
                f"NaN={num_nan}, Inf={num_inf}"
            )

    elif isinstance(x, torch.Tensor):
        finite = torch.isfinite(x).all().item()

        if not finite:
            num_nan = int(torch.isnan(x).sum().item())
            num_inf = int(torch.isinf(x).sum().item())

            raise RuntimeError(
                f"{name} contains invalid values: "
                f"NaN={num_nan}, Inf={num_inf}"
            )

    else:
        raise TypeError(
            f"{name} has unsupported type: {type(x)}"
        )


def feature_statistics(features, name):
    """
    Print feature statistics.
    Supports NumPy arrays and PyTorch tensors.
    """

    check_finite_tensor(features, name)

    if isinstance(features, torch.Tensor):
        features_np = (
            features.detach()
            .cpu()
            .numpy()
        )
    else:
        features_np = np.asarray(features)

    norms = np.linalg.norm(
        features_np,
        axis=1,
    )

    print(f"\n[{name} statistics]")
    print(f"Shape       : {features_np.shape}")
    print(f"Mean        : {features_np.mean():.8f}")
    print(f"Std         : {features_np.std():.8f}")
    print(f"Min         : {features_np.min():.8f}")
    print(f"Max         : {features_np.max():.8f}")
    print(f"Norm mean   : {norms.mean():.8f}")
    print(f"Norm min    : {norms.min():.8f}")
    print(f"Norm max    : {norms.max():.8f}")


def load_json(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def save_json(obj, path):
    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            obj,
            f,
            indent=2,
            ensure_ascii=False,
        )


def is_valid_feature_cache(
    feature_path,
    label_path,
    expected_num_samples,
):
    """
    Check whether a feature cache exists and is valid.
    """

    if not os.path.exists(feature_path):
        return False

    if not os.path.exists(label_path):
        return False

    try:
        features = np.load(
            feature_path,
            mmap_mode="r",
        )

        labels = np.load(
            label_path,
            mmap_mode="r",
        )

        if len(features) != expected_num_samples:
            print(
                f"Invalid cache size: "
                f"{len(features)} != {expected_num_samples}"
            )
            return False

        if len(labels) != expected_num_samples:
            print(
                f"Invalid label size: "
                f"{len(labels)} != {expected_num_samples}"
            )
            return False

        if not np.isfinite(features).all():
            print(
                "Invalid cache: contains NaN or Inf."
            )
            return False

        if features.ndim != 2:
            print(
                f"Invalid feature dimension: "
                f"{features.shape}"
            )
            return False

        return True

    except Exception as exc:
        print(
            f"Cannot validate cache: {exc}"
        )
        return False


# ============================================================
# 4. Protocol
# ============================================================

def extract_protocol_indices(protocol):
    """
    Supported database keys:
        database_indices
        db_indices
        train_indices
        database
        db

    Supported query keys:
        query_indices
        val_indices
        test_indices
        query
        test
    """

    database_keys = [
        "train_pool_indices",
        "database_indices",
        "db_indices",
        "train_indices",
        "database",
        "db",
    ]

    query_keys = [
        "query_indices",
        "val_indices",
        "test_indices",
        "query",
        "test",
    ]

    database_indices = None
    query_indices = None

    for key in database_keys:
        if key in protocol:
            database_indices = protocol[key]
            break

    for key in query_keys:
        if key in protocol:
            query_indices = protocol[key]
            break

    if database_indices is None:
        raise KeyError(
            "Cannot find database indices. "
            f"Available keys: {list(protocol.keys())}"
        )

    if query_indices is None:
        raise KeyError(
            "Cannot find query indices. "
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

    return database_indices, query_indices


def validate_protocol(
    database_indices,
    query_indices,
    targets,
):
    """
    Validate:
    - Index ranges
    - No DB/query overlap
    - Label distribution
    """

    if len(database_indices) == 0:
        raise RuntimeError(
            "Database indices are empty."
        )

    if len(query_indices) == 0:
        raise RuntimeError(
            "Query indices are empty."
        )

    if database_indices.min() < 0:
        raise RuntimeError(
            "Database indices contain negative values."
        )

    if query_indices.min() < 0:
        raise RuntimeError(
            "Query indices contain negative values."
        )

    if database_indices.max() >= len(targets):
        raise RuntimeError(
            "Database index exceeds dataset size."
        )

    if query_indices.max() >= len(targets):
        raise RuntimeError(
            "Query index exceeds dataset size."
        )

    database_set = set(
        database_indices.tolist()
    )

    query_set = set(
        query_indices.tolist()
    )

    overlap = database_set.intersection(
        query_set
    )

    if len(overlap) > 0:
        raise RuntimeError(
            f"Database/query overlap detected: "
            f"{len(overlap)} samples."
        )

    database_labels = targets[
        database_indices
    ]

    query_labels = targets[
        query_indices
    ]

    print("\n" + "=" * 80)
    print("Protocol Validation")
    print("=" * 80)

    print(
        f"Database samples : {len(database_indices)}"
    )

    print(
        f"Query samples    : {len(query_indices)}"
    )

    print(
        f"Overlap          : {len(overlap)}"
    )

    print(
        f"Database classes : "
        f"{len(np.unique(database_labels))}"
    )

    print(
        f"Query classes    : "
        f"{len(np.unique(query_labels))}"
    )

    database_counts = np.bincount(
        database_labels,
        minlength=1000,
    )

    query_counts = np.bincount(
        query_labels,
        minlength=1000,
    )

    database_nonzero = database_counts[
        database_counts > 0
    ]

    query_nonzero = query_counts[
        query_counts > 0
    ]

    print(
        "Database samples/class: "
        f"min={database_nonzero.min()}, "
        f"max={database_counts.max()}"
    )

    print(
        "Query samples/class: "
        f"min={query_nonzero.min()}, "
        f"max={query_counts.max()}"
    )


# ============================================================
# 5. Dataset
# ============================================================

class IndexedImageDataset(Dataset):

    def __init__(
        self,
        image_folder_dataset,
        indices,
        processor,
    ):
        self.dataset = image_folder_dataset

        self.indices = np.asarray(
            indices,
            dtype=np.int64,
        )

        self.processor = processor

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        original_index = int(
            self.indices[index]
        )

        image, label = self.dataset[
            original_index
        ]

        if not isinstance(image, Image.Image):
            raise TypeError(
                "Expected PIL image. "
                "ImageFolder must use transform=None."
            )

        processed = self.processor(
            images=image,
            return_tensors="pt",
        )

        pixel_values = processed[
            "pixel_values"
        ].squeeze(0)

        return pixel_values, label


# ============================================================
# 6. Model
# ============================================================

def load_model():

    print("\n" + "=" * 80)
    print("Loading DINOv3")
    print("=" * 80)

    print(f"Model path : {MODEL_PATH}")
    print(f"Device     : {DEVICE}")
    print("Dtype      : torch.float32")

    processor = (
        AutoImageProcessor.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
        )
    )

    # Prefer the new Transformers argument.
    try:
        model = AutoModel.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
            dtype=torch.float32,
        )

    except TypeError:
        # Compatibility with older Transformers versions.
        model = AutoModel.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
            torch_dtype=torch.float32,
        )

    model = model.to(DEVICE)

    # Force every model parameter to FP32.
    model = model.float()

    model.eval()

    # Check model parameters for NaN/Inf.
    print(
        "Checking model parameters..."
    )

    invalid_parameters = []

    with torch.no_grad():
        for name, parameter in model.named_parameters():

            if not torch.isfinite(parameter).all():
                invalid_parameters.append(name)

    if len(invalid_parameters) > 0:
        raise RuntimeError(
            "Model parameters contain NaN or Inf. "
            f"Examples: {invalid_parameters[:10]}"
        )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    config = model.config

    print(
        f"Parameters : "
        f"{parameter_count / 1e6:.2f} M"
    )

    if hasattr(config, "hidden_size"):
        print(
            f"Hidden size: {config.hidden_size}"
        )

    if hasattr(config, "num_hidden_layers"):
        print(
            f"Layers     : "
            f"{config.num_hidden_layers}"
        )

    if hasattr(config, "num_attention_heads"):
        print(
            f"Heads      : "
            f"{config.num_attention_heads}"
        )

    if hasattr(config, "patch_size"):
        print(
            f"Patch size : {config.patch_size}"
        )

    print(
        "Model loaded successfully."
    )

    return processor, model


# ============================================================
# 7. Feature extraction
# ============================================================

@torch.no_grad()
def extract_features(
    model,
    dataloader,
    description,
):
    """
    Extract pooler_output features using FP32.

    Returns:
        features: float32 NumPy array [N, D]
        labels: int64 NumPy array [N]
    """

    all_features = []
    all_labels = []

    start_time = time.time()

    total_samples = len(
        dataloader.dataset
    )

    for step, batch in enumerate(
        dataloader
    ):

        pixel_values, labels = batch

        pixel_values = pixel_values.to(
            DEVICE,
            dtype=torch.float32,
            non_blocking=True,
        )

        check_finite_tensor(
            pixel_values,
            "Input pixel_values",
        )

        outputs = model(
            pixel_values=pixel_values,
        )

        if not hasattr(
            outputs,
            "pooler_output",
        ):
            raise RuntimeError(
                "Model output does not contain "
                "pooler_output. "
                f"Available fields: {outputs.keys()}"
            )

        features = outputs.pooler_output

        if features is None:
            raise RuntimeError(
                "outputs.pooler_output is None."
            )

        # Check before normalization.
        check_finite_tensor(
            features,
            "Raw pooler features",
        )

        features = features.float()

        # L2 normalization.
        features = F.normalize(
            features,
            p=2,
            dim=-1,
            eps=1e-12,
        )

        check_finite_tensor(
            features,
            "Normalized features",
        )

        features_np = (
            features.detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        labels_np = (
            labels.detach()
            .cpu()
            .numpy()
            .astype(np.int64)
        )

        all_features.append(
            features_np
        )

        all_labels.append(
            labels_np
        )

        processed_samples = min(
            (step + 1) * dataloader.batch_size,
            total_samples,
        )

        if (
            step == 0
            or (step + 1) % 20 == 0
            or processed_samples == total_samples
        ):
            elapsed = (
                time.time() - start_time
            )

            speed = (
                processed_samples
                / max(elapsed, 1e-6)
            )

            print(
                f"{description}: "
                f"{processed_samples}/{total_samples} "
                f"images | "
                f"{speed:.2f} img/s"
            )

    features = np.concatenate(
        all_features,
        axis=0,
    )

    labels = np.concatenate(
        all_labels,
        axis=0,
    )

    check_finite_tensor(
        features,
        f"{description} final features",
    )

    return features, labels


# ============================================================
# 8. Feature cache
# ============================================================

def save_features(
    features,
    labels,
    feature_path,
    label_path,
):
    check_finite_tensor(
        features,
        "Features before saving",
    )

    np.save(
        feature_path,
        features.astype(np.float32),
    )

    np.save(
        label_path,
        labels.astype(np.int64),
    )


def load_features(
    feature_path,
    label_path,
):
    features = np.load(
        feature_path
    )

    labels = np.load(
        label_path
    )

    check_finite_tensor(
        features,
        f"Cached features: {feature_path}",
    )

    return (
        features.astype(np.float32),
        labels.astype(np.int64),
    )


# ============================================================
# 9. KNN prediction
# ============================================================

def weighted_knn_predict(
    index,
    database_labels,
    query_features,
    k,
    temperature=0.07,
):
    """
    FAISS IndexFlatIP.

    Since features are L2-normalized,
    inner product equals cosine similarity.

    Weight:
        exp(similarity / temperature)
    """

    similarities, neighbors = index.search(
        query_features.astype(np.float32),
        k,
    )

    predictions = []

    num_classes = int(
        database_labels.max()
    ) + 1

    for row in range(
        len(query_features)
    ):

        neighbor_indices = (
            neighbors[row]
        )

        neighbor_similarities = (
            similarities[row]
        )

        valid = (
            neighbor_indices >= 0
        )

        neighbor_indices = (
            neighbor_indices[valid]
        )

        neighbor_similarities = (
            neighbor_similarities[valid]
        )

        neighbor_labels = (
            database_labels[
                neighbor_indices
            ]
        )

        weights = np.exp(
            neighbor_similarities
            / temperature
        )

        class_scores = np.zeros(
            num_classes,
            dtype=np.float64,
        )

        for label, weight in zip(
            neighbor_labels,
            weights,
        ):
            class_scores[int(label)] += (
                float(weight)
            )

        predictions.append(
            int(np.argmax(class_scores))
        )

    return np.asarray(
        predictions,
        dtype=np.int64,
    )


# ============================================================
# 10. KNN evaluation and FLOPs
# ============================================================

def evaluate_knn(
    database_features,
    database_labels,
    query_features,
    query_labels,
):
    feature_dim = (
        database_features.shape[1]
    )

    num_database = (
        len(database_features)
    )

    num_query = (
        len(query_features)
    )

    index = faiss.IndexFlatIP(
        feature_dim
    )

    index.add(
        database_features.astype(
            np.float32
        )
    )

    print("\n" + "=" * 80)
    print("KNN Evaluation")
    print("=" * 80)

    print(
        f"Database size : {num_database}"
    )

    print(
        f"Query size    : {num_query}"
    )

    print(
        f"Feature dim   : {feature_dim}"
    )

    results = {}

    for k in K_VALUES:

        print(
            f"\nEvaluating K={k}"
        )

        predictions = (
            weighted_knn_predict(
                index=index,
                database_labels=database_labels,
                query_features=query_features,
                k=k,
                temperature=TEMPERATURE,
            )
        )

        accuracy = (
            predictions == query_labels
        ).mean() * 100.0

        # ----------------------------------------------------
        # FLOPs
        # ----------------------------------------------------
        #
        # Similarity:
        # N_query * N_database * D MACs
        #
        # 1 MAC = 2 FLOPs
        #
        # Voting:
        # approximately 2 FLOPs per neighbor
        #
        # Top-k sorting and exp are not included.
        # ----------------------------------------------------

        similarity_flops = (
            num_query
            * num_database
            * feature_dim
            * FLOPS_PER_MAC
        )

        voting_flops = (
            num_query
            * k
            * 2
        )

        total_flops = (
            similarity_flops
            + voting_flops
        )

        result = {
            "k": int(k),
            "accuracy_percent": float(
                accuracy
            ),
            "num_database": int(
                num_database
            ),
            "num_query": int(
                num_query
            ),
            "feature_dim": int(
                feature_dim
            ),
            "temperature": float(
                TEMPERATURE
            ),
            "similarity_flops": int(
                similarity_flops
            ),
            "voting_flops": int(
                voting_flops
            ),
            "total_flops": int(
                total_flops
            ),
            "total_pflops": float(
                total_flops / 1e15
            ),
        }

        results[str(k)] = result

        print(
            f"Top-{k:<2d} Accuracy : "
            f"{accuracy:.4f}%"
        )

        print(
            f"Similarity FLOPs : "
            f"{similarity_flops:,}"
        )

        print(
            f"Voting FLOPs     : "
            f"{voting_flops:,}"
        )

        print(
            f"Total FLOPs      : "
            f"{total_flops:,}"
        )

        print(
            f"Total PFLOPs     : "
            f"{total_flops / 1e15:.8f}"
        )

    return results


# ============================================================
# 11. Main
# ============================================================

def _single_pool_main():

    print("\n" + "=" * 80)
    print("DINOv3 ImageNet KNN Evaluation")
    print("=" * 80)

    print(
        f"Model       : {MODEL_PATH}"
    )

    print(
        f"ImageNet val: {IMAGENET_VAL_DIR}"
    )

    print(
        f"Protocol    : {PROTOCOL_PATH}"
    )

    print(
        f"Cache       : {CACHE_DIR}"
    )

    print(
        f"Batch size  : {BATCH_SIZE}"
    )

    print(
        f"Num workers : {NUM_WORKERS}"
    )

    print(
        f"Device      : {DEVICE}"
    )

    print(
        "Precision   : FP32"
    )

    # --------------------------------------------------------
    # Load ImageNet validation dataset
    # --------------------------------------------------------

    print(
        "\nLoading ImageNet validation dataset..."
    )

    imagenet_dataset = datasets.ImageFolder(
        root=IMAGENET_VAL_DIR,
        transform=None,
    )

    print(
        "ImageNet validation samples: "
        f"{len(imagenet_dataset)}"
    )

    targets = np.asarray(
        imagenet_dataset.targets,
        dtype=np.int64,
    )

    # --------------------------------------------------------
    # Load and validate protocol
    # --------------------------------------------------------

    protocol = load_json(
        PROTOCOL_PATH
    )

    database_indices, query_indices = (
        extract_protocol_indices(
            protocol
        )
    )

    validate_protocol(
        database_indices=database_indices,
        query_indices=query_indices,
        targets=targets,
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    processor, model = load_model()

    # --------------------------------------------------------
    # Build datasets and dataloaders
    # --------------------------------------------------------

    database_dataset = IndexedImageDataset(
        image_folder_dataset=imagenet_dataset,
        indices=database_indices,
        processor=processor,
    )

    query_dataset = IndexedImageDataset(
        image_folder_dataset=imagenet_dataset,
        indices=query_indices,
        processor=processor,
    )

    database_loader = DataLoader(
        database_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
    )

    query_loader = DataLoader(
        query_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
    )

    # --------------------------------------------------------
    # Cache paths
    # --------------------------------------------------------

    database_feature_path = os.path.join(
        CACHE_DIR,
        "database_features.npy",
    )

    database_label_path = os.path.join(
        CACHE_DIR,
        "database_labels.npy",
    )

    query_feature_path = os.path.join(
        CACHE_DIR,
        "query_features.npy",
    )

    query_label_path = os.path.join(
        CACHE_DIR,
        "query_labels.npy",
    )

    # --------------------------------------------------------
    # Database features
    # --------------------------------------------------------

    valid_database_cache = (
        is_valid_feature_cache(
            feature_path=database_feature_path,
            label_path=database_label_path,
            expected_num_samples=len(
                database_indices
            ),
        )
    )

    if valid_database_cache:

        print(
            "\nLoading cached database features..."
        )

        database_features, database_labels = (
            load_features(
                feature_path=database_feature_path,
                label_path=database_label_path,
            )
        )

    else:

        print(
            "\nExtracting database features..."
        )

        database_features, database_labels = (
            extract_features(
                model=model,
                dataloader=database_loader,
                description="Database",
            )
        )

        save_features(
            features=database_features,
            labels=database_labels,
            feature_path=database_feature_path,
            label_path=database_label_path,
        )

        print(
            "Saved database features to:"
        )

        print(
            database_feature_path
        )

    # --------------------------------------------------------
    # Query features
    # --------------------------------------------------------

    valid_query_cache = (
        is_valid_feature_cache(
            feature_path=query_feature_path,
            label_path=query_label_path,
            expected_num_samples=len(
                query_indices
            ),
        )
    )

    if valid_query_cache:

        print(
            "\nLoading cached query features..."
        )

        query_features, query_labels = (
            load_features(
                feature_path=query_feature_path,
                label_path=query_label_path,
            )
        )

    else:

        print(
            "\nExtracting query features..."
        )

        query_features, query_labels = (
            extract_features(
                model=model,
                dataloader=query_loader,
                description="Query",
            )
        )

        save_features(
            features=query_features,
            labels=query_labels,
            feature_path=query_feature_path,
            label_path=query_label_path,
        )

        print(
            "Saved query features to:"
        )

        print(
            query_feature_path
        )

    # --------------------------------------------------------
    # Feature statistics
    # --------------------------------------------------------

    feature_statistics(
        database_features,
        "Database features",
    )

    feature_statistics(
        query_features,
        "Query features",
    )

    # --------------------------------------------------------
    # Shape and label checks
    # --------------------------------------------------------

    if database_features.ndim != 2:
        raise RuntimeError(
            "Database features must be 2D."
        )

    if query_features.ndim != 2:
        raise RuntimeError(
            "Query features must be 2D."
        )

    if (
        database_features.shape[1]
        != query_features.shape[1]
    ):
        raise RuntimeError(
            "Database/query feature dimensions "
            "do not match."
        )

    if len(database_features) != len(
        database_labels
    ):
        raise RuntimeError(
            "Database feature/label count mismatch."
        )

    if len(query_features) != len(
        query_labels
    ):
        raise RuntimeError(
            "Query feature/label count mismatch."
        )

    # --------------------------------------------------------
    # L2 normalization checks
    # --------------------------------------------------------

    database_norms = np.linalg.norm(
        database_features,
        axis=1,
    )

    query_norms = np.linalg.norm(
        query_features,
        axis=1,
    )

    if not np.allclose(
        database_norms,
        1.0,
        atol=1e-4,
    ):
        raise RuntimeError(
            "Database features are not properly "
            "L2-normalized."
        )

    if not np.allclose(
        query_norms,
        1.0,
        atol=1e-4,
    ):
        raise RuntimeError(
            "Query features are not properly "
            "L2-normalized."
        )

    print(
        "\nL2 normalization check passed."
    )

    # --------------------------------------------------------
    # Evaluate KNN
    # --------------------------------------------------------

    results = evaluate_knn(
        database_features=database_features,
        database_labels=database_labels,
        query_features=query_features,
        query_labels=query_labels,
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    output = {
        "model_name": MODEL_NAME,
        "model_path": MODEL_PATH,
        "protocol_path": PROTOCOL_PATH,
        "database_size": int(
            len(database_features)
        ),
        "query_size": int(
            len(query_features)
        ),
        "feature_dim": int(
            database_features.shape[1]
        ),
        "batch_size": int(
            BATCH_SIZE
        ),
        "num_workers": int(
            NUM_WORKERS
        ),
        "temperature": float(
            TEMPERATURE
        ),
        "normalization": "L2",
        "faiss_index": "IndexFlatIP",
        "distance": "cosine_similarity",
        "precision": "FP32",
        "feature_type": "pooler_output",
        "feature_flops_per_image": 123107377152,
        "feature_extraction_flops_per_image": 123107377152,
        "feature_extraction_flops": 6155368857600000,
        "feature_extraction_tflops": 6155.3688576,
        "flops_convention": (
            "1 MAC = 2 FLOPs; ViT-L/16 at 224px, tokens=197, hidden=1024, layers=24; feature extraction counted once for 45k database + 5k query"
        ),
        "results": results,
    }

    save_json(
        output,
        RESULT_PATH,
    )

    print("\n" + "=" * 80)
    print("Finished")
    print("=" * 80)

    print(
        "Results saved to:"
    )

    print(
        RESULT_PATH
    )


if False:  # entry point is defined below
    _single_pool_main()
def main():
    """Extract the 45k pool once, then evaluate cached 5/10/20/45-shot slices."""
    _single_pool_main()
    from vision_encoder_eval.workers.knn.different_shot.multishot_protocol import run_multishot_from_cache
    return run_multishot_from_cache(CACHE_ROOT, os.path.dirname(RESULT_PATH), "dinov3")


if __name__ == "__main__":
    main()

