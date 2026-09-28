import os
import sys
import json
import time
import random
import warnings
import traceback

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F
import faiss

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets


# ============================================================
# 1. Global settings
# ============================================================

# ------------------------------------------------------------
# PE repository
# ------------------------------------------------------------

REPO_DIR = (
    "/cache/models/repositories/perception_models"
)


# ------------------------------------------------------------
# ImageNet TRAINING set
#
# IMPORTANT:
# This is intentionally changed from the previous PE protocol.
# The new protocol is exactly the same protocol used by the
# SigLIP2 benchmark.
# ------------------------------------------------------------

IMAGENET_TRAIN = (
    "/workspace/root/val"
)


# ------------------------------------------------------------
# Unified 45-pool / 5-query protocol
# ------------------------------------------------------------

PROTOCOL_PATH = (
    "/cache/metaclip_knn/"
    "val_45shot_5query_seed42_protocol.json"
)


# ------------------------------------------------------------
# Cache
# ------------------------------------------------------------

CACHE_ROOT = (
    "/cache/pe_knn_45shot"
)


RESULT_FILE = os.path.join(
    CACHE_ROOT,
    "all_pe_knn_results_5_10_20_45shot.json"
)


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------

SEED = 42


# ------------------------------------------------------------
# Few-shot protocol
# ------------------------------------------------------------

TRAIN_SHOTS = [
    5,
    10,
    20,
    45,
]

TRAIN_POOL_PER_CLASS = 45
QUERY_PER_CLASS = 5


# ------------------------------------------------------------
# KNN
# ------------------------------------------------------------

K = 20

TEMPERATURE = 0.07


# ------------------------------------------------------------
# Runtime
# ------------------------------------------------------------

BATCH_SIZE = int(
    os.environ.get(
        "PE_BATCH_SIZE",
        "128"
    )
)

NUM_WORKERS = 8

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ------------------------------------------------------------
# FLOPs convention
# ------------------------------------------------------------

# 1 MAC = 2 FLOPs
FLOPS_PER_MAC = 2


# ------------------------------------------------------------
# Create cache root
# ------------------------------------------------------------

os.makedirs(
    CACHE_ROOT,
    exist_ok=True
)


# ============================================================
# 2. PE Models
# ============================================================

MODELS = {

    # --------------------------------------------------------
    # PE-Core-B16-224
    # --------------------------------------------------------

    "PE-Core-B16-224": {

        "checkpoint": (
            "/cache/models/model/"
            "PE-Core-B16-224/"
            "PE-Core-B16-224.pt"
        ),

        "input_size": 224,
        "patch_size": 16,
        "depth": 12,
        "hidden_dim": 768,
        "mlp_dim": 3072,
        "num_heads": 12,

        "has_cls_token": True,

    },


    # --------------------------------------------------------
    # PE-Core-G14-448
    # --------------------------------------------------------

    "PE-Core-G14-448": {

        "checkpoint": (
            "/cache/models/model/"
            "PE-Core-G14-448/"
            "PE-Core-G14-448.pt"
        ),

        "input_size": 448,
        "patch_size": 14,
        "depth": 50,
        "hidden_dim": 1536,
        "mlp_dim": 8960,
        "num_heads": 16,

        "has_cls_token": False,

    },


    # --------------------------------------------------------
    # PE-Lang-L14-448
    # --------------------------------------------------------

    "PE-Lang-L14-448": {

        "checkpoint": (
            "/cache/models/model/"
            "PE-Lang-L14-448/"
            "PE-Lang-L14-448.pt"
        ),

        "input_size": 448,
        "patch_size": 14,
        "depth": 24,
        "hidden_dim": 1024,
        "mlp_dim": 4096,
        "num_heads": 16,

        "has_cls_token": True,

    },

}


# ============================================================
# 3. Seed
# ============================================================

def seed_everything(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# ============================================================
# 4. PE import
# ============================================================

def import_pe():

    if REPO_DIR not in sys.path:

        sys.path.insert(
            0,
            REPO_DIR
        )

    import core.vision_encoder.pe as pe

    import core.vision_encoder.transforms as transforms

    return pe, transforms


# ============================================================
# 5. Dataset
# ============================================================

class SafeImageFolder(
    datasets.ImageFolder
):

    def __getitem__(self, index):

        path, target = (
            self.samples[index]
        )

        try:

            image = self.loader(path)

            image = image.convert("RGB")

            if self.transform is not None:

                image = self.transform(
                    image
                )

            return (
                image,
                target,
                path
            )

        except Exception as e:

            print(
                f"\n[WARNING] Failed image: "
                f"{path}"
            )

            print(
                f"Error: {e}"
            )

            return None


def safe_collate(batch):

    batch = [
        item
        for item in batch
        if item is not None
    ]

    if len(batch) == 0:

        return None

    images = torch.stack(
        [
            item[0]
            for item in batch
        ]
    )

    labels = torch.tensor(
        [
            item[1]
            for item in batch
        ],
        dtype=torch.long
    )

    paths = [
        item[2]
        for item in batch
    ]

    return (
        images,
        labels,
        paths
    )


class IndexedSubset(
    Dataset
):

    def __init__(
        self,
        dataset,
        indices
    ):

        self.dataset = dataset

        self.indices = np.asarray(
            indices,
            dtype=np.int64
        )

    def __len__(self):

        return len(
            self.indices
        )

    def __getitem__(
        self,
        index
    ):

        return self.dataset[
            int(
                self.indices[index]
            )
        ]


# ============================================================
# 6. Build / load EXACT unified protocol
# ============================================================

def build_or_load_protocol():

    os.makedirs(
        os.path.dirname(
            PROTOCOL_PATH
        ),
        exist_ok=True
    )


    # --------------------------------------------------------
    # Existing protocol
    # --------------------------------------------------------

    if os.path.isfile(
        PROTOCOL_PATH
    ):

        print(
            "\nLoading fixed unified protocol:"
        )

        print(
            PROTOCOL_PATH
        )

        with open(
            PROTOCOL_PATH,
            "r",
            encoding="utf-8"
        ) as f:

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


        # ----------------------------------------------------
        # Verify mandatory protocol fields
        # ----------------------------------------------------

        required = {

            "seed":
                SEED,

            "train_pool_per_class":
                TRAIN_POOL_PER_CLASS,

            "query_per_class":
                QUERY_PER_CLASS,

            "train_shots":
                TRAIN_SHOTS,

        }


        for key, expected in (
            required.items()
        ):

            actual = protocol.get(
                key
            )

            if actual != expected:

                raise RuntimeError(
                    f"Protocol field mismatch: "
                    f"{key} = {actual}, "
                    f"expected {expected}"
                )


        # The canonical protocol stores a descriptive dataset name rather than
        # a filesystem path. Index/count validation below is authoritative.


        print(
            f"Train pool/class: "
            f"{protocol['train_pool_per_class']}"
        )

        print(
            f"Query/class: "
            f"{protocol['query_per_class']}"
        )

        print(
            f"Train shots: "
            f"{protocol['train_shots']}"
        )

        print(
            f"Train pool size: "
            f"{protocol['train_pool_size']:,}"
        )

        print(
            f"Query size: "
            f"{protocol['query_size']:,}"
        )


        return protocol


    # --------------------------------------------------------
    # Create new protocol
    # --------------------------------------------------------

    print(
        "\nProtocol does not exist."
    )

    print(
        "Creating unified 45-pool / 5-query protocol..."
    )


    dataset = datasets.ImageFolder(
        IMAGENET_TRAIN,
        transform=None
    )


    if len(dataset.classes) != 1000:

        raise RuntimeError(
            "Expected 1000 ImageNet classes, "
            f"got {len(dataset.classes)}"
        )


    # --------------------------------------------------------
    # Group image indices by class
    # --------------------------------------------------------

    by_class = {
        c: []
        for c in range(1000)
    }


    for idx, (
        _,
        label
    ) in enumerate(
        dataset.samples
    ):

        by_class[label].append(
            idx
        )


    rng = random.Random(
        SEED
    )


    train_pool_by_class = {}

    query_by_class = {}


    train_indices_by_shot = {

        str(shot): []

        for shot in TRAIN_SHOTS

    }


    query_indices = []


    # --------------------------------------------------------
    # IMPORTANT:
    #
    # For every class:
    #
    # shuffle
    # first 45 -> train pool
    # next 5  -> query
    #
    # The first 5 / 10 / 20 / 45 of the SAME pool are used.
    # --------------------------------------------------------

    for c in range(1000):

        ids = by_class[c].copy()

        need = (
            TRAIN_POOL_PER_CLASS
            +
            QUERY_PER_CLASS
        )


        if len(ids) < need:

            raise RuntimeError(
                f"Class {c} contains "
                f"{len(ids)} images, "
                f"but {need} are required."
            )


        rng.shuffle(ids)


        pool = ids[
            :TRAIN_POOL_PER_CLASS
        ]

        query = ids[
            TRAIN_POOL_PER_CLASS:
            need
        ]


        train_pool_by_class[
            str(c)
        ] = pool


        query_by_class[
            str(c)
        ] = query


        query_indices.extend(
            query
        )


        for shot in TRAIN_SHOTS:

            train_indices_by_shot[
                str(shot)
            ].extend(
                pool[:shot]
            )


    # --------------------------------------------------------
    # Construct protocol
    # --------------------------------------------------------

    protocol = {

        "seed":
            SEED,

        "dataset":
            IMAGENET_TRAIN,

        "num_classes":
            1000,

        "train_pool_per_class":
            TRAIN_POOL_PER_CLASS,

        "query_per_class":
            QUERY_PER_CLASS,

        "train_shots":
            TRAIN_SHOTS,

        "train_pool_size":
            1000
            * TRAIN_POOL_PER_CLASS,

        "query_size":
            1000
            * QUERY_PER_CLASS,

        "train_pool_by_class":
            train_pool_by_class,

        "query_by_class":
            query_by_class,

        "train_indices_by_shot":
            train_indices_by_shot,

        "query_indices":
            query_indices,

    }


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    with open(
        PROTOCOL_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            protocol,
            f,
            indent=2
        )


    print(
        f"Created protocol:\n"
        f"{PROTOCOL_PATH}"
    )


    return protocol


# ============================================================
# 7. Load PE model
# ============================================================

def load_model(
    model_name,
    checkpoint_path,
    pe
):

    print(
        "\n"
        + "=" * 110
    )

    print(
        f"Loading model: "
        f"{model_name}"
    )

    print(
        "=" * 110
    )


    if not os.path.exists(
        checkpoint_path
    ):

        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{checkpoint_path}"
        )


    print(
        f"Checkpoint:\n"
        f"{checkpoint_path}"
    )


    # --------------------------------------------------------
    # Keep EXACT original PE loading method
    # --------------------------------------------------------

    model = (
        pe.VisionTransformer.from_config(
            model_name,
            pretrained=True,
            checkpoint_path=checkpoint_path
        )
    )


    model = model.to(
        DEVICE
    )


    model.eval()


    print(
        f"Model image size: "
        f"{model.image_size}"
    )


    return model


# ============================================================
# 8. Encode PE images
# ============================================================

@torch.inference_mode()
def encode_images(
    model,
    images
):

    images = images.to(
        DEVICE,
        non_blocking=True
    )


    use_amp = (
        DEVICE == "cuda"
    )


    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=use_amp
    ):

        features = model(
            images
        )


    if not isinstance(
        features,
        torch.Tensor
    ):

        raise RuntimeError(
            "Model output is not Tensor: "
            f"{type(features)}"
        )


    # --------------------------------------------------------
    # [B, D]
    # --------------------------------------------------------

    if features.ndim == 2:

        pass


    # --------------------------------------------------------
    # [B, N, D]
    #
    # Original PE protocol:
    # CLS token
    # --------------------------------------------------------

    elif features.ndim == 3:

        features = (
            features[:, 0, :]
        )


    else:

        raise RuntimeError(
            "Unexpected output shape: "
            f"{features.shape}"
        )


    if features.ndim != 2:

        raise RuntimeError(
            "Expected [B, D], got: "
            f"{features.shape}"
        )


    # --------------------------------------------------------
    # L2 normalization
    #
    # Same as SigLIP2 protocol.
    # --------------------------------------------------------

    features = F.normalize(
        features.float(),
        dim=-1
    )


    return features


# ============================================================
# 9. Feature extraction
# ============================================================

@torch.inference_mode()
def extract_features(
    model,
    dataset,
    indices,
    transform,
    split_name
):

    dataset.transform = transform


    subset = IndexedSubset(
        dataset,
        indices
    )


    loader = DataLoader(

        subset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=True,

        persistent_workers=(
            NUM_WORKERS > 0
        ),

        collate_fn=safe_collate,

    )


    features_list = []

    labels_list = []


    total = len(
        subset
    )

    processed = 0


    start = time.time()


    print(
        "\n"
        + "=" * 110
    )

    print(
        f"Extracting {split_name}"
    )

    print(
        "=" * 110
    )

    print(
        f"Total images: "
        f"{total:,}"
    )


    for batch in loader:

        if batch is None:

            continue


        images, labels, paths = (
            batch
        )


        features = encode_images(
            model,
            images
        )


        features = (
            features
            .detach()
            .float()
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )


        features_list.append(
            features
        )


        labels_list.append(
            labels.numpy()
        )


        processed += len(
            labels
        )


        elapsed = (
            time.time()
            - start
        )


        speed = (
            processed
            /
            max(
                elapsed,
                1e-6
            )
        )


        eta = (

            (
                total
                - processed
            )
            /
            speed

            if speed > 0
            else 0

        )


        print(
            f"\r{split_name}: "
            f"{processed:,}/"
            f"{total:,} "
            f"({speed:.1f} img/s, "
            f"ETA {eta / 60:.1f} min)",
            end="",
            flush=True
        )


    print()


    if len(
        features_list
    ) == 0:

        raise RuntimeError(
            f"No valid images in "
            f"{split_name}"
        )


    features = np.concatenate(
        features_list,
        axis=0
    )


    labels = np.concatenate(
        labels_list,
        axis=0
    )


    # --------------------------------------------------------
    # Final L2 normalization
    # --------------------------------------------------------

    norms = np.linalg.norm(
        features,
        axis=1,
        keepdims=True
    )


    features = (
        features
        /
        np.maximum(
            norms,
            1e-12
        )
    )


    print(
        f"{split_name} feature shape: "
        f"{features.shape}"
    )


    print(
        f"{split_name} label shape: "
        f"{labels.shape}"
    )


    return (
        features,
        labels
    )


# ============================================================
# 10. Similarity-weighted KNN
# ============================================================

def run_knn(
    train_features,
    train_labels,
    query_features,
    query_labels,
    k=20
):

    train_features = np.ascontiguousarray(
        train_features,
        dtype=np.float32
    )


    query_features = np.ascontiguousarray(
        query_features,
        dtype=np.float32
    )


    feature_dim = (
        train_features.shape[1]
    )


    index = faiss.IndexFlatIP(
        feature_dim
    )


    index.add(
        train_features
    )


    # --------------------------------------------------------
    # Exhaustive search
    # --------------------------------------------------------

    start = time.time()


    similarities, neighbor_indices = (
        index.search(
            query_features,
            k
        )
    )


    search_time = (
        time.time()
        - start
    )


    neighbor_labels = (
        train_labels[
            neighbor_indices
        ]
    )


    # --------------------------------------------------------
    # Similarity-weighted voting
    #
    # EXACT SAME as SigLIP2 protocol:
    #
    # exp((sim - max(sim)) / temperature)
    # --------------------------------------------------------

    predictions = np.empty(
        len(query_labels),
        dtype=np.int64
    )


    for i in range(
        len(query_labels)
    ):

        labels_i = (
            neighbor_labels[i]
        )


        similarities_i = (
            similarities[i]
        )


        weights = np.exp(
            (
                similarities_i
                -
                similarities_i.max()
            )
            /
            TEMPERATURE
        )


        class_scores = {}


        for label, weight in zip(
            labels_i,
            weights
        ):

            label = int(
                label
            )


            class_scores[label] = (
                class_scores.get(
                    label,
                    0.0
                )
                +
                float(weight)
            )


        predictions[i] = max(
            class_scores.items(),
            key=lambda x: x[1]
        )[0]


    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    accuracy = float(
        (
            predictions
            ==
            query_labels
        ).mean()
    )


    # --------------------------------------------------------
    # Same-class neighbor rate
    # --------------------------------------------------------

    same_class = float(
        (
            neighbor_labels
            ==
            query_labels[:, None]
        ).mean()
    )


    return (
        accuracy,
        same_class,
        search_time
    )


# ============================================================
# 11. FLOPs
# ============================================================

def calculate_vit_flops(
    image_size,
    patch_size,
    depth,
    width,
    heads,
    mlp_dim,
    embed_dim
):

    if (
        image_size
        %
        patch_size
        != 0
    ):

        raise ValueError(
            f"Image size {image_size} "
            f"is not divisible by "
            f"patch size {patch_size}"
        )


    # --------------------------------------------------------
    # Number of patches
    #
    # PE Core G14 does not have CLS according to the model
    # configuration, but the FLOPs convention here follows
    # the same ViT accounting used in the SigLIP2 script:
    # +1 token.
    # --------------------------------------------------------

    n = (
        image_size
        //
        patch_size
    ) ** 2 + 1


    # --------------------------------------------------------
    # Patch embedding
    # --------------------------------------------------------

    flops = (

        FLOPS_PER_MAC

        *
        (
            image_size
            //
            patch_size
        ) ** 2

        *
        (
            3
            *
            patch_size
            *
            patch_size
        )

        *
        width

    )


    # --------------------------------------------------------
    # Transformer blocks
    # --------------------------------------------------------

    head_dim = (
        width
        //
        heads
    )


    for _ in range(
        depth
    ):

        # QKV

        flops += (

            FLOPS_PER_MAC
            *
            n
            *
            width
            *
            (
                3
                *
                width
            )

        )


        # QK^T + AV

        flops += (

            FLOPS_PER_MAC
            *
            heads
            *
            n
            *
            n
            *
            head_dim
            *
            2

        )


        # Attention output projection

        flops += (

            FLOPS_PER_MAC
            *
            n
            *
            width
            *
            width

        )


        # MLP

        flops += (

            FLOPS_PER_MAC
            *
            n
            *
            width
            *
            mlp_dim

        )


        flops += (

            FLOPS_PER_MAC
            *
            n
            *
            mlp_dim
            *
            width

        )


    # --------------------------------------------------------
    # Projection head
    # --------------------------------------------------------

    flops += (

        FLOPS_PER_MAC
        *
        width
        *
        embed_dim

    )


    return int(
        flops
    )


def calculate_total_flops(
    per_image,
    n_train,
    n_query,
    dim
):

    # --------------------------------------------------------
    # Feature extraction
    # --------------------------------------------------------

    feature_flops = (

        per_image
        *
        (
            n_train
            +
            n_query
        )

    )


    # --------------------------------------------------------
    # Exhaustive FAISS IndexFlatIP
    #
    # 1 dot product:
    # D MACs = 2D FLOPs
    # --------------------------------------------------------

    search_flops = (

        FLOPS_PER_MAC
        *
        n_train
        *
        n_query
        *
        dim

    )


    # --------------------------------------------------------
    # Voting
    #
    # Keep EXACT same accounting convention as SigLIP2 code.
    # --------------------------------------------------------

    voting_flops = (
        n_query
        *
        K
        *
        5
    )


    total = (
        feature_flops
        +
        search_flops
        +
        voting_flops
    )


    return {

        "feature_flops":
            int(
                feature_flops
            ),

        "search_flops":
            int(
                search_flops
            ),

        "voting_flops":
            int(
                voting_flops
            ),

        "total_flops":
            int(
                total
            ),

        "total_tflops":
            total
            /
            1e12,

        "total_pflops":
            total
            /
            1e15,

    }


# ============================================================
# 12. Cache utilities
# ============================================================

def get_cache_paths(
    model_name
):

    model_cache = os.path.join(
        CACHE_ROOT,
        model_name
    )


    os.makedirs(
        model_cache,
        exist_ok=True
    )


    return {

        "query_features":
            os.path.join(
                model_cache,
                "query_45pool5query_seed42_features.npy"
            ),

        "query_labels":
            os.path.join(
                model_cache,
                "query_45pool5query_seed42_labels.npy"
            ),

        "query_indices":
            os.path.join(
                model_cache,
                "query_45pool5query_seed42_indices.npy"
            ),

    }


def get_train_cache_paths(
    model_name,
    shot
):

    model_cache = os.path.join(
        CACHE_ROOT,
        model_name
    )


    os.makedirs(
        model_cache,
        exist_ok=True
    )


    return {

        "features":
            os.path.join(
                model_cache,
                f"train_{shot}shot_features.npy"
            ),

        "labels":
            os.path.join(
                model_cache,
                f"train_{shot}shot_labels.npy"
            ),

        "indices":
            os.path.join(
                model_cache,
                f"train_{shot}shot_indices.npy"
            ),

    }


# ============================================================
# 13. Load or extract query
# ============================================================

def load_or_extract_query(
    model,
    dataset,
    indices,
    transform,
    model_name
):

    paths = get_cache_paths(
        model_name
    )


    expected = np.asarray(
        indices,
        dtype=np.int64
    )


    if all(
        os.path.isfile(
            p
        )
        for p in paths.values()
    ):

        cached_indices = np.load(
            paths["query_indices"]
        )


        if np.array_equal(
            cached_indices,
            expected
        ):

            print(
                "\nLoading query cache:"
            )

            print(
                paths["query_features"]
            )


            features = np.load(
                paths["query_features"],
                mmap_mode="r"
            )


            labels = np.load(
                paths["query_labels"]
            )


            return (
                features,
                labels,
                cached_indices
            )


        print(
            "Query cache index mismatch."
        )

        print(
            "Extracting query again."
        )


    features, labels = (
        extract_features(
            model,
            dataset,
            expected,
            transform,
            "Query"
        )
    )


    np.save(
        paths["query_features"],
        features
    )


    np.save(
        paths["query_labels"],
        labels
    )


    np.save(
        paths["query_indices"],
        expected
    )


    return (
        features,
        labels,
        expected
    )


# ============================================================
# 14. Load or extract train
# ============================================================

def load_or_extract_train(
    model,
    dataset,
    indices,
    transform,
    model_name,
    shot
):

    paths = get_train_cache_paths(
        model_name,
        shot
    )


    expected = np.asarray(
        indices,
        dtype=np.int64
    )


    if all(
        os.path.isfile(
            p
        )
        for p in paths.values()
    ):

        cached_indices = np.load(
            paths["indices"]
        )


        if np.array_equal(
            cached_indices,
            expected
        ):

            print(
                f"\nLoading "
                f"{shot}-shot cache:"
            )

            print(
                paths["features"]
            )


            features = np.load(
                paths["features"],
                mmap_mode="r"
            )


            labels = np.load(
                paths["labels"]
            )


            return (
                features,
                labels,
                cached_indices
            )


        print(
            f"{shot}-shot cache index mismatch."
        )

        print(
            "Extracting again."
        )


    features, labels = (
        extract_features(
            model,
            dataset,
            expected,
            transform,
            f"Train {shot}-shot"
        )
    )


    np.save(
        paths["features"],
        features
    )


    np.save(
        paths["labels"],
        labels
    )


    np.save(
        paths["indices"],
        expected
    )


    return (
        features,
        labels,
        expected
    )


# ============================================================
# 15. Process one PE model
# ============================================================

def process_model(
    model_name,
    config,
    protocol,
    pe,
    transforms
):

    print(
        "\n\n"
        + "#"
        * 110
    )

    print(
        f"# Processing "
        f"{model_name}"
    )

    print(
        "#"
        * 110
    )


    model = None


    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------

    checkpoint = (
        config["checkpoint"]
    )


    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model(
        model_name,
        checkpoint,
        pe
    )


    # --------------------------------------------------------
    # PE preprocessing
    #
    # Keep EXACT original PE preprocessing.
    # --------------------------------------------------------

    transform = (
        transforms.get_image_transform(
            model.image_size
        )
    )


    # --------------------------------------------------------
    # Base ImageNet dataset
    # --------------------------------------------------------

    base_dataset = SafeImageFolder(
        IMAGENET_TRAIN,
        transform=transform
    )


    # --------------------------------------------------------
    # Query
    # --------------------------------------------------------

    query_indices = np.asarray(
        protocol["query_indices"],
        dtype=np.int64
    )


    (
        query_features,
        query_labels,
        query_cached_ids
    ) = load_or_extract_query(

        model=model,

        dataset=base_dataset,

        indices=query_indices,

        transform=transform,

        model_name=model_name

    )


    # --------------------------------------------------------
    # Verify query indices
    # --------------------------------------------------------

    if not np.array_equal(
        query_cached_ids,
        query_indices
    ):

        raise RuntimeError(
            "Query cache verification failed."
        )


    # --------------------------------------------------------
    # Query size
    # --------------------------------------------------------

    expected_query_size = (
        1000
        *
        QUERY_PER_CLASS
    )


    if len(
        query_features
    ) != expected_query_size:

        raise RuntimeError(
            "Unexpected query feature count: "
            f"{len(query_features)} "
            f"vs expected "
            f"{expected_query_size}"
        )


    # --------------------------------------------------------
    # Feature dimension
    # --------------------------------------------------------

    feature_dim = int(
        query_features.shape[1]
    )


    # --------------------------------------------------------
    # Model parameter count
    # --------------------------------------------------------

    num_params = sum(
        p.numel()
        for p in model.parameters()
    )


    print(
        f"\nFeature dim: "
        f"{feature_dim}"
    )


    print(
        f"Parameters: "
        f"{num_params / 1e6:.2f}M"
    )


    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    model_results = []

    # Extract/load the ordered 45-shot database pool exactly once.
    full_train_indices = np.asarray(
        protocol["train_indices_by_shot"]["45"], dtype=np.int64
    )
    full_train_features, full_train_labels, full_train_cached_ids = load_or_extract_train(
        model=model,
        dataset=base_dataset,
        indices=full_train_indices,
        transform=transform,
        model_name=model_name,
        shot=45,
    )
    if not np.array_equal(full_train_cached_ids, full_train_indices):
        raise RuntimeError("45-shot train cache verification failed.")
    full_pos = {int(idx): pos for pos, idx in enumerate(full_train_cached_ids)}


    # ========================================================
    # Different few-shot settings
    # ========================================================

    for shot in TRAIN_SHOTS:

        print(
            "\n"
            + "=" * 110
        )

        print(
            f"{model_name} | "
            f"{shot}-shot"
        )

        print(
            "=" * 110
        )


        # ----------------------------------------------------
        # Train indices
        # ----------------------------------------------------

        train_indices = np.asarray(

            protocol[
                "train_indices_by_shot"
            ][
                str(shot)
            ],

            dtype=np.int64

        )


        expected_train_size = (
            1000
            *
            shot
        )


        if len(
            train_indices
        ) != expected_train_size:

            raise RuntimeError(

                f"{shot}-shot train size "
                f"mismatch: "

                f"{len(train_indices)} "
                f"vs "

                f"{expected_train_size}"

            )


        # ----------------------------------------------------
        # Train features
        # ----------------------------------------------------

        take = np.asarray([full_pos[int(idx)] for idx in train_indices], dtype=np.int64)
        train_features = full_train_features[take]
        train_labels = full_train_labels[take]
        train_cached_ids = full_train_cached_ids[take]


        # ----------------------------------------------------
        # Verify cache
        # ----------------------------------------------------

        if not np.array_equal(
            train_cached_ids,
            train_indices
        ):

            raise RuntimeError(
                f"Train {shot}-shot "
                "cache verification failed."
            )


        # ----------------------------------------------------
        # Verify train/query no overlap
        # ----------------------------------------------------

        overlap = np.intersect1d(

            train_cached_ids,

            query_cached_ids

        )


        if len(
            overlap
        ):

            raise RuntimeError(

                f"Train/query overlap "
                f"detected: "
                f"{len(overlap)}"

            )


        # ----------------------------------------------------
        # Verify feature dimensions
        # ----------------------------------------------------

        if (
            train_features.shape[1]
            !=
            feature_dim
        ):

            raise RuntimeError(

                f"Feature dimension "
                f"mismatch: "

                f"{train_features.shape[1]} "
                f"vs "

                f"{feature_dim}"

            )


        # ----------------------------------------------------
        # KNN
        # ----------------------------------------------------

        (
            accuracy,
            same_class_rate,
            search_time
        ) = run_knn(

            train_features,

            train_labels,

            query_features,

            query_labels,

            K

        )


        # ----------------------------------------------------
        # FLOPs
        # ----------------------------------------------------

        per_image_flops = (
            calculate_vit_flops(

                image_size=
                    config["input_size"],

                patch_size=
                    config["patch_size"],

                depth=
                    config["depth"],

                width=
                    config["hidden_dim"],

                heads=
                    config["num_heads"],

                mlp_dim=
                    config["mlp_dim"],

                embed_dim=
                    feature_dim

            )
        )


        flops = (
            calculate_total_flops(

                per_image=
                    per_image_flops,

                n_train=
                    len(train_features),

                n_query=
                    len(query_features),

                dim=
                    feature_dim

            )
        )


        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        result = {

            "model":
                model_name,

            "architecture":
                model_name,

            "checkpoint":
                checkpoint,


            "input_size":
                config["input_size"],

            "patch_size":
                config["patch_size"],

            "depth":
                config["depth"],

            "hidden_dim":
                config["hidden_dim"],

            "mlp_dim":
                config["mlp_dim"],

            "num_heads":
                config["num_heads"],


            "parameters":
                num_params,


            "feature_dim":
                feature_dim,


            # ------------------------------------------------
            # Protocol
            # ------------------------------------------------

            "protocol_path":
                PROTOCOL_PATH,

            "seed":
                SEED,

            "train_pool_per_class":
                TRAIN_POOL_PER_CLASS,

            "query_per_class":
                QUERY_PER_CLASS,

            "train_shots":
                shot,

            "database_size":
                len(
                    train_features
                ),

            "query_size":
                len(
                    query_features
                ),


            # ------------------------------------------------
            # KNN
            # ------------------------------------------------

            "K":
                K,

            "temperature":
                TEMPERATURE,

            "accuracy":
                accuracy
                *
                100.0,

            "same_class_neighbor_rate":
                same_class_rate,

            "faiss":
                "IndexFlatIP",

            "normalization":
                "L2",

            "similarity_weighted":
                True,


            # ------------------------------------------------
            # Timing
            # ------------------------------------------------

            "faiss_search_seconds":
                search_time,


            # ------------------------------------------------
            # FLOPs
            # ------------------------------------------------

            "feature_flops_per_image":
                per_image_flops,

            "feature_gflops_per_image":
                per_image_flops
                /
                1e9,


            "feature_flops":
                flops[
                    "feature_flops"
                ],

            "search_flops":
                flops[
                    "search_flops"
                ],

            "voting_flops":
                flops[
                    "voting_flops"
                ],

            "total_flops":
                flops[
                    "total_flops"
                ],

            "total_tflops":
                flops[
                    "total_tflops"
                ],

            "total_pflops":
                flops[
                    "total_pflops"
                ],


            # ------------------------------------------------
            # Cache
            # ------------------------------------------------

            "train_indices_cache":
                f"train_{shot}shot",

            "query_indices_cache":
                "query_45pool5query_seed42",

        }


        model_results.append(
            result
        )


        # ----------------------------------------------------
        # Save per-shot result
        # ----------------------------------------------------

        model_cache = os.path.join(
            CACHE_ROOT,
            model_name
        )


        os.makedirs(
            model_cache,
            exist_ok=True
        )


        shot_result_file = os.path.join(

            model_cache,

            f"train_{shot}shot_results.json"

        )


        with open(
            shot_result_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                result,
                f,
                indent=2,
                ensure_ascii=False
            )


        # ----------------------------------------------------
        # Console
        # ----------------------------------------------------

        print(
            "\n"
            + "-" * 110
        )


        print(
            f"Model: "
            f"{model_name}"
        )


        print(
            f"Shot: "
            f"{shot}"
        )


        print(
            f"Database: "
            f"{len(train_features):,}"
        )


        print(
            f"Query: "
            f"{len(query_features):,}"
        )


        print(
            f"Feature dim: "
            f"{feature_dim}"
        )


        print(
            f"K: "
            f"{K}"
        )


        print(
            f"Top-1: "
            f"{accuracy * 100:.4f}%"
        )


        print(
            f"Same-class neighbor rate: "
            f"{same_class_rate:.6f}"
        )


        print(
            f"Feature GFLOPs/image: "
            f"{per_image_flops / 1e9:.3f}"
        )


        print(
            f"KNN search GFLOPs: "
            f"{flops['search_flops'] / 1e9:.3f}"
        )


        print(
            f"Total KNN TFLOPs: "
            f"{flops['total_tflops']:.6f}"
        )


        print(
            f"Total KNN PFLOPs: "
            f"{flops['total_pflops']:.6f}"
        )


        print(
            f"FAISS search time: "
            f"{search_time / 60:.2f} min"
        )


        print(
            f"Saved: "
            f"{shot_result_file}"
        )


    # --------------------------------------------------------
    # Free model
    # --------------------------------------------------------

    del model


    if torch.cuda.is_available():

        torch.cuda.empty_cache()


    return model_results


# ============================================================
# 16. Main
# ============================================================

def main():

    seed_everything(
        SEED
    )


    print(
        "=" * 110
    )

    print(
        "PE ImageNet KNN Benchmark"
    )

    print(
        "Unified 45-pool / 5-query Protocol"
    )

    print(
        "=" * 110
    )


    print(
        f"Device: "
        f"{DEVICE}"
    )


    print(
        f"ImageNet train: "
        f"{IMAGENET_TRAIN}"
    )


    print(
        f"Protocol: "
        f"{PROTOCOL_PATH}"
    )


    print(
        f"Cache: "
        f"{CACHE_ROOT}"
    )


    print(
        f"Train shots: "
        f"{TRAIN_SHOTS}"
    )


    print(
        f"Train pool/class: "
        f"{TRAIN_POOL_PER_CLASS}"
    )


    print(
        f"Query/class: "
        f"{QUERY_PER_CLASS}"
    )


    print(
        f"K: "
        f"{K}"
    )


    print(
        f"Temperature: "
        f"{TEMPERATURE}"
    )


    print(
        f"Batch size: "
        f"{BATCH_SIZE}"
    )


    print(
        f"Workers: "
        f"{NUM_WORKERS}"
    )


    print(
        "=" * 110
    )


    # --------------------------------------------------------
    # Check ImageNet
    # --------------------------------------------------------

    if not os.path.isdir(
        IMAGENET_TRAIN
    ):

        raise FileNotFoundError(
            IMAGENET_TRAIN
        )


    # --------------------------------------------------------
    # Build/load exact shared protocol
    # --------------------------------------------------------

    protocol = (
        build_or_load_protocol()
    )


    # --------------------------------------------------------
    # Verify protocol sizes
    # --------------------------------------------------------

    if len(
        protocol["query_indices"]
    ) != (
        1000
        *
        QUERY_PER_CLASS
    ):

        raise RuntimeError(
            "Query protocol size is incorrect."
        )


    for shot in TRAIN_SHOTS:

        expected = (
            1000
            *
            shot
        )

        actual = len(
            protocol[
                "train_indices_by_shot"
            ][
                str(shot)
            ]
        )

        if actual != expected:

            raise RuntimeError(

                f"{shot}-shot protocol "
                f"size mismatch: "

                f"{actual} vs {expected}"

            )


    # --------------------------------------------------------
    # Verify all train/query overlap
    # --------------------------------------------------------

    query_ids = np.asarray(
        protocol[
            "query_indices"
        ],
        dtype=np.int64
    )


    for shot in TRAIN_SHOTS:

        train_ids = np.asarray(

            protocol[
                "train_indices_by_shot"
            ][
                str(shot)
            ],

            dtype=np.int64

        )


        overlap = np.intersect1d(
            train_ids,
            query_ids
        )


        if len(
            overlap
        ):

            raise RuntimeError(

                f"Protocol has "
                f"train/query overlap "
                f"for {shot}-shot: "
                f"{len(overlap)}"

            )


    # --------------------------------------------------------
    # Import PE
    # --------------------------------------------------------

    pe, transforms = (
        import_pe()
    )


    all_results = []


    # --------------------------------------------------------
    # Process all PE models
    # --------------------------------------------------------

    for model_name, config in (
        MODELS.items()
    ):

        model = None

        try:

            results = process_model(

                model_name,

                config,

                protocol,

                pe,

                transforms

            )


            all_results.extend(
                results
            )


        except Exception as e:

            print(
                "\n"
                + "=" * 110
            )


            print(
                f"[ERROR] "
                f"{model_name}"
            )


            print(
                "=" * 110
            )


            print(
                repr(e)
            )


            traceback.print_exc()


            if model is not None:

                del model


            if torch.cuda.is_available():

                torch.cuda.empty_cache()


    # --------------------------------------------------------
    # Save all results
    # --------------------------------------------------------

    output = {

        "protocol": protocol,

        "results": all_results,

    }


    with open(
        RESULT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False
        )


    # ========================================================
    # Final table
    # ========================================================

    print(
        "\n\n"
        + "=" * 130
    )


    print(
        "FINAL RESULTS"
    )


    print(
        "=" * 130
    )


    print(

        f"{'Model':<28}"

        f"{'Shot':>6}"

        f"{'Train':>9}"

        f"{'Query':>8}"

        f"{'Dim':>7}"

        f"{'Top1':>10}"

        f"{'GFLOPs/img':>14}"

        f"{'PFLOPs':>14}"

    )


    print(
        "-" * 130
    )


    for result in all_results:

        print(

            f"{result['model']:<28}"

            f"{result['train_shots']:>6}"

            f"{result['database_size']:>9}"

            f"{result['query_size']:>8}"

            f"{result['feature_dim']:>7}"

            f"{result['accuracy']:>9.3f}%"

            f"{result['feature_gflops_per_image']:>14.3f}"

            f"{result['total_pflops']:>14.6f}"

        )


    print(
        "=" * 130
    )


    print(
        "\nAll results saved to:"
    )


    print(
        RESULT_FILE
    )


    print(
        "\nProtocol:"
    )


    print(
        PROTOCOL_PATH
    )


    print(
        "\n"
        "Protocol summary:"
    )


    print(
        "  1000 classes"
    )


    print(
        "  45 train images/class in fixed pool"
    )


    print(
        "  5 query images/class"
    )


    print(
        "  5-shot  = 5,000 database images"
    )


    print(
        "  10-shot = 10,000 database images"
    )


    print(
        "  20-shot = 20,000 database images"
    )


    print(
        "  45-shot = 45,000 database images"
    )


    print(
        "  Query    = 5,000 images"
    )


    print(
        "  K        = 20"
    )


    print(
        "  Seed     = 42"
    )


    print(
        "  FAISS    = IndexFlatIP"
    )


    print(
        "  Feature  = L2 normalized"
    )


    print(
        "  Weight   = exp((sim-max)/0.07)"
    )


    print(
        "=" * 110
    )


# ============================================================
# 17. Entry
# ============================================================

if __name__ == "__main__":

    main()
