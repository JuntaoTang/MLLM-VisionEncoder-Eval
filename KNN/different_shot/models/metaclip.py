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

import faiss
import open_clip


# ============================================================
# 0. GLOBAL SETTINGS
# ============================================================

SEED = 42

IMAGENET_VAL = "/workspace/root/val"

CACHE_ROOT = "/cache/metaclip_knn"

NUM_CLASSES = 1000

# ============================================================
# IMPORTANT:
# Train database sizes per class
# ============================================================

TRAIN_SHOTS = [5, 10, 20, 45]

# Maximum train pool per class.
# This MUST be >= max(TRAIN_SHOTS).
TRAIN_POOL_PER_CLASS = max(TRAIN_SHOTS)

# Fixed query images per class
QUERY_PER_CLASS = 5

# Total images needed per class
TOTAL_PROTOCOL_IMAGES_PER_CLASS = (
    TRAIN_POOL_PER_CLASS + QUERY_PER_CLASS
)

# Only evaluate K=20
K = 20

TEMPERATURE = 0.07

BATCH_SIZE = 128
NUM_WORKERS = 8

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Protocol paths
# ============================================================

PROTOCOL_NAME = (
    "val_45shot_5query_seed42_protocol.json"
)

PROTOCOL_PATH = os.path.join(
    CACHE_ROOT,
    PROTOCOL_NAME
)

# Query cache explicitly tied to this protocol.
#
# This avoids accidentally loading an old query_features.npy
# generated from a different protocol.
QUERY_CACHE_TAG = (
    "val_45shot_5query_seed42"
)


# ============================================================
# 1. MODELS
# ============================================================

MODELS = [
    {
        "name": "MetaCLIP2-H14-v1.2",
        "arch": "ViT-H-14-quickgelu",
        "pretrained": "/workspace/.cache/huggingface/hub/models--timm--vit_huge_patch14_clip_224.metaclip_altogether/snapshots/79ca86b54e97d41bd9f555bbb272bb1dcaa11f75/open_clip_model.safetensors",
    },
]


# ============================================================
# 2. UTILS
# ============================================================

def print_separator(title):

    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)


def set_seed(seed=SEED):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# ============================================================
# 3. DATASET WRAPPER
# ============================================================

class ImageNetSubset(Dataset):

    def __init__(
        self,
        base_dataset,
        indices,
        transform
    ):

        self.base_dataset = base_dataset

        self.indices = np.asarray(
            indices,
            dtype=np.int64
        )

        self.transform = transform

    def __len__(self):

        return len(self.indices)

    def __getitem__(self, i):

        idx = int(self.indices[i])

        path, label = (
            self.base_dataset.samples[idx]
        )

        image = Image.open(
            path
        ).convert("RGB")

        image = self.transform(image)

        return image, label


# ============================================================
# 4. BUILD / LOAD FIXED PROTOCOL
# ============================================================

def build_protocol(dataset):

    protocol_path = PROTOCOL_PATH

    os.makedirs(
        CACHE_ROOT,
        exist_ok=True
    )

    expected_train_pool = (
        NUM_CLASSES
        * TRAIN_POOL_PER_CLASS
    )

    expected_query = (
        NUM_CLASSES
        * QUERY_PER_CLASS
    )

    # --------------------------------------------------------
    # Load existing protocol
    # --------------------------------------------------------

    if os.path.isfile(protocol_path):

        print_separator(
            "LOADING EXISTING 45-SHOT PROTOCOL"
        )

        with open(
            protocol_path,
            "r"
        ) as f:

            protocol = json.load(f)

        # ----------------------------------------------------
        # Strict protocol validation
        # ----------------------------------------------------

        assert protocol["seed"] == SEED

        assert (
            protocol["train_pool_per_class"]
            == TRAIN_POOL_PER_CLASS
        ), (
            f"Existing protocol uses "
            f"{protocol['train_pool_per_class']} train images/class, "
            f"but current protocol requires "
            f"{TRAIN_POOL_PER_CLASS}."
        )

        assert (
            protocol["query_per_class"]
            == QUERY_PER_CLASS
        )

        assert (
            protocol["num_train_pool"]
            == expected_train_pool
        )

        assert (
            protocol["num_query"]
            == expected_query
        )

        train_pool_indices = np.asarray(
            protocol["train_pool_indices"],
            dtype=np.int64
        )

        query_indices = np.asarray(
            protocol["query_indices"],
            dtype=np.int64
        )

        assert (
            len(train_pool_indices)
            == expected_train_pool
        )

        assert (
            len(query_indices)
            == expected_query
        )

        print(
            "Protocol reused:",
            protocol_path
        )

        print(
            "Train pool/class:",
            TRAIN_POOL_PER_CLASS
        )

        print(
            "Query/class:",
            QUERY_PER_CLASS
        )

        print(
            "Train pool total:",
            len(train_pool_indices)
        )

        print(
            "Query total:",
            len(query_indices)
        )

        return (
            train_pool_indices,
            query_indices
        )

    # --------------------------------------------------------
    # Create new protocol
    # --------------------------------------------------------

    print_separator(
        "CREATING FIXED 45-SHOT PROTOCOL"
    )

    print(
        "Train pool per class:",
        TRAIN_POOL_PER_CLASS
    )

    print(
        "Query per class:",
        QUERY_PER_CLASS
    )

    print(
        "Total required per class:",
        TOTAL_PROTOCOL_IMAGES_PER_CLASS
    )

    rng = np.random.RandomState(SEED)

    class_to_indices = {
        c: []
        for c in range(NUM_CLASSES)
    }

    for idx, (_, label) in enumerate(
        dataset.samples
    ):

        class_to_indices[label].append(idx)

    train_pool_indices = []

    query_indices = []

    # --------------------------------------------------------
    # Sample each class independently
    # --------------------------------------------------------

    for class_id in range(NUM_CLASSES):

        indices = np.asarray(
            class_to_indices[class_id],
            dtype=np.int64
        )

        required = (
            TRAIN_POOL_PER_CLASS
            + QUERY_PER_CLASS
        )

        if len(indices) < required:

            raise RuntimeError(
                f"Class {class_id} "
                f"({dataset.classes[class_id]}) "
                f"contains only {len(indices)} images, "
                f"but {required} images are required."
            )

        # IMPORTANT:
        # Randomly select exactly 50 images:
        #
        #   first 45 -> training pool
        #   last 5   -> fixed query
        #
        selected = rng.choice(
            indices,
            size=required,
            replace=False
        )

        selected = selected.tolist()

        # ----------------------------------------------------
        # Train pool
        # ----------------------------------------------------

        train_pool_indices.extend(
            selected[
                :TRAIN_POOL_PER_CLASS
            ]
        )

        # ----------------------------------------------------
        # Fixed query
        # ----------------------------------------------------

        query_indices.extend(
            selected[
                TRAIN_POOL_PER_CLASS:
            ]
        )

    train_pool_indices = np.asarray(
        train_pool_indices,
        dtype=np.int64
    )

    query_indices = np.asarray(
        query_indices,
        dtype=np.int64
    )

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    assert (
        len(train_pool_indices)
        == expected_train_pool
    )

    assert (
        len(query_indices)
        == expected_query
    )

    assert (
        len(
            set(train_pool_indices.tolist())
            &
            set(query_indices.tolist())
        )
        == 0
    )

    # --------------------------------------------------------
    # Save protocol
    # --------------------------------------------------------

    protocol = {

        "seed": SEED,

        "dataset": "ImageNet-1K validation",

        "dataset_path": IMAGENET_VAL,

        "num_classes": NUM_CLASSES,

        "train_pool_per_class":
            TRAIN_POOL_PER_CLASS,

        "query_per_class":
            QUERY_PER_CLASS,

        "num_train_pool":
            expected_train_pool,

        "num_query":
            expected_query,

        "total_required_per_class":
            TOTAL_PROTOCOL_IMAGES_PER_CLASS,

        "train_shots":
            TRAIN_SHOTS,

        "train_pool_indices":
            train_pool_indices.tolist(),

        "query_indices":
            query_indices.tolist(),

    }

    with open(
        protocol_path,
        "w"
    ) as f:

        json.dump(
            protocol,
            f,
            indent=2
        )

    print(
        "New protocol saved:",
        protocol_path
    )

    return (
        train_pool_indices,
        query_indices
    )


# ============================================================
# 5. BUILD SHOT INDICES
# ============================================================

def build_shot_indices(
    train_pool_indices,
    shots
):

    if shots > TRAIN_POOL_PER_CLASS:

        raise ValueError(
            f"Requested {shots}-shot, "
            f"but train pool only contains "
            f"{TRAIN_POOL_PER_CLASS} images/class."
        )

    current_indices = []

    # IMPORTANT:
    # Do NOT hard-code 50 here.
    #
    # Each class occupies exactly
    # TRAIN_POOL_PER_CLASS positions
    # in train_pool_indices.

    for class_id in range(NUM_CLASSES):

        start = (
            class_id
            * TRAIN_POOL_PER_CLASS
        )

        end = (
            start
            + shots
        )

        current_indices.extend(
            train_pool_indices[
                start:end
            ]
        )

    current_indices = np.asarray(
        current_indices,
        dtype=np.int64
    )

    expected = (
        NUM_CLASSES
        * shots
    )

    assert len(current_indices) == expected

    return current_indices


# ============================================================
# 6. VERIFY PROTOCOL
# ============================================================

def verify_protocol(
    dataset,
    train_pool_indices,
    query_indices
):

    expected_train_pool = (
        NUM_CLASSES
        * TRAIN_POOL_PER_CLASS
    )

    expected_query = (
        NUM_CLASSES
        * QUERY_PER_CLASS
    )

    train_pool_labels = np.asarray(
        [
            dataset.samples[i][1]
            for i in train_pool_indices
        ],
        dtype=np.int64
    )

    query_labels = np.asarray(
        [
            dataset.samples[i][1]
            for i in query_indices
        ],
        dtype=np.int64
    )

    # --------------------------------------------------------
    # Size checks
    # --------------------------------------------------------

    assert (
        len(train_pool_indices)
        == expected_train_pool
    )

    assert (
        len(query_indices)
        == expected_query
    )

    # --------------------------------------------------------
    # No overlap
    # --------------------------------------------------------

    assert (
        len(
            set(train_pool_indices.tolist())
            &
            set(query_indices.tolist())
        )
        == 0
    )

    # --------------------------------------------------------
    # Per-class counts
    # --------------------------------------------------------

    train_counts = np.bincount(
        train_pool_labels,
        minlength=NUM_CLASSES
    )

    query_counts = np.bincount(
        query_labels,
        minlength=NUM_CLASSES
    )

    assert np.all(
        train_counts
        == TRAIN_POOL_PER_CLASS
    )

    assert np.all(
        query_counts
        == QUERY_PER_CLASS
    )

    print_separator(
        "PROTOCOL VERIFIED"
    )

    print(
        "Train pool images :",
        len(train_pool_indices)
    )

    print(
        "Query images      :",
        len(query_indices)
    )

    print(
        "Classes           :",
        NUM_CLASSES
    )

    print(
        "Train pool/class  :",
        TRAIN_POOL_PER_CLASS
    )

    print(
        "Query/class       :",
        QUERY_PER_CLASS
    )

    print(
        "Train/query overlap: 0"
    )

    return query_labels


# ============================================================
# 7. VISION FLOPs
# ============================================================

def get_vit_config(model):

    visual = model.visual

    if not hasattr(
        visual,
        "conv1"
    ):

        raise RuntimeError(
            "visual.conv1 not found. "
            "Unsupported visual encoder."
        )

    patch_size = (
        visual.conv1.kernel_size[0]
    )

    embed_dim = (
        visual.conv1.out_channels
    )

    image_size = visual.image_size

    if isinstance(
        image_size,
        tuple
    ):

        image_size = image_size[0]

    transformer = (
        visual.transformer
    )

    layers = len(
        transformer.resblocks
    )

    first_block = (
        transformer.resblocks[0]
    )

    mlp = first_block.mlp

    mlp_dim = (
        mlp.c_fc.out_features
    )

    heads = (
        first_block.attn.num_heads
    )

    return {

        "image_size": int(image_size),

        "patch_size": int(patch_size),

        "embed_dim": int(embed_dim),

        "layers": int(layers),

        "mlp_dim": int(mlp_dim),

        "heads": int(heads),

    }


def estimate_vit_flops(model):

    """
    Analytic ViT forward FLOPs.

    Convention:
        1 multiply-add = 2 FLOPs

    Includes:
        patch embedding
        QKV
        attention QK
        attention AV
        output projection
        MLP
        visual projection
    """

    cfg = get_vit_config(model)

    S = cfg["image_size"]

    P = cfg["patch_size"]

    D = cfg["embed_dim"]

    L = cfg["layers"]

    M = cfg["mlp_dim"]

    grid = S // P

    N = (
        grid * grid
        + 1
    )

    # Patch embedding
    patch_flops = (
        2
        * grid
        * grid
        * (P * P * 3)
        * D
    )

    # QKV
    qkv_flops = (
        2
        * N
        * D
        * (3 * D)
    )

    # Attention score Q @ K^T
    attention_score_flops = (
        2
        * N
        * N
        * D
    )

    # Attention @ V
    attention_value_flops = (
        2
        * N
        * N
        * D
    )

    # Output projection
    attention_projection_flops = (
        2
        * N
        * D
        * D
    )

    # MLP
    mlp_flops = (
        2
        * N
        * D
        * M
        +
        2
        * N
        * M
        * D
    )

    block_flops = (
        qkv_flops
        + attention_score_flops
        + attention_value_flops
        + attention_projection_flops
        + mlp_flops
    )

    transformer_flops = (
        L
        * block_flops
    )

    # Final CLIP visual projection
    projection_flops = 0

    visual_projection = getattr(
        model.visual,
        "proj",
        None
    )

    if visual_projection is not None:

        if hasattr(
            visual_projection,
            "shape"
        ):

            out_dim = (
                visual_projection.shape[-1]
            )

            projection_flops = (
                2
                * D
                * out_dim
            )

    total = (
        patch_flops
        + transformer_flops
        + projection_flops
    )

    return {

        "image_size": S,

        "patch_size": P,

        "tokens": N,

        "embed_dim": D,

        "mlp_dim": M,

        "layers": L,

        "heads": cfg["heads"],

        "vision_flops_per_image":
            int(total),

        "vision_flops_per_image_gflops":
            total / 1e9,

    }


# ============================================================
# 8. FEATURE EXTRACTION
# ============================================================

@torch.no_grad()
def extract_features(
    model,
    loader,
    total_images
):

    features = []

    start_time = time.time()

    processed = 0

    for images, labels in loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        image_features = (
            model.encode_image(images)
        )

        # FP32 + L2 normalization
        image_features = F.normalize(
            image_features.float(),
            dim=-1
        )

        features.append(
            image_features.cpu().numpy()
        )

        processed += (
            images.shape[0]
        )

        if (
            processed % 640 == 0
            or processed == total_images
        ):

            print(
                f"Processed "
                f"{processed:6d}/"
                f"{total_images}"
            )

    features = np.concatenate(
        features,
        axis=0
    ).astype(np.float32)

    elapsed = (
        time.time()
        - start_time
    )

    return (
        features,
        elapsed
    )


# ============================================================
# 9. DISTANCE-WEIGHTED KNN
# ============================================================

def weighted_knn_accuracy(
    similarities,
    neighbor_labels,
    query_labels,
    temperature
):

    logits = (
        similarities
        / temperature
    )

    logits -= logits.max(
        axis=1,
        keepdims=True
    )

    weights = np.exp(
        logits
    )

    predictions = np.empty(
        len(query_labels),
        dtype=np.int64
    )

    for i in range(
        len(query_labels)
    ):

        label_scores = {}

        for j in range(
            neighbor_labels.shape[1]
        ):

            label = int(
                neighbor_labels[i, j]
            )

            weight = float(
                weights[i, j]
            )

            label_scores[label] = (
                label_scores.get(
                    label,
                    0.0
                )
                + weight
            )

        predictions[i] = max(
            label_scores,
            key=label_scores.get
        )

    accuracy = (
        predictions
        == query_labels
    ).mean()

    return float(accuracy)


# ============================================================
# 10. PROCESS ONE MODEL
# ============================================================

def process_model(
    model_info,
    dataset,
    train_pool_indices,
    query_indices,
    query_labels
):

    name = model_info["name"]

    arch = model_info["arch"]

    pretrained = model_info["pretrained"]

    print_separator(
        f"MODEL: {name}"
    )

    print(
        "Architecture:",
        arch
    )

    print(
        "Pretrained:",
        pretrained
    )

    model_cache = os.path.join(
        CACHE_ROOT,
        name
    )

    os.makedirs(
        model_cache,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print(
        "\nLoading OpenCLIP model..."
    )

    load_start = time.time()

    model, _, preprocess = (
        open_clip.create_model_and_transforms(
            arch,
            pretrained=pretrained,
            device=DEVICE
        )
    )

    model.eval()

    load_time = (
        time.time()
        - load_start
    )

    print(
        f"Model loaded: "
        f"{load_time:.2f}s"
    )

    # --------------------------------------------------------
    # Parameters
    # --------------------------------------------------------

    num_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Parameters: "
        f"{num_params:,}"
    )

    # --------------------------------------------------------
    # FLOPs
    # --------------------------------------------------------

    print(
        "\nCalculating vision FLOPs..."
    )

    flops_info = estimate_vit_flops(
        model
    )

    vision_flops_per_image = (
        flops_info[
            "vision_flops_per_image"
        ]
    )

    print(
        f"Input resolution: "
        f"{flops_info['image_size']}"
    )

    print(
        f"Patch size: "
        f"{flops_info['patch_size']}"
    )

    print(
        f"Tokens: "
        f"{flops_info['tokens']}"
    )

    print(
        f"Hidden dimension: "
        f"{flops_info['embed_dim']}"
    )

    print(
        f"Layers: "
        f"{flops_info['layers']}"
    )

    print(
        f"Vision FLOPs/img: "
        f"{flops_info['vision_flops_per_image_gflops']:.4f} GFLOPs"
    )

    # --------------------------------------------------------
    # Query features
    #
    # IMPORTANT:
    # Use protocol-specific filenames.
    # --------------------------------------------------------

    query_feature_file = os.path.join(
        model_cache,
        f"{QUERY_CACHE_TAG}_query_features.npy"
    )

    query_label_file = os.path.join(
        model_cache,
        f"{QUERY_CACHE_TAG}_query_labels.npy"
    )

    query_index_file = os.path.join(
        model_cache,
        f"{QUERY_CACHE_TAG}_query_indices.npy"
    )

    query_dataset = ImageNetSubset(
        dataset,
        query_indices,
        preprocess
    )

    query_loader = DataLoader(
        query_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    use_cached_query = False

    if (
        os.path.isfile(query_feature_file)
        and os.path.isfile(query_label_file)
        and os.path.isfile(query_index_file)
    ):

        cached_query_features = np.load(
            query_feature_file
        )

        cached_query_labels = np.load(
            query_label_file
        )

        cached_query_indices = np.load(
            query_index_file
        )

        # Strict validation
        if (
            cached_query_features.shape[0]
            == len(query_indices)
            and
            cached_query_labels.shape[0]
            == len(query_indices)
            and
            cached_query_features.shape[1]
            > 0
            and
            np.array_equal(
                cached_query_labels,
                query_labels
            )
            and
            np.array_equal(
                cached_query_indices,
                query_indices
            )
        ):

            use_cached_query = True

            query_features = (
                cached_query_features
            )

            print_separator(
                f"{name} — LOADING QUERY FEATURES"
            )

            print(
                "Cached query features loaded."
            )

        else:

            print(
                "Existing query cache does not "
                "match current protocol."
            )

            print(
                "Re-extracting query features."
            )

    if use_cached_query:

        query_time = 0.0

    else:

        print_separator(
            f"{name} — QUERY FEATURES"
        )

        query_features, query_time = (
            extract_features(
                model,
                query_loader,
                len(query_indices)
            )
        )

        np.save(
            query_feature_file,
            query_features
        )

        np.save(
            query_label_file,
            query_labels
        )

        np.save(
            query_index_file,
            query_indices
        )

    feature_dim = (
        query_features.shape[1]
    )

    query_vision_flops = (
        vision_flops_per_image
        * len(query_indices)
    )

    # --------------------------------------------------------
    # Run all shot sizes
    # --------------------------------------------------------

    shot_results = {}

    # Extract/load the full ordered 45-shot pool exactly once.
    full_train_indices = build_shot_indices(train_pool_indices, TRAIN_POOL_PER_CLASS)
    full_train_labels = np.asarray(
        [dataset.samples[i][1] for i in full_train_indices], dtype=np.int64
    )
    full_cache = os.path.join(model_cache, "train_45shot")
    os.makedirs(full_cache, exist_ok=True)
    full_feature_file = os.path.join(full_cache, "train_features.npy")
    full_label_file = os.path.join(full_cache, "train_labels.npy")
    full_index_file = os.path.join(full_cache, "train_indices.npy")
    use_full_cache = False
    if all(os.path.isfile(x) for x in (full_feature_file, full_label_file, full_index_file)):
        full_train_features = np.load(full_feature_file)
        cached_full_labels = np.load(full_label_file)
        cached_full_indices = np.load(full_index_file)
        use_full_cache = (
            full_train_features.shape[0] == len(full_train_indices)
            and full_train_features.shape[1] == feature_dim
            and np.array_equal(cached_full_labels, full_train_labels)
            and np.array_equal(cached_full_indices, np.asarray(full_train_indices))
        )
    if use_full_cache:
        print("Loading verified cached 45-shot train features...")
    else:
        print_separator(f"{name} — EXTRACT 45-SHOT FEATURE POOL ONCE")
        full_dataset = ImageNetSubset(dataset, full_train_indices, preprocess)
        full_loader = DataLoader(
            full_dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True
        )
        full_train_features, _ = extract_features(
            model, full_loader, len(full_train_indices)
        )
        np.save(full_feature_file, full_train_features)
        np.save(full_label_file, full_train_labels)
        np.save(full_index_file, np.asarray(full_train_indices, dtype=np.int64))
    full_pos = {int(idx): pos for pos, idx in enumerate(full_train_indices)}

    for shots in TRAIN_SHOTS:

        print_separator(
            f"{name} — {shots}-SHOT DATABASE"
        )

        current_train_indices = (
            build_shot_indices(
                train_pool_indices,
                shots
            )
        )

        num_train = len(
            current_train_indices
        )

        expected_num_train = (
            NUM_CLASSES
            * shots
        )

        assert (
            num_train
            == expected_num_train
        )

        current_train_labels = np.asarray(
            [
                dataset.samples[i][1]
                for i in current_train_indices
            ],
            dtype=np.int64
        )

        # ----------------------------------------------------
        # Cache paths
        # ----------------------------------------------------

        shot_cache = os.path.join(
            model_cache,
            f"train_{shots}shot"
        )

        os.makedirs(
            shot_cache,
            exist_ok=True
        )

        train_feature_file = os.path.join(
            shot_cache,
            "train_features.npy"
        )

        train_label_file = os.path.join(
            shot_cache,
            "train_labels.npy"
        )

        result_file = os.path.join(
            shot_cache,
            "results.json"
        )

        # ----------------------------------------------------
        # Slice this shot from the single verified 45k feature pool.
        # ----------------------------------------------------

        take = np.asarray(
            [full_pos[int(idx)] for idx in current_train_indices],
            dtype=np.int64,
        )
        train_features = full_train_features[take]
        train_labels = full_train_labels[take]
        train_time = 0.0

        # ----------------------------------------------------
        # KNN: K=20 only
        # ----------------------------------------------------

        print_separator(
            f"{name} — {shots}-SHOT FAISS KNN"
        )

        print(
            "Database size:",
            num_train
        )

        print(
            "Query size:",
            len(query_features)
        )

        print(
            "Feature dim:",
            feature_dim
        )

        if num_train < K:

            raise RuntimeError(
                f"Database has {num_train} images, "
                f"but K={K}."
            )

        index = faiss.IndexFlatIP(
            feature_dim
        )

        index.add(
            np.ascontiguousarray(
                train_features
            )
        )

        knn_start = time.time()

        similarities, neighbor_indices = (
            index.search(
                np.ascontiguousarray(
                    query_features
                ),
                K
            )
        )

        knn_time = (
            time.time()
            - knn_start
        )

        neighbor_labels = (
            train_labels[
                neighbor_indices
            ]
        )

        acc = weighted_knn_accuracy(
            similarities,
            neighbor_labels,
            query_labels,
            TEMPERATURE
        )

        acc_percent = (
            acc * 100
        )

        print(
            f"Train {shots}/class | "
            f"Images: {num_train} | "
            f"K=20 Top-1: {acc_percent:.4f}%"
        )

        print(
            f"FAISS search time: "
            f"{knn_time:.4f}s"
        )

        # ----------------------------------------------------
        # FLOPs
        # ----------------------------------------------------

        train_vision_flops = (
            vision_flops_per_image
            * num_train
        )

        total_feature_flops = (
            train_vision_flops
            + query_vision_flops
        )

        # Exact inner-product similarity FLOPs.
        #
        # 2 FLOPs per feature dimension.
        #
        knn_similarity_flops = (
            2
            * len(query_features)
            * num_train
            * feature_dim
        )

        total_knn_flops = (
            total_feature_flops
            + knn_similarity_flops
        )

        print(
            f"Feature FLOPs: "
            f"{total_feature_flops / 1e12:.4f} TFLOPs"
        )

        print(
            f"KNN similarity FLOPs: "
            f"{knn_similarity_flops / 1e12:.4f} TFLOPs"
        )

        print(
            f"Total KNN FLOPs: "
            f"{total_knn_flops / 1e12:.4f} TFLOPs"
        )

        print(
            f"Total KNN PFLOPs: "
            f"{total_knn_flops / 1e15:.6f} PFLOPs"
        )

        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        result = {

            "model": name,

            "architecture": arch,

            "pretrained": pretrained,

            "parameters": int(
                num_params
            ),

            "feature_dim": int(
                feature_dim
            ),

            "dataset": {

                "path": IMAGENET_VAL,

                "seed": SEED,

                "num_classes":
                    NUM_CLASSES,

                "database_images":
                    num_train,

                "query_images":
                    len(query_indices),

                "database_per_class":
                    shots,

                "query_per_class":
                    QUERY_PER_CLASS,

            },

            "protocol": {

                "protocol_file":
                    PROTOCOL_PATH,

                "train_pool_per_class":
                    TRAIN_POOL_PER_CLASS,

                "query_per_class":
                    QUERY_PER_CLASS,

                "train_shots":
                    TRAIN_SHOTS,

            },

            "knn": {

                "index":
                    "FAISS IndexFlatIP",

                "metric":
                    "cosine similarity",

                "temperature":
                    TEMPERATURE,

                "k":
                    K,

                "top1_accuracy_percent":
                    acc_percent,

                "search_time_seconds":
                    knn_time,

            },

            "flops": {

                "vision": {

                    "per_image":
                        int(
                            vision_flops_per_image
                        ),

                    "database":
                        int(
                            train_vision_flops
                        ),

                    "query":
                        int(
                            query_vision_flops
                        ),

                    "total_feature_flops":
                        int(
                            total_feature_flops
                        ),

                    "total_feature_tflops":
                        (
                            total_feature_flops
                            / 1e12
                        ),

                },

                "knn_similarity": {

                    "flops":
                        int(
                            knn_similarity_flops
                        ),

                    "tflops":
                        (
                            knn_similarity_flops
                            / 1e12
                        ),

                },

                "total_knn_experiment": {

                    "flops":
                        int(
                            total_knn_flops
                        ),

                    "tflops":
                        (
                            total_knn_flops
                            / 1e12
                        ),

                    "pflops":
                        (
                            total_knn_flops
                            / 1e15
                        ),

                },

            },

            "runtime": {

                "model_load_seconds":
                    load_time,

                "database_feature_seconds":
                    train_time,

                "query_feature_seconds":
                    query_time,

                "knn_search_seconds":
                    knn_time,

            },

            "cache": {

                "database_features":
                    train_feature_file,

                "database_labels":
                    train_label_file,

                "query_features":
                    query_feature_file,

                "query_labels":
                    query_label_file,

                "query_indices":
                    query_index_file,

            },

        }

        with open(
            result_file,
            "w"
        ) as f:

            json.dump(
                result,
                f,
                indent=2
            )

        shot_results[str(shots)] = result

        # Free CPU memory.
        del train_features
        del index

    # --------------------------------------------------------
    # Free GPU memory
    # --------------------------------------------------------

    del model

    del query_features

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    return shot_results


# ============================================================
# 11. RUN ALL MODELS
# ============================================================

def main():

    set_seed(SEED)

    os.makedirs(
        CACHE_ROOT,
        exist_ok=True
    )

    print_separator(
        "LOADING IMAGENET VALIDATION"
    )

    dataset = datasets.ImageFolder(
        IMAGENET_VAL
    )

    assert (
        len(dataset.classes)
        == NUM_CLASSES
    )

    print(
        "Dataset:",
        IMAGENET_VAL
    )

    print(
        "Images:",
        len(dataset)
    )

    print(
        "Classes:",
        len(dataset.classes)
    )

    # --------------------------------------------------------
    # Protocol
    # --------------------------------------------------------

    train_pool_indices, query_indices = (
        build_protocol(dataset)
    )

    query_labels = verify_protocol(
        dataset,
        train_pool_indices,
        query_indices
    )

    # --------------------------------------------------------
    # Print protocol summary
    # --------------------------------------------------------

    print_separator(
        "EXPERIMENT CONFIGURATION"
    )

    print(
        "Train shots:",
        TRAIN_SHOTS
    )

    print(
        "Train pool/class:",
        TRAIN_POOL_PER_CLASS
    )

    print(
        "Query/class:",
        QUERY_PER_CLASS
    )

    print(
        "Total query:",
        len(query_indices)
    )

    print(
        "K:",
        K
    )

    print(
        "Temperature:",
        TEMPERATURE
    )

    print(
        "Protocol:",
        PROTOCOL_PATH
    )

    # --------------------------------------------------------
    # Run models
    # --------------------------------------------------------

    all_results = {}

    failed_models = {}

    global_start = time.time()

    for model_info in MODELS:

        name = model_info["name"]

        try:

            result = process_model(
                model_info,
                dataset,
                train_pool_indices,
                query_indices,
                query_labels
            )

            all_results[name] = result

        except Exception as e:

            print_separator(
                f"FAILED: {name}"
            )

            print(
                "Error:",
                repr(e)
            )

            failed_models[name] = {

                "architecture":
                    model_info["arch"],

                "pretrained":
                    model_info["pretrained"],

                "error":
                    repr(e),

            }

            if torch.cuda.is_available():

                torch.cuda.empty_cache()

    global_time = (
        time.time()
        - global_start
    )

    # --------------------------------------------------------
    # Global results
    # --------------------------------------------------------

    global_result = {

        "protocol": {

            "dataset":
                "ImageNet-1K validation",

            "dataset_path":
                IMAGENET_VAL,

            "seed":
                SEED,

            "train_shots":
                TRAIN_SHOTS,

            "train_pool_per_class":
                TRAIN_POOL_PER_CLASS,

            "query_per_class":
                QUERY_PER_CLASS,

            "query_images":
                NUM_CLASSES
                * QUERY_PER_CLASS,

            "knn_k":
                K,

            "temperature":
                TEMPERATURE,

            "feature_normalization":
                "L2",

            "faiss_index":
                "IndexFlatIP",

            "protocol_file":
                PROTOCOL_PATH,

        },

        "models":
            all_results,

        "failed_models":
            failed_models,

        "total_wall_clock_seconds":
            global_time,

    }

    global_result_file = os.path.join(
        CACHE_ROOT,
        "all_metaclip_knn_results_5_10_20_45shot.json"
    )

    with open(
        global_result_file,
        "w"
    ) as f:

        json.dump(
            global_result,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Final accuracy table
    # --------------------------------------------------------

    print_separator(
        "FINAL KNN ACCURACY COMPARISON"
    )

    header = (
        f"{'Model':35s} "
        + " ".join(
            f"{shots}-shot".rjust(10)
            for shots in TRAIN_SHOTS
        )
    )

    print(header)

    print(
        "-" * 85
    )

    for name, shot_results in all_results.items():

        values = []

        for shots in TRAIN_SHOTS:

            result = shot_results.get(
                str(shots)
            )

            if result is None:

                values.append(
                    float("nan")
                )

            else:

                values.append(
                    result[
                        "knn"
                    ][
                        "top1_accuracy_percent"
                    ]
                )

        value_string = " ".join(
            f"{v:10.4f}"
            for v in values
        )

        print(
            f"{name:35s} "
            f"{value_string}"
        )

    # --------------------------------------------------------
    # Final FLOPs table
    # --------------------------------------------------------

    print_separator(
        "FINAL FLOPs COMPARISON"
    )

    print(
        f"{'Model':30s} "
        f"{'Shot':>6s} "
        f"{'Feature(T)':>14s} "
        f"{'KNN(T)':>14s} "
        f"{'Total(T)':>14s} "
        f"{'Total(P)':>14s}"
    )

    print(
        "-" * 115
    )

    for name, shot_results in all_results.items():

        for shots in TRAIN_SHOTS:

            result = shot_results.get(
                str(shots)
            )

            if result is None:

                continue

            flops = result["flops"]

            feature_tflops = (
                flops[
                    "vision"
                ][
                    "total_feature_tflops"
                ]
            )

            knn_tflops = (
                flops[
                    "knn_similarity"
                ][
                    "tflops"
                ]
            )

            total_tflops = (
                flops[
                    "total_knn_experiment"
                ][
                    "tflops"
                ]
            )

            total_pflops = (
                flops[
                    "total_knn_experiment"
                ][
                    "pflops"
                ]
            )

            print(
                f"{name:30s} "
                f"{shots:6d} "
                f"{feature_tflops:14.4f} "
                f"{knn_tflops:14.4f} "
                f"{total_tflops:14.4f} "
                f"{total_pflops:14.6f}"
            )

    # --------------------------------------------------------
    # Failed models
    # --------------------------------------------------------

    if failed_models:

        print_separator(
            "FAILED MODELS"
        )

        for name, info in failed_models.items():

            print(
                f"\n{name}"
            )

            print(
                "Architecture:",
                info["architecture"]
            )

            print(
                "Pretrained:",
                info["pretrained"]
            )

            print(
                "Error:",
                info["error"]
            )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    print_separator(
        "DONE"
    )

    print(
        "Global result:"
    )

    print(
        global_result_file
    )


if __name__ == "__main__":

    main()
