import os
import sys
import json
import time
import math
import random
import warnings
import inspect
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np

import torch
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from torchvision import datasets, transforms

import faiss
import open_clip


# ============================================================
# 1. Global Settings
# ============================================================

SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# ImageNet
# ============================================================

IMAGENET_TRAIN = "/cache/imagenet-1k/standard/train"


# ============================================================
# Protocol
# ============================================================

TRAIN_SHOTS = [5, 10, 20, 45]

TRAIN_POOL_PER_CLASS = 45

QUERY_PER_CLASS = 5

K = 20

TEMPERATURE = 0.07

PROTOCOL_PATH = (
    "/cache/metaclip_knn/"
    "val_45shot_5query_seed42_protocol.json"
)


# ============================================================
# Cache
# ============================================================

CACHE_ROOT = "/cache/metaclip2_worldwide_knn"

FEATURE_ROOT = os.path.join(
    CACHE_ROOT,
    "features"
)

RESULT_ROOT = os.path.join(
    CACHE_ROOT,
    "results"
)

os.makedirs(
    FEATURE_ROOT,
    exist_ok=True
)

os.makedirs(
    RESULT_ROOT,
    exist_ok=True
)


ALL_RESULTS_PATH = os.path.join(
    RESULT_ROOT,
    "all_metaclip2_worldwide_knn_results_5_10_20_45shot.json"
)


# ============================================================
# Runtime
# ============================================================

BATCH_SIZE = 128

NUM_WORKERS = 8

STORE_DTYPE = np.float16


# ============================================================
# 2. MetaCLIP2 Models
# ============================================================

PT_MODELS = [

    {
        "name": "MetaCLIP2-B16-224-worldwide",
        "arch": "ViT-B-16",
        "image_size": 224,
        "checkpoint":
            "/workspace/.cache/clip/"
            "metaclip2_b16_224px_worldwide.pt",
    },

    {
        "name": "MetaCLIP2-B16-384-worldwide",
        "arch": "ViT-B-16",
        "image_size": 384,
        "checkpoint":
            "/workspace/.cache/clip/"
            "metaclip2_b16_384px_worldwide.pt",
    },

    {
        "name": "MetaCLIP2-B32-224-worldwide",
        "arch": "ViT-B-32",
        "image_size": 224,
        "checkpoint":
            "/workspace/.cache/clip/"
            "metaclip2_b32_224px_worldwide.pt",
    },

    {
        "name": "MetaCLIP2-B32-224-mt5-worldwide",
        "arch": "ViT-B-32",
        "image_size": 224,
        "checkpoint":
            "/workspace/.cache/clip/"
            "metaclip2_b32_224px_mt5_worldwide.pt",
    },

    {
        "name": "MetaCLIP2-B32-384-worldwide",
        "arch": "ViT-B-32",
        "image_size": 384,
        "checkpoint":
            "/workspace/.cache/clip/"
            "metaclip2_b32_384px_worldwide.pt",
    },

    {
        "name": "MetaCLIP2-S16-224-worldwide",
        "arch": "ViT-S-16",
        "image_size": 224,
        "checkpoint":
            "/workspace/.cache/clip/"
            "metaclip2_s16_224px_worldwide.pt",
    },

    {
        "name": "MetaCLIP2-S16-224-mt5-worldwide",
        "arch": "ViT-S-16",
        "image_size": 224,
        "checkpoint":
            "/workspace/.cache/clip/"
            "metaclip2_s16_224px_mt5_worldwide.pt",
    },

    {
        "name": "MetaCLIP2-S16-384-worldwide",
        "arch": "ViT-S-16",
        "image_size": 384,
        "checkpoint":
            "/workspace/.cache/clip/"
            "metaclip2_s16_384px_worldwide.pt",
    },

    {
        "name": "MetaCLIP2-M16-224-worldwide",
        "arch": "ViT-M-16",
        "image_size": 224,
        "checkpoint":
            "/workspace/.cache/clip/"
            "metaclip2_m16_224px_worldwide.pt",
    },

    {
        "name": "MetaCLIP2-M16-224-mt5-worldwide",
        "arch": "ViT-M-16",
        "image_size": 224,
        "checkpoint":
            "/workspace/.cache/clip/"
            "metaclip2_m16_224px_mt5_worldwide.pt",
    },

    {
        "name": "MetaCLIP2-M16-384-worldwide",
        "arch": "ViT-M-16",
        "image_size": 384,
        "checkpoint":
            "/workspace/.cache/clip/"
            "metaclip2_m16_384px_worldwide.pt",
    },

    {
        "name": "MetaCLIP2-L14-224-worldwide",
        "arch": "ViT-L-14",
        "image_size": 224,
        "checkpoint":
            "/workspace/.cache/clip/"
            "metaclip2_l14_224px_worldwide.pt",
    },

    {
        "name": "MetaCLIP2-H14-378-worldwide",
        "arch": "ViT-H-14",
        "image_size": 378,
        "checkpoint":
            "/workspace/.cache/clip/"
            "metaclip2_h14_378px_worldwide.pt",
    },
]


# ============================================================
# 3. Seed
# ============================================================

def seed_everything(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


seed_everything(SEED)


# ============================================================
# 4. Protocol
# ============================================================

def build_or_load_protocol():

    os.makedirs(
        os.path.dirname(PROTOCOL_PATH),
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load existing protocol
    # --------------------------------------------------------

    if os.path.exists(PROTOCOL_PATH):

        print("=" * 100)
        print("Loading existing protocol:")
        print(PROTOCOL_PATH)
        print("=" * 100)

        with open(
            PROTOCOL_PATH,
            "r"
        ) as f:

            protocol = json.load(f)

        # ----------------------------------------------------
        # Validate protocol
        # ----------------------------------------------------

        if protocol["seed"] != SEED:
            raise RuntimeError(
                "Existing protocol seed mismatch."
            )

        if (
            protocol["train_pool_per_class"]
            != TRAIN_POOL_PER_CLASS
        ):
            raise RuntimeError(
                "Existing protocol train pool mismatch."
            )

        if (
            protocol["query_per_class"]
            != QUERY_PER_CLASS
        ):
            raise RuntimeError(
                "Existing protocol query mismatch."
            )

        if (
            protocol["train_shots"]
            != TRAIN_SHOTS
        ):
            raise RuntimeError(
                "Existing protocol train shots mismatch."
            )

        print(
            f"Train pool: "
            f"{protocol['train_pool_size']}"
        )

        print(
            f"Query: "
            f"{protocol['query_size']}"
        )

        for shot in TRAIN_SHOTS:

            print(
                f"{shot}-shot: "
                f"{len(protocol['train_indices_by_shot'][str(shot)])}"
                f" images"
            )

        return protocol

    # --------------------------------------------------------
    # Build new protocol
    # --------------------------------------------------------

    print("=" * 100)
    print("Building new ImageNet protocol")
    print("=" * 100)

    dataset = datasets.ImageFolder(
        IMAGENET_TRAIN
    )

    num_classes = len(
        dataset.classes
    )

    print(
        f"ImageNet classes: "
        f"{num_classes}"
    )

    if num_classes != 1000:

        raise RuntimeError(
            f"Expected 1000 classes, "
            f"got {num_classes}"
        )

    # --------------------------------------------------------
    # Group indices by class
    # --------------------------------------------------------

    class_to_indices = {
        c: []
        for c in range(num_classes)
    }

    for idx, (_, label) in enumerate(
        dataset.samples
    ):

        class_to_indices[label].append(
            idx
        )

    rng = random.Random(SEED)

    train_pool_by_class = {}

    query_by_class = {}

    train_indices_by_shot = {
        str(shot): []
        for shot in TRAIN_SHOTS
    }

    query_indices = []

    # --------------------------------------------------------
    # Per-class sampling
    # --------------------------------------------------------

    for c in range(num_classes):

        indices = class_to_indices[c].copy()

        required = (
            TRAIN_POOL_PER_CLASS
            + QUERY_PER_CLASS
        )

        if len(indices) < required:

            raise RuntimeError(
                f"Class {c} has only "
                f"{len(indices)} images, "
                f"but {required} required."
            )

        rng.shuffle(indices)

        selected = indices[:required]

        train_pool = selected[
            :TRAIN_POOL_PER_CLASS
        ]

        query = selected[
            TRAIN_POOL_PER_CLASS:
        ]

        train_pool_by_class[str(c)] = (
            train_pool
        )

        query_by_class[str(c)] = (
            query
        )

        # ----------------------------------------------------
        # Nested training sets
        # ----------------------------------------------------

        for shot in TRAIN_SHOTS:

            train_indices_by_shot[
                str(shot)
            ].extend(
                train_pool[:shot]
            )

        query_indices.extend(
            query
        )

    # --------------------------------------------------------
    # Protocol object
    # --------------------------------------------------------

    protocol = {

        "seed":
            SEED,

        "dataset":
            IMAGENET_TRAIN,

        "num_classes":
            num_classes,

        "train_pool_per_class":
            TRAIN_POOL_PER_CLASS,

        "query_per_class":
            QUERY_PER_CLASS,

        "train_shots":
            TRAIN_SHOTS,

        "train_pool_size":
            num_classes
            * TRAIN_POOL_PER_CLASS,

        "query_size":
            num_classes
            * QUERY_PER_CLASS,

        "train_indices_by_shot":
            train_indices_by_shot,

        "train_pool_by_class":
            train_pool_by_class,

        "query_by_class":
            query_by_class,

        "query_indices":
            query_indices,
    }

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    with open(
        PROTOCOL_PATH,
        "w"
    ) as f:

        json.dump(
            protocol,
            f,
            indent=2
        )

    print(
        f"Saved protocol:\n"
        f"{PROTOCOL_PATH}"
    )

    return protocol


# ============================================================
# 5. Dataset
# ============================================================

class IndexedImageFolder(Dataset):

    def __init__(
        self,
        root,
        indices,
        transform=None
    ):

        self.dataset = datasets.ImageFolder(
            root=root,
            transform=transform
        )

        self.indices = list(indices)

    def __len__(self):

        return len(
            self.indices
        )

    def __getitem__(
        self,
        idx
    ):

        real_idx = self.indices[idx]

        image, label = (
            self.dataset[real_idx]
        )

        return image, label


# ============================================================
# 6. Checkpoint Loading
# ============================================================

def clean_state_dict(
    state_dict
):

    new_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):

            key = key[
                len("module.") :
            ]

        new_state_dict[key] = value

    return new_state_dict


def load_checkpoint(
    path
):

    print(
        f"Loading checkpoint:\n"
        f"{path}"
    )

    checkpoint = torch.load(
        path,
        map_location="cpu"
    )

    if isinstance(
        checkpoint,
        dict
    ):

        if "state_dict" in checkpoint:

            checkpoint = checkpoint[
                "state_dict"
            ]

        elif "model" in checkpoint:

            checkpoint = checkpoint[
                "model"
            ]

    checkpoint = clean_state_dict(
        checkpoint
    )

    return checkpoint


# ============================================================
# 7. Positional Embedding Helper
# ============================================================

def resize_positional_embedding(
    pos_embed,
    target_num_tokens
):

    """
    Resize ViT positional embedding if necessary.

    pos_embed:
        [1, N, C]

    The first token is treated as CLS.
    """

    if pos_embed.ndim != 3:

        return pos_embed

    old_num_tokens = (
        pos_embed.shape[1]
    )

    if (
        old_num_tokens
        == target_num_tokens
    ):

        return pos_embed

    cls_pos = pos_embed[
        :, :1, :
    ]

    patch_pos = pos_embed[
        :, 1:, :
    ]

    old_grid = int(
        math.sqrt(
            patch_pos.shape[1]
        )
    )

    new_patch_tokens = (
        target_num_tokens - 1
    )

    new_grid = int(
        math.sqrt(
            new_patch_tokens
        )
    )

    if (
        old_grid * old_grid
        != patch_pos.shape[1]
    ):

        raise RuntimeError(
            "Cannot infer old positional "
            "embedding grid."
        )

    if (
        new_grid * new_grid
        != new_patch_tokens
    ):

        raise RuntimeError(
            "Cannot infer new positional "
            "embedding grid."
        )

    patch_pos = patch_pos.reshape(
        1,
        old_grid,
        old_grid,
        -1
    )

    patch_pos = patch_pos.permute(
        0,
        3,
        1,
        2
    )

    patch_pos = F.interpolate(
        patch_pos.float(),
        size=(
            new_grid,
            new_grid
        ),
        mode="bicubic",
        align_corners=False
    )

    patch_pos = patch_pos.permute(
        0,
        2,
        3,
        1
    )

    patch_pos = patch_pos.reshape(
        1,
        new_grid * new_grid,
        -1
    )

    patch_pos = patch_pos.to(
        pos_embed.dtype
    )

    return torch.cat(
        [
            cls_pos,
            patch_pos
        ],
        dim=1
    )


# ============================================================
# 8. Build MetaCLIP2 Vision-only Model
# ============================================================

def build_metaclip2_vision_only(
    arch,
    checkpoint_path,
    image_size
):

    print("-" * 80)

    print(
        f"Building OpenCLIP model: "
        f"{arch}, image_size={image_size}"
    )

    # --------------------------------------------------------
    # Inspect installed OpenCLIP API
    # --------------------------------------------------------

    create_model_signature = (
        inspect.signature(
            open_clip.create_model
        )
    )

    supports_force_image_size = (
        "force_image_size"
        in create_model_signature.parameters
    )

    print(
        f"OpenCLIP supports "
        f"force_image_size: "
        f"{supports_force_image_size}"
    )

    # --------------------------------------------------------
    # Build architecture
    # --------------------------------------------------------

    if supports_force_image_size:

        print(
            "Creating model with "
            "force_image_size=..."
        )

        model = open_clip.create_model(
            arch,
            pretrained=None,
            force_image_size=image_size
        )

    else:

        print(
            "Creating model without "
            "force_image_size."
        )

        model = open_clip.create_model(
            arch,
            pretrained=None
        )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint = load_checkpoint(
        checkpoint_path
    )

    # --------------------------------------------------------
    # Extract visual weights
    # --------------------------------------------------------

    visual_state = {}

    for key, value in checkpoint.items():

        if key.startswith(
            "visual."
        ):

            visual_state[
                key[len("visual."):]
            ] = value

        elif key.startswith(
            "module.visual."
        ):

            visual_state[
                key[len("module.visual."):]
            ] = value

    # --------------------------------------------------------
    # Fallback:
    # checkpoint itself may already be vision-only
    # --------------------------------------------------------

    if len(visual_state) == 0:

        for key, value in checkpoint.items():

            if (
                key.startswith("conv1.")
                or key.startswith(
                    "class_embedding"
                )
                or key.startswith(
                    "positional_embedding"
                )
                or key.startswith(
                    "transformer."
                )
                or key.startswith(
                    "ln_pre."
                )
                or key.startswith(
                    "ln_post."
                )
            ):

                visual_state[key] = value

    if len(visual_state) == 0:

        raise RuntimeError(
            "Could not find visual weights "
            "in MetaCLIP2 checkpoint."
        )

    # --------------------------------------------------------
    # Positional embedding compatibility
    # --------------------------------------------------------

    if (
        "positional_embedding"
        in visual_state
    ):

        ckpt_pos = (
            visual_state[
                "positional_embedding"
            ]
        )

        model_pos = (
            model.visual.positional_embedding
        )

        if (
            ckpt_pos.shape
            != model_pos.shape
        ):

            print(
                "Positional embedding shape mismatch:"
            )

            print(
                f"  checkpoint: "
                f"{tuple(ckpt_pos.shape)}"
            )

            print(
                f"  model: "
                f"{tuple(model_pos.shape)}"
            )

            # ------------------------------------------------
            # If model image size was not directly configured,
            # resize checkpoint positional embedding.
            # ------------------------------------------------

            visual_state[
                "positional_embedding"
            ] = resize_positional_embedding(
                ckpt_pos,
                model_pos.shape[1]
            )

            print(
                "Positional embedding resized."
            )

    # --------------------------------------------------------
    # Load visual weights
    # --------------------------------------------------------

    missing, unexpected = (
        model.visual.load_state_dict(
            visual_state,
            strict=False
        )
    )

    print(
        f"Visual tensors loaded: "
        f"{len(visual_state)}"
    )

    if len(missing) > 0:

        print(
            f"Missing visual keys: "
            f"{len(missing)}"
        )

        for key in missing[:20]:

            print(
                f"  missing: {key}"
            )

    if len(unexpected) > 0:

        print(
            f"Unexpected visual keys: "
            f"{len(unexpected)}"
        )

        for key in unexpected[:20]:

            print(
                f"  unexpected: {key}"
            )

    # --------------------------------------------------------
    # Verify model image size
    # --------------------------------------------------------

    if hasattr(
        model.visual,
        "image_size"
    ):

        print(
            f"Visual image_size: "
            f"{model.visual.image_size}"
        )

    if hasattr(
        model.visual,
        "grid_size"
    ):

        print(
            f"Visual grid_size: "
            f"{model.visual.grid_size}"
        )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    model = model.to(
        DEVICE
    )

    model.eval()

    return model


# ============================================================
# 9. Transform
# ============================================================

def build_transform(
    image_size
):

    return transforms.Compose([

        transforms.Resize(
            image_size,
            interpolation=
                transforms.InterpolationMode.BICUBIC
        ),

        transforms.CenterCrop(
            image_size
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=(
                0.48145466,
                0.4578275,
                0.40821073
            ),

            std=(
                0.26862954,
                0.26130258,
                0.27577711
            )
        ),
    ])


# ============================================================
# 10. Feature Extraction
# ============================================================

@torch.no_grad()
def extract_features(
    model,
    loader,
    description=""
):

    all_features = []

    all_labels = []

    total = len(
        loader.dataset
    )

    print(
        f"Extracting features: "
        f"{description}"
    )

    start_time = time.time()

    processed = 0

    for images, labels in loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        features = model.encode_image(
            images
        )

        # ----------------------------------------------------
        # L2 normalize
        # ----------------------------------------------------

        features = F.normalize(
            features.float(),
            dim=-1
        )

        features = (
            features
            .cpu()
            .numpy()
        )

        all_features.append(
            features
        )

        all_labels.append(
            labels.numpy()
        )

        processed += (
            images.shape[0]
        )

        if (
            processed == total
            or
            processed % (
                BATCH_SIZE * 20
            ) == 0
        ):

            elapsed = (
                time.time()
                - start_time
            )

            speed = (
                processed / elapsed
                if elapsed > 0
                else 0
            )

            print(
                f"  {processed}/{total} "
                f"({processed / total * 100:.1f}%) "
                f"{speed:.1f} img/s"
            )

    features = np.concatenate(
        all_features,
        axis=0
    )

    labels = np.concatenate(
        all_labels,
        axis=0
    )

    return (
        features,
        labels
    )


# ============================================================
# 11. Cache Paths
# ============================================================

def get_model_cache_dir(
    model_name
):

    safe_name = (
        model_name
        .replace("/", "_")
        .replace(" ", "_")
    )

    return os.path.join(
        FEATURE_ROOT,
        safe_name
    )


def query_feature_paths(
    model_name
):

    model_dir = (
        get_model_cache_dir(
            model_name
        )
    )

    return (

        os.path.join(
            model_dir,
            "query_45pool5query_seed42_features.npy"
        ),

        os.path.join(
            model_dir,
            "query_45pool5query_seed42_labels.npy"
        ),
    )


def train_feature_paths(
    model_name,
    shot
):

    model_dir = (
        get_model_cache_dir(
            model_name
        )
    )

    shot_dir = os.path.join(
        model_dir,
        f"train_{shot}shot"
    )

    os.makedirs(
        shot_dir,
        exist_ok=True
    )

    return (

        os.path.join(
            shot_dir,
            "train_features.npy"
        ),

        os.path.join(
            shot_dir,
            "train_labels.npy"
        ),
    )


# ============================================================
# 12. Query Features
# ============================================================

def get_query_features(
    model,
    model_cfg,
    protocol
):

    model_name = model_cfg[
        "name"
    ]

    feature_path, label_path = (
        query_feature_paths(
            model_name
        )
    )

    model_dir = (
        get_model_cache_dir(
            model_name
        )
    )

    os.makedirs(
        model_dir,
        exist_ok=True
    )

    expected_size = (
        1000
        * QUERY_PER_CLASS
    )

    # --------------------------------------------------------
    # Existing cache
    # --------------------------------------------------------

    if (
        os.path.exists(feature_path)
        and
        os.path.exists(label_path)
    ):

        print(
            "\nLoading cached query features:"
        )

        print(
            feature_path
        )

        features = np.load(
            feature_path,
            mmap_mode="r"
        )

        labels = np.load(
            label_path,
            mmap_mode="r"
        )

        if (
            len(features)
            == expected_size
            and
            len(labels)
            == expected_size
        ):

            print(
                f"Query cache valid: "
                f"{len(features)}"
            )

            return (
                features,
                labels
            )

        print(
            "Query cache size mismatch."
        )

    # --------------------------------------------------------
    # Extract
    # --------------------------------------------------------

    transform = build_transform(
        model_cfg[
            "image_size"
        ]
    )

    query_indices = (
        protocol[
            "query_indices"
        ]
    )

    dataset = IndexedImageFolder(
        root=IMAGENET_TRAIN,
        indices=query_indices,
        transform=transform
    )

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

    features, labels = (
        extract_features(
            model,
            loader,
            description=(
                f"{model_name} "
                f"QUERY "
                f"({len(dataset)})"
            )
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    np.save(
        feature_path,
        features.astype(
            STORE_DTYPE
        )
    )

    np.save(
        label_path,
        labels
    )

    print(
        f"Saved query features:"
    )

    print(
        feature_path
    )

    return (

        np.load(
            feature_path,
            mmap_mode="r"
        ),

        np.load(
            label_path,
            mmap_mode="r"
        ),
    )


# ============================================================
# 13. Train Features
# ============================================================

def get_train_features(
    model,
    model_cfg,
    protocol,
    shot
):

    model_name = model_cfg[
        "name"
    ]

    feature_path, label_path = (
        train_feature_paths(
            model_name,
            shot
        )
    )

    expected_size = (
        1000 * shot
    )

    # --------------------------------------------------------
    # Existing cache
    # --------------------------------------------------------

    if (
        os.path.exists(feature_path)
        and
        os.path.exists(label_path)
    ):

        print(
            f"\nLoading cached "
            f"{shot}-shot features:"
        )

        print(
            feature_path
        )

        features = np.load(
            feature_path,
            mmap_mode="r"
        )

        labels = np.load(
            label_path,
            mmap_mode="r"
        )

        if (
            len(features)
            == expected_size
            and
            len(labels)
            == expected_size
        ):

            print(
                f"Train cache valid: "
                f"{len(features)}"
            )

            return (
                features,
                labels
            )

        print(
            "Train cache size mismatch."
        )

    # --------------------------------------------------------
    # Extract
    # --------------------------------------------------------

    transform = build_transform(
        model_cfg[
            "image_size"
        ]
    )

    indices = (
        protocol[
            "train_indices_by_shot"
        ][
            str(shot)
        ]
    )

    dataset = IndexedImageFolder(
        root=IMAGENET_TRAIN,
        indices=indices,
        transform=transform
    )

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

    features, labels = (
        extract_features(
            model,
            loader,
            description=(
                f"{model_name} "
                f"TRAIN {shot}-shot "
                f"({len(dataset)})"
            )
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    np.save(
        feature_path,
        features.astype(
            STORE_DTYPE
        )
    )

    np.save(
        label_path,
        labels
    )

    print(
        f"Saved train features:"
    )

    print(
        feature_path
    )

    return (

        np.load(
            feature_path,
            mmap_mode="r"
        ),

        np.load(
            label_path,
            mmap_mode="r"
        ),
    )


# ============================================================
# 14. ViT FLOPs
# ============================================================

def get_model_dimensions(
    arch
):

    configs = {

        "ViT-S-16": {

            "patch": 16,
            "width": 384,
            "layers": 12,
            "heads": 6,
            "mlp_ratio": 4,
            "embed_dim": 384,
        },

        "ViT-B-16": {

            "patch": 16,
            "width": 768,
            "layers": 12,
            "heads": 12,
            "mlp_ratio": 4,
            "embed_dim": 512,
        },

        "ViT-B-32": {

            "patch": 32,
            "width": 768,
            "layers": 12,
            "heads": 12,
            "mlp_ratio": 4,
            "embed_dim": 512,
        },

        "ViT-M-16": {

            "patch": 16,
            "width": 1024,
            "layers": 24,
            "heads": 16,
            "mlp_ratio": 4,
            "embed_dim": 512,
        },

        "ViT-L-14": {

            "patch": 14,
            "width": 1024,
            "layers": 24,
            "heads": 16,
            "mlp_ratio": 4,
            "embed_dim": 768,
        },

        "ViT-H-14": {

            "patch": 14,
            "width": 1280,
            "layers": 32,
            "heads": 16,
            "mlp_ratio": 4,
            "embed_dim": 1024,
        },
    }

    if arch not in configs:

        raise ValueError(
            f"Unknown architecture: "
            f"{arch}"
        )

    return configs[arch]


def estimate_vit_flops_per_image(
    arch,
    image_size
):

    cfg = get_model_dimensions(
        arch
    )

    patch = cfg["patch"]

    width = cfg["width"]

    layers = cfg["layers"]

    mlp_ratio = cfg["mlp_ratio"]

    embed_dim = cfg["embed_dim"]

    grid = image_size // patch

    num_tokens = (
        grid * grid + 1
    )

    mlp_hidden = (
        width * mlp_ratio
    )

    # --------------------------------------------------------
    # Patch embedding
    # --------------------------------------------------------

    patch_flops = (
        grid
        * grid
        * patch
        * patch
        * 3
        * width
        * 2
    )

    # --------------------------------------------------------
    # QKV
    # --------------------------------------------------------

    qkv_flops = (
        num_tokens
        * width
        * (3 * width)
        * 2
    )

    # --------------------------------------------------------
    # Attention QK
    # --------------------------------------------------------

    qk_flops = (
        num_tokens
        * num_tokens
        * width
        * 2
    )

    # --------------------------------------------------------
    # Attention AV
    # --------------------------------------------------------

    av_flops = (
        num_tokens
        * num_tokens
        * width
        * 2
    )

    # --------------------------------------------------------
    # Output projection
    # --------------------------------------------------------

    out_proj_flops = (
        num_tokens
        * width
        * width
        * 2
    )

    # --------------------------------------------------------
    # MLP
    # --------------------------------------------------------

    mlp_flops = (

        num_tokens
        * width
        * mlp_hidden
        * 2

        +

        num_tokens
        * mlp_hidden
        * width
        * 2
    )

    transformer_per_layer = (
        qkv_flops
        + qk_flops
        + av_flops
        + out_proj_flops
        + mlp_flops
    )

    transformer_flops = (
        transformer_per_layer
        * layers
    )

    # --------------------------------------------------------
    # Final projection
    # --------------------------------------------------------

    projection_flops = (
        width
        * embed_dim
        * 2
    )

    total_flops = (
        patch_flops
        + transformer_flops
        + projection_flops
    )

    return int(
        total_flops
    )


# ============================================================
# 15. KNN FLOPs
# ============================================================

def calculate_knn_flops(
    vision_flops_per_image,
    num_train,
    num_query,
    feature_dim
):

    # --------------------------------------------------------
    # Feature extraction
    # --------------------------------------------------------

    feature_flops = (
        vision_flops_per_image
        * (
            num_train
            + num_query
        )
    )

    # --------------------------------------------------------
    # Exact similarity
    #
    # 2 FLOPs per multiply-add
    # --------------------------------------------------------

    similarity_flops = (
        2
        * num_query
        * num_train
        * feature_dim
    )

    total_flops = (
        feature_flops
        + similarity_flops
    )

    return {

        "feature_extraction_flops":
            int(feature_flops),

        "knn_similarity_flops":
            int(similarity_flops),

        "total_flops":
            int(total_flops),

        "feature_extraction_tflops":
            feature_flops / 1e12,

        "knn_similarity_tflops":
            similarity_flops / 1e12,

        "total_tflops":
            total_flops / 1e12,

        "total_pflops":
            total_flops / 1e15,
    }


# ============================================================
# 16. KNN
# ============================================================

def run_knn(
    train_features,
    train_labels,
    query_features,
    query_labels,
    k=20,
    temperature=0.07
):

    # --------------------------------------------------------
    # float32 for FAISS
    # --------------------------------------------------------

    train_features = np.asarray(
        train_features,
        dtype=np.float32
    )

    query_features = np.asarray(
        query_features,
        dtype=np.float32
    )

    train_labels = np.asarray(
        train_labels
    )

    query_labels = np.asarray(
        query_labels
    )

    # --------------------------------------------------------
    # L2 normalization
    # --------------------------------------------------------

    train_norm = np.linalg.norm(
        train_features,
        axis=1,
        keepdims=True
    )

    query_norm = np.linalg.norm(
        query_features,
        axis=1,
        keepdims=True
    )

    train_features /= np.maximum(
        train_norm,
        1e-12
    )

    query_features /= np.maximum(
        query_norm,
        1e-12
    )

    feature_dim = (
        train_features.shape[1]
    )

    # --------------------------------------------------------
    # FAISS
    # --------------------------------------------------------

    index = faiss.IndexFlatIP(
        feature_dim
    )

    index.add(
        train_features
    )

    similarities, neighbors = (
        index.search(
            query_features,
            k
        )
    )

    # --------------------------------------------------------
    # Temperature weighted voting
    # --------------------------------------------------------

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

    neighbor_labels = (
        train_labels[
            neighbors
        ]
    )

    predictions = np.zeros(
        len(query_labels),
        dtype=np.int64
    )

    for i in range(
        len(query_labels)
    ):

        vote = {}

        for label, weight in zip(
            neighbor_labels[i],
            weights[i]
        ):

            label = int(
                label
            )

            vote[label] = (
                vote.get(
                    label,
                    0.0
                )
                + float(weight)
            )

        predictions[i] = max(
            vote,
            key=vote.get
        )

    accuracy = (
        predictions
        == query_labels
    ).mean()

    # --------------------------------------------------------
    # Same-class neighbor rate
    # --------------------------------------------------------

    same_class = (
        neighbor_labels
        == query_labels[:, None]
    )

    neighbor_rate = (
        same_class.mean()
    )

    return {

        "accuracy":
            float(accuracy),

        "same_class_neighbor_rate":
            float(neighbor_rate),

        "predictions":
            predictions,

        "similarities":
            similarities,

        "neighbors":
            neighbors,
    }


# ============================================================
# 17. Parameter Count
# ============================================================

def count_parameters(
    model
):

    return sum(
        p.numel()
        for p in model.parameters()
    )


def count_visual_parameters(
    model
):

    if not hasattr(
        model,
        "visual"
    ):

        return None

    return sum(
        p.numel()
        for p in model.visual.parameters()
    )


# ============================================================
# 18. Process One Model
# ============================================================

def process_model(
    model_cfg,
    protocol
):

    model_name = model_cfg[
        "name"
    ]

    print("\n")
    print("=" * 100)
    print(
        f"MODEL: {model_name}"
    )
    print("=" * 100)

    checkpoint = model_cfg[
        "checkpoint"
    ]

    # --------------------------------------------------------
    # Check checkpoint
    # --------------------------------------------------------

    if not os.path.exists(
        checkpoint
    ):

        print(
            "[SKIP] Checkpoint does not exist:"
        )

        print(
            checkpoint
        )

        return []

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model_start = time.time()

    model = build_metaclip2_vision_only(
        arch=model_cfg[
            "arch"
        ],

        checkpoint_path=checkpoint,

        image_size=model_cfg[
            "image_size"
        ]
    )

    model_load_time = (
        time.time()
        - model_start
    )

    # --------------------------------------------------------
    # Parameters
    # --------------------------------------------------------

    total_params = (
        count_parameters(
            model
        )
    )

    visual_params = (
        count_visual_parameters(
            model
        )
    )

    print(
        f"Total params: "
        f"{total_params / 1e6:.2f} M"
    )

    if visual_params is not None:

        print(
            f"Vision params: "
            f"{visual_params / 1e6:.2f} M"
        )

    # --------------------------------------------------------
    # FLOPs
    # --------------------------------------------------------

    vision_flops_per_image = (
        estimate_vit_flops_per_image(
            model_cfg["arch"],
            model_cfg["image_size"]
        )
    )

    print(
        f"Vision FLOPs/image: "
        f"{vision_flops_per_image / 1e9:.4f} GFLOPs"
    )

    # --------------------------------------------------------
    # Query features
    #
    # IMPORTANT:
    # Extract only once.
    # Reused by 5/10/20/45-shot.
    # --------------------------------------------------------

    query_features, query_labels = (
        get_query_features(
            model,
            model_cfg,
            protocol
        )
    )

    query_size = len(
        query_features
    )

    feature_dim = (
        query_features.shape[1]
    )

    print(
        f"Query size: "
        f"{query_size}"
    )

    print(
        f"Feature dimension: "
        f"{feature_dim}"
    )

    model_results = []

    # ========================================================
    # Each shot
    # ========================================================

    for shot in TRAIN_SHOTS:

        print("\n")
        print("-" * 100)

        print(
            f"{model_name} | "
            f"{shot}-SHOT | "
            f"K={K}"
        )

        print("-" * 100)

        # ----------------------------------------------------
        # Train features
        # ----------------------------------------------------

        train_features, train_labels = (
            get_train_features(
                model,
                model_cfg,
                protocol,
                shot
            )
        )

        num_train = len(
            train_features
        )

        # ----------------------------------------------------
        # KNN
        # ----------------------------------------------------

        knn_start = time.time()

        knn_result = run_knn(
            train_features=
                train_features,

            train_labels=
                train_labels,

            query_features=
                query_features,

            query_labels=
                query_labels,

            k=K,

            temperature=
                TEMPERATURE
        )

        knn_time = (
            time.time()
            - knn_start
        )

        # ----------------------------------------------------
        # FLOPs
        # ----------------------------------------------------

        flops = calculate_knn_flops(
            vision_flops_per_image=
                vision_flops_per_image,

            num_train=
                num_train,

            num_query=
                query_size,

            feature_dim=
                feature_dim
        )

        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        result = {

            "model":
                model_name,

            "arch":
                model_cfg["arch"],

            "image_size":
                model_cfg["image_size"],

            "checkpoint":
                checkpoint,

            "total_params":
                int(
                    total_params
                ),

            "vision_params":
                (
                    int(
                        visual_params
                    )
                    if visual_params is not None
                    else None
                ),

            "feature_dim":
                int(
                    feature_dim
                ),

            "train_shots":
                int(
                    shot
                ),

            "train_pool_per_class":
                TRAIN_POOL_PER_CLASS,

            "query_per_class":
                QUERY_PER_CLASS,

            "database_size":
                int(
                    num_train
                ),

            "query_size":
                int(
                    query_size
                ),

            "num_classes":
                1000,

            "K":
                K,

            "temperature":
                TEMPERATURE,

            "seed":
                SEED,

            "protocol":
                PROTOCOL_PATH,

            "feature_normalization":
                "L2",

            "faiss_index":
                "IndexFlatIP",

            "knn_accuracy":
                knn_result[
                    "accuracy"
                ],

            "top1_accuracy":
                knn_result[
                    "accuracy"
                ],

            "same_class_neighbor_rate":
                knn_result[
                    "same_class_neighbor_rate"
                ],

            "vision_flops_per_image":
                int(
                    vision_flops_per_image
                ),

            "vision_gflops_per_image":
                vision_flops_per_image
                / 1e9,

            "feature_extraction_flops":
                flops[
                    "feature_extraction_flops"
                ],

            "knn_similarity_flops":
                flops[
                    "knn_similarity_flops"
                ],

            "total_flops":
                flops[
                    "total_flops"
                ],

            "feature_extraction_tflops":
                flops[
                    "feature_extraction_tflops"
                ],

            "knn_similarity_tflops":
                flops[
                    "knn_similarity_tflops"
                ],

            "total_tflops":
                flops[
                    "total_tflops"
                ],

            "total_pflops":
                flops[
                    "total_pflops"
                ],

            "knn_time_seconds":
                knn_time,

            "model_load_time_seconds":
                model_load_time,
        }

        model_results.append(
            result
        )

        # ----------------------------------------------------
        # Save shot result
        # ----------------------------------------------------

        model_dir = (
            get_model_cache_dir(
                model_name
            )
        )

        shot_dir = os.path.join(
            model_dir,
            f"train_{shot}shot"
        )

        os.makedirs(
            shot_dir,
            exist_ok=True
        )

        shot_result_path = (
            os.path.join(
                shot_dir,
                "results.json"
            )
        )

        with open(
            shot_result_path,
            "w"
        ) as f:

            json.dump(
                result,
                f,
                indent=2
            )

        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        print(
            "\nKNN Result"
        )

        print(
            f"  Shot: "
            f"{shot}"
        )

        print(
            f"  Train: "
            f"{num_train}"
        )

        print(
            f"  Query: "
            f"{query_size}"
        )

        print(
            f"  K: "
            f"{K}"
        )

        print(
            f"  Top-1: "
            f"{knn_result['accuracy'] * 100:.4f}%"
        )

        print(
            f"  Same-class neighbor rate: "
            f"{knn_result['same_class_neighbor_rate'] * 100:.4f}%"
        )

        print(
            f"  Vision FLOPs/image: "
            f"{vision_flops_per_image / 1e9:.4f} GFLOPs"
        )

        print(
            f"  Feature extraction: "
            f"{flops['feature_extraction_tflops']:.4f} TFLOPs"
        )

        print(
            f"  KNN similarity: "
            f"{flops['knn_similarity_tflops']:.4f} TFLOPs"
        )

        print(
            f"  Total: "
            f"{flops['total_tflops']:.4f} TFLOPs"
        )

        print(
            f"  Total: "
            f"{flops['total_pflops']:.6f} PFLOPs"
        )

        print(
            f"  KNN time: "
            f"{knn_time:.2f} sec"
        )

    # --------------------------------------------------------
    # Free model
    # --------------------------------------------------------

    del model

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    return model_results


# ============================================================
# 19. Main
# ============================================================

def main():

    print("=" * 100)
    print(
        "MetaCLIP2 Worldwide "
        "ImageNet KNN"
    )
    print("=" * 100)

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"ImageNet train: "
        f"{IMAGENET_TRAIN}"
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
        f"K: {K}"
    )

    print(
        f"Temperature: "
        f"{TEMPERATURE}"
    )

    print(
        f"Seed: {SEED}"
    )

    print(
        f"Batch size: "
        f"{BATCH_SIZE}"
    )

    print(
        f"Num workers: "
        f"{NUM_WORKERS}"
    )

    # --------------------------------------------------------
    # Dataset check
    # --------------------------------------------------------

    if not os.path.isdir(
        IMAGENET_TRAIN
    ):

        raise FileNotFoundError(
            f"ImageNet train directory not found:\n"
            f"{IMAGENET_TRAIN}"
        )

    # --------------------------------------------------------
    # Protocol
    # --------------------------------------------------------

    protocol = (
        build_or_load_protocol()
    )

    # --------------------------------------------------------
    # Protocol summary
    # --------------------------------------------------------

    print("\n")

    print("=" * 100)
    print(
        "PROTOCOL SUMMARY"
    )
    print("=" * 100)

    print(
        f"Train pool: "
        f"{protocol['train_pool_size']}"
    )

    print(
        f"Query: "
        f"{protocol['query_size']}"
    )

    for shot in TRAIN_SHOTS:

        size = len(
            protocol[
                "train_indices_by_shot"
            ][
                str(shot)
            ]
        )

        print(
            f"{shot}-shot: "
            f"{size} images"
        )

    # --------------------------------------------------------
    # Process models
    # --------------------------------------------------------

    all_results = []

    for model_cfg in PT_MODELS:

        try:

            results = process_model(
                model_cfg,
                protocol
            )

            all_results.extend(
                results
            )

        except Exception as e:

            print("\n")

            print("=" * 100)

            print(
                f"[ERROR] "
                f"{model_cfg['name']}"
            )

            print("=" * 100)

            print(
                repr(e)
            )

            import traceback

            traceback.print_exc()

            # Continue next model
            continue

    # --------------------------------------------------------
    # Save all results
    # --------------------------------------------------------

    final_output = {

        "protocol": {

            "seed":
                SEED,

            "dataset":
                IMAGENET_TRAIN,

            "train_shots":
                TRAIN_SHOTS,

            "train_pool_per_class":
                TRAIN_POOL_PER_CLASS,

            "query_per_class":
                QUERY_PER_CLASS,

            "K":
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

        "results":
            all_results,
    }

    with open(
        ALL_RESULTS_PATH,
        "w"
    ) as f:

        json.dump(
            final_output,
            f,
            indent=2
        )

    # ========================================================
    # Final Summary
    # ========================================================

    print("\n")

    print("=" * 120)
    print(
        "FINAL SUMMARY"
    )
    print("=" * 120)

    header = (

        f"{'Model':45s} "
        f"{'Shot':>6s} "
        f"{'Train':>8s} "
        f"{'K':>4s} "
        f"{'Top1':>10s} "
        f"{'GFLOPs/img':>12s} "
        f"{'Total PFLOPs':>14s}"
    )

    print(
        header
    )

    print(
        "-" * 120
    )

    for r in all_results:

        print(

            f"{r['model'][:45]:45s} "

            f"{r['train_shots']:6d} "

            f"{r['database_size']:8d} "

            f"{r['K']:4d} "

            f"{r['knn_accuracy'] * 100:9.4f}% "

            f"{r['vision_gflops_per_image']:12.4f} "

            f"{r['total_pflops']:14.6f}"
        )

    print(
        "-" * 120
    )

    print(
        "\nSaved all results:"
    )

    print(
        ALL_RESULTS_PATH
    )

    print(
        f"\nTotal result entries: "
        f"{len(all_results)}"
    )


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":

    main()
