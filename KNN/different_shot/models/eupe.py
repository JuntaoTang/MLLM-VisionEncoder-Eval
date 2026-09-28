import os
import sys
import json
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np

import torch
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import Dataset, DataLoader

import faiss


# ============================================================
# 0. Paths
# ============================================================

REPO_DIR = "/cache/models/repositories/eupe"

MODEL_DIRS = {
    "EUPE-ViT-B": "/cache/models/model/EUPE-ViT-B",
    "EUPE-ViT-S": "/cache/models/model/EUPE-ViT-S",
    "EUPE-ViT-T": "/cache/models/model/EUPE-ViT-T",
    "EUPE-ConvNeXt-B": "/cache/models/model/EUPE-ConvNeXt-B",
}

IMAGENET_VAL_ROOT = "/workspace/root/val"

PROTOCOL_PATH = (
    "/cache/metaclip_knn/"
    "val_45shot_5query_seed42_protocol.json"
)

OUTPUT_ROOT = "/cache/eupe_knn"

FEATURE_ROOT = os.path.join(
    OUTPUT_ROOT,
    "features_45pool",
)

RESULT_ROOT = os.path.join(
    OUTPUT_ROOT,
    "results",
)

os.makedirs(
    FEATURE_ROOT,
    exist_ok=True,
)

os.makedirs(
    RESULT_ROOT,
    exist_ok=True,
)


# ============================================================
# 1. Unified KNN protocol
# ============================================================

NUM_CLASSES = 1000

DATABASE_PER_CLASS = 45
QUERY_PER_CLASS = 5

NUM_DATABASE = (
    NUM_CLASSES
    * DATABASE_PER_CLASS
)

NUM_QUERY = (
    NUM_CLASSES
    * QUERY_PER_CLASS
)

K_VALUES = [1, 5, 10, 20]

TEMPERATURE = 0.07

BATCH_SIZE = 128
NUM_WORKERS = 8

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# ------------------------------------------------------------
# EUPE official preprocessing example uses 256x256.
#
# We use this for all EUPE models to keep the input protocol
# identical.
# ------------------------------------------------------------

IMG_SIZE = 256

MEAN = (
    0.485,
    0.456,
    0.406,
)

STD = (
    0.229,
    0.224,
    0.225,
)


# ============================================================
# 2. Import official EUPE repo
# ============================================================

if not os.path.isdir(REPO_DIR):

    raise FileNotFoundError(
        "\nEUPE repository was not found.\n\n"
        "Please clone it first:\n\n"
        "cd /cache/models/repositories\n"
        "git clone "
        "https://github.com/facebookresearch/eupe.git\n\n"
        f"Expected:\n{REPO_DIR}\n"
    )

sys.path.insert(
    0,
    REPO_DIR,
)

print("=" * 90)
print("EUPE repository")
print("=" * 90)
print(
    f"REPO_DIR = {REPO_DIR}"
)


# ============================================================
# 3. Torchvision transform
# ============================================================

from torchvision import transforms


TRANSFORM = transforms.Compose(
    [
        transforms.Resize(
            (IMG_SIZE, IMG_SIZE),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=MEAN,
            std=STD,
        ),
    ]
)


# ============================================================
# 4. ImageNet Dataset
# ============================================================

class ImageNetValDataset(
    Dataset
):

    def __init__(
        self,
        root,
        transform,
    ):

        self.root = root
        self.transform = transform

        self.samples = []

        class_dirs = sorted(
            [
                d
                for d in os.listdir(root)
                if os.path.isdir(
                    os.path.join(
                        root,
                        d,
                    )
                )
            ]
        )

        if len(class_dirs) != NUM_CLASSES:

            raise RuntimeError(
                f"Expected {NUM_CLASSES} "
                f"class directories, "
                f"found {len(class_dirs)}"
            )

        self.class_to_idx = {
            name: i
            for i, name in enumerate(
                class_dirs
            )
        }

        for class_name in class_dirs:

            class_dir = os.path.join(
                root,
                class_name,
            )

            files = sorted(
                [
                    f
                    for f in os.listdir(
                        class_dir
                    )
                    if f.lower().endswith(
                        (
                            ".jpg",
                            ".jpeg",
                            ".png",
                            ".webp",
                        )
                    )
                ]
            )

            label = self.class_to_idx[
                class_name
            ]

            for filename in files:

                path = os.path.join(
                    class_dir,
                    filename,
                )

                self.samples.append(
                    (
                        path,
                        label,
                    )
                )

        print(
            f"ImageNet val images: "
            f"{len(self.samples):,}"
        )

    def __len__(self):

        return len(
            self.samples
        )

    def __getitem__(
        self,
        index,
    ):

        path, label = (
            self.samples[index]
        )

        image = Image.open(
            path
        ).convert("RGB")

        image = self.transform(
            image
        )

        return (
            image,
            label,
        )


class IndexedDataset(
    Dataset
):

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

        return len(
            self.indices
        )

    def __getitem__(
        self,
        index,
    ):

        return self.dataset[
            int(self.indices[index])
        ]


def collate_fn(batch):

    images = torch.stack(
        [
            x[0]
            for x in batch
        ],
        dim=0,
    )

    labels = torch.tensor(
        [
            x[1]
            for x in batch
        ],
        dtype=torch.long,
    )

    return images, labels


# ============================================================
# 5. Load existing protocol
# ============================================================

def load_protocol():

    if not os.path.isfile(
        PROTOCOL_PATH
    ):

        raise FileNotFoundError(
            f"Protocol not found:\n"
            f"{PROTOCOL_PATH}"
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

    print()
    print("=" * 90)
    print("KNN protocol")
    print("=" * 90)

    print(
        f"Protocol file: "
        f"{PROTOCOL_PATH}"
    )

    print(
        "Keys:",
        list(protocol.keys()),
    )

    return protocol


def extract_indices(
    protocol
):

    database_indices = None
    query_indices = None

    database_keys = [
        "train_pool_indices",
        "database_indices",
        "db_indices",
        "train_indices",
        "gallery_indices",
        "reference_indices",
    ]

    query_keys = [
        "query_indices",
        "test_indices",
        "val_indices",
        "query",
    ]

    for key in database_keys:

        if key in protocol:

            database_indices = (
                protocol[key]
            )

            break

    for key in query_keys:

        if key in protocol:

            query_indices = (
                protocol[key]
            )

            break

    # --------------------------------------------------------
    # Nested format
    # --------------------------------------------------------

    if database_indices is None:

        for parent in [
            "database",
            "db",
            "gallery",
            "train",
        ]:

            if parent not in protocol:
                continue

            obj = protocol[parent]

            if isinstance(
                obj,
                dict,
            ):

                for key in [
                    "indices",
                    "image_indices",
                    "samples",
                ]:

                    if key in obj:

                        database_indices = (
                            obj[key]
                        )

                        break

            if database_indices is not None:
                break

    if query_indices is None:

        for parent in [
            "query",
            "test",
            "val",
        ]:

            if parent not in protocol:
                continue

            obj = protocol[parent]

            if isinstance(
                obj,
                dict,
            ):

                for key in [
                    "indices",
                    "image_indices",
                    "samples",
                ]:

                    if key in obj:

                        query_indices = (
                            obj[key]
                        )

                        break

            if query_indices is not None:
                break

    if database_indices is None:

        raise RuntimeError(
            "Cannot find database indices.\n"
            f"Protocol keys: "
            f"{list(protocol.keys())}"
        )

    if query_indices is None:

        raise RuntimeError(
            "Cannot find query indices.\n"
            f"Protocol keys: "
            f"{list(protocol.keys())}"
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
        f"Database: "
        f"{len(database_indices):,}"
    )

    print(
        f"Query: "
        f"{len(query_indices):,}"
    )

    if len(database_indices) != NUM_DATABASE:

        raise RuntimeError(
            f"Expected {NUM_DATABASE} "
            f"database images, got "
            f"{len(database_indices)}"
        )

    if len(query_indices) != NUM_QUERY:

        raise RuntimeError(
            f"Expected {NUM_QUERY} "
            f"query images, got "
            f"{len(query_indices)}"
        )

    overlap = np.intersect1d(
        database_indices,
        query_indices,
    )

    if len(overlap) > 0:

        raise RuntimeError(
            f"Database/query overlap: "
            f"{len(overlap)}"
        )

    print(
        f"Database/query overlap: "
        f"{len(overlap)}"
    )

    return (
        database_indices,
        query_indices,
    )


# ============================================================
# 6. Load EUPE model
# ============================================================

def get_model_entrypoint(
    model_name
):

    mapping = {

        "EUPE-ViT-B":
            "eupe_vitb16",

        "EUPE-ViT-S":
            "eupe_vits16",

        "EUPE-ViT-T":
            "eupe_vitt16",

        "EUPE-ConvNeXt-B":
            "eupe_convnext_base",

    }

    if model_name not in mapping:

        raise KeyError(
            f"Unknown model: "
            f"{model_name}"
        )

    return mapping[
        model_name
    ]


def load_eupe_model(
    model_name,
    model_dir,
):

    checkpoint_candidates = [
        os.path.join(
            model_dir,
            f"{model_name}.pt",
        ),
        os.path.join(
            model_dir,
            model_name
            .replace(
                "EUPE-",
                "",
            )
            + ".pt",
        ),
    ]

    checkpoint = None

    for path in checkpoint_candidates:

        if os.path.isfile(path):

            checkpoint = path

            break

    # --------------------------------------------------------
    # Your naming is:
    #
    # EUPE-ViT-B/
    #     EUPE-ViT-B.pt
    #
    # EUPE-ConvNeXt-B/
    #     EUPE-ConvNeXt-B.pt
    # --------------------------------------------------------

    if checkpoint is None:

        pts = [
            f
            for f in os.listdir(
                model_dir
            )
            if f.endswith(".pt")
        ]

        if len(pts) == 1:

            checkpoint = os.path.join(
                model_dir,
                pts[0],
            )

        else:

            raise FileNotFoundError(
                f"No unique .pt checkpoint "
                f"found in {model_dir}\n"
                f"Found: {pts}"
            )

    entrypoint = (
        get_model_entrypoint(
            model_name
        )
    )

    print()
    print("=" * 90)
    print(
        f"Loading {model_name}"
    )
    print("=" * 90)

    print(
        f"Entrypoint: "
        f"{entrypoint}"
    )

    print(
        f"Checkpoint: "
        f"{checkpoint}"
    )

    # --------------------------------------------------------
    # Official EUPE torch.hub interface
    # --------------------------------------------------------

    model = torch.hub.load(
        REPO_DIR,
        entrypoint,
        source="local",
        weights=checkpoint,
        pretrained=True,
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    return model


# ============================================================
# 7. EUPE feature extraction
# ============================================================

@torch.inference_mode()
def forward_eupe(
    model,
    images,
    model_name,
):

    """
    Official EUPE ViT:
        outputs["x_norm_clstoken"]

    Official EUPE ConvNeXt:
        obtain the final global representation.

    We normalize the resulting feature outside this function.
    """

    # --------------------------------------------------------
    # ViT
    # --------------------------------------------------------

    if model_name.startswith(
        "EUPE-ViT"
    ):

        outputs = (
            model.forward_features(
                images
            )
        )

        if isinstance(
            outputs,
            dict,
        ):

            if (
                "x_norm_clstoken"
                in outputs
            ):

                features = outputs[
                    "x_norm_clstoken"
                ]

            elif (
                "x_norm_clstoken"
                in outputs.keys()
            ):

                features = outputs[
                    "x_norm_clstoken"
                ]

            else:

                raise RuntimeError(
                    "EUPE ViT output does not "
                    "contain x_norm_clstoken.\n"
                    f"Keys: "
                    f"{list(outputs.keys())}"
                )

        else:

            raise RuntimeError(
                "Unexpected EUPE ViT "
                "forward_features output: "
                f"{type(outputs)}"
            )

        # [B, 1, D] -> [B, D]

        if features.ndim == 3:

            features = features[
                :, 0
            ]

        return features

    # --------------------------------------------------------
    # ConvNeXt
    # --------------------------------------------------------

    if model_name.startswith(
        "EUPE-ConvNeXt"
    ):

        outputs = (
            model.forward_features(
                images
            )
        )

        # Some EUPE implementations return
        # a tensor directly.
        if torch.is_tensor(
            outputs
        ):

            features = outputs

        elif isinstance(
            outputs,
            dict,
        ):

            possible_keys = [
                "x_norm_clstoken",
                "x_norm_global",
                "x_norm",
                "features",
                "x",
            ]

            features = None

            for key in possible_keys:

                if key in outputs:

                    features = (
                        outputs[key]
                    )

                    break

            if features is None:

                raise RuntimeError(
                    "Could not identify "
                    "EUPE ConvNeXt feature.\n"
                    f"Output keys: "
                    f"{list(outputs.keys())}"
                )

        else:

            raise RuntimeError(
                "Unexpected EUPE ConvNeXt "
                "forward_features output: "
                f"{type(outputs)}"
            )

        # ----------------------------------------------------
        # ConvNeXt output can be:
        #
        # [B,C,H,W]
        # [B,C]
        # [B,1,C]
        # ----------------------------------------------------

        if features.ndim == 4:

            features = (
                features
                .mean(
                    dim=(-2, -1)
                )
            )

        elif features.ndim == 3:

            features = (
                features.mean(
                    dim=1
                )
            )

        elif features.ndim != 2:

            raise RuntimeError(
                f"Unexpected feature shape: "
                f"{features.shape}"
            )

        return features

    raise RuntimeError(
        f"Unsupported model: "
        f"{model_name}"
    )


@torch.inference_mode()
def extract_features(
    model,
    dataloader,
    model_name,
    split_name,
):

    model.eval()

    feature_list = []
    label_list = []

    total = len(
        dataloader.dataset
    )

    processed = 0

    start_time = time.time()

    for batch_idx, (
        images,
        labels,
    ) in enumerate(
        dataloader
    ):

        images = images.to(
            DEVICE,
            non_blocking=True,
        )

        # ----------------------------------------------------
        # Official EUPE forward_features
        # ----------------------------------------------------

        features = forward_eupe(
            model,
            images,
            model_name,
        )

        features = features.float()

        # ----------------------------------------------------
        # L2 normalization
        # ----------------------------------------------------

        features = F.normalize(
            features,
            p=2,
            dim=-1,
        )

        feature_list.append(
            features.cpu().numpy()
        )

        label_list.append(
            labels.numpy()
        )

        processed += (
            images.shape[0]
        )

        if (
            batch_idx % 20 == 0
            or processed >= total
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

            eta = (
                (
                    total
                    - processed
                )
                / speed
                if speed > 0
                else 0
            )

            print(
                f"[{split_name}] "
                f"{processed:,}/"
                f"{total:,} "
                f"("
                f"{processed / total * 100:.2f}%"
                f") | "
                f"{speed:.1f} img/s | "
                f"ETA "
                f"{eta / 60:.2f} min"
            )

    features = np.concatenate(
        feature_list,
        axis=0,
    )

    labels = np.concatenate(
        label_list,
        axis=0,
    )

    return (
        features.astype(
            np.float32
        ),
        labels.astype(
            np.int64
        ),
    )


# ============================================================
# 8. FLOPs
# ============================================================

def flops_vit(
    image_size,
    patch_size,
    hidden_size,
    depth,
    num_heads,
    mlp_ratio=4.0,
):

    grid = (
        image_size
        // patch_size
    )

    num_patches = (
        grid * grid
    )

    # 1 CLS + patches

    num_tokens = (
        num_patches + 1
    )

    mlp_hidden = int(
        hidden_size
        * mlp_ratio
    )

    # --------------------------------------------------------
    # Patch embedding
    # --------------------------------------------------------

    patch_flops = (
        2
        * num_patches
        * patch_size
        * patch_size
        * 3
        * hidden_size
    )

    # --------------------------------------------------------
    # QKV
    # --------------------------------------------------------

    qkv = (
        2
        * num_tokens
        * hidden_size
        * 3
        * hidden_size
    )

    # --------------------------------------------------------
    # QK^T
    # --------------------------------------------------------

    head_dim = (
        hidden_size
        // num_heads
    )

    attn_qk = (
        2
        * num_heads
        * num_tokens
        * num_tokens
        * head_dim
    )

    # --------------------------------------------------------
    # Attention V
    # --------------------------------------------------------

    attn_v = (
        2
        * num_heads
        * num_tokens
        * num_tokens
        * head_dim
    )

    # --------------------------------------------------------
    # Output projection
    # --------------------------------------------------------

    proj = (
        2
        * num_tokens
        * hidden_size
        * hidden_size
    )

    attention = (
        qkv
        + attn_qk
        + attn_v
        + proj
    )

    # --------------------------------------------------------
    # MLP
    # --------------------------------------------------------

    mlp = (
        2
        * num_tokens
        * hidden_size
        * mlp_hidden
        +
        2
        * num_tokens
        * mlp_hidden
        * hidden_size
    )

    per_layer = (
        attention
        + mlp
    )

    total = (
        patch_flops
        + depth * per_layer
    )

    return total


def flops_convnext_b(
    image_size=256,
):

    """
    ConvNeXt-B:

        depths = [3, 3, 27, 3]
        dims   = [128, 256, 512, 1024]

    We analytically count the major ConvNeXt operations.

    For depthwise 7x7:
        2 * H * W * C * 7 * 7

    Pointwise 1x1:
        2 * H * W * Cin * Cout

    MLP:
        C -> 4C -> C

    Downsampling:
        2 * H * W * Cin * Cout

    LayerNorm / activations are ignored because their FLOPs
    are comparatively small.
    """

    depths = [
        3,
        3,
        27,
        3,
    ]

    dims = [
        128,
        256,
        512,
        1024,
    ]

    total = 0

    # --------------------------------------------------------
    # Stem:
    #
    # Conv 4x4 stride 4
    # 3 -> 128
    # --------------------------------------------------------

    h = (
        image_size
        // 4
    )

    w = h

    total += (
        2
        * h
        * w
        * 3
        * 4
        * 4
        * 128
    )

    # --------------------------------------------------------
    # Stages
    # --------------------------------------------------------

    for stage in range(4):

        c = dims[stage]

        d = depths[stage]

        # ----------------------------------------------------
        # ConvNeXt blocks
        # ----------------------------------------------------

        for _ in range(d):

            # Depthwise 7x7

            total += (
                2
                * h
                * w
                * c
                * 7
                * 7
            )

            # Pointwise C -> 4C

            total += (
                2
                * h
                * w
                * c
                * 4
                * c
            )

            # Pointwise 4C -> C

            total += (
                2
                * h
                * w
                * 4
                * c
                * c
            )

        # ----------------------------------------------------
        # Downsample after stages 0,1,2
        # ----------------------------------------------------

        if stage < 3:

            next_c = dims[
                stage + 1
            ]

            h2 = h // 2
            w2 = w // 2

            total += (
                2
                * h2
                * w2
                * c
                * next_c
                * 2
                * 2
            )

            h = h2
            w = w2

    return total


def calculate_encoder_flops(
    model_name
):

    # --------------------------------------------------------
    # Official EUPE architecture
    #
    # ViT-B:
    #   12 layers
    #   12 heads
    #   dim 768
    #
    # ViT-S:
    #   12 layers
    #   6 heads
    #   dim 384
    #
    # ViT-T:
    #   12 layers
    #   3 heads
    #   dim 192
    #
    # --------------------------------------------------------

    if model_name == "EUPE-ViT-B":

        return flops_vit(
            image_size=IMG_SIZE,
            patch_size=16,
            hidden_size=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
        )

    if model_name == "EUPE-ViT-S":

        return flops_vit(
            image_size=IMG_SIZE,
            patch_size=16,
            hidden_size=384,
            depth=12,
            num_heads=6,
            mlp_ratio=4.0,
        )

    if model_name == "EUPE-ViT-T":

        return flops_vit(
            image_size=IMG_SIZE,
            patch_size=16,
            hidden_size=192,
            depth=12,
            num_heads=3,
            mlp_ratio=4.0,
        )

    if model_name == "EUPE-ConvNeXt-B":

        return flops_convnext_b(
            image_size=IMG_SIZE
        )

    raise KeyError(
        model_name
    )


# ============================================================
# 9. KNN total FLOPs
# ============================================================

def calculate_knn_flops(
    encoder_flops_per_image,
    feature_dim,
):

    encoder_total = (
        encoder_flops_per_image
        * (
            NUM_DATABASE
            + NUM_QUERY
        )
    )

    # Exact cosine/IP similarity:
    #
    # 5000 x 5000 dot products
    #
    # each dot product ~= 2D FLOPs

    similarity_flops = (
        2
        * NUM_DATABASE
        * NUM_QUERY
        * feature_dim
    )

    total = (
        encoder_total
        + similarity_flops
    )

    return {
        "encoder_total_flops":
            encoder_total,

        "knn_similarity_flops":
            similarity_flops,

        "knn_total_flops":
            total,

        "knn_total_tflops":
            total / 1e12,

        "knn_total_pflops":
            total / 1e15,
    }


# ============================================================
# 10. Feature cache
# ============================================================

def get_cache_paths(
    model_name
):

    directory = os.path.join(
        FEATURE_ROOT,
        model_name,
    )

    os.makedirs(
        directory,
        exist_ok=True,
    )

    return {
        "database_features":
            os.path.join(
                directory,
                "database_features.npy",
            ),

        "database_labels":
            os.path.join(
                directory,
                "database_labels.npy",
            ),

        "query_features":
            os.path.join(
                directory,
                "query_features.npy",
            ),

        "query_labels":
            os.path.join(
                directory,
                "query_labels.npy",
            ),
    }


def cache_exists(
    paths
):

    return all(
        os.path.isfile(path)
        for path in paths.values()
    )


def save_cache(
    paths,
    database_features,
    database_labels,
    query_features,
    query_labels,
):

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


# ============================================================
# 11. Weighted KNN
# ============================================================

def run_knn(
    database_features,
    database_labels,
    query_features,
    query_labels,
):

    database_features = (
        np.ascontiguousarray(
            database_features,
            dtype=np.float32,
        )
    )

    query_features = (
        np.ascontiguousarray(
            query_features,
            dtype=np.float32,
        )
    )

    feature_dim = (
        database_features.shape[1]
    )

    print()
    print("=" * 90)
    print("FAISS KNN")
    print("=" * 90)

    print(
        f"Feature dimension: "
        f"{feature_dim}"
    )

    index = faiss.IndexFlatIP(
        feature_dim
    )

    index.add(
        database_features
    )

    search_start = time.time()

    similarities, neighbors = (
        index.search(
            query_features,
            max(K_VALUES),
        )
    )

    search_time = (
        time.time()
        - search_start
    )

    print(
        f"FAISS search time: "
        f"{search_time:.3f} sec"
    )

    results = {}

    for k in K_VALUES:

        sims = similarities[
            :, :k
        ]

        inds = neighbors[
            :, :k
        ]

        labels = (
            database_labels[
                inds
            ]
        )

        # ----------------------------------------------------
        # Weighted vote
        # ----------------------------------------------------

        scaled = (
            sims
            / TEMPERATURE
        )

        scaled -= (
            scaled.max(
                axis=1,
                keepdims=True,
            )
        )

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

            score = {}

            for j in range(k):

                cls = int(
                    labels[i, j]
                )

                weight = float(
                    weights[i, j]
                )

                score[cls] = (
                    score.get(
                        cls,
                        0.0,
                    )
                    + weight
                )

            predictions[i] = max(
                score,
                key=score.get,
            )

        correct = (
            predictions
            == query_labels
        )

        accuracy = (
            correct.mean()
        )

        results[str(k)] = {
            "accuracy":
                float(accuracy),

            "correct":
                int(correct.sum()),

            "total":
                int(len(query_labels)),
        }

        print(
            f"K={k:2d} | "
            f"Accuracy="
            f"{accuracy * 100:.4f}% | "
            f"Correct="
            f"{correct.sum():,}/"
            f"{len(query_labels):,}"
        )

    return (
        results,
        search_time,
    )


# ============================================================
# 12. Run one model
# ============================================================

def run_model(
    model_name,
    model_dir,
    base_dataset,
    database_indices,
    query_indices,
):

    print("\n\n")
    print("#" * 100)
    print(
        f"# {model_name}"
    )
    print("#" * 100)

    checkpoint_files = [
        f
        for f in os.listdir(
            model_dir
        )
        if f.endswith(".pt")
    ]

    if len(checkpoint_files) == 0:

        raise FileNotFoundError(
            f"No .pt checkpoint in "
            f"{model_dir}"
        )

    print(
        f"Checkpoint(s): "
        f"{checkpoint_files}"
    )

    # --------------------------------------------------------
    # Feature cache
    # --------------------------------------------------------

    cache_paths = get_cache_paths(
        model_name
    )

    if cache_exists(
        cache_paths
    ):

        print(
            "\nFeature cache found."
        )

        database_features = np.load(
            cache_paths[
                "database_features"
            ],
            mmap_mode="r",
        )

        database_labels = np.load(
            cache_paths[
                "database_labels"
            ]
        )

        query_features = np.load(
            cache_paths[
                "query_features"
            ],
            mmap_mode="r",
        )

        query_labels = np.load(
            cache_paths[
                "query_labels"
            ]
        )

        # We don't need the model if
        # feature cache already exists.

        model = None

    else:

        model = load_eupe_model(
            model_name,
            model_dir,
        )

        database_dataset = (
            IndexedDataset(
                base_dataset,
                database_indices,
            )
        )

        query_dataset = (
            IndexedDataset(
                base_dataset,
                query_indices,
            )
        )

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
        # Database
        # ----------------------------------------------------

        print()
        print("=" * 90)
        print(
            "Extract DATABASE features"
        )
        print("=" * 90)

        database_features, database_labels = (
            extract_features(
                model=model,
                dataloader=database_loader,
                model_name=model_name,
                split_name="DATABASE",
            )
        )

        # ----------------------------------------------------
        # Query
        # ----------------------------------------------------

        print()
        print("=" * 90)
        print(
            "Extract QUERY features"
        )
        print("=" * 90)

        query_features, query_labels = (
            extract_features(
                model=model,
                dataloader=query_loader,
                model_name=model_name,
                split_name="QUERY",
            )
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        save_cache(
            cache_paths,
            database_features,
            database_labels,
            query_features,
            query_labels,
        )

        print(
            f"\nFeature cache saved to:\n"
            f"{os.path.dirname(cache_paths['database_features'])}"
        )

    # --------------------------------------------------------
    # Feature dimension
    # --------------------------------------------------------

    feature_dim = (
        database_features.shape[1]
    )

    print()
    print(
        f"Database feature shape: "
        f"{database_features.shape}"
    )

    print(
        f"Query feature shape: "
        f"{query_features.shape}"
    )

    print(
        f"Feature dimension: "
        f"{feature_dim}"
    )

    # --------------------------------------------------------
    # Encoder FLOPs
    # --------------------------------------------------------

    encoder_flops = (
        calculate_encoder_flops(
            model_name
        )
    )

    knn_flops = calculate_knn_flops(
        encoder_flops_per_image=(
            encoder_flops
        ),
        feature_dim=feature_dim,
    )

    print()
    print("=" * 90)
    print(
        "FLOPs"
    )
    print("=" * 90)

    print(
        f"Encoder FLOPs/image: "
        f"{encoder_flops / 1e9:.6f} GFLOPs"
    )

    print(
        f"Images encoded: "
        f"{NUM_DATABASE + NUM_QUERY:,}"
    )

    print(
        f"Encoder total: "
        f"{knn_flops['encoder_total_flops'] / 1e12:.6f} TFLOPs"
    )

    print(
        f"KNN similarity: "
        f"{knn_flops['knn_similarity_flops'] / 1e12:.6f} TFLOPs"
    )

    print(
        f"KNN total: "
        f"{knn_flops['knn_total_flops'] / 1e12:.6f} TFLOPs"
    )

    print(
        f"KNN total: "
        f"{knn_flops['knn_total_pflops']:.6f} PFLOPs"
    )

    # --------------------------------------------------------
    # KNN
    # --------------------------------------------------------

    results, search_time = run_knn(
        database_features,
        database_labels,
        query_features,
        query_labels,
    )

    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    result = {

        "model": model_name,

        "model_dir": model_dir,

        "checkpoint": checkpoint_files,

        "protocol": {

            "protocol_file":
                PROTOCOL_PATH,

            "num_classes":
                NUM_CLASSES,

            "database_per_class":
                DATABASE_PER_CLASS,

            "query_per_class":
                QUERY_PER_CLASS,

            "num_database":
                NUM_DATABASE,

            "num_query":
                NUM_QUERY,

            "seed":
                42,
        },

        "preprocessing": {

            "image_size":
                IMG_SIZE,

            "interpolation":
                "bicubic",

            "mean":
                MEAN,

            "std":
                STD,
        },

        "feature": {

            "source":
                (
                    "x_norm_clstoken"
                    if model_name.startswith(
                        "EUPE-ViT"
                    )
                    else "ConvNeXt global feature"
                ),

            "l2_normalized":
                True,

            "dimension":
                int(feature_dim),

            "dtype":
                "float32",
        },

        "knn": {

            "index":
                "FAISS IndexFlatIP",

            "temperature":
                TEMPERATURE,

            "k_values":
                K_VALUES,

            "search_time_seconds":
                search_time,

            "results":
                results,
        },

        "flops": {

            "encoder_flops_per_image":
                int(encoder_flops),

            "encoder_total_flops":
                int(
                    knn_flops[
                        "encoder_total_flops"
                    ]
                ),

            "knn_similarity_flops":
                int(
                    knn_flops[
                        "knn_similarity_flops"
                    ]
                ),

            "knn_total_flops":
                int(
                    knn_flops[
                        "knn_total_flops"
                    ]
                ),

            "knn_total_tflops":
                float(
                    knn_flops[
                        "knn_total_tflops"
                    ]
                ),

            "knn_total_pflops":
                float(
                    knn_flops[
                        "knn_total_pflops"
                    ]
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

    print()
    print("=" * 90)
    print(
        f"{model_name} RESULT"
    )
    print("=" * 90)

    for k in K_VALUES:

        print(
            f"K={k:2d}: "
            f"{results[str(k)]['accuracy'] * 100:.4f}%"
        )

    print(
        f"\nFeature dim: "
        f"{feature_dim}"
    )

    print(
        f"Encoder: "
        f"{encoder_flops / 1e9:.4f} GFLOPs/image"
    )

    print(
        f"KNN total: "
        f"{knn_flops['knn_total_pflops']:.6f} PFLOPs"
    )

    print(
        f"Saved: "
        f"{result_path}"
    )

    # --------------------------------------------------------
    # Release model
    # --------------------------------------------------------

    if model is not None:

        del model

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    return result


# ============================================================
# 13. Main
# ============================================================

def _single_pool_main():

    print()
    print("#" * 100)
    print(
        "# EUPE ImageNet KNN Evaluation"
    )
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
        f"Image size: "
        f"{IMG_SIZE}"
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
        f"Database: "
        f"{NUM_DATABASE:,}"
    )

    print(
        f"Query: "
        f"{NUM_QUERY:,}"
    )

    print(
        f"K: "
        f"{K_VALUES}"
    )

    print(
        f"Temperature: "
        f"{TEMPERATURE}"
    )

    # --------------------------------------------------------
    # Check ImageNet
    # --------------------------------------------------------

    if not os.path.isdir(
        IMAGENET_VAL_ROOT
    ):

        raise FileNotFoundError(
            IMAGENET_VAL_ROOT
        )

    # --------------------------------------------------------
    # Protocol
    # --------------------------------------------------------

    protocol = load_protocol()

    (
        database_indices,
        query_indices,
    ) = extract_indices(
        protocol
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print(
        "Building ImageNet dataset"
    )
    print("=" * 90)

    base_dataset = (
        ImageNetValDataset(
            IMAGENET_VAL_ROOT,
            TRANSFORM,
        )
    )

    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------

    all_results = {}

    total_start = time.time()

    for model_name, model_dir in (
        MODEL_DIRS.items()
    ):

        result = run_model(
            model_name=model_name,
            model_dir=model_dir,
            base_dataset=base_dataset,
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
    # Combined JSON
    # --------------------------------------------------------

    combined_path = os.path.join(
        RESULT_ROOT,
        "EUPE_all_results.json",
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

    print()
    print()
    print("#" * 120)
    print(
        "# FINAL SUMMARY"
    )
    print("#" * 120)

    print(
        f"\n"
        f"{'Model':<22}"
        f"{'K=1':>12}"
        f"{'K=5':>12}"
        f"{'K=10':>12}"
        f"{'K=20':>12}"
        f"{'Dim':>10}"
        f"{'Encoder GFLOPs':>18}"
        f"{'Total PFLOPs':>18}"
    )

    print(
        "-" * 120
    )

    for model_name, result in (
        all_results.items()
    ):

        r = result[
            "knn"
        ][
            "results"
        ]

        dim = result[
            "feature"
        ][
            "dimension"
        ]

        encoder_gflops = (
            result[
                "flops"
            ][
                "encoder_flops_per_image"
            ]
            / 1e9
        )

        total_pflops = result[
            "flops"
        ][
            "knn_total_pflops"
        ]

        print(
            f"{model_name:<22}"
            f"{r['1']['accuracy'] * 100:>11.4f}%"
            f"{r['5']['accuracy'] * 100:>11.4f}%"
            f"{r['10']['accuracy'] * 100:>11.4f}%"
            f"{r['20']['accuracy'] * 100:>11.4f}%"
            f"{dim:>10}"
            f"{encoder_gflops:>18.4f}"
            f"{total_pflops:>18.6f}"
        )

    print(
        "-" * 120
    )

    print(
        f"\nTotal wall time: "
        f"{total_time / 3600:.2f} hours"
    )

    print(
        f"\nCombined result:"
        f"\n{combined_path}"
    )

    print(
        "\nDone."
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
    return run_multishot_from_cache(FEATURE_ROOT, RESULT_ROOT, "eupe")


if __name__ == "__main__":
    main()

