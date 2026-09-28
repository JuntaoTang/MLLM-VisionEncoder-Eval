import os
import sys
import json
import time
import random
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F
import faiss

# The local TokLIP checkpoints use the legacy trusted PyTorch serialization format.
_torch_load = torch.load
def _torch_load_legacy_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load(*args, **kwargs)
torch.load = _torch_load_legacy_compat

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets


# ============================================================
# 1. SETTINGS
# ============================================================

SEED = 42

# ------------------------------------------------------------
# TokLIP repository
# ------------------------------------------------------------

REPO_DIR = "/cache/models/repositories/TokLIP"

if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
TOKLIP_SRC = os.path.join(REPO_DIR, "src")
if TOKLIP_SRC not in sys.path:
    sys.path.insert(0, TOKLIP_SRC)

# TokLIP resolves tokenizer assets relative to its repository root.
os.chdir(REPO_DIR)


# ------------------------------------------------------------
# ImageNet validation
# ------------------------------------------------------------

IMAGENET_ROOT = "/workspace/root/val"


# ------------------------------------------------------------
# Fixed benchmark protocol
# ------------------------------------------------------------

PROTOCOL_PATH = (
    "/cache/metaclip_knn/"
    "val_45shot_5query_seed42_protocol.json"
)


# ------------------------------------------------------------
# TokLIP models
#
# IMPORTANT:
# These names/configs are taken directly from your
# previously working TokLIP loading code.
# ------------------------------------------------------------

MODELS = {

    "TokLIP-S-256": {

        "model":
            "ViT-SO400M-16-SigLIP2-256-toklip",

        "image_size":
            256,

        "checkpoint":
            "/cache/models/model/"
            "toklip_s_256/TokLIP_S_256.pt",

        "batch_size":
            256,
    },

    "TokLIP-L-384": {

        "model":
            "ViT-SO400M-16-SigLIP2-384-toklip",

        "image_size":
            384,

        "checkpoint":
            "/cache/models/model/"
            "toklip_l_384/TokLIP_L_384.pt",

        "batch_size":
            128,
    },
}


# ------------------------------------------------------------
# Cache
# ------------------------------------------------------------

CACHE_ROOT = "/cache/toklip_fixed_knn_cache_45pool"

RESULT_ROOT = "/cache/toklip_fixed_knn_results_45pool"

os.makedirs(
    CACHE_ROOT,
    exist_ok=True
)

os.makedirs(
    RESULT_ROOT,
    exist_ok=True
)


# ------------------------------------------------------------
# Runtime
# ------------------------------------------------------------

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

NUM_WORKERS = 8


# ------------------------------------------------------------
# Feature
# ------------------------------------------------------------

FEATURE_DTYPE = np.float32


# ------------------------------------------------------------
# KNN
# ------------------------------------------------------------

K_VALUES = [
    1,
    5,
    10,
    20,
]

MAX_K = max(K_VALUES)

TEMPERATURE = 0.07


# ------------------------------------------------------------
# FLOPs convention
# ------------------------------------------------------------

# 1 MAC = 2 FLOPs

MAC_TO_FLOPS = 2


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

    # --------------------------------------------------------
    # Same philosophy as previous benchmark code:
    # optimize throughput rather than force deterministic CUDA
    # --------------------------------------------------------

    torch.backends.cudnn.benchmark = True

    torch.backends.cudnn.deterministic = False


# ============================================================
# 3. DATASET
# ============================================================

class SafeImageNetSubset(Dataset):

    def __init__(
        self,
        root,
        indices,
        preprocess,
    ):

        self.dataset = datasets.ImageFolder(
            root=root
        )

        self.indices = np.asarray(
            indices,
            dtype=np.int64
        )

        self.preprocess = preprocess


    def __len__(self):

        return len(self.indices)


    def __getitem__(
        self,
        idx
    ):

        real_idx = int(
            self.indices[idx]
        )

        path, label = (
            self.dataset.samples[
                real_idx
            ]
        )

        try:

            image = Image.open(
                path
            ).convert("RGB")

            image = self.preprocess(
                image
            )

            return (
                image,
                label,
                real_idx
            )

        except Exception as e:

            print()
            print(
                "[WARNING] Failed image:"
            )

            print(
                f"index = {real_idx}"
            )

            print(
                f"path  = {path}"
            )

            print(
                f"error = {repr(e)}"
            )

            return (
                None,
                label,
                real_idx
            )


# ============================================================
# 4. SAFE COLLATE
# ============================================================

def safe_collate(batch):

    valid = []

    for item in batch:

        image, label, index = item

        if image is None:
            continue

        valid.append(
            item
        )

    if len(valid) == 0:

        return None

    images = torch.stack(
        [
            x[0]
            for x in valid
        ],
        dim=0
    )

    labels = torch.tensor(
        [
            x[1]
            for x in valid
        ],
        dtype=torch.long
    )

    indices = torch.tensor(
        [
            x[2]
            for x in valid
        ],
        dtype=torch.long
    )

    return (
        images,
        labels,
        indices
    )


# ============================================================
# 5. LOAD FIXED PROTOCOL
# ============================================================

def load_protocol():

    if not os.path.exists(
        PROTOCOL_PATH
    ):

        raise FileNotFoundError(
            f"\nProtocol not found:\n"
            f"{PROTOCOL_PATH}"
        )

    with open(
        PROTOCOL_PATH,
        "r"
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
    print("=" * 110)
    print("LOADING FIXED KNN PROTOCOL")
    print("=" * 110)

    print(
        f"Protocol: {PROTOCOL_PATH}"
    )


    # --------------------------------------------------------
    # Database
    # --------------------------------------------------------

    database_indices = None

    for key in [

        "train_pool_indices",
        "database_indices",

        "db_indices",

        "train_indices",

    ]:

        if key in protocol:

            database_indices = (
                protocol[key]
            )

            break


    # --------------------------------------------------------
    # Query
    # --------------------------------------------------------

    query_indices = None

    for key in [

        "query_indices",

        "val_indices",

        "test_indices",

    ]:

        if key in protocol:

            query_indices = (
                protocol[key]
            )

            break


    if database_indices is None:

        raise KeyError(
            "Cannot find database indices "
            "in protocol."
        )


    if query_indices is None:

        raise KeyError(
            "Cannot find query indices "
            "in protocol."
        )


    database_indices = np.asarray(
        [
            int(x)
            for x in database_indices
        ],
        dtype=np.int64
    )

    query_indices = np.asarray(
        [
            int(x)
            for x in query_indices
        ],
        dtype=np.int64
    )


    overlap = np.intersect1d(
        database_indices,
        query_indices
    )


    print(
        f"Database size : "
        f"{len(database_indices):,}"
    )

    print(
        f"Query size    : "
        f"{len(query_indices):,}"
    )

    print(
        f"Overlap       : "
        f"{len(overlap):,}"
    )


    if len(overlap) != 0:

        raise RuntimeError(
            "Database/query overlap detected!"
        )


    print()

    return (
        database_indices,
        query_indices
    )


# ============================================================
# 6. BUILD DATALOADER
# ============================================================

def build_loader(
    root,
    indices,
    preprocess,
    batch_size,
):

    dataset = SafeImageNetSubset(
        root=root,
        indices=indices,
        preprocess=preprocess,
    )


    loader = DataLoader(

        dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=True,

        persistent_workers=(
            NUM_WORKERS > 0
        ),

        drop_last=False,

        collate_fn=safe_collate,
    )


    return loader


# ============================================================
# 7. LOAD TOKLIP
# ============================================================

def load_toklip_model(
    config
):

    checkpoint_path = (
        config["checkpoint"]
    )

    model_name = (
        config["model"]
    )

    image_size = (
        config["image_size"]
    )


    if not os.path.exists(
        checkpoint_path
    ):

        raise FileNotFoundError(
            checkpoint_path
        )


    print()
    print("=" * 110)
    print("LOADING TOKLIP")
    print("=" * 110)

    print(
        f"Model      : {model_name}"
    )

    print(
        f"Image size : {image_size}"
    )

    print(
        f"Checkpoint : {checkpoint_path}"
    )


    # --------------------------------------------------------
    # IMPORTANT:
    #
    # This is the SAME loading mechanism from your previous
    # working TokLIP code.
    #
    # Do NOT replace this with:
    #
    # open_clip.create_model_and_transforms(
    #     "ViT-B-32", ...
    # )
    #
    # TokLIP is not a generic ViT-B/32 checkpoint.
    # --------------------------------------------------------

    from create_toklip import create_toklip


    model, _, preprocess_val = (
        create_toklip(

            model=model_name,

            model_path=checkpoint_path,

            image_size=image_size,

            device=DEVICE,
        )
    )


    model.eval()


    print()
    print(
        "Model loaded successfully."
    )

    print(
        "Device:",
        next(
            model.parameters()
        ).device
    )


    if hasattr(
        model.visual,
        "image_size"
    ):

        print(
            "model.visual.image_size:",
            model.visual.image_size
        )


    print(
        "Preprocessing: "
        "official TokLIP preprocess_val"
    )


    return (
        model,
        preprocess_val
    )


# ============================================================
# 8. FEATURE EXTRACTION
# ============================================================

@torch.inference_mode()
def extract_features(
    model,
    loader,
    split_name,
):

    model.eval()


    feature_list = []

    label_list = []

    index_list = []


    total = len(
        loader.dataset
    )

    processed = 0

    failed = 0


    # --------------------------------------------------------
    # CUDA synchronization for accurate timing
    # --------------------------------------------------------

    if torch.cuda.is_available():

        torch.cuda.synchronize()


    start_time = time.time()


    print()
    print("=" * 110)

    print(
        f"EXTRACT {split_name}"
    )

    print("=" * 110)


    for batch_idx, batch in enumerate(
        loader
    ):

        if batch is None:

            failed += 1

            continue


        images, labels, indices = batch


        images = images.to(
            DEVICE,
            non_blocking=True
        )


        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Use the actual TokLIP image encoder.
        #
        # Do NOT call generic ViT code.
        #
        # Do NOT model.half().
        #
        # TokLIP VQ contains FP32 components.
        # ----------------------------------------------------

        features = model.encode_image(
            images
        )


        # ----------------------------------------------------
        # Make sure output is [B, D]
        # ----------------------------------------------------

        if features.ndim > 2:

            features = features.flatten(
                start_dim=1
            )


        features = features.float()


        # ----------------------------------------------------
        # L2 normalization
        #
        # Same as SigLIP / DINOv2 benchmark protocol.
        # ----------------------------------------------------

        features = F.normalize(
            features,
            p=2,
            dim=-1
        )


        features = features.float().cpu()


        feature_list.append(
            features
        )

        label_list.append(
            labels.cpu()
        )

        index_list.append(
            indices.cpu()
        )


        processed += len(images)


        elapsed = (
            time.time()
            - start_time
        )


        speed = (
            processed
            / max(
                elapsed,
                1e-8
            )
        )


        eta = (
            total - processed
        ) / max(
            speed,
            1e-8
        )


        if (
            batch_idx == 0
            or batch_idx % 10 == 0
            or processed >= total
        ):

            print(
                f"\r[{split_name}] "
                f"{processed:,}/{total:,} "
                f"| {speed:.2f} img/s "
                f"| ETA {eta / 60:.2f} min",
                end="",
                flush=True
            )


    print()


    if len(feature_list) == 0:

        raise RuntimeError(
            f"No features extracted for "
            f"{split_name}"
        )


    features = torch.cat(
        feature_list,
        dim=0
    ).numpy().astype(
        FEATURE_DTYPE
    )


    labels = torch.cat(
        label_list,
        dim=0
    ).numpy()


    extracted_indices = torch.cat(
        index_list,
        dim=0
    ).numpy()


    elapsed = (
        time.time()
        - start_time
    )


    speed = (
        len(features)
        / max(
            elapsed,
            1e-8
        )
    )


    print()
    print(
        f"{split_name} finished:"
    )

    print(
        f"  Features : {features.shape}"
    )

    print(
        f"  Labels   : {labels.shape}"
    )

    print(
        f"  Indices  : {extracted_indices.shape}"
    )

    print(
        f"  Failed   : {failed}"
    )

    print(
        f"  Time     : {elapsed:.2f} sec"
    )

    print(
        f"  Speed    : {speed:.2f} img/s"
    )


    # --------------------------------------------------------
    # Sanity check
    # --------------------------------------------------------

    finite = np.isfinite(
        features
    ).all()


    norms = np.linalg.norm(
        features,
        axis=1
    )


    print()
    print(
        "Feature sanity:"
    )

    print(
        f"  Finite    : {finite}"
    )

    print(
        f"  Norm mean : {norms.mean():.8f}"
    )

    print(
        f"  Norm std  : {norms.std():.8e}"
    )

    print(
        f"  Norm min  : {norms.min():.8f}"
    )

    print(
        f"  Norm max  : {norms.max():.8f}"
    )


    if not finite:

        raise RuntimeError(
            f"{split_name} contains "
            "NaN/Inf features."
        )


    return (
        features,
        labels,
        extracted_indices,
        elapsed,
    )


# ============================================================
# 9. TOKLIP ARCHITECTURE INSPECTION
# ============================================================

def first_existing_attr(
    obj,
    names,
):

    for name in names:

        if hasattr(
            obj,
            name
        ):

            value = getattr(
                obj,
                name
            )

            if value is not None:

                return value

    return None


def get_module_by_names(
    root,
    names,
):

    for name in names:

        module = getattr(
            root,
            name,
            None
        )

        if module is not None:

            return module


    return None


def infer_toklip_vit_config(
    model,
    requested_image_size,
):

    """
    Infer the ACTUAL TokLIP visual transformer architecture
    from the loaded model.

    This replaces the old generic OpenCLIP ViT-B/32 inference.

    It specifically supports structures such as:

        model.visual.trunk
        trunk.blocks
        block.attn.qkv
        block.mlp.fc1

    and standard OpenCLIP-style structures as fallback.
    """


    visual = model.visual


    print()
    print("=" * 110)
    print("INSPECTING TOKLIP VISUAL ARCHITECTURE")
    print("=" * 110)


    print(
        "visual type:",
        type(visual).__name__
    )


    # ========================================================
    # 9.1 Determine transformer/trunk
    # ========================================================

    trunk = get_module_by_names(
        visual,
        [
            "trunk",
            "transformer",
        ]
    )


    if trunk is None:

        raise RuntimeError(
            "Cannot find TokLIP visual trunk/"
            "transformer."
        )


    print(
        "trunk type:",
        type(trunk).__name__
    )


    # ========================================================
    # 9.2 Image size
    # ========================================================

    image_size = first_existing_attr(
        visual,
        [
            "image_size",
            "input_resolution",
        ]
    )


    if image_size is None:

        image_size = first_existing_attr(
            trunk,
            [
                "image_size",
                "input_resolution",
            ]
        )


    if image_size is None:

        image_size = requested_image_size


    if isinstance(
        image_size,
        (tuple, list)
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
    # IMPORTANT:
    #
    # The old failed code printed:
    #
    # model.visual.image_size = (224, 224)
    #
    # even though the checkpoint was TokLIP-S-256.
    #
    # Therefore, for FLOPs, trust the explicitly loaded
    # TokLIP configuration requested by create_toklip()
    # when the visual object reports a suspicious resolution.
    #
    # We validate against the requested model image_size.
    # --------------------------------------------------------

    if (
        requested_image_size is not None
        and
        (
            image_height
            != int(requested_image_size)
            or
            image_width
            != int(requested_image_size)
        )
    ):

        print()
        print(
            "[INFO] visual.image_size reports "
            f"{image_height}x{image_width}, "
            f"but TokLIP config requests "
            f"{requested_image_size}x{requested_image_size}."
        )

        print(
            "[INFO] Using requested TokLIP "
            "input resolution for FLOPs."
        )

        image_height = int(
            requested_image_size
        )

        image_width = int(
            requested_image_size
        )


    # ========================================================
    # 9.3 Blocks
    # ========================================================

    blocks = first_existing_attr(
        trunk,
        [
            "blocks",
            "resblocks",
        ]
    )


    if blocks is None:

        blocks = first_existing_attr(
            visual,
            [
                "blocks",
                "resblocks",
            ]
        )


    if blocks is None:

        raise RuntimeError(
            "Cannot find transformer blocks "
            "in TokLIP visual encoder."
        )


    try:

        num_layers = len(
            blocks
        )

    except Exception:

        num_layers = first_existing_attr(
            trunk,
            [
                "layers",
                "num_layers",
                "depth",
            ]
        )


        if num_layers is None:

            raise RuntimeError(
                "Cannot determine TokLIP "
                "transformer depth."
            )


        num_layers = int(
            num_layers
        )


    # ========================================================
    # 9.4 First block
    # ========================================================

    block0 = blocks[0]


    print(
        "block type:",
        type(block0).__name__
    )


    # ========================================================
    # 9.5 Hidden dimension
    # ========================================================

    hidden_size = None


    # --------------------------------------------------------
    # Try model attributes
    # --------------------------------------------------------

    hidden_size = first_existing_attr(
        trunk,
        [
            "width",
            "embed_dim",
            "hidden_size",
            "dim",
        ]
    )


    if hidden_size is None:

        hidden_size = first_existing_attr(
            visual,
            [
                "width",
                "embed_dim",
                "hidden_size",
            ]
        )


    # --------------------------------------------------------
    # Infer from attention QKV
    # --------------------------------------------------------

    attn = first_existing_attr(
        block0,
        [
            "attn",
            "attention",
        ]
    )


    qkv = None


    if attn is not None:

        qkv = first_existing_attr(
            attn,
            [
                "qkv",
            ]
        )


    if (
        hidden_size is None
        and
        qkv is not None
        and
        hasattr(qkv, "weight")
    ):

        qkv_out = (
            qkv.weight.shape[0]
        )

        if qkv_out % 3 == 0:

            hidden_size = (
                qkv_out // 3
            )


    # --------------------------------------------------------
    # Infer from block norm
    # --------------------------------------------------------

    if hidden_size is None:

        for name in [
            "norm1",
            "ln_1",
            "norm",
        ]:

            norm = getattr(
                block0,
                name,
                None
            )

            if (
                norm is not None
                and
                hasattr(norm, "normalized_shape")
            ):

                shape = norm.normalized_shape

                if isinstance(
                    shape,
                    (tuple, list)
                ):

                    hidden_size = int(
                        shape[-1]
                    )

                else:

                    hidden_size = int(
                        shape
                    )

                break


    if hidden_size is None:

        raise RuntimeError(
            "Cannot determine TokLIP "
            "hidden dimension."
        )


    hidden_size = int(
        hidden_size
    )


    # ========================================================
    # 9.6 Attention heads
    # ========================================================

    heads = None


    if attn is not None:

        heads = first_existing_attr(
            attn,
            [
                "num_heads",
                "heads",
                "n_heads",
            ]
        )


    if heads is None:

        heads = first_existing_attr(
            block0,
            [
                "num_heads",
                "heads",
                "n_heads",
            ]
        )


    if heads is None:

        heads = first_existing_attr(
            trunk,
            [
                "num_heads",
                "heads",
                "n_heads",
            ]
        )


    # --------------------------------------------------------
    # Infer using head_dim
    # --------------------------------------------------------

    if heads is None and attn is not None:

        head_dim = first_existing_attr(
            attn,
            [
                "head_dim",
            ]
        )

        if (
            head_dim is not None
            and
            int(head_dim) > 0
        ):

            if (
                hidden_size
                % int(head_dim)
                == 0
            ):

                heads = (
                    hidden_size
                    // int(head_dim)
                )


    # --------------------------------------------------------
    # Infer from qkv and common TokLIP dimensions
    #
    # This is only a fallback. We do NOT guess blindly.
    # --------------------------------------------------------

    if heads is None:

        # Common SigLIP SO400M attention uses 16 heads.
        # However, only use this if architecture metadata
        # strongly indicates the SO400M family.
        model_string = str(
            type(trunk).__name__
        ).lower()

        visual_string = str(
            type(visual).__name__
        ).lower()

        combined = (
            model_string
            + " "
            + visual_string
        ).lower()


        if (
            "siglip" in combined
            and hidden_size == 1152
        ):

            heads = 16


    if heads is None:

        raise RuntimeError(
            "\nCannot determine TokLIP attention heads.\n"
            "The model itself was loaded, but its attention "
            "metadata is not exposed in a standard location.\n"
            "Please inspect block0.attn."
        )


    heads = int(
        heads
    )


    # ========================================================
    # 9.7 MLP dimension
    # ========================================================

    mlp_dim = None


    mlp = first_existing_attr(
        block0,
        [
            "mlp",
            "ffn",
        ]
    )


    if mlp is not None:

        fc1 = first_existing_attr(
            mlp,
            [
                "fc1",
                "c_fc",
                "w1",
            ]
        )


        if (
            fc1 is not None
            and
            hasattr(
                fc1,
                "out_features"
            )
        ):

            mlp_dim = int(
                fc1.out_features
            )


        # ----------------------------------------------------
        # Sequential MLP
        # ----------------------------------------------------

        if mlp_dim is None:

            try:

                for module in mlp:

                    if hasattr(
                        module,
                        "out_features"
                    ):

                        mlp_dim = int(
                            module.out_features
                        )

                        break

            except Exception:

                pass


    # --------------------------------------------------------
    # Infer from state dimensions
    # --------------------------------------------------------

    if mlp_dim is None:

        if mlp is not None:

            for module in mlp.modules():

                if (
                    hasattr(
                        module,
                        "weight"
                    )
                    and
                    module.weight.ndim == 2
                ):

                    out_dim = (
                        module.weight.shape[0]
                    )

                    in_dim = (
                        module.weight.shape[1]
                    )

                    if (
                        in_dim == hidden_size
                        and
                        out_dim != hidden_size
                    ):

                        mlp_dim = int(
                            out_dim
                        )

                        break


    if mlp_dim is None:

        # SigLIP SO400M convention:
        # 4304 hidden dimension.
        #
        # Only use when the actual model indicates
        # hidden size 1152.
        if hidden_size == 1152:

            mlp_dim = 4304


    if mlp_dim is None:

        raise RuntimeError(
            "Cannot determine TokLIP MLP dimension."
        )


    mlp_dim = int(
        mlp_dim
    )


    # ========================================================
    # 9.8 Patch size
    # ========================================================

    patch_size = None


    # --------------------------------------------------------
    # Standard conv patch embed
    # --------------------------------------------------------

    conv1 = getattr(
        visual,
        "conv1",
        None
    )


    if conv1 is not None:

        if hasattr(
            conv1,
            "kernel_size"
        ):

            kernel = (
                conv1.kernel_size
            )

            if isinstance(
                kernel,
                tuple
            ):

                patch_size = int(
                    kernel[0]
                )

            else:

                patch_size = int(
                    kernel
                )


    # --------------------------------------------------------
    # Patch embed module
    # --------------------------------------------------------

    if patch_size is None:

        patch_embed = first_existing_attr(
            visual,
            [
                "patch_embed",
            ]
        )


        if patch_embed is None:

            patch_embed = first_existing_attr(
                trunk,
                [
                    "patch_embed",
                ]
            )


        if patch_embed is not None:

            proj = first_existing_attr(
                patch_embed,
                [
                    "proj",
                    "projection",
                ]
            )


            if proj is not None:

                if hasattr(
                    proj,
                    "kernel_size"
                ):

                    kernel = (
                        proj.kernel_size
                    )

                    if isinstance(
                        kernel,
                        tuple
                    ):

                        patch_size = int(
                            kernel[0]
                        )

                    else:

                        patch_size = int(
                            kernel
                        )


    # --------------------------------------------------------
    # Fallback based on TokLIP model name
    # --------------------------------------------------------

    if patch_size is None:

        patch_size = 16


    patch_size = int(
        patch_size
    )


    # ========================================================
    # 9.9 Patch/token count
    # ========================================================

    if (
        image_height
        % patch_size
        != 0
        or
        image_width
        % patch_size
        != 0
    ):

        raise RuntimeError(
            f"Image size {image_height}x{image_width} "
            f"is not divisible by patch size {patch_size}."
        )


    grid_h = (
        image_height
        // patch_size
    )

    grid_w = (
        image_width
        // patch_size
    )


    num_patches = (
        grid_h
        * grid_w
    )


    # --------------------------------------------------------
    # Determine whether CLS token exists.
    #
    # SigLIP normally does NOT use CLS token for the
    # transformer sequence.
    #
    # We inspect positional embedding shape where possible.
    # --------------------------------------------------------

    num_tokens = num_patches


    pos_embed = first_existing_attr(
        trunk,
        [
            "pos_embed",
        ]
    )


    if pos_embed is None:

        pos_embed = first_existing_attr(
            visual,
            [
                "positional_embedding",
                "pos_embed",
            ]
        )


    if pos_embed is not None:

        if hasattr(
            pos_embed,
            "shape"
        ):

            pos_shape = tuple(
                pos_embed.shape
            )


            if len(pos_shape) >= 2:

                pos_tokens = (
                    pos_shape[-2]
                )


                if (
                    pos_tokens
                    == num_patches + 1
                ):

                    num_tokens = (
                        num_patches + 1
                    )

                elif (
                    pos_tokens
                    == num_patches
                ):

                    num_tokens = (
                        num_patches
                    )


    # --------------------------------------------------------
    # If visual pooling indicates CLS token, detect it.
    # --------------------------------------------------------

    class_token = first_existing_attr(
        visual,
        [
            "class_embedding",
            "cls_token",
            "class_token",
        ]
    )


    if class_token is not None:

        num_tokens = (
            num_patches + 1
        )


    # ========================================================
    # 9.10 Print architecture
    # ========================================================

    print()
    print(
        "Detected TokLIP architecture:"
    )

    print(
        f"  Image size       : "
        f"{image_height} x {image_width}"
    )

    print(
        f"  Patch size       : "
        f"{patch_size}"
    )

    print(
        f"  Patch grid       : "
        f"{grid_h} x {grid_w}"
    )

    print(
        f"  Num patches      : "
        f"{num_patches}"
    )

    print(
        f"  Num tokens       : "
        f"{num_tokens}"
    )

    print(
        f"  Hidden size      : "
        f"{hidden_size}"
    )

    print(
        f"  Transformer depth: "
        f"{num_layers}"
    )

    print(
        f"  Attention heads  : "
        f"{heads}"
    )

    print(
        f"  MLP dimension    : "
        f"{mlp_dim}"
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
            heads,

        "mlp_dim":
            mlp_dim,
    }


# ============================================================
# 10. TOKLIP FEATURE FLOPs
# ============================================================

def calculate_toklip_flops(
    config
):

    """
    Structural FLOPs estimate for the TokLIP visual transformer.

    Convention:
        1 MAC = 2 FLOPs

    Included:
        - patch embedding
        - QKV projection
        - attention output projection
        - QK^T
        - AV
        - MLP

    Not included:
        - LayerNorm
        - activation
        - softmax
        - bias
        - residual additions
        - positional embedding additions
        - final normalization

    These non-MAC operations are tiny relative to the
    matrix multiplications and are omitted consistently
    with the other ViT FLOPs calculations.
    """


    H = int(
        config["image_height"]
    )

    W = int(
        config["image_width"]
    )

    P = int(
        config["patch_size"]
    )

    N = int(
        config["num_tokens"]
    )

    D = int(
        config["hidden_size"]
    )

    L = int(
        config["num_layers"]
    )

    M = int(
        config["mlp_dim"]
    )

    num_patches = int(
        config["num_patches"]
    )


    # ========================================================
    # Patch embedding
    # ========================================================

    patch_embed_macs = (
        num_patches
        * P
        * P
        * 3
        * D
    )


    # ========================================================
    # Attention projections
    #
    # Q + K + V + output
    #
    # 4 * N * D^2
    # ========================================================

    qkv_output_macs = (
        4
        * N
        * D
        * D
    )


    # ========================================================
    # Attention QK^T
    # ========================================================

    attention_qk_macs = (
        N
        * N
        * D
    )


    # ========================================================
    # Attention AV
    # ========================================================

    attention_av_macs = (
        N
        * N
        * D
    )


    # ========================================================
    # MLP
    #
    # fc1 + fc2
    #
    # 2 * N * D * M
    # ========================================================

    mlp_macs = (
        2
        * N
        * D
        * M
    )


    # ========================================================
    # Per transformer block
    # ========================================================

    block_macs = (
        qkv_output_macs
        + attention_qk_macs
        + attention_av_macs
        + mlp_macs
    )


    # ========================================================
    # All transformer blocks
    # ========================================================

    transformer_macs = (
        L
        * block_macs
    )


    # ========================================================
    # Total
    # ========================================================

    total_macs = (
        patch_embed_macs
        + transformer_macs
    )


    total_flops = (
        total_macs
        * MAC_TO_FLOPS
    )


    return {

        **config,

        "patch_embed_macs":
            int(patch_embed_macs),

        "qkv_output_macs_per_block":
            int(qkv_output_macs),

        "attention_qk_macs_per_block":
            int(attention_qk_macs),

        "attention_av_macs_per_block":
            int(attention_av_macs),

        "mlp_macs_per_block":
            int(mlp_macs),

        "block_macs":
            int(block_macs),

        "transformer_macs":
            int(transformer_macs),

        "total_macs_per_image":
            int(total_macs),

        "feature_flops_per_image":
            int(total_flops),
    }


# ============================================================
# 11. KNN FLOPs
# ============================================================

def calculate_knn_flops(
    num_database,
    num_query,
    feature_dim,
    max_k,
):

    """
    FAISS IndexFlatIP:

        Q * DB * D MACs

    1 MAC = 2 FLOPs
    """


    similarity_macs = (
        num_query
        * num_database
        * feature_dim
    )


    similarity_flops = (
        similarity_macs
        * MAC_TO_FLOPS
    )


    # --------------------------------------------------------
    # Weighted voting.
    #
    # Approximate 5 scalar FLOPs per neighbor:
    #
    #   subtraction
    #   division
    #   exp
    #   multiplication
    #   accumulation
    #
    # This is tiny compared with feature extraction.
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

        "similarity_macs":
            int(similarity_macs),

        "similarity_flops":
            int(similarity_flops),

        "voting_flops":
            int(voting_flops),

        "total_knn_flops":
            int(total_knn_flops),
    }


# ============================================================
# 12. WEIGHTED KNN
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
    # L2 normalized feature
    #
    # Inner Product = cosine similarity
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
        f"Feature D: {feature_dim}"
    )

    print(
        "Index    : IndexFlatIP"
    )

    print(
        f"Temperature: {TEMPERATURE}"
    )


    # ========================================================
    # Search ONCE with K=20
    # ========================================================

    if torch.cuda.is_available():

        torch.cuda.synchronize()


    start = time.time()


    similarities, indices = (
        index.search(
            query_features,
            MAX_K
        )
    )


    search_time = (
        time.time()
        - start
    )


    print()
    print(
        f"FAISS search time: "
        f"{search_time:.4f} sec"
    )


    results = {}


    # ========================================================
    # Evaluate all K from same top-20 result
    # ========================================================

    for k in K_VALUES:

        start = time.time()


        sims_k = (
            similarities[
                :, :k
            ]
        )


        inds_k = (
            indices[
                :, :k
            ]
        )


        labels_k = (
            database_labels[
                inds_k
            ]
        )


        # ----------------------------------------------------
        # Stable temperature weighting
        # ----------------------------------------------------

        scaled = (
            sims_k
            - sims_k.max(
                axis=1,
                keepdims=True
            )
        ) / TEMPERATURE


        weights = np.exp(
            scaled
        )


        predictions = np.empty(
            len(query_labels),
            dtype=np.int64
        )


        # ----------------------------------------------------
        # Weighted class voting
        # ----------------------------------------------------

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
                    return_inverse=True
                )
            )


            class_weights = (
                np.bincount(
                    inverse,
                    weights=neighbor_weights
                )
            )


            predictions[i] = (
                unique_labels[
                    np.argmax(
                        class_weights
                    )
                ]
            )


        accuracy = (
            predictions
            == query_labels
        ).mean() * 100.0


        voting_time = (
            time.time()
            - start
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
        search_time
    )


# ============================================================
# 13. CACHE PATHS
# ============================================================

def get_cache_paths(
    model_name
):

    model_dir = os.path.join(
        CACHE_ROOT,
        model_name
    )


    os.makedirs(
        model_dir,
        exist_ok=True
    )


    return {

        "database_features":
            os.path.join(
                model_dir,
                "database_features.npy"
            ),

        "database_labels":
            os.path.join(
                model_dir,
                "database_labels.npy"
            ),

        "database_indices":
            os.path.join(
                model_dir,
                "database_indices.npy"
            ),

        "query_features":
            os.path.join(
                model_dir,
                "query_features.npy"
            ),

        "query_labels":
            os.path.join(
                model_dir,
                "query_labels.npy"
            ),

        "query_indices":
            os.path.join(
                model_dir,
                "query_indices.npy"
            ),
    }


# ============================================================
# 14. VALIDATE CACHE
# ============================================================

def cache_valid(
    feature_path,
    label_path,
    index_path,
    expected_size,
):

    if not all(
        os.path.exists(p)
        for p in [
            feature_path,
            label_path,
            index_path,
        ]
    ):

        return False


    try:

        features = np.load(
            feature_path,
            mmap_mode="r"
        )

        labels = np.load(
            label_path,
            mmap_mode="r"
        )

        indices = np.load(
            index_path,
            mmap_mode="r"
        )


        if len(features) != expected_size:
            return False

        if len(labels) != expected_size:
            return False

        if len(indices) != expected_size:
            return False


        return True


    except Exception:

        return False


# ============================================================
# 15. EVALUATE ONE TOKLIP MODEL
# ============================================================

def evaluate_model(
    model_name,
    config,
    database_indices,
    query_indices,
):

    print()
    print()
    print("#" * 110)

    print(
        f"MODEL: {model_name}"
    )

    print("#" * 110)


    print(
        f"TokLIP config : "
        f"{config['model']}"
    )

    print(
        f"Image size    : "
        f"{config['image_size']}"
    )

    print(
        f"Checkpoint    : "
        f"{config['checkpoint']}"
    )

    print(
        f"Device        : "
        f"{DEVICE}"
    )

    print(
        "Precision     : FP32"
    )

    print(
        f"Batch size    : "
        f"{config['batch_size']}"
    )


    # ========================================================
    # Load model
    # ========================================================

    model, preprocess = (
        load_toklip_model(
            config
        )
    )


    # ========================================================
    # Parameters
    # ========================================================

    num_params = sum(
        p.numel()
        for p in model.parameters()
    )


    # ========================================================
    # Infer actual architecture
    # ========================================================

    vit_config = (
        infer_toklip_vit_config(
            model=model,
            requested_image_size=config[
                "image_size"
            ],
        )
    )


    # ========================================================
    # Calculate feature FLOPs
    # ========================================================

    flops_info = (
        calculate_toklip_flops(
            vit_config
        )
    )


    feature_flops_per_image = (
        flops_info[
            "feature_flops_per_image"
        ]
    )


    # ========================================================
    # Model summary
    # ========================================================

    print()
    print("=" * 110)
    print("TOKLIP VISUAL ENCODER")
    print("=" * 110)


    print(
        f"Parameters      : "
        f"{num_params / 1e6:.2f} M"
    )

    print(
        f"Image size      : "
        f"{vit_config['image_height']} x "
        f"{vit_config['image_width']}"
    )

    print(
        f"Patch size      : "
        f"{vit_config['patch_size']}"
    )

    print(
        f"Patch grid      : "
        f"{vit_config['grid_h']} x "
        f"{vit_config['grid_w']}"
    )

    print(
        f"Num patches     : "
        f"{vit_config['num_patches']}"
    )

    print(
        f"Tokens          : "
        f"{vit_config['num_tokens']}"
    )

    print(
        f"Hidden dim      : "
        f"{vit_config['hidden_size']}"
    )

    print(
        f"Layers          : "
        f"{vit_config['num_layers']}"
    )

    print(
        f"Heads           : "
        f"{vit_config['num_heads']}"
    )

    print(
        f"MLP dim         : "
        f"{vit_config['mlp_dim']}"
    )


    print()
    print(
        f"Feature FLOPs/image : "
        f"{feature_flops_per_image / 1e9:.6f} GFLOPs"
    )


    # ========================================================
    # Dataset size
    # ========================================================

    num_database = len(
        database_indices
    )

    num_query = len(
        query_indices
    )

    total_images = (
        num_database
        + num_query
    )


    # ========================================================
    # Feature extraction FLOPs
    #
    # IMPORTANT:
    # This is calculated even when features are loaded from
    # cache, because it represents benchmark computational
    # cost rather than current runtime.
    # ========================================================

    feature_extraction_flops = (
        total_images
        * feature_flops_per_image
    )


    database_feature_flops = (
        num_database
        * feature_flops_per_image
    )


    query_feature_flops = (
        num_query
        * feature_flops_per_image
    )


    # ========================================================
    # Cache
    # ========================================================

    paths = get_cache_paths(
        model_name
    )


    # ========================================================
    # DataLoaders
    # ========================================================

    database_loader = build_loader(
        root=IMAGENET_ROOT,
        indices=database_indices,
        preprocess=preprocess,
        batch_size=config[
            "batch_size"
        ],
    )


    query_loader = build_loader(
        root=IMAGENET_ROOT,
        indices=query_indices,
        preprocess=preprocess,
        batch_size=config[
            "batch_size"
        ],
    )


    # ========================================================
    # Database features
    # ========================================================

    database_cache_ok = cache_valid(

        paths[
            "database_features"
        ],

        paths[
            "database_labels"
        ],

        paths[
            "database_indices"
        ],

        num_database,
    )


    if database_cache_ok:

        print()
        print(
            "Loading cached database features..."
        )


        database_features = np.load(
            paths[
                "database_features"
            ]
        )


        database_labels = np.load(
            paths[
                "database_labels"
            ]
        )


        database_cached_indices = np.load(
            paths[
                "database_indices"
            ]
        )


        # ----------------------------------------------------
        # Make sure cache corresponds exactly to protocol
        # ----------------------------------------------------

        if not np.array_equal(
            database_cached_indices,
            database_indices
        ):

            print(
                "[WARNING] Cached database indices "
                "do not match protocol."
            )

            database_cache_ok = False


    if not database_cache_ok:

        print()
        print("=" * 110)
        print("EXTRACT DATABASE FEATURES")
        print("=" * 110)


        (
            database_features,
            database_labels,
            database_extracted_indices,
            database_extract_time,
        ) = extract_features(

            model=model,

            loader=database_loader,

            split_name="DATABASE",
        )


        np.save(
            paths[
                "database_features"
            ],
            database_features
        )


        np.save(
            paths[
                "database_labels"
            ],
            database_labels
        )


        np.save(
            paths[
                "database_indices"
            ],
            database_extracted_indices
        )


    else:

        database_extract_time = 0.0


    # ========================================================
    # Query features
    # ========================================================

    query_cache_ok = cache_valid(

        paths[
            "query_features"
        ],

        paths[
            "query_labels"
        ],

        paths[
            "query_indices"
        ],

        num_query,
    )


    if query_cache_ok:

        print()
        print(
            "Loading cached query features..."
        )


        query_features = np.load(
            paths[
                "query_features"
            ]
        )


        query_labels = np.load(
            paths[
                "query_labels"
            ]
        )


        query_cached_indices = np.load(
            paths[
                "query_indices"
            ]
        )


        if not np.array_equal(
            query_cached_indices,
            query_indices
        ):

            print(
                "[WARNING] Cached query indices "
                "do not match protocol."
            )

            query_cache_ok = False


    if not query_cache_ok:

        print()
        print("=" * 110)
        print("EXTRACT QUERY FEATURES")
        print("=" * 110)


        (
            query_features,
            query_labels,
            query_extracted_indices,
            query_extract_time,
        ) = extract_features(

            model=model,

            loader=query_loader,

            split_name="QUERY",
        )


        np.save(
            paths[
                "query_features"
            ],
            query_features
        )


        np.save(
            paths[
                "query_labels"
            ],
            query_labels
        )


        np.save(
            paths[
                "query_indices"
            ],
            query_extracted_indices
        )


    else:

        query_extract_time = 0.0


    # ========================================================
    # Validate
    # ========================================================

    if len(database_features) != num_database:

        raise RuntimeError(
            "Database feature count mismatch."
        )


    if len(query_features) != num_query:

        raise RuntimeError(
            "Query feature count mismatch."
        )


    feature_dim = int(
        database_features.shape[1]
    )


    if (
        query_features.shape[1]
        != feature_dim
    ):

        raise RuntimeError(
            "Database/query feature dimensions "
            "do not match."
        )


    # ========================================================
    # Verify L2 normalization
    # ========================================================

    database_norms = np.linalg.norm(
        database_features,
        axis=1
    )


    query_norms = np.linalg.norm(
        query_features,
        axis=1
    )


    print()
    print(
        "=" * 110
    )

    print(
        "FEATURE SUMMARY"
    )

    print(
        "=" * 110
    )

    print(
        f"Database features : "
        f"{database_features.shape}"
    )

    print(
        f"Query features    : "
        f"{query_features.shape}"
    )

    print(
        f"Feature dimension  : "
        f"{feature_dim}"
    )

    print(
        f"Database norm mean : "
        f"{database_norms.mean():.8f}"
    )

    print(
        f"Query norm mean    : "
        f"{query_norms.mean():.8f}"
    )


    # ========================================================
    # KNN FLOPs
    # ========================================================

    knn_flops = (
        calculate_knn_flops(

            num_database=
                num_database,

            num_query=
                num_query,

            feature_dim=
                feature_dim,

            max_k=
                MAX_K,
        )
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


    total_knn_search_flops = (
        knn_flops[
            "total_knn_flops"
        ]
    )


    # ========================================================
    # TOTAL FLOPs
    #
    # Feature extraction
    # +
    # FAISS similarity search
    # +
    # voting
    # ========================================================

    total_flops = (
        feature_extraction_flops
        + total_knn_search_flops
    )


    # ========================================================
    # FLOPs summary
    # ========================================================

    print()
    print("=" * 110)
    print("FLOPs SUMMARY")
    print("=" * 110)


    print(
        "Convention      : "
        "1 MAC = 2 FLOPs"
    )


    print()
    print(
        f"Feature FLOPs/image : "
        f"{feature_flops_per_image / 1e9:.6f} GFLOPs"
    )


    print(
        f"Database features   : "
        f"{database_feature_flops / 1e12:.6f} TFLOPs"
    )


    print(
        f"Query features      : "
        f"{query_feature_flops / 1e12:.6f} TFLOPs"
    )


    print(
        f"Feature extraction  : "
        f"{feature_extraction_flops / 1e15:.9f} PFLOPs"
    )


    print()
    print(
        f"KNN similarity      : "
        f"{similarity_flops / 1e15:.9f} PFLOPs"
    )


    print(
        f"KNN voting          : "
        f"{voting_flops / 1e15:.9f} PFLOPs"
    )


    print(
        f"KNN search+voting   : "
        f"{total_knn_search_flops / 1e15:.9f} PFLOPs"
    )


    print()
    print(
        f"KNN TOTAL FLOPs     : "
        f"{total_flops / 1e15:.9f} PFLOPs"
    )


    # ========================================================
    # Run KNN
    # ========================================================

    (
        knn_results,
        search_time
    ) = weighted_knn(

        database_features=
            database_features,

        database_labels=
            database_labels,

        query_features=
            query_features,

        query_labels=
            query_labels,
    )


    # ========================================================
    # Result JSON
    # ========================================================

    result = {

        "model":
            model_name,

        "toklip_model":
            config["model"],

        "checkpoint":
            config["checkpoint"],

        "repository":
            REPO_DIR,


        # ----------------------------------------------------
        # Protocol
        # ----------------------------------------------------

        "protocol": {

            "protocol_file":
                PROTOCOL_PATH,

            "seed":
                SEED,

            "dataset":
                "ImageNet validation",

            "database_size":
                num_database,

            "query_size":
                num_query,

            "database_per_class":
                5,

            "query_per_class":
                5,

            "database_query_overlap":
                0,

            "feature_normalization":
                "L2",

            "knn_metric":
                "cosine_similarity_via_normalized_IP",

            "index":
                "FAISS IndexFlatIP",

            "temperature":
                TEMPERATURE,

            "k_values":
                K_VALUES,
        },


        # ----------------------------------------------------
        # Runtime
        # ----------------------------------------------------

        "runtime": {

            "device":
                DEVICE,

            "precision":
                "FP32",

            "database_batch_size":
                config["batch_size"],

            "query_batch_size":
                config["batch_size"],

            "num_workers":
                NUM_WORKERS,
        },


        # ----------------------------------------------------
        # Feature
        # ----------------------------------------------------

        "feature": {

            "type":
                "TokLIP image embedding",

            "source":
                "model.encode_image(images)",

            "preprocess":
                "official TokLIP preprocess_val",

            "l2_normalized":
                True,

            "dtype":
                "float32",

            "dimension":
                feature_dim,
        },


        # ----------------------------------------------------
        # Model config
        # ----------------------------------------------------

        "model_config": {

            "parameters":
                int(num_params),

            "parameters_million":
                float(
                    num_params / 1e6
                ),

            "image_height":
                int(
                    vit_config[
                        "image_height"
                    ]
                ),

            "image_width":
                int(
                    vit_config[
                        "image_width"
                    ]
                ),

            "patch_size":
                int(
                    vit_config[
                        "patch_size"
                    ]
                ),

            "grid_h":
                int(
                    vit_config[
                        "grid_h"
                    ]
                ),

            "grid_w":
                int(
                    vit_config[
                        "grid_w"
                    ]
                ),

            "num_patches":
                int(
                    vit_config[
                        "num_patches"
                    ]
                ),

            "num_tokens":
                int(
                    vit_config[
                        "num_tokens"
                    ]
                ),

            "hidden_size":
                int(
                    vit_config[
                        "hidden_size"
                    ]
                ),

            "num_layers":
                int(
                    vit_config[
                        "num_layers"
                    ]
                ),

            "num_heads":
                int(
                    vit_config[
                        "num_heads"
                    ]
                ),

            "mlp_dim":
                int(
                    vit_config[
                        "mlp_dim"
                    ]
                ),
        },


        # ----------------------------------------------------
        # FLOPs
        # ----------------------------------------------------

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
                    database_feature_flops
                ),

            "query_feature_flops":
                int(
                    query_feature_flops
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

            "knn_total_search_flops":
                int(
                    total_knn_search_flops
                ),

            "knn_total_search_pflops":
                float(
                    total_knn_search_flops
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


        # ----------------------------------------------------
        # Timing
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # KNN
        # ----------------------------------------------------

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
    # Save individual JSON
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
    # Final result
    # ========================================================

    print()
    print("#" * 110)

    print(
        f"FINAL RESULT: {model_name}"
    )

    print("#" * 110)


    for k in K_VALUES:

        print(
            f"K={k:2d}: "
            f"{knn_results[k]['accuracy']:.4f}%"
        )


    print()

    print(
        f"Feature extraction : "
        f"{feature_extraction_flops / 1e15:.9f} PFLOPs"
    )

    print(
        f"KNN search+voting  : "
        f"{total_knn_search_flops / 1e15:.9f} PFLOPs"
    )

    print(
        f"KNN TOTAL          : "
        f"{total_flops / 1e15:.9f} PFLOPs"
    )


    print()

    print(
        "Result saved:"
    )

    print(
        result_path
    )


    # ========================================================
    # Release
    # ========================================================

    del model

    del preprocess

    del database_loader

    del query_loader

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

        torch.cuda.ipc_collect()


    return result


# ============================================================
# 16. MAIN
# ============================================================

def _single_pool_main():

    set_seed(
        SEED
    )


    print()
    print("=" * 110)
    print("TokLIP ImageNet Fixed-Protocol KNN Benchmark")
    print("=" * 110)


    print()
    print(
        f"Repository : {REPO_DIR}"
    )

    print(
        f"ImageNet   : {IMAGENET_ROOT}"
    )

    print(
        f"Protocol   : {PROTOCOL_PATH}"
    )

    print(
        f"Cache      : {CACHE_ROOT}"
    )

    print(
        f"Results    : {RESULT_ROOT}"
    )

    print(
        f"Device     : {DEVICE}"
    )

    print(
        "Precision  : FP32"
    )

    print(
        f"Workers    : {NUM_WORKERS}"
    )

    print(
        f"K          : {K_VALUES}"
    )

    print(
        f"Temperature: {TEMPERATURE}"
    )

    print(
        "Metric     : cosine similarity "
        "(L2-normalized IndexFlatIP)"
    )


    if torch.cuda.is_available():

        print()
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        print(
            "CUDA:",
            torch.version.cuda
        )


    # ========================================================
    # Load protocol
    # ========================================================

    (
        database_indices,
        query_indices,
    ) = load_protocol()


    # ========================================================
    # Load ImageNet once for protocol validation
    # ========================================================

    print()
    print("=" * 110)
    print("LOADING IMAGENET VALIDATION")
    print("=" * 110)


    imagenet = datasets.ImageFolder(
        root=IMAGENET_ROOT
    )


    print(
        f"Total images : "
        f"{len(imagenet):,}"
    )

    print(
        f"Classes      : "
        f"{len(imagenet.classes)}"
    )


    # --------------------------------------------------------
    # Verify indices are in range
    # --------------------------------------------------------

    if (
        database_indices.min()
        < 0
        or
        database_indices.max()
        >= len(imagenet)
    ):

        raise RuntimeError(
            "Database protocol indices "
            "are out of ImageNet range."
        )


    if (
        query_indices.min()
        < 0
        or
        query_indices.max()
        >= len(imagenet)
    ):

        raise RuntimeError(
            "Query protocol indices "
            "are out of ImageNet range."
        )


    # --------------------------------------------------------
    # Verify 5-shot / 5-query per class
    # --------------------------------------------------------

    db_labels = np.asarray(
        imagenet.targets
    )[database_indices]


    query_labels = np.asarray(
        imagenet.targets
    )[query_indices]


    db_counts = np.bincount(
        db_labels,
        minlength=len(
            imagenet.classes
        )
    )


    query_counts = np.bincount(
        query_labels,
        minlength=len(
            imagenet.classes
        )
    )


    print()
    print(
        "Protocol class-count check:"
    )

    print(
        f"  DB min/max    : "
        f"{db_counts.min()} / "
        f"{db_counts.max()}"
    )

    print(
        f"  Query min/max : "
        f"{query_counts.min()} / "
        f"{query_counts.max()}"
    )


    if not np.all(
        db_counts == 45
    ):

        raise RuntimeError(
            "Database is not exactly "
            "45 images per class."
        )


    if not np.all(
        query_counts == 5
    ):

        raise RuntimeError(
            "Query is not exactly "
            "5 images per class."
        )


    print(
        "  PASS: 45 DB + 5 Query per class"
    )


    # ========================================================
    # Evaluate all TokLIP models
    # ========================================================

    all_results = {}


    for model_name, config in MODELS.items():

        result = evaluate_model(

            model_name=
                model_name,

            config=
                config,

            database_indices=
                database_indices,

            query_indices=
                query_indices,
        )


        all_results[
            model_name
        ] = result


    # ========================================================
    # Combined JSON
    # ========================================================

    combined_path = os.path.join(
        RESULT_ROOT,
        "toklip_fixed_protocol_all_results.json"
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
    # Final summary
    # ========================================================

    print()
    print()
    print("=" * 150)
    print("TOKLIP FINAL SUMMARY")
    print("=" * 150)


    print(
        f"{'Model':<20}"
        f"{'K=1':>12}"
        f"{'K=5':>12}"
        f"{'K=10':>12}"
        f"{'K=20':>12}"
        f"{'Feature PFLOPs':>20}"
        f"{'KNN PFLOPs':>20}"
        f"{'TOTAL PFLOPs':>20}"
    )


    print("-" * 150)


    for model_name, result in (
        all_results.items()
    ):

        knn = result[
            "knn"
        ]

        flops = result[
            "flops"
        ]


        print(

            f"{model_name:<20}"

            f"{knn['1']['top1_accuracy']:>11.4f}%"

            f"{knn['5']['top1_accuracy']:>11.4f}%"

            f"{knn['10']['top1_accuracy']:>11.4f}%"

            f"{knn['20']['top1_accuracy']:>11.4f}%"

            f"{flops['feature_extraction_pflops']:>20.6f}"

            f"{flops['knn_total_search_pflops']:>20.9f}"

            f"{flops['total_pflops']:>20.6f}"
        )


    print("-" * 150)


    print()
    print(
        "Combined results:"
    )

    print(
        combined_path
    )


# ============================================================
# ENTRY
# ============================================================

if False:  # entry point is defined below
    _single_pool_main()
def main():
    """Extract the 45k pool once, then evaluate cached 5/10/20/45-shot slices."""
    _single_pool_main()
    from multishot_protocol import run_multishot_from_cache
    return run_multishot_from_cache(CACHE_ROOT, RESULT_ROOT, "toklip")


if __name__ == "__main__":
    main()

