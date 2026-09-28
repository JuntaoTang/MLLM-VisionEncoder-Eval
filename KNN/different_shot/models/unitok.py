import os

# ============================================================
# Thread settings
# ============================================================

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import gc
import json
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# ============================================================
# UniTok repo
# ============================================================

REPO_DIR = "/cache/models/repositories/UniTok"

if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from models.unitok import UniTok
from utils.config import Args

# Compatibility with newer timm factory kwargs.
from models import vitamin as _unitok_vitamin
_hybrid_embed_init = _unitok_vitamin.HybridEmbed.__init__
def _hybrid_embed_init_compat(self, *args, **kwargs):
    kwargs.pop("device", None)
    kwargs.pop("dtype", None)
    return _hybrid_embed_init(self, *args, **kwargs)
_unitok_vitamin.HybridEmbed.__init__ = _hybrid_embed_init_compat

_geglu_init = _unitok_vitamin.GeGluMlp.__init__
def _geglu_init_compat(self, *args, **kwargs):
    for key in ("norm_layer", "bias", "device", "dtype"):
        kwargs.pop(key, None)
    return _geglu_init(self, *args, **kwargs)
_unitok_vitamin.GeGluMlp.__init__ = _geglu_init_compat



# ============================================================
# Paths
# ============================================================

CKPT_PATH = (
    "/cache/models/model/"
    "unitok_attn/unitok_tokenizer.pth"
)

TRAIN_PATH = (
    "/workspace/root/val"
)

# ------------------------------------------------------------
# IMPORTANT:
# This is the SAME fixed protocol used by the previous
# MetaCLIP / SigLIP / PE / UniAR experiments.
#
# 5,000 database + 5,000 query
# seed = 42
# no overlap
# ------------------------------------------------------------

PROTOCOL_FILE = (
    "/cache/metaclip_knn/"
    "val_45shot_5query_seed42_protocol.json"
)

CACHE_DIR = (
    "/cache/unitok_knn_cache_45pool"
)

os.makedirs(
    CACHE_DIR,
    exist_ok=True,
)


# ============================================================
# Settings
# ============================================================

SEED = 42

K_VALUES = [1, 5, 10, 20]

BATCH_SIZE = 256
NUM_WORKERS = 8

STORE_DTYPE = np.float16

FAISS_USE_GPU = True

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Representation
#
# encoder:
#   mean pooled encoder tokens
#
# quantized:
#   quantized tokenizer representation
#
# final:
#   model.encode_image()
#
# For a tokenizer comparison, you can run all three.
# ============================================================

REPRESENTATION = "final"

assert REPRESENTATION in [
    "encoder",
    "quantized",
    "final",
]


# ============================================================
# Transform
#
# SAME protocol as previous models
# ============================================================

IMG_SIZE = 256

RESIZE_SIZE = int(
    IMG_SIZE * 1.125
)

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
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    ),
])


# ============================================================
# Safe ImageFolder
# ============================================================

class SafeImageFolder(
    datasets.ImageFolder
):

    def __init__(
        self,
        root,
        transform=None,
    ):

        print()
        print("=" * 100)
        print("Loading ImageNet")
        print("=" * 100)

        super().__init__(
            root=root,
            transform=transform,
        )

        print(
            f"Images : {len(self):,}"
        )

        print(
            f"Classes: {len(self.classes)}"
        )


# ============================================================
# Load UniTok
#
# EXACTLY follows user's original UniTok
# checkpoint-loading method.
# ============================================================

def load_model():

    print()
    print("=" * 100)
    print("Loading UniTok")
    print("=" * 100)

    print(
        f"Checkpoint:\n{CKPT_PATH}"
    )

    ckpt = torch.load(
        CKPT_PATH,
        map_location="cpu",
    )

    # --------------------------------------------------------
    # Load model args from checkpoint
    # --------------------------------------------------------

    model_args = Args()

    model_args.load_state_dict(
        ckpt["args"]
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = UniTok(
        model_args
    )

    # --------------------------------------------------------
    # Load trainer.unitok
    # --------------------------------------------------------

    state_dict = (
        ckpt["trainer"]["unitok"]
    )

    new_state_dict = {}

    for k, v in state_dict.items():

        if k.startswith("module."):

            k = k[len("module."):]

        new_state_dict[k] = v

    missing, unexpected = (
        model.load_state_dict(
            new_state_dict,
            strict=False,
        )
    )

    print(
        f"Missing keys    : "
        f"{len(missing)}"
    )

    print(
        f"Unexpected keys : "
        f"{len(unexpected)}"
    )

    # --------------------------------------------------------
    # Critical modules
    # --------------------------------------------------------

    for c in [
        "encoder",
        "quantizer",
        "quant_proj",
    ]:

        critical_missing = [
            k for k in missing
            if c in k
        ]

        if critical_missing:

            raise RuntimeError(
                f"Critical module {c} missing: "
                f"{critical_missing[:10]}"
            )

    model = (
        model
        .to(DEVICE)
        .eval()
    )

    print()
    print(
        "UniTok loaded successfully."
    )

    return model


# ============================================================
# Load fixed protocol
# ============================================================

def load_protocol():

    print()
    print("=" * 100)
    print("Loading fixed KNN protocol")
    print("=" * 100)

    print(
        f"Protocol:\n{PROTOCOL_FILE}"
    )

    with open(
        PROTOCOL_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        protocol = json.load(f)


        # Canonical fixed 45-pool/5-query protocol plus legacy aliases used below.

        protocol.setdefault("train_indices", protocol["train_pool_indices"])

        protocol.setdefault("database_indices", protocol["train_pool_indices"])

        protocol.setdefault("db_indices", protocol["train_pool_indices"])

        protocol.setdefault("num_train", 45000)

        protocol.setdefault("train_per_class", 45)

        protocol.setdefault("images_per_class", 50)

    return protocol


# ============================================================
# Extract indices from protocol
# ============================================================

def get_protocol_indices(
    protocol,
    dataset_size,
):

    # --------------------------------------------------------
    # Expected protocol:
    #
    # train_indices
    # query_indices
    #
    # Both are indices into ImageNet validation set.
    # --------------------------------------------------------

    if (
        "train_indices" not in protocol
        or
        "query_indices" not in protocol
    ):

        raise KeyError(
            "Protocol must contain "
            "'train_indices' and "
            "'query_indices'."
        )

    database_indices = np.asarray(
        protocol["train_indices"],
        dtype=np.int64,
    )

    query_indices = np.asarray(
        protocol["query_indices"],
        dtype=np.int64,
    )

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    if (
        database_indices.min() < 0
        or
        database_indices.max() >= dataset_size
    ):

        raise RuntimeError(
            "Database indices exceed "
            "dataset size."
        )

    if (
        query_indices.min() < 0
        or
        query_indices.max() >= dataset_size
    ):

        raise RuntimeError(
            "Query indices exceed "
            "dataset size."
        )

    overlap = np.intersect1d(
        database_indices,
        query_indices,
    )

    if len(overlap) != 0:

        raise RuntimeError(
            f"Database/query overlap: "
            f"{len(overlap)}"
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
    )


# ============================================================
# Feature extraction
# ============================================================

@torch.no_grad()
def extract_features(
    model,
    dataset,
    split_name,
):

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

    all_features = []
    all_labels = []

    print()
    print("=" * 100)
    print(
        f"Extracting {REPRESENTATION} "
        f"features: {split_name}"
    )
    print("=" * 100)

    t0 = time.time()

    for batch_idx, (
        images,
        labels,
    ) in enumerate(loader):

        images = images.to(
            DEVICE,
            non_blocking=True,
        )

        # ====================================================
        # Encoder representation
        # ====================================================

        if REPRESENTATION == "encoder":

            tokens = model.encoder(
                images
            )

            if isinstance(
                tokens,
                (tuple, list),
            ):

                tokens = tokens[0]

            features = (
                tokens
                .float()
                .mean(dim=1)
            )

        # ====================================================
        # Quantized representation
        # ====================================================

        elif REPRESENTATION == "quantized":

            tokens = model.encoder(
                images
            )

            if isinstance(
                tokens,
                (tuple, list),
            ):

                tokens = tokens[0]

            tokens = model.quant_proj(
                tokens.float()
            )

            indices = (
                model.quantizer.f_to_idx(
                    tokens
                )
            )

            tokens = (
                model.quantizer.idx_to_f(
                    indices
                ).float()
            )

            features = (
                tokens.mean(dim=1)
            )

        # ====================================================
        # Final representation
        # ====================================================

        elif REPRESENTATION == "final":

            features = (
                model.encode_image(
                    images,
                    normalize=False,
                )
                .float()
            )

        else:

            raise ValueError(
                REPRESENTATION
            )

        # ====================================================
        # Unified protocol:
        #
        # L2 normalize BEFORE KNN
        # ====================================================

        features = F.normalize(
            features,
            p=2,
            dim=1,
        )

        # ====================================================
        # FP16 cache
        # ====================================================

        all_features.append(
            features
            .cpu()
            .numpy()
            .astype(STORE_DTYPE)
        )

        all_labels.append(
            labels.numpy()
        )

        # ====================================================
        # Progress
        # ====================================================

        if (
            batch_idx % 10 == 0
            or
            batch_idx == len(loader) - 1
        ):

            processed = min(
                (batch_idx + 1)
                * BATCH_SIZE,
                len(dataset),
            )

            elapsed = (
                time.time() - t0
            )

            speed = (
                processed
                /
                max(elapsed, 1e-6)
            )

            eta = (
                (len(dataset) - processed)
                /
                max(speed, 1e-6)
            )

            print(
                f"\r{split_name}: "
                f"{processed:,}/"
                f"{len(dataset):,} | "
                f"{speed:.1f} img/s | "
                f"ETA {eta/60:.2f} min",
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

    print()
    print(
        f"{split_name} features:"
    )

    print(
        f"  shape = {features.shape}"
    )

    print(
        f"  dtype = {features.dtype}"
    )

    return (
        features,
        labels,
    )


# ============================================================
# Similarity-weighted KNN
#
# Unified with previous UniAR protocol:
#
#   L2 normalize
#   IndexFlatIP
#   cosine similarity
#   exponential similarity weighting
# ============================================================

def evaluate_knn(
    database_features,
    database_labels,
    query_features,
    query_labels,
):

    import faiss

    print()
    print("=" * 100)
    print("FAISS KNN")
    print("=" * 100)

    Xdb = np.ascontiguousarray(
        database_features,
        dtype=np.float32,
    )

    Xq = np.ascontiguousarray(
        query_features,
        dtype=np.float32,
    )

    labels_db = np.asarray(
        database_labels
    )

    labels_q = np.asarray(
        query_labels
    )

    D = Xdb.shape[1]

    print(
        f"Database : {len(Xdb):,}"
    )

    print(
        f"Query    : {len(Xq):,}"
    )

    print(
        f"Dimension: {D}"
    )

    # --------------------------------------------------------
    # L2-normalized -> inner product = cosine similarity
    # --------------------------------------------------------

    use_gpu = (
        FAISS_USE_GPU
        and faiss.get_num_gpus() > 0
    )

    if use_gpu:

        print(
            f"FAISS GPU: "
            f"{faiss.get_num_gpus()} GPU(s)"
        )

        res = (
            faiss.StandardGpuResources()
        )

        index = faiss.GpuIndexFlatIP(
            res,
            D,
        )

    else:

        print(
            "FAISS CPU: IndexFlatIP"
        )

        index = faiss.IndexFlatIP(
            D
        )

    # --------------------------------------------------------
    # Add database
    # --------------------------------------------------------

    t0 = time.time()

    index.add(
        Xdb
    )

    print(
        f"Index add time: "
        f"{time.time()-t0:.2f} s"
    )

    max_k = max(
        K_VALUES
    )

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    t0 = time.time()

    print(
        f"Searching top-{max_k}..."
    )

    similarities, indices = (
        index.search(
            Xq,
            max_k,
        )
    )

    search_time = (
        time.time() - t0
    )

    print(
        f"Search time: "
        f"{search_time/60:.3f} min"
    )

    # --------------------------------------------------------
    # Weighted voting
    # --------------------------------------------------------

    temperature = 0.07

    neighbor_labels = (
        labels_db[indices]
    )

    num_classes = int(
        max(
            labels_db.max(),
            labels_q.max(),
        )
        + 1
    )

    results = {}

    for K in K_VALUES:

        top1 = 0
        top5 = 0
        top10 = 0
        top20 = 0

        sims = (
            similarities[:, :K]
        )

        neigh = (
            neighbor_labels[:, :K]
        )

        # ----------------------------------------------------
        # Numerically stable exponential weighting
        # ----------------------------------------------------

        weights = np.exp(
            (
                sims
                -
                sims.max(
                    axis=1,
                    keepdims=True
                )
            )
            /
            temperature
        )

        for i in range(
            len(labels_q)
        ):

            scores = np.bincount(
                neigh[i],
                weights=weights[i],
                minlength=num_classes,
            )

            ranked = np.argsort(
                -scores
            )

            target = labels_q[i]

            if target == ranked[0]:
                top1 += 1

            if target in ranked[:5]:
                top5 += 1

            if target in ranked[:10]:
                top10 += 1

            if target in ranked[:20]:
                top20 += 1

        n = len(
            labels_q
        )

        results[K] = {
            "top1": 100.0 * top1 / n,
            "top5": 100.0 * top5 / n,
            "top10": 100.0 * top10 / n,
            "top20": 100.0 * top20 / n,
        }

        print()
        print(
            f"K = {K}"
        )

        print(
            f"  Top-1  : "
            f"{results[K]['top1']:.4f}%"
        )

        print(
            f"  Top-5  : "
            f"{results[K]['top5']:.4f}%"
        )

        print(
            f"  Top-10 : "
            f"{results[K]['top10']:.4f}%"
        )

        print(
            f"  Top-20 : "
            f"{results[K]['top20']:.4f}%"
        )

    return results


# ============================================================
# FLOPs profiling
#
# We profile the ACTUAL UniTok path corresponding to
# REPRESENTATION.
#
# Then:
#
# Feature extraction FLOPs
#     = 10,000 × FLOPs/image
#
# Exact KNN search:
#
#     2 × N_query × N_database × D
#
# Total:
#
#     feature extraction
#     +
#     exact KNN search
# ============================================================

@torch.no_grad()
def profile_forward(
    model,
    images,
):

    if REPRESENTATION == "encoder":

        tokens = model.encoder(
            images
        )

        if isinstance(
            tokens,
            (tuple, list),
        ):

            tokens = tokens[0]

        features = (
            tokens
            .float()
            .mean(dim=1)
        )

    elif REPRESENTATION == "quantized":

        tokens = model.encoder(
            images
        )

        if isinstance(
            tokens,
            (tuple, list),
        ):

            tokens = tokens[0]

        tokens = model.quant_proj(
            tokens.float()
        )

        indices = (
            model.quantizer.f_to_idx(
                tokens
            )
        )

        tokens = (
            model.quantizer.idx_to_f(
                indices
            ).float()
        )

        features = (
            tokens.mean(dim=1)
        )

    elif REPRESENTATION == "final":

        features = (
            model.encode_image(
                images,
                normalize=False,
            )
            .float()
        )

    else:

        raise ValueError(
            REPRESENTATION
        )

    features = F.normalize(
        features,
        p=2,
        dim=1,
    )

    return features


def calculate_total_flops(
    model,
    query_dataset,
    database_count,
    query_count,
    feature_dim,
):

    print()
    print("=" * 100)
    print("KNN TOTAL FLOPs")
    print("=" * 100)

    # --------------------------------------------------------
    # One real image
    # --------------------------------------------------------

    loader = DataLoader(
        query_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    images, _ = next(
        iter(loader)
    )

    images = images.to(
        DEVICE,
        non_blocking=True,
    )

    # --------------------------------------------------------
    # Warmup
    # --------------------------------------------------------

    print()
    print(
        "Warmup UniTok..."
    )

    with torch.no_grad():

        for _ in range(2):

            _ = profile_forward(
                model,
                images,
            )

    if DEVICE.type == "cuda":

        torch.cuda.synchronize()

    # --------------------------------------------------------
    # torch.profiler
    # --------------------------------------------------------

    activities = [
        torch.profiler.ProfilerActivity.CPU
    ]

    if DEVICE.type == "cuda":

        activities.append(
            torch.profiler.ProfilerActivity.CUDA
        )

    print()
    print(
        "Profiling actual UniTok forward..."
    )

    with torch.profiler.profile(
        activities=activities,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_flops=True,
    ) as prof:

        with torch.no_grad():

            _ = profile_forward(
                model,
                images,
            )

    if DEVICE.type == "cuda":

        torch.cuda.synchronize()

    # --------------------------------------------------------
    # Sum profiler FLOPs
    # --------------------------------------------------------

    model_flops_per_image = 0.0

    for event in (
        prof.key_averages()
    ):

        flops = getattr(
            event,
            "flops",
            0,
        )

        if flops is None:

            flops = 0

        model_flops_per_image += (
            float(flops)
        )

    # --------------------------------------------------------
    # Feature extraction
    # --------------------------------------------------------

    total_images = (
        database_count
        +
        query_count
    )

    feature_extraction_flops = (
        total_images
        *
        model_flops_per_image
    )

    # --------------------------------------------------------
    # Exact exhaustive KNN
    #
    # Dot product:
    #
    #     D multiplications
    #     D additions
    #
    # => approximately 2D FLOPs
    # --------------------------------------------------------

    knn_search_flops = (
        2
        *
        query_count
        *
        database_count
        *
        feature_dim
    )

    # --------------------------------------------------------
    # Total
    # --------------------------------------------------------

    total_flops = (
        feature_extraction_flops
        +
        knn_search_flops
    )

    # --------------------------------------------------------
    # Units
    # --------------------------------------------------------

    model_gflops = (
        model_flops_per_image
        / 1e9
    )

    feature_gflops = (
        feature_extraction_flops
        / 1e9
    )

    feature_pflops = (
        feature_extraction_flops
        / 1e15
    )

    search_gflops = (
        knn_search_flops
        / 1e9
    )

    search_pflops = (
        knn_search_flops
        / 1e15
    )

    total_gflops = (
        total_flops
        / 1e9
    )

    total_pflops = (
        total_flops
        / 1e15
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print()
    print(
        "----- UniTok Feature Extraction -----"
    )

    print(
        f"Representation      : "
        f"{REPRESENTATION}"
    )

    print(
        f"Feature dimension   : "
        f"{feature_dim:,}"
    )

    print(
        f"FLOPs / image       : "
        f"{model_gflops:.6f} GFLOPs"
    )

    print(
        f"Database images     : "
        f"{database_count:,}"
    )

    print(
        f"Query images        : "
        f"{query_count:,}"
    )

    print(
        f"Total images        : "
        f"{total_images:,}"
    )

    print(
        f"Feature extraction  : "
        f"{feature_gflops:.6f} GFLOPs"
    )

    print(
        f"Feature extraction  : "
        f"{feature_pflops:.9f} PFLOPs"
    )

    print()
    print(
        "----- Exact KNN Search -----"
    )

    print(
        f"Formula             : "
        f"2 × {query_count:,} × "
        f"{database_count:,} × "
        f"{feature_dim:,}"
    )

    print(
        f"KNN search          : "
        f"{search_gflops:.6f} GFLOPs"
    )

    print(
        f"KNN search          : "
        f"{search_pflops:.9f} PFLOPs"
    )

    print()
    print(
        "----- KNN TOTAL -----"
    )

    print(
        f"Feature extraction  : "
        f"{feature_gflops:.6f} GFLOPs"
    )

    print(
        f"+ KNN search        : "
        f"{search_gflops:.6f} GFLOPs"
    )

    print(
        "-" * 70
    )

    print(
        f"KNN TOTAL           : "
        f"{total_gflops:.6f} GFLOPs"
    )

    print(
        f"KNN TOTAL           : "
        f"{total_pflops:.9f} PFLOPs"
    )

    print("=" * 100)

    return {
        "model_flops_per_image": (
            float(model_flops_per_image)
        ),

        "model_gflops_per_image": (
            float(model_gflops)
        ),

        "database_images": (
            int(database_count)
        ),

        "query_images": (
            int(query_count)
        ),

        "total_images": (
            int(total_images)
        ),

        "feature_dimension": (
            int(feature_dim)
        ),

        "feature_extraction_flops": (
            float(feature_extraction_flops)
        ),

        "feature_extraction_gflops": (
            float(feature_gflops)
        ),

        "feature_extraction_pflops": (
            float(feature_pflops)
        ),

        "knn_search_flops": (
            int(knn_search_flops)
        ),

        "knn_search_gflops": (
            float(search_gflops)
        ),

        "knn_search_pflops": (
            float(search_pflops)
        ),

        "knn_total_flops": (
            float(total_flops)
        ),

        "knn_total_gflops": (
            float(total_gflops)
        ),

        "knn_total_pflops": (
            float(total_pflops)
        ),
    }


# ============================================================
# Main
# ============================================================

def _single_pool_main():

    print()
    print("=" * 100)
    print(
        "UniTok KNN - Unified Protocol"
    )
    print("=" * 100)

    print(
        f"Representation : "
        f"{REPRESENTATION}"
    )

    print(
        f"Checkpoint     : "
        f"{CKPT_PATH}"
    )

    print(
        f"Protocol       : "
        f"{PROTOCOL_FILE}"
    )

    print(
        f"ImageNet       : "
        f"{TRAIN_PATH}"
    )

    print(
        f"Image size     : "
        f"{IMG_SIZE}"
    )

    print(
        f"Resize         : "
        f"{RESIZE_SIZE}"
    )

    print(
        f"Batch size     : "
        f"{BATCH_SIZE}"
    )

    print(
        f"K values       : "
        f"{K_VALUES}"
    )

    print(
        f"Seed           : "
        f"{SEED}"
    )

    print(
        f"Device         : "
        f"{DEVICE}"
    )

    # ========================================================
    # Dataset
    # ========================================================

    dataset = SafeImageFolder(
        TRAIN_PATH,
        transform=transform,
    )

    # ========================================================
    # Protocol
    # ========================================================

    protocol = load_protocol()

    database_indices, query_indices = (
        get_protocol_indices(
            protocol,
            len(dataset),
        )
    )

    database_dataset = Subset(
        dataset,
        database_indices.tolist(),
    )

    query_dataset = Subset(
        dataset,
        query_indices.tolist(),
    )

    # ========================================================
    # Cache
    # ========================================================

    cache_key = (
        f"{REPRESENTATION}"
        f"_5k5k_seed{SEED}"
    )

    cache_dir = os.path.join(
        CACHE_DIR,
        cache_key,
    )

    os.makedirs(
        cache_dir,
        exist_ok=True,
    )

    db_feat_file = os.path.join(
        cache_dir,
        "database_features.npy",
    )

    db_label_file = os.path.join(
        cache_dir,
        "database_labels.npy",
    )

    q_feat_file = os.path.join(
        cache_dir,
        "query_features.npy",
    )

    q_label_file = os.path.join(
        cache_dir,
        "query_labels.npy",
    )

    print()
    print(
        f"Cache directory:\n{cache_dir}"
    )

    # ========================================================
    # Load / extract
    # ========================================================

    cache_exists = all(
        os.path.exists(p)
        for p in [
            db_feat_file,
            db_label_file,
            q_feat_file,
            q_label_file,
        ]
    )

    if cache_exists:

        print()
        print(
            "[Cache HIT]"
        )

        database_features = np.load(
            db_feat_file
        )

        database_labels = np.load(
            db_label_file
        )

        query_features = np.load(
            q_feat_file
        )

        query_labels = np.load(
            q_label_file
        )

        print(
            f"Database features: "
            f"{database_features.shape} "
            f"{database_features.dtype}"
        )

        print(
            f"Query features   : "
            f"{query_features.shape} "
            f"{query_features.dtype}"
        )

        # ----------------------------------------------------
        # Need model for FLOPs profiling
        # ----------------------------------------------------

        model = load_model()

    else:

        print()
        print(
            "[Cache MISS]"
        )

        model = load_model()

        # ----------------------------------------------------
        # Database
        # ----------------------------------------------------

        database_features, database_labels = (
            extract_features(
                model,
                database_dataset,
                "Database",
            )
        )

        # ----------------------------------------------------
        # Query
        # ----------------------------------------------------

        query_features, query_labels = (
            extract_features(
                model,
                query_dataset,
                "Query",
            )
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        np.save(
            db_feat_file,
            database_features,
        )

        np.save(
            db_label_file,
            database_labels,
        )

        np.save(
            q_feat_file,
            query_features,
        )

        np.save(
            q_label_file,
            query_labels,
        )

        print()
        print(
            "Features saved."
        )

    # ========================================================
    # Statistics
    # ========================================================

    feature_dim = (
        database_features.shape[1]
    )

    print()
    print("=" * 100)
    print("FEATURE STATISTICS")
    print("=" * 100)

    db_norm = np.linalg.norm(
        database_features.astype(
            np.float32
        ),
        axis=1,
    )

    q_norm = np.linalg.norm(
        query_features.astype(
            np.float32
        ),
        axis=1,
    )

    print(
        f"Dimension       : "
        f"{feature_dim}"
    )

    print(
        f"Database norm   : "
        f"mean={db_norm.mean():.6f}, "
        f"std={db_norm.std():.6f}"
    )

    print(
        f"Query norm      : "
        f"mean={q_norm.mean():.6f}, "
        f"std={q_norm.std():.6f}"
    )

    # ========================================================
    # KNN
    # ========================================================

    knn_results = evaluate_knn(
        database_features,
        database_labels,
        query_features,
        query_labels,
    )

    # ========================================================
    # FLOPs
    # ========================================================

    flops = calculate_total_flops(
        model=model,
        query_dataset=query_dataset,
        database_count=len(
            database_features
        ),
        query_count=len(
            query_features
        ),
        feature_dim=feature_dim,
    )

    # ========================================================
    # Final result
    # ========================================================

    print()
    print("=" * 100)
    print("FINAL RESULT")
    print("=" * 100)

    print(
        f"Representation : "
        f"{REPRESENTATION}"
    )

    print(
        f"Database       : "
        f"{len(database_features):,}"
    )

    print(
        f"Query          : "
        f"{len(query_features):,}"
    )

    print(
        f"Dimension      : "
        f"{feature_dim}"
    )

    print()

    print(
        f"{'K':>6} "
        f"{'Top-1':>12} "
        f"{'Top-5':>12} "
        f"{'Top-10':>12} "
        f"{'Top-20':>12}"
    )

    print(
        "-" * 70
    )

    for K in K_VALUES:

        r = knn_results[K]

        print(
            f"{K:>6} "
            f"{r['top1']:>11.4f}% "
            f"{r['top5']:>11.4f}% "
            f"{r['top10']:>11.4f}% "
            f"{r['top20']:>11.4f}%"
        )

    print()
    print(
        "FLOPs:"
    )

    print(
        f"  Feature extraction : "
        f"{flops['feature_extraction_gflops']:.6f} GFLOPs "
        f"({flops['feature_extraction_pflops']:.9f} PFLOPs)"
    )

    print(
        f"  Exact KNN search   : "
        f"{flops['knn_search_gflops']:.6f} GFLOPs "
        f"({flops['knn_search_pflops']:.9f} PFLOPs)"
    )

    print(
        f"  KNN TOTAL          : "
        f"{flops['knn_total_gflops']:.6f} GFLOPs "
        f"({flops['knn_total_pflops']:.9f} PFLOPs)"
    )

    print("=" * 100)

    # ========================================================
    # Save JSON
    # ========================================================

    result = {

        "model": "UniTok",

        "checkpoint": CKPT_PATH,

        "representation": REPRESENTATION,

        "protocol": {

            "protocol_file": (
                PROTOCOL_FILE
            ),

            "seed": SEED,

            "database_images": int(
                len(database_features)
            ),

            "query_images": int(
                len(query_features)
            ),

            "no_overlap": True,

            "image_size": IMG_SIZE,

            "resize_size": RESIZE_SIZE,

            "normalization": {
                "mean": [
                    0.5,
                    0.5,
                    0.5,
                ],
                "std": [
                    0.5,
                    0.5,
                    0.5,
                ],
            },

            "l2_normalized": True,

            "faiss_index": (
                "IndexFlatIP"
            ),

            "similarity": (
                "cosine similarity"
            ),

            "temperature": 0.07,

            "K_values": K_VALUES,
        },

        "knn": {
            str(K): knn_results[K]
            for K in K_VALUES
        },

        "flops": flops,
    }

    result_file = os.path.join(
        cache_dir,
        "results.json",
    )

    with open(
        result_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        f"Results saved to:\n"
        f"{result_file}"
    )

    print()
    print(
        f"Features saved to:\n"
        f"{cache_dir}"
    )

    # ========================================================
    # Release
    # ========================================================

    del model

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()


# ============================================================
# Entry
# ============================================================

if False:  # entry point is defined below
    _single_pool_main()
def main():
    """Extract the 45k pool once, then evaluate cached 5/10/20/45-shot slices."""
    _single_pool_main()
    from multishot_protocol import run_multishot_from_cache
    return run_multishot_from_cache(CACHE_DIR, CACHE_DIR, "unitok")


if __name__ == "__main__":
    main()

