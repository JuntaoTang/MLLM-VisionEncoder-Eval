import os
import gc
import json
import time
import random
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets

from PIL import Image

from transformers import AutoImageProcessor, AutoModel


# ============================================================
# 1. Basic Settings
# ============================================================

SEED = 42

# K values for KNN evaluation
K_VALUES = [1, 5, 10, 20]

# Different database shots
SHOT_VALUES = [5, 10, 20, 45]

BATCH_SIZE = int(
    os.environ.get(
        "WEBSSL_BATCH_SIZE",
        "256",
    )
)

NUM_WORKERS = int(
    os.environ.get(
        "WEBSSL_NUM_WORKERS",
        "8",
    )
)

# Feature cache dtype
STORE_DTYPE = np.float16

TEMPERATURE = 0.07

FAISS_USE_GPU = True

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# 2. ImageNet
#
# IMPORTANT:
# This experiment uses ImageNet validation set.
# ============================================================

IMAGENET_ROOT = (
    "/workspace/root/val"
)


# ============================================================
# 3. Fixed Unified Protocol
#
# 5-shot:
#     5 images/class × 1000 classes = 5,000 database
#
# 10-shot:
#     10 images/class × 1000 classes = 10,000 database
#
# 20-shot:
#     20 images/class × 1000 classes = 20,000 database
#
# 45-shot:
#     45 images/class × 1000 classes = 45,000 database
#
# Query:
#     5 images/class × 1000 classes = 5,000 query
#
# Seed:
#     42
#
# No database/query overlap.
# ============================================================

PROTOCOL_FILE = (
    "/cache/metaclip_knn/"
    "val_45shot_5query_seed42_protocol.json"
)

NUM_QUERY = 5000

NUM_DATABASE_BY_SHOT = {

    5:
        5000,

    10:
        10000,

    20:
        20000,

    45:
        45000,
}


# ============================================================
# 4. WebSSL-MAE Models
# ============================================================

MODEL_CONFIGS = [

    {
        "name":
            "webssl-mae300m",

        "path":
            "/cache/models/model/"
            "webssl-mae300m-full2b-224",
    },

    {
        "name":
            "webssl-mae1b",

        "path":
            "/cache/models/model/"
            "webssl-mae1b-full2b-224",
    },

    {
        "name":
            "webssl-mae3b",

        "path":
            "/cache/models/model/"
            "webssl-mae3b-full2b-224",
    },

]


# ============================================================
# 5. Cache
# ============================================================

CACHE_ROOT = (
    "/cache/knn_cache/"
    "webssl_mae_unified_45shot5query_seed42"
)


# ============================================================
# 6. Seed
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
# 7. Device information
# ============================================================

print()
print("=" * 120)
print("WebSSL-MAE Unified ImageNet KNN")
print("=" * 120)

print(
    f"Device: {DEVICE}"
)

if torch.cuda.is_available():

    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"GPU memory: "
        f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
    )

print("=" * 120)


# ============================================================
# 8. Safe ImageFolder
#
# Preserve ImageFolder ordering.
# ============================================================

class SafeImageFolder(
    datasets.ImageFolder
):

    def __init__(
        self,
        root,
    ):

        print()
        print("=" * 120)
        print("Loading ImageNet")
        print("=" * 120)

        super().__init__(
            root=root,
            transform=None,
        )

        print(
            f"Images : {len(self):,}"
        )

        print(
            f"Classes: {len(self.classes)}"
        )

        if len(self.classes) != 1000:

            print(
                f"WARNING: expected 1000 classes, "
                f"found {len(self.classes)}"
            )


# ============================================================
# 9. Indexed Dataset
# ============================================================

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

        real_index = int(
            self.indices[index]
        )

        image, label = (
            self.dataset[
                real_index
            ]
        )

        return image, label


# ============================================================
# 10. Load fixed protocol
#
# Supports:
#
# train_indices_by_shot:
# {
#     "5":  [...],
#     "10": [...],
#     "20": [...],
#     "45": [...]
# }
#
# and several fallback key names.
# ============================================================

def load_protocol():

    print()
    print("=" * 120)
    print("Loading fixed unified protocol")
    print("=" * 120)

    print(
        f"Protocol:\n"
        f"{PROTOCOL_FILE}"
    )

    if not os.path.exists(
        PROTOCOL_FILE
    ):

        raise FileNotFoundError(
            f"Protocol not found:\n"
            f"{PROTOCOL_FILE}"
        )

    with open(
        PROTOCOL_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        protocol = json.load(f)

    # Canonical protocol stores one ordered 45-image-per-class pool.
    # Derive the legacy per-shot views in memory without changing the file.
    if "train_indices_by_shot" not in protocol and "train_pool_indices" in protocol:
        pool = np.asarray(protocol["train_pool_indices"], dtype=np.int64).reshape(1000, 45)
        protocol["train_indices_by_shot"] = {
            str(shot): pool[:, :shot].reshape(-1).tolist()
            for shot in SHOT_VALUES
        }

    # ========================================================
    # Query
    # ========================================================

    query_indices = None

    for key in [
        "query_indices",
        "val_indices",
        "test_indices",
    ]:

        if key in protocol:

            query_indices = np.asarray(
                protocol[key],
                dtype=np.int64,
            )

            break

    if query_indices is None:

        raise KeyError(
            "Cannot find query indices in protocol."
        )

    if len(query_indices) != NUM_QUERY:

        raise RuntimeError(
            f"Expected {NUM_QUERY} query images, "
            f"got {len(query_indices)}"
        )

    # ========================================================
    # Database indices by shot
    # ========================================================

    database_indices_by_shot = {}

    # --------------------------------------------------------
    # Preferred protocol format
    # --------------------------------------------------------

    if "train_indices_by_shot" in protocol:

        raw = protocol[
            "train_indices_by_shot"
        ]

        for shot in SHOT_VALUES:

            key = str(shot)

            if key not in raw:

                raise KeyError(
                    f"Missing "
                    f"train_indices_by_shot['{key}']"
                )

            indices = np.asarray(
                raw[key],
                dtype=np.int64,
            )

            expected = (
                NUM_DATABASE_BY_SHOT[
                    shot
                ]
            )

            if len(indices) != expected:

                raise RuntimeError(
                    f"{shot}-shot expected "
                    f"{expected} images, "
                    f"got {len(indices)}"
                )

            database_indices_by_shot[
                shot
            ] = indices

    # --------------------------------------------------------
    # Fallback protocol formats
    # --------------------------------------------------------

    else:

        for shot in SHOT_VALUES:

            possible_keys = [

                f"train_indices_{shot}shot",

                f"train_{shot}shot_indices",

                f"{shot}shot_indices",

            ]

            found = None

            for key in possible_keys:

                if key in protocol:

                    found = np.asarray(
                        protocol[key],
                        dtype=np.int64,
                    )

                    break

            if found is None:

                raise KeyError(
                    f"Cannot find database indices "
                    f"for {shot}-shot."
                )

            expected = (
                NUM_DATABASE_BY_SHOT[
                    shot
                ]
            )

            if len(found) != expected:

                raise RuntimeError(
                    f"{shot}-shot expected "
                    f"{expected} images, "
                    f"got {len(found)}"
                )

            database_indices_by_shot[
                shot
            ] = found

    # ========================================================
    # Check overlap
    # ========================================================

    for shot in SHOT_VALUES:

        db_indices = (
            database_indices_by_shot[
                shot
            ]
        )

        overlap = np.intersect1d(
            db_indices,
            query_indices,
        )

        if len(overlap) != 0:

            raise RuntimeError(
                f"{shot}-shot database/query "
                f"overlap detected: "
                f"{len(overlap)}"
            )

    # ========================================================
    # Check nested protocol
    #
    # 5 ⊂ 10 ⊂ 20 ⊂ 45
    # ========================================================

    for smaller, larger in [

        (5, 10),

        (10, 20),

        (20, 45),

    ]:

        small_set = set(
            database_indices_by_shot[
                smaller
            ]
        )

        large_set = set(
            database_indices_by_shot[
                larger
            ]
        )

        if not small_set.issubset(
            large_set
        ):

            raise RuntimeError(
                f"Protocol violation: "
                f"{smaller}-shot is not a subset "
                f"of {larger}-shot."
            )

    # ========================================================
    # Print
    # ========================================================

    print()

    for shot in SHOT_VALUES:

        print(
            f"{shot:>2}-shot database : "
            f"{len(database_indices_by_shot[shot]):,}"
        )

    print(
        f"Query              : "
        f"{len(query_indices):,}"
    )

    print(
        "Database/query overlap: 0"
    )

    print()

    return (
        database_indices_by_shot,
        query_indices,
    )


# ============================================================
# 11. Official WebSSL-MAE processor
# ============================================================

def build_processor(
    model_path,
):

    print()
    print("=" * 120)
    print("Loading official WebSSL-MAE AutoImageProcessor")
    print("=" * 120)

    print(
        model_path
    )

    processor = (
        AutoImageProcessor.from_pretrained(
            model_path,
            local_files_only=True,
        )
    )

    print()
    print("Processor:")
    print(processor)

    print()
    print("-" * 120)
    print("Official preprocessing")
    print("-" * 120)

    print(
        f"Processor type : "
        f"{getattr(processor, 'image_processor_type', 'N/A')}"
    )

    print(
        f"do_resize      : "
        f"{getattr(processor, 'do_resize', 'N/A')}"
    )

    print(
        f"size           : "
        f"{getattr(processor, 'size', 'N/A')}"
    )

    print(
        f"do_center_crop : "
        f"{getattr(processor, 'do_center_crop', 'N/A')}"
    )

    print(
        f"crop_size      : "
        f"{getattr(processor, 'crop_size', 'N/A')}"
    )

    print(
        f"do_rescale     : "
        f"{getattr(processor, 'do_rescale', 'N/A')}"
    )

    print(
        f"rescale_factor : "
        f"{getattr(processor, 'rescale_factor', 'N/A')}"
    )

    print(
        f"do_normalize   : "
        f"{getattr(processor, 'do_normalize', 'N/A')}"
    )

    print(
        f"image_mean     : "
        f"{getattr(processor, 'image_mean', 'N/A')}"
    )

    print(
        f"image_std      : "
        f"{getattr(processor, 'image_std', 'N/A')}"
    )

    print(
        f"resample       : "
        f"{getattr(processor, 'resample', 'N/A')}"
    )

    print("-" * 120)

    return processor


# ============================================================
# 12. Collate
# ============================================================

def make_collate_fn(
    processor,
):

    def collate_fn(
        batch,
    ):

        images = [
            item[0]
            for item in batch
        ]

        labels = [
            item[1]
            for item in batch
        ]

        inputs = processor(
            images=images,
            return_tensors="pt",
        )

        labels = torch.tensor(
            labels,
            dtype=torch.long,
        )

        return (
            inputs,
            labels,
        )

    return collate_fn


# ============================================================
# 13. Load WebSSL-MAE
# ============================================================

def load_webssl_mae(
    model_path,
):

    print()
    print("=" * 120)
    print("Loading WebSSL-MAE")
    print("=" * 120)

    print(
        f"Path:\n"
        f"{model_path}"
    )

    if not os.path.exists(
        model_path
    ):

        raise FileNotFoundError(
            f"Model path does not exist:\n"
            f"{model_path}"
        )

    # --------------------------------------------------------
    # Processor
    # --------------------------------------------------------

    processor = (
        build_processor(
            model_path
        )
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = AutoModel.from_pretrained(
        model_path,
        local_files_only=True,
    )

    model = (
        model
        .to(DEVICE)
        .eval()
    )

    config = model.config

    print()
    print("-" * 120)
    print("Model configuration")
    print("-" * 120)

    print(
        f"Model type      : "
        f"{getattr(config, 'model_type', 'N/A')}"
    )

    print(
        f"Hidden size     : "
        f"{getattr(config, 'hidden_size', 'N/A')}"
    )

    print(
        f"Layers          : "
        f"{getattr(config, 'num_hidden_layers', 'N/A')}"
    )

    print(
        f"Attention heads : "
        f"{getattr(config, 'num_attention_heads', 'N/A')}"
    )

    print(
        f"Image size      : "
        f"{getattr(config, 'image_size', 'N/A')}"
    )

    print(
        f"Patch size      : "
        f"{getattr(config, 'patch_size', 'N/A')}"
    )

    print("-" * 120)

    feat_dim = int(
        config.hidden_size
    )

    # --------------------------------------------------------
    # Test forward
    # --------------------------------------------------------

    dummy_image = Image.new(
        "RGB",
        (224, 224),
        color=(128, 128, 128),
    )

    inputs = processor(
        images=dummy_image,
        return_tensors="pt",
    )

    inputs = {
        k: v.to(DEVICE)
        for k, v in inputs.items()
    }

    with torch.no_grad():

        outputs = model(
            **inputs
        )

    last_hidden_state = (
        outputs.last_hidden_state
    )

    print()
    print(
        f"Output shape: "
        f"{tuple(last_hidden_state.shape)}"
    )

    if last_hidden_state.ndim != 3:

        raise RuntimeError(
            "Unexpected last_hidden_state "
            "dimension."
        )

    if (
        last_hidden_state.shape[-1]
        != feat_dim
    ):

        raise RuntimeError(
            "Feature dimension mismatch."
        )

    cls_feature = (
        last_hidden_state[
            :,
            0,
            :
        ]
    )

    print(
        f"CLS feature shape: "
        f"{tuple(cls_feature.shape)}"
    )

    print()
    print(
        "WebSSL-MAE loaded successfully."
    )

    print(
        "Feature = last_hidden_state[:, 0, :]"
    )

    return (
        model,
        processor,
        feat_dim,
    )


# ============================================================
# 14. Feature extraction
# ============================================================

@torch.no_grad()
def extract_features(
    model,
    loader,
    split_name,
):

    all_features = []
    all_labels = []

    total = len(
        loader.dataset
    )

    processed = 0

    t0 = time.time()

    print()
    print("=" * 120)

    print(
        f"Extracting {split_name} features"
    )

    print("=" * 120)

    for inputs, labels in loader:

        inputs = {
            k: v.to(
                DEVICE,
                non_blocking=True,
            )
            for k, v in inputs.items()
        }

        outputs = model(
            **inputs
        )

        # ----------------------------------------------------
        # CLS
        # ----------------------------------------------------

        features = (
            outputs
            .last_hidden_state[
                :,
                0,
                :
            ]
        )

        # ----------------------------------------------------
        # L2 normalization
        # ----------------------------------------------------

        features = F.normalize(
            features.float(),
            p=2,
            dim=1,
        )

        # ----------------------------------------------------
        # Cache FP16
        # ----------------------------------------------------

        features_np = (
            features
            .cpu()
            .numpy()
            .astype(STORE_DTYPE)
        )

        labels_np = (
            labels
            .cpu()
            .numpy()
        )

        all_features.append(
            features_np
        )

        all_labels.append(
            labels_np
        )

        processed += len(
            labels
        )

        if (
            processed % 1024 == 0
            or processed == total
        ):

            elapsed = (
                time.time() - t0
            )

            speed = (
                processed
                /
                max(
                    elapsed,
                    1e-6,
                )
            )

            eta = (
                (total - processed)
                /
                max(
                    speed,
                    1e-6,
                )
            )

            print(
                f"\r"
                f"{split_name}: "
                f"{processed:,}/"
                f"{total:,} | "
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

    elapsed = (
        time.time() - t0
    )

    print()
    print(
        f"{split_name} complete."
    )

    print(
        f"Shape : {features.shape}"
    )

    print(
        f"Dtype : {features.dtype}"
    )

    print(
        f"Time  : {elapsed/60:.2f} min"
    )

    print(
        f"Speed : "
        f"{len(features)/max(elapsed, 1e-6):.1f} img/s"
    )

    return (
        features,
        labels,
    )


# ============================================================
# 15. FAISS KNN
#
# L2-normalized features
# IndexFlatIP
# Cosine similarity
# Temperature-weighted voting
# ============================================================

def evaluate_knn(
    database_features,
    database_labels,
    query_features,
    query_labels,
):

    import faiss

    print()
    print("=" * 120)
    print("FAISS KNN")
    print("=" * 120)

    Xdb = np.ascontiguousarray(
        database_features,
        dtype=np.float32,
    )

    Xq = np.ascontiguousarray(
        query_features,
        dtype=np.float32,
    )

    labels_db = np.asarray(
        database_labels,
        dtype=np.int64,
    )

    labels_q = np.asarray(
        query_labels,
        dtype=np.int64,
    )

    feature_dim = Xdb.shape[1]

    print(
        f"Database : {len(Xdb):,}"
    )

    print(
        f"Query    : {len(Xq):,}"
    )

    print(
        f"Dimension: {feature_dim}"
    )

    # --------------------------------------------------------
    # FAISS
    # --------------------------------------------------------

    use_gpu = (
        FAISS_USE_GPU
        and
        faiss.get_num_gpus() > 0
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
            feature_dim,
        )

    else:

        print(
            "FAISS CPU: IndexFlatIP"
        )

        index = faiss.IndexFlatIP(
            feature_dim
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
        f"{time.time() - t0:.3f} s"
    )

    max_k = max(
        K_VALUES
    )

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    print()
    print(
        f"Searching top-{max_k}..."
    )

    t0 = time.time()

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
        f"{search_time:.3f} s"
    )

    # --------------------------------------------------------
    # Neighbor labels
    # --------------------------------------------------------

    neighbor_labels = (
        labels_db[
            indices
        ]
    )

    num_classes = int(
        max(
            labels_db.max(),
            labels_q.max(),
        )
        + 1
    )

    results = {}

    # --------------------------------------------------------
    # K values
    # --------------------------------------------------------

    for K in K_VALUES:

        sims = (
            similarities[
                :,
                :K
            ]
        )

        neigh = (
            neighbor_labels[
                :,
                :K
            ]
        )

        # ----------------------------------------------------
        # Stable exponential weighting
        # ----------------------------------------------------

        weights = np.exp(
            (
                sims
                -
                sims.max(
                    axis=1,
                    keepdims=True,
                )
            )
            /
            TEMPERATURE
        )

        top1 = 0
        top5 = 0
        top10 = 0
        top20 = 0

        # ----------------------------------------------------
        # Voting
        # ----------------------------------------------------

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

            "top1":
                100.0 * top1 / n,

            "top5":
                100.0 * top5 / n,

            "top10":
                100.0 * top10 / n,

            "top20":
                100.0 * top20 / n,

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

    # --------------------------------------------------------
    # Release FAISS resources
    # --------------------------------------------------------

    del index

    if use_gpu:

        del res

    return results


# ============================================================
# 16. Actual model forward FLOPs
# ============================================================

@torch.no_grad()
def profile_forward(
    model,
    processor,
):

    # --------------------------------------------------------
    # Dummy image
    # --------------------------------------------------------

    dummy_image = Image.new(
        "RGB",
        (224, 224),
        color=(128, 128, 128),
    )

    inputs = processor(
        images=dummy_image,
        return_tensors="pt",
    )

    inputs = {
        k: v.to(DEVICE)
        for k, v in inputs.items()
    }

    # --------------------------------------------------------
    # Warmup
    # --------------------------------------------------------

    for _ in range(2):

        outputs = model(
            **inputs
        )

        _ = (
            outputs
            .last_hidden_state[
                :,
                0,
                :
            ]
        )

    if DEVICE.type == "cuda":

        torch.cuda.synchronize()

    # --------------------------------------------------------
    # Profiler
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
        "Profiling actual WebSSL-MAE forward..."
    )

    with torch.profiler.profile(
        activities=activities,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_flops=True,
    ) as prof:

        outputs = model(
            **inputs
        )

        _ = (
            outputs
            .last_hidden_state[
                :,
                0,
                :
            ]
        )

    if DEVICE.type == "cuda":

        torch.cuda.synchronize()

    total_flops = 0.0

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

        total_flops += float(
            flops
        )

    print(
        f"Actual forward FLOPs: "
        f"{total_flops:.0f}"
    )

    return total_flops


# ============================================================
# 17. Calculate FLOPs for one shot
#
# Total:
#
#     Feature extraction
#       +
#     Exact KNN search
#
# Feature extraction:
#
#     (N_database + N_query)
#       ×
#     model FLOPs/image
#
# KNN:
#
#     2 × N_query × N_database × D
# ============================================================

def calculate_shot_flops(
    model_flops_per_image,
    feature_dim,
    database_images,
    query_images=NUM_QUERY,
):

    # --------------------------------------------------------
    # Feature extraction FLOPs
    # --------------------------------------------------------

    total_images = (
        database_images
        +
        query_images
    )

    feature_extraction_flops = (
        total_images
        *
        model_flops_per_image
    )

    # --------------------------------------------------------
    # Exact KNN search FLOPs
    #
    # 2 FLOPs / dimension:
    # multiplication + addition
    # --------------------------------------------------------

    knn_search_flops = (
        2
        *
        query_images
        *
        database_images
        *
        feature_dim
    )

    # --------------------------------------------------------
    # Total
    # --------------------------------------------------------

    knn_total_flops = (
        feature_extraction_flops
        +
        knn_search_flops
    )

    return {

        "database_images":
            int(database_images),

        "query_images":
            int(query_images),

        "total_images":
            int(total_images),

        "feature_dimension":
            int(feature_dim),

        "model_flops_per_image":
            float(model_flops_per_image),

        "model_gflops_per_image":
            float(
                model_flops_per_image
                / 1e9
            ),

        "model_tflops_per_image":
            float(
                model_flops_per_image
                / 1e12
            ),

        "feature_extraction_flops":
            float(
                feature_extraction_flops
            ),

        "feature_extraction_gflops":
            float(
                feature_extraction_flops
                / 1e9
            ),

        "feature_extraction_tflops":
            float(
                feature_extraction_flops
                / 1e12
            ),

        "feature_extraction_pflops":
            float(
                feature_extraction_flops
                / 1e15
            ),

        "knn_search_flops":
            int(
                knn_search_flops
            ),

        "knn_search_gflops":
            float(
                knn_search_flops
                / 1e9
            ),

        "knn_search_tflops":
            float(
                knn_search_flops
                / 1e12
            ),

        "knn_search_pflops":
            float(
                knn_search_flops
                / 1e15
            ),

        "knn_total_flops":
            float(
                knn_total_flops
            ),

        "knn_total_gflops":
            float(
                knn_total_flops
                / 1e9
            ),

        "knn_total_tflops":
            float(
                knn_total_flops
                / 1e12
            ),

        "knn_total_pflops":
            float(
                knn_total_flops
                / 1e15
            ),
    }


# ============================================================
# 18. Calculate FLOPs for all shots
# ============================================================

def calculate_all_shot_flops(
    model_flops_per_image,
    feature_dim,
):

    all_flops = {}

    for shot in SHOT_VALUES:

        database_images = (
            NUM_DATABASE_BY_SHOT[
                shot
            ]
        )

        flops = (
            calculate_shot_flops(
                model_flops_per_image=
                    model_flops_per_image,

                feature_dim=
                    feature_dim,

                database_images=
                    database_images,

                query_images=
                    NUM_QUERY,
            )
        )

        all_flops[
            str(shot)
        ] = flops

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print()
    print("=" * 140)
    print("KNN FLOPs BY SHOT")
    print("=" * 140)

    print(
        f"{'Shot':>10} "
        f"{'Database':>12} "
        f"{'Query':>10} "
        f"{'Feature PFLOPs':>20} "
        f"{'Search PFLOPs':>20} "
        f"{'KNN Total PFLOPs':>22}"
    )

    print(
        "-" * 140
    )

    for shot in SHOT_VALUES:

        f = all_flops[
            str(shot)
        ]

        print(
            f"{shot:>7}-shot "
            f"{f['database_images']:>12,} "
            f"{f['query_images']:>10,} "
            f"{f['feature_extraction_pflops']:>20.9f} "
            f"{f['knn_search_pflops']:>20.9f} "
            f"{f['knn_total_pflops']:>22.9f}"
        )

    print("=" * 140)

    return all_flops


# ============================================================
# 19. Main
# ============================================================

def main():

    seed_everything(
        SEED
    )

    os.makedirs(
        CACHE_ROOT,
        exist_ok=True,
    )

    print()
    print("=" * 120)
    print(
        "WebSSL-MAE ImageNet KNN - Unified 5/10/20/45-shot Protocol"
    )
    print("=" * 120)

    print(
        f"ImageNet       : "
        f"{IMAGENET_ROOT}"
    )

    print(
        f"Protocol       : "
        f"{PROTOCOL_FILE}"
    )

    print(
        f"Shots          : "
        f"{SHOT_VALUES}"
    )

    print(
        f"Database       : "
        f"5k / 10k / 20k / 45k"
    )

    print(
        f"Query          : "
        f"{NUM_QUERY:,}"
    )

    print(
        f"Seed           : "
        f"{SEED}"
    )

    print(
        f"Batch size     : "
        f"{BATCH_SIZE}"
    )

    print(
        f"Workers        : "
        f"{NUM_WORKERS}"
    )

    print(
        f"K values       : "
        f"{K_VALUES}"
    )

    print(
        "Feature        : "
        "CLS token"
    )

    print(
        "Normalization  : "
        "L2"
    )

    print(
        "FAISS          : "
        "IndexFlatIP"
    )

    print(
        "Similarity     : "
        "Cosine"
    )

    print(
        f"Temperature    : "
        f"{TEMPERATURE}"
    )

    print("=" * 120)

    # ========================================================
    # Dataset
    # ========================================================

    dataset = SafeImageFolder(
        IMAGENET_ROOT
    )

    # ========================================================
    # Protocol
    # ========================================================

    (
        database_indices_by_shot,
        query_indices,
    ) = load_protocol()

    # ========================================================
    # Check all indices
    # ========================================================

    for shot in SHOT_VALUES:

        indices = (
            database_indices_by_shot[
                shot
            ]
        )

        if (
            indices.min() < 0
            or
            indices.max()
            >= len(dataset)
        ):

            raise RuntimeError(
                f"{shot}-shot database index "
                f"exceeds dataset."
            )

    if (
        query_indices.min() < 0
        or
        query_indices.max()
        >= len(dataset)
    ):

        raise RuntimeError(
            "Query index exceeds dataset."
        )

    # ========================================================
    # Query dataset
    #
    # Query is identical for all shots.
    # ========================================================

    query_dataset = (
        IndexedDataset(
            dataset,
            query_indices,
        )
    )

    # ========================================================
    # Model loop
    # ========================================================

    all_results = []

    for config in MODEL_CONFIGS:

        model_name = (
            config["name"]
        )

        model_path = (
            config["path"]
        )

        print()
        print()
        print("#" * 120)
        print(
            f"# MODEL: {model_name}"
        )
        print(
            f"# PATH : {model_path}"
        )
        print("#" * 120)

        # ====================================================
        # Model cache
        # ====================================================

        model_cache = os.path.join(
            CACHE_ROOT,
            model_name,
        )

        os.makedirs(
            model_cache,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # Cache database at 45-shot.
        # All smaller shots are subsets.
        # ----------------------------------------------------

        database_feature_file = (
            os.path.join(
                model_cache,
                "database_45shot_features.npy",
            )
        )

        database_label_file = (
            os.path.join(
                model_cache,
                "database_45shot_labels.npy",
            )
        )

        database_index_file = (
            os.path.join(
                model_cache,
                "database_45shot_indices.npy",
            )
        )

        query_feature_file = (
            os.path.join(
                model_cache,
                "query_features.npy",
            )
        )

        query_label_file = (
            os.path.join(
                model_cache,
                "query_labels.npy",
            )
        )

        flops_file = (
            os.path.join(
                model_cache,
                "flops_by_shot.json",
            )
        )

        result_file = (
            os.path.join(
                model_cache,
                "results_by_shot.json",
            )
        )

        # ====================================================
        # 45-shot database indices
        # ====================================================

        database_45_indices = (
            database_indices_by_shot[
                45
            ]
        )

        # ====================================================
        # Cache check
        # ====================================================

        cache_exists = all(

            os.path.exists(p)

            for p in [

                database_feature_file,

                database_label_file,

                database_index_file,

                query_feature_file,

                query_label_file,

            ]
        )

        # ====================================================
        # Load / extract
        # ====================================================

        if cache_exists:

            print()
            print(
                "[CACHE HIT]"
            )

            database_features = np.load(
                database_feature_file
            )

            database_labels = np.load(
                database_label_file
            )

            cached_database_indices = (
                np.load(
                    database_index_file
                )
            )

            query_features = np.load(
                query_feature_file
            )

            query_labels = np.load(
                query_label_file
            )

            # ------------------------------------------------
            # Verify database index cache
            # ------------------------------------------------

            if not np.array_equal(
                cached_database_indices,
                database_45_indices,
            ):

                raise RuntimeError(
                    "Cached 45-shot database indices "
                    "do not match current protocol."
                )

            # ------------------------------------------------
            # Verify sizes
            # ------------------------------------------------

            if (
                len(database_features)
                != 45000
            ):

                raise RuntimeError(
                    "Cached database feature size "
                    "is not 45,000."
                )

            if (
                len(query_features)
                != NUM_QUERY
            ):

                raise RuntimeError(
                    "Cached query feature size "
                    f"is not {NUM_QUERY}."
                )

            print(
                f"Database features: "
                f"{database_features.shape} "
                f"{database_features.dtype}"
            )

            print(
                f"Query features: "
                f"{query_features.shape} "
                f"{query_features.dtype}"
            )

            feature_dim = (
                database_features.shape[1]
            )

            # Need model for FLOPs
            (
                model,
                processor,
                _,
            ) = load_webssl_mae(
                model_path
            )

        else:

            print()
            print(
                "[CACHE MISS]"
            )

            # ------------------------------------------------
            # Load model
            # ------------------------------------------------

            (
                model,
                processor,
                feature_dim,
            ) = load_webssl_mae(
                model_path
            )

            # ------------------------------------------------
            # Collate
            # ------------------------------------------------

            collate_fn = (
                make_collate_fn(
                    processor
                )
            )

            # ------------------------------------------------
            # 45-shot database dataset
            # ------------------------------------------------

            database_45_dataset = (
                IndexedDataset(
                    dataset,
                    database_45_indices,
                )
            )

            # ------------------------------------------------
            # DataLoader
            # ------------------------------------------------

            database_loader = DataLoader(

                database_45_dataset,

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

            # ------------------------------------------------
            # Extract 45k database features ONCE
            # ------------------------------------------------

            (
                database_features,
                database_labels,
            ) = extract_features(

                model,

                database_loader,

                "45-shot Database",
            )

            # ------------------------------------------------
            # Extract query features ONCE
            # ------------------------------------------------

            (
                query_features,
                query_labels,
            ) = extract_features(

                model,

                query_loader,

                "Query",
            )

            # ------------------------------------------------
            # Save
            # ------------------------------------------------

            np.save(
                database_feature_file,
                database_features,
            )

            np.save(
                database_label_file,
                database_labels,
            )

            np.save(
                database_index_file,
                database_45_indices,
            )

            np.save(
                query_feature_file,
                query_features,
            )

            np.save(
                query_label_file,
                query_labels,
            )

            print()
            print(
                "45-shot database features "
                "and query features saved."
            )

        # ====================================================
        # Feature statistics
        # ====================================================

        print()
        print("=" * 120)
        print("FEATURE STATISTICS")
        print("=" * 120)

        db_float = (
            database_features
            .astype(np.float32)
        )

        q_float = (
            query_features
            .astype(np.float32)
        )

        db_norm = np.linalg.norm(
            db_float,
            axis=1,
        )

        q_norm = np.linalg.norm(
            q_float,
            axis=1,
        )

        print(
            f"Feature dimension : "
            f"{feature_dim}"
        )

        print(
            f"45-shot DB norm   : "
            f"mean={db_norm.mean():.6f}, "
            f"std={db_norm.std():.6f}"
        )

        print(
            f"Query norm        : "
            f"mean={q_norm.mean():.6f}, "
            f"std={q_norm.std():.6f}"
        )

        # ====================================================
        # Verify query labels
        # ====================================================

        if len(query_labels) != NUM_QUERY:

            raise RuntimeError(
                f"Expected {NUM_QUERY} query labels, "
                f"got {len(query_labels)}"
            )

        # ====================================================
        # Build global index -> position mapping
        #
        # database_features corresponds exactly to
        # database_indices_by_shot[45].
        # ====================================================

        database_index_to_position = {

            int(global_idx):
                position

            for position, global_idx
            in enumerate(
                database_45_indices
            )
        }

        # ====================================================
        # Model FLOPs
        # ====================================================

        model_flops_per_image = (
            profile_forward(
                model,
                processor,
            )
        )

        print()
        print("=" * 120)
        print("MODEL FORWARD FLOPs")
        print("=" * 120)

        print(
            f"Model FLOPs/image : "
            f"{model_flops_per_image / 1e9:.6f} GFLOPs"
        )

        print(
            f"Model FLOPs/image : "
            f"{model_flops_per_image / 1e12:.6f} TFLOPs"
        )

        # ====================================================
        # FLOPs for all shots
        # ====================================================

        shot_flops = (
            calculate_all_shot_flops(
                model_flops_per_image=
                    model_flops_per_image,

                feature_dim=
                    feature_dim,
            )
        )

        # ====================================================
        # KNN for every shot
        # ====================================================

        all_shot_results = {}

        for shot in SHOT_VALUES:

            print()
            print()
            print("#" * 120)
            print(
                f"# MODEL: {model_name}"
            )
            print(
                f"# SHOT : {shot}"
            )
            print("#" * 120)

            # ------------------------------------------------
            # Current shot indices
            # ------------------------------------------------

            shot_database_indices = (
                database_indices_by_shot[
                    shot
                ]
            )

            expected_database_size = (
                NUM_DATABASE_BY_SHOT[
                    shot
                ]
            )

            # ------------------------------------------------
            # Map global dataset index
            # to 45-shot feature position
            # ------------------------------------------------

            shot_positions = np.asarray(

                [
                    database_index_to_position[
                        int(idx)
                    ]

                    for idx
                    in shot_database_indices
                ],

                dtype=np.int64,
            )

            # ------------------------------------------------
            # Slice 45-shot feature cache
            # ------------------------------------------------

            shot_database_features = (
                database_features[
                    shot_positions
                ]
            )

            shot_database_labels = (
                database_labels[
                    shot_positions
                ]
            )

            # ------------------------------------------------
            # Sanity checks
            # ------------------------------------------------

            if (
                len(shot_database_features)
                != expected_database_size
            ):

                raise RuntimeError(
                    f"{shot}-shot feature count mismatch."
                )

            if (
                len(shot_database_labels)
                != expected_database_size
            ):

                raise RuntimeError(
                    f"{shot}-shot label count mismatch."
                )

            # ------------------------------------------------
            # Check labels against original ImageFolder
            # ------------------------------------------------

            expected_labels = np.asarray(

                [
                    dataset.targets[
                        int(idx)
                    ]

                    for idx
                    in shot_database_indices
                ],

                dtype=np.int64,
            )

            if not np.array_equal(
                shot_database_labels,
                expected_labels,
            ):

                raise RuntimeError(
                    f"{shot}-shot cached labels "
                    f"do not match ImageFolder."
                )

            # ------------------------------------------------
            # Print
            # ------------------------------------------------

            print(
                f"Shot       : "
                f"{shot}"
            )

            print(
                f"Database   : "
                f"{len(shot_database_features):,}"
            )

            print(
                f"Query      : "
                f"{len(query_features):,}"
            )

            print(
                f"Feature dim: "
                f"{feature_dim}"
            )

            # ------------------------------------------------
            # KNN
            # ------------------------------------------------

            knn_results = evaluate_knn(

                database_features=
                    shot_database_features,

                database_labels=
                    shot_database_labels,

                query_features=
                    query_features,

                query_labels=
                    query_labels,
            )

            # ------------------------------------------------
            # FLOPs
            # ------------------------------------------------

            flops = shot_flops[
                str(shot)
            ]

            # ------------------------------------------------
            # Store
            # ------------------------------------------------

            all_shot_results[
                str(shot)
            ] = {

                "shot":
                    shot,

                "database_images":
                    expected_database_size,

                "query_images":
                    NUM_QUERY,

                "knn":
                    {
                        str(K):
                            knn_results[K]

                        for K in K_VALUES
                    },

                "flops":
                    flops,
            }

            # ------------------------------------------------
            # Print result
            # ------------------------------------------------

            print()
            print("=" * 120)
            print(
                f"FINAL RESULT - "
                f"{model_name} - {shot}-shot"
            )
            print("=" * 120)

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
                "----- FLOPs -----"
            )

            print(
                f"Database images     : "
                f"{flops['database_images']:,}"
            )

            print(
                f"Query images        : "
                f"{flops['query_images']:,}"
            )

            print(
                f"Total images        : "
                f"{flops['total_images']:,}"
            )

            print(
                f"Feature extraction : "
                f"{flops['feature_extraction_pflops']:.9f} PFLOPs"
            )

            print(
                f"Exact KNN search    : "
                f"{flops['knn_search_pflops']:.9f} PFLOPs"
            )

            print(
                f"KNN TOTAL           : "
                f"{flops['knn_total_pflops']:.9f} PFLOPs"
            )

            print(
                f"KNN TOTAL           : "
                f"{flops['knn_total_tflops']:.6f} TFLOPs"
            )

            print("=" * 120)

            # ------------------------------------------------
            # Release shot arrays
            # ------------------------------------------------

            del shot_database_features
            del shot_database_labels
            del shot_positions

            gc.collect()

            if torch.cuda.is_available():

                torch.cuda.empty_cache()

        # ====================================================
        # Save FLOPs
        # ====================================================

        with open(
            flops_file,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                {

                    "model":
                        model_name,

                    "feature_dimension":
                        int(feature_dim),

                    "model_flops_per_image":
                        float(
                            model_flops_per_image
                        ),

                    "shots":
                        shot_flops,

                },
                f,
                indent=2,
            )

        # ====================================================
        # Save model result
        # ====================================================

        result = {

            "model":
                model_name,

            "model_path":
                model_path,

            "feature":
                "last_hidden_state[:, 0, :]",

            "feature_dimension":
                int(feature_dim),

            "model_flops_per_image":
                float(
                    model_flops_per_image
                ),

            "protocol": {

                "protocol_file":
                    PROTOCOL_FILE,

                "seed":
                    SEED,

                "database_images_by_shot":
                    {
                        str(k): int(v)

                        for k, v
                        in NUM_DATABASE_BY_SHOT.items()
                    },

                "query_images":
                    NUM_QUERY,

                "no_overlap":
                    True,

                "preprocessing":
                    "official AutoImageProcessor",

                "l2_normalized":
                    True,

                "feature_dtype":
                    "float16",

                "faiss_index":
                    "IndexFlatIP",

                "similarity":
                    "cosine similarity",

                "temperature":
                    TEMPERATURE,

                "K_values":
                    K_VALUES,

            },

            "shots":
                all_shot_results,
        }

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
            f"Result saved to:\n"
            f"{result_file}"
        )

        print(
            f"FLOPs saved to:\n"
            f"{flops_file}"
        )

        # ====================================================
        # Add to global results
        # ====================================================

        all_results.append({

            "model":
                model_name,

            "feature_dim":
                int(feature_dim),

            "model_flops_per_image":
                float(
                    model_flops_per_image
                ),

            "shots":
                all_shot_results,

        })

        # ====================================================
        # Release model
        # ====================================================

        del model

        del processor

        del database_features
        del database_labels

        del query_features
        del query_labels

        gc.collect()

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

    # ========================================================
    # Final summary
    # ========================================================

    print()
    print()
    print("=" * 150)
    print(
        "ALL WebSSL-MAE RESULTS"
    )
    print("=" * 150)

    print(
        f"{'Model':20s} "
        f"{'Shot':>9s} "
        f"{'DB':>10s} "
        f"{'K=1':>11s} "
        f"{'K=5':>11s} "
        f"{'K=10':>11s} "
        f"{'K=20':>11s} "
        f"{'Feature PFLOPs':>18s} "
        f"{'Search PFLOPs':>18s} "
        f"{'TOTAL PFLOPs':>18s}"
    )

    print(
        "-" * 150
    )

    for model_result in all_results:

        model_name = (
            model_result["model"]
        )

        for shot in SHOT_VALUES:

            shot_result = (
                model_result[
                    "shots"
                ][
                    str(shot)
                ]
            )

            knn = (
                shot_result[
                    "knn"
                ]
            )

            flops = (
                shot_result[
                    "flops"
                ]
            )

            print(
                f"{model_name:20s} "
                f"{shot:>7}-shot "
                f"{flops['database_images']:>10,} "
                f"{knn['1']['top1']:>10.4f}% "
                f"{knn['5']['top1']:>10.4f}% "
                f"{knn['10']['top1']:>10.4f}% "
                f"{knn['20']['top1']:>10.4f}% "
                f"{flops['feature_extraction_pflops']:>18.9f} "
                f"{flops['knn_search_pflops']:>18.9f} "
                f"{flops['knn_total_pflops']:>18.9f}"
            )

    print("=" * 150)

    # ========================================================
    # Final summary JSON
    # ========================================================

    summary_file = os.path.join(
        CACHE_ROOT,
        "summary.json",
    )

    summary = {

        "protocol": {

            "dataset":
                IMAGENET_ROOT,

            "protocol_file":
                PROTOCOL_FILE,

            "seed":
                SEED,

            "shots":
                SHOT_VALUES,

            "database_images_by_shot":
                {
                    str(k): int(v)

                    for k, v
                    in NUM_DATABASE_BY_SHOT.items()
                },

            "query_images":
                NUM_QUERY,

            "feature":
                "CLS token",

            "normalization":
                "L2",

            "faiss":
                "IndexFlatIP",

            "similarity":
                "cosine",

            "temperature":
                TEMPERATURE,

            "K_values":
                K_VALUES,

        },

        "results":
            all_results,
    }

    with open(
        summary_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        f"Summary saved to:\n"
        f"{summary_file}"
    )

    print()
    print(
        "✓ WebSSL-MAE unified "
        "5/10/20/45-shot KNN completed."
    )


# ============================================================
# 20. Entry
# ============================================================

if __name__ == "__main__":

    main()
