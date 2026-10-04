#!/usr/bin/env python
"""Single-surface DINOv2-style linear probing for visual tokenizers."""

import argparse
import gc
import hashlib
import itertools
import json
import logging
import os
from pathlib import Path
import random
import sys
from vision_encoder_eval.core.runtime import asset_path

import numpy as np
import torch
from torch import nn
from fvcore.common.checkpoint import Checkpointer


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = Path(asset_path('runtime'))
DINO_ROOT = Path(asset_path('third_party', 'dinov2'))
if str(DINO_ROOT) not in sys.path:
    sys.path.insert(0, str(DINO_ROOT))

import dinov2.distributed as distributed
from dinov2.data import SamplerType, make_data_loader, make_dataset
from dinov2.eval.metrics import MetricType, build_metric
from dinov2.eval.utils import evaluate
from dinov2.logging import MetricLogger, setup_logging

from vision_encoder_eval.workers.linear.feature_extractors import (
    DINOV3_CONVNEXT_SPECS,
    DINO_VIT_SPECS,
    EUPE_SPECS,
    PE_SPECS,
    PIXIO_SPECS,
    WEBSSL_DINO_SPECS,
    WEBSSL_MAE_SPECS,
    FeatureBundle,
    load_feature_bundle,
)


LOGGER = logging.getLogger("dinov2")
PROTOCOL_VERSION = "tokenizer_linear_probe_dinov2_single_surface_v2"
MC2_MODEL_NAMES = (
    "mc2_h14_378",
    "mc2_g14_224",
    "mc2_g14_378",
    "mc2_s16_224",
    "mc2_s16_384",
    "mc2_s16_224_mt5",
    "mc2_m16_224",
    "mc2_m16_384",
    "mc2_m16_224_mt5",
    "mc2_b32_224",
    "mc2_b32_384",
    "mc2_b32_224_mt5",
    "mc2_b16_224",
    "mc2_b16_384",
    "mc2_l14_224",
)
MC2_RAW_CHECKPOINT_FILENAMES = {
    "mc2_s16_224": "metaclip2_s16_224px_worldwide.pt",
    "mc2_s16_384": "metaclip2_s16_384px_worldwide.pt",
    "mc2_s16_224_mt5": "metaclip2_s16_224px_mt5_worldwide.pt",
    "mc2_m16_224": "metaclip2_m16_224px_worldwide.pt",
    "mc2_m16_384": "metaclip2_m16_384px_worldwide.pt",
    "mc2_m16_224_mt5": "metaclip2_m16_224px_mt5_worldwide.pt",
    "mc2_b32_224": "metaclip2_b32_224px_worldwide.pt",
    "mc2_b32_384": "metaclip2_b32_384px_worldwide.pt",
    "mc2_b32_224_mt5": "metaclip2_b32_224px_mt5_worldwide.pt",
    "mc2_b16_224": "metaclip2_b16_224px_worldwide.pt",
    "mc2_b16_384": "metaclip2_b16_384px_worldwide.pt",
    "mc2_l14_224": "metaclip2_l14_224px_worldwide.pt",
}
SIGLIP2_MODEL_NAMES = (
    "siglip2_b32_256",
    "siglip2_b16_224",
    "siglip2_b16_256",
    "siglip2_b16_384",
    "siglip2_b16_512",
    "siglip2_l16_256",
    "siglip2_l16_384",
    "siglip2_l16_512",
    "siglip2_sm14_224",
    "siglip2_sm14_384",
    "siglip2_sm16_256",
    "siglip2_sm16_384",
    "siglip2_sm16_512",
    "siglip2_g16_256",
    "siglip2_g16_384",
)
RAEV2_MODEL_NAMES = (
    "dinov3",
    "raev2",
    "raev2_dinov3l_k7",
    "ijepa",
)
NEW_CONTINUOUS_MODEL_NAMES = (
    *PE_SPECS,
    *DINO_VIT_SPECS,
    *DINOV3_CONVNEXT_SPECS,
    *WEBSSL_DINO_SPECS,
    *WEBSSL_MAE_SPECS,
    *EUPE_SPECS,
    *PIXIO_SPECS,
)
MODEL_NAMES = (
    "metaclip",
    "clip_openai__l14",
    "mc1_b32_224_400m",
    "mc1_b16_224_400m",
    "mc1_l14_224_400m",
    "mc1_b32_224_2.5b",
    "mc1_b16_224_2.5b",
    "mc1_l14_224_2.5b",
    "mc1_h14_224_2.5b",
    "mc1_g14_224_2.5b",
    "mc1_h14_224_v1.2",
    *MC2_MODEL_NAMES,
    *SIGLIP2_MODEL_NAMES,
    *RAEV2_MODEL_NAMES,
    *NEW_CONTINUOUS_MODEL_NAMES,
    "toklip_s",
    "toklip_l",
    "unitok",
    "uniar_bsq",
    "vilau",
    "vqgan",
)
OUTPUT_NAMES = {
    "metaclip": "metaclip_b16_2pt5b",
    "clip_openai__l14": "clip_openai__l14",
    "mc1_b32_224_400m": "mc1_b32_224_400m",
    "mc1_b16_224_400m": "mc1_b16_224_400m",
    "mc1_l14_224_400m": "mc1_l14_224_400m",
    "mc1_b32_224_2.5b": "mc1_b32_224_2.5b",
    "mc1_b16_224_2.5b": "mc1_b16_224_2.5b",
    "mc1_l14_224_2.5b": "mc1_l14_224_2.5b",
    "mc1_h14_224_2.5b": "mc1_h14_224_2.5b",
    "mc1_g14_224_2.5b": "mc1_g14_224_2.5b",
    "mc1_h14_224_v1.2": "mc1_h14_224_v1.2",
    **{model_name: model_name for model_name in MC2_MODEL_NAMES},
    **{model_name: model_name for model_name in SIGLIP2_MODEL_NAMES},
    **{model_name: model_name for model_name in RAEV2_MODEL_NAMES},
    **{model_name: model_name for model_name in NEW_CONTINUOUS_MODEL_NAMES},
    **{model_name: f"{model_name}_cls" for model_name in WEBSSL_MAE_SPECS},
    # Preserve the invalid pre-LayerNorm/no-BN Pixio runs and write the corrected
    # MAE-style probing protocol to distinct output directories.
    **{model_name: f"{model_name}_mae_bn" for model_name in PIXIO_SPECS},
    "toklip_s": "toklip_s_semantic_256",
    "toklip_l": "toklip_l_semantic_384",
    "unitok": "unitok",
    "uniar_bsq": "uniar_bsq_final27_256",
    "vilau": "vilau_7b_256_semantic_penultimate",
    "vqgan": "vqgan_imagenet_f16_16384",
}

# Optimization-protocol constants are intentionally not command-line options.
BATCH_SIZE = 1024
# Frozen feature extraction may be split into smaller chunks without changing
# the optimization batch.  The default preserves all existing run protocols.
FEATURE_MICROBATCH_SIZE = 1024
EVAL_BATCH_SIZE = 1024
EPOCHS = 10
EPOCH_LENGTH = 1250
MAX_UPDATES = EPOCHS * EPOCH_LENGTH
EVAL_PERIOD_UPDATES = 1250
SEED = 0
NUM_CLASSES = 1000
BASE_LEARNING_RATES = (
    0.0001,
    0.0002,
    0.0005,
    0.001,
    0.002,
    0.005,
    0.01,
    0.02,
    0.05,
    0.1,
    0.2,
    0.3,
    0.5,
)


def _lr_token(value: float) -> str:
    return format(float(value), ".12g").replace(".", "_").replace("+", "p").replace("-", "m")


class FrozenFeatureModel(nn.Module):
    """Run a frozen encoder in chunks and return one ordinary FP32 batch."""

    def __init__(self, bundle: FeatureBundle, device: torch.device, microbatch_size: int):
        super().__init__()
        self.encoder = bundle.encoder
        self.autocast_context = bundle.autocast_context
        self.device = device
        self.microbatch_size = int(microbatch_size)
        if self.microbatch_size <= 0:
            raise ValueError("Feature microbatch size must be positive")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self.encoder.eval()
        feature_chunks = []
        for image_chunk in images.split(self.microbatch_size, dim=0):
            device_images = image_chunk.to(self.device, non_blocking=True)
            with torch.inference_mode():
                with self.autocast_context():
                    chunk_features = self.encoder(device_images)
            del device_images
            if chunk_features.ndim != 2 or chunk_features.shape[0] != image_chunk.shape[0]:
                raise RuntimeError(
                    "Feature extractor must return [B,D] for every microbatch, "
                    f"got {tuple(chunk_features.shape)}"
                )
            # Clone outside the local inference_mode context so training-time
            # nn.Linear can save this tensor for its weight gradient.
            feature_chunks.append(chunk_features.float().clone())

        features = torch.cat(feature_chunks, dim=0)
        if features.shape[0] != images.shape[0]:
            raise RuntimeError(
                f"Expected {images.shape[0]} concatenated features, got {features.shape[0]}"
            )
        return features


class LinearHead(nn.Module):
    def __init__(self, in_dim: int, base_lr: float, effective_lr: float):
        super().__init__()
        self.base_lr = float(base_lr)
        self.effective_lr = float(effective_lr)
        self.linear = nn.Linear(in_dim, NUM_CLASSES, bias=True)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)


class BatchNormalizedLinearHead(LinearHead):
    """MAE-style non-affine BatchNorm followed by a linear classifier."""

    def __init__(self, in_dim: int, base_lr: float, effective_lr: float):
        super().__init__(in_dim, base_lr, effective_lr)
        self.batch_norm = nn.BatchNorm1d(
            in_dim,
            eps=1e-6,
            momentum=0.1,
            affine=False,
            track_running_stats=True,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(self.batch_norm(features))


class LinearHeadGrid(nn.Module):
    def __init__(self, heads: dict[str, LinearHead]):
        super().__init__()
        self.heads = nn.ModuleDict(heads)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        return {name: head(features) for name, head in self.heads.items()}


class LinearPostprocessor(nn.Module):
    def __init__(self, head: LinearHead):
        super().__init__()
        self.head = head

    def forward(self, features: torch.Tensor, targets: torch.Tensor):
        return {"preds": self.head(features), "target": targets}


def _build_parser():
    model_zoo = Path(asset_path('download', 'tokenizer/discrete'))
    continuous_model_zoo = model_zoo / "continuous"
    image_scripts = Path(asset_path('package', 'workers/tokbench/tokenzier_vae_scripts/image_scripts'))
    parser = argparse.ArgumentParser(
        description="DINOv2-style single-surface ImageNet linear probing for visual tokenizers"
    )
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument("--data-root", default=str(WORKSPACE / "data" / "imagenet1k"))
    parser.add_argument("--extra-root", default=None)
    parser.add_argument(
        "--output-root",
        default=str(WORKSPACE / "outputs" / "vae_linear_probing_dinov2_single_paperlr"),
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--eval-num-workers",
        type=int,
        default=None,
        help=(
            "Validation DataLoader workers. Defaults to --num-workers; set to 0 "
            "to avoid multiprocessing prefetch without changing validation batches."
        ),
    )
    parser.add_argument(
        "--feature-microbatch-size",
        type=int,
        default=FEATURE_MICROBATCH_SIZE,
        help=(
            "Frozen-backbone forward chunk size. The resulting FP32 features are "
            "concatenated before the linear heads; the optimization batch remains 1024."
        ),
    )
    parser.add_argument(
        "--eval-feature-microbatch-size",
        type=int,
        default=None,
        help=(
            "Frozen-backbone chunk size used only during validation. Defaults to "
            "--feature-microbatch-size and does not change the validation batch."
        ),
    )
    parser.add_argument(
        "--stop-after-epoch",
        type=int,
        default=EPOCHS,
        help=(
            "Stop after this epoch while retaining the full 10-epoch cosine "
            "schedule (default: 10). Re-run with a larger value to resume from "
            "the saved epoch checkpoint."
        ),
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--pixio-readout",
        choices=("post-ln", "pre-ln"),
        default="post-ln",
        help=(
            "Pixio CLS-token readout before the non-affine BatchNorm. "
            "The default is post-ln; pre-ln is provided for an ablation."
        ),
    )

    parser.add_argument("--image-scripts", type=Path, default=image_scripts)
    parser.add_argument(
        "--continuous-model-root",
        default=str(continuous_model_zoo),
        help="Root containing the downloaded PE, DINO, and WebSSL model directories.",
    )
    parser.add_argument("--metaclip-model", default="vit_base_patch16_clip_224.metaclip_2pt5b")
    parser.add_argument(
        "--metaclip-checkpoint",
        default=str(model_zoo / "MetaCLIP" / "vit_base_patch16_clip_224.metaclip_2pt5b"),
    )
    parser.add_argument(
        "--clip-openai-model-path",
        default=str(continuous_model_zoo / "clip_openai__l14"),
    )
    parser.add_argument(
        "--clip-meta-model",
        default="vit_large_patch14_clip_224.metaclip_2pt5b",
    )
    parser.add_argument(
        "--clip-meta-checkpoint",
        default=str(continuous_model_zoo / "mc1_l14_224_2.5b"),
    )
    parser.add_argument(
        "--mc1-b32-224-400m-checkpoint",
        default=str(continuous_model_zoo / "mc1_b32_224_400m"),
    )
    parser.add_argument(
        "--mc1-b16-224-400m-checkpoint",
        default=str(continuous_model_zoo / "mc1_b16_224_400m"),
    )
    parser.add_argument(
        "--mc1-l14-224-400m-checkpoint",
        default=str(continuous_model_zoo / "mc1_l14_224_400m"),
    )
    parser.add_argument(
        "--mc1-b32-224-2.5b-checkpoint",
        dest="mc1_b32_224_2_5b_checkpoint",
        default=str(continuous_model_zoo / "mc1_b32_224_2.5b"),
    )
    parser.add_argument(
        "--mc1-b16-224-2.5b-checkpoint",
        dest="mc1_b16_224_2_5b_checkpoint",
        default=str(continuous_model_zoo / "mc1_b16_224_2.5b"),
    )
    parser.add_argument(
        "--mc1-l14-224-2.5b-checkpoint",
        dest="mc1_l14_224_2_5b_checkpoint",
        default=str(continuous_model_zoo / "mc1_l14_224_2.5b"),
    )
    parser.add_argument(
        "--mc1-h14-224-2.5b-checkpoint",
        dest="mc1_h14_224_2_5b_checkpoint",
        default=str(continuous_model_zoo / "mc1_h14_224_2.5b"),
    )
    parser.add_argument(
        "--mc1-g14-224-2.5b-checkpoint",
        dest="mc1_g14_224_2_5b_checkpoint",
        default=str(continuous_model_zoo / "mc1_g14_224_2.5b"),
    )
    parser.add_argument(
        "--mc1-h14-224-v1.2-checkpoint",
        dest="mc1_h14_224_v1_2_checkpoint",
        default=str(continuous_model_zoo / "mc1_h14_224_v1.2"),
    )
    for model_name in MC2_MODEL_NAMES:
        checkpoint = continuous_model_zoo / model_name
        filename = MC2_RAW_CHECKPOINT_FILENAMES.get(model_name)
        if filename is not None:
            checkpoint /= filename
        parser.add_argument(
            f"--{model_name.replace('_', '-')}-checkpoint",
            default=str(checkpoint),
        )
    for model_name in SIGLIP2_MODEL_NAMES:
        parser.add_argument(
            f"--{model_name.replace('_', '-')}-model-path",
            default=str(continuous_model_zoo / model_name),
        )
    parser.add_argument(
        "--raev2-model-root",
        default=str(model_zoo / "RAEv2-models"),
    )
    parser.add_argument(
        "--raev2-path",
        default=str(image_scripts / "RAEv2"),
    )
    parser.add_argument(
        "--dinov3-path",
        default=str(image_scripts / "dinov3"),
    )
    parser.add_argument("--toklip-path", default=str(image_scripts / "TokLIP"))
    parser.add_argument("--toklip-s-checkpoint", default=str(model_zoo / "TokLIP" / "TokLIP_S_256.pt"))
    parser.add_argument("--toklip-l-checkpoint", default=str(model_zoo / "TokLIP" / "TokLIP_L_384.pt"))
    parser.add_argument("--toklip-vq-checkpoint", default=str(model_zoo / "TokLIP" / "vq_ds16_t2i.pt"))
    parser.add_argument("--unitok-path", default=str(image_scripts / "UniTok"))
    parser.add_argument(
        "--unitok-checkpoint",
        default=str(model_zoo / "unitok_20250227" / "unitok_tokenizer.pth"),
    )
    parser.add_argument(
        "--uniar-path",
        default=str(image_scripts / "UniAR"),
        help="Path to the official ShareLab-SII/UniAR source checkout.",
    )
    parser.add_argument(
        "--uniar-checkpoint",
        default=str(model_zoo / "uniar_bsq" / "bsq_encoder"),
        help="Directory containing the standalone UniAR BSQ encoder checkpoint.",
    )
    parser.add_argument(
        "--uniar-image-size",
        type=int,
        default=256,
        help="Square linear-probing crop; must be divisible by patch_size * merge_size (32).",
    )
    parser.add_argument("--vilau-path", default=str(image_scripts / "vila-u"))
    parser.add_argument("--vilau-model-path", default=str(model_zoo / "VILA-U" / "vila-u-7b-256"))
    parser.add_argument(
        "--vilau-siglip-config",
        default=str(model_zoo / "VILA-U" / "siglip-large-patch16-256"),
    )
    vqgan_dir = model_zoo / "taming_vqgan_imagenet_f16_16384"
    parser.add_argument(
        "--vqgan-path",
        default=str(image_scripts / "taming-transformers"),
    )
    parser.add_argument(
        "--vqgan-config",
        default=str(vqgan_dir / "model.yaml"),
    )
    parser.add_argument(
        "--vqgan-checkpoint",
        default=str(vqgan_dir / "last.ckpt"),
    )
    return parser


def _parse_args():
    return _build_parser().parse_args()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _configure_cuda_math() -> None:
    # DINOv2 enables TF32 for CUDA training.  Set it explicitly here because
    # some tokenizer dependencies change these process-global flags on import.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def _atomic_json_dump(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _metrics_history_has_iteration(path: Path, iteration: int) -> bool:
    if not path.is_file():
        return False
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"Invalid JSON in {path} at line {line_number}: {error}"
                ) from error
            if payload.get("iteration") == iteration:
                return True
    return False


def _protocol_fingerprint(protocol: dict) -> str:
    encoded = json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_or_validate_protocol(output_dir: Path, protocol: dict) -> None:
    protocol_path = output_dir / "protocol.json"
    if protocol_path.is_file():
        with protocol_path.open("r", encoding="utf-8") as handle:
            previous = json.load(handle)
        if previous != protocol:
            # Validation-only chunking is an execution detail: every official
            # validation example, outer batch, forward FLOP and metric remains
            # unchanged. It is therefore safe to lower after a validation OOM,
            # including when a completed training checkpoint already exists.
            validation_runtime_keys = {
                "validation_feature_extraction_microbatch_size",
                "fingerprint",
            }
            previous_validation_core = {
                key: value
                for key, value in previous.items()
                if key not in validation_runtime_keys
            }
            protocol_validation_core = {
                key: value
                for key, value in protocol.items()
                if key not in validation_runtime_keys
            }
            if previous_validation_core == protocol_validation_core:
                LOGGER.warning(
                    "Updating validation-only feature microbatch from %s to %s",
                    previous.get("validation_feature_extraction_microbatch_size"),
                    protocol.get("validation_feature_extraction_microbatch_size"),
                )
                _atomic_json_dump(protocol_path, protocol)
                return
            # A feature microbatch only chunks independent frozen-backbone forwards;
            # it does not change the optimization batch or schedule. Permit a failed
            # pre-training attempt to retry with a safer chunk size, but never mix
            # protocols after a checkpoint or metric has been produced.
            ignored_keys = {"feature_extraction_microbatch_size", "fingerprint"}
            previous_core = {
                key: value for key, value in previous.items() if key not in ignored_keys
            }
            protocol_core = {
                key: value for key, value in protocol.items() if key not in ignored_keys
            }
            progress_artifacts = [
                output_dir / "last_checkpoint",
                output_dir / "metrics_history.jsonl",
                output_dir / "results_eval_linear.json",
                *output_dir.glob("*.pth"),
            ]
            if previous_core == protocol_core and not any(
                path.exists() for path in progress_artifacts
            ):
                LOGGER.warning(
                    "Replacing the protocol from an incomplete pre-training attempt "
                    "with feature microbatch %d",
                    protocol["feature_extraction_microbatch_size"],
                )
                _atomic_json_dump(protocol_path, protocol)
                return
            raise RuntimeError(
                "The output directory contains a different protocol.json; "
                "refusing to mix incompatible runs. Choose a new --output-dir."
            )
        return
    _atomic_json_dump(protocol_path, protocol)


def _make_protocol(args, bundle: FeatureBundle, effective_lrs: list[float]) -> dict:
    use_batch_norm = args.model in PIXIO_SPECS
    protocol = {
        "version": PROTOCOL_VERSION,
        "model": args.model,
        "representation": bundle.representation,
        "transform": bundle.transform_description,
        "backbone_precision": bundle.backbone_precision,
        "linear_head_precision": "float32",
        "cuda_matmul_tf32": True,
        "cudnn_tf32": True,
        "checkpoint_paths": bundle.checkpoint_paths,
        "dataset": "ImageNet-1k",
        "single_gpu": True,
        "global_batch_size": BATCH_SIZE,
        "feature_extraction_microbatch_size": args.feature_microbatch_size,
        "validation_feature_extraction_microbatch_size": (
            args.eval_feature_microbatch_size
        ),
        "feature_microbatch_semantics": (
            "frozen backbone only; concatenate one FP32 [global_batch,D] tensor before linear heads"
        ),
        "validation_batch_size": EVAL_BATCH_SIZE,
        "gradient_accumulation_steps": 1,
        "epochs": EPOCHS,
        "epoch_length_updates": EPOCH_LENGTH,
        "max_updates": MAX_UPDATES,
        "eval_period_updates": EVAL_PERIOD_UPDATES,
        "seed": SEED,
        "num_classes": NUM_CLASSES,
        "readout_search": False,
        "multi_block_search": False,
        "feature_normalization": use_batch_norm,
        "loss": "cross_entropy",
        "optimizer": "SGD",
        "momentum": 0.9,
        "weight_decay": 0.0,
        "bias": True,
        "head_weight_init": "normal(mean=0,std=0.01)",
        "head_bias_init": 0.0,
        "lr_schedule": "cosine_to_zero",
        "warmup_updates": 0,
        "lr_scaling": "effective_lr = base_lr * global_batch_size / 256",
        "base_learning_rates": list(BASE_LEARNING_RATES),
        "effective_learning_rates": effective_lrs,
    }
    if use_batch_norm:
        protocol.update(
            {
                "feature_normalization_type": "BatchNorm1d",
                "feature_normalization_placement": (
                    "immediately before each linear classifier"
                ),
                "batch_norm_affine": False,
                "batch_norm_eps": 1e-6,
                "batch_norm_momentum": 0.1,
                "batch_norm_track_running_stats": True,
                "batch_norm_training_batch_size": BATCH_SIZE,
                "pixio_readout_protocol": (
                    f"mean of eight {args.pixio_readout} CLS tokens, followed by "
                    "non-affine BatchNorm"
                ),
            }
        )
    protocol["fingerprint"] = _protocol_fingerprint(protocol)
    return protocol


def _validate_feature_model(feature_model: FrozenFeatureModel, sample: torch.Tensor) -> int:
    features = feature_model(sample)
    if features.ndim != 2 or features.shape[0] != sample.shape[0]:
        raise RuntimeError(f"Feature extractor must return [B,D], got {tuple(features.shape)}")
    if features.dtype != torch.float32:
        raise RuntimeError(f"Feature extractor must return FP32, got {features.dtype}")
    if not torch.isfinite(features).all():
        raise RuntimeError("Feature extractor returned non-finite values")
    if any(parameter.requires_grad for parameter in feature_model.encoder.parameters()):
        raise RuntimeError("The feature encoder contains trainable parameters")
    return int(features.shape[1])


def _load_probe_feature_bundle(args, device: torch.device) -> FeatureBundle:
    """Load DINOv3 ViTs through the local official implementation when needed."""

    spec = DINO_VIT_SPECS.get(args.model)
    if spec is not None and spec[1] == "dinov3_vit":
        cached_script_dir = SCRIPT_DIR
        if str(cached_script_dir) not in sys.path:
            sys.path.insert(0, str(cached_script_dir))
        # This strict converter already supports the locally downloaded DINOv3 HF
        # checkpoints without requiring a Transformers upgrade.
        from vision_encoder_eval.workers.linear.linear_probe_cached import _load_local_dinov3_vit

        return _load_local_dinov3_vit(args, spec, device)
    return load_feature_bundle(args.model, args, device)


def _shutdown_loader_iterator(iterator) -> None:
    """Stop DataLoader workers before validation starts its worker pool."""

    shutdown_workers = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown_workers):
        shutdown_workers()


def _build_heads(
    in_dim: int,
    device: torch.device,
    *,
    use_batch_norm: bool = False,
):
    lr_scale = BATCH_SIZE * distributed.get_global_size() / 256.0
    effective_lrs = [float(base_lr * lr_scale) for base_lr in BASE_LEARNING_RATES]
    heads = {}
    parameter_groups = []
    for base_lr, effective_lr in zip(BASE_LEARNING_RATES, effective_lrs):
        name = f"classifier_fixed_readout_lr_{_lr_token(effective_lr)}"
        if name in heads:
            raise RuntimeError(f"Linear-head name collision: {name}")
        head_type = BatchNormalizedLinearHead if use_batch_norm else LinearHead
        head = head_type(in_dim, base_lr, effective_lr).to(device)
        heads[name] = head
        parameter_groups.append({"params": head.parameters(), "lr": effective_lr})
    if len(heads) != len(BASE_LEARNING_RATES):
        raise RuntimeError(f"Expected 13 linear heads, constructed {len(heads)}")
    return LinearHeadGrid(heads).to(device), parameter_groups, effective_lrs


@torch.no_grad()
def _evaluate_heads(
    feature_model: FrozenFeatureModel,
    head_grid: LinearHeadGrid,
    data_loader,
    iteration: int,
    output_dir: Path,
    metric=None,
    feature_microbatch_size: int | None = None,
):
    if metric is None:
        metric = build_metric(MetricType.MEAN_ACCURACY, num_classes=NUM_CLASSES)
    postprocessors = {name: LinearPostprocessor(head) for name, head in head_grid.heads.items()}
    metrics = {name: metric.clone() for name in head_grid.heads}
    was_training = head_grid.training
    training_microbatch_size = feature_model.microbatch_size
    if feature_microbatch_size is not None:
        feature_model.microbatch_size = feature_microbatch_size
    head_grid.eval()
    try:
        _stats, raw_results = evaluate(
            feature_model,
            data_loader,
            postprocessors,
            metrics,
            torch.cuda.current_device(),
        )
    finally:
        feature_model.microbatch_size = training_microbatch_size
        head_grid.train(was_training)

    classifiers = []
    for name, metric_values in raw_results.items():
        head = head_grid.heads[name]
        classifiers.append(
            {
                "name": name,
                "readout": "fixed_preselected_representation",
                "base_lr": head.base_lr,
                "configured_lr": head.base_lr,
                "effective_lr": head.effective_lr,
                "lr_boundary": (
                    "low"
                    if head.base_lr == min(BASE_LEARNING_RATES)
                    else "high"
                    if head.base_lr == max(BASE_LEARNING_RATES)
                    else "interior"
                ),
                "metrics": {key: float(value.item()) for key, value in metric_values.items()},
            }
        )
    classifiers.sort(key=lambda item: item["base_lr"])
    best = max(classifiers, key=lambda item: item["metrics"]["top-1"])
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "iteration": iteration,
        "best_classifier": {
            "name": best["name"],
            "accuracy": best["metrics"]["top-1"],
            "top5_accuracy": best["metrics"]["top-5"],
            "base_lr": best["base_lr"],
            "configured_lr": best["configured_lr"],
            "effective_lr": best["effective_lr"],
            "lr_boundary": best["lr_boundary"],
            "readout": best["readout"],
        },
        "classifiers": classifiers,
    }
    _atomic_json_dump(output_dir / "results_eval_linear.json", payload)
    with (output_dir / "metrics_history.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
    LOGGER.info("Validation at update %d: best=%s", iteration, payload["best_classifier"])
    return payload


def main() -> int:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This protocol requires one CUDA GPU")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"Expected exactly one visible GPU, found {torch.cuda.device_count()}; "
            "set CUDA_VISIBLE_DEVICES to one device."
        )
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.eval_num_workers is None:
        args.eval_num_workers = args.num_workers
    if args.eval_num_workers < 0:
        raise ValueError("--eval-num-workers must be non-negative")
    if args.feature_microbatch_size <= 0:
        raise ValueError("--feature-microbatch-size must be positive")
    if args.eval_feature_microbatch_size is None:
        args.eval_feature_microbatch_size = args.feature_microbatch_size
    if args.eval_feature_microbatch_size <= 0:
        raise ValueError("--eval-feature-microbatch-size must be positive")
    if not 1 <= args.stop_after_epoch <= EPOCHS:
        raise ValueError(f"--stop-after-epoch must be in [1, {EPOCHS}]")
    if args.model not in PIXIO_SPECS and args.pixio_readout != "post-ln":
        raise ValueError("--pixio-readout pre-ln is only valid for Pixio models")

    distributed.enable(overwrite=True)
    if distributed.get_global_size() != 1:
        raise RuntimeError(f"Expected world_size=1, got {distributed.get_global_size()}")
    _seed_everything(SEED)

    device = torch.device("cuda", torch.cuda.current_device())
    output_name = OUTPUT_NAMES[args.model]
    if args.model in PIXIO_SPECS and args.pixio_readout == "pre-ln":
        output_name = f"{args.model}_pre_ln_bn"
    output_dir = Path(args.output_dir or Path(args.output_root) / output_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output=str(output_dir), level=logging.INFO)

    data_root = Path(args.data_root).expanduser().resolve()
    extra_root = Path(args.extra_root).expanduser().resolve() if args.extra_root else data_root / "extra"
    if not data_root.is_dir():
        raise FileNotFoundError(f"Missing ImageNet root: {data_root}")
    if not extra_root.is_dir():
        raise FileNotFoundError(f"Missing ImageNet extra directory: {extra_root}")

    train_dataset_str = f"ImageNet:split=TRAIN:root={data_root}:extra={extra_root}"
    val_dataset_str = f"ImageNet:split=VAL:root={data_root}:extra={extra_root}"
    bundle = _load_probe_feature_bundle(args, device)
    _configure_cuda_math()
    feature_model = FrozenFeatureModel(
        bundle,
        device=device,
        microbatch_size=args.feature_microbatch_size,
    ).to(device).eval()

    train_dataset = make_dataset(dataset_str=train_dataset_str, transform=bundle.train_transform)
    val_dataset = make_dataset(dataset_str=val_dataset_str, transform=bundle.eval_transform)
    train_targets = np.asarray(train_dataset.get_targets(), dtype=np.int64)
    if len(np.unique(train_targets)) != NUM_CLASSES:
        raise RuntimeError(f"Expected {NUM_CLASSES} ImageNet classes, found {len(np.unique(train_targets))}")

    sample = train_dataset[0][0].unsqueeze(0).to(device)
    in_dim = _validate_feature_model(feature_model, sample)
    head_grid, parameter_groups, effective_lrs = _build_heads(
        in_dim,
        device,
        use_batch_norm=args.model in PIXIO_SPECS,
    )
    if any(parameter.requires_grad for parameter in feature_model.parameters()):
        raise RuntimeError("Frozen feature model unexpectedly has trainable parameters")

    protocol = _make_protocol(args, bundle, effective_lrs)
    _write_or_validate_protocol(output_dir, protocol)
    LOGGER.info("Protocol: %s", json.dumps(protocol, sort_keys=True))
    LOGGER.info("Feature dimension=%d; heads=%d", in_dim, len(head_grid.heads))

    optimizer = torch.optim.SGD(parameter_groups, momentum=0.9, weight_decay=0.0)
    head_parameter_ids = {id(parameter) for parameter in head_grid.parameters()}
    optimizer_parameter_ids = {
        id(parameter)
        for parameter_group in optimizer.param_groups
        for parameter in parameter_group["params"]
    }
    if optimizer_parameter_ids != head_parameter_ids:
        raise RuntimeError("Optimizer parameters must be exactly the 13 linear heads")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, MAX_UPDATES, eta_min=0.0)
    checkpointer = Checkpointer(head_grid, str(output_dir), optimizer=optimizer, scheduler=scheduler)
    checkpoint = checkpointer.resume_or_load("", resume=not args.no_resume)
    start_update = int(checkpoint.get("iteration", -1)) + 1
    stop_update = args.stop_after_epoch * EPOCH_LENGTH
    if start_update < 0 or start_update > MAX_UPDATES:
        raise RuntimeError(f"Invalid checkpoint update: {start_update}")
    if start_update > stop_update:
        raise RuntimeError(
            f"Checkpoint is already at update {start_update}, beyond requested cutoff "
            f"{stop_update}; use a larger --stop-after-epoch or a new --output-dir."
        )

    train_loader = make_data_loader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=args.num_workers,
        shuffle=True,
        seed=SEED,
        sampler_type=SamplerType.SHARDED_INFINITE,
        sampler_advance=start_update * BATCH_SIZE,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = make_data_loader(
        dataset=val_dataset,
        batch_size=EVAL_BATCH_SIZE,
        num_workers=args.eval_num_workers,
        shuffle=False,
        seed=SEED,
        sampler_type=SamplerType.DISTRIBUTED,
        drop_last=False,
        persistent_workers=False,
    )

    LOGGER.info(
        "Starting %s from update %d/%d with requested cutoff epoch=%d "
        "(%d updates), optimization global batch %d, and frozen-backbone "
        "feature microbatch %d (validation %d)",
        args.model,
        start_update,
        MAX_UPDATES,
        args.stop_after_epoch,
        stop_update,
        BATCH_SIZE,
        args.feature_microbatch_size,
        args.eval_feature_microbatch_size,
    )
    if (
        0 < start_update < MAX_UPDATES
        and start_update % EVAL_PERIOD_UPDATES == 0
        and not _metrics_history_has_iteration(
            output_dir / "metrics_history.jsonl", start_update
        )
    ):
        LOGGER.info(
            "Recovered checkpoint is missing validation at update %d; evaluating before training",
            start_update,
        )
        _evaluate_heads(
            feature_model,
            head_grid,
            val_loader,
            start_update,
            output_dir,
            feature_microbatch_size=args.eval_feature_microbatch_size,
        )

    if start_update < stop_update:
        metric_logger = MetricLogger(delimiter="  ")
        train_iterator = iter(train_loader)
        remaining_batches = itertools.islice(
            train_iterator, stop_update - start_update
        )
        update = start_update
        for images, labels in metric_logger.log_every(
            remaining_batches,
            10,
            "Training",
            stop_update,
            start_update,
        ):
            if images.shape[0] != BATCH_SIZE:
                raise RuntimeError(
                    f"Expected optimization batch {BATCH_SIZE}, got {images.shape[0]}"
                )
            labels = labels.to(device, non_blocking=True)
            features = feature_model(images)
            if features.shape[0] != BATCH_SIZE:
                raise RuntimeError(f"Expected feature batch {BATCH_SIZE}, got {features.shape[0]}")
            logits = head_grid(features)
            losses = [nn.functional.cross_entropy(output, labels) for output in logits.values()]
            loss = torch.stack(losses).sum()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            completed_updates = update + 1
            if update % 10 == 0:
                metric_logger.update(loss=float(loss.item()), lr=float(optimizer.param_groups[0]["lr"]))
            if completed_updates % EPOCH_LENGTH == 0 and completed_updates < MAX_UPDATES:
                checkpointer.save("running_checkpoint_linear_eval", iteration=update)
            if (
                completed_updates % EVAL_PERIOD_UPDATES == 0
                and completed_updates < MAX_UPDATES
                and completed_updates != stop_update
            ):
                _evaluate_heads(
                    feature_model,
                    head_grid,
                    val_loader,
                    completed_updates,
                    output_dir,
                    feature_microbatch_size=args.eval_feature_microbatch_size,
                )
            update += 1
            del images, labels, features, logits, losses, loss

        del remaining_batches
        _shutdown_loader_iterator(train_iterator)
        del train_iterator, train_loader
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        LOGGER.info(
            "Released training batches and DataLoader workers before cutoff validation"
        )

    if stop_update < MAX_UPDATES:
        if not _metrics_history_has_iteration(
            output_dir / "metrics_history.jsonl", stop_update
        ):
            _evaluate_heads(
                feature_model,
                head_grid,
                val_loader,
                stop_update,
                output_dir,
                feature_microbatch_size=args.eval_feature_microbatch_size,
            )
        LOGGER.info(
            "Stopped at epoch %d/%d (update %d); checkpoint and validation are "
            "complete. Re-run with a larger --stop-after-epoch to resume.",
            args.stop_after_epoch,
            EPOCHS,
            stop_update,
        )
        return 0

    checkpointer.save("model_final", iteration=MAX_UPDATES - 1)
    final_results = _evaluate_heads(
        feature_model,
        head_grid,
        val_loader,
        MAX_UPDATES,
        output_dir,
        feature_microbatch_size=args.eval_feature_microbatch_size,
    )
    LOGGER.info("Final result: %s", final_results["best_classifier"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
