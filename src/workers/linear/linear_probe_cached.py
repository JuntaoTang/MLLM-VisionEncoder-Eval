#!/usr/bin/env python
"""Batch-normalized ImageNet linear probe with deterministic feature caching."""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import os
from pathlib import Path
import sys
from vision_encoder_eval.core.runtime import asset_path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = Path(asset_path('runtime'))
BASE_SCRIPT_DIR = SCRIPT_DIR
if str(BASE_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_SCRIPT_DIR))

import vision_encoder_eval.workers.linear.probe_core as base
import vision_encoder_eval.workers.linear.feature_extractors as extractors
LOGGER = logging.getLogger("dinov2")
PROTOCOL_VERSION = "tokenizer_linear_probe_deterministic_cached_bn_v1"
CACHE_VERSION = "deterministic_frozen_features_v1"
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE / "outputs" / "vae_linear_probing_dinov2_single_noaug_cached_paperlr_bn"
)
DEFAULT_CACHE_ROOT = DEFAULT_OUTPUT_ROOT / "_feature_cache"

# The requested WebSSL protocol is CLS-only for both DINO and MAE families.
# MAE names already carry this suffix in the baseline; add it to WebSSL-DINO.
for _webssl_dino_model in base.WEBSSL_DINO_SPECS:
    base.OUTPUT_NAMES[_webssl_dino_model] = f"{_webssl_dino_model}_cls"


class BatchNormalizedLinearHead(nn.Module):
    """MAE-style frozen-affine BatchNorm1d followed by a linear classifier."""

    def __init__(self, in_dim: int, base_lr: float, effective_lr: float):
        super().__init__()
        self.base_lr = float(base_lr)
        self.effective_lr = float(effective_lr)
        self.batch_norm = nn.BatchNorm1d(
            in_dim,
            eps=1e-6,
            momentum=0.1,
            affine=False,
            track_running_stats=True,
        )
        self.linear = nn.Linear(in_dim, base.NUM_CLASSES, bias=True)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(self.batch_norm(features))


_BASE_EVALUATE_HEADS = base._evaluate_heads


class IndexedDataset(Dataset):
    """Attach stable dataset indices to samples while building a cache."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, target = self.dataset[index]
        return index, image, target


class CachedIndexDataset(Dataset):
    """The loader samples indices; the collator gathers complete feature batches."""

    def __init__(self, size: int):
        self.size = int(size)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return int(index)


class CachedBatchCollator:
    def __init__(self, features_path: Path, labels_path: Path):
        self.features_path = str(features_path)
        self.labels_path = str(labels_path)
        self._features = None
        self._labels = None

    def _open(self):
        if self._features is None:
            self._features = np.load(self.features_path, mmap_mode="r")
            self._labels = np.load(self.labels_path, mmap_mode="r")

    def __call__(self, indices):
        self._open()
        batch_indices = np.asarray(indices, dtype=np.int64)
        # Advanced indexing creates one writable, contiguous array per batch.
        # That avoids both per-sample copies and PyTorch's read-only NumPy warning.
        features = np.ascontiguousarray(self._features[batch_indices])
        labels = np.ascontiguousarray(self._labels[batch_indices])
        return torch.from_numpy(features), torch.from_numpy(labels)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_features"] = None
        state["_labels"] = None
        return state


class CachedFeatureTransfer(nn.Module):
    """Move cached FP32 features to the active GPU for heads and metrics."""

    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        # The shared evaluator temporarily reads/overrides this attribute on
        # frozen feature models. Cached transfers do not chunk anything, but
        # exposing the same interface keeps evaluation behavior compatible.
        self.microbatch_size = 1

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features.to(self.device, non_blocking=True)


class LocalDINOv3ConvNeXtGlobalEncoder(nn.Module):
    """Official DINOv3 ConvNeXt GAP readout for older Transformers releases."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.model.forward_features(images).get("x_norm_clstoken")
        if not torch.is_tensor(features) or features.ndim != 2:
            shape = None if not torch.is_tensor(features) else tuple(features.shape)
            raise RuntimeError(f"Unexpected DINOv3 ConvNeXt pooled shape: {shape}")
        return features.float()


class LocalDINOv3ViTClsPatchEncoder(nn.Module):
    """Concatenate official DINOv3 ViT CLS and mean patch representations."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        outputs = self.model.forward_features(images)
        cls_token = outputs.get("x_norm_clstoken")
        patch_tokens = outputs.get("x_norm_patchtokens")
        if (
            not torch.is_tensor(cls_token)
            or not torch.is_tensor(patch_tokens)
            or cls_token.ndim != 2
            or patch_tokens.ndim != 3
            or patch_tokens.shape[1] == 0
        ):
            cls_shape = None if not torch.is_tensor(cls_token) else tuple(cls_token.shape)
            patch_shape = (
                None if not torch.is_tensor(patch_tokens) else tuple(patch_tokens.shape)
            )
            raise RuntimeError(
                f"Unexpected DINOv3 ViT features: CLS={cls_shape}, patches={patch_shape}"
            )
        return torch.cat((cls_token, patch_tokens.mean(dim=1)), dim=-1).float()


def _dinov3_safetensor_files(model_dir: Path) -> list[Path]:
    checkpoint = model_dir / "model.safetensors"
    if checkpoint.is_file():
        return [checkpoint]
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing safetensors checkpoint in {model_dir}")
    with index_path.open("r", encoding="utf-8") as handle:
        index = json.load(handle)
    filenames = sorted(set(index.get("weight_map", {}).values()))
    checkpoints = [model_dir / filename for filename in filenames]
    missing = [str(path) for path in checkpoints if not path.is_file()]
    if not checkpoints or missing:
        raise FileNotFoundError(f"Missing DINOv3 checkpoint shards: {missing}")
    return checkpoints


def _load_local_dinov3_vit(args, spec, device: torch.device):
    """Strictly convert HF DINOv3 ViT weights to the local official model."""

    from safetensors import safe_open

    directory, expected_model_type = spec
    model_dir = Path(args.continuous_model_root).expanduser().resolve() / directory
    config_path = model_dir / "config.json"
    processor_path = model_dir / "preprocessor_config.json"
    for required_path in (config_path, processor_path):
        if not required_path.is_file():
            raise FileNotFoundError(f"Missing DINOv3 ViT file: {required_path}")
    checkpoints = _dinov3_safetensor_files(model_dir)

    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("model_type") != expected_model_type:
        raise RuntimeError(
            "DINOv3 ViT model_type mismatch: "
            f"expected {expected_model_type}, got {config.get('model_type')}"
        )
    bias_flags = {
        "q_proj": bool(config["query_bias"]),
        "k_proj": bool(config["key_bias"]),
        "v_proj": bool(config["value_bias"]),
    }
    qkv_bias = any(bias_flags.values())

    dinov3_path = Path(args.dinov3_path).expanduser().resolve()
    source_path = dinov3_path / "dinov3" / "models" / "vision_transformer.py"
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing compatible local DINOv3 source: {source_path}")
    if str(dinov3_path) not in sys.path:
        sys.path.insert(0, str(dinov3_path))
    from dinov3.models.vision_transformer import DinoVisionTransformer

    hidden_size = int(config["hidden_size"])
    num_heads = int(config["num_attention_heads"])
    gated_mlp = bool(config["use_gated_mlp"])
    ffn_ratio = float(config["intermediate_size"]) / hidden_size
    if gated_mlp:
        # The local SwiGLU converts its requested width to 2/3 internally.
        ffn_ratio *= 1.5
    norm_layer = (
        "layernormbf16"
        if float(config["layer_norm_eps"]) == 1e-5
        else "layernorm"
    )
    # Meta initialization avoids allocating a second FP32 copy of ViT-7B.
    with torch.device("meta"):
        model = DinoVisionTransformer(
            img_size=int(config["image_size"]),
            patch_size=int(config["patch_size"]),
            in_chans=int(config["num_channels"]),
            pos_embed_rope_base=float(config["rope_theta"]),
            pos_embed_rope_shift_coords=config.get("pos_embed_shift"),
            pos_embed_rope_jitter_coords=config.get("pos_embed_jitter"),
            pos_embed_rope_rescale_coords=config.get("pos_embed_rescale"),
            pos_embed_rope_dtype="bf16",
            embed_dim=hidden_size,
            depth=int(config["num_hidden_layers"]),
            num_heads=num_heads,
            ffn_ratio=ffn_ratio,
            qkv_bias=qkv_bias,
            drop_path_rate=float(config["drop_path_rate"]),
            layerscale_init=float(config["layerscale_value"]),
            norm_layer=norm_layer,
            ffn_layer="swiglu" if gated_mlp else "mlp",
            ffn_bias=bool(config["mlp_bias"]),
            proj_bias=bool(config["proj_bias"]),
            n_storage_tokens=int(config["num_register_tokens"]),
            mask_k_bias=False,
        )

    converted = {}
    qkv_parts = {}
    seen_hf_keys = set()
    for checkpoint in checkpoints:
        with safe_open(checkpoint, framework="pt", device="cpu") as handle:
            for hf_key in handle.keys():
                if hf_key in seen_hf_keys:
                    raise RuntimeError(f"Duplicate DINOv3 ViT key: {hf_key}")
                seen_hf_keys.add(hf_key)
                tensor = handle.get_tensor(hf_key).to(torch.bfloat16)
                if hf_key == "embeddings.cls_token":
                    local_key = "cls_token"
                elif hf_key == "embeddings.mask_token":
                    local_key = "mask_token"
                    tensor = tensor.squeeze(0)
                elif hf_key == "embeddings.register_tokens":
                    local_key = "storage_tokens"
                elif hf_key.startswith("embeddings.patch_embeddings."):
                    local_key = "patch_embed.proj." + hf_key.rsplit(".", 1)[1]
                elif hf_key.startswith("layer."):
                    parts = hf_key.split(".")
                    layer_index = parts[1]
                    suffix = ".".join(parts[2:])
                    if suffix.startswith("attention.") and any(
                        suffix.startswith(f"attention.{projection}")
                        for projection in ("q_proj", "k_proj", "v_proj")
                    ):
                        projection, parameter = parts[3], parts[4]
                        qkv_parts[(layer_index, parameter, projection)] = tensor
                        continue
                    replacements = {
                        "attention.o_proj.": "attn.proj.",
                        "layer_scale1.lambda1": "ls1.gamma",
                        "layer_scale2.lambda1": "ls2.gamma",
                        "norm1.": "norm1.",
                        "norm2.": "norm2.",
                        "mlp.up_proj.": "mlp.w2." if gated_mlp else "mlp.fc1.",
                        "mlp.gate_proj.": "mlp.w1.",
                        "mlp.down_proj.": "mlp.w3." if gated_mlp else "mlp.fc2.",
                    }
                    local_suffix = None
                    for source_prefix, target_prefix in replacements.items():
                        if suffix.startswith(source_prefix):
                            local_suffix = target_prefix + suffix.removeprefix(source_prefix)
                            break
                    if local_suffix is None:
                        raise RuntimeError(f"Unrecognized DINOv3 ViT key: {hf_key}")
                    local_key = f"blocks.{layer_index}.{local_suffix}"
                elif hf_key.startswith("norm."):
                    local_key = hf_key
                else:
                    raise RuntimeError(f"Unrecognized DINOv3 ViT key: {hf_key}")
                if local_key in converted:
                    raise RuntimeError(f"Duplicate converted DINOv3 ViT key: {local_key}")
                converted[local_key] = tensor

    for layer_index in range(int(config["num_hidden_layers"])):
        layer = str(layer_index)
        q_weight = qkv_parts.pop((layer, "weight", "q_proj"))
        k_weight = qkv_parts.pop((layer, "weight", "k_proj"))
        v_weight = qkv_parts.pop((layer, "weight", "v_proj"))
        converted[f"blocks.{layer}.attn.qkv.weight"] = torch.cat(
            (q_weight, k_weight, v_weight), dim=0
        )
        if qkv_bias:
            bias_parts = []
            for projection in ("q_proj", "k_proj", "v_proj"):
                if bias_flags[projection]:
                    bias_parts.append(
                        qkv_parts.pop((layer, "bias", projection))
                    )
                else:
                    bias_parts.append(
                        torch.zeros(hidden_size, dtype=torch.bfloat16)
                    )
            converted[f"blocks.{layer}.attn.qkv.bias"] = torch.cat(
                bias_parts, dim=0
            )
    if qkv_parts:
        raise RuntimeError(f"Unexpected unconsumed DINOv3 QKV keys: {sorted(qkv_parts)}")

    head_dim = hidden_size // num_heads
    periods = float(config["rope_theta"]) ** (
        2
        * torch.arange(head_dim // 4, dtype=torch.float32)
        / (head_dim // 2)
    )
    converted["rope_embed.periods"] = periods.to(torch.bfloat16)
    expected_state = model.state_dict()
    missing = sorted(set(expected_state) - set(converted))
    unexpected = sorted(set(converted) - set(expected_state))
    mismatched = sorted(
        key
        for key in set(expected_state) & set(converted)
        if expected_state[key].shape != converted[key].shape
    )
    if missing or unexpected or mismatched:
        raise RuntimeError(
            "Failed to strictly convert DINOv3 ViT weights: "
            f"missing={missing}, unexpected={unexpected}, mismatched={mismatched}"
        )
    model.load_state_dict(converted, strict=True, assign=True)
    del converted
    encoder = (
        LocalDINOv3ViTClsPatchEncoder(model)
        .to(device)
        .eval()
        .requires_grad_(False)
    )

    with processor_path.open("r", encoding="utf-8") as handle:
        processor = json.load(handle)
    crop_value = processor.get("crop_size")
    image_size = extractors._processor_square_size(
        crop_value if crop_value is not None else processor["size"],
        "crop_size" if crop_value is not None else "size",
    )
    resize_size = extractors._processor_square_size(processor["size"], "size")
    mean = tuple(float(value) for value in processor["image_mean"])
    std = tuple(float(value) for value in processor["image_std"])
    train_transform = extractors.make_classification_train_transform(
        crop_size=image_size,
        hflip_prob=0.5,
        mean=mean,
        std=std,
    )
    eval_transform = extractors.make_classification_eval_transform(
        resize_size=resize_size,
        crop_size=image_size,
        mean=mean,
        std=std,
    )
    register_count = int(config["num_register_tokens"])
    return extractors.FeatureBundle(
        encoder=encoder,
        train_transform=train_transform,
        eval_transform=eval_transform,
        autocast_context=lambda: torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ),
        representation=(
            "concat(final normalized CLS, mean(final normalized patch tokens)); "
            f"excludes {register_count} register token(s)"
        ),
        transform_description=(
            f"train=RandomResizedCrop({image_size})+HorizontalFlip(0.5), "
            f"eval=Resize({resize_size})+CenterCrop({image_size}), "
            f"mean={mean}, std={std}"
        ),
        backbone_precision="bfloat16 weights and autocast",
        checkpoint_paths=[
            *(str(path.resolve()) for path in checkpoints),
            str(config_path.resolve()),
            str(processor_path.resolve()),
        ],
    )


def _load_local_dinov3_convnext(args, spec, device: torch.device):
    """Strictly convert HF DINOv3 ConvNeXt weights to the local official model."""

    from safetensors import safe_open

    directory, expected_model_type = spec
    model_dir = Path(args.continuous_model_root).expanduser().resolve() / directory
    config_path = model_dir / "config.json"
    processor_path = model_dir / "preprocessor_config.json"
    checkpoint = model_dir / "model.safetensors"
    for required_path in (config_path, processor_path, checkpoint):
        if not required_path.is_file():
            raise FileNotFoundError(f"Missing DINOv3 ConvNeXt file: {required_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("model_type") != expected_model_type:
        raise RuntimeError(
            "DINOv3 ConvNeXt model_type mismatch: "
            f"expected {expected_model_type}, got {config.get('model_type')}"
        )

    dinov3_path = Path(args.dinov3_path).expanduser().resolve()
    source_path = dinov3_path / "dinov3" / "models" / "convnext.py"
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing compatible local DINOv3 source: {source_path}")
    if str(dinov3_path) not in sys.path:
        sys.path.insert(0, str(dinov3_path))
    from dinov3.models.convnext import ConvNeXt

    model = ConvNeXt(
        in_chans=int(config["num_channels"]),
        depths=list(config["depths"]),
        dims=list(config["hidden_sizes"]),
        drop_path_rate=float(config["drop_path_rate"]),
        layer_scale_init_value=float(config["layer_scale_init_value"]),
    )
    converted = {}
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        for hf_key in handle.keys():
            if hf_key.startswith("layer_norm."):
                local_key = "norm." + hf_key.removeprefix("layer_norm.")
            else:
                parts = hf_key.split(".")
                if len(parts) < 5 or parts[0] != "stages":
                    raise RuntimeError(f"Unrecognized DINOv3 ConvNeXt key: {hf_key}")
                stage = parts[1]
                if parts[2] == "downsample_layers":
                    local_key = ".".join(("downsample_layers", stage, *parts[3:]))
                elif parts[2] == "layers":
                    block = parts[3]
                    suffix = ".".join(parts[4:])
                    suffix = suffix.replace("depthwise_conv", "dwconv", 1)
                    suffix = suffix.replace("layer_norm", "norm", 1)
                    suffix = suffix.replace("pointwise_conv1", "pwconv1", 1)
                    suffix = suffix.replace("pointwise_conv2", "pwconv2", 1)
                    local_key = f"stages.{stage}.{block}.{suffix}"
                else:
                    raise RuntimeError(f"Unrecognized DINOv3 ConvNeXt key: {hf_key}")
            if local_key in converted:
                raise RuntimeError(f"Duplicate converted ConvNeXt key: {local_key}")
            converted[local_key] = handle.get_tensor(hf_key)

    # ``norms.3`` aliases the final ``norm`` in the official implementation,
    # so state_dict exposes the same two parameters under both names.
    converted["norms.3.weight"] = converted["norm.weight"]
    converted["norms.3.bias"] = converted["norm.bias"]
    expected_state = model.state_dict()
    missing = sorted(set(expected_state) - set(converted))
    unexpected = sorted(set(converted) - set(expected_state))
    mismatched = sorted(
        key
        for key in set(expected_state) & set(converted)
        if expected_state[key].shape != converted[key].shape
    )
    if missing or unexpected or mismatched:
        raise RuntimeError(
            "Failed to strictly convert DINOv3 ConvNeXt weights: "
            f"missing={missing}, unexpected={unexpected}, mismatched={mismatched}"
        )
    model.load_state_dict(converted, strict=True)
    encoder = (
        LocalDINOv3ConvNeXtGlobalEncoder(model)
        .to(device=device, dtype=torch.bfloat16)
        .eval()
        .requires_grad_(False)
    )

    with processor_path.open("r", encoding="utf-8") as handle:
        processor = json.load(handle)
    crop_value = processor.get("crop_size")
    image_size = extractors._processor_square_size(
        crop_value if crop_value is not None else processor["size"],
        "crop_size" if crop_value is not None else "size",
    )
    resize_size = extractors._processor_square_size(processor["size"], "size")
    mean = tuple(float(value) for value in processor["image_mean"])
    std = tuple(float(value) for value in processor["image_std"])
    train_transform = extractors.make_classification_train_transform(
        crop_size=image_size,
        hflip_prob=0.5,
        mean=mean,
        std=std,
    )
    eval_transform = extractors.make_classification_eval_transform(
        resize_size=resize_size,
        crop_size=image_size,
        mean=mean,
        std=std,
    )
    return extractors.FeatureBundle(
        encoder=encoder,
        train_transform=train_transform,
        eval_transform=eval_transform,
        autocast_context=lambda: torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ),
        representation=(
            "DINOv3 ConvNeXt final-stage GAP after the released final LayerNorm"
        ),
        transform_description=(
            f"train=RandomResizedCrop({image_size})+HorizontalFlip(0.5), "
            f"eval=Resize({resize_size})+CenterCrop({image_size}), "
            f"mean={mean}, std={std}"
        ),
        backbone_precision="bfloat16 weights and autocast",
        checkpoint_paths=[
            str(checkpoint.resolve()),
            str(config_path.resolve()),
            str(processor_path.resolve()),
        ],
    )


def _build_parser():
    parser = base._build_parser()
    parser.description = (
        "DINOv2-style ImageNet BatchNorm linear probing with deterministic "
        "preprocessing and one-time frozen-feature caching"
    )
    parser.set_defaults(output_root=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--cache-root",
        default=str(DEFAULT_CACHE_ROOT),
        help="Persistent root for deterministic train/validation feature caches.",
    )
    parser.add_argument(
        "--cache-batch-size",
        type=int,
        default=base.BATCH_SIZE,
        help="Image loader batch size while extracting each split once.",
    )
    cutoff_action = next(
        (action for action in parser._actions if action.dest == "stop_after_epoch"),
        None,
    )
    if cutoff_action is None:
        parser.add_argument(
            "--stop-after-epoch",
            type=int,
            default=base.EPOCHS,
            help=(
                "Stop at this epoch while retaining the original 10-epoch cosine "
                "schedule (default: 10). A smaller value can be used for screening; "
                "re-run with a larger value to resume the same probe."
            ),
        )
    else:
        cutoff_action.default = base.EPOCHS
        cutoff_action.help = (
            "Stop at this epoch while retaining the full cosine schedule "
            f"(default: {base.EPOCHS})."
        )
    return parser


def _parse_args():
    """Parse arguments through an overridable hook for protocol adapters."""

    return _build_parser().parse_args()


def _linear_head_class(_model_name: str):
    """Return the cached baseline's default BN head implementation."""

    return BatchNormalizedLinearHead


def _json_fingerprint(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _checkpoint_identity(path_string: str) -> dict:
    path = Path(path_string).expanduser().resolve()
    stat = path.stat()
    return {
        "path": str(path),
        "kind": "directory" if path.is_dir() else "file",
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _load_feature_bundle(model_name: str, args, device: torch.device):
    """Use a CLS-only readout for every WebSSL checkpoint."""

    if (
        model_name in extractors.DINO_VIT_SPECS
        and extractors.DINO_VIT_SPECS[model_name][1] == "dinov3_vit"
    ):
        return _load_local_dinov3_vit(
            args,
            extractors.DINO_VIT_SPECS[model_name],
            device,
        )
    if model_name in extractors.DINOV3_CONVNEXT_SPECS:
        return _load_local_dinov3_convnext(
            args,
            extractors.DINOV3_CONVNEXT_SPECS[model_name],
            device,
        )
    if model_name in base.WEBSSL_DINO_SPECS:
        bundle = extractors._load_hf_visual_encoder(
            args.continuous_model_root,
            base.WEBSSL_DINO_SPECS[model_name],
            "cls",
            device,
        )
        bundle.representation = "final normalized WebSSL-DINO encoder CLS token"
        return bundle
    if model_name in base.WEBSSL_MAE_SPECS:
        bundle = extractors._load_hf_visual_encoder(
            args.continuous_model_root,
            base.WEBSSL_MAE_SPECS[model_name],
            "cls",
            device,
        )
        bundle.representation = "final normalized WebSSL-MAE encoder CLS token"
        return bundle
    return base.load_feature_bundle(model_name, args, device)


def _cache_identity(args, bundle, data_root: Path, extra_root: Path) -> dict:
    return {
        "version": CACHE_VERSION,
        "model": args.model,
        "representation": bundle.representation,
        "backbone_precision": bundle.backbone_precision,
        "deterministic_transform": (
            "bundle.eval_transform for both ImageNet TRAIN and VAL; "
            + bundle.transform_description
        ),
        "data_root": str(data_root),
        "extra_root": str(extra_root),
        "checkpoints": [_checkpoint_identity(path) for path in bundle.checkpoint_paths],
        "dtype": "float32",
    }


def _load_valid_cache(cache_dir: Path, identity: dict):
    metadata_path = cache_dir / "metadata.json"
    if not metadata_path.is_file():
        return None
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("identity") != identity:
        return None
    for split in ("train", "val"):
        split_metadata = metadata["splits"][split]
        features_path = cache_dir / f"{split}_features.npy"
        labels_path = cache_dir / f"{split}_labels.npy"
        if not features_path.is_file() or not labels_path.is_file():
            return None
        features = np.load(features_path, mmap_mode="r")
        labels = np.load(labels_path, mmap_mode="r")
        if list(features.shape) != split_metadata["feature_shape"]:
            return None
        if list(labels.shape) != split_metadata["label_shape"]:
            return None
        if features.dtype != np.float32 or labels.dtype != np.int64:
            return None
    return metadata


@torch.no_grad()
def _extract_split(
    *,
    split: str,
    dataset,
    feature_model: base.FrozenFeatureModel,
    feature_dim: int,
    cache_dir: Path,
    batch_size: int,
    num_workers: int,
) -> dict:
    feature_tmp = cache_dir / f".{split}_features.npy.tmp"
    label_tmp = cache_dir / f".{split}_labels.npy.tmp"
    features_path = cache_dir / f"{split}_features.npy"
    labels_path = cache_dir / f"{split}_labels.npy"
    feature_array = np.lib.format.open_memmap(
        feature_tmp,
        mode="w+",
        dtype=np.float32,
        shape=(len(dataset), feature_dim),
    )
    label_array = np.lib.format.open_memmap(
        label_tmp,
        mode="w+",
        dtype=np.int64,
        shape=(len(dataset),),
    )
    loader = DataLoader(
        IndexedDataset(dataset),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )
    LOGGER.info(
        "Extracting deterministic %s features once: samples=%d batches=%d",
        split,
        len(dataset),
        len(loader),
    )
    for batch_index, (indices, images, labels) in enumerate(loader):
        batch_features = feature_model(images).detach().cpu().numpy()
        if batch_features.dtype != np.float32:
            raise RuntimeError(f"Expected FP32 cached features, got {batch_features.dtype}")
        if batch_features.shape != (len(indices), feature_dim):
            raise RuntimeError(
                f"Unexpected {split} feature shape {batch_features.shape}; "
                f"expected {(len(indices), feature_dim)}"
            )
        if not np.isfinite(batch_features).all():
            raise RuntimeError(f"Non-finite values in {split} feature batch {batch_index}")
        numpy_indices = indices.numpy()
        feature_array[numpy_indices] = batch_features
        label_array[numpy_indices] = labels.numpy()
        if batch_index % 100 == 0:
            LOGGER.info(
                "Cached %s batch %d/%d",
                split,
                batch_index + 1,
                len(loader),
            )

    expected_labels = np.asarray(dataset.get_targets(), dtype=np.int64)
    if not np.array_equal(label_array, expected_labels):
        raise RuntimeError(f"Cached {split} labels do not match dataset targets")
    feature_array.flush()
    label_array.flush()
    del feature_array, label_array, loader
    os.replace(feature_tmp, features_path)
    os.replace(label_tmp, labels_path)
    return {
        "feature_shape": [len(dataset), feature_dim],
        "label_shape": [len(dataset)],
    }


def _prepare_cache(args, bundle, feature_model, train_dataset, val_dataset, identity, cache_dir):
    metadata = _load_valid_cache(cache_dir, identity)
    if metadata is not None:
        LOGGER.info("Reusing complete deterministic feature cache: %s", cache_dir)
        return metadata

    sample = train_dataset[0][0].unsqueeze(0)
    feature_dim = base._validate_feature_model(feature_model, sample)
    cache_dir.mkdir(parents=True, exist_ok=True)
    split_metadata = {}
    split_metadata["train"] = _extract_split(
        split="train",
        dataset=train_dataset,
        feature_model=feature_model,
        feature_dim=feature_dim,
        cache_dir=cache_dir,
        batch_size=args.cache_batch_size,
        num_workers=args.num_workers,
    )
    split_metadata["val"] = _extract_split(
        split="val",
        dataset=val_dataset,
        feature_model=feature_model,
        feature_dim=feature_dim,
        cache_dir=cache_dir,
        batch_size=args.cache_batch_size,
        num_workers=args.num_workers,
    )
    metadata = {
        "identity": identity,
        "fingerprint": _json_fingerprint(identity),
        "splits": split_metadata,
    }
    base._atomic_json_dump(cache_dir / "metadata.json", metadata)
    LOGGER.info("Completed deterministic feature cache: %s", cache_dir)
    return metadata


def _make_protocol(args, bundle, effective_lrs, cache_fingerprint: str) -> dict:
    protocol = base._make_protocol(args, bundle, effective_lrs)
    protocol.pop("fingerprint", None)
    protocol.update(
        {
            "version": PROTOCOL_VERSION,
            "transform": (
                "train=bundle.eval_transform (deterministic, no augmentation); "
                "val=bundle.eval_transform; original bundle: "
                + bundle.transform_description
            ),
            "data_augmentation": False,
            "feature_extraction": "once per split, then persistent FP32 cache reuse",
            "feature_cache_dtype": "float32",
            "feature_cache_fingerprint": cache_fingerprint,
            "feature_normalization": True,
            "feature_normalization_type": "BatchNorm1d",
            "feature_normalization_placement": "immediately before each linear classifier",
            "batch_norm_affine": False,
            "batch_norm_fixed_scale": 1.0,
            "batch_norm_fixed_shift": 0.0,
            "batch_norm_eps": 1e-6,
            "batch_norm_momentum": 0.1,
            "batch_norm_track_running_stats": True,
            "batch_norm_training_batch_size": base.BATCH_SIZE,
            "batch_norm_applied_after_feature_cache_load": True,
            "training_cutoff_semantics": (
                "--stop-after-epoch is an execution cutoff; optimizer and cosine "
                "scheduler retain the full 10-epoch/12500-update horizon"
            ),
        }
    )
    protocol["fingerprint"] = base._protocol_fingerprint(protocol)
    return protocol


@torch.no_grad()
def _evaluate_heads_with_running_stats(
    feature_model,
    head_grid,
    data_loader,
    iteration: int,
    output_dir: Path,
):
    """Evaluate BN heads with running stats, then restore their prior mode."""

    was_training = head_grid.training
    head_grid.eval()
    try:
        return _BASE_EVALUATE_HEADS(
            feature_model,
            head_grid,
            data_loader,
            iteration,
            output_dir,
        )
    finally:
        head_grid.train(was_training)


def _make_cached_loader(
    *,
    features_path: Path,
    labels_path: Path,
    size: int,
    batch_size: int,
    num_workers: int,
    train: bool,
    sampler_advance: int = 0,
):
    return base.make_data_loader(
        dataset=CachedIndexDataset(size),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=train,
        seed=base.SEED,
        sampler_type=(base.SamplerType.SHARDED_INFINITE if train else base.SamplerType.DISTRIBUTED),
        sampler_advance=sampler_advance,
        drop_last=train,
        persistent_workers=num_workers > 0,
        collate_fn=CachedBatchCollator(features_path, labels_path),
    )


def _read_latest_result(output_dir: Path):
    path = output_dir / "results_eval_linear.json"
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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
    if args.feature_microbatch_size <= 0 or args.cache_batch_size <= 0:
        raise ValueError("Feature microbatch and cache batch sizes must be positive")
    if not 1 <= args.stop_after_epoch <= base.EPOCHS:
        raise ValueError(f"--stop-after-epoch must be in [1, {base.EPOCHS}]")

    base.PROTOCOL_VERSION = PROTOCOL_VERSION
    base.LinearHead = _linear_head_class(args.model)
    base.distributed.enable(overwrite=True)
    if base.distributed.get_global_size() != 1:
        raise RuntimeError(f"Expected world_size=1, got {base.distributed.get_global_size()}")
    base._seed_everything(base.SEED)
    device = torch.device("cuda", torch.cuda.current_device())

    output_dir = Path(
        args.output_dir or Path(args.output_root) / base.OUTPUT_NAMES[args.model]
    ).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base.setup_logging(output=str(output_dir), level=logging.INFO)

    data_root = Path(args.data_root).expanduser().resolve()
    extra_root = Path(args.extra_root).expanduser().resolve() if args.extra_root else data_root / "extra"
    if not data_root.is_dir():
        raise FileNotFoundError(f"Missing ImageNet root: {data_root}")
    if not extra_root.is_dir():
        raise FileNotFoundError(f"Missing ImageNet extra directory: {extra_root}")

    train_dataset_str = f"ImageNet:split=TRAIN:root={data_root}:extra={extra_root}"
    val_dataset_str = f"ImageNet:split=VAL:root={data_root}:extra={extra_root}"
    bundle = _load_feature_bundle(args.model, args, device)
    base._configure_cuda_math()
    feature_model = base.FrozenFeatureModel(
        bundle,
        device=device,
        microbatch_size=args.feature_microbatch_size,
    ).to(device).eval()
    # The key change: TRAIN uses the same deterministic preprocessing as VAL.
    train_dataset = base.make_dataset(dataset_str=train_dataset_str, transform=bundle.eval_transform)
    val_dataset = base.make_dataset(dataset_str=val_dataset_str, transform=bundle.eval_transform)
    train_targets = np.asarray(train_dataset.get_targets(), dtype=np.int64)
    if len(np.unique(train_targets)) != base.NUM_CLASSES:
        raise RuntimeError(
            f"Expected {base.NUM_CLASSES} ImageNet classes, found {len(np.unique(train_targets))}"
        )

    identity = _cache_identity(args, bundle, data_root, extra_root)
    cache_fingerprint = _json_fingerprint(identity)
    cache_dir = (
        Path(args.cache_root).expanduser().resolve()
        / base.OUTPUT_NAMES[args.model]
        / cache_fingerprint[:16]
    )
    metadata = _prepare_cache(
        args,
        bundle,
        feature_model,
        train_dataset,
        val_dataset,
        identity,
        cache_dir,
    )
    feature_dim = int(metadata["splits"]["train"]["feature_shape"][1])
    if metadata["splits"]["val"]["feature_shape"][1] != feature_dim:
        raise RuntimeError("Train and validation cache feature dimensions differ")

    # Release the tokenizer before allocating the LR grid; it is never used again.
    del feature_model, train_dataset, val_dataset
    bundle.encoder = nn.Identity()
    torch.cuda.empty_cache()

    head_grid, parameter_groups, effective_lrs = base._build_heads(feature_dim, device)
    protocol = _make_protocol(args, bundle, effective_lrs, metadata["fingerprint"])
    base._write_or_validate_protocol(output_dir, protocol)
    LOGGER.info("Protocol: %s", json.dumps(protocol, sort_keys=True))
    LOGGER.info("Cached feature dimension=%d; heads=%d", feature_dim, len(head_grid.heads))

    optimizer = torch.optim.SGD(parameter_groups, momentum=0.9, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, base.MAX_UPDATES, eta_min=0.0
    )
    checkpointer = base.Checkpointer(
        head_grid,
        str(output_dir),
        optimizer=optimizer,
        scheduler=scheduler,
    )
    checkpoint = checkpointer.resume_or_load("", resume=not args.no_resume)
    start_update = int(checkpoint.get("iteration", -1)) + 1
    stop_update = args.stop_after_epoch * base.EPOCH_LENGTH
    if start_update < 0 or start_update > base.MAX_UPDATES:
        raise RuntimeError(f"Invalid checkpoint update: {start_update}")
    if start_update > stop_update:
        raise RuntimeError(
            f"Checkpoint is already at update {start_update}, beyond requested cutoff "
            f"{stop_update}; use a larger --stop-after-epoch or a new --output-dir."
        )

    train_size = int(metadata["splits"]["train"]["feature_shape"][0])
    val_size = int(metadata["splits"]["val"]["feature_shape"][0])
    train_loader = _make_cached_loader(
        features_path=cache_dir / "train_features.npy",
        labels_path=cache_dir / "train_labels.npy",
        size=train_size,
        batch_size=base.BATCH_SIZE,
        num_workers=args.num_workers,
        train=True,
        sampler_advance=start_update * base.BATCH_SIZE,
    )
    val_loader = _make_cached_loader(
        features_path=cache_dir / "val_features.npy",
        labels_path=cache_dir / "val_labels.npy",
        size=val_size,
        batch_size=base.EVAL_BATCH_SIZE,
        num_workers=args.num_workers,
        train=False,
    )
    cached_feature_model = CachedFeatureTransfer(device).eval()
    metrics_path = output_dir / "metrics_history.jsonl"

    LOGGER.info(
        "Starting cached %s from update %d/%d; requested cutoff epoch=%d (%d updates)",
        args.model,
        start_update,
        base.MAX_UPDATES,
        args.stop_after_epoch,
        stop_update,
    )
    if (
        start_update > 0
        and start_update % base.EPOCH_LENGTH == 0
        and not base._metrics_history_has_iteration(metrics_path, start_update)
    ):
        _evaluate_heads_with_running_stats(
            cached_feature_model,
            head_grid,
            val_loader,
            start_update,
            output_dir,
        )

    if start_update < stop_update:
        metric_logger = base.MetricLogger(delimiter="  ")
        remaining_batches = itertools.islice(train_loader, stop_update - start_update)
        update = start_update
        for features, labels in metric_logger.log_every(
            remaining_batches,
            10,
            "Cached training",
            stop_update,
            start_update,
        ):
            labels = labels.to(device, non_blocking=True)
            features = cached_feature_model(features)
            logits = head_grid(features)
            losses = [nn.functional.cross_entropy(output, labels) for output in logits.values()]
            loss = torch.stack(losses).sum()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            completed_updates = update + 1
            if update % 10 == 0:
                metric_logger.update(
                    loss=float(loss.item()),
                    lr=float(optimizer.param_groups[0]["lr"]),
                )
            if completed_updates % base.EPOCH_LENGTH == 0:
                if completed_updates < base.MAX_UPDATES:
                    checkpointer.save("running_checkpoint_linear_eval", iteration=update)
                if not base._metrics_history_has_iteration(metrics_path, completed_updates):
                    _evaluate_heads_with_running_stats(
                        cached_feature_model,
                        head_grid,
                        val_loader,
                        completed_updates,
                        output_dir,
                    )
            update += 1

    if stop_update == base.MAX_UPDATES:
        checkpointer.save("model_final", iteration=base.MAX_UPDATES - 1)
    result = _read_latest_result(output_dir)
    if result is None or result.get("iteration") != stop_update:
        raise RuntimeError(f"Missing validation result at cutoff update {stop_update}")
    LOGGER.info(
        "Stopped at epoch %d/%d: %s",
        args.stop_after_epoch,
        base.EPOCHS,
        result["best_classifier"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
