import json
import os

import torch
import torch.nn as nn
from transformers import CLIPImageProcessor
from llava.utils import rank0_print

try:
    import open_clip
    import torchvision
    from open_clip.pretrained import get_pretrained_cfg
    from open_clip.transformer import _expand_token
except ImportError:
    print("OpenCLIP not installed")
    open_clip = None
    get_pretrained_cfg = None

HIDDEN_SIZE_DICT = {
    "ViT-L-14": 1024,
    "ViT-H-14-378-quickgelu": 1280,
}

DEFAULT_CLIP_PROCESSOR_PATH = "/cache/VTB/model/visual_encoder/clip-vit-large-patch14"

# Official SigLIP / SigLIP2 (open_clip webli) preprocessing.
_SIGLIP_IMAGE_MEAN = (0.5, 0.5, 0.5)
_SIGLIP_IMAGE_STD = (0.5, 0.5, 0.5)


def _is_siglip_model(model_name: str) -> bool:
    name_l = model_name.lower()
    return "siglip" in name_l or "so400m" in name_l


def _resolve_clip_processor_path() -> str:
    for key in ("VTB_CLIP_IMAGE_PROCESSOR", "VTB_CLIP_PROCESSOR_PATH"):
        path = os.environ.get(key)
        if path and os.path.isdir(path):
            return path
    if os.path.isdir(DEFAULT_CLIP_PROCESSOR_PATH):
        return DEFAULT_CLIP_PROCESSOR_PATH
    return "openai/clip-vit-large-patch14"


def _quick_gelu_from_hf_clip_config(processor_path: str) -> bool | None:
    config_path = os.path.join(processor_path, "config.json")
    if not os.path.isfile(config_path):
        return None
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)
    hidden_act = cfg.get("vision_config", {}).get("hidden_act")
    if hidden_act is None:
        return None
    return hidden_act == "quick_gelu"


def _resolve_force_quick_gelu(model_name: str, pretrained: str, processor_path: str) -> bool:
    env_override = os.environ.get("VTB_FORCE_QUICK_GELU")
    if env_override is not None:
        return env_override.lower() in ("1", "true", "yes")

    # SigLIP / timm trunks use GELU, not QuickGELU. Do not inherit from CLIP processor.
    if _is_siglip_model(model_name) or model_name.lower().startswith("timm/"):
        return False

    if get_pretrained_cfg is not None and isinstance(pretrained, str) and pretrained and not os.path.isfile(pretrained):
        try:
            pretrained_cfg = get_pretrained_cfg(model_name, pretrained)
            if pretrained_cfg is not None and pretrained_cfg.get("quick_gelu") is not None:
                return bool(pretrained_cfg["quick_gelu"])
        except Exception:
            pass

    # Local MetaCLIP / OpenAI .pt often expects QuickGELU; prefer CLIP processor hint.
    from_config = _quick_gelu_from_hf_clip_config(processor_path)
    if from_config is not None and not _is_siglip_model(model_name):
        return from_config
    return False


def _pretrained_cfg_for_local_weights(model_name: str) -> dict | None:
    """Look up open_clip tag preprocess when `pretrained` is a local file path.

    Passing a checkpoint path skips open_clip's tag→preprocess merge, so SigLIP
    (and similar) fall back to OpenAI CLIP mean/std + center-crop. Recover the
    official tag cfg (usually `webli` for SigLIP2).
    """
    if get_pretrained_cfg is None:
        return None
    tag = os.environ.get("VTB_OPENCLIP_PRETRAINED_TAG")
    tags: list[str] = []
    if tag:
        tags.append(tag)
    if _is_siglip_model(model_name):
        tags.append("webli")
    try:
        from open_clip.pretrained import list_pretrained_tags_by_model

        for t in list_pretrained_tags_by_model(model_name) or []:
            if t not in tags:
                tags.append(t)
    except Exception:
        pass
    for t in tags:
        try:
            cfg = get_pretrained_cfg(model_name, t)
        except Exception:
            cfg = None
        if cfg and any(cfg.get(k) is not None for k in ("mean", "std", "resize_mode", "interpolation")):
            return cfg
    return None


def _resolve_image_preprocess_kwargs(model_name: str, pretrained: str) -> dict:
    """Kwargs for create_model_and_transforms image preprocessing."""
    # Non-file tags already merge preprocess inside open_clip.
    if not (isinstance(pretrained, str) and os.path.isfile(pretrained)):
        # Still force SigLIP defaults if someone passes pretrained=None.
        if pretrained is None and _is_siglip_model(model_name):
            return dict(
                image_mean=_SIGLIP_IMAGE_MEAN,
                image_std=_SIGLIP_IMAGE_STD,
                image_interpolation="bicubic",
                image_resize_mode="squash",
            )
        return {}

    cfg = _pretrained_cfg_for_local_weights(model_name)
    if cfg:
        out = {}
        if cfg.get("mean") is not None:
            out["image_mean"] = tuple(cfg["mean"])
        if cfg.get("std") is not None:
            out["image_std"] = tuple(cfg["std"])
        if cfg.get("interpolation") is not None:
            out["image_interpolation"] = cfg["interpolation"]
        if cfg.get("resize_mode") is not None:
            out["image_resize_mode"] = cfg["resize_mode"]
        if out:
            return out

    if _is_siglip_model(model_name):
        return dict(
            image_mean=_SIGLIP_IMAGE_MEAN,
            image_std=_SIGLIP_IMAGE_STD,
            image_interpolation="bicubic",
            image_resize_mode="squash",
        )
    return {}


def _as_hw_size(size) -> tuple[int, int]:
    if isinstance(size, (tuple, list)):
        if len(size) == 1:
            s = int(size[0])
            return s, s
        return int(size[0]), int(size[1])
    s = int(size)
    return s, s


def _build_hf_image_processor(
    processor_path: str,
    *,
    resize_size,
    image_mean,
    image_std,
    resize_mode: str | None,
    use_squash: bool,
):
    """Map open_clip transforms onto a HF CLIPImageProcessor used by LLaVA data code."""
    h, w = _as_hw_size(resize_size)
    processor_kwargs = dict(
        image_mean=list(image_mean),
        image_std=list(image_std),
        do_resize=True,
        do_normalize=True,
        do_rescale=True,
    )
    if use_squash or resize_mode == "squash":
        # SigLIP / webli: direct resize to HxW, no center crop.
        processor_kwargs.update(
            size={"height": h, "width": w},
            crop_size={"height": h, "width": w},
            do_center_crop=False,
        )
    else:
        # CLIP-style shortest-edge resize + center crop.
        edge = min(h, w)
        processor_kwargs.update(
            size={"shortest_edge": edge},
            crop_size={"height": h, "width": w},
            do_center_crop=True,
        )
    if os.path.isdir(processor_path):
        processor_kwargs["local_files_only"] = True
    return CLIPImageProcessor.from_pretrained(processor_path, **processor_kwargs)


def _should_use_weights_only(pretrained) -> bool:
    """MetaCLIP / custom .pt dumps may embed numpy types; disable weights_only."""
    if isinstance(pretrained, str) and os.path.isfile(pretrained):
        return False
    return True


def _strip_module_prefix(state_dict: dict) -> dict:
    if not state_dict:
        return state_dict
    if all(k.startswith("module.") for k in state_dict):
        return {k[len("module.") :]: v for k, v in state_dict.items()}
    return {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}


def _local_checkpoint_state_dict(pretrained: str) -> dict | None:
    if not (isinstance(pretrained, str) and os.path.isfile(pretrained)):
        return None
    try:
        from open_clip.factory import load_state_dict

        return _strip_module_prefix(load_state_dict(pretrained, weights_only=False))
    except Exception:
        try:
            ckpt = torch.load(pretrained, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]
            if isinstance(ckpt, dict):
                return _strip_module_prefix(ckpt)
        except Exception:
            return None
    return None


def _infer_force_image_size(model_name: str, state_dict: dict | None) -> int | None:
    env = os.environ.get("VTB_FORCE_IMAGE_SIZE")
    if env:
        try:
            return int(env)
        except ValueError:
            pass
    if not state_dict:
        return None
    pos = state_dict.get("visual.positional_embedding")
    if pos is None or not hasattr(pos, "shape") or pos.ndim != 2:
        return None
    # grid^2 + 1 class token (standard CLIP ViT). SigLIP/timm trunks are skipped.
    n = int(pos.shape[0]) - 1
    if n <= 0:
        return None
    grid = int(round(n**0.5))
    if grid * grid != n:
        return None
    patch = None
    name_l = model_name.lower()
    for token in ("-14", "_14", "patch14"):
        if token in name_l:
            patch = 14
            break
    if patch is None:
        for token in ("-16", "_16", "patch16"):
            if token in name_l:
                patch = 16
                break
    if patch is None:
        for token in ("-32", "_32", "patch32"):
            if token in name_l:
                patch = 32
                break
    if patch is None:
        return None
    return grid * patch


def _infer_text_vocab_size(state_dict: dict | None) -> int | None:
    if not state_dict:
        return None
    token = state_dict.get("token_embedding.weight")
    if token is not None and hasattr(token, "shape") and getattr(token, "ndim", 0) == 2:
        return int(token.shape[0])
    return None


def _create_open_clip_with_text_cfg(
    model_name: str,
    pretrained: str,
    force_quick_gelu: bool,
    state_dict: dict | None,
    preprocess_kwargs: dict | None = None,
):
    """Retry full checkpoint load after aligning text vocab_size from the dump.

    MetaCLIP-2 distilled dumps keep a CLIP visual tower but use XLM-V / mT5 text
    vocab sizes that stock OpenCLIP model configs do not match.
    """
    from copy import deepcopy

    from open_clip.factory import get_model_config

    vocab = _infer_text_vocab_size(state_dict)
    if vocab is None:
        raise RuntimeError(f"Cannot infer text vocab size from {pretrained}")
    model_cfg = get_model_config(model_name)
    if model_cfg is None:
        raise RuntimeError(f"Unknown OpenCLIP model config: {model_name}")
    text_cfg = deepcopy(model_cfg["text_cfg"])
    if int(text_cfg.get("vocab_size", -1)) == vocab:
        raise RuntimeError(
            f"text_cfg vocab already matches ({vocab}); full load should have succeeded"
        )
    text_cfg["vocab_size"] = vocab
    force_image_size = _infer_force_image_size(model_name, state_dict)
    create_kwargs = dict(
        model_name=model_name,
        pretrained=pretrained,
        force_quick_gelu=force_quick_gelu,
        precision="fp32",
        device="cpu",
        weights_only=False,
        require_pretrained=True,
        text_cfg=text_cfg,
    )
    if force_image_size is not None:
        create_kwargs["force_image_size"] = force_image_size
        rank0_print(f"force_image_size: {force_image_size} (from checkpoint / env)")
    if preprocess_kwargs:
        create_kwargs.update(preprocess_kwargs)
    rank0_print(f"Retrying full load with text_cfg.vocab_size={vocab}")
    vision_tower, _, image_processor = open_clip.create_model_and_transforms(**create_kwargs)
    return vision_tower, image_processor


def _create_open_clip_with_local_visual(
    model_name: str,
    pretrained: str,
    force_quick_gelu: bool,
    preprocess_kwargs: dict | None = None,
):
    """Create OpenCLIP and load only visual.* weights from a local checkpoint.

    MetaCLIP-2 dumps use a worldwide text vocab (≈901k) that does not match stock
    OpenCLIP configs for most ViT sizes. LLaVA only needs the visual tower, so we
    load matching visual tensors and ignore text mismatches.
    """
    state_dict = _local_checkpoint_state_dict(pretrained)
    force_image_size = _infer_force_image_size(model_name, state_dict)
    # pretrained=None skips tag preprocess; apply overrides (esp. SigLIP webli).
    pp = dict(preprocess_kwargs or {})
    if not pp:
        pp = _resolve_image_preprocess_kwargs(model_name, pretrained)
    create_kwargs = dict(
        model_name=model_name,
        pretrained=None,
        force_quick_gelu=force_quick_gelu,
        precision="fp32",
        device="cpu",
    )
    if force_image_size is not None:
        create_kwargs["force_image_size"] = force_image_size
        rank0_print(f"force_image_size: {force_image_size} (from checkpoint / env)")
    if pp:
        create_kwargs.update(pp)

    vision_tower, _, image_processor = open_clip.create_model_and_transforms(**create_kwargs)
    if state_dict is None:
        raise RuntimeError(f"Could not read local checkpoint {pretrained}")

    model_sd = vision_tower.state_dict()
    filtered = {}
    skipped = 0
    for key, value in state_dict.items():
        if not key.startswith("visual."):
            continue
        if key in model_sd and model_sd[key].shape == value.shape:
            filtered[key] = value
        else:
            skipped += 1
    incompat = vision_tower.load_state_dict(filtered, strict=False)
    missing_visual = [k for k in incompat.missing_keys if k.startswith("visual.")]
    rank0_print(
        f"Loaded visual weights from {pretrained}: "
        f"{len(filtered)} tensors (skipped_shape={skipped}, "
        f"missing_visual={len(missing_visual)}, unexpected={len(incompat.unexpected_keys)})"
    )
    if len(filtered) == 0:
        raise RuntimeError(f"No visual weights loaded from {pretrained} into {model_name}")
    if missing_visual:
        raise RuntimeError(
            f"Incomplete visual load from {pretrained} into {model_name}: "
            f"missing {missing_visual[:5]}"
        )
    return vision_tower, image_processor


def _should_visual_only_metaclip_load(state_dict: dict | None) -> bool:
    """MetaCLIP-2 worldwide dumps use text vocab sizes stock OpenCLIP cannot load."""
    vocab = _infer_text_vocab_size(state_dict)
    return vocab is not None and vocab > 100_000


class OpenCLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.model_name = vision_tower.replace("open_clip_hub:", "")
        self.pretrained = args.vision_tower_pretrained
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")

        if not delay_load:
            rank0_print(f"Loading vision tower: {vision_tower}")
            self.load_model()
        elif getattr(args, "unfreeze_mm_vision_tower", False):
            # TODO: better detector is needed.
            rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `unfreeze_mm_vision_tower`: True.")
            self.load_model()
        elif hasattr(args, "mm_tunable_parts") and "mm_vision_tower" in args.mm_tunable_parts:
            rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `mm_tunable_parts` contains `mm_vision_tower`.")
            self.load_model()

    def load_model(self, device_map="auto"):
        pretrained = os.environ.get("VTB_VISION_WEIGHTS") or self.pretrained
        processor_path = _resolve_clip_processor_path()
        force_quick_gelu = _resolve_force_quick_gelu(self.model_name, pretrained, processor_path)
        preprocess_kwargs = _resolve_image_preprocess_kwargs(self.model_name, pretrained)

        rank0_print(f"Loading OpenCLIP model: {self.model_name}")
        rank0_print(f"Pretrained: {pretrained}")
        rank0_print(f"force_quick_gelu: {force_quick_gelu} (from config)")
        if preprocess_kwargs:
            rank0_print(f"image_preprocess overrides: {preprocess_kwargs}")
        weights_only = _should_use_weights_only(pretrained)
        local_file = isinstance(pretrained, str) and os.path.isfile(pretrained)
        vision_tower = None
        image_processor = None
        if local_file:
            # Prefer full load when possible; fall back to visual-only for MetaCLIP-2
            # worldwide vocab / architecture mismatches. Do not visual-only-fallback
            # .npz dumps (SigLIP2): a failed convert must error, not silently random-init.
            is_npz = str(pretrained).lower().endswith(".npz")
            state_dict_pre = _local_checkpoint_state_dict(pretrained)
            if not is_npz and _should_visual_only_metaclip_load(state_dict_pre):
                rank0_print(
                    "MetaCLIP-2 worldwide checkpoint detected; loading visual tower only"
                )
                vision_tower, image_processor = _create_open_clip_with_local_visual(
                    self.model_name, pretrained, force_quick_gelu, preprocess_kwargs
                )
            else:
                try:
                    create_kwargs = dict(
                        model_name=self.model_name,
                        pretrained=pretrained,
                        force_quick_gelu=force_quick_gelu,
                        precision="fp32",
                        device="cpu",
                        weights_only=weights_only,
                        require_pretrained=True,
                    )
                    force_image_size = _infer_force_image_size(
                        self.model_name, state_dict_pre
                    )
                    if force_image_size is not None:
                        create_kwargs["force_image_size"] = force_image_size
                        rank0_print(
                            f"force_image_size: {force_image_size} (from checkpoint / env)"
                        )
                    if preprocess_kwargs:
                        create_kwargs.update(preprocess_kwargs)
                    vision_tower, _, image_processor = open_clip.create_model_and_transforms(
                        **create_kwargs
                    )
                except Exception as exc:
                    if is_npz:
                        raise RuntimeError(
                            f"Failed to load local npz {pretrained} into {self.model_name}. "
                            f"NaFlex / mismatched big_vision checkpoints are unsupported. "
                            f"Underlying error: {exc}"
                        ) from exc
                    rank0_print(
                        f"Full OpenCLIP load failed ({type(exc).__name__}: {exc}); "
                        "retrying text_cfg vocab align, then visual-only"
                    )
                    state_dict = state_dict_pre or _local_checkpoint_state_dict(pretrained)
                    try:
                        vision_tower, image_processor = _create_open_clip_with_text_cfg(
                            self.model_name,
                            pretrained,
                            force_quick_gelu,
                            state_dict,
                            preprocess_kwargs,
                        )
                    except Exception as text_exc:
                        rank0_print(
                            f"text_cfg align failed ({type(text_exc).__name__}: {text_exc}); "
                            "retrying visual-only local load"
                        )
                        vision_tower, image_processor = _create_open_clip_with_local_visual(
                            self.model_name, pretrained, force_quick_gelu, preprocess_kwargs
                        )
        else:
            create_kwargs = dict(
                model_name=self.model_name,
                pretrained=pretrained,
                force_quick_gelu=force_quick_gelu,
                precision="fp32",
                device="cpu",
                weights_only=weights_only,
            )
            if preprocess_kwargs:
                create_kwargs.update(preprocess_kwargs)
            vision_tower, _, image_processor = open_clip.create_model_and_transforms(
                **create_kwargs
            )

        resize_transform = [t for t in image_processor.transforms if isinstance(t, torchvision.transforms.Resize)][0]
        normalize_transform = [t for t in image_processor.transforms if isinstance(t, torchvision.transforms.Normalize)][0]
        has_center_crop = any(
            isinstance(t, torchvision.transforms.CenterCrop) for t in image_processor.transforms
        )
        self.resize_transform_size = resize_transform.size  # 224 or (224, 224)
        visual = vision_tower.visual
        if hasattr(visual, "conv1"):
            self.patch_size = visual.conv1.kernel_size[0]  # 14 or 16
        elif hasattr(visual, "trunk") and hasattr(visual.trunk, "patch_embed"):
            self.patch_size = visual.trunk.patch_embed.proj.kernel_size[0]
        else:
            raise AttributeError(
                f"Cannot infer patch_size for OpenCLIP visual type {type(visual).__name__}"
            )

        resize_mode = (preprocess_kwargs or {}).get("image_resize_mode")
        use_squash = (not has_center_crop) or _is_siglip_model(self.model_name)
        self.image_processor = _build_hf_image_processor(
            processor_path,
            resize_size=resize_transform.size,
            image_mean=normalize_transform.mean,
            image_std=normalize_transform.std,
            resize_mode=resize_mode,
            use_squash=use_squash,
        )
        rank0_print(f"Loaded image processor from {processor_path}: {self.image_processor}")
        self.vision_tower = vision_tower.visual
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def _to_batch_first_nld(self, feats, batch_size):
        # Normalize to [B, N, D] across OpenCLIP variants.
        if feats.ndim != 3:
            return feats
        if feats.shape[0] == batch_size:
            return feats
        if feats.shape[1] == batch_size:
            return feats.permute(1, 0, 2).contiguous()
        raise ValueError(f"Unexpected OpenCLIP feature shape {tuple(feats.shape)} for batch size {batch_size}.")

    def _visual_has_cls_token(self) -> bool:
        """CLIP ViT has a CLS token; SigLIP / many timm trunks do not."""
        name_l = self.model_name.lower()
        if "siglip" in name_l:
            return False
        visual = self.vision_tower
        if hasattr(visual, "class_embedding"):
            return True
        trunk = getattr(visual, "trunk", None)
        if trunk is not None and getattr(trunk, "cls_token", None) is not None:
            return True
        return False

    def feature_select(self, image_forward_outs, batch_size):
        image_features = image_forward_outs[self.select_layer]
        image_features = self._to_batch_first_nld(image_features, batch_size)
        if self.select_feature == "patch":
            # Only drop the leading CLS/prefix token when the tower actually has one.
            if self._visual_has_cls_token():
                image_features = image_features[:, 1:]
        elif self.select_feature == "cls_patch":
            image_features = image_features
        elif self.select_feature == "conv_flatten":
            image_features = image_features.flatten(2).transpose(1, 2)
        else:
            raise ValueError(f"Unexpected select feature: {self.select_feature}")
        return image_features

    def _format_intermediates(self, image_forward_outs):
        image_features = image_forward_outs["image_intermediates"]
        prefix_features = image_forward_outs.get("image_intermediates_prefix")
        if prefix_features is None:
            return image_features
        return [torch.cat([prefix, image_feature], dim=1) for prefix, image_feature in zip(prefix_features, image_features)]

    def forward_visual(self, x, output_hidden_states=False):
        if hasattr(self.vision_tower, "forward_intermediates"):
            output_fmt = "NCHW" if self.select_feature == "conv_flatten" else "NLC"
            image_forward_outs = self.vision_tower.forward_intermediates(
                x,
                indices=None,
                output_fmt=output_fmt,
                output_extra_tokens=output_fmt == "NLC",
                intermediates_only=True,
            )
            return self._format_intermediates(image_forward_outs)
        else:

            def forward_openclip(self, x: torch.Tensor):
                features = []
                if hasattr(self, "_embeds"):
                    x = self._embeds(x)
                else:
                    x = self.conv1(x)  # shape = [*, width, grid, grid]
                    x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
                    x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

                    # class embeddings and positional embeddings
                    x = torch.cat(
                        [_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x],
                        dim=1,
                    )
                    # shape = [*, grid ** 2 + 1, width]
                    x = x + self.positional_embedding.to(x.dtype)

                    x = self.patch_dropout(x)
                    x = self.ln_pre(x)

                batch_first = getattr(self.transformer, "batch_first", False)
                if not batch_first:
                    x = x.permute(1, 0, 2)  # NLD -> LND
                for r in self.transformer.resblocks:
                    x = r(x, attn_mask=None)
                    features.append(x if batch_first else x.permute(1, 0, 2).contiguous())
                return features

            return forward_openclip(self.vision_tower, x)

    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.forward_visual(image.to(self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out, batch_size=1).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.forward_visual(images.to(self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs, batch_size=images.shape[0]).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        if hasattr(self.vision_tower, "conv1"):
            return self.vision_tower.conv1.weight.dtype
        if hasattr(self.vision_tower, "trunk"):
            return self.vision_tower.trunk.patch_embed.proj.weight.dtype
        raise NotImplementedError

    @property
    def device(self):
        if hasattr(self.vision_tower, "conv1"):
            return self.vision_tower.conv1.weight.device
        if hasattr(self.vision_tower, "trunk"):
            return self.vision_tower.trunk.patch_embed.proj.weight.device
        raise NotImplementedError

    @property
    def config(self):
        return None

    @property
    def hidden_size(self):
        if self.model_name in HIDDEN_SIZE_DICT:
            return HIDDEN_SIZE_DICT[self.model_name]
        # Prefer visual token width (transformer hidden size), which is the
        # feature dimension consumed by LLaVA's mm_projector.
        if hasattr(self.vision_tower, "width"):
            return self.vision_tower.width
        if hasattr(self.vision_tower, "conv1"):
            return self.vision_tower.conv1.weight.shape[0]
        if hasattr(self.vision_tower, "trunk") and hasattr(self.vision_tower.trunk, "num_features"):
            return self.vision_tower.trunk.num_features
        # Fallback: output embedding dim (may differ from token width).
        if hasattr(self.vision_tower, "output_dim"):
            return self.vision_tower.output_dim
        raise NotImplementedError(f"Unknown hidden size for OpenCLIP model: {self.model_name}")

    @property
    def num_patches(self):
        image_size = self.resize_transform_size if isinstance(self.resize_transform_size, int) else self.resize_transform_size[0]
        _num_patches = (image_size // self.patch_size) ** 2
        if "cls_patch" in self.select_feature:
            _num_patches += 1
        return _num_patches

    @property
    def image_size(self):
        return self.resize_transform_size

    @property
    def num_patches_per_side(self):
        image_size = self.resize_transform_size if isinstance(self.resize_transform_size, int) else self.resize_transform_size[0]
        return image_size // self.patch_size
