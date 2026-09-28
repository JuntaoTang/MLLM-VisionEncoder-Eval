import torch
import torch.nn as nn
import os

from transformers import AutoModel, AutoImageProcessor, AutoConfig, CLIPImageProcessor
from llava.utils import rank0_print


def _ensure_llava_processor_attrs(processor):
    """LLaVA dataset expects CLIP-like `.crop_size` dict; ViTImageProcessor lacks it."""
    size = getattr(processor, "size", None)
    if isinstance(size, dict):
        h = size.get("height") or size.get("shortest_edge") or size.get("longest_edge") or 224
        w = size.get("width") or h
    elif isinstance(size, (list, tuple)) and len(size) >= 2:
        h, w = int(size[0]), int(size[1])
    elif isinstance(size, int):
        h = w = size
    else:
        h = w = int(getattr(processor, "crop_size", 224) or 224)
        if isinstance(getattr(processor, "crop_size", None), dict):
            cs = processor.crop_size
            h = int(cs.get("height") or cs.get("shortest_edge") or 224)
            w = int(cs.get("width") or h)

    crop = getattr(processor, "crop_size", None)
    if not isinstance(crop, dict) or "height" not in crop or "width" not in crop:
        try:
            processor.crop_size = {"height": int(h), "width": int(w)}
        except Exception:
            # Some processors are frozen; attach as plain attribute.
            object.__setattr__(processor, "crop_size", {"height": int(h), "width": int(w)})

    # train.py also reads image_processor.size[0] or size["shortest_edge"]
    if not isinstance(getattr(processor, "size", None), (dict, list, tuple, int)):
        try:
            processor.size = {"shortest_edge": int(h), "height": int(h), "width": int(w)}
        except Exception:
            object.__setattr__(
                processor, "size", {"shortest_edge": int(h), "height": int(h), "width": int(w)}
            )
    return processor


class HFVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower.replace("hf:", "", 1)
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, "mm_vision_select_feature", None) or os.environ.get(
            "VTB_HF_SELECT_FEATURE", "patch"
        )
        # Always keep cfg_only so delay_load path exposes `.config` (LLaVA MetaModel
        # builds the projector from vision_tower.config before load_model).
        self.cfg_only = AutoConfig.from_pretrained(self.vision_tower_name, trust_remote_code=True)

        if not delay_load:
            self.load_model()

    def load_model(self, device_map=None):
        try:
            self.image_processor = AutoImageProcessor.from_pretrained(self.vision_tower_name)
        except Exception:
            if "448" in self.vision_tower_name:
                image_size = 448
                self.image_processor = CLIPImageProcessor(
                    size={"shortest_edge": image_size},
                    do_center_crop=True,
                    crop_size=image_size,
                )
            else:
                # Prefer local CLIP processor when offline.
                local_clip = os.environ.get("VTB_CLIP_IMAGE_PROCESSOR") or (
                    "/cache/ckpt/download/tokenizer/continuous/clip-vit-large-patch14"
                )
                if os.path.isdir(local_clip):
                    self.image_processor = CLIPImageProcessor.from_pretrained(local_clip)
                else:
                    self.image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.image_processor = _ensure_llava_processor_attrs(self.image_processor)
        rank0_print(f"Loaded image processor: {self.image_processor}")
        # Do not force .to("cuda") — DeepSpeed / accelerate place modules.
        kwargs = {"trust_remote_code": True, "torch_dtype": torch.bfloat16}
        if device_map is not None:
            kwargs["device_map"] = device_map
        self.vision_tower = AutoModel.from_pretrained(self.vision_tower_name, **kwargs)
        # Keep full-model config before optional unwrap to .vision_model.
        if not hasattr(self, "cfg_only") or self.cfg_only is None:
            self.cfg_only = self.vision_tower.config

        if hasattr(self.vision_tower, "vision_model"):
            self.vision_tower = self.vision_tower.vision_model
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        select_feature_type = self.select_feature

        if self.select_feature in ["slicefour_patch", "slicefour_cls_patch"]:
            select_every_k_layer = len(image_forward_outs.hidden_states) // 4
            image_features = torch.cat(
                [
                    image_forward_outs.hidden_states[i]
                    for i in range(
                        select_every_k_layer + self.select_layer,
                        len(image_forward_outs.hidden_states),
                        select_every_k_layer,
                    )
                ],
                dim=-1,
            )
            select_feature_type = select_feature_type.replace("slicefour_", "")
        else:
            image_features = image_forward_outs.hidden_states[self.select_layer]

        if select_feature_type == "patch":
            image_features = image_features[:, 1:]
        elif select_feature_type == "cls_patch":
            image_features = image_features
        else:
            raise ValueError(f"Unexpected select feature: {select_feature_type}")
        return image_features

    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True,
                )
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(
                images.to(device=self.device, dtype=self.dtype),
                output_hidden_states=True,
            )
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return next(self.vision_tower.parameters()).dtype

    @property
    def device(self):
        return next(self.vision_tower.parameters()).device

    @property
    def config(self):
        if self.is_loaded and hasattr(self, "vision_tower") and hasattr(self.vision_tower, "config"):
            return self.vision_tower.config
        return self.cfg_only

    @property
    def hidden_size(self):
        try:
            _hidden_size = self.config.hidden_size
        except Exception:
            _hidden_size = self.config.vision_config.hidden_size
        if "slicefour" in self.select_feature:
            _hidden_size *= 4
        return _hidden_size

    @property
    def num_patches(self):
        image_size = self.image_size
        patch_size = self.config.patch_size if hasattr(self.config, "patch_size") else self.config.vision_config.patch_size
        _num_patches = (image_size // patch_size) ** 2
        if "cls_patch" in self.select_feature:
            _num_patches += 1
        return _num_patches

    @property
    def num_patches_per_side(self):
        patch_size = self.config.patch_size if hasattr(self.config, "patch_size") else self.config.vision_config.patch_size
        return self.image_size // patch_size

    @property
    def image_size(self):
        if hasattr(self.config, "image_size"):
            return self.config.image_size
        return self.config.vision_config.image_size
