"""VLMEvalKit model adapter for VTB-trained LLaVA-NeXT checkpoints."""

from __future__ import annotations

from vision_encoder_eval.core.runtime import asset_path

import copy
import os
import warnings

import pandas as pd
import torch
from PIL import Image

from vision_encoder_eval.mllm.evaluation.checkpoint import resolve_checkpoint_dir
from vision_encoder_eval.mllm.evaluation.dataset_config import eval_dataset_type
from vision_encoder_eval.mllm.evaluation.mcq_utils import mcq_choice_map
from vision_encoder_eval.mllm.evaluation.vlm_utils import (
    COCO_STYLE_CAPTION_PROMPT,
    is_caption_dataset,
    open_vqa_answer_suffix,
    postprocess_model_output,
)
from vlmeval.smp import cn_string
from vlmeval.vlm.base import BaseModel


def _resolve_eval_checkpoint(checkpoint_dir: str, model_base: str | None = None) -> tuple[str, str | None]:
    resolved = resolve_checkpoint_dir(checkpoint_dir)
    if model_base is None:
        return resolved.path, resolved.model_base if resolved.kind == "adapter" else None
    return resolved.path, model_base


def _build_overwrite_config(
    model_path: str,
    vision_weights: str | None = None,
    vision_tower: str | None = None,
    processor_path: str | None = None,
) -> dict:
    """Remap mm_vision_tower when checkpoint config points at a stale / hub path.

    Offline hosts cannot resolve huggingface ids like openai/clip-vit-large-patch14.
    Keep open_clip_hub:* towers as-is (weights via VTB_VISION_WEIGHTS); remap HF CLIP
    hubs to a local processor directory when available.
    """
    import json

    overwrite: dict = {"delay_load": True}
    config_path = os.path.join(model_path, "config.json")
    saved = None
    if os.path.isfile(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        saved = cfg.get("mm_vision_tower") or cfg.get("vision_tower")

    # Local directory checkpoint / processor is always fine offline.
    if saved and os.path.isdir(str(saved)):
        return overwrite

    # OpenCLIP / custom hub schemes must not be rewritten to openai/clip-*.
    non_hf_prefixes = ("open_clip_hub:", "hf:", "vtb_ssl:", "siglip", "eva", "imagebind")
    if saved and str(saved).startswith(non_hf_prefixes):
        overwrite["mm_vision_tower"] = saved
        return overwrite
    if vision_tower and str(vision_tower).startswith(non_hf_prefixes):
        overwrite["mm_vision_tower"] = vision_tower
        return overwrite

    local_clip = (
        processor_path
        or os.environ.get("VTB_CLIP_IMAGE_PROCESSOR")
        or asset_path('download', 'tokenizer/continuous/clip-vit-large-patch14')
    )
    if vision_weights and os.path.isdir(vision_weights):
        overwrite["mm_vision_tower"] = vision_weights
    elif vision_tower and os.path.isdir(str(vision_tower)):
        overwrite["mm_vision_tower"] = vision_tower
    elif saved and os.path.isdir(str(saved)):
        overwrite["mm_vision_tower"] = saved
    elif local_clip and os.path.isdir(local_clip):
        # Hub CLIP id (openai/...) or missing tower → local processor tree.
        if (not saved) or "clip" in str(saved).lower() or str(saved).startswith("openai/"):
            overwrite["mm_vision_tower"] = local_clip

    return overwrite


def _resolve_model_name(model_path: str) -> str:
    import json

    config_path = os.path.join(model_path, "config.json")
    if os.path.isfile(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        if cfg.get("model_type") == "qwen3" or "Qwen3" in str(cfg.get("architectures", [])):
            return "qwen3"
        if cfg.get("model_type") == "qwen2" or "Qwen2" in str(cfg.get("architectures", [])):
            return "qwen"
        if cfg.get("model_type") in ("llama", "llava_llama") or "LlavaLlama" in str(cfg.get("architectures", [])):
            return "llava_llama_3"
    lower = model_path.lower()
    if "qwen3" in lower or "qwen_3" in lower:
        return "qwen3"
    if "qwen2" in lower or "qwen2.5" in lower or "qwen2p5" in lower or "qwen25" in lower:
        return "qwen"
    if "llama" in lower or "smol" in lower:
        return "llava_llama_3"
    return "qwen3"


class VTB_LLaVA(BaseModel):
    """LLaVA-NeXT checkpoint trained via VTB, evaluated through VLMEvalKit."""

    INSTALL_REQ = True
    INTERLEAVE = True

    DEFAULT_IMAGE_TOKEN = "<image>"
    IMAGE_TOKEN_INDEX = -200

    def __init__(
        self,
        model_path: str,
        conv_mode: str | None = None,
        vision_weights: str | None = None,
        vision_tower: str | None = None,
        processor_path: str | None = None,
        attn_implementation: str = "sdpa",
        max_new_tokens: int = 2048,
        model_base: str | None = None,
        **kwargs,
    ):
        assert model_path is not None

        from llava.conversation import SeparatorStyle, conv_templates
        from llava.mm_utils import KeywordsStoppingCriteria, process_images, tokenizer_image_token
        from llava.model.builder import load_pretrained_model
        from llava.utils import disable_torch_init

        if vision_weights:
            os.environ["VTB_VISION_WEIGHTS"] = vision_weights
        if processor_path and os.path.isdir(processor_path):
            os.environ["VTB_CLIP_IMAGE_PROCESSOR"] = processor_path

        model_path, adapter_base = _resolve_eval_checkpoint(model_path, model_base=model_base)
        if adapter_base is None and os.path.isfile(os.path.join(model_path, "mm_projector.bin")):
            adapter_base = kwargs.pop("llm_path", None)
            if adapter_base is None:
                raise ValueError(
                    f"Adapter checkpoint at {model_path} requires llm_path/model_base (base LLM path)."
                )
        kwargs.pop("llm_path", None)

        device_index = int(os.environ.get("VTB_EVAL_DEVICE_INDEX", "0"))
        device = torch.device(f"cuda:{device_index}")
        device_map = {"": device_index}

        disable_torch_init()
        name_source = adapter_base or model_path
        model_name = _resolve_model_name(name_source)
        overwrite_config = _build_overwrite_config(
            model_path,
            vision_weights,
            vision_tower,
            processor_path=processor_path,
        )

        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path,
            adapter_base,
            model_name,
            multimodal=True,
            torch_dtype="bfloat16",
            attn_implementation=attn_implementation,
            device_map=device_map,
            overwrite_config=overwrite_config,
        )
        if image_processor is None:
            raise RuntimeError(f"Failed to load vision tower for checkpoint {model_path}")

        model = model.to(device=device, dtype=torch.bfloat16).eval()
        # delay_load skips encoder weights in the LLaVA ckpt; parent `.to(bf16)`
        # also zeros EUPE-T LayerNorm affines. Overlay the saved tower in fp32.
        try:
            ssl_tower = model.get_model().get_vision_tower()
            if getattr(ssl_tower, "ssl_type", None) == "eupe" and hasattr(ssl_tower, "load_eupe_weights_from_llava_ckpt"):
                ssl_tower.load_eupe_weights_from_llava_ckpt(model_path, device=device)
        except Exception as exc:
            warnings.warn(f"EUPE ckpt overlay failed: {exc}")
        self.device = device

        if conv_mode is None:
            lower_paths = f"{model_path} {adapter_base or ''}".lower()
            if "smol" in lower_paths:
                conv_mode = "smollm2"
            elif "llama" in model_name:
                conv_mode = "llava_llama_3"
            elif model_name == "qwen3":
                conv_mode = "qwen_3"
            else:
                conv_mode = "qwen_2_5"

        self.conv_template = conv_mode
        self.conv_templates = conv_templates
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.process_images = process_images
        self.tokenizer_image_token = tokenizer_image_token
        self.KeywordStoppingCriteria = KeywordsStoppingCriteria
        self.SeparatorStyle = SeparatorStyle

        gen_kwargs = dict(
            do_sample=False,
            temperature=0,
            max_new_tokens=max_new_tokens,
            top_p=None,
            num_beams=1,
            use_cache=True,
        )
        reserved = {
            "model_base",
            "vision_weights",
            "vision_tower",
            "processor_path",
            "conv_mode",
            "attn_implementation",
            "max_new_tokens",
            "class",
        }
        gen_kwargs.update({k: v for k, v in kwargs.items() if k not in reserved})
        self.kwargs = gen_kwargs
        warnings.warn(f"VTB_LLaVA generation kwargs: {self.kwargs}")

    def use_custom_prompt(self, dataset):
        """Custom prompts for MCQ / VQA / caption; keep yes_no dataset-native."""
        assert dataset is not None
        etype = eval_dataset_type(dataset)
        if etype in ("mcq", "vqa", "caption"):
            return True
        if etype == "yes_no":
            return False
        if is_caption_dataset(dataset):
            return True
        if "VQAv2" in dataset or dataset == "VizWiz":
            return True
        from vlmeval.dataset import DATASET_TYPE

        dtype = DATASET_TYPE(dataset)
        return dtype in ("MCQ", "VQA", "Caption")

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        tgt_path = self.dump_image(line, dataset)

        # Constrained COCO-style caption prompt (lowers "The image shows..." fluff).
        if eval_dataset_type(dataset) == "caption" or is_caption_dataset(dataset):
            message = [dict(type="image", value=s) for s in tgt_path]
            message.append(dict(type="text", value=COCO_STYLE_CAPTION_PROMPT))
            return message

        question = line["question"]
        hint = line["hint"] if ("hint" in line and not pd.isna(line["hint"])) else None
        if hint is not None:
            question = hint + "\n" + question

        # Prefer A–Z columns; else expand MMMU-style ``options`` list.
        options = mcq_choice_map(line)
        for key, item in options.items():
            question += f"\n{key}. {item}"
        prompt = question

        if options:
            prompt += (
                "\n请直接回答选项字母。"
                if cn_string(prompt)
                else "\nAnswer with the option's letter from the given choices directly."
            )
        else:
            prompt += open_vqa_answer_suffix(dataset, prompt)

        message = [dict(type="image", value=s) for s in tgt_path]
        message.append(dict(type="text", value=prompt))
        return message

    def _prepare_image_tensor(self, images: list[Image.Image]) -> torch.Tensor | list[torch.Tensor]:
        tensors = self.process_images(images, self.image_processor, self.model.config)
        if isinstance(tensors, list):
            return [t.to(dtype=torch.bfloat16, device=self.device) for t in tensors]
        return tensors.to(dtype=torch.bfloat16, device=self.device)

    def generate_inner(self, message, dataset=None):
        content, images = "", []
        for msg in message:
            if msg["type"] == "text":
                content += msg["value"]
            else:
                images.append(Image.open(msg["value"]).convert("RGB"))
                content = f"{self.DEFAULT_IMAGE_TOKEN}\n{content}"

        image_tensor = self._prepare_image_tensor(images)

        conv = copy.deepcopy(self.conv_templates[self.conv_template])
        conv.tokenizer = self.tokenizer
        conv.append_message(conv.roles[0], content)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = self.tokenizer_image_token(
            prompt_question, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        input_ids = input_ids.unsqueeze(0).to(self.device)

        stop_str = conv.sep if conv.sep_style != self.SeparatorStyle.TWO else conv.sep2
        stopping_criteria = self.KeywordStoppingCriteria([stop_str], self.tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                attention_mask=torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device),
                images=image_tensor,
                stopping_criteria=[stopping_criteria],
                **self.kwargs,
            )

        # LLaVA-Qwen generate() runs from inputs_embeds and typically returns *only*
        # new tokens. Blindly slicing by input_ids length chops long answers (esp.
        # captions), leaving mid-sentence fragments like ", and there are…".
        in_len = int(input_ids.shape[1])
        if output_ids.shape[1] > in_len and torch.equal(output_ids[:, :in_len], input_ids):
            output_ids = output_ids[:, in_len:]
        text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return postprocess_model_output(text, self.tokenizer)
