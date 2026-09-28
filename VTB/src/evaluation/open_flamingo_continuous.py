"""Continuous (LLaVA-NeXT) adapter for OpenFlamingo-style continuation eval."""

from __future__ import annotations

import os
import sys
from typing import Sequence

import torch
from PIL import Image
from transformers import StoppingCriteria

VTB_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LLAVA_ROOT = os.path.join(VTB_ROOT, "third_party", "LLaVA-NeXT")
THIRD_PARTY = os.path.join(VTB_ROOT, "third_party")
for path in (VTB_ROOT, LLAVA_ROOT, THIRD_PARTY):
    if path not in sys.path:
        sys.path.insert(0, path)

from open_flamingo_eval.base import BaseOFEvalModel  # noqa: E402

from src.evaluation.checkpoint import resolve_checkpoint_dir  # noqa: E402
from src.evaluation.vtb_vlm import _build_overwrite_config, _resolve_model_name  # noqa: E402

IMAGE_TOKEN = "<image>"
IMAGE_TOKEN_INDEX = -200
IGNORE_INDEX = -100


class _KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        for keyword in keywords:
            ids = tokenizer(keyword).input_ids
            if len(ids) > 1 and ids[0] == tokenizer.bos_token_id:
                ids = ids[1:]
            if ids:
                self.keyword_ids.append(torch.tensor(ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if output_ids.shape[0] != 1:
            return False
        offset = min(output_ids.shape[1] - self.start_len, 3)
        self.keyword_ids = [k.to(output_ids.device) for k in self.keyword_ids]
        for kid in self.keyword_ids:
            if kid.numel() == 0:
                continue
            if torch.equal(output_ids[0, -kid.shape[0] :], kid):
                return True
        text = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        return any(k in text for k in self.keywords)


def load_continuous_of_model(
    *,
    model_path: str,
    llm_path: str,
    vision_weights: str | None = None,
    vision_tower: str | None = None,
    processor_path: str | None = None,
    attn_implementation: str = "sdpa",
    prompt_style: str = "chatml",
) -> "ContinuousOFEvalModel":
    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init

    if vision_weights:
        os.environ["VTB_VISION_WEIGHTS"] = vision_weights
    if processor_path and os.path.isdir(processor_path):
        os.environ["VTB_CLIP_IMAGE_PROCESSOR"] = processor_path

    resolved = resolve_checkpoint_dir(model_path)
    model_path = resolved.path
    adapter_base = resolved.model_base
    if adapter_base is None and os.path.isfile(os.path.join(model_path, "mm_projector.bin")):
        adapter_base = llm_path

    device_index = int(os.environ.get("VTB_EVAL_DEVICE_INDEX", "0"))
    device = torch.device(f"cuda:{device_index}" if torch.cuda.is_available() else "cpu")
    device_map = {"": device_index} if torch.cuda.is_available() else "cpu"

    disable_torch_init()
    name_source = adapter_base or model_path
    model_name = _resolve_model_name(name_source)
    overwrite_config = _build_overwrite_config(
        model_path, vision_weights, vision_tower, processor_path=processor_path
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
        raise RuntimeError(f"Failed to load vision tower for {model_path}")

    model = model.to(device=device, dtype=torch.bfloat16).eval()
    return ContinuousOFEvalModel(
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        process_images=process_images,
        tokenizer_image_token=tokenizer_image_token,
        device=device,
        prompt_style=prompt_style,
    )


class ContinuousOFEvalModel(BaseOFEvalModel):
    """OpenFlamingo continuation interface over LLaVA-NeXT continuous MLLM."""

    def __init__(
        self,
        model,
        tokenizer,
        image_processor,
        process_images,
        tokenizer_image_token,
        device: torch.device,
        prompt_style: str = "chatml",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.process_images = process_images
        self.tokenizer_image_token = tokenizer_image_token
        self.device = device
        self.prompt_style = prompt_style

    def _wrap_prompt(self, prompt: str) -> str:
        if self.prompt_style != "chatml":
            return prompt
        return (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def _prepare_images(self, images: Sequence[Image.Image]):
        tensors = self.process_images(list(images), self.image_processor, self.model.config)
        if isinstance(tensors, list):
            return [t.to(dtype=torch.bfloat16, device=self.device) for t in tensors]
        return tensors.to(dtype=torch.bfloat16, device=self.device)

    def _encode(self, prompt: str) -> torch.Tensor:
        ids = self.tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        return ids.unsqueeze(0).to(self.device)

    @torch.inference_mode()
    def generate(
        self,
        images: Sequence[Image.Image],
        prompt: str,
        *,
        max_new_tokens: int,
        num_beams: int = 1,
        stop_strings: Sequence[str] | None = None,
    ) -> str:
        image_tensor = self._prepare_images(images)
        wrapped = self._wrap_prompt(prompt)
        input_ids = self._encode(wrapped)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=self.device)

        stops = list(stop_strings or [])
        if self.prompt_style == "chatml":
            stops = list(dict.fromkeys(stops + ["<|im_end|>", "<|endofchunk|>", "\n"]))
        stopping = [_KeywordsStoppingCriteria(stops, self.tokenizer, input_ids)] if stops else None

        beams = max(1, int(num_beams))
        gen_kwargs = dict(
            do_sample=False,
            temperature=0.0,
            top_p=None,
            top_k=None,
            max_new_tokens=max_new_tokens,
            num_beams=beams,
            use_cache=True,
        )
        if stopping is not None:
            gen_kwargs["stopping_criteria"] = stopping

        output_ids = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            images=image_tensor,
            image_sizes=[img.size for img in images],
            **gen_kwargs,
        )
        in_len = int(input_ids.shape[1])
        if output_ids.shape[1] > in_len and torch.equal(output_ids[:, :in_len], input_ids):
            output_ids = output_ids[:, in_len:]
        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    @torch.inference_mode()
    def score_completions(
        self,
        images: Sequence[Image.Image],
        prompt: str,
        completions: Sequence[str],
        *,
        length_normalize: bool = False,
    ) -> list[float]:
        image_tensor = self._prepare_images(images)
        scores: list[float] = []
        for completion in completions:
            sep = "" if prompt.endswith((":", " ")) else " "
            wrapped_prompt = self._wrap_prompt(prompt)
            wrapped_full = wrapped_prompt + f"{sep}{completion}"
            input_ids = self._encode(wrapped_full)
            prompt_ids = self._encode(wrapped_prompt)
            prompt_len = int(prompt_ids.shape[1])

            labels = input_ids.clone()
            labels[:, :prompt_len] = IGNORE_INDEX
            labels[labels == IMAGE_TOKEN_INDEX] = IGNORE_INDEX
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=self.device)

            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=image_tensor,
                labels=labels,
                use_cache=False,
            )
            loss = getattr(out, "loss", None)
            if loss is None or not torch.isfinite(loss):
                scores.append(float("-inf"))
                continue
            n_tok = int((labels != IGNORE_INDEX).sum().item())
            n_tok = max(n_tok, 1)
            total_lp = float((-loss * n_tok).item())
            scores.append(total_lp / n_tok if length_normalize else total_lp)
        return scores
