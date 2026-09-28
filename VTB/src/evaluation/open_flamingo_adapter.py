"""Discrete MLLM adapter for OpenFlamingo-style continuation eval."""

from __future__ import annotations

import os
import sys
from typing import Sequence

import torch
import torchvision.transforms as T
from PIL import Image
from transformers import StoppingCriteria

VTB_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
THIRD_PARTY = os.path.join(VTB_ROOT, "third_party")
for path in (VTB_ROOT, THIRD_PARTY):
    if path not in sys.path:
        sys.path.insert(0, path)

from open_flamingo_eval.base import BaseOFEvalModel  # noqa: E402

from src.discrete.eval_utils import build_eval_tokenizer_cfg  # noqa: E402
from src.discrete.model.discrete_adapter import (  # noqa: E402
    IMAGE_TOKEN_INDEX,
    IGNORE_INDEX,
    DiscreteVisualAdapter,
)
from src.discrete.model.tokenizers.factory import build_visual_tokenizer  # noqa: E402
from src.discrete.train.common import load_arch_config  # noqa: E402


def _tokenizer_image_token(
    prompt: str,
    tokenizer,
    image_token_index: int = IMAGE_TOKEN_INDEX,
    return_tensors=None,
):
    """Match LLaVA-NeXT tokenizer_image_token (local copy; avoid vlmeval import)."""
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<image>")]

    def insert_separator(chunks, sep):
        return [ele for sublist in zip(chunks, [sep] * len(chunks)) for ele in sublist][:-1]

    input_ids: list[int] = []
    offset = 0
    if (
        len(prompt_chunks) > 0
        and len(prompt_chunks[0]) > 0
        and prompt_chunks[0][0] == tokenizer.bos_token_id
    ):
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for chunk in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(chunk[offset:])

    if return_tensors == "pt":
        return torch.tensor(input_ids, dtype=torch.long)
    return input_ids


class _KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            if cur_keyword_ids:
                self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if output_ids.shape[0] != 1:
            return False
        offset = min(output_ids.shape[1] - self.start_len, 3)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            if keyword_id.numel() == 0:
                continue
            if torch.equal(output_ids[0, -keyword_id.shape[0] :], keyword_id):
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False


def load_discrete_of_model(
    *,
    model_path: str,
    llm_path: str,
    hidden_size: int = 2048,
    vis_mode: str | None = None,
    tokenizer_cfg: dict | None = None,
    projector_cfg: dict | None = None,
    prompt_style: str = "chatml",
    **kwargs,
) -> "DiscreteOFEvalModel":
    """Build DiscreteOFEvalModel from a discrete checkpoint directory."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arch_cfg: dict = load_arch_config(model_path) if os.path.isdir(model_path) else {}
    resolved_vis = vis_mode or arch_cfg.get("vis_mode")
    if not resolved_vis:
        raise ValueError("vis_mode is required")

    tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(
        llm_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    tok_cfg = tokenizer_cfg or build_eval_tokenizer_cfg(
        resolved_vis, kwargs=kwargs, arch_cfg=arch_cfg
    )
    visual_tokenizer = build_visual_tokenizer(
        {"tokenizer": tok_cfg, "arch": {"vis_mode": resolved_vis}}
    )

    proj = projector_cfg or {}
    if not proj:
        arch = arch_cfg.get("projector_architecture") or arch_cfg.get("connector_architecture")
        if arch:
            proj = {"architecture": arch}
            hidden_dims = (
                arch_cfg.get("projector_hidden_dims")
                if arch_cfg.get("projector_hidden_dims") is not None
                else arch_cfg.get("connector_hidden_dims")
            )
            if hidden_dims is not None:
                proj["hidden_dims"] = hidden_dims

    model = DiscreteVisualAdapter(
        tokenizer=visual_tokenizer,
        llm=llm,
        hidden_size=hidden_size,
        vis_mode=resolved_vis,
        projector_cfg=proj,
    )
    model.set_phase(2)

    ckpt = os.path.join(model_path, "pytorch_model.bin")
    if os.path.isdir(model_path) and os.path.isfile(ckpt):
        model.load_state_dict(
            torch.load(ckpt, map_location="cpu", weights_only=True), strict=False
        )

    model = model.to(device=device, dtype=torch.bfloat16).eval()
    if resolved_vis == "toklip_post_quant" and hasattr(model.tokenizer, "_visual"):
        model.tokenizer._visual.half()

    return DiscreteOFEvalModel(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompt_style=prompt_style,
    )


class DiscreteOFEvalModel(BaseOFEvalModel):
    """Continuation / log-prob interface over DiscreteVisualAdapter.

    ``prompt_style``:
      - ``flamingo``: raw OpenFlamingo ``Output:`` / ``Short answer:`` continuation
      - ``chatml``: wrap the same text in Qwen ChatML (matches VTB discrete pretrain)
    """

    def __init__(
        self,
        model: DiscreteVisualAdapter,
        tokenizer,
        device: torch.device,
        prompt_style: str = "chatml",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.prompt_style = prompt_style
        image_size = int(getattr(model.tokenizer, "image_size", 256))
        self.transform = T.Compose(
            [T.Resize((image_size, image_size)), T.ToTensor()]
        )

    def _wrap_prompt(self, prompt: str) -> str:
        if self.prompt_style != "chatml":
            return prompt
        # Keep OpenFlamingo task text, but place it in the ChatML generation slot
        # used during VTB discrete pretrain / finetune.
        return (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def _pixels(self, images: Sequence[Image.Image]) -> torch.Tensor:
        if not images:
            raise ValueError("At least one image is required")
        return torch.stack([self.transform(img) for img in images]).to(
            device=self.device, dtype=torch.bfloat16
        )

    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        input_ids = _tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        return input_ids.unsqueeze(0).to(self.device)

    def _attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            return torch.ones_like(input_ids, dtype=torch.long)
        return input_ids.ne(pad_id).long()

    @torch.inference_mode()
    def generate(
        self,
        images: Sequence[Image.Image],
        prompt: str,
        *,
        max_new_tokens: int,
        num_beams: int = 3,
        stop_strings: Sequence[str] | None = None,
    ) -> str:
        pixel_values = self._pixels(images)
        input_ids = self._encode_prompt(self._wrap_prompt(prompt))
        attention_mask = self._attention_mask(input_ids)

        stops = list(stop_strings or [])
        if self.prompt_style == "chatml":
            stops = list(dict.fromkeys(stops + ["<|im_end|>", "<|endofchunk|>", "\n"]))
        stopping = None
        if stops:
            stopping = [_KeywordsStoppingCriteria(stops, self.tokenizer, input_ids)]

        gen_kwargs = dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            num_beams=max(1, int(num_beams)),
            use_cache=True,
        )
        if stopping is not None:
            gen_kwargs["stopping_criteria"] = stopping

        output_ids = self.model.generate(**gen_kwargs)
        text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        return text.strip()

    @torch.inference_mode()
    def score_completions(
        self,
        images: Sequence[Image.Image],
        prompt: str,
        completions: Sequence[str],
        *,
        length_normalize: bool = False,
    ) -> list[float]:
        """Log-probs of each completion under teacher forcing (续写打分)."""
        pixel_values = self._pixels(images)
        scores: list[float] = []
        for completion in completions:
            sep = "" if prompt.endswith((":", " ")) else " "
            # Completion must continue the assistant turn, not be wrapped into the user message.
            wrapped_prompt = self._wrap_prompt(prompt)
            wrapped_full = wrapped_prompt + f"{sep}{completion}"
            input_ids = self._encode_prompt(wrapped_full)
            attention_mask = self._attention_mask(input_ids)

            prompt_ids = self._encode_prompt(wrapped_prompt)
            prompt_len = int(prompt_ids.shape[1])

            labels = input_ids.clone()
            labels[:, :prompt_len] = IGNORE_INDEX
            labels[labels == IMAGE_TOKEN_INDEX] = IGNORE_INDEX

            out = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
            )
            loss = out.get("loss")
            if loss is None or not torch.isfinite(loss):
                scores.append(float("-inf"))
                continue
            n_tok = int((labels != IGNORE_INDEX).sum().item())
            n_tok = max(n_tok, 1)
            total_lp = float((-loss * n_tok).item())
            scores.append(total_lp / n_tok if length_normalize else total_lp)
        return scores
