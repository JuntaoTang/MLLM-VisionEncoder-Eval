"""VLMEvalKit model adapter for discrete MLLMs."""

from __future__ import annotations

from vision_encoder_eval.core.runtime import asset_path

import os
import threading
import time

import pandas as pd
import torch
from PIL import Image

from vision_encoder_eval.mllm.discrete.data.qwen_chat import build_qwen_chat_prompt
from vision_encoder_eval.mllm.discrete.eval_utils import build_eval_tokenizer_cfg
from vision_encoder_eval.mllm.discrete.model.discrete_adapter import DiscreteVisualAdapter
from vision_encoder_eval.mllm.discrete.model.tokenizers.factory import build_visual_tokenizer
from vision_encoder_eval.mllm.discrete.train.common import load_arch_config
from vision_encoder_eval.mllm.evaluation.dataset_config import eval_dataset_type
from vision_encoder_eval.mllm.evaluation.mcq_utils import mcq_choice_map
from vision_encoder_eval.mllm.evaluation.vlm_utils import (
    COCO_STYLE_CAPTION_PROMPT,
    IMAGE_TOKEN_INDEX,
    KeywordsStoppingCriteria,
    is_caption_dataset,
    open_vqa_answer_suffix,
    postprocess_model_output,
    split_thinking,
    strengthen_short_answer_prompt,
    tokenizer_image_token,
)
from vlmeval.smp import cn_string
from vlmeval.vlm.base import BaseModel

_PROGRESS_LOCK = threading.Lock()
_PROGRESS_COUNTS: dict[str, int] = {}


class VTB_Discrete_VLM(BaseModel):
  INSTALL_REQ = True
  INTERLEAVE = True
  DEFAULT_IMAGE_TOKEN = "<image>"

  def __init__(
      self,
      model_path: str,
      llm_path: str = asset_path('download', 'llm/Qwen3-1.7B'),
      hidden_size: int = 2048,
      **kwargs,
  ):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    arch_cfg: dict = load_arch_config(model_path) if os.path.isdir(model_path) else {}
    vis_mode = kwargs.get("vis_mode") or arch_cfg.get("vis_mode")
    if not vis_mode:
      raise ValueError("vis_mode is required for VTB_Discrete_VLM")

    print(f"[VTB_Discrete_VLM] Loading LLM from {llm_path} ...", flush=True)
    self.tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
    if self.tokenizer.pad_token is None:
      self.tokenizer.pad_token = self.tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(
        llm_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    tokenizer_cfg = build_eval_tokenizer_cfg(vis_mode, kwargs=kwargs, arch_cfg=arch_cfg)
    print(f"[VTB_Discrete_VLM] Loading {vis_mode} tokenizer ...", flush=True)
    visual_tokenizer = build_visual_tokenizer({
        "tokenizer": tokenizer_cfg,
        "arch": {"vis_mode": vis_mode},
    })

    projector_cfg = kwargs.get("projector") or kwargs.get("connector") or {}
    if not projector_cfg:
      arch = (
        arch_cfg.get("projector_architecture")
        or arch_cfg.get("connector_architecture")
      )
      if arch:
        projector_cfg = {"architecture": arch}
        hidden_dims = (
          arch_cfg.get("projector_hidden_dims")
          if arch_cfg.get("projector_hidden_dims") is not None
          else arch_cfg.get("connector_hidden_dims")
        )
        if hidden_dims is not None:
          projector_cfg["hidden_dims"] = hidden_dims

    model = DiscreteVisualAdapter(
        tokenizer=visual_tokenizer,
        llm=llm,
        hidden_size=hidden_size,
        vis_mode=vis_mode,
        projector_cfg=projector_cfg,
    )
    model.set_phase(2)

    ckpt = os.path.join(model_path, "pytorch_model.bin")
    if os.path.isdir(model_path) and os.path.isfile(ckpt):
      print(f"[VTB_Discrete_VLM] Loading weights from {ckpt} ...", flush=True)
      model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True), strict=False)

    self.model = model.to(device=self.device, dtype=torch.bfloat16).eval()
    if vis_mode == "toklip_post_quant" and hasattr(self.model.tokenizer, "_visual"):
      self.model.tokenizer._visual.half()

    self.default_max_new_tokens = int(kwargs.get("max_new_tokens", 2048))
    self.log_every = int(kwargs.get("log_every", os.environ.get("VTB_EVAL_LOG_EVERY", 50)))
    self.verbose_responses = bool(
        kwargs.get("verbose_responses", os.environ.get("VTB_EVAL_VERBOSE_RESPONSES", "0").lower() in ("1", "true", "yes"))
    )
    print(
        f"[VTB_Discrete_VLM] Ready on {self.device} ({time.time() - t0:.1f}s) vis_mode={vis_mode}",
        flush=True,
    )

  def split_thinking(self, text: str) -> tuple[str, str]:
    return split_thinking(text)

  def use_custom_prompt(self, dataset):
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

    return DATASET_TYPE(dataset) in ("MCQ", "VQA", "Caption")

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

  def _build_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
    pad_token_id = self.tokenizer.pad_token_id
    if pad_token_id is None:
      return torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)
    attention_mask = input_ids.ne(pad_token_id).long()
    if attention_mask.sum() == 0:
      return torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)
    return attention_mask

  def _use_keyword_stopping(self, dataset) -> bool:
    return not is_caption_dataset(dataset)

  def _max_new_tokens(self, dataset) -> int:
    return self.default_max_new_tokens

  def _log_infer_progress(self, dataset) -> None:
    key = str(dataset or "unknown")
    with _PROGRESS_LOCK:
      _PROGRESS_COUNTS[key] = _PROGRESS_COUNTS.get(key, 0) + 1
      count = _PROGRESS_COUNTS[key]
    if count == 1 or count % self.log_every == 0:
      print(f"[infer/{key}] {count} samples completed", flush=True)

  def generate_inner(self, message, dataset=None):
    content, images = "", []
    for msg in message:
      if msg["type"] == "text":
        content += strengthen_short_answer_prompt(msg["value"], dataset)
      else:
        images.append(Image.open(msg["value"]).convert("RGB"))
        content += self.DEFAULT_IMAGE_TOKEN + "\n"

    import torchvision.transforms as T

    image_size = self.model.tokenizer.image_size
    transform = T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])
    pixel_values = torch.stack([transform(img) for img in images]).to(
        device=self.device, dtype=torch.bfloat16
    )

    prompt_question, stop_str = build_qwen_chat_prompt(self.tokenizer, content)
    input_ids = tokenizer_image_token(
        prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    )
    input_ids = input_ids.unsqueeze(0).to(self.device)
    attention_mask = self._build_attention_mask(input_ids)

    stopping_criteria = None
    if self._use_keyword_stopping(dataset):
      stopping_criteria = [KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)]

    gen_kwargs = dict(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=False,
        temperature=0,
        max_new_tokens=self._max_new_tokens(dataset),
        top_p=None,
        num_beams=1,
        use_cache=True,
    )
    if stopping_criteria is not None:
      gen_kwargs["stopping_criteria"] = stopping_criteria

    with torch.inference_mode():
      output_ids = self.model.generate(**gen_kwargs)

    text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    text = postprocess_model_output(text, self.tokenizer)

    self._log_infer_progress(dataset)
    if self.verbose_responses:
      key = str(dataset or "unknown")
      print(
          f"[response/{key}] tokens={int(output_ids.shape[1])} chars={len(text)}\n{text}\n---",
          flush=True,
      )
    return text
