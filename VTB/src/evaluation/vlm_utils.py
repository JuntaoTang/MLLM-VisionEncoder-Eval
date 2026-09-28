"""Shared VLMEvalKit inference helpers aligned with VTB vtb_vlm.py."""

from __future__ import annotations

import re

import torch
from transformers import StoppingCriteria
from vlmeval.smp import listinstr

from src.data.qwen_chat import IMAGE_TOKEN_INDEX, build_qwen_chat_prompt, qwen_im_end_token

# Backward-compatible alias; requires tokenizer as the first argument.
build_qwen3_conv_prompt = build_qwen_chat_prompt


def tokenizer_image_token(
    prompt: str,
    tokenizer,
    image_token_index: int = IMAGE_TOKEN_INDEX,
    return_tensors=None,
):
    """Match LLaVA-NeXT llava/mm_utils.tokenizer_image_token."""
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


class KeywordsStoppingCriteria(StoppingCriteria):
    """Stop when generated text contains a keyword (VTB / LLaVA-NeXT)."""

    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if output_ids.shape[0] != 1:
            return False
        offset = min(output_ids.shape[1] - self.start_len, 3)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            if output_ids[0, -keyword_id.shape[0] :] == keyword_id:
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False


def is_short_vqa_dataset(dataset) -> bool:
    return isinstance(dataset, str) and listinstr(
        ["VQAv2", "VizWiz", "TextVQA", "GQA"], dataset
    )


def is_vizwiz_dataset(dataset) -> bool:
    return isinstance(dataset, str) and "VizWiz" in dataset


def is_caption_dataset(dataset) -> bool:
    return isinstance(dataset, str) and listinstr(["KARPATHY", "COCO_VAL", "Caption"], dataset)


# Karpathy/COCO-style constrained prompt (inference-time only).
COCO_STYLE_CAPTION_PROMPT = (
    "Generate a short image caption.\n"
    "Describe only the main objects and actions.\n"
    "Use one sentence.\n"
    'Do not start with "The image shows", "The image features", '
    '"This image", or "A photo of".\n'
    "Use the style of COCO captions.\n"
    "Output only the caption.\n"
    "Example:\n"
    "A dog running on the grass."
)

# LLaVA-1.5 VizWiz response-format prompt.
VIZWIZ_ANSWER_SUFFIX = (
    "\nWhen the provided information is insufficient, respond with 'Unanswerable'."
    "\nAnswer the question using a single word or phrase."
)
SHORT_VQA_ANSWER_SUFFIX = "\nAnswer the question using a single word or phrase."
SHORT_VQA_ANSWER_SUFFIX_CN = "\n请用一个词或短语回答。"


def open_vqa_answer_suffix(dataset, prompt: str) -> str:
    """Append LLaVA-style short-answer (and VizWiz unanswerable) instruction."""
    from vlmeval.smp import cn_string

    if is_vizwiz_dataset(dataset):
        return VIZWIZ_ANSWER_SUFFIX
    if cn_string(prompt):
        return SHORT_VQA_ANSWER_SUFFIX_CN
    return SHORT_VQA_ANSWER_SUFFIX


def strengthen_short_answer_prompt(text: str, dataset) -> str:
    if not is_short_vqa_dataset(dataset):
        return text
    return text


def split_thinking(text: str) -> tuple[str, str]:
    """Strip Qwen3 thinking blocks (aligned with VLMEvalKit inference.split_thinking)."""
    if "</think>" in text:
        splits = text.split("</think>")
        prediction = splits[-1].strip()
        if len(splits) == 2 and "<think>" in splits[0]:
            thinking = splits[0].split("<think>")[1].strip()
        else:
            thinking = "</think>".join(splits[:-1])
            thinking += "</think>"
    else:
        thinking = ""
        prediction = text
    return prediction, thinking


def postprocess_model_output(text: str, tokenizer=None) -> str:
    prediction, _ = split_thinking(text)
    if "<answer>" in prediction and "</answer>" in prediction:
        match = re.search(r"<answer>\s*(.*?)\s*</answer>", prediction, re.DOTALL)
        if match:
            prediction = match.group(1).strip()
    for sep in ("assistant\n", "ASSISTANT:", "<|im_start|>assistant"):
        if sep in prediction:
            prediction = prediction.split(sep)[-1].strip()
    stops = []
    if tokenizer is not None:
        stops.append(qwen_im_end_token(tokenizer))
    for stop in stops:
        if stop and stop in prediction:
            prediction = prediction.split(stop)[0].strip()
    return prediction.strip()
