"""Qwen3 ChatML helpers for discrete training and eval."""

from __future__ import annotations

import copy
from typing import Any

import torch

from vision_encoder_eval.mllm.data.qwen_chat import IMAGE_TOKEN_INDEX, build_qwen_chat_prompt, qwen_im_end_token

IGNORE_INDEX = -100

__all__ = [
    "IGNORE_INDEX",
    "IMAGE_TOKEN_INDEX",
    "build_qwen_chat_prompt",
    "qwen_im_end_token",
    "preprocess_qwen_conversation",
]


def preprocess_qwen_conversation(
    conversations: list[dict[str, Any]],
    tokenizer,
    *,
    max_length: int = 2048,
    system_message: str = "You are a helpful assistant.",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize one LLaVA-style conversation into (input_ids, labels).

    User/system turns are masked with IGNORE_INDEX; assistant tokens are supervised.
    Literal ``<image>`` tokens are remapped to IMAGE_TOKEN_INDEX (-200).
    """
    roles = {"human": "user", "gpt": "assistant", "user": "user", "assistant": "assistant"}

    tok = copy.deepcopy(tokenizer)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    tok.add_tokens(["<image>"], special_tokens=True)

    image_token_id = tok.convert_tokens_to_ids("<image>")
    im_start = tok.convert_tokens_to_ids("<|im_start|>")
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    if im_start is None or im_end is None:
        extra = tok.additional_special_tokens_ids or []
        if len(extra) >= 2:
            im_start, im_end = extra[0], extra[1]
        else:
            raise ValueError("Cannot resolve Qwen <|im_start|>/<|im_end|> token ids")

    # Keep special structural tokens unmasked (match LLaVA preprocess_qwen).
    unmask_tokens_idx = {198, im_start, im_end}

    chat_template = (
        "{% for message in messages %}"
        "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
    )
    tok.chat_template = chat_template

    source = list(conversations)
    if source and roles.get(source[0].get("from") or source[0].get("role"), "") != "user":
        source = source[1:]

    input_id: list[int] = []
    target: list[int] = []

    system_ids = tok.apply_chat_template([{"role": "system", "content": system_message}])
    input_id += system_ids
    target += [IGNORE_INDEX] * len(system_ids)

    for turn in source:
        role_raw = turn.get("role", turn.get("from"))
        content = turn.get("content", turn.get("value", ""))
        role = roles.get(role_raw, role_raw)
        encode_id = tok.apply_chat_template([{"role": role, "content": content}])
        input_id += encode_id
        if role in ("user", "system"):
            target += [IGNORE_INDEX] * len(encode_id)
        else:
            target += list(encode_id)

    if len(input_id) != len(target):
        raise RuntimeError(f"Token/label length mismatch: {len(input_id)} != {len(target)}")

    for idx, tid in enumerate(input_id):
        if tid in unmask_tokens_idx:
            target[idx] = tid
        if tid == image_token_id:
            input_id[idx] = IMAGE_TOKEN_INDEX

    if max_length and len(input_id) > max_length:
        input_id = input_id[:max_length]
        target = target[:max_length]

    return (
        torch.tensor(input_id, dtype=torch.long),
        torch.tensor(target, dtype=torch.long),
    )
