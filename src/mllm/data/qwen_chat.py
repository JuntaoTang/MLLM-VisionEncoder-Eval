"""Qwen3 ChatML prompt helpers shared by continuous/discrete eval."""

from __future__ import annotations

IMAGE_TOKEN_INDEX = -200
IGNORE_INDEX = -100

_IM_END_CANDIDATES = ("<|im_end|>",)


def qwen_im_end_token(tokenizer) -> str:
    """Return the tokenizer's im_end special token string."""
    for tok in _IM_END_CANDIDATES:
        try:
            tid = tokenizer.convert_tokens_to_ids(tok)
        except Exception:
            tid = None
        if tid is None:
            continue
        unk = getattr(tokenizer, "unk_token_id", None)
        if unk is not None and tid == unk:
            continue
        return tok
    vocab = getattr(tokenizer, "get_vocab", lambda: {})()
    for tok in _IM_END_CANDIDATES:
        if tok in vocab:
            return tok
    return ""


def build_qwen_chat_prompt(tokenizer, user_content: str) -> tuple[str, str]:
    """Build a single-turn Qwen3 ChatML prompt; returns (prompt, stop_str)."""
    stop = qwen_im_end_token(tokenizer)
    system = "<|im_start|>system\nYou are a helpful assistant."
    prompt = (
        f"{system}{stop}\n"
        f"<|im_start|>user\n{user_content}{stop}\n"
        f"<|im_start|>assistant\n"
    )
    return prompt, stop
