"""Discrete tokenizer factory boundary (selected backend is loaded lazily)."""

def build_visual_tokenizer(config):
    from ..mllm.discrete.model.tokenizers import build_visual_tokenizer as build
    return build(config)
