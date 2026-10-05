"""Frozen text extraction entry point; historical pooling remains explicit."""

def encode_captions(*args, **kwargs):
    from ..workers.alignment.encode_text import encode as implementation
    return implementation(*args,**kwargs)
