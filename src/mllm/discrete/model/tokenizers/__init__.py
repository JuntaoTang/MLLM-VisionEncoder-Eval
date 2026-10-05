"""Tokenizer construction is lazy; unselected backends need no dependencies."""
from importlib import import_module
from .base import BaseDiscreteTokenizer

__all__ = [
    "BaseDiscreteTokenizer",
    "TokLIPTokenizer",
    "UniTokTokenizer",
    "VilaUTokenizer",
    "build_visual_tokenizer",
]

def build_visual_tokenizer(cfg):
    from .factory import build_visual_tokenizer as build
    return build(cfg)

def __getattr__(name):
    modules = {'TokLIPTokenizer':'toklip.wrapper','UniTokTokenizer':'unitok.wrapper','VilaUTokenizer':'vilau.wrapper'}
    if name in modules:
        return getattr(import_module('.'+modules[name],__name__),name)
    raise AttributeError(name)
