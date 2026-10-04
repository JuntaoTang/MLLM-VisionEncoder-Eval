"""Model exports loaded only when selected."""
from importlib import import_module

__all__ = ['DiscreteVisualAdapter', 'BaseDiscreteTokenizer', 'build_visual_tokenizer']

def __getattr__(name):
    if name == 'DiscreteVisualAdapter':
        return import_module('.discrete_adapter', __name__).DiscreteVisualAdapter
    if name in {'BaseDiscreteTokenizer', 'build_visual_tokenizer'}:
        return getattr(import_module('.tokenizers', __name__),name)
    raise AttributeError(name)
