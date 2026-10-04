"""Reproducible vision-encoder evaluation framework."""

from .core.registry import METHODS, MethodSpec
from . import methods as _methods  # noqa: F401  # register migrated methods

__all__ = ["METHODS", "MethodSpec"]
__version__ = "0.1.0"
