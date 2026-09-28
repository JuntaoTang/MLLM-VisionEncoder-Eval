"""OpenFlamingo-style few-shot eval protocol for pretrained MLLMs."""

from .base import BaseOFEvalModel
from .evaluate import evaluate_captioning, evaluate_classification, evaluate_vqa
from .prompts import caption_prompt, classification_prompt, vqa_prompt

__all__ = [
    "BaseOFEvalModel",
    "caption_prompt",
    "vqa_prompt",
    "classification_prompt",
    "evaluate_captioning",
    "evaluate_vqa",
    "evaluate_classification",
]
