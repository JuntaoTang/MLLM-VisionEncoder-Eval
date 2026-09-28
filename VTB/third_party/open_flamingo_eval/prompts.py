"""OpenFlamingo / Flamingo prompt templates."""

from __future__ import annotations

from typing import Sequence

IMAGE_TOKEN = "<image>"
END_OF_CHUNK = "<|endofchunk|>"


def caption_prompt(
    captions: Sequence[str | None],
    *,
    include_images: Sequence[bool] | None = None,
) -> str:
    """Build caption ICL prompt.

    ``captions[-1]`` may be None for the query turn (model continues after ``Output:``).
    When ``include_images[i]`` is False, the image token is omitted (Flamingo 0-shot text demos).
    """
    include_images = include_images or [True] * len(captions)
    if len(include_images) != len(captions):
        raise ValueError("include_images must match captions length")

    parts: list[str] = []
    for i, cap in enumerate(captions):
        prefix = f"{IMAGE_TOKEN} " if include_images[i] else ""
        if cap is None:
            parts.append(f"{prefix}Output:")
        else:
            parts.append(f"{prefix}Output:{cap}{END_OF_CHUNK}")
    return "".join(parts)


def vqa_prompt(
    qa_pairs: Sequence[tuple[str, str | None]],
    *,
    include_images: Sequence[bool] | None = None,
) -> str:
    """Build VQA ICL prompt. Last answer may be None for the query turn."""
    include_images = include_images or [True] * len(qa_pairs)
    if len(include_images) != len(qa_pairs):
        raise ValueError("include_images must match qa_pairs length")

    parts: list[str] = []
    for i, (question, answer) in enumerate(qa_pairs):
        prefix = f"{IMAGE_TOKEN} " if include_images[i] else ""
        if answer is None:
            parts.append(f"{prefix}Question:{question} Short answer:")
        else:
            parts.append(
                f"{prefix}Question:{question} Short answer:{answer}{END_OF_CHUNK}"
            )
    return "".join(parts)


def classification_prompt(
    labels: Sequence[str | None],
    *,
    include_images: Sequence[bool] | None = None,
    template: str = "Output:{label}",
) -> str:
    """Build classification ICL prompt (query turn has label=None)."""
    include_images = include_images or [True] * len(labels)
    if len(include_images) != len(labels):
        raise ValueError("include_images must match labels length")

    parts: list[str] = []
    for i, label in enumerate(labels):
        prefix = f"{IMAGE_TOKEN} " if include_images[i] else ""
        if label is None:
            # Strip trailing label placeholder → leave "Output:" / "Answer:"
            bare = template.replace("{label}", "")
            parts.append(f"{prefix}{bare}")
        else:
            filled = template.format(label=label)
            parts.append(f"{prefix}{filled}{END_OF_CHUNK}")
    return "".join(parts)


def pope_prompt(
    questions: Sequence[tuple[str, str | None]],
    *,
    include_images: Sequence[bool] | None = None,
) -> str:
    """Yes/No POPE-style classification prompt."""
    include_images = include_images or [True] * len(questions)
    parts: list[str] = []
    for i, (question, answer) in enumerate(questions):
        prefix = f"{IMAGE_TOKEN} " if include_images[i] else ""
        if answer is None:
            parts.append(f"{prefix}Question:{question} Answer:")
        else:
            parts.append(f"{prefix}Question:{question} Answer:{answer}{END_OF_CHUNK}")
    return "".join(parts)


def caption_rank_prompt() -> str:
    """Query-only caption continuation prefix (model 续写 the caption)."""
    return f"{IMAGE_TOKEN} Output:"
