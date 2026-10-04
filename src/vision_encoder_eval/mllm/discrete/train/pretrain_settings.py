"""Resolve per-recipe pretrain settings from MLLM arch config."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


DEFAULT_PRETRAIN_TUNE = ("projector",)


@dataclass(frozen=True)
class PretrainSettings:
    tune_llm: bool
    tune_projector: bool
    batch: str | dict[str, Any] | None = None
    training_overrides: dict[str, Any] = field(default_factory=dict)

    @property
    def training_phase(self) -> int:
        """Map to legacy phase flags (2 when LLM is tuned)."""
        return 2 if self.tune_llm else 1

    def summary(self) -> str:
        parts = []
        if self.tune_llm:
            parts.append("llm")
        if self.tune_projector:
            parts.append("projector")
        tune = "+".join(parts) or "none"
        batch = self.batch if self.batch is not None else "pretrain"
        return f"tune={tune}, batch={batch}"


def _normalize_tune(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return DEFAULT_PRETRAIN_TUNE
    if isinstance(raw, str):
        raw = [raw]
    tune = tuple(str(x).strip().lower() for x in raw if str(x).strip())
    if not tune:
        return DEFAULT_PRETRAIN_TUNE
    allowed = {"llm", "projector"}
    unknown = set(tune) - allowed
    if unknown:
        raise ValueError(
            f"arch.pretrain.tune contains unknown modules {sorted(unknown)}; "
            f"expected subset of {sorted(allowed)}"
        )
    return tune


def resolve_pretrain_settings(cfg: dict[str, Any]) -> PretrainSettings:
    """Read arch.pretrain from a merged training config."""
    arch = cfg.get("arch") or {}
    pretrain = arch.get("pretrain") or {}
    tune = _normalize_tune(pretrain.get("tune") or pretrain.get("trainable"))
    batch = pretrain.get("batch")
    if batch is None:
        batch = pretrain.get("batch_from")
    training_overrides = dict(pretrain.get("training") or {})
    settings = PretrainSettings(
        tune_llm="llm" in tune,
        tune_projector="projector" in tune,
        batch=batch,
        training_overrides=training_overrides,
    )
    if not settings.tune_llm and not settings.tune_projector:
        raise ValueError("arch.pretrain.tune must include at least one of: llm, projector")
    return settings


def resolve_pretrain_batch(
    batch_cfg: dict[str, Any],
    *,
    finetune_tag: str,
    pretrain_settings: PretrainSettings,
) -> dict[str, int] | None:
    """Return explicit micro/accum batch dict, or None to use stage defaults."""
    batch = pretrain_settings.batch
    if batch is None:
        return None
    if isinstance(batch, dict):
        return {
            "per_device_train_batch_size": int(batch["per_device_train_batch_size"]),
            "gradient_accumulation_steps": int(batch["gradient_accumulation_steps"]),
        }
    if not isinstance(batch, str):
        raise ValueError(
            f"arch.pretrain.batch must be 'finetune', 'pretrain', or a dict; got {batch!r}"
        )
    key = batch.strip().lower()
    if key == "pretrain":
        return None
    if key != "finetune":
        raise ValueError(
            f"arch.pretrain.batch string must be 'finetune' or 'pretrain'; got {batch!r}"
        )
    stage_cfg = batch_cfg.get("finetune", {})
    if finetune_tag and isinstance(stage_cfg, dict):
        tagged = stage_cfg.get(finetune_tag)
        if isinstance(tagged, dict):
            stage_cfg = tagged
    if not isinstance(stage_cfg, dict):
        raise ValueError("arch.pretrain.batch=finetune but no finetune batch preset found")
    return {
        "per_device_train_batch_size": int(stage_cfg.get("per_device_train_batch_size", 4)),
        "gradient_accumulation_steps": int(stage_cfg.get("gradient_accumulation_steps", 8)),
    }
