"""Flat eval results layout: results/{llm}/{vision}/{dataset}/predictions.xlsx + summary.json."""

from __future__ import annotations

import ast
import glob
import json
import os
import re
import shutil
import string
import tempfile
from typing import Any

import pandas as pd

from vision_encoder_eval.mllm.evaluation.dataset_config import (
    DatasetEvalSpec,
    ensure_lmudata_images_link,
    prepare_dataset_tsv,
)
from vision_encoder_eval.mllm.evaluation.lmu_data import vqa_score
from vision_encoder_eval.mllm.evaluation.scoring import score_predictions_local

PREDICTIONS_XLSX = "predictions.xlsx"
SUMMARY_JSON = "summary.json"


def dataset_dir(results_root: str, dataset_name: str) -> str:
    return os.path.join(results_root, dataset_name)


def predictions_path(results_root: str, dataset_name: str) -> str:
    return os.path.join(dataset_dir(results_root, dataset_name), PREDICTIONS_XLSX)


def summary_path(results_root: str) -> str:
    return os.path.join(results_root, SUMMARY_JSON)


def prepare_eval_lmudata(
    source_dir: str,
    dataset_name: str,
    max_samples: int | None,
    sample_seed: int | None = None,
) -> str:
    """Build a temporary LMUData view (TSV + image symlink). Never stored under results/."""
    cache_dir = tempfile.mkdtemp(prefix=f"vtb_lmudata_{dataset_name}_")
    prepare_dataset_tsv(
        source_dir,
        cache_dir,
        DatasetEvalSpec(dataset_name, max_samples, sample_seed),
    )
    ensure_lmudata_images_link(source_dir, cache_dir)
    return cache_dir


def make_vlmeval_work_dir() -> str:
    return tempfile.mkdtemp(prefix="vtb_vlmeval_")


def find_vlmeval_xlsx(work_dir: str, model_key: str, dataset_name: str) -> str | None:
    patterns = [
        os.path.join(work_dir, model_key, "*", f"*_{dataset_name}*.xlsx"),
        os.path.join(work_dir, model_key, "*", f"{model_key}_{dataset_name}*.xlsx"),
        os.path.join(work_dir, f"{model_key}_{dataset_name}*.xlsx"),
        os.path.join(work_dir, "**", f"*_{dataset_name}*.xlsx"),
        os.path.join(work_dir, "**", f"{model_key}_{dataset_name}*.xlsx"),
    ]
    matches: list[str] = []
    for pattern in patterns:
        matches.extend(glob.glob(pattern, recursive=True))
    matches = sorted(set(matches))
    return matches[-1] if matches else None


def merge_prediction_xlsx(shards: list[str], dst_path: str) -> str:
    """Merge rank/GPU shards by index (VLMEvalKit multi-process output)."""
    frames = [pd.read_excel(p) for p in shards if os.path.isfile(p)]
    if not frames:
        raise FileNotFoundError(f"No prediction shards to merge: {shards}")
    merged = pd.concat(frames, ignore_index=True)
    if "index" in merged.columns:
        merged = merged.drop_duplicates(subset=["index"], keep="last").sort_values("index")
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    merged.to_excel(dst_path, index=False)
    return dst_path


def publish_predictions(src_xlsx: str, results_root: str, dataset_name: str) -> str:
    dst = predictions_path(results_root, dataset_name)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src_xlsx, dst)
    # VLMEvalKit MMMU_result_transfer expects ``id``; local TSVs only have ``index``.
    if "MMMU" in dataset_name.upper():
        try:
            df = pd.read_excel(dst)
            if "id" not in df.columns and "index" in df.columns:
                df["id"] = df["index"]
                df.to_excel(dst, index=False)
        except Exception:
            pass
    return dst


def _parse_vqa_references(raw: Any) -> list[str]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    text = str(raw).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return [text]
    if isinstance(parsed, list):
        return [str(x) for x in parsed if str(x).strip()]
    return [str(parsed)]


def _mcq_choices(row: pd.Series) -> list[str]:
    from vision_encoder_eval.mllm.evaluation.mcq_utils import mcq_choice_map

    choices = mcq_choice_map(row)
    if choices:
        return list(choices.keys())
    letters = []
    for ch in string.ascii_uppercase:
        if ch in row.index and pd.notna(row.get(ch)) and str(row[ch]).strip():
            letters.append(ch)
    return letters or list("ABCD")


def _extract_mcq_letter(prediction: str, choices: list[str]) -> str | None:
    try:
        from vlmeval.utils.matching_util import can_infer_option

        inferred = can_infer_option(str(prediction), choices)
        if inferred and inferred != "Z":
            return str(inferred).upper()
    except ImportError:
        pass

    text = str(prediction).upper()
    for ch in choices:
        if re.search(rf"\b{ch}\b", text):
            return ch
    m = re.match(r"^\s*([A-E])\b", text)
    return m.group(1) if m else None


def score_predictions(
    xlsx_path: str,
    dataset_name: str,
    eval_cfg: dict | None = None,
) -> dict[str, Any]:
    """Score predictions; pass eval_cfg to enable local Qwen judge for MMMU/MMBench."""
    from vision_encoder_eval.mllm.evaluation.dataset_config import eval_dataset_type

    try:
        return score_predictions_local(xlsx_path, dataset_name, eval_cfg=eval_cfg)
    except Exception as exc:
        # Caption/MCQ/VQA have dedicated metrics — never silently remap caption→MCQ 0%.
        etype = eval_dataset_type(dataset_name)
        if etype in ("caption", "mcq", "vqa", "yes_no"):
            raise RuntimeError(
                f"Scoring failed for {dataset_name} ({etype}): {exc}"
            ) from exc
        return _score_predictions_fallback(xlsx_path, dataset_name)


def _score_predictions_fallback(xlsx_path: str, dataset_name: str) -> dict[str, Any]:
    df = pd.read_excel(xlsx_path)
    if "prediction" not in df.columns:
        raise ValueError(f"{xlsx_path} has no prediction column")

    total = len(df)
    if total == 0:
        return {
            "dataset": dataset_name,
            "metric": "none",
            "accuracy": 0.0,
            "correct": 0,
            "total": 0,
            "predictions_file": xlsx_path,
        }

    name_upper = dataset_name.upper()
    if "VQA" in name_upper:
        scores = []
        for _, row in df.iterrows():
            refs = _parse_vqa_references(row.get("answer"))
            if not refs and pd.notna(row.get("multiple_choice_answer")):
                refs = [str(row["multiple_choice_answer"])]
            scores.append(vqa_score(str(row.get("prediction", "")), refs))
        accuracy = sum(scores) / len(scores)
        return {
            "dataset": dataset_name,
            "metric": "vqa_soft_accuracy",
            "accuracy": accuracy,
            "correct": None,
            "total": total,
            "predictions_file": xlsx_path,
        }

    if "answer" in df.columns:
        correct = 0
        for _, row in df.iterrows():
            gt = str(row["answer"]).strip().upper()
            if not gt:
                continue
            choices = _mcq_choices(row)
            pred = _extract_mcq_letter(str(row.get("prediction", "")), choices)
            if pred == gt:
                correct += 1
        accuracy = correct / total
        return {
            "dataset": dataset_name,
            "metric": "mcq_accuracy",
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "predictions_file": xlsx_path,
        }

    return {
        "dataset": dataset_name,
        "metric": "unknown",
        "accuracy": None,
        "correct": None,
        "total": total,
        "predictions_file": xlsx_path,
    }


def write_run_summary(results_root: str, payload: dict[str, Any]) -> str:
    path = summary_path(results_root)
    os.makedirs(results_root, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def cleanup_temp_dir(path: str | None) -> None:
    if path and os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
