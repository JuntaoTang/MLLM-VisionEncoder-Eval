"""Local (no API) scoring aligned with VLMEvalKit exact_matching / official metrics."""

from __future__ import annotations

import os
import sys
from typing import Any

import pandas as pd

from src.evaluation.dataset_config import eval_dataset_type

VTB_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VLMEVAL_ROOT = os.path.join(VTB_ROOT, "third_party", "VLMEvalKit")


def _ensure_vlmeval() -> None:
    for path in (VLMEVAL_ROOT, VTB_ROOT):
        if path not in sys.path:
            sys.path.insert(0, path)
    from src.evaluation.vlmeval_patches import apply_vlmeval_scoring_patches

    apply_vlmeval_scoring_patches()


def _read_total(xlsx_path: str) -> int:
    return len(pd.read_excel(xlsx_path))


def _fraction_or_percent(value: float) -> float:
    """Normalize VLMEvalKit scores that may be 0-1 or 0-100."""
    if value > 1.0:
        return value / 100.0
    return float(value)


def _overall_from_acc_df(acc_df: pd.DataFrame) -> float | None:
    if acc_df is None or len(acc_df) == 0:
        return None
    if "Overall" not in acc_df.columns:
        return None
    return _fraction_or_percent(float(acc_df["Overall"].max()))


def _vlmeval_records(result: Any) -> list[dict[str, Any]] | dict[str, Any] | None:
    if result is None:
        return None
    if isinstance(result, pd.DataFrame):
        return result.to_dict(orient="records")
    if isinstance(result, dict):
        return result
    return None


def _resolve_lmudata_dir(eval_cfg: dict | None = None) -> str:
    """Prefer configured path-based LMUData; never fall back to ~/LMUData downloads."""
    for candidate in (
        (eval_cfg or {}).get("lmudata_dir"),
        os.environ.get("LMUData"),
        "/cache/data/.lmudata",
    ):
        if candidate and os.path.isdir(str(candidate)):
            return str(candidate)
    return "/cache/data/.lmudata"


def _score_mcq_vlmeval(
    xlsx_path: str,
    dataset_name: str,
    *,
    eval_cfg: dict | None = None,
) -> dict[str, Any]:
    # Local TSVs use string indices (test_Accounting_10); unofficial ~/LMUData
    # downloads use numeric indices and break subset scoring asserts.
    os.environ["LMUData"] = _resolve_lmudata_dir(eval_cfg)
    _ensure_vlmeval()
    from vlmeval.dataset import build_dataset

    from src.evaluation.judge_config import build_judge_kwargs, uses_llm_judge

    dataset = build_dataset(dataset_name)
    if dataset is None:
        raise RuntimeError(f"VLMEvalKit cannot build dataset {dataset_name}")

    score_path = xlsx_path
    if "MMMU" in dataset_name.upper():
        from src.evaluation.mcq_utils import ensure_mcq_letter_columns_xlsx

        score_path = ensure_mcq_letter_columns_xlsx(xlsx_path)

    judge_kwargs = {"model": "exact_matching", "nproc": 4}
    judge_name = "exact_matching"
    if eval_cfg is not None and uses_llm_judge(dataset_name):
        judge_kwargs = build_judge_kwargs(eval_cfg, dataset_name)
        judge_name = str(judge_kwargs.get("model") or "exact_matching")
        if judge_name != "exact_matching":
            from src.evaluation.judge_manager import ensure_local_judge

            ensure_local_judge(eval_cfg)

    acc_df = dataset.evaluate(score_path, **judge_kwargs)
    accuracy = _overall_from_acc_df(acc_df)
    total = _read_total(score_path)
    correct = round(accuracy * total) if accuracy is not None else None
    metric = "mcq_exact_matching" if judge_name == "exact_matching" else f"mcq_judge:{judge_name}"
    return {
        "dataset": dataset_name,
        "metric": metric,
        "judge": judge_name,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "predictions_file": score_path,
        "vlmeval_acc": _vlmeval_records(acc_df),
    }


def _score_vqa_vlmeval(xlsx_path: str, dataset_name: str) -> dict[str, Any]:
    """VQAv2-style soft accuracy (official annotator agreement)."""
    _ensure_vlmeval()
    from vlmeval.dataset.utils.vqa_eval import hit_calculate, process_line

    df = pd.read_excel(xlsx_path)
    if "prediction" not in df.columns:
        raise ValueError(f"{xlsx_path} has no prediction column")
    if "answer" not in df.columns and "answers" in df.columns:
        df = df.copy()
        df["answer"] = df["answers"]
    df["prediction"] = df["prediction"].fillna("").astype(str)

    lines = [df.iloc[i] for i in range(len(df))]
    results = [process_line(line, method="vqa_score") for line in lines]
    scores = hit_calculate(results, dataset_name)
    accuracy = float(sum(scores) / len(scores)) if scores else 0.0
    return {
        "dataset": dataset_name,
        "metric": "vqa_soft_accuracy",
        "accuracy": accuracy,
        "correct": None,
        "total": len(df),
        "predictions_file": xlsx_path,
    }


def _score_vlmeval_dataset(xlsx_path: str, dataset_name: str, metric: str) -> dict[str, Any]:
    """Delegate to VLMEvalKit dataset.evaluate with exact_matching (no API judge)."""
    _ensure_vlmeval()
    from vlmeval.dataset import build_dataset

    dataset = build_dataset(dataset_name)
    if dataset is None:
        raise RuntimeError(f"VLMEvalKit cannot build dataset {dataset_name}")

    result = dataset.evaluate(xlsx_path, model="exact_matching", nproc=4)
    total = _read_total(xlsx_path)
    payload: dict[str, Any] = {
        "dataset": dataset_name,
        "metric": metric,
        "total": total,
        "predictions_file": xlsx_path,
        "vlmeval_acc": _vlmeval_records(result),
    }

    if isinstance(result, pd.DataFrame):
        if "Overall" in result.columns:
            accuracy = _overall_from_acc_df(result)
            payload["accuracy"] = accuracy
            payload["correct"] = round(accuracy * total) if accuracy is not None else None
            return payload

        row = result.iloc[0]
        cols = {str(c).lower(): c for c in result.columns}
        if "perception" in cols and "reasoning" in cols:
            perception = float(row[cols["perception"]])
            reasoning = float(row[cols["reasoning"]])
            payload["metric"] = "mme_perception_reasoning"
            payload["perception"] = perception
            payload["reasoning"] = reasoning
            payload["accuracy"] = perception + reasoning
            payload["correct"] = None
            return payload

    if isinstance(result, dict):
        cider = result.get("CIDEr")
        if cider is None:
            cider = result.get("Cider")
        if cider is not None:
            payload["metric"] = "caption_cider"
            payload["accuracy"] = _fraction_or_percent(float(cider))
            payload["caption_metrics"] = result
            payload["correct"] = None
            return payload

    raise RuntimeError(f"Unsupported VLMEvalKit evaluate result for {dataset_name}: {type(result)}")


def _score_caption_vlmeval(xlsx_path: str, dataset_name: str) -> dict[str, Any]:
    """Caption metrics (BLEU / ROUGE-L / CIDEr) via VLMEvalKit ImageCaptionDataset."""
    _ensure_vlmeval()
    from vlmeval.dataset.image_caption import ImageCaptionDataset

    result = ImageCaptionDataset.evaluate(xlsx_path)
    total = _read_total(xlsx_path)
    cider = result.get("CIDEr") or result.get("Cider")
    if cider is None:
        raise RuntimeError(f"No CIDEr score returned for {dataset_name}")
    return {
        "dataset": dataset_name,
        "metric": "caption_cider",
        "accuracy": _fraction_or_percent(float(cider)),
        "correct": None,
        "total": total,
        "predictions_file": xlsx_path,
        "caption_metrics": result,
        "vlmeval_acc": result,
    }


def _score_mme_local(xlsx_path: str) -> dict[str, Any]:
    """MME perception+reasoning; tolerate incomplete image pairs after subsample."""
    _ensure_vlmeval()
    from collections import defaultdict

    import numpy as np
    from vlmeval.dataset.utils.yorn import MME_rating, YOrN_Extraction
    from vlmeval.smp import dump, load

    aux = xlsx_path.replace(".xlsx", "_auxmatch.xlsx")
    if os.path.isfile(aux):
        data = load(aux)
    else:
        data = load(xlsx_path)
        if "score" not in data.columns:
            data = data.copy()
            data["extracted"] = [YOrN_Extraction(str(p)) for p in data["prediction"]]
            ans = data["answer"].astype(str).str.lower().str.strip()
            ext = data["extracted"].astype(str).str.lower().str.strip()
            data["score"] = ans == ext
            dump(data, aux)

    if "score" not in data.columns:
        raise RuntimeError(f"MME scoring requires a score column: {xlsx_path}")

    stats: dict[str, dict[str, list]] = defaultdict(dict)
    for i in range(len(data)):
        item = data.iloc[i]
        category = str(item["category"])
        image_path = item["image_path"]
        score = item["score"]
        stats[category].setdefault(image_path, []).append(float(score))

    n_pairs = sum(1 for cat in stats.values() for v in cat.values() if len(v) >= 2)
    n_single = sum(1 for cat in stats.values() for v in cat.values() if len(v) == 1)

    if n_pairs == 0:
        scores = [float(x) for x in data["score"].tolist() if pd.notna(x)]
        acc = float(np.mean(scores)) if scores else 0.0
        return {
            "dataset": "MME",
            "metric": "mme_yes_no_degraded",
            "accuracy": acc,
            "perception": None,
            "reasoning": None,
            "correct": None,
            "total": len(data),
            "predictions_file": xlsx_path,
            "warning": f"no intact MME pairs (singles={n_single}); reporting mean yes/no ACC",
        }

    keep_idx = [
        i
        for i in range(len(data))
        if len(stats[str(data.iloc[i]["category"])].get(data.iloc[i]["image_path"], [])) >= 2
    ]
    paired = data.iloc[keep_idx].reset_index(drop=True)
    tmp = xlsx_path.replace(".xlsx", "_paired_for_score.xlsx")
    dump(paired, tmp)
    try:
        result = MME_rating(tmp)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass

    if isinstance(result, pd.DataFrame):
        row = result.iloc[0]
        cols = {str(c).lower(): c for c in result.columns}
        perception = float(row[cols["perception"]]) if "perception" in cols else 0.0
        reasoning = float(row[cols["reasoning"]]) if "reasoning" in cols else 0.0
        records = _vlmeval_records(result)
    else:
        perception = float(result.get("perception", 0.0) or 0.0)
        reasoning = float(result.get("reasoning", 0.0) or 0.0)
        records = result

    return {
        "dataset": "MME",
        "metric": "mme_perception_reasoning",
        "accuracy": perception + reasoning,
        "perception": perception,
        "reasoning": reasoning,
        "correct": None,
        "total": len(data),
        "pairs_scored": n_pairs,
        "pairs_dropped_singles": n_single,
        "predictions_file": xlsx_path,
        "vlmeval_acc": records,
    }


def score_predictions_local(
    xlsx_path: str,
    dataset_name: str,
    *,
    eval_cfg: dict | None = None,
) -> dict[str, Any]:
    """Score predictions; optional local/API LLM judge for MMMU/MMBench MCQ."""
    etype = eval_dataset_type(dataset_name)

    if etype == "caption":
        return _score_caption_vlmeval(xlsx_path, dataset_name)
    if etype == "yes_no":
        if "MME" in dataset_name.upper():
            return _score_mme_local(xlsx_path)
        return _score_vlmeval_dataset(xlsx_path, dataset_name, "mme_yes_no")
    if etype == "vqa":
        # Soft VQA: VQAv2 / VizWiz (and similar) with multi-annotator answers in xlsx.
        # Prefer local soft accuracy; VLMEvalKit evaluate() needs `answers` and fails on path-based TSVs.
        name_u = dataset_name.upper()
        if any(k in name_u for k in ("VQAV2", "VIZWIZ")):
            return _score_vqa_vlmeval(xlsx_path, dataset_name)
        try:
            return _score_vlmeval_dataset(xlsx_path, dataset_name, "vqa_soft_accuracy")
        except Exception:
            return _score_vqa_vlmeval(xlsx_path, dataset_name)
    if etype == "mcq":
        return _score_mcq_vlmeval(xlsx_path, dataset_name, eval_cfg=eval_cfg)

    name_upper = dataset_name.upper()
    if "VQA" in name_upper:
        return _score_vqa_vlmeval(xlsx_path, dataset_name)
    return _score_mcq_vlmeval(xlsx_path, dataset_name, eval_cfg=eval_cfg)


def rescore_run(results_root: str, datasets: list[str] | None = None) -> dict[str, Any]:
    """Rescore existing predictions.xlsx files under a run directory."""
    summaries: dict[str, Any] = {}
    if datasets is None:
        datasets = [
            name
            for name in sorted(os.listdir(results_root))
            if os.path.isfile(os.path.join(results_root, name, "predictions.xlsx"))
        ]
    for dataset_name in datasets:
        xlsx = os.path.join(results_root, dataset_name, "predictions.xlsx")
        if not os.path.isfile(xlsx):
            continue
        summaries[dataset_name] = score_predictions_local(xlsx, dataset_name)
    return summaries
