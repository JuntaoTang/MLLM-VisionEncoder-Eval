"""Public evaluation interface for metrics on the fixed paper encoder panel."""
from collections.abc import Mapping
import csv
from importlib import import_module
from math import isfinite
from pathlib import Path

from .core.artifacts import write_json_atomic
from .core.hashing import sha256_file
from .data.ground_truth import BENCHMARKS, LLMS, ground_truth_path, load_ground_truth
from .encoders import encoder_panel


def _selected_llms(llms):
    selected = tuple(LLMS if llms is None else llms)
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(LLMS):
        raise ValueError(f"llms must be a nonempty, unique selection from {LLMS}")
    return tuple(llm for llm in LLMS if llm in selected)


def prediction_pairs(llms=None):
    """Canonical IDs only; no downstream labels are passed to predictors."""
    selected = _selected_llms(llms)
    return [{"encoder_id": spec.encoder_id, "llm": llm}
            for spec in encoder_panel() for llm in selected]


def _write_csv(path, rows, fields):
    path = Path(path)
    if path.resolve() == ground_truth_path().resolve():
        raise ValueError("cannot overwrite the bundled ground truth")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_prediction_template(output, *, llms=None):
    rows = [{**pair, "score": ""} for pair in prediction_pairs(llms)]
    _write_csv(output, rows, ["encoder_id", "llm", "score"])
    return len(rows)


def export_ground_truth(output):
    """Export original rounded labels and all task scores without recomputing them."""
    truth = load_ground_truth()
    rows = []
    for pair in prediction_pairs():
        encoder = truth["encoders"][pair["encoder_id"]]
        label = encoder["llms"][pair["llm"]]
        rows.append({**pair, "display_name": encoder["display_name"],
                     "family": encoder["family"], "average": label["average"],
                     **label["scores"]})
    _write_csv(output, rows, ["encoder_id", "llm", "display_name", "family",
                              "average", *BENCHMARKS])
    return len(rows)


def read_predictions(path, *, score_column="score"):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        required = {"encoder_id", "llm", score_column}
        if len(fields) != len(set(fields)) or not required.issubset(fields):
            raise ValueError(f"prediction CSV needs unique columns including {sorted(required)}")
        rows = list(reader)
        if any(None in row or any(value is None for value in row.values()) for row in rows):
            raise ValueError("prediction CSV rows must have the same number of fields as the header")
        return rows


def _validated_predictions(rows, *, llms, score_column):
    if score_column in {"encoder_id", "llm"}:
        raise ValueError("score_column must be distinct from encoder_id and llm")
    pairs = prediction_pairs(llms)
    expected = {(row["encoder_id"], row["llm"]) for row in pairs}
    predictions = {}
    for index, row in enumerate(rows, 1):
        if not isinstance(row, Mapping):
            raise ValueError(f"prediction row {index} must be a mapping")
        pair = (row.get("encoder_id"), row.get("llm"))
        if not all(isinstance(value, str) for value in pair) or pair not in expected:
            raise ValueError(f"unknown or unselected encoder/LLM in prediction row {index}: {pair}")
        if pair in predictions:
            raise ValueError(f"duplicate prediction for {pair}")
        raw = row.get(score_column)
        try:
            if isinstance(raw, bool):
                raise ValueError("boolean score")
            score = float(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"prediction score must be a finite number for {pair}: {raw!r}") from exc
        if not isfinite(score):
            raise ValueError(f"prediction score must be a finite number for {pair}: {raw!r}")
        predictions[pair] = score
    missing = expected - set(predictions)
    if missing:
        raise ValueError(f"expected exactly {len(expected)} predictions; missing {len(missing)} pairs: "
                         f"{sorted(missing)[:8]}")
    return [{**pair, "score": predictions[(pair["encoder_id"], pair["llm"])]} for pair in pairs]


def evaluate_predictions(rows, *, llms=None, score_column="score", direction="higher",
                         ground_truth=None, metric="custom"):
    """Join by IDs and correlate each LLM's 70 scores with its downstream labels.

    ``rows`` is an iterable of mappings with encoder_id, llm and a numeric score.
    ``direction='lower'`` negates predictions before computing correlations.
    Selecting ``llms`` still requires all 70 encoders for each selected LLM.
    """
    if direction not in {"higher", "lower"}:
        raise ValueError("direction must be higher or lower")
    selected = _selected_llms(llms)
    rows = _validated_predictions(rows, llms=selected, score_column=score_column)
    path = Path(ground_truth) if ground_truth is not None else ground_truth_path()
    truth = load_ground_truth(path)
    try:
        from scipy.stats import pearsonr, spearmanr
    except ImportError as exc:
        raise ValueError("correlation evaluation requires pip install -e '.[benchmark]'") from exc

    def correlate(predictions, labels):
        if len(set(labels)) == 1:
            return {"spearman": None, "pearson": None, "reason": "constant ground truth"}
        result = {"spearman": float(spearmanr(predictions, labels).statistic),
                  "pearson": float(pearsonr(predictions, labels).statistic)}
        if not all(isfinite(value) for value in result.values()):
            raise ValueError("correlation is undefined or numerically nonfinite")
        return result

    per_llm = {}
    for llm in selected:
        group = [row for row in rows if row["llm"] == llm]
        predictions = [row["score"] * (1 if direction == "higher" else -1) for row in group]
        if len(set(predictions)) == 1:
            raise ValueError(f"correlation is undefined for constant predictions: {llm}")
        labels = [truth["encoders"][row["encoder_id"]]["llms"][llm] for row in group]
        per_llm[llm] = {
            "n_encoders": len(group),
            **correlate(predictions, [label["average"] for label in labels]),
            "per_benchmark": {benchmark: correlate(predictions, [label["scores"][benchmark]
                                                                  for label in labels])
                              for benchmark in BENCHMARKS},
        }
    return {"schema_version": 1, "status": "complete", "metric": metric,
            "n_pairs": len(rows), "llms": list(selected),
            "scope": "full_panel" if selected == LLMS else "selected_llms",
            "score_column": score_column, "direction": direction,
            "protocol": "Per-LLM correlation over all 70 encoders; original rounded 11-task average",
            "ground_truth_sha256": sha256_file(path),
            "ground_truth_provenance": truth["provenance"], "per_llm": per_llm}


def evaluate_file(predictions, output, **options):
    """Evaluate a CSV and write correlations.json and correlations.csv."""
    rows = read_predictions(predictions, score_column=options.get("score_column", "score"))
    report = evaluate_predictions(rows, **options)
    report["predictions_sha256"] = sha256_file(predictions)
    output = Path(output)
    destinations = [output / "correlations.json", output / "correlations.csv"]
    protected = {Path(predictions).resolve(), ground_truth_path().resolve()}
    if options.get("ground_truth") is not None:
        protected.add(Path(options["ground_truth"]).resolve())
    if any(path.resolve() in protected for path in destinations):
        raise ValueError("report output cannot overwrite predictions or ground truth")
    records = []
    for llm, result in report["per_llm"].items():
        for target, values in [("average", result), *result["per_benchmark"].items()]:
            records.append({"llm": llm, "target": target, "n_encoders": result["n_encoders"],
                            "spearman": values["spearman"], "pearson": values["pearson"]})
    _write_csv(destinations[1], records, ["llm", "target", "n_encoders", "spearman", "pearson"])
    write_json_atomic(destinations[0], report)
    return report


def run_metric(entrypoint, output, *, llms=None, direction="higher", metric=None):
    """Run an importable module:function taking keyword encoder_id and llm."""
    module_name, separator, function_name = entrypoint.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError("metric entrypoint must be module:function")
    try:
        predictor = getattr(import_module(module_name), function_name)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"cannot load metric entrypoint {entrypoint}: {exc}") from exc
    if not callable(predictor):
        raise ValueError(f"metric entrypoint is not callable: {entrypoint}")
    selected = _selected_llms(llms)
    pairs = prediction_pairs(selected)
    rows = [{**pair, "score": predictor(**pair)} for pair in pairs]
    rows = _validated_predictions(rows, llms=selected, score_column="score")
    predictions = Path(output) / "predictions.csv"
    _write_csv(predictions, rows, ["encoder_id", "llm", "score"])
    return evaluate_file(predictions, output, llms=selected, direction=direction,
                         metric=metric or entrypoint)
