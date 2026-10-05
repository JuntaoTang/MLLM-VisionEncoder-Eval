"""The exact 70-encoder x 3-backbone downstream labels used in the paper."""
from math import fsum, isfinite
from pathlib import Path

from ..core.artifacts import read_json, write_json_atomic
from ..core.hashing import sha256_file
from ..encoders import encoder_panel


LLMS = ("qwen3", "qwen25", "smollm2")
BENCHMARKS = (
    "MMMU_TEST", "MMBench_TEST_EN_V11", "VQAv2_VAL", "ScienceQA_VAL",
    "ChartQA_TEST", "DocVQA_VAL", "TextVQA_VAL", "POPE",
    "GQA_TestDev_Balanced", "MSCOCO_KARPATHY_TEST", "FLICKR30K_KARPATHY_TEST",
)


def ground_truth_path():
    return Path(__file__).resolve().parents[1] / "resources/ground_truth.json"


def validate_ground_truth(value):
    expected = {spec.encoder_id for spec in encoder_panel()}
    if value.get("n_encoders") != 70 or value.get("n_pairs") != 210:
        raise ValueError("paper ground truth must declare 70 encoders and 210 pairs")
    if value.get("llms") != list(LLMS) or value.get("benchmarks") != list(BENCHMARKS):
        raise ValueError("paper ground truth must use the three backbones and eleven benchmarks")
    encoders = value.get("encoders", {})
    if set(encoders) != expected:
        raise ValueError(f"paper encoder pool mismatch: missing={sorted(expected - set(encoders))}, "
                         f"extra={sorted(set(encoders) - expected)}")
    order = value.get("order", [])
    if len(order) != 70 or set(order) != expected:
        raise ValueError("paper ground truth order must contain each encoder exactly once")
    for name, encoder in encoders.items():
        if encoder.get("stem") != name or set(encoder.get("llms", {})) != set(LLMS):
            raise ValueError(f"paper ground truth needs all three backbones for {name}")
        for llm, row in encoder["llms"].items():
            scores = row.get("scores", {})
            if set(scores) != set(BENCHMARKS):
                raise ValueError(f"paper ground truth needs all eleven benchmark scores: {name}/{llm}")
            numbers = [row.get("average"), *scores.values()]
            if any(isinstance(number, bool) or not isinstance(number, (int, float)) or
                   not isfinite(number) for number in numbers):
                raise ValueError(f"paper ground truth contains an invalid score: {name}/{llm}")
            # The published averages are rounded; preserve those labels exactly.
            if abs(row["average"] - fsum(scores.values()) / len(BENCHMARKS)) > 0.005 + 1e-10:
                raise ValueError(f"paper ground truth average differs from benchmark mean: {name}/{llm}")
    return value


def load_ground_truth(path=None):
    return validate_ground_truth(read_json(path or ground_truth_path()))


def ground_truth_rows(path=None):
    value = load_ground_truth(path)
    return [{"encoder_id": spec.encoder_id, "llm": llm,
             **value["encoders"][spec.encoder_id]["llms"][llm]}
            for spec in encoder_panel() for llm in LLMS]


def prepare_ground_truth(source, output):
    """Select only paper pairs and remove run-specific bookkeeping."""
    raw = read_json(source)
    names = {spec.encoder_id for spec in encoder_panel()}
    encoders = {}
    for name in raw["order"]:
        if name not in names:
            continue
        old = raw["encoders"][name]
        encoders[name] = {key: old[key] for key in
                          ("stem", "display_name", "type", "family", "rank", "mean")}
        encoders[name]["llms"] = {
            llm: {"average": old["llms"][llm]["average"],
                  "scores": {benchmark: old["llms"][llm]["scores"][benchmark]
                             for benchmark in BENCHMARKS}}
            for llm in LLMS}
    value = {"schema_version": 1, "n_encoders": 70, "n_pairs": 210,
             "llms": list(LLMS), "llm_labels": {llm: raw["llm_labels"][llm] for llm in LLMS},
             "benchmarks": list(BENCHMARKS), "sorted_by": raw["sorted_by"],
             "order": list(encoders), "encoders": encoders,
             "provenance": {"source_sha256": sha256_file(source),
                            "averages": "original two-decimal labels; not recomputed",
                            "score_units": "percentage; captioning CIDEr multiplied by 100"}}
    validate_ground_truth(value)
    write_json_atomic(output, value)
    return value
