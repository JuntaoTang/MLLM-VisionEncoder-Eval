"""Reproduce the paper's 210 RAVEL scores and downstream correlations."""
import copy
import csv
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
import time

import numpy as np

from .config.ravel import ravel_inputs
from .core.artifacts import read_json, write_json_atomic
from .core.hashing import sha256_file, sha256_json
from .data.ground_truth import BENCHMARKS, LLMS, ground_truth_path, load_ground_truth
from .methods.neighbors import binary_overlap, topk_neighbors
from .methods.patch_similarity import (
    fit_full_rank_patch_pca, transform_patches_full_rank, symmetric_chamfer_similarity,
)
from .methods.preprocessing import full_rank_pca_whiten_l2


MODELS = {"qwen3": "Qwen/Qwen3-1.7B", "qwen25": "Qwen/Qwen2.5-1.5B-Instruct",
          "smollm2": "HuggingFaceTB/SmolLM2-1.7B-Instruct"}
MODEL_REVISIONS = {"qwen3": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
                   "qwen25": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
                   "smollm2": "31b70e2e869a7173562077fd711b654946d38674"}
PAPER = {"qwen3": {"spearman": 0.820, "pearson": 0.816},
         "qwen25": {"spearman": 0.897, "pearson": 0.896},
         "smollm2": {"spearman": 0.833, "pearson": 0.845}}


def text_neighbors(values, *, k, device):
    whitened, meta = full_rank_pca_whiten_l2(values, whitening_eps=1e-4, eigenvalue_floor=1e-10)
    similarity = np.asarray(whitened @ whitened.T, dtype=np.float32)
    np.fill_diagonal(similarity, -np.inf)
    return topk_neighbors(similarity, k, device=device), meta


def visual_neighbors(values, *, k, device):
    state = fit_full_rank_patch_pca(values, eigenvalue_floor=1e-10)
    whitened = transform_patches_full_rank(values, state, whitening_eps=1e-4)
    similarity = symmetric_chamfer_similarity(whitened, device=device, normalize_tokens=False)
    return topk_neighbors(similarity, k, device=device), state["meta"]


def cached_visual_neighbors(source, hashes, output_root, *, k, device):
    module_root = Path(__file__).parent
    identity = {"encoder_id": source["encoder"],
                "inputs": {key: hashes[source[key]] for key in ("patches", "patch_audit", "samples")},
                "k": k, "device": device, "whitening_eps": 1e-4, "eigenvalue_floor": 1e-10,
                "code": {path.name: sha256_file(path) for path in [Path(__file__), *[
                    module_root / "methods" / (name + ".py")
                    for name in ("preprocessing", "patch_similarity", "neighbors")]]},
                "environment": {name: version(name) for name in ("numpy", "torch", "scikit-learn")}}
    fingerprint = sha256_json(identity)
    output = Path(output_root) / "ravel_visual_neighbors" / fingerprint[:12]
    record_path, array_path = output / "audit.json", output / "neighbors.npy"
    if record_path.exists():
        record = read_json(record_path)
        if record.get("identity") != identity or record.get("neighbors_sha256") != sha256_file(array_path):
            raise ValueError(f"visual neighbor cache changed: {source['encoder']}")
        return np.load(array_path, allow_pickle=False), record["pca"]
    neighbors, meta = visual_neighbors(np.load(source["patches"], mmap_mode="r", allow_pickle=False),
                                      k=k, device=device)
    output.mkdir(parents=True, exist_ok=True)
    temporary = array_path.with_suffix(".tmp.npy")
    np.save(temporary, neighbors, allow_pickle=False)
    temporary.replace(array_path)
    write_json_atomic(record_path, {"identity": identity, "pca": meta,
                                   "neighbors_sha256": sha256_file(array_path)})
    return neighbors, meta


def prepare_visual_cache(local, *, device="cuda:0", k=100):
    import torch
    from threadpoolctl import threadpool_limits

    ground_truth = load_ground_truth(local["paths"].get("ground_truth", ground_truth_path()))
    features = Path(local["paths"]["features"])
    samples = Path(local["paths"].get("ravel_sample_manifest", features.parent /
        "samples/lcs558k/n1000_seed42/gw/outputs/gw_reproduction/manifests/alignment_sample_n1000_seed42.json"))
    checked = check_visual_inputs(local, samples, ground_truth, use_existing_visual_cache=True)
    if checked["issues"]:
        raise ValueError("visual cache preparation requires aligned audited patches")
    patches = Path(local["paths"].get("ravel_patches_dir", features /
        "lcs558k/n1000_seed42/gw/outputs/lcs_patch_ravel/seed42_lang43/patches"))
    hashes = {str(samples): sha256_file(samples)}
    torch.set_num_threads(8)
    with threadpool_limits(limits=8):
        for index, encoder in enumerate(ground_truth["encoders"], 1):
            started = time.monotonic()
            path = patches / f"{encoder}_patch_n1000_seed42.npy"
            source = {"encoder": encoder, "patches": str(path), "patch_audit": str(path.with_suffix(".json")),
                      "samples": str(samples)}
            print(f"Preparing visual neighbors [{index}/70]: {encoder}", flush=True)
            for key in ("patches", "patch_audit"):
                hashes[source[key]] = sha256_file(source[key])
            cached_visual_neighbors(source, hashes, local["output_root"], k=k, device=device)
            print(f"Completed visual neighbors [{index}/70]: {encoder} ({time.monotonic()-started:.1f}s)", flush=True)


def summarize_scores(rows, ground_truth):
    from scipy.stats import pearsonr, spearmanr

    expected = {(name, llm) for name in ground_truth["encoders"] for llm in LLMS}
    pairs = [(row["encoder_id"], row["llm"]) for row in rows]
    if len(pairs) != 210 or len(set(pairs)) != 210 or set(pairs) != expected:
        raise ValueError("RAVEL correlation requires exactly the 210 paper pairs")
    if not all(np.isfinite(row["ravel_score"]) for row in rows):
        raise ValueError("RAVEL correlation cannot include nonfinite scores")
    per_llm = {}
    for llm in LLMS:
        selected = [row for row in rows if row["llm"] == llm]
        predictions = [row["ravel_score"] for row in selected]
        labels = [ground_truth["encoders"][row["encoder_id"]]["llms"][llm] for row in selected]
        correlations = {"spearman": float(spearmanr(predictions, [row["average"] for row in labels]).statistic),
                        "pearson": float(pearsonr(predictions, [row["average"] for row in labels]).statistic)}
        if not all(np.isfinite(value) for value in correlations.values()):
            raise ValueError("RAVEL correlation is undefined for constant scores")
        per_llm[llm] = {"n_encoders": 70, **correlations, "paper": PAPER[llm],
                       "delta": {name: value - PAPER[llm][name] for name, value in correlations.items()},
                       "matches_published_precision": all(round(value, 3) == PAPER[llm][name]
                                                          for name, value in correlations.items()),
                       "per_benchmark": {benchmark: {
                           "spearman": float(spearmanr(predictions, [row["scores"][benchmark] for row in labels]).statistic),
                           "pearson": float(pearsonr(predictions, [row["scores"][benchmark] for row in labels]).statistic)}
                           for benchmark in BENCHMARKS}}
    return {"status": "complete", "n_pairs": 210, "per_llm": per_llm,
            "matches_published_precision": all(row["matches_published_precision"] for row in per_llm.values())}


def check_visual_inputs(local, samples, ground_truth, *, use_existing_visual_cache=False):
    """Check paper visual provenance before any model download or computation."""
    document = read_json(samples)
    rows = document.get("records", [])
    if document.get("n_samples") != 1000 or len(rows) != 1000 or len({row["index"] for row in rows}) != 1000:
        raise ValueError("paper RAVEL requires 1000 unique ordered sample records")
    if not all(isinstance(row.get("text"), str) and row["text"].strip() for row in rows):
        raise ValueError("paper RAVEL sample manifest requires a caption for every row")
    sample_hash = sha256_file(samples)
    index_hash = sha256(np.asarray([row["index"] for row in rows], dtype=np.int64).tobytes()).hexdigest()
    base = Path(local["paths"]["features"]) / "lcs558k/n1000_seed42/gw/outputs"
    patches = Path(local["paths"].get("ravel_patches_dir", base / "lcs_patch_ravel/seed42_lang43/patches"))
    issues, warnings = [], []
    for encoder in ground_truth["encoders"]:
        path = patches / f"{encoder}_patch_n1000_seed42.npy"
        audit_path = path.with_suffix(".json")
        if not path.is_file() or not audit_path.is_file():
            issues.append({"encoder_id": encoder, "issue": "visual patch array or audit missing"})
            continue
        audit = read_json(audit_path)
        if audit.get("sample_manifest_sha256") != sample_hash or audit.get("sample_index_array_sha256", index_hash) != index_hash:
            issues.append({"encoder_id": encoder, "issue": "visual sample provenance mismatch"})
        if audit.get("feature_layer") not in (-1, "final"):
            (warnings if use_existing_visual_cache else issues).append({
                "encoder_id": encoder, "issue": "final visual layer is not verified in audit",
                "recorded_layer": audit.get("feature_layer")})
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.ndim != 3 or len(array) != 1000 or audit.get("shape") != list(array.shape) or audit.get("dtype") != str(array.dtype):
            issues.append({"encoder_id": encoder, "issue": "visual shape or dtype mismatch"})
    return {"status": "blocked" if issues else "inputs_checked", "pairs": 210,
            "samples": str(samples), "issues": issues, "warnings": warnings,
            "protocol": {"visual_layer": "existing_cache_unverified" if warnings else "final",
                         "text_layer": -2, "text_pooling": "mean_valid_tokens",
                         "n_samples": 1000, "k": 100, "whitening_eps": 1e-4},
            "text_preparation": "extract from the original frozen backbones; last-token caches are not used",
            "required_models": MODELS}


def mean_penultimate_hidden_state(hidden_states, attention_mask):
    hidden = hidden_states[-2].float()
    mask = attention_mask.unsqueeze(-1).float()
    lengths = mask.sum(1)
    if (lengths <= 0).any():
        raise ValueError("cannot mean-pool a caption without valid tokens")
    return (hidden * mask).sum(1) / lengths


def prepare_text(local, samples, *, device):
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer
    import torch

    sample_hash = sha256_file(samples)
    document = read_json(samples)
    rows = document["records"]
    if len(rows) != 1000 or len({row["index"] for row in rows}) != 1000:
        raise ValueError("paper RAVEL requires 1000 unique ordered sample records")
    model_cache = Path(local["paths"].get("ravel_model_cache",
                       Path(local["paths"]["features"]).parent / "models/ravel"))
    models, audits = {}, {}
    for llm, repo in MODELS.items():
        print(f"Preparing frozen backbone: {repo}", flush=True)
        models[llm] = Path(snapshot_download(repo, revision=MODEL_REVISIONS[llm], cache_dir=str(model_cache),
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"], max_workers=4))
        audits[llm] = {"model_id": repo, "revision": models[llm].name,
                       "files": {path.name: sha256_file(path) for path in sorted(models[llm].iterdir())
                                 if path.is_file() and path.suffix in {".json", ".safetensors", ".txt", ".model", ".tiktoken"}}}
    identity = {"sample_manifest_sha256": sample_hash, "models": audits,
                "layer": -2, "pooling": "mean_valid_tokens", "padding_side": "right",
                "truncation": False, "dtype": "bfloat16", "batch_size": 32,
                "attention": "sdpa", "device": device,
                "torch": version("torch"), "transformers": version("transformers"),
                "extractor_sha256": sha256_file(Path(__file__))}
    output = Path(local["output_root"]) / "ravel_paper_features" / sha256_json(identity)[:12]
    output.mkdir(parents=True, exist_ok=True)
    for llm in LLMS:
        target = output / f"{llm}_penultimate_mean_n1000_seed42.npy"
        audit_file = target.with_suffix(".audit.json")
        if target.exists() and audit_file.exists():
            audit = read_json(audit_file)
            if audit.get("extraction_identity") != identity or audit.get("feature_file_sha256") != sha256_file(target):
                raise ValueError(f"text feature cache changed: {target}")
            print(f"Reusing verified mean-pooled text: {llm}", flush=True)
            continue
        tokenizer = AutoTokenizer.from_pretrained(models[llm], local_files_only=True)
        tokenizer.padding_side = "right"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModel.from_pretrained(models[llm], torch_dtype=torch.bfloat16,
            attn_implementation="sdpa", local_files_only=True).to(device).eval()
        values = []
        with torch.inference_mode():
            for start in range(0, len(rows), 32):
                inputs = tokenizer([row["text"] for row in rows[start:start + 32]],
                                   padding=True, truncation=False, return_tensors="pt").to(device)
                result = model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
                pooled = mean_penultimate_hidden_state(result.hidden_states, inputs["attention_mask"])
                values.append(pooled.cpu().numpy())
                del result, pooled, inputs
        array = np.concatenate(values).astype(np.float32)
        temporary = target.with_suffix(".tmp.npy")
        np.save(temporary, array, allow_pickle=False)
        temporary.replace(target)
        write_json_atomic(audit_file, {"feature_surface": "penultimate_hidden_state",
            "token_policy": "mean_valid_tokens", "sample_manifest_sha256": sample_hash,
            "sample_manifest": str(samples), "shape": list(array.shape), "dtype": str(array.dtype),
            "feature_file_sha256": sha256_file(target), "extraction_identity": identity})
        print(f"Extracted aligned mean-pooled text: {llm} {array.shape}", flush=True)
        del model, array, values
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    return output, identity


def run(local, *, device=None, k=100, check_only=False, use_existing_visual_cache=False):
    ground_path = local["paths"].get("ground_truth", ground_truth_path())
    ground_truth = load_ground_truth(ground_path)
    # The audited visual cache pins this ordered manifest; fresh text uses it directly.
    sample_path = Path(local["paths"].get("ravel_sample_manifest",
        Path(local["paths"]["features"]).parent /
        "samples/lcs558k/n1000_seed42/gw/outputs/gw_reproduction/manifests/alignment_sample_n1000_seed42.json"))
    if k != 100:
        raise ValueError("paper Table 1 RAVEL requires k=100")
    checked = check_visual_inputs(local, sample_path, ground_truth,
                                  use_existing_visual_cache=use_existing_visual_cache)
    if check_only:
        return checked
    if checked["issues"]:
        raise ValueError(f"paper visual inputs are not verified ({len(checked['issues'])} issues); run ravel_paper --check")
    import torch
    from threadpoolctl import threadpool_limits

    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(8)
    with threadpool_limits(limits=8):
        text_root, extraction = prepare_text(local, sample_path, device=device)
        paper_local = copy.deepcopy(local)
        paper_local["paths"]["ravel_text_dir"] = str(text_root)
        sources, hashes = ravel_inputs(paper_local, "all", "all", text_pooling="mean",
                                      require_final_visual_layer=not use_existing_visual_cache)
        module_root = Path(__file__).parent
        identity = {"inputs": hashes, "extraction": extraction,
                    "ground_truth_sha256": sha256_file(ground_path),
                    "protocol": {**checked["protocol"], "k": k, "eigenvalue_floor": 1e-10,
                                 "device": device, "blas_threads": 8},
                    "visual_provenance_warnings": checked["warnings"],
                    "code": {str(path.relative_to(module_root)): sha256_file(path) for path in
                             [Path(__file__), *[module_root / "methods" / (name + ".py")
                             for name in ("ravel", "preprocessing", "patch_similarity", "neighbors")]]},
                    "environment": {name: version(name) for name in
                                    ("numpy", "torch", "scikit-learn", "scipy", "transformers")}}
        fingerprint = sha256_json(identity)
        output = Path(local["output_root"]) / "ravel_paper" / fingerprint[:12]
        output.mkdir(parents=True, exist_ok=True)
        write_json_atomic(output / "protocol.json", identity)
        texts = {}
        for source in sources:
            llm = source["text_encoder"]
            if llm not in texts:
                texts[llm], _ = text_neighbors(np.load(source["text"], allow_pickle=False), k=k, device=device)
        visual = {source["encoder"]: source for source in sources}
        score_rows = []
        for index, (encoder, source) in enumerate(visual.items(), 1):
            started = time.monotonic()
            record_path = output / (encoder + ".json")
            neighbor_path = output / (encoder + ".neighbors.npy")
            if record_path.exists():
                record = read_json(record_path)
                if record["fingerprint"] != fingerprint or record["neighbors_sha256"] != sha256_file(neighbor_path):
                    raise ValueError(f"visual result cache changed: {encoder}")
                neighbors = np.load(neighbor_path, allow_pickle=False)
                if record["scores"] != {llm: binary_overlap(neighbors, texts[llm]) for llm in LLMS}:
                    raise ValueError(f"cached RAVEL scores changed: {encoder}")
            else:
                neighbors, meta = cached_visual_neighbors(source, hashes, local["output_root"], k=k, device=device)
                np.save(neighbor_path, neighbors, allow_pickle=False)
                record = {"status": "success", "encoder_id": encoder, "fingerprint": fingerprint,
                          "scores": {llm: binary_overlap(neighbors, texts[llm]) for llm in LLMS},
                          "pca": meta, "neighbors_sha256": sha256_file(neighbor_path),
                          "seconds": time.monotonic() - started}
                write_json_atomic(record_path, record)
            score_rows.extend({"encoder_id": encoder, "llm": llm, "ravel_score": record["scores"][llm],
                               "ground_truth": ground_truth["encoders"][encoder]["llms"][llm]["average"]}
                              for llm in LLMS)
            write_json_atomic(output / "progress.json", {"status": "running", "completed_pairs": len(score_rows),
                                                       "total_pairs": 210, "last_encoder": encoder})
            print(f"[{index:02d}/70] {encoder}: {record['scores']} ({time.monotonic() - started:.1f}s)", flush=True)
        report = summarize_scores(score_rows, ground_truth)
        report.update(fingerprint=fingerprint, output_directory=str(output),
                      protocol=identity["protocol"], ground_truth_sha256=identity["ground_truth_sha256"],
                      feature_scope=f"{checked['protocol']['visual_layer']} visual patches; fresh penultimate mean-pooled text",
                      paper_protocol_verified=not checked["warnings"],
                      visual_provenance_warnings=checked["warnings"],
                      labels_precision="original published two-decimal averages")
        with (output / "scores.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["encoder_id", "llm", "ravel_score", "ground_truth"])
            writer.writeheader()
            writer.writerows(score_rows)
        write_json_atomic(output / "correlations.json", report)
        write_json_atomic(output / "progress.json", {"status": "complete", "completed_pairs": 210, "total_pairs": 210})
        return report
