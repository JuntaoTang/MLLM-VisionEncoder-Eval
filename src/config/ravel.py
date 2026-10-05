"""Prepare executable RAVEL recipes from audited caches or explicit paired arrays."""
from hashlib import sha256
from pathlib import Path
import re

import numpy as np

from .schema import ConfigError
from ..core.artifacts import read_json, write_json_atomic
from ..core.hashing import sha256_file
from ..data.features import create_feature_manifest
from ..encoders import encoder_panel


TEXT_ENCODERS = ("qwen25", "qwen3", "smollm2")


def ravel_inputs(local, encoder="clip_openai__l14", text_encoder="qwen25", *,
                 patches=None, text=None, sample_ids=None, text_pooling="mean",
                 require_final_visual_layer=True):
    if text_pooling not in {"lasttok", "mean"}:
        raise ConfigError("RAVEL text pooling must be lasttok or mean")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", encoder) or text_encoder not in {*TEXT_ENCODERS, "all"}:
        raise ConfigError("RAVEL encoder IDs must be safe identifiers and text encoders must be declared")
    direct = (patches, text, sample_ids)
    if any(direct):
        if not all(direct) or encoder == "all" or text_encoder == "all":
            raise ConfigError("provide --patches, --text and --sample-ids together for one RAVEL run")
        sources = [{"encoder": encoder, "text_encoder": text_encoder,
                    "patches": str(Path(patches).expanduser().resolve()),
                    "text": str(Path(text).expanduser().resolve()),
                    "samples": str(Path(sample_ids).expanduser().resolve()),
                    "dataset": "user_supplied_paired_features"}]
    else:
        paths = local["paths"]
        if not paths.get("features"):
            raise ConfigError("configure paths.features or pass --patches, --text and --sample-ids")
        features = Path(paths["features"])
        base = features / "lcs558k/n1000_seed42/gw/outputs"
        patch_root = Path(paths.get("ravel_patches_dir", base / "lcs_patch_ravel/seed42_lang43/patches"))
        text_root = Path(paths.get("ravel_text_dir", base / "gw_reproduction/features/text"))
        samples = Path(paths.get("ravel_sample_manifest", features.parent /
            "samples/lcs558k/n1000_seed42/gw/outputs/gw_reproduction/manifests/alignment_sample_n1000_seed42.json"))
        encoders = [spec.encoder_id for spec in encoder_panel()] if encoder == "all" else [encoder]
        texts = TEXT_ENCODERS if text_encoder == "all" else [text_encoder]
        sources = [{"encoder": name, "text_encoder": llm,
                    "patches": str(patch_root / f"{name}_patch_n1000_seed42.npy"),
                    "patch_audit": str(patch_root / f"{name}_patch_n1000_seed42.json"),
                    "text": str(text_root / f"{llm}_penultimate_{text_pooling}_n1000_seed42.npy"),
                    "text_audit": str(text_root / f"{llm}_penultimate_{text_pooling}_n1000_seed42.audit.json"),
                    "samples": str(samples), "dataset": "lcs558k_n1000_seed42"}
                   for name in encoders for llm in texts]
    hashes = {}
    for source in sources:
        for key in ("patches", "text", "samples", "patch_audit", "text_audit"):
            if key not in source:
                continue
            filename = source[key]
            if filename not in hashes:
                if not Path(filename).is_file():
                    raise ConfigError(f"RAVEL {key} missing: {filename}; configure cache paths or supply paired arrays")
                hashes[filename] = sha256_file(filename)
        sample = read_json(source["samples"])
        if direct[0]:
            ids = sample
        else:
            if not isinstance(sample, dict):
                raise ConfigError("RAVEL archived sample manifest must contain records")
            rows = sample.get("records", [])
            if len(rows) != 1000 or sample.get("n_samples") != len(rows):
                raise ConfigError("RAVEL archived n1000 cache requires exactly 1000 ordered samples")
            indices = np.asarray([row["index"] for row in rows], dtype=np.int64)
            ids = [f"lcs558k:{index}" for index in indices]
            for key in ("patch_audit", "text_audit"):
                audit = read_json(source[key])
                if audit.get("sample_manifest_sha256") != hashes[source["samples"]]:
                    raise ConfigError(f"RAVEL sample provenance mismatch: {source[key]}")
            patch_audit = read_json(source["patch_audit"])
            if require_final_visual_layer and text_pooling == "mean" and patch_audit.get("feature_layer") not in (-1, "final"):
                raise ConfigError(f"paper RAVEL requires verified final-layer visual patches: {source['patch_audit']}")
            # Special exports pin the ordered manifest without a second index hash.
            index_hash = patch_audit.get("sample_index_array_sha256")
            if index_hash is not None and index_hash != sha256(indices.tobytes()).hexdigest():
                raise ConfigError("RAVEL patch row order differs from the sample manifest")
            text_audit = read_json(source["text_audit"])
            if text_pooling == "mean" and (
                    text_audit.get("feature_surface") != "penultimate_hidden_state" or
                    text_audit.get("token_policy") != "mean_valid_tokens"):
                raise ConfigError("paper RAVEL requires audited penultimate mean-pooled text features")
            if text_audit.get("feature_file_sha256") != hashes[source["text"]]:
                raise ConfigError("RAVEL text cache differs from its audited content")
        if not isinstance(ids, list) or not ids or not all(isinstance(item, str) and item for item in ids) or len(set(ids)) != len(ids):
            raise ConfigError("RAVEL needs a nonempty list of unique sample IDs in exact row order")
        for key, rank in (("patches", 3), ("text", 2)):
            array = np.load(source[key], mmap_mode="r", allow_pickle=False)
            if array.ndim != rank or len(array) != len(ids) or any(dim < 1 for dim in array.shape) or array.dtype.kind not in "fiu":
                raise ConfigError(f"RAVEL {key} must have rank {rank} and one row per sample ID")
            if not direct[0]:
                audit = read_json(source["patch_audit" if key == "patches" else "text_audit"])
                if audit.get("shape") != list(array.shape) or audit.get("dtype") != str(array.dtype):
                    raise ConfigError(f"RAVEL cache shape/dtype differs from its audit: {source[key]}")
        source["sample_ids"] = ids
    return sources, hashes


def generate_ravel_suite(output, sources, *, k=100, device="cuda:0", full_panel=False):
    if any(not 0 < k < len(source["sample_ids"]) for source in sources):
        raise ConfigError("RAVEL k must be between 1 and number_of_samples - 1")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    experiments = []
    manifests = {}
    for source in sources:
        row_files = {}
        for key in ("patches", "text"):
            path = source[key]
            if path not in manifests:
                filename = f"rows_{len(manifests):03d}.json"
                create_feature_manifest(path, source["sample_ids"], output / filename)
                manifests[path] = filename
            row_files[key] = str(output / manifests[path])
        name = f"ravel_{source['encoder']}__{source['text_encoder']}"
        filename = name + ".json"
        experiments.append(filename)
        config = {"schema_version": 1, "experiment": {"name": name, "kind": "method"},
                  "dataset": source["dataset"], "method": "ravel",
                  "inputs": {"visual_features": source["patches"], "text_features": source["text"],
                             "visual_manifest": row_files["patches"], "text_manifest": row_files["text"],
                             "sample_manifest": source["samples"]},
                  "protocol": {"k": k, "whitening_eps": 1e-4, "eigenvalue_floor": 1e-10, "device": device},
                  "runtime": {"environment": "core", "device": device}}
        for key in ("patch_audit", "text_audit"):
            if key in source:
                config["inputs"][key] = source[key]
        if full_panel:
            config["report_cell"] = {"encoder_id": source["encoder"],
                                     "column_id": "ravel__" + source["text_encoder"], "metric": "final_score"}
        write_json_atomic(output / filename, config)
    reporting = {"format": "long_table"}
    if full_panel:
        texts = list(dict.fromkeys(source["text_encoder"] for source in sources))
        reporting = {"format": "panel_table", "columns": [
            {"id": "ravel__" + llm, "method": "ravel", "dataset": sources[0]["dataset"], "unit": "unitless"}
            for llm in texts]}
    write_json_atomic(output / "suite.json", {"schema_version": 1,
        "experiment": {"name": "ravel_cached", "kind": "suite"},
        "experiments": experiments, "reporting": reporting})
