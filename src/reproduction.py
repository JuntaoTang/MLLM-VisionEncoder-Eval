"""Repository reproduction shortcuts; scientific execution stays in core."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from .config import ConfigError, resolve_config
from .config.loader import _expand_environment, _load_document, _resolve_local_paths
from .config.schema import validate_local_config
from .core.artifacts import read_json, write_json_atomic
from .core.experiment import matrix_configs, plan, run_experiment
from .core.hashing import sha256_file, sha256_json
from .core.runtime import repository_root
from .workers.registry import get_worker


RECIPES = {
    "ravel_paper": ("ravel_paper", "210 paper RAVEL pairs; fresh mean-pooled text and ground-truth correlations"),
    "ravel": ("ravel", "Paper-protocol RAVEL on audited final-layer patches and penultimate mean text"),
    "linear_probe": ("linear_probe", "CLIP linear probe; ImageNet images required"),
    "alignment_probe": ("alignment_probe", "CLIP/Qwen2.5 alignment on COCO"),
    "ckax": ("ckax", "CKA-X with calibration features and labels"),
    "ckax_budget": ("ckax_budget", "CKA-X calibration budget sweep"),
    "zero_shot": ("zero_shot", "MetaCLIP2 zero-shot on ImageNet validation"),
    "zero_shot_clip_benchmark": ("zero_shot_clip_benchmark", "CLIP benchmark zero-shot recipe"),
    "tokbench": ("tokbench", "Summarize existing TokBench scores"),
    "law": ("law", "Law A/C scores; trained models and SPair-71k required"),
    "mllm_train": ("mllm_continuous_train", "Configured continuous MLLM pretrain + finetune"),
    "mllm_eval": ("mllm_eval", "Configured continuous MLLM evaluation"),
    "mllm_discrete_train": ("mllm_discrete_train", "Configured discrete MLLM pretrain + finetune"),
    "mllm_discrete_eval": ("mllm_discrete_eval", "Configured discrete MLLM evaluation"),
    "all": ("paper_all", "Ten component recipes; not the complete paper model grid"),
}
PAIRED = {"ravel", "rsa", "cca", "gw", "mutualnn"}


def load_local(path):
    path = Path(path).resolve()
    local = dict(_expand_environment(_load_document(path)))
    validate_local_config(local)
    _resolve_local_paths(local, config_dir=path.parent)
    return local


def initialize_local(path, root, data_root):
    import yaml
    from .data.ground_truth import ground_truth_path

    path = Path(path).resolve()
    if path.exists():
        raise ConfigError(f"local config already exists; edit it instead: {path}")
    local = dict(_load_document(root / "configs/local.example.yaml"))
    data_root = Path(data_root).resolve()
    local["paths"].update(
        ground_truth=str(ground_truth_path()),
        features=str(data_root / "features"),
        knn_exports=str(data_root / "features/imagenet1k/train200k"),
        knn_protocol=str(data_root / "protocols/knn/imagenet_train_195pool_5query_seed42.json"),
    )
    for name, candidate in {"datasets": Path("/cache/data"),
                            "trained": Path("/cache/ckpt/trained")}.items():
        if candidate.is_dir():
            local["paths"][name] = str(candidate)
    local["output_root"] = str(root / "runs")
    local["resource_root"] = str(root)
    for group in ("core", "knn"):
        local["environments"][group]["python"] = sys.executable
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(local, sort_keys=False), encoding="utf-8")
    print(f"Created {path}")
    print("RAVEL/kNN cache paths and bundled paper ground truth configured. Set other assets and worker interpreters as needed.")


def check_config(resolved, ancestors=()):
    """Check existence and dependency discovery without hashing large datasets."""
    expanded = matrix_configs(resolved)
    if expanded is not None:
        return [item for child in expanded for item in check_config(child, ancestors)]
    source = resolved.value["sources"]["experiment"]
    if source in ancestors:
        raise ConfigError(f"cyclic suite: {source}")
    experiment = resolved.value["experiment"]
    if experiment["experiment"]["kind"] == "suite":
        records = []
        for child in experiment["experiments"]:
            try:
                records.extend(check_config(resolve_config(
                    local_path=resolved.value["sources"]["local"], experiment_path=child
                ), (*ancestors, source)))
            except (ConfigError, KeyError, ValueError, OSError) as exc:
                records.append({"name": Path(child).stem, "issues": [str(exc)]})
        return records
    local = resolved.value["local"]
    issues = []
    try:
        execution = plan(resolved)
    except (ConfigError, KeyError, ValueError) as exc:
        execution = {}
        issues.append(str(exc))
    for name in experiment.get("required_paths", []):
        value = local["paths"].get(name)
        if not value or not Path(value).exists():
            issues.append(f"paths.{name}: missing {value}")
    for name, value in experiment.get("inputs", {}).items():
        if not isinstance(value, str) or not Path(value).exists():
            issues.append(f"inputs.{name}: missing {value}")
    for step in execution.get("steps", []):
        group = step.get("environment", get_worker(step["worker"]).environment)
        python = local["environments"][group]["python"]
        if not Path(python).is_file() or not os.access(python, os.X_OK):
            issues.append(f"environments.{group}.python: not executable {python}")
            continue
        try:
            result = subprocess.run([
                python, "-m", "vision_encoder_eval.workers.diagnostics",
                "--worker", step["worker"], "--resource-root", local.get("resource_root", str(repository_root()))
            ], capture_output=True, text=True, timeout=30, check=False)
            if result.returncode:
                issues.append(f"{step['worker']}: {result.stderr.strip()}")
            else:
                diagnostics = json.loads(result.stdout)
                issues.extend(f"{step['worker']}: {name}: {reason}"
                              for name, reason in diagnostics["missing"].items())
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            issues.append(f"{step['worker']}: {exc}")
    if execution.get("method"):
        from .core.experiment import method_python
        python = method_python(resolved)
        modules = {"numpy"}
        method = execution["method"]
        modules.update({"ravel": {"torch", "sklearn"}, "rsa": {"torch"},
                        "cca": {"torch"}, "gw": {"scipy"},
                        "mutualnn": set(), "knn": {"faiss"}}[method])
        try:
            probe = subprocess.run([python, "-c",
                "import importlib.util,json,sys; print(json.dumps([n for n in sys.argv[1:] if importlib.util.find_spec(n) is None]))",
                *sorted(modules)], capture_output=True, text=True, check=False, timeout=30)
            if probe.returncode:
                issues.append(f"method interpreter: {probe.stderr.strip()}")
            else:
                issues.extend(f"method dependency: {name} is not installed" for name in json.loads(probe.stdout))
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            issues.append(f"method interpreter: {exc}")
    return [{"name": experiment["experiment"]["name"], "issues": list(dict.fromkeys(issues))}]


def print_checks(records):
    for record in records:
        print(f"{'BLOCKED' if record['issues'] else 'CHECKED'} {record['name']}")
        for issue in record["issues"]:
            print(f"  - {issue}")
    print("Checks cover paths and critical dependencies. Execution also validates input content and model loading.")
    return not any(record["issues"] for record in records)


def generated_suite(output, identity, generate):
    """Reuse only recipes with unchanged generation inputs and config content."""
    receipt_path = output / "generation_receipt.json"
    if output.exists():
        if not receipt_path.is_file():
            raise ConfigError(f"unrecognized generated directory; choose --generated-dir: {output}")
        receipt = read_json(receipt_path)
        if receipt.get("identity") != identity:
            raise ConfigError("generation inputs changed; choose a new --generated-dir")
        expected = receipt.get("files", {})
        actual = {p.name: sha256_file(p) for p in output.glob("*.json") if p != receipt_path}
        if not expected or actual != expected or "suite.json" not in expected:
            raise ConfigError("generated configs changed; choose a new --generated-dir")
    else:
        generate(output)
        files = {p.name: sha256_file(p) for p in output.glob("*.json")}
        write_json_atomic(receipt_path, {"identity": identity, "files": files})
    return output / "suite.json"


def select_config(args, root, local):
    target = args.target
    if target == "ravel" and not args.manifest:
        from .config.ravel import ravel_inputs, generate_ravel_suite
        import torch

        sources, hashes = ravel_inputs(local, args.encoder, args.text_encoder,
            patches=args.patches, text=args.text, sample_ids=args.sample_ids)
        device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        identity = {"kind": "ravel", "sources": [{key: value for key, value in source.items()
                    if key != "sample_ids"} for source in sources], "hashes": hashes,
                    "k": args.k, "device": device,
                    "generator_sha256": sha256_file(Path(__file__).parent / "config/ravel.py")}
        output = Path(args.generated_dir or Path(local["output_root"]) / "recipes" /
                      ("ravel-" + sha256_json(identity)[:12])).resolve()
        return generated_suite(output, identity, lambda dest: generate_ravel_suite(
            dest, sources, k=args.k, device=device, full_panel=args.encoder == "all"))
    if target in {"knn", "knn70"}:
        from .config.knn_panel import generate_knn_panel
        exports = Path(local["paths"].get("knn_exports", "__missing_knn_exports__"))
        for name in ("export_manifest.json", "labels.npy", "source_indices.npy"):
            if not (exports / name).is_file():
                raise ConfigError(f"paths.knn_exports: missing {exports / name}")
        identity = {"kind": "knn70", "exports": str(exports), "generator_sha256": sha256_file(Path(__file__).parent / "config/knn_panel.py"),
                    "inputs": {name: sha256_file(exports / name) for name in
                               ("export_manifest.json", "labels.npy", "source_indices.npy")}}
        output = Path(args.generated_dir or Path(local["output_root"]) / "recipes" / ("knn70-" + sha256_json(identity)[:12])).resolve()
        return generated_suite(output, identity, lambda dest: generate_knn_panel(args.local, dest))
    if args.manifest:
        if target not in PAIRED | {"paired70"}:
            raise ConfigError("--manifest applies only to ravel/rsa/cca/gw/mutualnn/paired70")
        from .config.feature_panel import generate_feature_panel
        manifest = Path(args.manifest).resolve()
        document = dict(_load_document(manifest))
        if target in PAIRED:
            if target not in document.get("methods", {}):
                raise ConfigError(f"manifest must explicitly configure methods.{target}")
            document["methods"] = {target: document["methods"][target]}
        # Preserve relative paths when filtering a reviewed panel manifest.
        def absolute_features(item):
            if isinstance(item, dict):
                result = {}
                for key, value in item.items():
                    if key in {"features", "manifest"} and isinstance(value, str):
                        path = Path(value).expanduser()
                        result[key] = str((manifest.parent / path).resolve())
                    else:
                        result[key] = absolute_features(value)
                return result
            return item
        document = absolute_features(document)
        identity = {"kind": target, "manifest": str(manifest), "sha256": sha256_file(manifest),
                    "generator_sha256": sha256_file(Path(__file__).parent / "config/feature_panel.py")}
        output = Path(args.generated_dir or Path(local["output_root"]) / "recipes" / (target + "-" + sha256_json(identity)[:12])).resolve()
        def generate(dest):
            selected = dest.parent / (dest.name + ".manifest.json")
            write_json_atomic(selected, document)
            generate_feature_panel(selected, dest)
        return generated_suite(output, identity, generate)
    if target in (PAIRED - {"ravel"}) | {"paired70"}:
        raise ConfigError(f"{target} requires --manifest with reviewed caches, ordered row manifests and method parameters")
    if target in RECIPES:
        return root / "configs/experiments" / (RECIPES[target][0] + ".yaml")
    return Path(target).resolve()


def main(argv=None):
    root = repository_root()
    parser = argparse.ArgumentParser(description="Run a named reproduction recipe or an experiment config.")
    parser.add_argument("target", nargs="?", help="recipe name or config path")
    parser.add_argument("--local", default=str(root / "configs/local.yaml"))
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--init-local", action="store_true")
    parser.add_argument("--data-root", default="/cache/vision_encoder_eval_data")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check", action="store_true", help="check paths/dependencies; do not compute")
    modes.add_argument("--dry-run", action="store_true", help="show plan; do not compute")
    parser.add_argument("--manifest", help="reviewed canonical70 paired-feature manifest")
    parser.add_argument("--encoder", default="clip_openai__l14", help="RAVEL visual encoder ID, or all")
    parser.add_argument("--text-encoder", default="qwen25", choices=["qwen25", "qwen3", "smollm2", "all"])
    parser.add_argument("--patches", help="RAVEL [N,T,D] patch array (.npy)")
    parser.add_argument("--text", help="RAVEL [N,D] text array (.npy)")
    parser.add_argument("--sample-ids", help="JSON list of shared sample IDs in exact feature-row order")
    parser.add_argument("--k", type=int, default=100, help="RAVEL neighborhood size")
    parser.add_argument("--device", help="RAVEL device; default CUDA when available, otherwise CPU")
    parser.add_argument("--use-existing-visual-cache", action="store_true",
                        help="RAVEL paper attempt using existing patches; record unverified visual layers")
    parser.add_argument("--generated-dir")
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    args = parser.parse_args(argv)
    try:
        if args.list or not (args.target or args.init_local):
            for name, (_, description) in RECIPES.items():
                print(f"{name:30} {description}")
            print(f"{'knn / knn70':28} All 70 cached encoders, six nested shot counts; FAISS CPU")
            print(f"{'paired70':28} Reviewed paired-feature panel; requires --manifest")
            print(f"{'rsa / cca / gw / mutualnn':28} One method from a reviewed panel; requires --manifest")
            return 0
        if args.init_local:
            initialize_local(args.local, root, args.data_root)
            if not args.target:
                return 0
        if not Path(args.local).is_file():
            raise ConfigError(f"local config is missing: {args.local}; run scripts/reproduce.sh --init-local")
        local = load_local(args.local)
        if args.target == "ravel_paper":
            from .ravel_paper import run
            if args.manifest or args.patches or args.text or args.sample_ids or args.overrides or args.generated_dir:
                raise ConfigError("ravel_paper uses the audited full panel configured in --local")
            result = run(local, device=args.device, k=args.k, check_only=args.check or args.dry_run,
                         use_existing_visual_cache=args.use_existing_visual_cache)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 2 if result["status"] == "blocked" else 0
        if args.use_existing_visual_cache:
            raise ConfigError("--use-existing-visual-cache applies only to ravel_paper")
        config = select_config(args, root, local)
        resolved = resolve_config(local_path=args.local, experiment_path=config, overrides=args.overrides)
        if args.check:
            return 0 if print_checks(check_config(resolved)) else 2
        if not args.dry_run and not print_checks(check_config(resolved)):
            return 2
        result = run_experiment(resolved, dry_run=args.dry_run)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except (ConfigError, KeyError, ValueError, OSError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
