from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .config import ConfigError, resolve_config
from .core.artifacts import read_json, validate_result_payload, write_json_atomic
from .core.registry import METHODS
from .data import DatasetManifest, ManifestError
from .core.artifacts import ArtifactError
from .core.experiment import run_experiment, preflight as check_preflight


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vision-encoder-eval")
    subparsers = parser.add_subparsers(dest="command", required=True)

    config = subparsers.add_parser("config", help="resolve and validate configuration")
    config_subparsers = config.add_subparsers(dest="config_command", required=True)
    resolve = config_subparsers.add_parser("resolve")
    resolve.add_argument("--local", required=True)
    resolve.add_argument("--experiment", required=True)
    resolve.add_argument("--set", action="append", default=[], dest="overrides")
    resolve.add_argument("--output")
    panel = config_subparsers.add_parser('knn-panel',help='generate the canonical cached-feature 70-model CPU suite')
    panel.add_argument('--local',required=True)
    panel.add_argument('--output',required=True)
    feature_panel = config_subparsers.add_parser('feature-panel',help='generate a canonical70 paired-feature suite and strict table')
    feature_panel.add_argument('--manifest',required=True)
    feature_panel.add_argument('--output',required=True)

    data = subparsers.add_parser("data", help="validate dataset manifests")
    data_subparsers = data.add_subparsers(dest="data_command", required=True)
    validate = data_subparsers.add_parser("validate")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--data-root", required=True)
    feature = data_subparsers.add_parser('feature-manifest',help='pin exported features to ordered sample IDs')
    feature.add_argument('--features',required=True)
    feature.add_argument('--sample-ids',required=True,help='JSON list in the exact exported row order')
    feature.add_argument('--output',required=True)
    ground_truth = data_subparsers.add_parser('ground-truth', help='validate the bundled 210 paper labels')
    ground_truth.add_argument('--path', help='override the bundled ground truth')
    ground_truth.add_argument('--import-from', dest='import_from', help='prepare paper labels from an archived table')
    ground_truth.add_argument('--output', help='destination required with --import-from')

    benchmark = subparsers.add_parser("benchmark", help="evaluate custom metrics against the paper ground truth")
    benchmark_subparsers = benchmark.add_subparsers(dest="benchmark_command", required=True)
    export = benchmark_subparsers.add_parser("export-ground-truth", help="export all 210 labels as CSV")
    export.add_argument("--output", required=True)
    template = benchmark_subparsers.add_parser("template", help="write canonical prediction IDs with empty scores")
    template.add_argument("--output", required=True)
    template.add_argument("--llm", action="append", dest="llms", choices=["qwen3", "qwen25", "smollm2"])
    for name in ("evaluate", "run"):
        command = benchmark_subparsers.add_parser(name)
        command.add_argument("--output", required=True, help="report directory")
        command.add_argument("--llm", action="append", dest="llms", choices=["qwen3", "qwen25", "smollm2"])
        command.add_argument("--direction", choices=["higher", "lower"], default="higher")
        if name == "evaluate":
            command.add_argument("--predictions", required=True, help="CSV with encoder_id,llm,score")
            command.add_argument("--score-column", default="score")
            command.add_argument("--metric", default="custom", help="method name recorded in the report")
        else:
            command.add_argument("--metric", required=True, help="importable module:function predictor")

    methods = subparsers.add_parser("methods", help="inspect registered methods")
    methods.add_argument("action", choices=["list"])
    workers = subparsers.add_parser('workers', help='inspect isolated worker entry points')
    workers.add_argument('action', choices=['list'])

    preflight = subparsers.add_parser("preflight", help="check a local setup")
    preflight.add_argument("--local", required=True)
    preflight.add_argument("--experiment", required=True)
    preflight.add_argument("--set", action="append", default=[], dest="overrides")

    experiment = subparsers.add_parser("experiment", help="run a configured experiment")
    experiment_subparsers = experiment.add_subparsers(dest="experiment_command", required=True)
    run = experiment_subparsers.add_parser("run")
    run.add_argument("--local", required=True)
    run.add_argument("--experiment", required=True)
    run.add_argument("--set", action="append", default=[], dest="overrides")
    run.add_argument("--dry-run", action="store_true")

    return parser


def _print_json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _resolve(args: argparse.Namespace):
    return resolve_config(
        local_path=args.local,
        experiment_path=args.experiment,
        overrides=args.overrides,
    )


def _run(args: argparse.Namespace) -> int:
    if args.command == "benchmark":
        from .benchmark import evaluate_file, export_ground_truth, run_metric, write_prediction_template
        if args.benchmark_command == "export-ground-truth":
            count = export_ground_truth(args.output)
            _print_json({"status": "success", "pairs": count, "output": args.output})
        elif args.benchmark_command == "template":
            count = write_prediction_template(args.output, llms=args.llms)
            _print_json({"status": "success", "pairs": count, "output": args.output})
        elif args.benchmark_command == "evaluate":
            _print_json(evaluate_file(args.predictions, args.output, llms=args.llms,
                                     score_column=args.score_column, direction=args.direction, metric=args.metric))
        else:
            _print_json(run_metric(args.metric, args.output, llms=args.llms, direction=args.direction))
        return 0

    if args.command=='config' and args.config_command=='feature-panel':
        from .config.feature_panel import generate_feature_panel
        _print_json(generate_feature_panel(args.manifest,args.output))
        return 0
    if args.command=='config' and args.config_command=='knn-panel':
        from .config.knn_panel import generate_knn_panel
        _print_json(generate_knn_panel(args.local,args.output))
        return 0

    if args.command == "config" and args.config_command == "resolve":
        resolved = _resolve(args)
        if args.output:
            resolved.write(args.output)
        else:
            _print_json({**resolved.value, "config_sha256": resolved.sha256})
        return 0

    if args.command == "data" and args.data_command == "validate":
        manifest = DatasetManifest.load(args.manifest)
        manifest.validate_files(data_root=args.data_root)
        _print_json(
            {
                "status": "success",
                "dataset_id": manifest.dataset_id,
                "split": manifest.split,
                "samples": len(manifest.samples),
                "manifest_sha256": manifest.sha256,
            }
        )
        return 0

    if args.command=='data' and args.data_command=='feature-manifest':
        from .data.features import create_feature_manifest
        value = create_feature_manifest(args.features,read_json(args.sample_ids),args.output)
        _print_json({'status':'success','samples':len(value['sample_ids']),'output':args.output})
        return 0

    if args.command == 'data' and args.data_command == 'ground-truth':
        from .data.ground_truth import ground_truth_path, load_ground_truth, prepare_ground_truth
        from .core.hashing import sha256_file
        if args.import_from:
            if not args.output or args.path:
                raise ValueError('--import-from requires --output and cannot be combined with --path')
            value = prepare_ground_truth(args.import_from, args.output)
            path = Path(args.output).resolve()
        else:
            if args.output:
                raise ValueError('--output requires --import-from')
            path = Path(args.path).resolve() if args.path else ground_truth_path()
            value = load_ground_truth(path)
        _print_json({'status': 'success', 'encoders': value['n_encoders'],
                     'pairs': value['n_pairs'], 'llms': value['llms'],
                     'benchmarks': len(value['benchmarks']),
                     'path': str(path), 'sha256': sha256_file(path)})
        return 0

    if args.command == "methods" and args.action == "list":
        _print_json(
            [
                {
                    "name": spec.name,
                    "description": spec.description,
                    "requires": list(spec.requires),
                    'execution_mode': spec.execution_mode,
                }
                for spec in METHODS
            ]
        )
        return 0

    if args.command == 'workers':
        from .workers.registry import WORKERS
        from dataclasses import asdict
        _print_json({name:asdict(spec) for name,spec in sorted(WORKERS.items())})
        return 0

    if args.command == "preflight":
        payload = check_preflight(_resolve(args))
        _print_json(payload)
        return 0 if payload['status'] == 'success' else 2

    if args.command == "experiment" and args.experiment_command == "run":
        _print_json(run_experiment(_resolve(args), dry_run=args.dry_run))
        return 0

    raise AssertionError(f"unhandled command: {args.command}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        status = _run(args)
    except (ConfigError, ManifestError, ArtifactError, KeyError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    raise SystemExit(status)


if __name__ == "__main__":
    main()
