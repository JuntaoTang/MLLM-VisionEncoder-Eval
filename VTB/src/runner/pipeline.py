import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.evaluation.run_eval import run_evaluation
from src.runner.llava_train import run_training_stage
from src.utils.config import (
    CONTINUOUS_CONFIGS,
    VTB_ROOT,
    allocate_stage_log_dir,
    load_run_context,
    materialize_single_model_config,
    normalize_model_list,
    resolve_mode_runtime,
)
from src.utils.mllm_registry import register_trained_model


def _is_discrete_config(config_path: str) -> bool:
    norm = os.path.abspath(config_path).replace("\\", "/")
    return "/configs/discrete/" in norm or norm.endswith("/discrete/runtime.yaml")


def _resolve_log_dir(ctx) -> str:
    log_dir = ctx.runtime.get("log_dir", "logs/continuous")
    if not os.path.isabs(log_dir):
        log_dir = os.path.join(VTB_ROOT, log_dir)
    return log_dir


def _parse_gpu_list(raw) -> list[str]:
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def _resolve_overlap_settings(runtime_cfg: dict) -> dict | None:
    """Return overlap settings when test || next-model pretrain share the same GPU pool."""
    pipe = runtime_cfg.get("pipeline") or {}
    if not pipe.get("overlap_test_pretrain"):
        return None

    runtime = runtime_cfg.get("runtime") or {}
    eval_cfg = runtime_cfg.get("eval") or {}
    gpus = _parse_gpu_list(pipe.get("overlap_gpus") or "")
    if not gpus:
        gpus = _parse_gpu_list(eval_cfg.get("cuda_visible_devices") or "")
    if not gpus:
        gpus = _parse_gpu_list(runtime.get("cuda_visible_devices", "0,1,2,3,4,5,6,7"))
    if not gpus:
        print(
            "[warn] overlap_test_pretrain enabled but no GPUs resolved; "
            "falling back to sequential pipeline.",
            file=sys.stderr,
        )
        return None

    base_port = int(runtime.get("master_port", 29501))
    return {
        "gpus": gpus,
        "overlap_master_port": base_port + 100,
    }


def _run_eval_in_subprocess(
    config_path: str,
    log_path: str | None,
    eval_model: str | None,
    physical_gpus: list[str],
) -> int:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(physical_gpus)
    code = (
        "import sys\n"
        "from src.evaluation.run_eval import run_evaluation\n"
        f"sys.exit(run_evaluation({config_path!r}, log_path={log_path!r}, eval_model={eval_model!r}))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        cwd=VTB_ROOT,
    )
    return int(proc.returncode)


def _run_pretrain_in_subprocess(
    config_path: str,
    train_config_path: str | None,
    log_path: str,
    physical_gpus: list[str],
    master_port: int,
) -> int:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(physical_gpus)
    code = (
        "import sys\n"
        "from src.utils.config import load_run_context\n"
        "from src.runner.llava_train import run_training_stage\n"
        "ctx = load_run_context("
        f"config_path={config_path!r}, mode='continuous', "
        f"train_config_path={train_config_path!r})\n"
        "sys.exit(run_training_stage("
        "ctx, 'pretrain', log_path="
        f"{log_path!r}, num_gpus={len(physical_gpus)}, master_port={master_port}))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        cwd=VTB_ROOT,
    )
    return int(proc.returncode)


def _run_one_continuous(
    config_path: str,
    train_config_path: str | None,
    stages: list[str],
    eval_model: str | None,
    *,
    skip_stages: set[str] | None = None,
) -> int:
    kwargs = {"config_path": config_path, "eval_model": eval_model, "mode": "continuous"}
    if train_config_path:
        kwargs["train_config_path"] = train_config_path
    ctx = load_run_context(**kwargs)
    skip = skip_stages or set()

    print(f"[continuous] {ctx.run_slug}  stages={', '.join(stages)}")

    for stage in stages:
        if stage in skip:
            print(f"[{stage}] skip (already completed via overlap)")
            continue

        run_log_dir = allocate_stage_log_dir(_resolve_log_dir(ctx), ctx.run_slug, stage)
        print(f"[{stage}] log: {run_log_dir}")

        if stage in ("pretrain", "finetune"):
            log_path = os.path.join(run_log_dir, f"{stage}.log")
            code = run_training_stage(ctx, stage, log_path=log_path)
            if code != 0:
                return code
            print(f"registered: {register_trained_model(ctx, stage, log_dir=run_log_dir)}")
        elif stage in ("test", "eval"):
            eval_log = os.path.join(run_log_dir, "eval.log")
            code = run_evaluation(config_path, log_path=eval_log, eval_model=eval_model)
            if code != 0:
                return code
        else:
            print(f"Unknown stage: {stage}", file=sys.stderr)
            return 1

    print(f"Done: {ctx.run_slug}")
    return 0


def _run_overlap_test_pretrain(
    *,
    current_config: str,
    next_config: str,
    train_config_path: str | None,
    eval_model: str | None,
    overlap: dict,
    current_label: str,
    next_label: str,
) -> tuple[int, int]:
    current_ctx = load_run_context(
        config_path=current_config,
        eval_model=eval_model,
        mode="continuous",
        train_config_path=train_config_path,
    )
    next_ctx = load_run_context(
        config_path=next_config,
        eval_model=eval_model,
        mode="continuous",
        train_config_path=train_config_path,
    )

    current_test_log_dir = allocate_stage_log_dir(
        _resolve_log_dir(current_ctx), current_ctx.run_slug, "test"
    )
    next_pretrain_log_dir = allocate_stage_log_dir(
        _resolve_log_dir(next_ctx), next_ctx.run_slug, "pretrain"
    )
    eval_log = os.path.join(current_test_log_dir, "eval.log")
    pretrain_log = os.path.join(next_pretrain_log_dir, "pretrain.log")

    gpus = overlap["gpus"]
    gpu_label = ",".join(gpus)
    print(
        f"[overlap] {current_label} test || {next_label} pretrain "
        f"on the same GPUs [{gpu_label}]"
    )
    print(f"  test log: {current_test_log_dir}")
    print(f"  pretrain log: {next_pretrain_log_dir}")

    def _run_test() -> int:
        return _run_eval_in_subprocess(
            current_config,
            log_path=eval_log,
            eval_model=eval_model,
            physical_gpus=gpus,
        )

    def _run_pretrain() -> int:
        return _run_pretrain_in_subprocess(
            next_config,
            train_config_path=train_config_path,
            log_path=pretrain_log,
            physical_gpus=gpus,
            master_port=int(overlap["overlap_master_port"]),
        )

    test_code = 1
    pretrain_code = 1
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            pool.submit(_run_test): "test",
            pool.submit(_run_pretrain): "pretrain",
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                code = int(future.result())
            except Exception as exc:
                print(f"[overlap] {name} failed: {exc}", file=sys.stderr)
                code = 1
            if name == "test":
                test_code = code
            else:
                pretrain_code = code

    if pretrain_code == 0:
        print(
            f"registered: {register_trained_model(next_ctx, 'pretrain', log_dir=next_pretrain_log_dir)}"
        )
    return test_code, pretrain_code


def _run_continuous_pipeline(
    config_path: str,
    train_config_path: str | None = None,
    stages_override: list[str] | None = None,
    eval_model: str | None = None,
) -> int:
    runtime_cfg, _ = resolve_mode_runtime(config_path, "continuous")
    stages = stages_override or runtime_cfg.get("stages") or []
    if not stages:
        print("No stages configured. Set `stages` in configs/runtime.yaml (continuous).")
        return 1

    models = normalize_model_list(runtime_cfg, "mllm", "mllms")
    if not models and not eval_model:
        print(
            "No mllm configured. Set `mllm:` (string or list) in configs/runtime.yaml.",
            file=sys.stderr,
        )
        return 1
    if not models:
        models = [None]

    overlap = _resolve_overlap_settings(runtime_cfg)
    overlap_enabled = overlap is not None and "test" in stages and "pretrain" in stages
    if overlap_enabled:
        print(
            "[continuous] overlap: test || next pretrain on shared GPUs "
            f"[{','.join(overlap['gpus'])}]"
        )

    print(f"[continuous] queue={len(models)} model(s)  stages={', '.join(stages)}")
    from src.utils.finish_tracker import is_finished

    skip_next_pretrain = False
    for i, mllm in enumerate(models, 1):
        label = mllm or eval_model or "(eval)"
        if mllm and is_finished("continuous", mllm) and set(stages) >= {"pretrain", "finetune", "test"}:
            print(f"\n===== [{i}/{len(models)}] skip {label} (in results/finish.json) =====")
            skip_next_pretrain = False
            continue
        print(f"\n===== [{i}/{len(models)}] {label} =====")
        try:
            from src.evaluation.judge_manager import release_eval_gpu_resources

            release_eval_gpu_resources(runtime_cfg.get("eval"))
        except Exception as exc:
            print(f"[warn] release_eval_gpu_resources failed: {exc}", file=sys.stderr)

        one_config = (
            materialize_single_model_config(config_path, "continuous", mllm, "mllm")
            if mllm
            else config_path
        )

        skip_stages: set[str] = set()
        if skip_next_pretrain and "pretrain" in stages:
            skip_stages.add("pretrain")
            skip_next_pretrain = False

        remaining_stages = [s for s in stages if s not in skip_stages]
        if overlap_enabled and i < len(models) and "test" in remaining_stages:
            pre_overlap = [s for s in remaining_stages if s != "test"]
            if pre_overlap:
                code = _run_one_continuous(
                    config_path=one_config,
                    train_config_path=train_config_path,
                    stages=pre_overlap,
                    eval_model=eval_model,
                    skip_stages=skip_stages,
                )
                if code != 0:
                    print(f"Failed on {label} (exit={code})", file=sys.stderr)
                    return code

            next_mllm = models[i]
            next_label = next_mllm or eval_model or "(eval)"
            next_config = materialize_single_model_config(
                config_path, "continuous", next_mllm, "mllm"
            )
            test_code, pretrain_code = _run_overlap_test_pretrain(
                current_config=one_config,
                next_config=next_config,
                train_config_path=train_config_path,
                eval_model=eval_model,
                overlap=overlap,
                current_label=label,
                next_label=next_label,
            )
            if test_code != 0:
                print(f"Failed on {label} test (exit={test_code})", file=sys.stderr)
                return test_code
            if pretrain_code != 0:
                print(f"Failed on {next_label} pretrain (exit={pretrain_code})", file=sys.stderr)
                return pretrain_code
            skip_next_pretrain = True
            continue

        code = _run_one_continuous(
            config_path=one_config,
            train_config_path=train_config_path,
            stages=remaining_stages,
            eval_model=eval_model,
            skip_stages=skip_stages,
        )
        if code != 0:
            print(f"Failed on {label} (exit={code})", file=sys.stderr)
            return code

    print("\nAll continuous jobs done.")
    return 0


def run_pipeline(
    config_path: str,
    train_config_path: str | None = None,
    stages_override: list[str] | None = None,
    eval_model: str | None = None,
    *,
    mode: str | None = None,
    recipe_override: str | None = None,
    finetune_tag_override: str | None = None,
    data_config_path: str | None = None,
) -> int:
    if not os.path.isabs(config_path):
        config_path = os.path.join(VTB_ROOT, config_path)

    use_discrete = mode == "discrete" or (mode is None and _is_discrete_config(config_path))
    if use_discrete:
        from src.discrete.pipeline import run_pipeline as run_discrete_pipeline

        return run_discrete_pipeline(
            config_path=config_path,
            train_config_path=train_config_path,
            data_config_path=data_config_path,
            stages_override=stages_override,
            recipe_override=recipe_override,
            finetune_tag_override=finetune_tag_override,
            eval_model=eval_model,
        )

    if train_config_path is None:
        runtime_cfg, _ = resolve_mode_runtime(config_path, "continuous")
        train_config_path = runtime_cfg.get("train_config") or os.path.join(
            CONTINUOUS_CONFIGS, "train.yaml"
        )
    return _run_continuous_pipeline(
        config_path=config_path,
        train_config_path=train_config_path,
        stages_override=stages_override,
        eval_model=eval_model,
    )
