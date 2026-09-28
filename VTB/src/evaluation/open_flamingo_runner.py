"""Run OpenFlamingo-style few-shot eval for discrete / continuous MLLMs.

Examples:
  # Discrete
  python -m src.evaluation.open_flamingo_runner configs/runtime.yaml \\
      --mode discrete --recipe qwen3/toklip_l_384 \\
      --datasets MSCOCO_KARPATHY_TEST VQAv2_VAL POPE \\
      --shots 0 4 --max-samples 128 --use-checkpoint pretrain

  # Continuous
  python -m src.evaluation.open_flamingo_runner configs/runtime.yaml \\
      --mode continuous --mllm qwen3/clip_openai__l14_mlp2x \\
      --datasets MSCOCO_KARPATHY_TEST VQAv2_VAL POPE \\
      --shots 0 4 --max-samples 128 --use-checkpoint pretrain
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any

VTB_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
THIRD_PARTY = os.path.join(VTB_ROOT, "third_party")
LLAVA_ROOT = os.path.join(THIRD_PARTY, "LLaVA-NeXT")
for path in (VTB_ROOT, THIRD_PARTY):
    if path not in sys.path:
        sys.path.insert(0, path)

from src.utils.config import install_cuda_stub_env, install_offline_hf_env  # noqa: E402

install_cuda_stub_env()
install_offline_hf_env()

from open_flamingo_eval.evaluate import DEFAULT_DATASET_TASK, evaluate_dataset  # noqa: E402

from src.evaluation.checkpoint import pick_eval_checkpoint_dir  # noqa: E402


def _results_dir(ctx, tag: str = "open_flamingo") -> str:
    root = getattr(ctx, "results_dir", None) or os.path.join(
        VTB_ROOT, "results", "open_flamingo"
    )
    out = os.path.join(root, tag)
    os.makedirs(out, exist_ok=True)
    return out


def _save_result(out_dir: str, result: dict[str, Any]) -> str:
    name = f"{result['dataset']}_{result['num_shots']}shot"
    path = os.path.join(out_dir, f"{name}.json")
    slim = {k: v for k, v in result.items() if k != "predictions"}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(slim, f, indent=2, ensure_ascii=False)
    pred_path = os.path.join(out_dir, f"{name}_predictions.json")
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(result.get("predictions", []), f, indent=2, ensure_ascii=False)
    return path


def _load_discrete_model(ctx, checkpoint_dir: str, of_cfg: dict[str, Any]):
    from src.discrete.eval_utils import build_vlmeval_model_entry
    from src.evaluation.open_flamingo_adapter import load_discrete_of_model

    entry = build_vlmeval_model_entry(ctx, checkpoint_dir)
    prompt_style = str(
        os.environ.get("VTB_OF_PROMPT_STYLE") or of_cfg.get("prompt_style", "chatml")
    )
    model = load_discrete_of_model(
        model_path=entry["model_path"],
        llm_path=entry["llm_path"],
        hidden_size=int(entry.get("hidden_size", 2048)),
        vis_mode=entry.get("vis_mode"),
        projector_cfg=entry.get("projector"),
        prompt_style=prompt_style,
        unitok_checkpoint_path=entry.get("unitok_checkpoint_path"),
        num_query=entry.get("num_query"),
        vilau_checkpoint_path=entry.get("vilau_checkpoint_path"),
        tokenizer_checkpoint_path=entry.get("tokenizer_checkpoint_path"),
        toklip_model_config=entry.get("toklip_model_config"),
        toklip_vqgan_checkpoint_path=entry.get("toklip_vqgan_checkpoint_path"),
        image_size=entry.get("image_size"),
        post_quant_embed_dim=entry.get("post_quant_embed_dim"),
        embed_dim=entry.get("embed_dim"),
        num_tokens=entry.get("num_tokens"),
        rq_depth=entry.get("rq_depth"),
    )
    return model, prompt_style


def _load_continuous_model(ctx, checkpoint_dir: str, of_cfg: dict[str, Any]):
    from src.evaluation.open_flamingo_continuous import load_continuous_of_model

    if LLAVA_ROOT not in sys.path:
        sys.path.insert(0, LLAVA_ROOT)

    ve = ctx.vision_encoder or {}
    ve_type = ve.get("type", "open_clip_hub")
    vision_tower = ve.get("vision_tower")
    if not vision_tower and ve_type in ("dinov3", "raev2", "ijepa", "pe", "eupe", "pixio"):
        vision_tower = f"vtb_ssl:{ve_type}"
    elif not vision_tower and ve_type == "open_clip_hub" and ve.get("model_name"):
        vision_tower = f"open_clip_hub:{ve['model_name']}"
    elif not vision_tower and ve_type == "hf_clip":
        vision_tower = ve.get("model_name_or_path") or ve.get("weights_path")
    elif not vision_tower and ve_type in ("hf", "hf_vision"):
        path = ve.get("model_name_or_path") or ve.get("vision_tower") or ve.get("weights_path")
        vision_tower = f"hf:{path}" if path else None

    weights = ve.get("weights_path")
    if weights and ve_type != "hf_clip":
        os.environ["VTB_VISION_WEIGHTS"] = str(weights)
    force_image_size = ve.get("force_image_size")
    if force_image_size is not None:
        os.environ["VTB_FORCE_IMAGE_SIZE"] = str(int(force_image_size))
    force_qg = ve.get("force_quick_gelu")
    model_name = str(ve.get("model_name") or ve.get("vision_tower") or "")
    if force_qg is None and "siglip" in model_name.lower():
        force_qg = False
    if force_qg is True:
        os.environ["VTB_FORCE_QUICK_GELU"] = "1"
    elif force_qg is False:
        os.environ["VTB_FORCE_QUICK_GELU"] = "0"
    clip_processor = ve.get("processor_path")
    if clip_processor and os.path.isdir(str(clip_processor)):
        os.environ["VTB_CLIP_IMAGE_PROCESSOR"] = str(clip_processor)
    if ve_type in ("dinov3", "raev2", "ijepa"):
        os.environ["VTB_ROOT"] = VTB_ROOT
        image_size = ve.get("image_size") or force_image_size
        if image_size is not None:
            os.environ["VTB_SSL_IMAGE_SIZE"] = str(int(image_size))
        layers = ve.get("layers")
        if layers is not None:
            if isinstance(layers, (list, tuple)):
                os.environ["VTB_SSL_LAYERS"] = ".".join(str(int(x)) for x in layers)
            else:
                os.environ["VTB_SSL_LAYERS"] = str(layers)
        dinov3_repo = ve.get("dinov3_repo") or os.path.join(VTB_ROOT, "third_party", "dinov3")
        if os.path.isdir(dinov3_repo):
            os.environ["VTB_DINOV3_REPO_DIR"] = dinov3_repo
            os.environ["DINOV3_REPO_DIR"] = dinov3_repo

    prompt_style = str(
        os.environ.get("VTB_OF_PROMPT_STYLE") or of_cfg.get("prompt_style", "chatml")
    )
    attn_impl = str((ctx.eval or {}).get("attn_implementation", "sdpa"))
    model = load_continuous_of_model(
        model_path=checkpoint_dir,
        llm_path=ctx.llm["model_name_or_path"],
        vision_weights=weights if ve_type != "hf_clip" else None,
        vision_tower=vision_tower,
        processor_path=ve.get("processor_path"),
        attn_implementation=attn_impl,
        prompt_style=prompt_style,
    )
    return model, prompt_style


def run_open_flamingo_eval(
    config_path: str | None = None,
    *,
    mode: str = "discrete",
    recipe: str | None = None,
    mllm: str | None = None,
    datasets: list[str] | None = None,
    shots: list[int] | None = None,
    max_samples: int | None = None,
    seed: int = 42,
    use_checkpoint: str | None = None,
    cuda_device: str | None = None,
    output_dir: str | None = None,
    num_beams: int | None = None,
    task: str | None = None,
    num_negatives: int | None = None,
) -> dict[str, Any]:
    if cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
    os.environ.setdefault("VTB_EVAL_DEVICE_INDEX", "0")

    from src.utils.config import DEFAULT_RUNTIME_CONFIG as CONT_DEFAULT

    config_path = config_path or CONT_DEFAULT
    mode = mode.lower().strip()
    if mode not in ("discrete", "continuous"):
        raise ValueError(f"Unsupported mode={mode}")

    if mode == "discrete":
        from src.discrete.config import DEFAULT_RUNTIME_CONFIG, load_run_context

        config_path = config_path or DEFAULT_RUNTIME_CONFIG
        ctx = load_run_context(config_path, recipe_override=recipe or mllm)
    else:
        from src.utils.config import load_run_context

        ctx = load_run_context(
            config_path,
            mode="continuous",
            mllm_override=mllm or recipe,
        )

    eval_cfg = dict(ctx.eval or {})
    of_cfg = dict(eval_cfg.get("open_flamingo") or {})

    task_override = task or of_cfg.get("task")
    datasets = datasets or of_cfg.get("datasets") or [
        "MSCOCO_KARPATHY_TEST",
        "VQAv2_VAL",
        "POPE",
    ]
    if task_override == "caption_rank":
        datasets = datasets or ["MSCOCO_KARPATHY_TEST"]
        shots = [0]
    else:
        shots = shots if shots is not None else list(of_cfg.get("shots") or [0, 4])
    max_samples = (
        max_samples
        if max_samples is not None
        else int(of_cfg.get("max_samples", eval_cfg.get("max_samples", 1000)))
    )
    seed = int(of_cfg.get("sample_seed", eval_cfg.get("sample_seed", seed)))
    use_ckpt = use_checkpoint or of_cfg.get("use_checkpoint") or "pretrain"
    ctx.eval["use_checkpoint"] = use_ckpt
    beams = (
        int(num_beams)
        if num_beams is not None
        else int(of_cfg.get("num_beams", 3))
    )
    n_neg = (
        int(num_negatives)
        if num_negatives is not None
        else int(of_cfg.get("num_negatives", 9))
    )

    lmudata_dir = (
        of_cfg.get("lmudata_dir")
        or eval_cfg.get("lmudata_dir")
        or getattr(ctx, "eval_data_dir", None)
        or os.environ.get("LMUData")
        or "/cache/data/.lmudata"
    )
    os.environ["LMUData"] = lmudata_dir

    resolved = pick_eval_checkpoint_dir(ctx, ctx.llm["model_name_or_path"])
    checkpoint_dir = resolved.path
    print(f"[open_flamingo] mode={mode}", flush=True)
    print(f"[open_flamingo] checkpoint ({use_ckpt}): {checkpoint_dir}", flush=True)
    print(f"[open_flamingo] LMUData: {lmudata_dir}", flush=True)
    print(
        f"[open_flamingo] task={task_override or 'auto'} datasets={datasets} "
        f"shots={shots} max_samples={max_samples} beams={beams} "
        f"num_negatives={n_neg}",
        flush=True,
    )

    if mode == "discrete":
        model, prompt_style = _load_discrete_model(ctx, checkpoint_dir, of_cfg)
    else:
        model, prompt_style = _load_continuous_model(ctx, checkpoint_dir, of_cfg)
    print(f"[open_flamingo] prompt_style={prompt_style}", flush=True)

    stamp = datetime.now().strftime("%m_%d_%H%M%S")
    out_dir = output_dir or os.path.join(_results_dir(ctx), stamp)
    os.makedirs(out_dir, exist_ok=True)

    summary: dict[str, Any] = {
        "backend": "open_flamingo",
        "mode": mode,
        "task": task_override or "auto",
        "checkpoint": checkpoint_dir,
        "use_checkpoint": use_ckpt,
        "lmudata_dir": lmudata_dir,
        "max_samples": max_samples,
        "sample_seed": seed,
        "num_beams": beams,
        "num_negatives": n_neg,
        "shots": shots,
        "datasets": {},
    }

    for dataset_name in datasets:
        task_name = task_override or DEFAULT_DATASET_TASK.get(dataset_name)
        summary["datasets"][dataset_name] = {}
        shot_list = [0] if task_name == "caption_rank" else shots
        for shot in shot_list:
            kwargs: dict[str, Any] = dict(
                max_samples=max_samples,
                seed=seed,
                num_shots=int(shot),
                zero_shot_text_demos=int(of_cfg.get("zero_shot_text_demos", 2)),
            )
            if task_name == "captioning":
                kwargs["max_new_tokens"] = int(of_cfg.get("caption_max_new_tokens", 20))
                kwargs["num_beams"] = beams
            elif task_name == "vqa":
                kwargs["max_new_tokens"] = int(of_cfg.get("vqa_max_new_tokens", 5))
                kwargs["num_beams"] = beams
            elif task_name == "caption_rank":
                kwargs["num_negatives"] = n_neg

            result = evaluate_dataset(
                model,
                dataset_name,
                lmudata_dir=lmudata_dir,
                task=task_name,
                **kwargs,
            )
            path = _save_result(out_dir, result)
            extra = ""
            if task_name == "caption_rank":
                extra = (
                    f" R@5={result.get('recall@5', 0):.4f} "
                    f"MRR={result.get('mrr', 0):.4f} "
                    f"mean_rank={result.get('mean_rank', 0):.2f}"
                )
            print(
                f"[open_flamingo] {dataset_name} {shot}-shot "
                f"{result['metric']}={result['score']:.4f}{extra} -> {path}",
                flush=True,
            )
            entry = {
                "metric": result["metric"],
                "score": result["score"],
                "num_samples": result["num_samples"],
                "path": os.path.relpath(path, out_dir),
            }
            if task_name == "caption_rank":
                for k in ("recall@1", "recall@5", "mrr", "mean_rank", "num_negatives"):
                    if k in result:
                        entry[k] = result[k]
            summary["datasets"][dataset_name][f"{shot}shot"] = entry

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[open_flamingo] summary -> {summary_path}", flush=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    from src.utils.config import DEFAULT_RUNTIME_CONFIG

    parser = argparse.ArgumentParser(description="OpenFlamingo-style MLLM eval")
    parser.add_argument(
        "config",
        nargs="?",
        default=DEFAULT_RUNTIME_CONFIG,
        help="Runtime yaml (default: configs/runtime.yaml)",
    )
    parser.add_argument(
        "--mode",
        choices=["discrete", "continuous"],
        default="discrete",
        help="Model family (default: discrete)",
    )
    parser.add_argument(
        "--recipe",
        default=None,
        help="Discrete recipe name, e.g. qwen3/toklip_l_384",
    )
    parser.add_argument(
        "--mllm",
        default=None,
        help="Continuous mllm recipe, e.g. qwen3/clip_openai__l14_mlp2x",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="LMUData dataset names (default: COCO / VQAv2 / POPE)",
    )
    parser.add_argument(
        "--task",
        choices=["captioning", "vqa", "classification", "caption_rank"],
        default=None,
        help="Force task (use caption_rank for log-prob caption ranking)",
    )
    parser.add_argument("--shots", nargs="+", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-beams", type=int, default=None)
    parser.add_argument(
        "--num-negatives",
        type=int,
        default=None,
        help="Negatives per image for caption_rank (default: 9 → 10-way)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--use-checkpoint",
        choices=["pretrain", "finetune"],
        default=None,
        help="Which stage checkpoint to evaluate (default: pretrain)",
    )
    parser.add_argument(
        "--prompt-style",
        choices=["chatml", "flamingo"],
        default=None,
        help="chatml (default, matches VTB training) or raw flamingo continuation",
    )
    parser.add_argument("--cuda-device", default=None, help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)

    if args.prompt_style:
        os.environ["VTB_OF_PROMPT_STYLE"] = args.prompt_style

    run_open_flamingo_eval(
        args.config,
        mode=args.mode,
        recipe=args.recipe,
        mllm=args.mllm,
        datasets=args.datasets,
        shots=args.shots,
        max_samples=args.max_samples,
        seed=args.seed,
        use_checkpoint=args.use_checkpoint,
        cuda_device=args.cuda_device,
        output_dir=args.output_dir,
        num_beams=args.num_beams,
        task=args.task,
        num_negatives=args.num_negatives,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
