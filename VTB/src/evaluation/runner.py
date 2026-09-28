import json
import logging
import os
import shutil
import sys
import warnings

import pandas as pd
import torch
from PIL import Image
from transformers import GenerationConfig

from src.evaluation import results as results_layout
from src.evaluation.lmu_data import exact_match, load_lmu_tsv, vqa_score
from src.utils.config import (
    PRETRAINED_ROOT,
    apply_offline_hf_env,
    install_offline_hf_env,
    load_config,
    load_run_context,
    resolve_llava_project,
)

# OpenCLIP vision tower uses delay_load and reloads frozen weights from disk;
# checkpoint vision keys are intentionally unused and produce noisy HF logs.
_LOAD_LOGGERS = (
    "transformers.modeling_utils",
    "transformers.generation.utils",
    "transformers.configuration_utils",
)


def _quiet_transformers_logs():
    for name in _LOAD_LOGGERS:
        logging.getLogger(name).setLevel(logging.ERROR)


def _resolve_model_name(checkpoint_dir: str) -> str:
    """Use a LLaVA-compatible name so builder selects the llama code path."""
    config_path = os.path.join(checkpoint_dir, "config.json")
    if os.path.isfile(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        if cfg.get("model_type") in ("llama", "llava_llama") or "LlavaLlama" in str(cfg.get("architectures", [])):
            return "llava_llama_3"
    lower = checkpoint_dir.lower()
    if "smol" in lower or "llama" in lower:
        return "llava_llama_3"
    parent = os.path.basename(os.path.dirname(checkpoint_dir.rstrip("/")))
    return parent or "llava_llama_3"


def _ensure_llava_on_path(llava_project: str) -> None:
    llava_root = os.path.abspath(llava_project)
    if llava_root not in sys.path:
        sys.path.insert(0, llava_root)


def _resolve_eval_checkpoint(checkpoint_dir: str) -> str:
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    markers = ("model.safetensors", "pytorch_model.bin")
    if any(os.path.isfile(os.path.join(checkpoint_dir, name)) for name in markers):
        return checkpoint_dir

    ckpts = []
    for name in os.listdir(checkpoint_dir):
        if name.startswith("checkpoint-"):
            path = os.path.join(checkpoint_dir, name)
            if os.path.isdir(path) and any(os.path.isfile(os.path.join(path, m)) for m in markers):
                try:
                    step = int(name.split("-", 1)[1])
                except (IndexError, ValueError):
                    step = -1
                ckpts.append((step, path))

    if not ckpts:
        raise FileNotFoundError(f"No HF checkpoint found under {checkpoint_dir}")

    ckpts.sort(key=lambda x: x[0])
    return ckpts[-1][1]


def _load_llava_model(
    checkpoint_dir: str,
    llava_project: str,
    attn_implementation: str = "sdpa",
    vision_weights: str | None = None,
    force_image_size: int | None = None,
    vision_encoder: dict | None = None,
):
    _ensure_llava_on_path(llava_project)
    _quiet_transformers_logs()

    install_offline_hf_env()
    if vision_weights:
        os.environ["VTB_VISION_WEIGHTS"] = vision_weights
    else:
        os.environ.setdefault(
            "VTB_VISION_WEIGHTS",
            os.path.join(PRETRAINED_ROOT, "visual_encoder", "vit_l14_metaclip.safetensors"),
        )
    if force_image_size is not None:
        os.environ["VTB_FORCE_IMAGE_SIZE"] = str(int(force_image_size))
    ve = vision_encoder or {}
    ve_type = ve.get("type", "")
    if ve_type in ("dinov3", "raev2", "ijepa"):
        from src.utils.config import VTB_ROOT

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
    if ve_type in ("pe", "eupe", "pixio"):
        from src.utils.config import VTB_ROOT

        os.environ["VTB_ROOT"] = VTB_ROOT
        image_size = ve.get("image_size") or force_image_size
        if image_size is not None:
            os.environ["VTB_SSL_IMAGE_SIZE"] = str(int(image_size))
        pe_repo = ve.get("pe_repo") or os.path.join(VTB_ROOT, "third_party", "perception_models")
        if os.path.isdir(pe_repo):
            os.environ["VTB_PE_REPO_DIR"] = pe_repo
        pe_config = ve.get("pe_config") or ve.get("model_name")
        if pe_config:
            os.environ["VTB_PE_CONFIG"] = str(pe_config)
        eupe_repo = ve.get("eupe_repo") or os.path.join(VTB_ROOT, "third_party", "eupe")
        if os.path.isdir(eupe_repo):
            os.environ["VTB_EUPE_REPO_DIR"] = eupe_repo
        if ve.get("eupe_hub"):
            os.environ["VTB_EUPE_HUB"] = str(ve["eupe_hub"])
        pixio_repo = ve.get("pixio_repo") or os.path.join(VTB_ROOT, "third_party", "pixio")
        if os.path.isdir(pixio_repo):
            os.environ["VTB_PIXIO_REPO_DIR"] = pixio_repo
        if ve.get("pixio_hub"):
            os.environ["VTB_PIXIO_HUB"] = str(ve["pixio_hub"])
    if ve_type in ("hf", "hf_vision") and ve.get("select_feature"):
        os.environ["VTB_HF_SELECT_FEATURE"] = str(ve["select_feature"])

    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import SeparatorStyle, conv_templates
    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init

    disable_torch_init()
    model_path = _resolve_eval_checkpoint(checkpoint_dir)
    model_name = _resolve_model_name(model_path)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*Some weights of.*were not used when initializing.*",
        )
        warnings.filterwarnings(
            "ignore",
            message=".*You are using a model of type.*",
        )
        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path,
            None,
            model_name,
            multimodal=True,
            torch_dtype="bfloat16",
            attn_implementation=attn_implementation,
            overwrite_config={"delay_load": True},
        )

    if image_processor is None:
        raise RuntimeError(
            f"Failed to load vision tower for checkpoint {model_path}. "
            "Ensure the finetune checkpoint is a full LLaVA multimodal model."
        )

    model.eval()
    try:
        ssl_tower = model.get_model().get_vision_tower()
        if getattr(ssl_tower, "ssl_type", None) == "eupe" and hasattr(ssl_tower, "load_eupe_weights_from_llava_ckpt"):
            ssl_tower.load_eupe_weights_from_llava_ckpt(model_path, device="cuda")
    except Exception:
        pass
    return {
        "tokenizer": tokenizer,
        "model": model,
        "image_processor": image_processor,
        "model_path": model_path,
        "DEFAULT_IMAGE_TOKEN": DEFAULT_IMAGE_TOKEN,
        "IMAGE_TOKEN_INDEX": IMAGE_TOKEN_INDEX,
        "conv_templates": conv_templates,
        "SeparatorStyle": SeparatorStyle,
        "process_images": process_images,
        "tokenizer_image_token": tokenizer_image_token,
    }


def _prepare_images(image: Image.Image, bundle: dict) -> torch.Tensor | list[torch.Tensor]:
    tensors = bundle["process_images"]([image], bundle["image_processor"], bundle["model"].config)
    if isinstance(tensors, list):
        return [t.to(dtype=torch.bfloat16, device="cuda") for t in tensors]
    return tensors.to(dtype=torch.bfloat16, device="cuda")


def _build_prompt(question: str, conv_mode: str, tokenizer, bundle: dict) -> tuple[str, object]:
    conv = bundle["conv_templates"][conv_mode].copy()
    conv.tokenizer = tokenizer
    qs = (
        f"{bundle['DEFAULT_IMAGE_TOKEN']}\n"
        f"{question}\n"
        "Answer the question using a single word or phrase."
    )
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt(), conv


def _extract_short_answer(text: str) -> str:
    text = text.strip()
    for sep in ("assistant", "<|eot_id|>", "\n\n"):
        if sep in text:
            text = text.split(sep)[0]
    return text.strip().split("\n")[0].strip()


def _decode_generation(tokenizer, input_ids, output_ids) -> tuple[str, str]:
    # LLaVA generate via inputs_embeds often returns new tokens only — do not slice
    # by input length unless the prompt ids are literally prefixed.
    in_len = int(input_ids.shape[1])
    if output_ids.shape[1] > in_len and torch.equal(output_ids[:, :in_len], input_ids):
        new_ids = output_ids[:, in_len:]
        raw = tokenizer.batch_decode(new_ids, skip_special_tokens=True)[0].strip()
    else:
        raw = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    if raw:
        return _extract_short_answer(raw), raw
    return "", ""


def _generate_answer(image: Image.Image, question: str, bundle: dict, conv_mode: str, max_new_tokens: int) -> tuple[str, str]:
    from llava.mm_utils import KeywordsStoppingCriteria

    tokenizer = bundle["tokenizer"]
    model = bundle["model"]

    prompt, conv = _build_prompt(question, conv_mode, tokenizer, bundle)
    input_ids = (
        bundle["tokenizer_image_token"](
            prompt,
            tokenizer,
            bundle["IMAGE_TOKEN_INDEX"],
            return_tensors="pt",
        )
        .unsqueeze(0)
        .to("cuda")
    )
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)
    images = _prepare_images(image, bundle)

    stop_str = conv.sep if conv.sep_style != bundle["SeparatorStyle"].TWO else conv.sep2
    stopping = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
    stop_token_ids = getattr(conv, "stop_token_ids", None)

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    eos_token_id = stop_token_ids if stop_token_ids else tokenizer.eos_token_id
    gen_config = GenerationConfig.from_model_config(model.config)
    gen_config.do_sample = False
    gen_config.max_new_tokens = max_new_tokens
    gen_config.pad_token_id = pad_token_id
    gen_config.eos_token_id = eos_token_id
    gen_config.temperature = None
    gen_config.top_p = None
    gen_config.top_k = None

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            images=images,
            generation_config=gen_config,
            use_cache=True,
            stopping_criteria=[stopping],
        )

    return _decode_generation(tokenizer, input_ids, output_ids)


def _eval_output_dir(results_dir: str, dataset_name: str) -> str:
    """results/{llm}/{vision_encoder}/{dataset}/"""
    return os.path.join(results_dir, dataset_name)


def _reset_eval_output_dir(output_dir: str) -> None:
    """Remove previous eval artifacts so re-running the same test fully overwrites."""
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)


def _write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _append_prediction(path: str, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_readable_header(path: str, run_config: dict, total: int) -> None:
    lines = [
        "=" * 80,
        "VTB Eval Results",
        "=" * 80,
        f"Experiment : {run_config.get('run_slug', run_config.get('experiment', '?'))}",
        f"Dataset    : {run_config.get('dataset')}",
        f"Checkpoint : {run_config.get('checkpoint')}",
        f"Samples    : {total}",
        "=" * 80,
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _append_readable_prediction(path: str, idx: int, total: int, record: dict) -> None:
    header = f"[{idx:>{len(str(total))}}/{total}]  index={record.get('index', '?')}"
    if record.get("image"):
        header += f"  image={record['image']}"

    lines = ["", "=" * 80, header, "-" * 80, ""]

    question = record.get("question", "").strip()
    if question:
        lines.append("问题：")
        lines.append(f"  {question}")
        lines.append("")

    if record.get("skipped"):
        lines.append(f"状态：跳过 ({record.get('skip_reason', 'unknown')})")
        gt = record.get("ground_truth", "")
        if gt:
            lines.append("")
            lines.append("正确答案：")
            lines.append(f"  {gt}")
    else:
        model_answer = record.get("model_answer", "").strip()
        gt = record.get("ground_truth", "").strip()

        lines.append("模型答案：")
        for line in model_answer.splitlines() or ["(empty)"]:
            lines.append(f"  {line}")
        lines.append("")
        lines.append("正确答案：")
        for line in gt.splitlines() or ["(empty)"]:
            lines.append(f"  {line}")
        lines.append("")

        vqa = record.get("vqa_score", 0.0)
        exact = "✓" if record.get("exact_match") else "✗"
        lines.append(f"评分：VQA {vqa:.2f}  |  精确匹配 {exact}")

    lines.append("")
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))


def evaluate(
    config_path: str,
    dataset_name: str = "VQAv2_VAL",
    max_samples: int | None = None,
    result_dir: str | None = None,
):
    config = load_config(config_path)
    ctx = load_run_context(config_path)
    eval_cfg = ctx.eval or {}

    llava_project = resolve_llava_project(ctx.paths)
    conv_mode = eval_cfg.get("conv_mode", "llava_llama_3")
    if max_samples is None:
        max_samples = eval_cfg.get("max_samples")
    max_new_tokens = int(eval_cfg.get("max_new_tokens", 64))
    log_every = int(eval_cfg.get("log_every", 50))
    attn_implementation = eval_cfg.get("attn_implementation", "sdpa")

    lmudata_dir = config.eval_data_dir
    tsv_path = os.path.join(lmudata_dir, f"{dataset_name}.tsv")
    image_dir = os.path.join(lmudata_dir, "images", dataset_name)

    vision_weights = ctx.vision_encoder.get("weights_path")
    force_image_size = ctx.vision_encoder.get("force_image_size")
    print(f"Loading checkpoint from {config.checkpoint_dir} ...")
    bundle = _load_llava_model(
        config.checkpoint_dir,
        llava_project,
        attn_implementation=attn_implementation,
        vision_weights=vision_weights,
        force_image_size=force_image_size,
        vision_encoder=ctx.vision_encoder,
    )
    print(f"Using checkpoint: {bundle['model_path']}")
    print(
        "Note: vision encoder weights are loaded from local files "
        f"({vision_weights or os.environ.get('VTB_VISION_WEIGHTS')}); "
        "checkpoint vision keys are skipped by design."
    )

    samples = load_lmu_tsv(tsv_path, image_dir, max_samples=max_samples)
    if not samples:
        raise RuntimeError(f"No eval samples loaded from {tsv_path}")

    sample_note = "all" if max_samples is None else str(max_samples)
    print(f"Evaluating {len(samples)} samples from {dataset_name} (limit={sample_note}) ...")

    checkpoint_path = bundle["model_path"]
    dataset_dir = result_dir or results_layout.dataset_dir(config.results_dir, dataset_name)
    _reset_eval_output_dir(dataset_dir)
    print(f"Output: {dataset_dir}/predictions.xlsx")

    xlsx_path = results_layout.predictions_path(config.results_dir, dataset_name)
    rows_for_xlsx = []
    vqa_scores = []
    exact_correct = 0
    skipped = 0

    for idx, sample in enumerate(samples, start=1):
        if not os.path.isfile(sample["image"]):
            skipped += 1
            gt_skip = sample["answer"]
            rows_for_xlsx.append(
                {
                    "index": sample["index"],
                    "question": sample["question"],
                    "answer": json.dumps(sample.get("all_answers") or ([gt_skip] if gt_skip else [])),
                    "image_path": sample["image"],
                    "multiple_choice_answer": gt_skip,
                    "prediction": "",
                    "skipped": True,
                    "skip_reason": "image_not_found",
                }
            )
            continue

        image = Image.open(sample["image"]).convert("RGB")
        pred_text, pred_raw = _generate_answer(
            image,
            sample["question"],
            bundle,
            conv_mode=conv_mode,
            max_new_tokens=max_new_tokens,
        )

        gt = sample["answer"]
        all_answers = sample.get("all_answers") or ([gt] if gt else [])
        score = vqa_score(pred_text, all_answers) if all_answers else 0.0
        is_exact = exact_match(pred_text, gt) if gt else False

        vqa_scores.append(score)
        if is_exact:
            exact_correct += 1

        record = {
            "index": sample["index"],
            "question": sample["question"],
            "answer": json.dumps(all_answers),
            "image_path": sample["image"],
            "multiple_choice_answer": gt,
            "prediction": pred_text,
            "vqa_score": score,
            "exact_match": is_exact,
            "skipped": False,
        }
        rows_for_xlsx.append(record)

        if idx % log_every == 0 or idx == len(samples):
            avg_vqa = sum(vqa_scores) / max(len(vqa_scores), 1)
            print(
                f"  [{idx}/{len(samples)}] "
                f"vqa_acc={avg_vqa:.2%} exact={exact_correct}/{len(vqa_scores)} skipped={skipped}"
            )

    pd.DataFrame(rows_for_xlsx).to_excel(xlsx_path, index=False)

    total = len(vqa_scores)
    summary = {
        "dataset": dataset_name,
        "backend": "vtb",
        "metric": "vqa_soft_accuracy",
        "accuracy": sum(vqa_scores) / max(total, 1),
        "exact_accuracy": exact_correct / max(total, 1),
        "total": total,
        "correct": exact_correct,
        "skipped": skipped,
        "max_samples": max_samples,
        "checkpoint": checkpoint_path,
        "predictions_file": xlsx_path,
    }

    print(f"Predictions: {xlsx_path}")
    print(
        f"VQA accuracy: {summary['accuracy']:.2%} ({total} samples, skipped {skipped}) | "
        f"Exact match: {summary['exact_accuracy']:.2%} ({exact_correct}/{total})"
    )
    return summary


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.evaluation.runner <config_path> [dataset_name]")
        sys.exit(1)
    ds = sys.argv[2] if len(sys.argv) > 2 else "VQAv2_VAL"
    evaluate(sys.argv[1], ds)
