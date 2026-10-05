
from vision_encoder_eval.core.runtime import asset_path
import os
import shutil
import subprocess
import sys
from typing import Optional

from vision_encoder_eval.mllm.utils.config import (
    HF_HOME_DEFAULT,
    PRETRAINED_ROOT,
    RunContext,
    VTB_ROOT,
    apply_cuda_stub_env,
    apply_offline_hf_env,
    format_stage_data_summary,
    resolve_llava_project,
)


def _vision_tower_args(vision: dict) -> tuple[str, Optional[str], str]:
    select_layer = str(vision.get("select_layer", -2))
    tower_type = vision.get("type", "open_clip_hub")

    if tower_type in ("dinov3", "raev2", "ijepa"):
        tower = f"vtb_ssl:{tower_type}"
        weights = vision.get("weights_path")
        return tower, weights, select_layer

    if tower_type in ("pe", "eupe", "pixio"):
        tower = f"vtb_ssl:{tower_type}"
        weights = vision.get("weights_path") or vision.get("model_name_or_path")
        return tower, weights, select_layer

    # Generic HuggingFace transformers vision towers (DINOv2 / DINO / Web-SSL / Pixio / DINOv3-HF).
    # Must use hf: prefix so builder routes to HFVisionTower (absolute paths otherwise hit CLIP).
    if tower_type in ("hf", "hf_vision"):
        path = (
            vision.get("model_name_or_path")
            or vision.get("vision_tower")
            or vision.get("weights_path")
        )
        if not path:
            raise ValueError("hf vision_encoder requires model_name_or_path")
        return f"hf:{path}", None, select_layer

    if tower_type == "hf_clip" or vision.get("vision_tower"):
        tower = (
            vision.get("vision_tower")
            or vision.get("model_name_or_path")
            or vision.get("weights_path")
            or "openai/clip-vit-large-patch14"
        )
        weights = vision.get("weights_path")
        if weights and os.path.isdir(weights):
            tower = weights
        if os.path.isdir(str(tower)):
            return tower, None, select_layer
        # Offline: remap hub ids to local CLIP processor/weights when available.
        local_clip = os.path.join(
            os.environ.get("VTB_CLIP_IMAGE_PROCESSOR", ""),
        ) if os.environ.get("VTB_CLIP_IMAGE_PROCESSOR") else ""
        fallback = asset_path('download', 'tokenizer/continuous/clip-vit-large-patch14')
        if (not os.path.isdir(str(tower))) and ("clip-vit-large-patch14" in str(tower) or str(tower).startswith("openai/")):
            for cand in (local_clip, fallback):
                if cand and os.path.isdir(cand):
                    tower = cand
                    break
        return tower, None, select_layer

    model_name = vision.get("model_name", "ViT-L-14")
    tower = f"open_clip_hub:{model_name}"
    pretrained = vision.get("pretrained", "metaclip_fullcc")
    weights = vision.get("weights_path")
    if weights:
        pretrained = weights
    return tower, pretrained, select_layer


def _resolve_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    if not os.path.isabs(path):
        path = os.path.join(VTB_ROOT, path)
    return os.path.abspath(path)


def _is_full_hf_checkpoint(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    markers = ("trainer_state.json", "pytorch_model.bin", "model.safetensors")
    return any(os.path.isfile(os.path.join(path, name)) for name in markers)


def _mm_projector_in(path: str) -> Optional[str]:
    direct = os.path.join(path, "mm_projector.bin")
    if os.path.isfile(direct):
        return direct
    return None


def _find_mm_projector(pretrain_dir: str) -> Optional[str]:
    if not os.path.isdir(pretrain_dir):
        return None
    direct = _mm_projector_in(pretrain_dir)
    if direct:
        return direct
    saved = os.path.join(pretrain_dir, "saved_weights")
    search_dirs = [pretrain_dir]
    if os.path.isdir(saved):
        search_dirs.insert(0, saved)
    checkpoints = []
    for base in search_dirs:
        checkpoints.extend(
            d for d in os.listdir(base) if d.startswith("checkpoint-") and os.path.isdir(os.path.join(base, d))
        )
    checkpoints = sorted(set(checkpoints), key=lambda x: int(x.split("-")[-1]))
    for ckpt in reversed(checkpoints):
        for base in search_dirs:
            path = os.path.join(base, ckpt, "mm_projector.bin")
            if os.path.isfile(path):
                return path
    return None


def _archive_adapter_checkpoints(output_dir: str) -> None:
    """Move adapter-only checkpoint-* aside so LLaVA won't try broken auto-resume."""
    if not os.path.isdir(output_dir):
        return
    saved = os.path.join(output_dir, "saved_weights")
    os.makedirs(saved, exist_ok=True)
    for name in os.listdir(output_dir):
        if not name.startswith("checkpoint-"):
            continue
        src = os.path.join(output_dir, name)
        if not os.path.isdir(src) or _is_full_hf_checkpoint(src):
            continue
        dest = os.path.join(saved, name)
        if os.path.exists(dest):
            continue
        shutil.move(src, dest)
        print(f"Archived adapter checkpoint: {src} -> {dest}")


def _resolve_resume_and_init(ctx: RunContext, stage: str) -> tuple[Optional[str], Optional[str]]:
    resume_key = f"{stage}_resume"
    resume_raw = ctx.checkpoints.get(resume_key)
    resume_path = _resolve_path(resume_raw)

    init_path = None
    if stage == "finetune":
        init_path = _resolve_path(ctx.checkpoints.get("finetune_init"))
        if not init_path:
            init_path = _find_mm_projector(ctx.pretrain_dir)

    resume_for_cmd = None
    if resume_path:
        if _is_full_hf_checkpoint(resume_path):
            resume_for_cmd = resume_path
        else:
            mm = _mm_projector_in(resume_path)
            if mm:
                init_path = mm
                print(
                    f"{resume_key} points to adapter-only weights; "
                    f"loading {mm} without optimizer resume."
                )
            else:
                raise FileNotFoundError(f"Invalid checkpoint path (no weights found): {resume_path}")

    return resume_for_cmd, init_path


def _build_train_cmd(
    ctx: RunContext,
    stage: str,
    resume_path: Optional[str],
    init_path: Optional[str],
    *,
    num_gpus_override: Optional[int] = None,
    master_port_override: Optional[int] = None,
) -> list[str]:
    if stage not in ("pretrain", "finetune"):
        raise ValueError(f"Not a training stage: {stage}")

    train = ctx.train[stage]
    data = ctx.data.get(stage, {})
    batch = ctx.stage_batch(stage)
    num_gpus = int(num_gpus_override if num_gpus_override is not None else batch["num_gpus"])
    micro = batch["per_device_train_batch_size"]
    accum = batch["gradient_accumulation_steps"]
    output_dir = ctx.stage_output_dir(stage)

    llava_root = resolve_llava_project(ctx.paths)
    train_script = os.path.join(llava_root, "llava", "train", "train_mem.py")
    if not os.path.isfile(train_script):
        raise FileNotFoundError(f"LLaVA train script not found: {train_script}")

    vision_tower, vision_pretrained, select_layer = _vision_tower_args(ctx.vision_encoder)
    projector_type = ctx.projector.get("type", "mlp2x_gelu")
    runtime = ctx.runtime

    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={num_gpus}",
        f"--master_port={master_port_override if master_port_override is not None else runtime.get('master_port', 29501)}",
        train_script,
        "--model_name_or_path", ctx.llm["model_name_or_path"],
        "--version", ctx.llm.get("model_version", "llava_llama_3"),
        "--data_path", data["data_path"],
        "--image_folder", data["image_folder"],
        "--vision_tower", vision_tower,
        "--mm_vision_select_layer", select_layer,
        "--mm_vision_select_feature", str(ctx.vision_encoder.get("select_feature", "patch")),
        "--mm_projector_type", projector_type,
        "--mm_patch_merge_type", "flat",
        "--image_aspect_ratio", "square",
        "--mm_use_im_start_end", "False",
        "--mm_use_im_patch_token", "False",
        "--mm_tunable_parts", train.get("tunable_parts", "mm_mlp_adapter"),
        "--bf16", str(train.get("bf16", True)),
        "--output_dir", output_dir,
        "--num_train_epochs", str(train.get("num_train_epochs", 1)),
        "--per_device_train_batch_size", str(micro),
        "--gradient_accumulation_steps", str(accum),
        "--dataloader_num_workers", str(runtime.get("dataloader_num_workers", 8)),
        "--dataloader_pin_memory", "True",
        "--dataloader_persistent_workers", "True" if int(runtime.get("dataloader_num_workers", 8)) > 0 else "False",
        "--save_strategy", str(train.get("save_strategy", "epoch")),
        "--save_total_limit", str(train.get("save_total_limit", 1)),
        "--learning_rate", str(train.get("learning_rate", 1e-5)),
        "--weight_decay", str(train.get("weight_decay", 0.0)),
        "--warmup_ratio", str(train.get("warmup_ratio", 0.03)),
        "--lr_scheduler_type", str(train.get("lr_scheduler_type", "cosine")),
        "--logging_steps", str(train.get("logging_steps", 10)),
        "--model_max_length", str(train.get("model_max_length", 8192)),
        "--gradient_checkpointing", str(train.get("gradient_checkpointing", False)),
        "--lazy_preprocess", str(train.get("lazy_preprocess", True)),
        "--report_to", str(train.get("report_to", "none")),
        "--attn_implementation", str(train.get("attn_implementation", "sdpa")),
        "--eval_strategy", "no",
    ]

    if vision_pretrained:
        cmd.extend(["--vision_tower_pretrained", vision_pretrained])

    if train.get("max_grad_norm") is not None:
        cmd.extend(["--max_grad_norm", str(train["max_grad_norm"])])

    if train.get("dataloader_drop_last"):
        cmd.extend(["--dataloader_drop_last", "True"])

    gc_kwargs = train.get("gradient_checkpointing_kwargs")
    if gc_kwargs and train.get("gradient_checkpointing"):
        cmd.extend(["--gradient_checkpointing_kwargs", gc_kwargs])

    if train.get("tf32"):
        cmd.extend(["--tf32", str(train["tf32"])])

    if train.get("max_steps") is not None:
        cmd.extend(["--max_steps", str(train["max_steps"])])

    if train.get("group_by_modality_length"):
        cmd.extend(["--group_by_modality_length", "True"])

    mm_lr = train.get("mm_projector_lr")
    if mm_lr is not None and stage == "finetune":
        cmd.extend(["--mm_projector_lr", str(mm_lr)])

    if resume_path:
        cmd.extend(["--resume_from_checkpoint", resume_path])

    if init_path:
        cmd.extend(["--pretrain_mm_mlp_adapter", init_path])

    return cmd, output_dir, micro, accum, num_gpus


def setup_env(ctx: RunContext, *, cuda_visible_devices: Optional[str] = None) -> dict:
    llava_root = resolve_llava_project(ctx.paths)
    hf_home = ctx.paths.get("hf_home", HF_HOME_DEFAULT)
    weights = ctx.vision_encoder.get("weights_path")
    tower_type = ctx.vision_encoder.get("type", "open_clip_hub")

    env = apply_cuda_stub_env(apply_offline_hf_env())
    env["PYTHONPATH"] = f"{llava_root}:{env.get('PYTHONPATH', '')}"
    # Prefer an already-selected device (orchestrator / smoke --gpu) over runtime yaml.
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    elif os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, ""):
        env["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
    else:
        env["CUDA_VISIBLE_DEVICES"] = str(ctx.runtime.get("cuda_visible_devices", "0"))
    env["OMP_NUM_THREADS"] = "8"
    env["HF_HOME"] = hf_home
    env["HUGGINGFACE_HUB_CACHE"] = os.path.join(hf_home, "hub")
    # Mid-run CE OOMs often leave ~15GB free but need a large contiguous chunk.
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    clip_processor = ctx.vision_encoder.get("processor_path")
    if not clip_processor:
        default_processor = os.path.join(
            os.path.dirname(weights or ""),
            "clip-vit-large-patch14",
        ) if weights else ""
        if default_processor and os.path.isdir(default_processor):
            clip_processor = default_processor
        clip_fallback = os.path.join(PRETRAINED_ROOT, "visual_encoder", "clip-vit-large-patch14")
        if os.path.isdir(clip_fallback):
            clip_processor = clip_fallback
    if clip_processor and os.path.isdir(clip_processor):
        env["VTB_CLIP_IMAGE_PROCESSOR"] = clip_processor
    force_qg = ctx.vision_encoder.get("force_quick_gelu")
    model_name = str(ctx.vision_encoder.get("model_name") or ctx.vision_encoder.get("vision_tower") or "")
    if force_qg is None and "siglip" in model_name.lower():
        force_qg = False
    if force_qg is True:
        env["VTB_FORCE_QUICK_GELU"] = "1"
    elif force_qg is False:
        env["VTB_FORCE_QUICK_GELU"] = "0"
    if weights and tower_type != "hf_clip":
        env["VTB_VISION_WEIGHTS"] = weights
    force_image_size = ctx.vision_encoder.get("force_image_size")
    if force_image_size is not None:
        env["VTB_FORCE_IMAGE_SIZE"] = str(int(force_image_size))
    # SSL continuous towers (DINOv3 / RAEv2 / I-JEPA / PE / EUPE / Pixio)
    if tower_type in ("dinov3", "raev2", "ijepa", "pe", "eupe", "pixio"):
        env["VTB_ROOT"] = VTB_ROOT
        image_size = ctx.vision_encoder.get("image_size") or force_image_size
        if image_size is not None:
            env["VTB_SSL_IMAGE_SIZE"] = str(int(image_size))
        layers = ctx.vision_encoder.get("layers")
        if layers is not None:
            if isinstance(layers, (list, tuple)):
                env["VTB_SSL_LAYERS"] = ".".join(str(int(x)) for x in layers)
            else:
                env["VTB_SSL_LAYERS"] = str(layers)
        dinov3_repo = ctx.vision_encoder.get("dinov3_repo") or os.path.join(
            VTB_ROOT, "third_party", "dinov3"
        )
        if os.path.isdir(dinov3_repo):
            env["VTB_DINOV3_REPO_DIR"] = dinov3_repo
            env["DINOV3_REPO_DIR"] = dinov3_repo
        pe_repo = ctx.vision_encoder.get("pe_repo") or os.path.join(
            VTB_ROOT, "third_party", "perception_models"
        )
        if os.path.isdir(pe_repo):
            env["VTB_PE_REPO_DIR"] = pe_repo
        pe_config = ctx.vision_encoder.get("pe_config") or ctx.vision_encoder.get("model_name")
        if pe_config:
            env["VTB_PE_CONFIG"] = str(pe_config)
        eupe_repo = ctx.vision_encoder.get("eupe_repo") or os.path.join(
            VTB_ROOT, "third_party", "eupe"
        )
        if os.path.isdir(eupe_repo):
            env["VTB_EUPE_REPO_DIR"] = eupe_repo
        eupe_hub = ctx.vision_encoder.get("eupe_hub")
        if eupe_hub:
            env["VTB_EUPE_HUB"] = str(eupe_hub)
        dinov3_backbone = ctx.vision_encoder.get("dinov3_backbone")
        if dinov3_backbone:
            env["VTB_DINOV3_BACKBONE"] = str(dinov3_backbone)
        pixio_repo = ctx.vision_encoder.get("pixio_repo") or os.path.join(
            VTB_ROOT, "third_party", "pixio"
        )
        if os.path.isdir(pixio_repo):
            env["VTB_PIXIO_REPO_DIR"] = pixio_repo
        pixio_hub = ctx.vision_encoder.get("pixio_hub")
        if pixio_hub:
            env["VTB_PIXIO_HUB"] = str(pixio_hub)
    if tower_type in ("hf", "hf_vision"):
        # Local HF dirs: keep transformers offline-friendly.
        env.setdefault("TRANSFORMERS_OFFLINE", "1")
        env.setdefault("HF_HUB_OFFLINE", "1")
        select_feature = ctx.vision_encoder.get("select_feature")
        if select_feature:
            env["VTB_HF_SELECT_FEATURE"] = str(select_feature)
    return env


def run_training_stage(
    ctx: RunContext,
    stage: str,
    log_path: Optional[str] = None,
    *,
    cuda_visible_devices: Optional[str] = None,
    num_gpus: Optional[int] = None,
    master_port: Optional[int] = None,
) -> int:
    from vision_encoder_eval.mllm.utils.checkpoint_layout import prepare_stage_output_dir

    output_dir = prepare_stage_output_dir(ctx, stage)
    resume_path, init_path = _resolve_resume_and_init(ctx, stage)
    cmd, output_dir, micro, accum, effective_gpus = _build_train_cmd(
        ctx,
        stage,
        resume_path,
        init_path,
        num_gpus_override=num_gpus,
        master_port_override=master_port,
    )
    global_batch = micro * accum * effective_gpus
    llava_root = resolve_llava_project(ctx.paths)
    env = setup_env(ctx, cuda_visible_devices=cuda_visible_devices)

    data = ctx.data.get(stage, {})
    header = (
        f"=== {stage.upper()} ===\n"
        f"LLM: {ctx.llm['model_name_or_path']}\n"
        f"Vision: {ctx.vision_encoder.get('display_name') or ctx.vision_encoder.get('vision_tower') or ctx.vision_encoder.get('model_name')}\n"
        f"Data: {format_stage_data_summary(data)}\n"
        f"micro={micro} x accum={accum} x gpus={effective_gpus} => global_batch={global_batch}\n"
        f"LR: {ctx.train[stage].get('learning_rate')}\n"
        f"Output: {output_dir}\n"
    )
    if init_path:
        header += f"Init weights: {init_path}\n"
    if resume_path:
        header += f"Resume: {resume_path}\n"
    print(header)

    os.makedirs(os.path.dirname(log_path) if log_path else output_dir, exist_ok=True)
    stdout = open(log_path, "a") if log_path else None
    if stdout:
        stdout.write(header)

    proc = subprocess.run(
        cmd,
        env=env,
        cwd=llava_root,
        stdout=stdout or None,
        stderr=subprocess.STDOUT if stdout else None,
    )
    if stdout:
        stdout.close()
    if proc.returncode == 0:
        from vision_encoder_eval.mllm.utils.checkpoint_layout import finalize_stage_checkpoint

        finalize_stage_checkpoint(output_dir)
    return proc.returncode
