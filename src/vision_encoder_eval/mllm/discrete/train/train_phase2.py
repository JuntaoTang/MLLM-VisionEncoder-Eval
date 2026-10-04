"""Phase 2 training: fine-tune projector + LLM jointly."""

from __future__ import annotations

import argparse
import os

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from vision_encoder_eval.mllm.discrete.data.dataset import build_training_dataset, collate_fn
from vision_encoder_eval.mllm.discrete.model.discrete_adapter import DiscreteVisualAdapter
from vision_encoder_eval.mllm.discrete.model.tokenizers.factory import build_visual_tokenizer
from vision_encoder_eval.mllm.discrete.model.vision_config import build_arch_config, resolve_vis_mode
from vision_encoder_eval.mllm.discrete.train.common import (
    DiscreteTrainer,
    build_training_args,
    configure_model_max_length,
    get_mm_projector_lr,
    load_arch_config,
    save_arch_config,
)


def _load_phase1_adapter_weights(model, ckpt_path: str) -> None:
    """Load Phase 1 trainable vision adapter weights into Phase 2 model."""
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    adapter_sd = {
        k: v
        for k, v in sd.items()
        if k.startswith("projector.") or "visual_embed" in k
    }
    if not adapter_sd:
        print(
            f"Warning: no projector/visual_embed weights found in {ckpt_path}; "
            "Phase 2 starts with a randomly initialized vision adapter.",
            flush=True,
        )
        return
    model.load_state_dict(adapter_sd, strict=False)
    loaded = ", ".join(sorted(adapter_sd.keys()))
    print(f"Loaded phase1 adapter weights from {ckpt_path}: {loaded}", flush=True)


def build_model(cfg: dict, phase1_checkpoint: str | None = None):
    llm_cfg = cfg["llm"]
    vis_mode = resolve_vis_mode(cfg)
    if phase1_checkpoint:
        phase1_arch = load_arch_config(phase1_checkpoint)
        vis_mode = phase1_arch.get("vis_mode", vis_mode)

    print(f"Loading LLM: {llm_cfg['model_name_or_path']}")
    llm = AutoModelForCausalLM.from_pretrained(
        llm_cfg["model_name_or_path"], torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(llm_cfg["model_name_or_path"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer_vq = build_visual_tokenizer(cfg)
    hidden_size = llm_cfg.get("hidden_size", llm.config.hidden_size)
    model = DiscreteVisualAdapter(
        tokenizer=tokenizer_vq,
        llm=llm,
        hidden_size=hidden_size,
        vis_mode=vis_mode,
        projector_cfg=cfg.get("projector"),
    )
    if phase1_checkpoint:
        ckpt_path = os.path.join(phase1_checkpoint, "pytorch_model.bin")
        if os.path.isfile(ckpt_path):
            _load_phase1_adapter_weights(model, ckpt_path)
    model.set_phase(2)
    model = model.to(torch.bfloat16)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.get_trainable_params())
    print(f"Vision mode: {model.vis_mode}")
    print(f"Total params: {total:,} | Trainable: {trainable:,}")
    return model, tokenizer, tokenizer_vq


def train(config_path: str, output_dir: str, phase1_checkpoint: str | None = None, resume_from_checkpoint: str | None = None):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    resume_from_checkpoint = resume_from_checkpoint or cfg.get("resume_from_checkpoint")
    model, tokenizer, vq = build_model(cfg, phase1_checkpoint if not resume_from_checkpoint else None)
    dc = cfg["data"]
    tc = cfg.get("training", {})
    datasets = dc.get("datasets")
    if not datasets:
        datasets = [{
            "data_path": dc["data_path"],
            "image_folder": dc["image_folder"],
            "sampling_strategy": dc.get("sampling_strategy", "all"),
        }]
    max_length = int(dc.get("max_length") or tc.get("model_max_length", 8192))
    configure_model_max_length(model, tokenizer, max_length)
    dataset = build_training_dataset(
        datasets=datasets,
        tokenizer=tokenizer,
        image_size=vq.image_size,
        max_length=max_length,
        seed=int(tc.get("seed", 42)),
    )
    print(f"Training dataset size: {len(dataset)}", flush=True)
    tc.setdefault("gradient_checkpointing", True)
    args = build_training_args(output_dir=output_dir, tc=tc, phase=2)
    mm_projector_lr = get_mm_projector_lr(tc)
    if mm_projector_lr is not None:
        print(f"mm_projector_lr: {mm_projector_lr}", flush=True)
    trainer = DiscreteTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collate_fn,
        mm_projector_lr=mm_projector_lr,
    )
    if resume_from_checkpoint:
        print(f"Resuming finetune from {resume_from_checkpoint}", flush=True)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(output_dir)
    trainer.save_state()
    save_arch_config(output_dir, build_arch_config(cfg, model.vis_mode))
    print(f"Phase 2 done. Model saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--output-dir", default="./logs/phase2")
    parser.add_argument("--phase1-checkpoint", default=None)
    parser.add_argument("--resume-from-checkpoint", default=None)
    args = parser.parse_args()
    train(args.config, args.output_dir, args.phase1_checkpoint, args.resume_from_checkpoint)
