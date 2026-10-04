"""Phase 1 training: pretrain (trainable modules configured per MLLM recipe)."""

from __future__ import annotations

import argparse

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from vision_encoder_eval.mllm.discrete.data.dataset import LLaVADataset, build_training_dataset, collate_fn
from vision_encoder_eval.mllm.discrete.model.discrete_adapter import DiscreteVisualAdapter
from vision_encoder_eval.mllm.discrete.model.tokenizers.factory import build_visual_tokenizer
from vision_encoder_eval.mllm.discrete.model.vision_config import build_arch_config, resolve_vis_mode
from vision_encoder_eval.mllm.discrete.train.common import (
    DiscreteTrainer,
    build_training_args,
    configure_model_max_length,
    get_mm_projector_lr,
    save_arch_config,
)
from vision_encoder_eval.mllm.discrete.train.pretrain_settings import resolve_pretrain_settings


def build_model(cfg: dict):
    llm_cfg = cfg["llm"]
    vis_mode = resolve_vis_mode(cfg)
    pretrain = resolve_pretrain_settings(cfg)

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
    model.set_trainable_modules(
        llm=pretrain.tune_llm,
        projector=pretrain.tune_projector,
    )
    model = model.to(torch.bfloat16)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.get_trainable_params())
    print(f"Vision mode: {model.vis_mode}")
    print(f"Pretrain: {pretrain.summary()}")
    print(f"Total params: {total:,} | Trainable: {trainable:,}")
    return model, tokenizer, tokenizer_vq


def train(config_path: str, output_dir: str, resume_from_checkpoint: str | None = None):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    resume_from_checkpoint = resume_from_checkpoint or cfg.get("resume_from_checkpoint")
    pretrain = resolve_pretrain_settings(cfg)
    model, tokenizer, vq = build_model(cfg)
    dc = cfg["data"]
    tc = dict(cfg.get("training", {}))
    tc.update(pretrain.training_overrides)
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
        seed=int(cfg.get("training", {}).get("seed", 42)),
    )
    print(f"Training dataset size: {len(dataset)}", flush=True)
    args = build_training_args(
        output_dir=output_dir,
        tc=tc,
        phase=pretrain.training_phase,
    )
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
        print(f"Resuming pretrain from {resume_from_checkpoint}", flush=True)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(output_dir)
    trainer.save_state()
    save_arch_config(output_dir, build_arch_config(cfg, model.vis_mode))
    print(f"Phase 1 done. Model saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--output-dir", default="./logs/phase1")
    parser.add_argument("--resume-from-checkpoint", default=None)
    args = parser.parse_args()
    train(args.config, args.output_dir, args.resume_from_checkpoint)
