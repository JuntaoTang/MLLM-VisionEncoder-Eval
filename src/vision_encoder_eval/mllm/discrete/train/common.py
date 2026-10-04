"""Shared helpers for discrete MLLM training."""



from __future__ import annotations



import json

from typing import Any



from transformers import PreTrainedTokenizer, Trainer, TrainingArguments

from transformers.trainer_pt_utils import get_parameter_names

from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS





def configure_model_max_length(model, tokenizer: PreTrainedTokenizer, max_length: int) -> int:

    max_length = int(max_length)

    tokenizer.model_max_length = max_length

    model.tokenizer_model_max_length = max_length

    return max_length





def _get_training_float(tc: dict, *keys: str, default: float) -> float:

    for key in keys:

        if key in tc and tc[key] is not None:

            return float(tc[key])

    return default





def build_training_args(

    *,

    output_dir: str,

    tc: dict,

    phase: int,

    num_workers: int = 8,

) -> TrainingArguments:

    gradient_checkpointing = bool(tc.get("gradient_checkpointing", phase == 2))

    save_strategy = tc.get("save_strategy", "no")
    if save_strategy is False:
        save_strategy = "no"

    kwargs = dict(

        output_dir=output_dir,

        per_device_train_batch_size=int(tc.get("batch_size", 8 if phase == 1 else 4)),

        gradient_accumulation_steps=int(tc.get("grad_accum", 2 if phase == 1 else 4)),

        learning_rate=_get_training_float(

            tc, "lr", "learning_rate", default=1e-4 if phase == 1 else 1e-5

        ),

        warmup_ratio=float(tc.get("warmup_ratio", 0.03)),

        lr_scheduler_type=str(tc.get("lr_scheduler_type", "cosine")),

        num_train_epochs=float(tc.get("num_epochs", 1)),

        weight_decay=float(tc.get("weight_decay", 0.0)),

        max_grad_norm=float(tc.get("max_grad_norm", 1.0)),

        bf16=bool(tc.get("bf16", True)),

        logging_steps=int(tc.get("logging_steps", 10)),

        save_strategy=str(save_strategy),

        save_total_limit=int(tc.get("save_total_limit", 1)),

        remove_unused_columns=False,

        report_to=str(tc.get("report_to", "none")),

        dataloader_num_workers=int(tc.get("num_workers", num_workers)),

        dataloader_drop_last=bool(tc.get("dataloader_drop_last", False)),

        seed=int(tc.get("seed", 42)),

        save_safetensors=False,

        ddp_find_unused_parameters=phase == 1,

    )

    if gradient_checkpointing:

        kwargs["gradient_checkpointing"] = True

        gc_kwargs = tc.get("gradient_checkpointing_kwargs")

        if gc_kwargs:

            if isinstance(gc_kwargs, str):

                gc_kwargs = json.loads(gc_kwargs)

            kwargs["gradient_checkpointing_kwargs"] = gc_kwargs

    if tc.get("tf32"):

        kwargs["tf32"] = True

    if tc.get("max_steps") is not None:
        kwargs["max_steps"] = int(tc["max_steps"])

    if tc.get("save_steps") is not None:
        kwargs["save_steps"] = int(tc["save_steps"])

    return TrainingArguments(**kwargs)





def get_mm_projector_lr(tc: dict) -> float | None:

    if tc.get("mm_projector_lr") is None:

        return None

    return float(tc["mm_projector_lr"])





class DiscreteTrainer(Trainer):

    """Trainer with optional higher LR for projector parameters (VTB-style)."""



    def __init__(self, *args, mm_projector_lr: float | None = None, **kwargs):

        self.mm_projector_lr = mm_projector_lr

        super().__init__(*args, **kwargs)



    def create_optimizer(self):

        if self.mm_projector_lr is None:

            return super().create_optimizer()



        opt_model = self.model

        decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)

        decay_parameters = [name for name in decay_parameters if "bias" not in name]

        projector_parameters = [

            name for name, _ in opt_model.named_parameters() if "projector" in name

        ]

        optimizer_grouped_parameters = [

            {

                "params": [

                    p for n, p in opt_model.named_parameters()

                    if n in decay_parameters and n not in projector_parameters and p.requires_grad

                ],

                "weight_decay": self.args.weight_decay,

            },

            {

                "params": [

                    p for n, p in opt_model.named_parameters()

                    if n not in decay_parameters and n not in projector_parameters and p.requires_grad

                ],

                "weight_decay": 0.0,

            },

            {

                "params": [

                    p for n, p in opt_model.named_parameters()

                    if n in decay_parameters and n in projector_parameters and p.requires_grad

                ],

                "weight_decay": self.args.weight_decay,

                "lr": self.mm_projector_lr,

            },

            {

                "params": [

                    p for n, p in opt_model.named_parameters()

                    if n not in decay_parameters and n in projector_parameters and p.requires_grad

                ],

                "weight_decay": 0.0,

                "lr": self.mm_projector_lr,

            },

        ]

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer





def save_arch_config(output_dir: str, arch_cfg: dict[str, Any]) -> None:

    import os

    import yaml



    if not arch_cfg:

        return

    path = os.path.join(output_dir, "arch_config.yaml")

    with open(path, "w", encoding="utf-8") as f:

        yaml.dump(arch_cfg, f, sort_keys=False)





def load_arch_config(model_path: str) -> dict[str, Any]:

    import os

    import yaml



    path = os.path.join(model_path, "arch_config.yaml")

    if not os.path.isfile(path):

        return {}

    with open(path, encoding="utf-8") as f:

        return yaml.safe_load(f) or {}





def extract_trainer_metrics(output_dir: str) -> dict[str, Any]:

    import os



    path = os.path.join(output_dir, "trainer_state.json")

    if not os.path.isfile(path):

        return {"status": "missing_trainer_state"}

    with open(path, encoding="utf-8") as f:

        state = json.load(f)

    history = state.get("log_history", [])

    losses = [float(x["loss"]) for x in history if "loss" in x]

    metrics: dict[str, Any] = {

        "status": "ok",

        "train_runtime": state.get("train_runtime"),

        "train_loss": state.get("train_loss"),

        "epoch": state.get("epoch"),

    }

    if losses:

        metrics["initial_loss"] = losses[0]

        metrics["final_loss"] = losses[-1]

        metrics["loss_curve"] = losses

    return metrics

