from __future__ import annotations
import math
from functools import partial
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from ..workers.alignment.sail_optim import Lion, cosine_lr

class StarMLP(nn.Module):
    def __init__(self, input_dim, output_dim, width_factor=8, activation=None):
        super().__init__()
        self.f1 = nn.Linear(input_dim, width_factor * input_dim)
        self.f2 = nn.Linear(input_dim, width_factor * input_dim)
        self.act = activation
        self.g = nn.Linear(width_factor * input_dim, output_dim)

    def forward(self, x):
        x1 = torch.clamp(self.f1(x), min=-1e3, max=1e3)
        x2 = torch.clamp(self.f2(x), min=-1e3, max=1e3)
        x = self.act(x1) * x2 if self.act else x1 * x2
        return self.g(x)

class SiglipMLP(nn.Module):
    def __init__(self, input_dim, output_dim, intermediate_dim=None):
        super().__init__()
        intermediate_dim = intermediate_dim or 4 * input_dim
        self.proj = nn.Sequential(
            nn.Linear(input_dim, intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, output_dim),
        )

    def forward(self, x):
        return self.proj(x)

class AlignmentLayer(nn.Module):
    def __init__(self, vision_dim, text_dim, target_dimension=1024,
                 linear_type="linear", logit_scale=20.0, logit_bias=-10.0,
                 width_factor=8):
        super().__init__()
        if linear_type == "star":
            Linear = partial(StarMLP, width_factor=width_factor, activation=nn.ReLU6())
        elif linear_type == "mlp":
            Linear = SiglipMLP
        else:
            Linear = nn.Linear

        self.vision_layer_norm = nn.LayerNorm(vision_dim)
        self.vision_mapping_network = Linear(vision_dim, target_dimension)
        self.text_layer_norm = nn.LayerNorm(text_dim)
        self.text_mapping_network = Linear(text_dim, target_dimension)
        self.logit_scale = nn.Parameter(torch.zeros(1))
        self.logit_bias = nn.Parameter(torch.zeros(1))
        self._init(logit_scale, logit_bias)

    def _init(self, scale, bias):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        self.logit_scale.data.fill_(float(np.log(scale)))
        self.logit_bias.data.fill_(float(bias))

    def encode_image(self, v):
        return self.vision_mapping_network(self.vision_layer_norm(v))

    def encode_text(self, t):
        return self.text_mapping_network(self.text_layer_norm(t))

def siglip_loss(img, txt, logit_scale, logit_bias, extra_txt=None,
                reduction="mean"):
    """SAIL model/loss.py::SigLipLoss — extra positives are concatenated as a
    second N x N block and the whole thing is averaged."""
    img = F.normalize(img, dim=-1)
    txt = F.normalize(txt, dim=-1)
    logits = logit_scale * img @ txt.t() + logit_bias
    n = logits.shape[0]
    labels = 2 * torch.eye(n, device=logits.device, dtype=logits.dtype) - 1
    loss = F.logsigmoid(labels * logits)
    if extra_txt is not None:
        extra = F.normalize(extra_txt, dim=-1)
        extra_logits = logit_scale * img @ extra.t() + logit_bias
        loss = torch.cat([loss, F.logsigmoid(labels * extra_logits)], dim=1)
    # SAIL averages over the whole N x (N or 2N) matrix; "sum_over_n" is the
    # scale the first COCO-only runs used. AdamW is near scale-invariant, so
    # this shifts scores by well under a point.
    return -loss.mean() if reduction == "mean" else -loss.sum() / n

def clip_loss(img, txt, logit_scale, logit_bias, extra_txt=None):
    img = F.normalize(img, dim=-1)
    txt = F.normalize(txt, dim=-1)
    logits = logit_scale * img @ txt.t()
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
    if extra_txt is not None:
        el = logit_scale * img @ F.normalize(extra_txt, dim=-1).t()
        loss = 0.5 * loss + 0.5 * 0.5 * (F.cross_entropy(el, labels)
                                         + F.cross_entropy(el.t(), labels))
    return loss

def _recalls(ranks: np.ndarray, ks=(1, 5, 10)) -> dict[str, float]:
    return {f"R@{k}": float((ranks < k).mean() * 100.0) for k in ks}

@torch.no_grad()
def retrieval(model, V: torch.Tensor, T: torch.Tensor, ks=(1, 5, 10)) -> dict:
    """V: [N, Dv]; T: [N, C, Dt]. Returns 1-caption and C-caption protocols."""
    model.eval()
    n, c = T.shape[0], T.shape[1]
    ie = F.normalize(model.encode_image(V), dim=-1)                    # [N, d]
    te = F.normalize(model.encode_text(T.reshape(n * c, -1)), dim=-1)  # [N*C, d]

    out = {}

    # --- protocol A: one caption per image (the N x N pair matrix) ---
    te1 = te.reshape(n, c, -1)[:, 0]
    sim = ie @ te1.t()
    i2t = (sim > sim.diag().unsqueeze(1)).sum(1).cpu().numpy()
    t2i = (sim.t() > sim.diag().unsqueeze(1)).sum(1).cpu().numpy()
    out["i2t_1cap"] = _recalls(i2t, ks)
    out["t2i_1cap"] = _recalls(t2i, ks)

    # --- protocol B: standard COCO, all C captions per image ---
    sim5 = ie @ te.t()                                   # [N, N*C]
    owner = torch.arange(n, device=V.device).repeat_interleave(c)
    gt = owner.unsqueeze(0) == torch.arange(n, device=V.device).unsqueeze(1)  # [N, N*C]
    # image -> text: rank of best-scoring ground-truth caption
    best_gt = sim5.masked_fill(~gt, float("-inf")).max(dim=1).values
    i2t5 = (sim5 > best_gt.unsqueeze(1)).sum(1).cpu().numpy()
    # text -> image: rank of the owning image
    sim5t = sim5.t()                                     # [N*C, N]
    gt_score = sim5t.gather(1, owner.unsqueeze(1))
    t2i5 = (sim5t > gt_score).sum(1).cpu().numpy()
    out["i2t_5cap"] = _recalls(i2t5, ks)
    out["t2i_5cap"] = _recalls(t2i5, ks)

    for tag in ("1cap", "5cap"):
        vals = list(out[f"i2t_{tag}"].values()) + list(out[f"t2i_{tag}"].values())
        out[f"mean_recall_{tag}"] = float(np.mean(vals))
    return out

def train_once(Vtr, Ttr, Vva, Tva, *, args, wd: float, lr: float, device: str,
               seed: int, progress: tqdm | None = None):
    torch.manual_seed(seed)
    model = AlignmentLayer(
        Vtr.shape[1], Ttr.shape[-1],
        target_dimension=args.target_dimension,
        linear_type=args.linear_type,
        logit_scale=args.logit_scale,
        logit_bias=args.logit_bias,
        width_factor=args.width_factor,
    ).to(device)
    if args.optimizer == "lion":
        opt = Lion(model.parameters(), lr=lr, betas=(args.beta1, args.beta2),
                   weight_decay=wd)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd,
                                betas=(args.beta1, args.beta2))
    # SAIL main.py: cosine schedule with warmup = ceil(0.1 * total_steps)
    sched = cosine_lr(opt, lr, math.ceil(0.1 * args.steps), args.steps)

    n = Vtr.shape[0]
    bs = min(args.batch_size, n)
    g = torch.Generator(device="cpu").manual_seed(seed)
    loss_fn = (partial(siglip_loss, reduction=args.loss_reduction)
               if args.loss == "siglip" else clip_loss)

    best = {"val": -1.0, "step": -1, "state": None}
    last_loss = float("nan")
    for step in range(1, args.steps + 1):
        model.train()
        bi = (torch.arange(n, device=device) if bs >= n
              else torch.randperm(n, generator=g)[:bs].to(device))
        v = Vtr[bi]
        if args.train_captions > 1:
            ci = torch.randint(0, args.train_captions, (len(bi),), generator=g).to(device)
            t = Ttr[bi, ci]
        else:
            t = Ttr[bi, 0]
        extra = None
        if args.extra_positive and Ttr.shape[1] > 1:
            extra = model.encode_text(Ttr[bi, 1])
        loss = loss_fn(model.encode_image(v), model.encode_text(t),
                       model.logit_scale.exp(), model.logit_bias, extra)
        opt.zero_grad(set_to_none=True)
        sched(step - 1)
        loss.backward()
        if args.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        opt.step()
        last_loss = float(loss.item())

        track = args.select == "val" and Vva is not None and len(Vva) > 0
        if track and (step % args.eval_every == 0 or step == args.steps):
            m = retrieval(model, Vva, Tva)
            score = m["mean_recall_1cap"]
            if score > best["val"]:
                best = {"val": score, "step": step,
                        "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
        if progress is not None:
            progress.update(1)

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    else:
        best = {"val": float("nan"), "step": args.steps, "state": None}
    return model, {"val_mean_recall": best["val"], "best_step": best["step"],
                   "final_train_loss": last_loss, "wd": wd, "lr": lr}
