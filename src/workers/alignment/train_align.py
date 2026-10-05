#!/usr/bin/env python3
"""SAIL-style alignment probing on COCO-2K (stage 2 + retrieval eval).

For one (visual tokenizer, LLM) pair: freeze both backbones, take the cached
mean-pooled embeddings, and train only a lightweight alignment layer
(LayerNorm + mapping network on each side, learnable logit scale/bias) with the
SigLIP objective — the same recipe as SAIL's scripts/alignment_probing.sh.

Train on the 1600-image COCO-2K train split (a slice of which is held out for
hyper-parameter / step selection), then score image<->text retrieval on the
400-image test split.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import vision_encoder_eval.workers.alignment.common as common
from vision_encoder_eval.workers.alignment.sail_optim import Lion, cosine_lr


# --------------------------------------------------------------------------- #
# alignment layer (port of SAIL model/linear.py + model/sail_model.py)
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# retrieval metrics
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# one training run
# --------------------------------------------------------------------------- #
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


def run_combo(slug, llm, train_pair, test_pair, split, args, device):
    (V_tr, T_tr), (V_te, T_te) = train_pair, test_pair
    tr, va, te = split
    Vtr, Ttr = V_tr[tr], T_tr[tr]
    Vva, Tva = (V_tr[va], T_tr[va]) if len(va) else (None, None)
    Vte, Tte = V_te[te], T_te[te]

    grid = [(wd, lr) for wd in args.wd_grid for lr in args.lr_grid]
    total = len(grid) * args.steps
    bar = tqdm(total=total, desc=f"{llm}/{slug}", unit="step", leave=False)
    best_model, best_info = None, None
    for wd, lr in grid:
        model, info = train_once(Vtr, Ttr, Vva, Tva, args=args, wd=wd, lr=lr,
                                 device=device, seed=args.seed, progress=bar)
        if best_info is None or info["val_mean_recall"] > best_info["val_mean_recall"]:
            best_model, best_info = model, info
    bar.close()

    test = retrieval(best_model, Vte, Tte)
    return {
        "slug": slug,
        "llm": llm,
        "vision_dim": int(V_tr.shape[1]),
        "text_dim": int(T_tr.shape[-1]),
        "n_train": int(len(tr)),
        "n_val": int(len(va)),
        "n_test": int(len(te)),
        "selection": best_info,
        "test": test,
        "score": test["mean_recall_1cap"],
        "score_5cap": test["mean_recall_5cap"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", required=True, choices=sorted(common.LLMS))
    ap.add_argument("--train-dataset", default="coco2k", choices=sorted(common.DATASETS))
    ap.add_argument("--test-dataset", default="", choices=[""] + sorted(common.DATASETS),
                    help="empty = same dataset, split by --n-train (in-domain mode)")
    ap.add_argument("--extra-positive", action="store_true",
                    help="SAIL's second positive caption (caption field index 1)")
    ap.add_argument("--slugs", default="", help="comma list; empty = all 70")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--text-pool", default="mean", choices=["mean", "last"])
    # alignment layer / optimisation (SAIL alignment_probing.sh defaults)
    ap.add_argument("--linear-type", default="linear", choices=["linear", "mlp", "star"])
    ap.add_argument("--target-dimension", type=int, default=1024)
    ap.add_argument("--width-factor", type=int, default=8)
    ap.add_argument("--loss", default="siglip", choices=["siglip", "clip"])
    ap.add_argument("--loss-reduction", default="mean",
                    choices=["mean", "sum_over_n"],
                    help="mean = SAIL's form; sum_over_n reproduces the first "
                         "COCO-only runs")
    ap.add_argument("--logit-scale", type=float, default=20.0)
    ap.add_argument("--logit-bias", type=float, default=-10.0)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--optimizer", default="lion", choices=["lion", "adamw"],
                    help="SAIL's default is lion")
    ap.add_argument("--epochs", type=int, default=0,
                    help=">0 overrides --steps: steps = ceil(n_train/batch) * epochs")
    ap.add_argument("--grad-clip-norm", type=float, default=None,
                    help="SAIL leaves this unset (no clipping)")
    ap.add_argument("--select", default="final", choices=["final", "val"],
                    help="final = SAIL's fixed schedule; val = early-stop on a "
                         "held-out slice (a deviation)")
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.99)
    ap.add_argument("--lr-grid", default="1e-3")
    ap.add_argument("--wd-grid", default="1e-4,1e-2,1e-1")
    ap.add_argument("--train-captions", type=int, default=1,
                    help="captions per train image sampled per step (1..5)")
    ap.add_argument("--n-train", type=int, default=1600,
                    help="training images out of the 2000 COCO-2K pairs")
    ap.add_argument("--split-seed", type=int, default=-1,
                    help="-1 = use the fixed order in coco2k.json (reproduces the "
                         "1600/400 baseline); >=0 = reshuffle the 2000 pairs")
    ap.add_argument("--val-frac", type=float, default=0.1,
                    help="fraction of the train split held out for hp/step selection")
    ap.add_argument("--val-size", type=int, default=0,
                    help="absolute val size; overrides --val-frac when > 0")
    ap.add_argument("--tag", default="",
                    help="write under results/sweeps/<tag>/ instead of results/align/")
    ap.add_argument("--train-budget", type=int, default=0,
                    help="cross-dataset mode: use the first N training pairs "
                         "(0 = all). --n-train only applies to in-domain splits.")
    ap.add_argument("--tf32", action="store_true",
                    help="TF32 matmuls; ~8x faster on A100, needed for the large budgets")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    args.lr_grid = [float(x) for x in args.lr_grid.split(",") if x]
    args.wd_grid = [float(x) for x in args.wd_grid.split(",") if x]

    common.ensure_dirs()
    device = args.device
    cross = bool(args.test_dataset) and args.test_dataset != args.train_dataset
    tr_ds = args.train_dataset
    te_ds = args.test_dataset or args.train_dataset

    def load_text(ds: str) -> torch.Tensor:
        p = common.text_cache(ds, args.llm) / f"emb_{args.text_pool}.npy"
        if not p.exists():
            raise SystemExit(f"missing {p} - run "
                             f"encode_text.py --dataset {ds} --llm {args.llm}")
        return torch.from_numpy(np.load(p)).to(device)

    T_tr = load_text(tr_ds)
    T_te = load_text(te_ds) if cross else T_tr

    n_tr_total = len(common.load_dataset(tr_ds)["samples"])
    if cross:
        # every training pair trains; the whole eval dataset is the gallery.
        # --n-train N takes the first N pairs (cross prefix): the datasets are
        # nested, so budgets are subsets of one another.
        budget = args.train_budget if args.train_budget > 0 else n_tr_total
        ids = np.arange(min(budget, n_tr_total))
        n_val = args.val_size if args.val_size > 0 else int(round(args.val_frac * len(ids)))
        n_val = min(max(0, n_val), len(ids) - 1)
        perm = np.random.default_rng(args.seed).permutation(len(ids))
        va = torch.as_tensor(ids[perm[:n_val]], device=device)
        tr = torch.as_tensor(ids[perm[n_val:]], device=device)
        te = torch.arange(len(common.load_dataset(te_ds)["samples"]), device=device)
    else:
        samples = common.load_dataset(tr_ds)["samples"]
        all_ids = np.array([s["idx"] for s in samples])
        if args.split_seed >= 0:
            all_ids = all_ids[np.random.default_rng(args.split_seed)
                              .permutation(len(all_ids))]
        if not 0 < args.n_train < len(all_ids):
            raise SystemExit(f"--n-train must be in (0, {len(all_ids)})")
        train_ids, test_ids = all_ids[: args.n_train], all_ids[args.n_train :]
        n_val = args.val_size if args.val_size > 0 else int(round(args.val_frac
                                                                 * len(train_ids)))
        n_val = max(1, min(n_val, len(train_ids) - 1))
        perm = np.random.default_rng(args.seed).permutation(len(train_ids))
        va = torch.as_tensor(train_ids[perm[:n_val]], device=device)
        tr = torch.as_tensor(train_ids[perm[n_val:]], device=device)
        te = torch.as_tensor(test_ids, device=device)

    slugs = ([s.strip() for s in args.slugs.split(",") if s.strip()]
             if args.slugs else common.read_tokenizer_list())
    slugs = slugs[args.shard :: args.num_shards]

    if args.epochs > 0:
        per_epoch = max(1, math.ceil(len(tr) / min(args.batch_size, len(tr))))
        args.steps = per_epoch * args.epochs
        print(f"  --epochs {args.epochs} x {per_epoch} batch(es)/epoch "
              f"-> {args.steps} steps", flush=True)

    out_dir = (common.RESULTS_DIR / "sweeps" / args.tag / args.llm if args.tag
               else common.RESULTS_DIR / "align" / args.llm)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{args.llm}{'/' + args.tag if args.tag else ''}: {len(slugs)} tokenizers | "
          f"train={tr_ds}({len(tr)}) val={len(va)} test={te_ds}({len(te)}) | "
          f"extra_positive={args.extra_positive} | "
          f"grid={len(args.lr_grid)*len(args.wd_grid)}", flush=True)

    bar = tqdm(slugs, desc=f"{args.llm} combos", unit="combo")
    for slug in bar:
        bar.set_postfix_str(slug)
        out_json = out_dir / f"{slug}.json"
        if out_json.exists() and not args.overwrite:
            tqdm.write(f"[skip] {args.llm}/{slug}")
            continue
        vis_tr = common.vision_cache(tr_ds, slug) / "emb.npy"
        vis_te = common.vision_cache(te_ds, slug) / "emb.npy"
        missing = [str(p) for p in ({vis_tr, vis_te}) if not p.exists()]
        if missing:
            tqdm.write(f"[miss] {args.llm}/{slug}: no vision cache ({missing[0]})")
            continue
        t0 = time.time()
        try:
            V_tr = torch.from_numpy(np.load(vis_tr)).to(device)
            V_te = torch.from_numpy(np.load(vis_te)).to(device) if cross else V_tr
            res = run_combo(slug, args.llm, (V_tr, T_tr), (V_te, T_te),
                            (tr, va, te), args, device)
            res["seconds"] = round(time.time() - t0, 1)
            res["split"] = {"train_dataset": tr_ds, "test_dataset": te_ds,
                            "cross_dataset": cross, "n_train": args.n_train,
                            "split_seed": args.split_seed, "val_frac": args.val_frac,
                            "tag": args.tag, "n_gallery": int(len(te)),
                            "extra_positive": bool(args.extra_positive)}
            res["config"] = {k: v for k, v in vars(args).items()
                             if k not in ("slugs", "shard", "num_shards", "overwrite")}
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)
            tqdm.write(
                f"[ok]   {args.llm}/{slug}  MR(1cap)={res['score']:.2f} "
                f"MR(5cap)={res['score_5cap']:.2f} "
                f"i2t@1={res['test']['i2t_1cap']['R@1']:.1f} "
                f"t2i@1={res['test']['t2i_1cap']['R@1']:.1f} ({res['seconds']}s)"
            )
            del V_tr, V_te
            torch.cuda.empty_cache()
        except Exception as exc:
            tqdm.write(f"[FAIL] {args.llm}/{slug}: {exc}")
            traceback.print_exc()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
