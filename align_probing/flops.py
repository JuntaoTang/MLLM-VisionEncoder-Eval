#!/usr/bin/env python3
"""FLOP accounting for the COCO-2K alignment-probing experiment.

Stage 1 (measure): run one forward per frozen backbone under torch's
FlopCounterMode to get FLOPs/image for each of the 70 visual tokenizers and
FLOPs/batch for each of the 3 LLMs at the caption lengths actually used.

Stage 2 (report): multiply by the real workload counts, and add the alignment
layer's train/eval cost, which is computed exactly from the shapes recorded in
every results/**/<slug>.json.
"""

from __future__ import annotations

import argparse
import json
import traceback

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode
from tqdm import tqdm

import common

common.bootstrap()

import encode_vision as ev  # noqa: E402  (bootstraps VTB paths)

VISION_JSON = common.RESULTS_DIR / "flops_vision.json"
TEXT_JSON = common.RESULTS_DIR / "flops_text.json"


# --------------------------------------------------------------------------- #
# stage 1a: vision towers
# --------------------------------------------------------------------------- #
@torch.no_grad()
def measure_vision(slug: str, device: str) -> dict:
    data = common.load_dataset("coco2k")
    sample = data["samples"][:1]
    mode, cfg_path = common.resolve_slug(slug)
    cfg = common.load_cfg(cfg_path)

    if mode == "continuous":
        vision = cfg["vision_encoder"]
        tower = common.build_continuous_tower(vision, device)
        ds = ev.ContinuousImages(sample, tower.image_processor)
        model, extractor = tower, (lambda px: tower(px))
    else:
        tok, vis_mode = common.build_discrete_tokenizer(cfg, device)
        ds = ev.DiscreteImages(sample, int(tok.image_size))
        model, extractor = tok, (lambda px: ev._discrete_features(tok, vis_mode, px))

    px = ds[0][0].unsqueeze(0).to(device)
    dtype = next(model.parameters()).dtype
    if dtype in (torch.float16, torch.bfloat16):
        px = px.to(dtype)

    counter = FlopCounterMode(display=False)
    with counter:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=dtype == torch.float32):
            extractor(px)
    flops = int(counter.get_total_flops())
    params = int(sum(p.numel() for p in model.parameters()))
    del model
    torch.cuda.empty_cache()
    return {"slug": slug, "mode": mode, "flops_per_image": flops, "params": params,
            "input_hw": list(px.shape[-2:])}


# --------------------------------------------------------------------------- #
# stage 1b: LLM text encoders (exact per-batch, matching encode_text.py)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def measure_text(llm: str, device: str, batch: int, max_length: int,
                 dataset: str = "coco2k") -> dict:
    from transformers import AutoModel, AutoTokenizer

    path = common.LLMS[llm]["path"]
    data = common.load_dataset(dataset)
    texts = [c for s in data["samples"] for c in s["captions"][:data["captions_per_image"]]]

    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModel.from_pretrained(path, torch_dtype=torch.bfloat16,
                                      trust_remote_code=True).to(device).eval()

    total, shapes = 0, []
    for i in tqdm(range(0, len(texts), batch), desc=f"{llm} batches", unit="batch",
                  leave=False):
        enc = tok(texts[i : i + batch], return_tensors="pt", padding=True,
                  truncation=True, max_length=max_length)
        enc = {k: v.to(device) for k, v in enc.items()}
        counter = FlopCounterMode(display=False)
        with counter:
            model(**enc, use_cache=False)
        total += int(counter.get_total_flops())
        shapes.append(list(enc["input_ids"].shape))

    params = int(sum(p.numel() for p in model.parameters()))
    del model
    torch.cuda.empty_cache()
    return {"llm": llm, "dataset": dataset, "max_length": max_length,
            "flops_total": total, "n_captions": len(texts),
            "params": params, "n_batches": len(shapes),
            "mean_padded_len": float(np.mean([s[1] for s in shapes]))}


# --------------------------------------------------------------------------- #
# stage 2: alignment layer, computed exactly from the recorded shapes
# --------------------------------------------------------------------------- #
def align_flops(res: dict) -> dict:
    """FLOPs for one (tokenizer, LLM) alignment run, from its stored config.

    The head is LayerNorm + Linear on each side; only the two GEMMs matter
    (2*D_in*D_out FLOPs per sample per side). Training counts fwd+bwd as 3x
    forward. Retrieval evals are forward-only, plus the similarity matmuls.
    """
    c = res["config"]
    dv, dt, d = res["vision_dim"], res["text_dim"], c["target_dimension"]
    if c["linear_type"] != "linear":
        raise SystemExit(f"align_flops only models linear_type=linear, got {c['linear_type']}")
    n_grid = len(c["lr_grid"]) * len(c["wd_grid"])
    steps, bs = c["steps"], min(c["batch_size"], res["n_train"])
    caps = 5  # captions per image, both eval protocols use all five

    f_img = 2 * dv * d          # per image, forward
    f_txt = 2 * dt * d          # per caption, forward

    # training: fwd+bwd ~= 3x fwd, on bs images + bs captions per step
    train = n_grid * steps * 3 * bs * (f_img + f_txt)
    # contrastive logits per step: bs x bs x d, fwd+bwd
    train += n_grid * steps * 3 * 2 * bs * bs * d

    # validation: one retrieval pass every eval_every steps
    n_eval = steps // c["eval_every"]
    nv = res["n_val"]
    val = n_grid * n_eval * (nv * f_img + nv * caps * f_txt
                             + 2 * nv * (nv * caps) * d)
    # test: one pass with the selected checkpoint
    nt = res["n_test"]
    test = nt * f_img + nt * caps * f_txt + 2 * nt * (nt * caps) * d

    return {"train": train, "val": val, "test": test, "total": train + val + test}


def human(x: float) -> str:
    for unit in ("", "K", "M", "G", "T", "P", "E"):
        if abs(x) < 1000:
            return f"{x:.3g}{unit}"
        x /= 1000
    return f"{x:.3g}Z"


def report(*, include_eval: bool = False, scope: str = "all") -> None:
    """Per-tokenizer and total FLOPs.

    scope="base" counts only the original 210-combo run; "all" adds the nine
    robustness variants (2100 runs). include_eval=False drops the retrieval
    passes (validation selection + final test) and keeps encoding + alignment
    training only.
    """
    vis = {r["slug"]: r for r in json.load(open(VISION_JSON, encoding="utf-8"))}
    txt = {r["llm"]: r for r in json.load(open(TEXT_JSON, encoding="utf-8"))}
    data = common.load_dataset("coco2k")
    n_img = data["n"]

    slugs = common.read_tokenizer_list()
    missing = [s for s in slugs if s not in vis]
    if missing:
        raise SystemExit(f"missing vision FLOPs for {len(missing)}: {missing[:8]}")

    roots = [("base", common.RESULTS_DIR / "align")]
    sweeps = common.RESULTS_DIR / "sweeps"
    if scope == "all" and sweeps.is_dir():
        roots += [(d.name, d) for d in sorted(sweeps.iterdir()) if d.is_dir()]

    parts = ("train", "val", "test") if include_eval else ("train",)
    align = {s: 0 for s in slugs}
    n_runs = 0
    for _, root in roots:
        for llm in common.LLMS:
            for p in (root / llm).glob("*.json"):
                with open(p, encoding="utf-8") as f:
                    fl = align_flops(json.load(f))
                align[p.stem] += sum(fl[k] for k in parts)
                n_runs += 1

    enc = {s: vis[s]["flops_per_image"] * n_img for s in slugs}
    text_total = sum(txt[l]["flops_total"] for l in common.LLMS)
    text_each = text_total / len(slugs)          # shared cost, amortised
    per_tok = {s: enc[s] + align[s] + text_each for s in slugs}

    # float64: the int64 totals overflow once the sweep variants are included
    vals = np.array([per_tok[s] for s in slugs], dtype=float)
    ev = np.array([enc[s] for s in slugs], dtype=float)
    av = np.array([align[s] for s in slugs], dtype=float)

    label = "encoding + alignment training" if not include_eval else "everything"
    print(f"=== FLOPs per tokenizer — {label} ===")
    print(f"scope: {scope} ({n_runs} alignment runs over {len(roots)} split variant(s), "
          f"{len(common.LLMS)} LLMs each)")
    print(f"{n_img} images/tokenizer, {len(slugs)} tokenizers\n")

    print(f"{'component':34s}{'mean/tokenizer':>16s}{'x70 total':>12s}{'share':>8s}")
    rows = (("vision pre-encoding", ev, float(ev.sum())),
            ("alignment training (3 LLMs)", av, float(av.sum())),
            ("text pre-encoding (amortised)",
             np.full(len(slugs), text_each), float(text_total)))
    for name, arr, tot in rows:
        print(f"{name:34s}{human(arr.mean()):>16s}{human(tot):>12s}"
              f"{100*tot/vals.sum():7.1f}%")
    print(f"{'AVERAGE PER TOKENIZER':34s}{human(vals.mean()):>16s}"
          f"{human(vals.sum()):>12s}{100:7.1f}%")

    print(f"\nspread across the 70 tokenizers:")
    print(f"  median {human(np.median(vals))}   min {human(vals.min())} "
          f"({slugs[int(vals.argmin())]})   max {human(vals.max())} "
          f"({slugs[int(vals.argmax())]})")
    print(f"  vision-encoding share of the mean: {100*ev.mean()/vals.mean():.1f}%")

    out = common.RESULTS_DIR / (
        f"flops_per_tokenizer_{scope}{'_with_eval' if include_eval else ''}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"scope": scope, "include_eval": include_eval,
                   "n_images": n_img, "n_alignment_runs": n_runs,
                   "mean_per_tokenizer": float(vals.mean()),
                   "median_per_tokenizer": float(np.median(vals)),
                   "total": float(vals.sum()),
                   "mean_vision_encoding": float(ev.mean()),
                   "mean_alignment_training": float(av.mean()),
                   "text_amortised_each": float(text_each),
                   "per_tokenizer": {s: {"vision_encoding": enc[s],
                                         "alignment_training": align[s],
                                         "total": per_tok[s]} for s in slugs}},
                  f, indent=2)
    print(f"\nwrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="report",
                    choices=["vision", "text", "report", "all"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--slugs", default="")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--text-batch", type=int, default=128)
    ap.add_argument("--text-dataset", default="coco2k", choices=sorted(common.DATASETS))
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--scope", default="all", choices=["base", "all"],
                    help="base = original 210 runs; all = + the 9 sweep variants")
    ap.add_argument("--include-eval", action="store_true",
                    help="also count the val/test retrieval passes")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    common.ensure_dirs()

    if args.stage in ("vision", "all"):
        slugs = ([s.strip() for s in args.slugs.split(",") if s.strip()]
                 if args.slugs else common.read_tokenizer_list())
        slugs = slugs[args.shard :: args.num_shards]
        out_path = (common.RESULTS_DIR / f"flops_vision_shard{args.shard}.json"
                    if args.num_shards > 1 else VISION_JSON)
        done = {}
        if out_path.exists() and not args.overwrite:
            done = {r["slug"]: r for r in json.load(open(out_path, encoding="utf-8"))}
        bar = tqdm(slugs, desc=f"vision FLOPs shard{args.shard}", unit="tok")
        for slug in bar:
            bar.set_postfix_str(slug)
            if slug in done:
                continue
            try:
                r = measure_vision(slug, args.device)
                done[slug] = r
                tqdm.write(f"[ok]   {slug:28s} {human(r['flops_per_image']):>9s}/img "
                           f"params={human(r['params'])}")
            except Exception as exc:
                tqdm.write(f"[FAIL] {slug}: {exc}")
                traceback.print_exc()
            finally:
                torch.cuda.empty_cache()
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(list(done.values()), f, indent=2)

    if args.stage in ("text", "all"):
        text_json = (TEXT_JSON if args.text_dataset == "coco2k"
                     else common.RESULTS_DIR / f"flops_text_{args.text_dataset}.json")
        done = {}
        if text_json.exists() and not args.overwrite:
            done = {r["llm"]: r for r in json.load(open(text_json, encoding="utf-8"))}
        for llm in tqdm(sorted(common.LLMS), desc="text FLOPs", unit="llm"):
            if llm in done:
                continue
            r = measure_text(llm, args.device, args.text_batch, args.max_length,
                             args.text_dataset)
            done[llm] = r
            tqdm.write(f"[ok]   {llm:10s} {human(r['flops_total'])} total "
                       f"({r['n_batches']} batches, mean len "
                       f"{r['mean_padded_len']:.1f})")
            with open(text_json, "w", encoding="utf-8") as f:
                json.dump(list(done.values()), f, indent=2)

    if args.stage in ("report", "all"):
        report(include_eval=args.include_eval, scope=args.scope)


if __name__ == "__main__":
    main()
