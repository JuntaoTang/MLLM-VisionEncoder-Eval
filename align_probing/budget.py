#!/usr/bin/env python3
"""FLOP budget calculator for the SAIL-style CC3M-train / COCO-eval design.

Vision and text costs come from the measured tables (results/flops_vision.json,
results/flops_text.json); the alignment layer is modelled analytically with the
same 2xMAC convention.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

import common


def human(x: float) -> str:
    for u in ("", "K", "M", "G", "T", "P", "E"):
        if abs(x) < 1000:
            return f"{x:.4g}{u}"
        x /= 1000
    return f"{x:.4g}Z"


def build(args) -> dict:
    vis = {r["slug"]: r for r in json.load(open(common.RESULTS_DIR / "flops_vision.json"))}
    txt = {r["llm"]: r for r in json.load(open(common.RESULTS_DIR / "flops_text.json"))}
    slugs = common.read_tokenizer_list()
    d = args.target_dimension

    # measured FLOPs per text token, per LLM (attention is sub-dominant at these lengths)
    tok_cost = {l: txt[l]["flops_total"] / (txt[l]["n_captions"] * txt[l]["mean_padded_len"])
                for l in txt}
    text_dims, vis_dims = {}, {}
    for l in common.LLMS:
        text_dims[l] = int(json.load(open(common.text_cache("coco2k", l) / "meta.json"))["dim"])
    for s in slugs:
        vis_dims[s] = int(json.load(open(common.vision_cache("coco2k", s) / "meta.json"))["dim"])

    n_val = int(round(args.val_frac * args.n_train))
    n_tr = args.n_train - n_val
    bs = min(args.batch_size, n_tr)
    n_eval = max(1, args.steps // args.eval_every)
    n_pos = 2 if args.extra_positive else 1     # main caption + optional extra positive

    # ---- text encoding: CC3M train captions (+ COCO test captions if not cached) ----
    cc3m_tokens = args.raw_len + (args.extra_len if args.extra_positive else 0)
    n_tok = args.n_train * cc3m_tokens
    if not args.reuse_coco_cache:
        n_tok += args.n_test * args.test_captions * args.coco_len
    text_total = sum(tok_cost[l] * n_tok for l in common.LLMS)
    text_each = text_total / len(slugs)

    n_images = args.n_train + (0 if args.reuse_coco_cache else args.n_test)

    enc, ali = [], []
    for s in slugs:
        enc.append(vis[s]["flops_per_image"] * n_images)
        f_img = 2 * vis_dims[s] * d
        a = 0
        for l in common.LLMS:
            f_txt = 2 * text_dims[l] * d
            # training step: fwd+bwd (3x) over bs images and bs*n_pos captions,
            # plus n_pos similarity matrices of bs x bs x d
            step = 3 * bs * (f_img + n_pos * f_txt) + 3 * n_pos * 2 * bs * bs * d
            a += args.grid * args.steps * step
            # val retrieval (forward only), n_pos=1 caption per val image
            a += args.grid * n_eval * (n_val * f_img + n_val * f_txt
                                       + 2 * n_val * n_val * d)
            # final test on the COCO gallery
            a += (args.n_test * f_img + args.n_test * args.test_captions * f_txt
                  + 2 * args.n_test * (args.n_test * args.test_captions) * d)
        ali.append(a)

    enc = np.array(enc, float)
    ali = np.array(ali, float)
    tot = enc + ali + text_each
    return {"slugs": slugs, "enc": enc, "ali": ali, "text_each": text_each,
            "text_total": text_total, "tot": tot, "bs": bs, "n_tr": n_tr,
            "n_val": n_val, "n_images": n_images, "n_tok": n_tok}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--test-captions", type=int, default=5)
    ap.add_argument("--extra-positive", action="store_true",
                    help="SAIL's second positive caption (longSV)")
    ap.add_argument("--raw-len", type=float, default=10.8, help="tokens, CC3M raw_caption")
    ap.add_argument("--extra-len", type=float, default=188.3, help="tokens, longSV")
    ap.add_argument("--coco-len", type=float, default=25.6)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch-size", type=int, default=32768)
    ap.add_argument("--grid", type=int, default=3)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--target-dimension", type=int, default=2048)
    ap.add_argument("--reuse-coco-cache", action="store_true")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    b = build(args)
    tot, enc, ali = b["tot"], b["enc"], b["ali"]
    print(f"=== {args.label or 'budget'}: {args.n_train} CC3M train / {args.n_test} COCO test ===")
    print(f"train {b['n_tr']} + val {b['n_val']} | batch {b['bs']} | {args.steps} steps "
          f"x {args.grid} grid | d={args.target_dimension} | "
          f"extra_positive={args.extra_positive}")
    print(f"{b['n_images']} images/tokenizer, {b['n_tok']:,} text tokens/LLM\n")
    print(f"{'component':32s}{'mean/tokenizer':>16s}{'x70':>12s}")
    print(f"{'vision encoding':32s}{human(enc.mean()):>16s}{human(enc.sum()):>12s}")
    print(f"{'alignment (3 LLMs)':32s}{human(ali.mean()):>16s}{human(ali.sum()):>12s}")
    print(f"{'text encoding (amortised)':32s}{human(b['text_each']):>16s}{human(b['text_total']):>12s}")
    print(f"{'TOTAL':32s}{human(tot.mean()):>16s}{human(tot.sum()):>12s}")
    print(f"median {human(np.median(tot))}  max {human(tot.max())} "
          f"({b['slugs'][int(tot.argmax())]})")


if __name__ == "__main__":
    main()
