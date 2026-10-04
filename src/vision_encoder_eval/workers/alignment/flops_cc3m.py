#!/usr/bin/env python3
"""FLOPs of the SAIL-protocol baseline (train CC3M-2K -> eval COCO-2K).

Everything is measured or derived from what actually ran:
  * vision encoding: measured FLOPs/image (results/flops_vision.json) x
    (2000 CC3M + 2000 COCO) images
  * text encoding: measured totals for CC3M (raw + longSV, max_len 128) and
    COCO (5 captions, max_len 64)
  * alignment layer: analytic, from the config stored in every
    results/sweeps/<tag>/<llm>/<slug>.json (2xMAC convention, fwd+bwd = 3x fwd)
"""

from __future__ import annotations

import argparse
import json

import numpy as np

import vision_encoder_eval.workers.alignment.common as common
from vision_encoder_eval.workers.alignment.flops import human


def head_flops(res: dict) -> dict:
    c = res["config"]
    if c["linear_type"] != "linear":
        raise SystemExit("only linear heads are modelled")
    dv, dt, d = res["vision_dim"], res["text_dim"], c["target_dimension"]
    n_tr, n_val, n_te = res["n_train"], res["n_val"], res["n_test"]
    grid = len(c["lr_grid"]) * len(c["wd_grid"])
    steps = c["steps"]
    bs = min(c["batch_size"], n_tr)
    n_pos = 2 if c.get("extra_positive") else 1
    f_img, f_txt = 2 * dv * d, 2 * dt * d

    # per step: bs images, bs*n_pos captions through the heads, n_pos bs x bs logit blocks
    step = 3 * (bs * f_img + n_pos * bs * f_txt) + 3 * n_pos * 2 * bs * bs * d
    train = grid * steps * step

    val = 0
    if c.get("select") == "val" and n_val > 0:
        n_eval = steps // c["eval_every"]
        val = grid * n_eval * (n_val * f_img + n_val * f_txt + 2 * n_val * n_val * d)

    caps = 5
    test = n_te * f_img + n_te * caps * f_txt + 2 * n_te * (n_te * caps) * d
    return {"train": train, "val": val, "test": test}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="cc3m2k")
    ap.add_argument("--include-eval", action="store_true")
    args = ap.parse_args()

    slugs = common.read_tokenizer_list()
    vis = {r["slug"]: r for r in json.load(open(common.RESULTS_DIR / "flops_vision.json"))}
    txt_coco = sum(r["flops_total"] for r in json.load(open(common.RESULTS_DIR / "flops_text.json")))
    txt_cc3m = sum(r["flops_total"] for r in
                   json.load(open(common.RESULTS_DIR / "flops_text_cc3m2k.json")))

    n_cc3m = len(common.load_dataset("cc3m2k")["samples"])
    n_coco = len(common.load_dataset("coco2k")["samples"])

    root = common.RESULTS_DIR / "sweeps" / args.tag
    enc, train, val, test = {}, {}, {}, {}
    for s in slugs:
        enc[s] = vis[s]["flops_per_image"] * (n_cc3m + n_coco)
        train[s] = val[s] = test[s] = 0
        for l in common.LLMS:
            r = json.load(open(root / l / f"{s}.json"))
            h = head_flops(r)
            train[s] += h["train"]; val[s] += h["val"]; test[s] += h["test"]

    text_each = (txt_coco + txt_cc3m) / len(slugs)
    E = np.array([enc[s] for s in slugs], float)
    Tr = np.array([train[s] for s in slugs], float)
    Va = np.array([val[s] for s in slugs], float)
    Te = np.array([test[s] for s in slugs], float)
    tot = E + Tr + text_each + (Va + Te if args.include_eval else 0)

    r0 = json.load(open(root / "qwen3" / f"{slugs[0]}.json"))
    c = r0["config"]
    print(f"=== FLOPs: SAIL baseline, {args.tag} "
          f"({'incl.' if args.include_eval else 'excl.'} eval) ===")
    print(f"train {r0['n_train']} CC3M (val {r0['n_val']}), test {r0['n_test']} COCO | "
          f"{c['optimizer']} lr={c['lr_grid']} steps={c['steps']} d={c['target_dimension']} "
          f"extra_positive={c.get('extra_positive')}\n")
    print(f"{'component':40s}{'mean/tokenizer':>15s}{'x70':>11s}")
    rows = [("vision encoding (2000 CC3M + 2000 COCO)", E),
            ("alignment-layer training (3 LLMs)", Tr),
            ("text encoding (amortised; measured)", np.full(len(slugs), text_each))]
    if args.include_eval:
        rows += [("validation retrieval", Va), ("COCO test retrieval", Te)]
    for name, a in rows:
        print(f"{name:40s}{human(a.mean()):>15s}{human(a.sum()):>11s}")
    print(f"{'TOTAL':40s}{human(tot.mean()):>15s}{human(tot.sum()):>11s}")
    print(f"\nmedian {human(np.median(tot))}   min {human(tot.min())} "
          f"({slugs[int(tot.argmin())]})   max {human(tot.max())} "
          f"({slugs[int(tot.argmax())]})")
    print(f"MACs convention (half): mean {human(tot.mean()/2)}")
    print(f"not counted: validation = {human(Va.sum())} (no val split in this config), "
          f"COCO test retrieval = {human(Te.mean())}/tokenizer")

    out = root / f"flops{'_with_eval' if args.include_eval else ''}.json"
    json.dump({"tag": args.tag, "include_eval": args.include_eval,
               "mean_per_tokenizer": float(tot.mean()),
               "median_per_tokenizer": float(np.median(tot)),
               "total": float(tot.sum()),
               "mean_vision_encoding": float(E.mean()),
               "mean_alignment_training": float(Tr.mean()),
               "text_amortised_each": float(text_each),
               "mean_test_eval": float(Te.mean()),
               "per_tokenizer": {s: {"vision_encoding": enc[s], "alignment_training": train[s],
                                     "test_eval": test[s]} for s in slugs}},
              open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
