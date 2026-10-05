# -*- coding: utf-8 -*-
"""
make_cm_final.py  (derive the converged cross-modal CSV)
=========================================================
Reads the raw cross-modal CSV written by compute_crossmodal_stats.py and keeps
the six columns used downstream:

    cm_cka, cm_r2   x   {full, lmu, ocr}

The two scalar columns of the descriptor are cm_cka and cm_r2 on the
full image set; the LMU/OCR variants are subsets, and cm_r2_alpha / cm_margin
are diagnostics, not features.

Usage (from the CKA-X/ root):
    # Qwen2.5 encoder (the default text encoder, see README §10)
    python scripts/make_cm_final.py \
        --src results/reference/crossmodal_stats_qwen25_full.csv \
        --out_name crossmodal_stats_final_qwen25_full
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vision_encoder_eval.workers.ckax.scripts.ckax_common import RESULTS_DIR  # noqa: E402

KEEP = ["cm_cka", "cm_r2", "cm_cka_lmu", "cm_r2_lmu", "cm_cka_ocr", "cm_r2_ocr"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str,
                    default=os.path.join(RESULTS_DIR, "crossmodal_stats_qwen.csv"),
                    help="8-col cross-modal CSV from compute_crossmodal_stats.py")
    ap.add_argument("--out_name", type=str, default="crossmodal_stats_final",
                    help="output 6-col CSV basename under results/reference/ "
                         "(e.g. crossmodal_stats_final_qwen25)")
    args = ap.parse_args()

    out = os.path.join(RESULTS_DIR, args.out_name + ".csv")
    if not os.path.isfile(args.src):
        raise SystemExit(f"  [ERROR] {args.src} not found; "
                         f"run compute_crossmodal_stats.py first")
    with open(args.src, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        header = [h for h in rd.fieldnames if h in KEEP]
        rows = [(r["tokenizer"], [r[h] for h in header]) for r in rd]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["tokenizer"] + header)
        for tok, vals in rows:
            w.writerow([tok] + vals)
    print(f"  wrote {out} ({len(header)} cols: " + ", ".join(header) + ")")


if __name__ == "__main__":
    main()
