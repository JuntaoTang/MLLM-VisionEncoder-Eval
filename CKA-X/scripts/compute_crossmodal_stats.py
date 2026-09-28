# -*- coding: utf-8 -*-
"""
compute_crossmodal_stats.py
=============================
Per-tokenizer cross-modal metrics between image features and CLIP text features
on the shared calibration images.

Metrics (one row per tokenizer, saved as crossmodal_stats.csv):
  cm_cka        linear CKA(visual features, text features)
  cm_r2         ridge R-squared: predict text from visual features
  cm_margin     alignment margin: cos(pred, true) - cos(pred, random)

  cm_cka_lmu    subset: LMU images only
  cm_r2_lmu     subset: LMU images only
  cm_cka_ocr    subset: OCR-VQA images only
  cm_r2_ocr     subset: OCR-VQA images only

All CKA computed on same random subset of n_cka images for consistency.
ridge R^2 computed on full set (half-fit, half-eval split).

Output: results/reference/crossmodal_stats_<out_name>.csv (one row per
        tokenizer in features_diverse); run.sh uses --out_name
        crossmodal_stats_qwen25_full, which make_cm_final.py then reduces to
        artifacts/crossmodal_stats_final_qwen25_full.csv

Usage (from the CKA-X/ root):
    python scripts/compute_crossmodal_stats.py --device cuda
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ckax_common import resolve_data_path, ensure_results_dir
from compute_cka_kernel import linear_cka_ssx

SEED = 42
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
# Subsampled CKA columns: fraction of the rows, so the default scales with
# whatever feature set is handed to the script (--full_cka uses every row).
N_CKA_FRACTION = 0.2


def centered_gram(X):
    Xc = X.float() - X.float().mean(0)
    return Xc @ Xc.T


def cka_from_grams(K, L):
    den = torch.sqrt((K * K).sum() * (L * L).sum()) + 1e-12
    return float((K * L).sum() / den)


def ridge_loocv(X, Y, alphas, device):
    X = X.to(device).float()
    Y = Y.to(device).float()
    mu = X.mean(0)
    sd = X.std(0).clamp(min=1e-6)
    Xn = (X - mu) / sd
    G = Xn.T @ Xn
    lam, V = torch.linalg.eigh(G)
    lam = lam.clamp(min=0)
    B = Xn @ V
    C = V.T @ (Xn.T @ Y)
    best_a, best_sse = alphas[0], float("inf")
    for a in alphas:
        w = 1.0 / (lam + a)
        Yhat = (B * w) @ C
        h = ((B * B) @ w).clamp(max=0.9999)
        loo = (Yhat - h.unsqueeze(1) * Y) / (1 - h).unsqueeze(1)
        sse = float(((loo - Y) ** 2).sum())
        if sse < best_sse:
            best_a, best_sse = a, sse
    w = 1.0 / (lam + best_a)
    W = V @ (torch.diag(w) @ C)
    return Xn @ W, best_a


def compute_metrics(vis_feat, txt_feat, device):
    n = vis_feat.shape[0]
    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(n, generator=g)
    n_fit = n // 2
    # eval must be same length as fit; when n is odd drop the last sample
    fi, ei = perm[:n_fit], perm[n_fit:2 * n_fit]
    pred_eval, best_a = ridge_loocv(vis_feat[fi], txt_feat[fi], ALPHAS, device)
    Ye = txt_feat[ei].to(device)
    pn = F.normalize(pred_eval.float(), dim=-1)
    yn = F.normalize(Ye.float(), dim=-1)
    cos_true = (pn * yn).sum(1)
    S = pn @ yn.T
    m = pn.shape[0]
    cos_rand = float((S.sum() - cos_true.sum()) / max(m * (m - 1), 1))
    margin = float(cos_true.mean()) - cos_rand
    ss_res = float(((Ye - pred_eval) ** 2).sum())
    ss_tot = float(((Ye - Ye.mean(0)) ** 2).sum()) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    return r2, margin, best_a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diverse_dir", type=str, default=None)
    ap.add_argument("--text_features_pt", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--n_cka", type=int, default=None,
                    help="images per encoder for the (subsampled) CKA "
                         "columns; default = 20%% of the rows")
    ap.add_argument("--full_cka", action="store_true",
                    help="compute cm_cka on ALL images via the ||Xc'Yc||_F^2 "
                         "identity (no image subsample). cm_r2 already "
                         "uses all images (half-split); this makes the CKA "
                         "columns subsample-free too. Slower (full data) but "
                         "exact.")
    ap.add_argument("--out_name", type=str, default="crossmodal_stats",
                    help="output csv basename (without .csv) under "
                         "results/reference/. Default = crossmodal_stats; "
                         "use e.g. crossmodal_stats_qwen for Qwen3 encoder.")
    args = ap.parse_args()

    if args.diverse_dir is None:
        args.diverse_dir = resolve_data_path("features_diverse")
    if args.text_features_pt is None:
        sample_dir = resolve_data_path(os.path.join("sample_data"))
        args.text_features_pt = os.path.join(sample_dir, "text_features", "text_features.pt")

    diverse_dir = Path(args.diverse_dir)
    device = args.device if torch.cuda.is_available() else "cpu"

    toks = sorted(td.name for td in diverse_dir.iterdir()
                  if td.is_dir() and (td / "visual_features.pt").exists())
    print(f"  Tokenizers in features_diverse: {len(toks)}")

    print(f"  Loading text features from {args.text_features_pt}")
    if not os.path.isfile(args.text_features_pt):
        raise SystemExit(f"  [ERROR] text_features.pt not found. Run extract_text_features.py first.")
    T = torch.load(args.text_features_pt, map_location="cpu", weights_only=False).float()
    print(f"  Text features: {T.shape}")

    ref_tok = toks[0]
    ref_paths_file = diverse_dir / ref_tok / "image_paths.txt"
    img_names = [os.path.basename(l.strip()) for l in open(str(ref_paths_file)).readlines() if l.strip()]
    is_lmu = torch.tensor([1 if n.startswith("lmu_") else 0 for n in img_names])
    is_ocr = 1 - is_lmu
    lmu_idx = torch.where(is_lmu == 1)[0]
    ocr_idx = torch.where(is_ocr == 1)[0]
    print(f"  LMU images: {len(lmu_idx)}, OCR-VQA images: {len(ocr_idx)}")

    if args.full_cka:
        print("  cm_cka on ALL images (full_cka); skipping subsample grams")
    else:
        n_cka = (min(args.n_cka, T.shape[0]) if args.n_cka
                 else max(3, int(round(N_CKA_FRACTION * T.shape[0]))))
        g = torch.Generator().manual_seed(SEED)
        sub = torch.randperm(T.shape[0], generator=g)[:n_cka].sort().values
        Tsub = T[sub]

        print("  Computing text Gram matrix...")
        Ktxt_full = centered_gram(Tsub.to(device)).cpu()
        print("  done")

        n_cka_lmu = min(n_cka // 2, len(lmu_idx))
        n_cka_ocr = min(n_cka // 2, len(ocr_idx))
        sub_lmu = lmu_idx[torch.randperm(len(lmu_idx), generator=g)[:n_cka_lmu]].sort().values
        sub_ocr = ocr_idx[torch.randperm(len(ocr_idx), generator=g)[:n_cka_ocr]].sort().values
        Tsub_lmu = T[sub_lmu]
        Tsub_ocr = T[sub_ocr]
        Klmu_txt = centered_gram(Tsub_lmu.to(device)).cpu()
        Kocr_txt = centered_gram(Tsub_ocr.to(device)).cpu()
        print(f"  CKA subset sizes: full={n_cka}, lmu={n_cka_lmu}, ocr={n_cka_ocr}")

    out_dir = ensure_results_dir()
    out_csv = os.path.join(out_dir, args.out_name + ".csv")
    csv_path = out_csv
    with open(csv_path, "w", newline="", encoding="utf-8") as fout:
        w = csv.writer(fout, lineterminator="\n")
        w.writerow(["tokenizer", "cm_cka", "cm_r2", "cm_margin",
                     "cm_cka_lmu", "cm_r2_lmu",
                     "cm_cka_ocr", "cm_r2_ocr",
                     "cm_r2_alpha"])

        for tok in tqdm(toks, desc="  Tokenizers", unit="tok"):
            vf_pt = diverse_dir / tok / "visual_features.pt"
            V = torch.load(str(vf_pt), map_location="cpu", weights_only=False).float()
            assert V.shape[0] == T.shape[0], f"row mismatch: {tok} {V.shape[0]} vs text {T.shape[0]}"

            if args.full_cka:
                # Exact CKA on ALL images (no subsample), via ||Xc'Yc||_F^2.
                Vd = V.to(device)
                Td = T.to(device)
                cm_cka = linear_cka_ssx(Vd, Td)
                Vl, Tl = V[lmu_idx].to(device), T[lmu_idx].to(device)
                cm_cka_lmu = linear_cka_ssx(Vl, Tl)
                Vo, To = V[ocr_idx].to(device), T[ocr_idx].to(device)
                cm_cka_ocr = linear_cka_ssx(Vo, To)
                del Vd, Td, Vl, Tl, Vo, To
            else:
                Vsub = V[sub]
                Kvis_full = centered_gram(Vsub.to(device)).cpu()
                cm_cka = cka_from_grams(Kvis_full, Ktxt_full)
                del Vsub, Kvis_full

                Vsub_lmu = V[sub_lmu]
                Klmu_vis = centered_gram(Vsub_lmu.to(device)).cpu()
                cm_cka_lmu = cka_from_grams(Klmu_vis, Klmu_txt)
                del Vsub_lmu, Klmu_vis

                Vsub_ocr = V[sub_ocr]
                Kocr_vis = centered_gram(Vsub_ocr.to(device)).cpu()
                cm_cka_ocr = cka_from_grams(Kocr_vis, Kocr_txt)
                del Vsub_ocr, Kocr_vis

            cm_r2, cm_margin, best_a = compute_metrics(V, T, device)
            cm_r2_lmu, _, _ = compute_metrics(V[lmu_idx], T[lmu_idx], device)
            cm_r2_ocr, _, _ = compute_metrics(V[ocr_idx], T[ocr_idx], device)
            del V

            w.writerow([tok,
                        f"{cm_cka:.6f}", f"{cm_r2:.6f}", f"{cm_margin:.6f}",
                        f"{cm_cka_lmu:.6f}", f"{cm_r2_lmu:.6f}",
                        f"{cm_cka_ocr:.6f}", f"{cm_r2_ocr:.6f}",
                        f"{best_a:.6f}"])

    print(f"\n  Saved: {os.path.relpath(csv_path)}")


if __name__ == "__main__":
    main()
