# -*- coding: utf-8 -*-
"""
compute_cka_kernel.py
======================
Standalone pairwise CKA kernel from ANY features_diverse-style dir
(features_diverse, features_llm, ...). Output is drop-in compatible
with any consumer that reads file["cka"]).

Math/seed: fixed-seed row subset, centered Gram matrices, linear CKA.

Usage (from the CKA-X/ root):
    python scripts/compute_cka_kernel.py --feature_dir features_diverse \
        --full --out results/rebuild/cka_diverse.pt [--reference clip_openai__l14]
"""
import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ckax_common import list_feature_tokenizers, RESULTS_DIR  # noqa: E402


# Non---full CKA subsample, as a fraction of the available rows, so the
# default scales with whatever feature set is handed to the script.
N_CKA_FRACTION = 0.2


def centered_gram(X):
    Xc = X.float() - X.float().mean(0)
    return Xc @ Xc.T


def cka_from_grams(K, L):
    den = torch.sqrt((K * K).sum() * (L * L).sum()) + 1e-12
    return float((K * L).sum() / den)


def linear_cka_ssx(X, Y):
    """Linear CKA via the ||Xc'Yc||_F^2 identity (exact, no n x n gram).

    CKA_lin = <Xc Xc', Yc Yc'>_F / sqrt(<Xc Xc'>_F <Yc Yc'>_F)
             = ||Xc' Yc||_F^2 / (||Xc' Xc||_F * ||Yc' Yc||_F)
    Only needs d x d (d <= 1664) matrices instead of n x n grams, so
    full-data pairwise CKA is cheap and memory-light. Mathematically
    identical to the centered-gram version up to float rounding.
    X, Y: (n, d1), (n, d2) float tensors on the same device.
    """
    Xc = X - X.mean(0, keepdim=True)
    Yc = Y - Y.mean(0, keepdim=True)
    M = Xc.T @ Yc                      # (d1, d2)
    num = float((M * M).sum())
    sx = float(((Xc.T @ Xc) ** 2).sum())
    sy = float(((Yc.T @ Yc) ** 2).sum())
    return num / (torch.sqrt(torch.tensor(sx * sy)).item() + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_dir", type=str, required=True)
    ap.add_argument("--n_cka", type=int, default=None,
                    help="images per encoder for the non---full kernel "
                         "(default: 20%% of the available rows, so the "
                         "subsample scales with the feature set)")
    ap.add_argument("--full", action="store_true",
                    help="use ALL images (streaming pairwise CKA: features "
                         "stay on CPU, gram matrices built pair-by-pair on "
                         "GPU; peak RAM ~10-15 GB regardless of image count)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reference", type=str, default="clip_openai__l14")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    if args.out is None:
        args.out = os.path.join(RESULTS_DIR, "cka_kernel.pt")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    device = args.device
    if device != "cpu" and not torch.cuda.is_available():
        print("  [WARN] cuda unavailable; using cpu")
        device = "cpu"

    toks = list_feature_tokenizers(args.feature_dir)
    if not toks:
        raise SystemExit("  [ERROR] no tokenizers in " + args.feature_dir)
    feats, Ns = {}, {}
    for t in toks:
        f = torch.load(Path(args.feature_dir) / t / "visual_features.pt",
                       map_location="cpu", weights_only=False)
        feats[t] = f.float()
        Ns[t] = f.shape[0]
    if len(set(Ns.values())) != 1:
        raise SystemExit(f"  [ERROR] row count mismatch: {set(Ns.values())}")
    N = Ns[toks[0]]
    if args.full:
        n_cka = N
    elif args.n_cka:
        n_cka = min(args.n_cka, N)
    else:
        n_cka = max(3, int(round(N_CKA_FRACTION * N)))
    g = torch.Generator().manual_seed(args.seed)
    sub = None if args.full else \
        torch.randperm(N, generator=g)[:n_cka].sort().values

    ref = args.reference if args.reference in toks else toks[0]
    if ref != args.reference:
        print(f"  [WARN] reference {args.reference} absent; using {ref}")
    print(f"  {len(toks)} tokenizers, N={N}, n_cka={n_cka} "
          f"({'FULL' if args.full else 'subsample'}), ref={ref}")

    if args.full:
        # Streaming full-data mode via the ||Xc'Yc||_F^2 CKA identity:
        # no n x n gram ever materialized -> peak RAM ~= features (~10 GB),
        # each pair costs two d x d matmuls. Exact same value as the
        # centered-gram version up to float rounding.
        kernel = torch.eye(len(toks))
        cka_to_ref = {}
        idx = {t: i for i, t in enumerate(toks)}
        ri = idx[ref]
        for i in range(len(toks)):
            Xi = feats[toks[i]].to(device)
            for j in range(i + 1, len(toks)):
                Xj = feats[toks[j]].to(device)
                v = linear_cka_ssx(Xi, Xj)
                kernel[i, j] = kernel[j, i] = v
                del Xj
            if toks[i] == ref:
                cka_to_ref[ref] = 1.0
            del Xi
            torch.cuda.empty_cache()
            print(f"  kernel row {i + 1}/{len(toks)} done")
        # CKA-to-reference from the already-computed kernel row of `ref`.
        for j, t in enumerate(toks):
            if t != ref:
                cka_to_ref[t] = float(kernel[ri, j])
    else:
        grams, cka_to_ref, Kref = {}, {}, None
        for i, t in enumerate(toks, 1):
            X = feats[t][sub]
            K = centered_gram(X.to(device)).cpu()
            grams[t] = K
            if t == ref:
                Kref = K
                cka_to_ref[t] = 1.0
            del X
            print(f"  [{i:>2}/{len(toks)}] {t:<28} gram ready")
        del feats
        for t in toks:
            if t != ref:
                cka_to_ref[t] = cka_from_grams(grams[t], Kref)

        n = len(toks)
        kernel = torch.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                v = cka_from_grams(grams[toks[i]], grams[toks[j]])
                kernel[i, j] = kernel[j, i] = v
            print(f"  kernel row {i + 1}/{n} done")

    torch.save({
        "tok_names": toks,
        "cka": {"toks": toks, "cka_to_ref": cka_to_ref,
                "kernel": kernel, "n_cka": n_cka, "reference": ref},
        "meta": {"feature_dir": os.path.abspath(args.feature_dir),
                 "seed": args.seed, "n_cka": n_cka, "n_images": N},
    }, args.out)
    print(f"\n  Saved: {args.out}")
    top = sorted(cka_to_ref.items(), key=lambda kv: -kv[1])[:6]
    print("  CKA-to-ref top: "
          + ", ".join(f"{t}={v:.3f}" for t, v in top))


if __name__ == "__main__":
    main()