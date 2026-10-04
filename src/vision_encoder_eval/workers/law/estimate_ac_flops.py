#!/usr/bin/env python3
"""Analytical FLOPs for the AC policy (k'=8) on the 42-encoder paper pool.

AC policy cost per LLM
----------------------
  C-score : vision-only forward on all 42 encoders (SPair-71k JPEGs)
  Stage-1 : projector-only pretrain on all 42 (needed for A-score)
  A-score : 100 frozen forwards of the Stage-1 model
  Stage-2 : full LLM+projector finetune on a uniform random k'=8 / 42

Expected cost of one (LLM, tokenizer), then average over tokenizers:

    E[F] = F_C + F_S1 + F_A + (8/42) * F_S2

C is encoder-level and shared across LLMs. Per-LLM averages charge C in
full (that LLM's policy needs it). The 3-LLM campaign charges C once.

Training multipliers (Kaplan): frozen module = 1x forward; trained module
= 3x (fwd + 2x bwd). Vision encoder is frozen in both stages. Stage-1
tunes only mm_mlp_adapter; Stage-2 tunes adapter + LLM.

This is an architecture-level estimate, not a profiler dump.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vision_encoder_eval.workers.law.common import MLLM_CFG_ROOT, RESULTS_DIR, atomic_json  # noqa: E402
from vision_encoder_eval.workers.law.fit_ac import K_PRIME_HEADLINE, PAPER_ENCODERS  # noqa: E402

N_PRETRAIN = 558_128
N_FINETUNE = 664_903
N_SPAIR_JPEG = 1_800
N_A_SCORE = 100
FT_IMAGE_FRAC = 0.933
K_PRIME = K_PRIME_HEADLINE
N_ENC = len(PAPER_ENCODERS)

# Sampled with each LLM tokenizer (2000 random rows, seed=0), <image> stripped.
# +2 / +16 covers BOS/EOS and a short chat template.
TEXT = {
    "qwen25": {"pre": 20.8 + 2.0, "ft": 201.2 + 16.0},
    "qwen3": {"pre": 20.8 + 2.0, "ft": 201.2 + 16.0},
    "smollm2": {"pre": 22.2 + 2.0, "ft": 217.9 + 16.0},
}

LLMS = {
    "qwen25": dict(
        name="Qwen2.5-1.5B",
        d=1536,
        L=28,
        nq=12,
        nkv=2,
        hd=128,
        inter=8960,
        vocab=151936,
    ),
    "qwen3": dict(
        name="Qwen3-1.7B",
        d=2048,
        L=28,
        nq=16,
        nkv=8,
        hd=128,
        inter=6144,
        vocab=151936,
    ),
    "smollm2": dict(
        name="SmolLM2-1.7B",
        d=2048,
        L=24,
        nq=32,
        nkv=32,
        hd=64,
        inter=8192,
        vocab=49152,
    ),
}

# (d, L, mlp, patch, extra_tokens, swiglu)
# extra_tokens = CLS / register tokens inside the ViT (not LLM visual tokens).
ARCH = {
    "so400m": (1152, 27, 4304, 14, 0, False),
    "siglip_g": (1536, 40, 6144, 16, 0, False),
    "vit_l": (1024, 24, 4096, 16, 0, False),
    "vit_b": (768, 12, 3072, 16, 0, False),
    "clip_l14": (1024, 24, 4096, 14, 1, False),
    "bigg": (1664, 48, 8192, 14, 1, False),
    "vit_h14": (1280, 32, 5120, 14, 1, False),
    "vit_l14": (1024, 24, 4096, 14, 1, False),
    "vit_b16": (768, 12, 3072, 16, 1, False),
    "vit_s16": (384, 12, 1536, 16, 1, False),
    "pe_g": (1536, 50, 8960, 14, 0, False),
    "pe_l": (1024, 23, 4096, 14, 1, False),
    "pe_b": (768, 12, 3072, 16, 1, False),
    "pixio_l": (1024, 24, 4096, 16, 8, False),
    "pixio_b": (768, 12, 3072, 16, 8, False),
    "pixio_h": (1280, 32, 5120, 16, 8, False),
    "dinov2_g": (1536, 40, 4096, 14, 1, True),
    "dinov2_l": (1024, 24, 4096, 14, 1, False),
    "dinov2_b": (768, 12, 3072, 14, 1, False),
    "dinov2_s": (384, 12, 1536, 14, 1, False),
    "eupe_b": (768, 12, 3072, 16, 5, False),
    "eupe_s": (384, 12, 1536, 16, 5, False),
    "eupe_t": (192, 12, 768, 16, 5, False),
    "ijepa_h": (1280, 32, 5120, 14, 1, False),
    "dino_b16": (768, 12, 3072, 16, 1, False),
    "dino_s16": (384, 12, 1536, 16, 1, False),
    "dino_b8": (768, 12, 3072, 8, 1, False),
    "dino_s8": (384, 12, 1536, 8, 1, False),
    "mae3b": (3072, 26, 12288, 14, 1, False),
}

# (arch_key, image_size) — visual tokens default to (image/patch)^2
ENC = {
    "siglip2_sm14_384": ("so400m", 384),
    "siglip2_g16_384": ("siglip_g", 384),
    "siglip2_l16_384": ("vit_l", 384),
    "siglip2_sm16_512": ("so400m", 512),
    "siglip2_sm16_384": ("so400m", 384),
    "siglip2_g16_256": ("siglip_g", 256),
    "mc2_g14_378": ("bigg", 378),
    "siglip2_sm14_224": ("so400m", 224),
    "pe_core_g14_448": ("pe_g", 448),
    "siglip2_sm16_256": ("so400m", 256),
    "siglip2_l16_256": ("vit_l", 256),
    "mc1_g14_224_2.5b": ("bigg", 224),
    "mc2_g14_224": ("bigg", 224),
    "siglip2_b16_512": ("vit_b", 512),
    "mc1_h14_224_v1.2": ("vit_h14", 224),
    "clip_openai__l14": ("clip_l14", 224),
    "mc1_h14_224_2.5b": ("vit_h14", 224),
    "mc1_l14_224_2.5b": ("vit_l14", 224),
    "siglip2_b16_256": ("vit_b", 256),
    "siglip2_b16_224": ("vit_b", 224),
    "mc2_l14_224": ("vit_l14", 224),
    "pe_lang_l14_448": ("pe_l", 448),
    "pe_core_b16_224": ("pe_b", 224),
    "mc1_b16_224_400m": ("vit_b16", 224),
    "mc1_b16_224_2.5b": ("vit_b16", 224),
    "pixio_vitl16": ("pixio_l", 256),
    "dinov2_giant": ("dinov2_g", 518),
    "dinov2_large": ("dinov2_l", 518),
    "pixio_vitb16": ("pixio_b", 256),
    "eupe_vit_b": ("eupe_b", 256),
    "eupe_vit_s": ("eupe_s", 256),
    "mc2_s16_224": ("vit_s16", 224),
    "dinov2_base": ("dinov2_b", 518),
    "dinov2_small": ("dinov2_s", 518),
    "pixio_vith16": ("pixio_h", 256),
    "ijepa_vith14": ("ijepa_h", 224),
    "dino_vitb16": ("dino_b16", 224),
    "webssl_mae3b_full2b_224": ("mae3b", 224),
    "eupe_vit_t": ("eupe_t", 256),
    "dino_vits8": ("dino_s8", 224),
    "dino_vitb8": ("dino_b8", 224),
    "dino_vits16": ("dino_s16", 224),
}


def visual_tokens_from_yaml(vision_id: str) -> int | None:
    path = MLLM_CFG_ROOT / "qwen25" / f"{vision_id}_mlp2x.yaml"
    if not path.is_file():
        return None
    data = yaml.safe_load(path.read_text()) or {}
    vt = ((data.get("batch") or {}).get("finetune") or {}).get("visual_tokens")
    return int(vt) if vt else None


def vit_fwd_flops(n: int, d: int, L: int, mlp: int, patch: int, swiglu: bool) -> float:
    patch_embed = 2.0 * n * d * (patch * patch * 3)
    attn_lin = L * (2.0 * n * 3.0 * d * d + 2.0 * n * d * d)
    attn_qkav = L * (4.0 * n * n * d)
    mlp_f = L * ((6.0 if swiglu else 4.0) * n * d * mlp)
    return patch_embed + attn_lin + attn_qkav + mlp_f


def llm_matmul_params(cfg: dict) -> tuple[float, float]:
    d, L, nq, nkv, hd, inter, vocab = (
        cfg["d"],
        cfg["L"],
        cfg["nq"],
        cfg["nkv"],
        cfg["hd"],
        cfg["inter"],
        cfg["vocab"],
    )
    qkv_o = d * (nq * hd) + 2 * d * (nkv * hd) + (nq * hd) * d
    mlp = 3 * d * inter  # SwiGLU
    return float(L * (qkv_o + mlp)), float(vocab * d)


def llm_fwd_flops(T: float, cfg: dict) -> float:
    nonembed, embed = llm_matmul_params(cfg)
    matmul = 2.0 * nonembed * T + 2.0 * embed * T
    attn = 4.0 * cfg["L"] * T * T * cfg["d"]
    return matmul + attn


def proj_fwd_flops(n_vis: float, d_v: int, d_llm: int) -> float:
    # mlp2x: Linear(d_v, d_llm) + GELU + Linear(d_llm, d_llm)
    return 2.0 * n_vis * d_v * d_llm + 2.0 * n_vis * d_llm * d_llm


def encoder_spec(vision_id: str) -> dict:
    arch_key, image_size = ENC[vision_id]
    d, L, mlp, patch, extra, swiglu = ARCH[arch_key]
    n_patch = (image_size // patch) ** 2
    vt = visual_tokens_from_yaml(vision_id)
    n_vis = float(vt if vt is not None else n_patch)
    n_vit = n_patch + extra
    f_vit = vit_fwd_flops(n_vit, d, L, mlp, patch, swiglu)
    return {
        "vision_id": vision_id,
        "arch": arch_key,
        "d": d,
        "L": L,
        "mlp": mlp,
        "patch": patch,
        "image_size": image_size,
        "n_patch": n_patch,
        "n_vis": n_vis,
        "n_vit": n_vit,
        "swiglu": swiglu,
        "F_vit_fwd": f_vit,
    }


def one_pair(spec: dict, llm_key: str) -> dict:
    cfg = LLMS[llm_key]
    d_llm = cfg["d"]
    n_vis = spec["n_vis"]
    f_vit = spec["F_vit_fwd"]
    f_proj = proj_fwd_flops(n_vis, spec["d"], d_llm)
    t_pre = n_vis + TEXT[llm_key]["pre"]
    t_ft = FT_IMAGE_FRAC * n_vis + TEXT[llm_key]["ft"]
    f_llm_pre = llm_fwd_flops(t_pre, cfg)
    f_llm_ft = llm_fwd_flops(t_ft, cfg)

    f_c = N_SPAIR_JPEG * f_vit
    f_a = N_A_SCORE * (f_vit + f_proj + f_llm_pre)
    # S1: vit frozen, proj trained, LLM frozen
    f_s1 = N_PRETRAIN * (f_vit + 3.0 * f_proj + f_llm_pre)
    # S2: vit frozen, proj+LLM trained
    f_s2 = N_FINETUNE * (f_vit + 3.0 * f_proj + 3.0 * f_llm_ft)
    f_full = f_s1 + f_s2
    f_policy = f_c + f_s1 + f_a + (K_PRIME / N_ENC) * f_s2
    return {
        "llm": llm_key,
        "vision_id": spec["vision_id"],
        "n_vis": n_vis,
        "T_pre": t_pre,
        "T_ft": t_ft,
        "F_vit_fwd": f_vit,
        "F_C": f_c,
        "F_A": f_a,
        "F_S1": f_s1,
        "F_S2": f_s2,
        "F_full": f_full,
        "F_policy_expected": f_policy,
    }


def eflops(x: float) -> float:
    return x / 1e18


def pflops(x: float) -> float:
    return x / 1e15


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def main() -> None:
    missing = [vid for vid, _, _ in PAPER_ENCODERS if vid not in ENC]
    if missing:
        raise SystemExit(f"missing encoder arch: {missing}")

    specs = [encoder_spec(vid) for vid, _, _ in PAPER_ENCODERS]
    rows = []
    per_llm: dict[str, list[dict]] = {}
    for llm_key in LLMS:
        per_llm[llm_key] = [one_pair(sp, llm_key) for sp in specs]
        rows.extend(per_llm[llm_key])

    summary = {}
    lines = [
        "AC policy FLOPs  (analytical, k'=8, 42 encoders)",
        "=" * 64,
        "",
        "E[F | LLM, tokenizer] = F_C + F_S1 + F_A + (8/42) F_S2",
        "then average over the 42 tokenizers (and over the 3 LLMs).",
        "",
        f"N_pretrain={N_PRETRAIN}  N_finetune={N_FINETUNE}  N_SPair={N_SPAIR_JPEG}",
        "S1: vit 1x, projector 3x, LLM 1x (frozen).  S2: vit 1x, projector 3x, LLM 3x.",
        "C is encoder-only (shared across LLMs). A-score = 100 Stage-1 forwards.",
        "",
    ]

    for llm_key, cfg in LLMS.items():
        recs = per_llm[llm_key]
        m_c = mean([r["F_C"] for r in recs])
        m_s1 = mean([r["F_S1"] for r in recs])
        m_s2 = mean([r["F_S2"] for r in recs])
        m_a = mean([r["F_A"] for r in recs])
        m_pol = mean([r["F_policy_expected"] for r in recs])
        m_full = mean([r["F_full"] for r in recs])
        campaign = N_ENC * (m_c + m_s1 + m_a) + K_PRIME * m_s2
        full_campaign = N_ENC * (m_c + m_full + m_a)
        summary[llm_key] = {
            "name": cfg["name"],
            "mean_F_C_EFLOP": eflops(m_c),
            "mean_F_S1_EFLOP": eflops(m_s1),
            "mean_F_S2_EFLOP": eflops(m_s2),
            "mean_F_A_EFLOP": eflops(m_a),
            "mean_policy_EFLOP": eflops(m_pol),
            "mean_full_S1S2_EFLOP": eflops(m_full),
            "campaign_policy_EFLOP": eflops(campaign),
            "campaign_full_EFLOP": eflops(full_campaign),
            "policy_over_full": m_pol / m_full,
        }
        lines += [
            f"{cfg['name']}  (mean over {N_ENC} tokenizers)",
            f"  mean F_C           {eflops(m_c):8.4f} EFLOP",
            f"  mean F_S1          {eflops(m_s1):8.4f} EFLOP",
            f"  mean F_S2          {eflops(m_s2):8.4f} EFLOP",
            f"  mean F_A           {eflops(m_a):8.4f} EFLOP",
            f"  mean AC-policy E[F]{eflops(m_pol):8.4f} EFLOP   (= C+S1+A+(8/42)S2)",
            f"  mean full S1+S2    {eflops(m_full):8.4f} EFLOP",
            f"  policy / full      {m_pol / m_full:8.3f}",
            f"  campaign (42 C/S1 + 8 S2) {eflops(campaign):8.3f} EFLOP",
            "",
        ]

    # Average over LLMs. C counted fully in each per-LLM mean.
    pol_means = [summary[k]["mean_policy_EFLOP"] for k in LLMS]
    s1_means = [summary[k]["mean_F_S1_EFLOP"] for k in LLMS]
    s2_means = [summary[k]["mean_F_S2_EFLOP"] for k in LLMS]
    full_means = [summary[k]["mean_full_S1S2_EFLOP"] for k in LLMS]
    c_mean = summary["qwen25"]["mean_F_C_EFLOP"]  # identical across LLMs
    headline = sum(pol_means) / 3
    headline_c_once = headline - (2.0 / 3.0) * c_mean

    campaign_3llm_c_once = (
        N_ENC * (c_mean * 1e18)
        + sum(N_ENC * (summary[k]["mean_F_S1_EFLOP"] + summary[k]["mean_F_A_EFLOP"]) * 1e18 for k in LLMS)
        + sum(K_PRIME * summary[k]["mean_F_S2_EFLOP"] * 1e18 for k in LLMS)
    )
    per_pair_c_once = campaign_3llm_c_once / (3 * N_ENC)

    summary["average"] = {
        "mean_over_llm_and_tokenizer_EFLOP": headline,
        "mean_C_shared_once_EFLOP": eflops(per_pair_c_once),
        "mean_F_C_EFLOP": c_mean,
        "mean_F_S1_EFLOP": sum(s1_means) / 3,
        "mean_F_S2_EFLOP": sum(s2_means) / 3,
        "mean_full_S1S2_EFLOP": sum(full_means) / 3,
        "campaign_3llm_C_once_EFLOP": eflops(campaign_3llm_c_once),
        "k_prime": K_PRIME,
        "n_encoders": N_ENC,
        "n_llms": 3,
    }

    lines += [
        "Average over 3 LLMs × 42 tokenizers",
        f"  E[F] (C charged per LLM)     {headline:8.4f} EFLOP",
        f"  E[F] (C charged once / 3)    {headline_c_once:8.4f} EFLOP",
        f"  mean F_S1                    {sum(s1_means)/3:8.4f} EFLOP",
        f"  mean F_S2                    {sum(s2_means)/3:8.4f} EFLOP",
        f"  mean full S1+S2              {sum(full_means)/3:8.4f} EFLOP",
        f"  3-LLM campaign (C once)      {eflops(campaign_3llm_c_once):8.3f} EFLOP",
        "",
        "Headline number: mean over LLM and tokenizer of E[F] = "
        f"{headline:.3f} EFLOP  ({headline*1e3:.1f} PFLOP).",
        f"C-score is <{c_mean*1e3:.2f} PFLOP/encoder — negligible vs Stage-1.",
        "",
        "1 EFLOP = 1e18 FLOPs.  Assumptions: Kaplan 2ND/token forward;",
        "trained modules 3x; attention 4 L T^2 d; mlp2x projector;",
        "text lengths from actual tokenizers (see TEXT in this file).",
    ]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    txt_path = RESULTS_DIR / "ac_policy_flops.txt"
    txt_path.write_text("\n".join(lines) + "\n")

    csv_path = RESULTS_DIR / "ac_policy_flops.csv"
    header = [
        "llm",
        "vision_id",
        "n_vis",
        "T_pre",
        "T_ft",
        "F_C_EFLOP",
        "F_S1_EFLOP",
        "F_S2_EFLOP",
        "F_A_EFLOP",
        "F_policy_expected_EFLOP",
        "F_full_S1S2_EFLOP",
    ]
    with csv_path.open("w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(
                ",".join(
                    [
                        r["llm"],
                        r["vision_id"],
                        f"{r['n_vis']:.1f}",
                        f"{r['T_pre']:.2f}",
                        f"{r['T_ft']:.2f}",
                        f"{eflops(r['F_C']):.6f}",
                        f"{eflops(r['F_S1']):.6f}",
                        f"{eflops(r['F_S2']):.6f}",
                        f"{eflops(r['F_A']):.8f}",
                        f"{eflops(r['F_policy_expected']):.6f}",
                        f"{eflops(r['F_full']):.6f}",
                    ]
                )
                + "\n"
            )

    spec_rows = [
        {
            "vision_id": s["vision_id"],
            "arch": s["arch"],
            "d": s["d"],
            "L": s["L"],
            "n_vis": s["n_vis"],
            "n_vit": s["n_vit"],
            "F_vit_fwd_GFLOP": s["F_vit_fwd"] / 1e9,
        }
        for s in specs
    ]
    atomic_json(
        RESULTS_DIR / "ac_policy_flops.json",
        {
            "k_prime": K_PRIME,
            "n_encoders": N_ENC,
            "formula": "E[F]=F_C+F_S1+F_A+(k'/42)*F_S2",
            "units": "EFLOP = 1e18 FLOPs",
            "text_tokens": TEXT,
            "n_pretrain": N_PRETRAIN,
            "n_finetune": N_FINETUNE,
            "n_spair": N_SPAIR_JPEG,
            "summary": summary,
            "encoders": spec_rows,
        },
    )
    print("\n".join(lines))
    print(f"\nwrote {txt_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
