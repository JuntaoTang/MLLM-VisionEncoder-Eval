#!/usr/bin/env python3
"""AC-policy FLOPs on the 70-encoder GT table, k'=8, including Stage-1 pretrain.

Per (LLM, encoder):
    E[F] = F_C + F_S1 + F_A + (k' / N) * F_S2

F_S1 is projector-only pretraining on LLaVA-558k (the 'pretrain' the table
must include). F_S2 is amortized because only k' of N encoders get Stage-2.
N is the AC-scorable pool for that LLM (encoders with A and C).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS_DIR, atomic_json  # noqa: E402
from estimate_ac_flops import (  # noqa: E402
    ARCH,
    ENC,
    FT_IMAGE_FRAC,
    K_PRIME,
    LLMS,
    N_A_SCORE,
    N_FINETUNE,
    N_PRETRAIN,
    N_SPAIR_JPEG,
    TEXT,
    eflops,
    encoder_spec,
    llm_fwd_flops,
    mean,
    proj_fwd_flops,
    visual_tokens_from_yaml,
)

GT_PATH = Path("/home/ma-user/work_space/VTB/results/ground_truth.json")
DISCRETE = {"toklip_l_384", "toklip_s_256", "unitok_attn", "vilau_256", "uniar_bsq"}
N_GT = 70
# Discrete: no SPair C-score; still charge Stage-1 pretrain + amortized Stage-2.
DISCRETE_N_VIS = {
    "toklip_l_384": 576,
    "toklip_s_256": 256,
    "unitok_attn": 256,
    "vilau_256": 256,
    "uniar_bsq": 1024,
}
DISCRETE_D_PROJ = {
    "uniar_bsq": 4608,  # post_quant_embed_dim
}

# Extra ViT families not in the 42-encoder paper table.
ARCH.update(
    {
        "vit_m16": (512, 12, 2048, 16, 1, False),
        "vit_b32": (768, 12, 3072, 32, 1, False),
        "siglip_b32": (768, 12, 3072, 32, 0, False),
        "dinov3_l": (1024, 24, 4096, 16, 5, False),  # 4 registers + CLS
        "mae1b": (1536, 40, 6144, 14, 1, False),
        "mae300m": (1024, 24, 4096, 16, 1, False),
        "dino1b": (1536, 40, 6144, 14, 1, False),
        "convnext_b": (1024, 0, 0, 32, 0, False),  # special-cased below
    }
)
ENC.update(
    {
        "siglip2_b16_384": ("vit_b", 384),
        "siglip2_l16_512": ("vit_l", 512),
        "siglip2_b32_256": ("siglip_b32", 256),
        "mc1_l14_224_400m": ("vit_l14", 224),
        "mc2_b16_384": ("vit_b16", 384),
        "mc2_h14_378": ("vit_h14", 378),
        "mc2_b16_224": ("vit_b16", 224),
        "mc2_m16_384": ("vit_m16", 384),
        "mc2_m16_224_mt5": ("vit_m16", 224),
        "mc2_m16_224": ("vit_m16", 224),
        "mc2_s16_384": ("vit_s16", 384),
        "mc2_s16_224_mt5": ("vit_s16", 224),
        "mc2_b32_384": ("vit_b32", 384),
        "mc2_b32_224": ("vit_b32", 224),
        "mc2_b32_224_mt5": ("vit_b32", 224),
        "mc1_b32_224_2.5b": ("vit_b32", 224),
        "mc1_b32_224_400m": ("vit_b32", 224),
        "dinov3_vitl16": ("dinov3_l", 256),
        "raev2_dinov3l_k7": ("dinov3_l", 256),
        "webssl_dino1b_full2b_224": ("dino1b", 224),
        "webssl_mae1b_full2b_224": ("mae1b", 224),
        "webssl_mae300m_full2b_224": ("mae300m", 224),
        "eupe_convnext_b": ("convnext_b", 256),
        "toklip_l_384": ("so400m", 384),
        "toklip_s_256": ("so400m", 256),
        "unitok_attn": ("vit_l", 256),
        "vilau_256": ("vit_l", 256),
        "uniar_bsq": ("so400m", 512),
    }
)

# ConvNeXt-B paper: 15.4 GFLOP @224; scale by (256/224)^2.
CONVNEXT_B_FWD = 15.4e9 * (256.0 / 224.0) ** 2


def spec_or_convnext(vid: str) -> dict:
    spec = encoder_spec(vid)
    if spec["arch"] == "convnext_b":
        vt = visual_tokens_from_yaml(vid)
        n_vis = float(vt if vt is not None else 64)
        spec["n_vis"] = n_vis
        spec["n_vit"] = int(n_vis)
        spec["F_vit_fwd"] = CONVNEXT_B_FWD
        spec["d"] = 1024
    if vid in DISCRETE_N_VIS:
        spec["n_vis"] = float(DISCRETE_N_VIS[vid])
        spec["n_vit"] = int(spec["n_vis"]) + (1 if spec["arch"] != "so400m" else 0)
    spec["d_proj"] = int(DISCRETE_D_PROJ.get(vid, spec["d"]))
    spec["has_c"] = vid not in DISCRETE
    return spec


def one_pair_n70(spec: dict, llm_key: str) -> dict:
    cfg = LLMS[llm_key]
    d_llm = cfg["d"]
    n_vis = spec["n_vis"]
    f_vit = spec["F_vit_fwd"]
    f_proj = proj_fwd_flops(n_vis, spec["d_proj"], d_llm)
    t_pre = n_vis + TEXT[llm_key]["pre"]
    t_ft = FT_IMAGE_FRAC * n_vis + TEXT[llm_key]["ft"]
    f_llm_pre = llm_fwd_flops(t_pre, cfg)
    f_llm_ft = llm_fwd_flops(t_ft, cfg)
    f_c = (N_SPAIR_JPEG * f_vit) if spec.get("has_c", True) else 0.0
    f_a = N_A_SCORE * (f_vit + f_proj + f_llm_pre)
    f_s1 = N_PRETRAIN * (f_vit + 3.0 * f_proj + f_llm_pre)
    f_s2 = N_FINETUNE * (f_vit + 3.0 * f_proj + 3.0 * f_llm_ft)
    f_full = f_s1 + f_s2
    f_policy = f_c + f_s1 + f_a + (K_PRIME / N_GT) * f_s2
    return {
        "llm": llm_key,
        "vision_id": spec["vision_id"],
        "n_vis": n_vis,
        "F_C": f_c,
        "F_A": f_a,
        "F_S1": f_s1,
        "F_S2": f_s2,
        "F_full": f_full,
        "F_policy_expected": f_policy,
    }


def main() -> None:
    gt = json.loads(GT_PATH.read_text())
    vids = list(gt["order"])
    if len(vids) != N_GT:
        raise SystemExit(f"expected {N_GT} GT encoders, got {len(vids)}")
    missing_arch = [v for v in vids if v not in ENC]
    if missing_arch:
        raise SystemExit(f"missing arch mapping: {missing_arch}")

    lines = [
        "AC Policy FLOPs  (ground_truth.json, N=70, k'=8, includes Stage-1)",
        "=" * 68,
        "",
        "Search space = all 70 tokenizers in ground_truth.json (same N for 3 LLMs).",
        "E[F per encoder] = F_C + F_S1 + F_A + (8/70) F_S2",
        "  F_S1 : projector-only pretrain on LLaVA-558k  (included)",
        "  F_S2 : LLM+projector finetune on LLaVA-665k, amortized 8/70",
        "  F_C  : SPair forwards (0 for 5 discrete tokenizers)",
        "  F_A  : 100 Stage-1 NLL forwards",
        f"  N_pretrain={N_PRETRAIN}  N_finetune={N_FINETUNE}  k'={K_PRIME}  N={N_GT}",
        "",
    ]
    summary = {}
    for llm_key, cfg in LLMS.items():
        n = N_GT
        recs = []
        for vid in vids:
            spec = spec_or_convnext(vid)
            rec = one_pair_n70(spec, llm_key)
            recs.append(rec)
        m_c = mean([r["F_C"] for r in recs])
        m_s1 = mean([r["F_S1"] for r in recs])
        m_s2 = mean([r["F_S2"] for r in recs])
        m_a = mean([r["F_A"] for r in recs])
        m_pol = mean([r["F_policy_expected"] for r in recs])
        m_full = mean([r["F_full"] for r in recs])
        campaign = n * (m_c + m_s1 + m_a) + K_PRIME * m_s2
        summary[llm_key] = {
            "name": cfg["name"],
            "n": n,
            "mean_F_C_EFLOP": eflops(m_c),
            "mean_F_S1_EFLOP": eflops(m_s1),
            "mean_F_S2_EFLOP": eflops(m_s2),
            "mean_F_A_EFLOP": eflops(m_a),
            "mean_policy_EFLOP": eflops(m_pol),
            "mean_full_S1S2_EFLOP": eflops(m_full),
            "campaign_EFLOP": eflops(campaign),
            "s2_share": K_PRIME / n,
        }
        s = summary[llm_key]
        lines += [
            f"{cfg['name']}  N={n}  k'/N={K_PRIME}/{n}={K_PRIME/n:.4f}",
            f"  F_C              {s['mean_F_C_EFLOP']:8.4f} EFLOP",
            f"  F_S1 (pretrain)  {s['mean_F_S1_EFLOP']:8.4f} EFLOP   <- included",
            f"  F_A              {s['mean_F_A_EFLOP']:8.4f} EFLOP",
            f"  (8/70) F_S2      {eflops(m_s2 * K_PRIME / n):8.4f} EFLOP",
            f"  E[F] per encoder {s['mean_policy_EFLOP']:8.4f} EFLOP",
            f"  full S1+S2       {s['mean_full_S1S2_EFLOP']:8.4f} EFLOP",
            f"  campaign (70 C/S1 + 8 S2) {s['campaign_EFLOP']:8.2f} EFLOP",
            "",
        ]

    pol = [summary[k]["mean_policy_EFLOP"] for k in LLMS]
    s1 = [summary[k]["mean_F_S1_EFLOP"] for k in LLMS]
    headline = sum(pol) / 3
    lines += [
        "Average over 3 LLMs  (table FLOPs column)",
        f"  E[F] per evaluated MLLM   {headline:.3f} EFLOP   = {headline*1e18:.2e} FLOPs",
        f"    of which Stage-1 pretrain {sum(s1)/3:.3f} EFLOP ({100*sum(s1)/3/headline:.0f}%)",
        f"  vs full S1+S2             {sum(summary[k]['mean_full_S1S2_EFLOP'] for k in LLMS)/3:.3f} EFLOP",
        "",
        "Table cell suggestion: "
        + f"${headline:.2f}$ EFLOPs"
        + f"  or  ${headline:.2f}\\!\\times\\!10^{{18}}$",
    ]
    summary["average"] = {
        "mean_policy_EFLOP": headline,
        "mean_policy_FLOPs": headline * 1e18,
        "mean_S1_EFLOP": sum(s1) / 3,
        "k_prime": K_PRIME,
        "formula": "F_C + F_S1 + F_A + (k'/N) F_S2",
    }
    text = "\n".join(lines)
    (RESULTS_DIR / "ac_gt70_flops.txt").write_text(text + "\n")
    atomic_json(RESULTS_DIR / "ac_gt70_flops.json", summary)
    print(text)
    print(f"\nwrote {RESULTS_DIR / 'ac_gt70_flops.txt'}")


if __name__ == "__main__":
    main()
