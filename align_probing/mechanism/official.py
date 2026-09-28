"""Load the paper's own per-tokenizer scores (seed 42 = Table 1; seeds 43/44 = full ladder)."""
import csv, json
import numpy as np
import analyze as A

R = "/home/ma-user/work_space/RAVEL/experiments"
S, LLMS = A.SLUGS, A.LLMS


def _canon(x): return x.replace("_mlp2x", "").strip()


def load():
    out = {}  # out[seed][stage][llm] -> np.array over S
    rows = list(csv.DictReader(open(f"{R}/missing_tokenizers_n2000/results/unified70_aligned_scores.csv")))
    m42 = {("MutualNN", "No PCA/whitening"): "S0_mutualnn",
           ("MutualNN", "PCA-256 + whitening"): "S1_whiten",
           ("Patch-RAVEL", "PCA-256 + whitening"): "S3_ravel"}
    d = {}
    for r in rows:
        k = m42.get((r["method"], r["preprocessing"]))
        if k: d[(k, r["base_model"], _canon(r["tokenizer_config"]))] = float(r["alignment_score"])
    out[42] = {st: {l: np.array([d[(st, l, s)] for s in S]) for l in LLMS} for st in set(m42.values())}
    stage = {"mutual_nn": "S0_mutualnn", "pca_whitening": "S1_whiten",
             "patch": "S2_patch", "all_weighting": "S3_ravel"}
    for sd in (43, 44):
        d = {}
        for r in csv.DictReader(open(f"{R}/formal_cumulative_multiseed/seed{sd}/results/per_tokenizer_scores.csv")):
            d[(stage[r["stage"]], r["base_model"], _canon(r["tokenizer_config"]))] = float(r["final_score"])
        out[sd] = {st: {l: np.array([d[(st, l, s)] for s in S]) for l in LLMS} for st in stage.values()}
    return out


def sail(l, key="score_5cap"):
    return np.array([json.load(open(f"/cache/wangky/align_probe/results/sweeps/cc3m2k/{l}/{s}.json"))[key]
                     for s in S])


if __name__ == "__main__":
    O = load()
    for sd in O:
        for st in sorted(O[sd]):
            rh = [A.spearman(O[sd][st][l], A.gt(l)) for l in LLMS]
            t1 = [A.gt(l)[int(np.argmax(O[sd][st][l]))] for l in LLMS]
            print(sd, f"{st:12s}", " ".join(f"{x:.3f}" for x in rh), f"avg {np.mean(rh):.3f} top1 {np.mean(t1):.2f}")
    rh = [A.spearman(sail(l), A.gt(l)) for l in LLMS]; t1 = [A.gt(l)[int(np.argmax(sail(l)))] for l in LLMS]
    print("SAIL        ", " ".join(f"{x:.3f}" for x in rh), f"avg {np.mean(rh):.3f} top1 {np.mean(t1):.2f}")
