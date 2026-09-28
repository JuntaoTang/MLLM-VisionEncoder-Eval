#!/usr/bin/env python3
"""Training-budget sweep summary for the SAIL alignment-probing baseline."""
import json, sys
import numpy as np
from correlate import spearman, pearson
from flops_cc3m import head_flops
import common

S = common.read_tokenizer_list()
GT = json.load(open("/home/ma-user/work_space/VTB/results/ground_truth.json"))["encoders"]
LL = [("qwen3", "Qwen3-1.7B"), ("qwen25", "Qwen2.5-1.5B"), ("smollm2", "SmolLM2-1.7B")]
BUD = [int(x) for x in (sys.argv[1:] or [500, 1000, 2000, 3000, 5000, 10000])]
VIS = {r["slug"]: r for r in json.load(open(common.RESULTS_DIR / "flops_vision.json"))}
TX_COCO = sum(r["flops_total"] for r in json.load(open(common.RESULTS_DIR / "flops_text.json")))
_cc = json.load(open(common.RESULTS_DIR / "flops_text_cc3m2k.json"))
TX_PER_CAP = sum(r["flops_total"] for r in _cc) / _cc[0]["n_captions"]   # 2000 imgs x 2 captions


def res(N, l, s): return json.load(open(common.RESULTS_DIR / "sweeps" / f"budget_n{N}" / l / f"{s}.json"))


def flops(N, include_eval=True):
    """Mean FLOPs per tokenizer: vision + text encoding, alignment training, and
    (include_eval) the COCO-2K retrieval pass that produces the score."""
    parts = ("train", "val", "test") if include_eval else ("train",)
    tot = []
    for s in S:
        enc = VIS[s]["flops_per_image"] * (N + 2000)          # N CC3M train + 2000 COCO eval
        h = [head_flops(res(N, l, s)) for l, _ in LL]
        tot.append(enc + sum(x[k] for x in h for k in parts))
    text = (TX_PER_CAP * N * 2 + TX_COCO) / len(S)
    return np.mean(tot) + text


rows = []
for N in BUD:
    r = {"N": N}
    for l, name in LL:
        x = np.array([res(N, l, s)["score_5cap"] for s in S])
        g = np.array([GT[s]["llms"][l]["average"] for s in S])
        r[l] = (spearman(x, g), pearson(x, g), g[int(np.argmax(x))], x.mean())
    r["flops"] = flops(N, True)
    r["flops_no_eval"] = flops(N, False)
    rows.append(r)

print(f"{'N':>7s}" + "".join(f"{n:>26s}" for _, n in LL) + f"{'mean MR':>9s}{'PFLOPs':>9s}{'(no eval)':>11s}")
print(f"{'':7s}" + "".join(f"{'rho      r    Top-1':>26s}" for _ in LL))
for r in rows:
    line = f"{r['N']:>7d}"
    for l, _ in LL:
        rho, pr, t1, _ = r[l]; line += f"{rho:9.3f}{pr:8.3f}{t1:9.2f}"
    line += (f"{np.mean([r[l][3] for l,_ in LL]):9.2f}{r['flops']/1e15:9.2f}"
             f"{r['flops_no_eval']/1e15:11.2f}")
    print(line)

with open(common.RESULTS_DIR / "budget_sweep_summary.txt", "w") as f:
    f.write("Training-budget sweep - SAIL alignment probing\n")
    f.write("Train: first N CC3M pairs (nested). Eval: COCO-2K, 2000 images, 5-caption protocol.\n")
    f.write("Score = MR(5cap) = mean of i2t/t2i R@1/5/10. GT = downstream 11-benchmark average.\n")
    f.write("PFLOPs = mean cost per tokenizer: vision+text encoding + alignment training\n"
            "         + the COCO-2K retrieval evaluation (2xMAC convention).\n\n")
    f.write("N\tLLM\tspearman\tpearson\ttop1\tmean_score\n")
    for r in rows:
        for l, name in LL:
            rho, pr, t1, mu = r[l]
            f.write(f"{r['N']}\t{name}\t{rho:.4f}\t{pr:.4f}\t{t1:.2f}\t{mu:.2f}\n")
    f.write("\nN\tPFLOPs_per_tokenizer\tPFLOPs_excluding_eval\n")
    for r in rows:
        f.write(f"{r['N']}\t{r['flops']/1e15:.3f}\t{r['flops_no_eval']/1e15:.3f}\n")

with open(common.RESULTS_DIR / "budget_sweep_scores.txt", "w") as f:
    f.write("n_train\ttokenizer\tllm\tscore\n")
    for N in BUD:
        for s in S:
            for l, name in LL:
                f.write(f"{N}\t{s}\t{name}\t{res(N, l, s)['score_5cap']:.2f}\n")
print(f"\nwrote {common.RESULTS_DIR}/budget_sweep_summary.txt")
print(f"wrote {common.RESULTS_DIR}/budget_sweep_scores.txt  ({len(BUD)*210} rows)")
