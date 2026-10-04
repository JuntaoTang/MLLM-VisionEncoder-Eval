# -*- coding: utf-8 -*-
"""
plot_budget_sweep.py
=========================
Figure and tables for the CKA-X (GBDT) labeling-budget sweep.

Reads :
    results/reference/budget_sweep.json     (100 splits, seed 0, k'=8..64)
    results/reference/fullinfo_gbdt.json        (full-information 5-fold reference)
Writes (default --outdir = the CKA-X/ root):
    budget_sweep_gbdt_<caliber>.png / .pdf   line plot, Times New Roman, no title
    budget_sweep_table.tex              LaTeX tables (booktabs)
    budget_sweep_table.md               markdown tables

Calibers (--caliber):
    fold_mean (default) : metric per split, then mean +/- std over the 100
                          splits; error bars shown.
    pooled              : all held-out predictions of all 100 splits
                          concatenated, one metric over n_splits*(N-k') points;
                          monotone in k' (no per-split noise), no error bars.

Figure style
------------
Colours follow the reference figure (Okabe-Ito colourblind-safe triple):
    Qwen2.5-1.5B  blue   #0072B2  circle   (legend first)
    Qwen3-1.7B    orange #D55E00  square   (legend second)
    SmolLM2-1.7B  green  #009E73  triangle (legend third)
Tables keep the column order used throughout (Qwen3 / Qwen2.5 / SmolLM2).
"""
import argparse
import json
import os

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(SCRIPT_DIR)              # the CKA-X/ root

# Display names for the backbones of the reference figure; a backbone the map
# does not know is shown under its own key (see set_backbones()).
LABEL = {"qwen3": "Qwen3-1.7B", "qwen25": "Qwen2.5-1.5B",
         "smollm2": "SmolLM2-1.7B"}

# figure order and colours (Okabe-Ito, matching the reference figure)
SERIES_STYLE = [
    ("qwen25", "#0072B2", "o"),     # blue, circle
    ("qwen3", "#D55E00", "s"),      # orange-red, square
    ("smollm2", "#009E73", "^"),    # bluish green, triangle
]
EXTRA_STYLE = [("#56B4E9", "v"), ("#E69F00", "D"), ("#CC79A7", "P"),
               ("#999999", "x"), ("#F0E442", "*")]
QWEN25 = "qwen25"

# Filled in from the sweep file by set_backbones(); the defaults below keep
# the module usable on its own.
KEYS = [b for b, _, _ in SERIES_STYLE]
TABLE_ORDER = list(KEYS)
PLOT_SERIES = list(SERIES_STYLE)


def set_backbones(keys):
    """Order the backbones that the sweep file actually contains.

    The reference figure's order and colours are kept for the backbones it
    knows; anything else is appended with an automatic style, so the figure is
    not tied to one experiment's backbone set."""
    global KEYS, TABLE_ORDER, PLOT_SERIES
    keys = list(keys)
    KEYS = keys
    TABLE_ORDER = list(keys)
    styles = {b: (c, m) for b, c, m in SERIES_STYLE}
    for i, b in enumerate(keys):
        styles.setdefault(b, EXTRA_STYLE[i % len(EXTRA_STYLE)])
        LABEL.setdefault(b, b)
    known = [b for b, _, _ in SERIES_STYLE if b in keys]
    PLOT_SERIES = [(b, styles[b][0], styles[b][1])
                   for b in known + [b for b in keys if b not in known]]


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_fold_mean(sweep, field="spearman"):
    """{bb: (means, stds)} from the fold-mean caliber."""
    out = {}
    for bb in KEYS:
        m, s = [], []
        for k in sweep["protocol"]["k_sweep"]:
            r = (sweep["backbones"].get(bb) or {}).get(str(k))
            if r is None:
                m.append(np.nan)
                s.append(np.nan)
            else:
                c = r["fold_mean"][field]
                m.append(c["mean"])
                s.append(c["std"])
        out[bb] = (np.array(m), np.array(s))
    return out


def collect_pooled(sweep, field="spearman"):
    """{bb: values} from the pooled caliber."""
    out = {}
    for bb in KEYS:
        v = []
        for k in sweep["protocol"]["k_sweep"]:
            r = (sweep["backbones"].get(bb) or {}).get(str(k))
            v.append(np.nan if r is None else r["pooled"][field])
        out[bb] = np.array(v)
    return out


def fmt(m, s=None, nd=3):
    if m is None or (isinstance(m, float) and np.isnan(m)):
        return "--"
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return f"{m:.{nd}f}"
    return f"{m:.{nd}f} +/- {s:.{nd}f}"


def md_block(k_list, series, title, nd=3):
    fm = isinstance(next(iter(series.values())), tuple)
    M = [f"## {title}", "",
         "| labeled encoders | " + " | ".join(LABEL[b] for b in TABLE_ORDER)
         + " |",
         "|" + "---|" * (len(TABLE_ORDER) + 1)]
    for i, k in enumerate(k_list):
        cells = []
        for bb in TABLE_ORDER:
            if fm:
                cells.append(fmt(series[bb][0][i], series[bb][1][i], nd))
            else:
                cells.append(fmt(series[bb][i], None, nd))
        M.append(f"| {k} | " + " | ".join(cells) + " |")
    M.append("")
    return M


def write_tables(outdir, k_list, ref, rho, pear, top1, pooled_rho):
    """LaTeX (fold-mean) and markdown (both calibers) tables."""
    # ---------------- LaTeX: three-backbone fold-mean table ----------------
    L = []
    L.append("% CKA-X (GBDT) labeling-budget sweep -- fold-mean caliber")
    L.append("% requires \\usepackage{booktabs} in the preamble")
    L.append("% protocol: seed 0, 100 random labeled subsets per k', "
             "unified text encoder")
    L.append("% generated by CKA-X/scripts/plot_budget_sweep.py")
    L.append("\\begin{table}[t]")
    L.append("\\centering")
    L.append("\\small")
    L.append("\\begin{tabular}{l" + "c" * len(TABLE_ORDER) + "}")
    L.append("\\toprule")
    L.append("Labeled encoders & " +
             " & ".join(f"{LABEL[b]} $\\rho$" for b in TABLE_ORDER) + " \\\\")
    L.append("\\midrule")
    for i, k in enumerate(k_list):
        cells = ["$%.3f \\pm %.3f$" % (rho[bb][0][i], rho[bb][1][i])
                 for bb in TABLE_ORDER]
        L.append(f"{k} & " + " & ".join(cells) + " \\\\")
    L.append("\\midrule")
    cells = ["$%.3f \\pm %.3f$" % (ref[bb]["rho"][0], ref[bb]["rho"][1])
             for bb in TABLE_ORDER]
    L.append("full info (5-fold) & " + " & ".join(cells) + " \\\\")
    L.append("\\bottomrule")
    L.append("\\end{tabular}")
    L.append("\\caption{Data efficiency of CKA-X with the gradient-boosted "
             "regressor. Each cell is the mean $\\pm$ std of Spearman's $\\rho$ "
             "on the held-out encoders over 100 random splits that label the "
             "given number of candidate encoders; the last row is the "
             "full-information 5-fold reference. Fold-mean caliber.}")
    L.append("\\label{tab:budget_sweep_gbdt}")
    L.append("\\end{table}")
    L.append("")
    # ---- single-backbone budget table
    L.append("% single-backbone budget table "
             "(single backbone: rho / r / Top-1)")
    L.append("\\begin{table}[t]")
    L.append("\\centering")
    L.append("\\small")
    L.append("\\begin{tabular}{lccc}")
    L.append("\\toprule")
    L.append("Labeled encoders & Spearman $\\rho$ & Pearson $r$ "
             "& Top-1 \\\\")
    L.append("\\midrule")
    for i, k in enumerate(k_list):
        L.append("%d & $%.3f \\pm %.3f$ & $%.3f \\pm %.3f$ & $%.2f \\pm %.2f$ \\\\"
                 % (k, rho[QWEN25][0][i], rho[QWEN25][1][i],
                    pear[QWEN25][0][i], pear[QWEN25][1][i],
                    top1[QWEN25][0][i], top1[QWEN25][1][i]))
    L.append("\\midrule")
    L.append("full info (5-fold) & $%.3f \\pm %.3f$ & $%.3f \\pm %.3f$ "
             "& $%.2f \\pm %.2f$ \\\\"
             % (ref[QWEN25]["rho"][0], ref[QWEN25]["rho"][1],
                ref[QWEN25]["pearson"][0], ref[QWEN25]["pearson"][1],
                ref[QWEN25]["top1"][0], ref[QWEN25]["top1"][1]))
    L.append("\\bottomrule")
    L.append("\\end{tabular}")
    L.append("\\caption{Data efficiency of CKA-X (gradient-boosted regressor, "
             "fold-mean caliber) on the Qwen2.5-1.5B target: mean $\\pm$ std "
             "over 100 random splits per labeled-set size; the last row is the "
             "full-information 5-fold reference.}")
    L.append("\\label{tab:budget_sweep_gbdt_qwen25}")
    L.append("\\end{table}")
    with open(os.path.join(outdir, "budget_sweep_table.tex"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

    # ---------------- markdown: both calibers ----------------
    M = ["# CKA-X (GBDT) labeling-budget sweep",
         "",
         "- protocol: 100 random splits per labeled-set size, seed 0; the "
         "labeled encoders form the reference pool and the remaining encoders "
         "are held out (the full pool, unified text encoder)",
         "- fold-mean = per-split metric, then mean +/- std over the 100 splits "
         "(per-split caliber)",
         "- pooled = all held-out predictions of all splits concatenated, one "
         "metric over n_splits*(N - labeled) points",
         ""]
    M += md_block(k_list, rho, "Spearman rho - fold-mean (mean +/- std)")
    M += md_block(k_list, pooled_rho, "Spearman rho - pooled")
    M.append("## Pearson r - fold-mean (mean +/- std)")
    M.append("")
    M.append("| labeled encoders | " + " | ".join(LABEL[b] for b in TABLE_ORDER) + " |")
    M.append("|" + "---|" * (len(TABLE_ORDER) + 1))
    for i, k in enumerate(k_list):
        M.append(f"| {k} | " + " | ".join(
            fmt(pear[bb][0][i], pear[bb][1][i]) for bb in TABLE_ORDER) + " |")
    M.append("| full info (5-fold) | " + " | ".join(
        fmt(ref[bb]["pearson"][0], ref[bb]["pearson"][1])
        for bb in TABLE_ORDER) + " |")
    M.append("")
    M.append("## Top-1 - fold-mean (mean +/- std; GT score of the top-ranked "
             "held-out encoder)")
    M.append("")
    M.append("| labeled encoders | " + " | ".join(LABEL[b] for b in TABLE_ORDER) + " |")
    M.append("|" + "---|" * (len(TABLE_ORDER) + 1))
    for i, k in enumerate(k_list):
        M.append(f"| {k} | " + " | ".join(
            fmt(top1[bb][0][i], top1[bb][1][i], 2) for bb in TABLE_ORDER) + " |")
    M.append("| full info (5-fold) | " + " | ".join(
        fmt(ref[bb]["top1"][0], ref[bb]["top1"][1], 2)
        for bb in TABLE_ORDER) + " |")
    with open(os.path.join(outdir, "budget_sweep_table.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(M) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", type=str, default=os.path.join(
        PROJ_DIR, "results", "reference", "budget_sweep.json"))
    ap.add_argument("--ref", type=str, default=os.path.join(
        PROJ_DIR, "results", "reference", "fullinfo_gbdt.json"))
    ap.add_argument("--outdir", type=str, default=PROJ_DIR)
    ap.add_argument("--caliber", choices=["fold_mean", "pooled"],
                    default="fold_mean")
    ap.add_argument("--basename", type=str, default=None,
                    help="default: budget_sweep_gbdt_<caliber>")
    ap.add_argument("--fs_label", type=float, default=23)
    ap.add_argument("--fs_tick", type=float, default=20)
    ap.add_argument("--fs_legend", type=float, default=19)
    ap.add_argument("--marker_size", type=float, default=8.5)
    ap.add_argument("--line_width", type=float, default=2.2)
    ap.add_argument("--ylim", type=str, default=None)
    ap.add_argument("--figsize", type=str, default="7.4,5.2",
                    help="figure size in inches, e.g. 7.4,4.2 for a flatter "
                         "figure that costs less vertical space at the same "
                         "linewidth (font point sizes are unchanged)")
    ap.add_argument("--no_err", action="store_true",
                    help="omit the std error bars (fold-mean caliber)")
    ap.add_argument("--legend_frame", action="store_true",
                    help="draw the legend inside a translucent floating box "
                         "(default: no frame)")
    ap.add_argument("--legend_above", action="store_true",
                    help="horizontal legend centred above the axes (useful for "
                         "small figures where the lower-right corner collides "
                         "with error bars)")
    ap.add_argument("--legend_alpha", type=float, default=0.75,
                    help="legend frame transparency with --legend_frame")
    ap.add_argument("--legend_round", action="store_true",
                    help="rounded legend corners instead of a plain rectangle")
    ap.add_argument("--xlabel", type=str,
                    default="Number of labeled encoders",
                    help="x-axis label (no symbol: the "
                         "notations differ)")
    args = ap.parse_args()

    if args.basename is None:
        tag = "foldmean" if args.caliber == "fold_mean" else "pooled"
        args.basename = f"budget_sweep_{tag}"
    if args.ylim is None:
        args.ylim = "0.25,1.05" if args.caliber == "fold_mean" else "0.30,0.90"

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "mathtext.fontset": "stix",
        "font.size": args.fs_tick,
        "axes.labelsize": args.fs_label,
        "xtick.labelsize": args.fs_tick,
        "ytick.labelsize": args.fs_tick,
        "legend.fontsize": args.fs_legend,
        "axes.linewidth": 1.3,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "savefig.dpi": 400,
    })

    sweep = load(args.sweep)
    set_backbones((sweep.get("backbones") or {}).keys())
    k_list = list(sweep["protocol"]["k_sweep"])
    ks = np.array(k_list, dtype=float)

    rho = collect_fold_mean(sweep, "spearman")
    pear = collect_fold_mean(sweep, "pearson")
    top1 = collect_fold_mean(sweep, "top1")
    pooled_rho = collect_pooled(sweep, "spearman")

    # full-information 5-fold reference
    ref = {}
    if os.path.isfile(args.ref):
        d = load(args.ref)
        for bb in KEYS:
            for key, v in d["results"].items():
                if key.split("|")[-1] == bb:
                    ref[bb] = {
                        "rho": (v["fold_mean_rho"][0], v["fold_mean_rho"][1]),
                        "pearson": (v["fold_mean_pearson"][0],
                                    v["fold_mean_pearson"][1]),
                        "top1": (v["fold_mean_top1_gt"][0],
                                 v["fold_mean_top1_gt"][1])}
                    break
    if not ref:
        print("  [WARN] no full-information reference found")

    # ---------------- figure ----------------
    figw, figh = [float(x) for x in args.figsize.split(",")]
    fig, ax = plt.subplots(figsize=(figw, figh))
    if args.caliber == "pooled":
        for bb, color, marker in PLOT_SERIES:
            ax.plot(ks, pooled_rho[bb], "-", color=color, marker=marker,
                    markersize=args.marker_size, linewidth=args.line_width, label=LABEL[bb])
    else:
        for bb, color, marker in PLOT_SERIES:
            m, s = rho[bb]
            if args.no_err:
                ax.plot(ks, m, "-", color=color, marker=marker, markersize=args.marker_size, linewidth=args.line_width, label=LABEL[bb])
            else:
                ax.errorbar(ks, m, yerr=s, fmt="-" + marker, color=color,
                            markersize=args.marker_size, linewidth=args.line_width,
                            elinewidth=0.5 * args.line_width,
                            capsize=0.5 * args.marker_size,
                            capthick=0.5 * args.line_width, label=LABEL[bb])
    ax.set_xlabel(args.xlabel)
    ax.set_ylabel("Spearman")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in k_list])
    lo, hi = [float(x) for x in args.ylim.split(",")]
    ax.set_ylim(lo, hi)
    # y ticks: 0.1 spacing restricted to the visible range, two decimals
    from matplotlib.ticker import FormatStrFormatter
    yticks = np.round(np.arange(np.ceil(lo / 0.1) * 0.1, hi + 1e-9, 0.1), 2)
    ax.set_yticks(yticks)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.grid(True, which="major", linestyle=":", linewidth=0.8, alpha=0.35)
    ax.set_axisbelow(True)
    # reference-figure style: left/bottom spines only; legend without a frame
    # by default (pass --legend_frame for the boxed translucent variant)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    legend_kw = dict(borderpad=0.45, labelspacing=0.35,
                     handlelength=1.7, handletextpad=0.6, borderaxespad=0.7)
    if args.legend_above:
        # horizontal legend centred above the axes: avoids collisions with
        # error bars in small figures
        legend_kw.update(loc="lower center", bbox_to_anchor=(0.5, 1.02),
                         ncol=3, borderaxespad=0.2)
    else:
        legend_kw["loc"] = "lower right"
    if args.legend_frame:
        ax.legend(frameon=True, fancybox=args.legend_round,
                  facecolor="white", edgecolor="0.6",
                  framealpha=args.legend_alpha, shadow=True, **legend_kw)
    else:
        ax.legend(frameon=False, **legend_kw)
    fig.tight_layout()

    os.makedirs(args.outdir, exist_ok=True)
    for ext in ("png", "pdf"):
        p = os.path.join(args.outdir, f"{args.basename}.{ext}")
        fig.savefig(p, bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.close(fig)

    if ref:
        write_tables(args.outdir, k_list, ref, rho, pear, top1, pooled_rho)
        for n in ("budget_sweep_table.tex", "budget_sweep_table.md"):
            print(f"  Saved: {os.path.join(args.outdir, n)}")

    # ---------------- console table ----------------
    print(f"\n  Spearman, {args.caliber} (GBDT)")
    print("    k'".ljust(7) + "".join(f"{LABEL[b]:>22s}" for b in TABLE_ORDER))
    for i, k in enumerate(k_list):
        line = f"    {k:<4d} "
        for bb in TABLE_ORDER:
            if args.caliber == "pooled":
                line += f"{fmt(pooled_rho[bb][i]):>22s}"
            else:
                line += f"{fmt(rho[bb][0][i], rho[bb][1][i]):>22s}"
        print(line)
    if ref and args.caliber == "fold_mean":
        line = "    full  "
        for bb in TABLE_ORDER:
            line += f"{fmt(ref[bb]['rho'][0], ref[bb]['rho'][1]):>22s}"
        print(line)


if __name__ == "__main__":
    main()


