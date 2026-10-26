"""
Figure 5: violin plots comparing raw / detrended (linear, spline) / bosperrus-corrected (additive,
multiplicative and log, see sphere.py) / SERN-corrected centrality correlations against the uncropped
("original") sphere benchmark. One figure per correlation method (Pearson,
Spearman) -- see sphere.py's CORRELATION_METHODS -- and node subset ("all"
crop nodes; "affected" = nearest quartile to the border, written as a
separate figure with an "_affected" suffix).

Run:
    cd truncated_graphs/
    pixi run python src/correction_benchmark/plot_violins.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu

from load_results import load_sphere

ROOT = Path(__file__).resolve().parent.parent.parent

METHOD_PALETTE = {"raw": "grey", "linear": "#D9C27E", "spline": "#A8842A", "bosperrus": "#8ABBD6", "bosperrus_mult": "#3A7DB0", "bosperrus_log": "#1B3F66",
                  "SERN": "#E59EDD"}
METHOD_ORDER = ["raw", "linear", "spline", "bosperrus", "bosperrus_mult", "bosperrus_log", "SERN"]
METHOD_LABELS = {"raw": "raw", "linear": "linear detr.", "spline": "spline detr.", "bosperrus": "bosperrus add.", "bosperrus_mult": "bosperrus mult.",
                 "bosperrus_log": "bosperrus log", "SERN": "SERN"}
ROW_ORDER = ["delaunay", "delaunay_pruned", "knn", "rnn"]
ROW_LABELS = {"delaunay": "Delaunay", "delaunay_pruned": "Delaunay\n(pruned)", "knn": "$k$-NN", "rnn": "$r$-NN"}
CORR_LABEL = {"pearson": "Pearson $r$", "spearman": "Spearman $\\rho$"}
FIG_WIDTH = 9  # this manuscript's full text width (matches figure3.svg)

# Pairs (a, b, bracket height, MWU alternative). Pairs ordered (worse, better)
# by assumed correction quality are one-sided ("less": is b's correlation
# significantly *larger* than a's?); additive vs. multiplicative, additive vs.
# log and detrending vs. bosperrus have no assumed order, so they are two-sided.
SIG_PAIRS = [
    ("raw", "linear",               1.04, "less"),
    ("spline", "bosperrus",         1.04, "two-sided"),
    ("bosperrus_mult", "bosperrus_log", 1.04, "two-sided"),
    ("raw", "spline",               1.14, "less"),
    ("bosperrus", "bosperrus_mult", 1.14, "two-sided"),
    ("bosperrus_log", "SERN",       1.14, "less"),
    ("bosperrus", "bosperrus_log",  1.24, "two-sided"),
    ("spline", "bosperrus_log",     1.34, "two-sided"),
    ("raw", "bosperrus_log",        1.44, "less"),
    ("raw", "SERN",                 1.54, "less"),
]
METHODS = METHOD_ORDER
N_SAMPLE = 200


def _get_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def _draw_bracket(ax, x1, x2, y, stars, fontsize=7):
    h = 0.025
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y],
            lw=0.8, color="k", clip_on=False)
    ax.text((x1 + x2) / 2, y + h, stars,
            ha="center", va="bottom", fontsize=fontsize, clip_on=False)


def load_data():
    all_dfs = load_sphere(ROOT / "results" / "correction_benchmark")
    all_dfs = all_dfs[all_dfs["measure"].isin(
        ["betweenness", "degree", "closeness", "clustering", "pagerank"]
    )].copy()

    all_dfs = all_dfs.rename(columns={
        "original vs. on crop": "raw",
        "original vs. BOSPERRUS corrected on crop": "bosperrus",
        "original vs. BOSPERRUS multiplicative on crop": "bosperrus_mult",
        "original vs. BOSPERRUS log on crop": "bosperrus_log",
        "original vs. linear detrended on crop": "linear",
        "original vs. spline detrended on crop": "spline",
        "original vs. SERN corrected on crop": "SERN",
    })
    all_dfs["radius"] = all_dfs["radius"].fillna(0)
    # r-NN r = 0.05 is excluded from figure 5 (fragmented crop graph; see sphere.py), also for runs
    # made before it was removed from sphere.py
    all_dfs = all_dfs[~((all_dfs["graph_type"] == "rnn") & np.isclose(all_dfs["radius"], 0.05))]
    all_dfs["k"] = all_dfs["k"].fillna(0)

    if "node_subset" not in all_dfs:
        all_dfs["node_subset"] = "all"
    for col in METHODS:
        all_dfs[col] = np.abs(all_dfs[col])

    all_dfs = all_dfs.reset_index(drop=True)
    all_dfs = all_dfs.iloc[all_dfs[METHODS].dropna(how="all").index]
    return all_dfs


def make_violin_plot(all_dfs, corr_method, out_dir, node_subset="all"):
    df_method = all_dfs[(all_dfs["corr_method"] == corr_method) & (all_dfs["node_subset"] == node_subset)]
    df_long = df_method.melt(
        id_vars=["measure", "graph_type", "coord_type"],
        value_vars=METHODS,
        var_name="method",
        value_name="correlation",
    ).dropna(subset=["correlation"])
    measure_order = sorted(df_long["measure"].unique())

    g = sns.catplot(
        data=df_long,
        x="method", y="correlation", hue="method",
        col="measure", row="graph_type",
        kind="violin",
        order=METHOD_ORDER,
        row_order=ROW_ORDER,
        col_order=measure_order,
        palette=METHOD_PALETTE,
        inner="box",
        cut=0,
        linewidth=0.8,
        height=2.5, aspect=0.9,
        sharey=True,
        legend=False,
    )
    # Rescale to the manuscript's full text width, preserving proportions.
    w, h = g.figure.get_size_inches()
    g.figure.set_size_inches(FIG_WIDTH, h * FIG_WIDTH / w)

    g.set_titles("")
    for ax, col in zip(g.axes[0], measure_order):
        ax.set_title(col.capitalize(), fontsize=10)
    for ax in g.axes.flat:
        ax.set_xlabel("")
        ax.set_ylabel("")
    for ax_row, row_key in zip(g.axes[:, 0], ROW_ORDER):
        ax_row.set_ylabel(f"{ROW_LABELS[row_key]}\n{CORR_LABEL[corr_method]}")
    for ax in g.axes[-1]:
        ax.set_xticks(range(len(METHOD_ORDER)))
        ax.set_xticklabels([METHOD_LABELS[m] for m in METHOD_ORDER], rotation=40, ha="right", fontsize=8)

    # Paired lines + one-sided MWU significance brackets per facet
    rng = np.random.default_rng(42)
    for row_idx, row_key in enumerate(ROW_ORDER):
        for col_idx, col_key in enumerate(measure_order):
            ax = g.axes[row_idx][col_idx]
            subset = df_method[
                (df_method["graph_type"] == row_key) &
                (df_method["measure"] == col_key)
            ][METHODS]

            # --- paired lines (sampled subset for readability) ---
            s = subset.dropna(how="all")
            idx = rng.choice(len(s), size=min(N_SAMPLE, len(s)), replace=False)
            sample = s.iloc[idx]

            xs_all, ys_all = [], []
            for _, row_data in sample.iterrows():
                pts = [(i, v) for i, v in enumerate(
                    [row_data[m] for m in METHODS]
                ) if not np.isnan(v)]
                if len(pts) >= 2:
                    xs_all.extend([p[0] for p in pts] + [np.nan])
                    ys_all.extend([p[1] for p in pts] + [np.nan])
            if xs_all:
                ax.plot(xs_all, ys_all, color="k", alpha=0.05, lw=0.4,
                        zorder=0, solid_capstyle="round")

            # --- one-sided MWU significance brackets: is the second method's
            # correlation significantly *larger* than the first's? ---
            for a, b, y, alternative in SIG_PAIRS:
                va = subset[a].dropna()
                vb = subset[b].dropna()
                if len(va) > 1 and len(vb) > 1:
                    _, p = mannwhitneyu(va, vb, alternative=alternative)
                    _draw_bracket(ax, METHOD_ORDER.index(a), METHOD_ORDER.index(b), y, _get_stars(p))

    g.set(ylim=(-0.05, 1.68), yticks=[0, 0.5, 1])
    g.figure.tight_layout()
    suffix = "" if node_subset == "all" else f"_{node_subset}"
    out = out_dir / f"correlations_violin_{corr_method}{suffix}.pdf"
    g.figure.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(g.figure)


def main():
    all_dfs = load_data()
    out_dir = ROOT / "result_plots" / "correction_benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    for corr_method in ["pearson", "spearman"]:
        for node_subset in sorted(all_dfs["node_subset"].unique()):
            make_violin_plot(all_dfs, corr_method, out_dir, node_subset=node_subset)


if __name__ == "__main__":
    main()
