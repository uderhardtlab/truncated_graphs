"""
Figure 5: violin plots comparing raw / bosperrus-corrected / SERN-corrected
centrality correlations against the uncropped ("original") sphere benchmark.
One figure per correlation method (Pearson, Spearman) -- see sphere.py's
CORRELATION_METHODS.

Run:
    cd truncated_graphs/
    pixi run python src/figure5/plot_violins.py
"""
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parent.parent.parent

METHOD_PALETTE = {"raw": "grey", "bosperrus": "#8ABBD6", "SERN": "#E59EDD"}
METHOD_ORDER = ["raw", "bosperrus", "SERN"]
ROW_ORDER = ["delaunay", "knn", "rnn"]
ROW_LABELS = {"delaunay": "Delaunay", "knn": "$k$-NN", "rnn": "$r$-NN"}
CORR_LABEL = {"pearson": "Pearson $r$", "spearman": "Spearman $\\rho$"}
FIG_WIDTH = 9  # this manuscript's full text width (matches figure3.svg)

# Adjacent/spanning pairs, ordered (worse, better) by assumed correction
# quality; the MWU test below is one-sided on this ordering, since we only
# care whether the second method's correlations are significantly *larger*.
SIG_PAIRS = [
    ("raw", "bosperrus", 0, 1, 1.04),   # adjacent, lowest bracket
    ("bosperrus", "SERN", 1, 2, 1.11),  # adjacent
    ("raw", "SERN",       0, 2, 1.19),  # spanning, highest bracket
]
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
    dfs = [
        pd.read_csv(f).rename(columns={"Unnamed: 0": "measure"})
        for f in glob.glob(str(ROOT / "results" / "figure5" / "*.csv"))
    ]
    all_dfs = pd.concat(dfs)
    all_dfs = all_dfs[all_dfs["measure"].isin(
        ["betweenness", "degree", "closeness", "clustering", "pagerank"]
    )].copy()

    all_dfs = all_dfs.rename(columns={
        "original vs. on crop": "raw",
        "original vs. BOSPERRUS corrected on crop": "bosperrus",
        "original vs. SERN corrected on crop": "SERN",
    })
    all_dfs["radius"] = all_dfs["radius"].fillna(0)
    all_dfs["k"] = all_dfs["k"].fillna(0)

    for col in ("bosperrus", "SERN", "raw"):
        all_dfs[col] = np.abs(all_dfs[col])

    all_dfs = all_dfs.reset_index(drop=True)
    all_dfs = all_dfs.iloc[all_dfs[["bosperrus", "SERN", "raw"]].dropna(how="all").index]
    return all_dfs


def make_violin_plot(all_dfs, corr_method, out_dir):
    df_method = all_dfs[all_dfs["corr_method"] == corr_method]
    df_long = df_method.melt(
        id_vars=["measure", "graph_type", "coord_type"],
        value_vars=["raw", "bosperrus", "SERN"],
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
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)

    # Paired lines + one-sided MWU significance brackets per facet
    rng = np.random.default_rng(42)
    for row_idx, row_key in enumerate(ROW_ORDER):
        for col_idx, col_key in enumerate(measure_order):
            ax = g.axes[row_idx][col_idx]
            subset = df_method[
                (df_method["graph_type"] == row_key) &
                (df_method["measure"] == col_key)
            ][["raw", "bosperrus", "SERN"]]

            # --- paired lines (sampled subset for readability) ---
            s = subset.dropna(how="all")
            idx = rng.choice(len(s), size=min(N_SAMPLE, len(s)), replace=False)
            sample = s.iloc[idx]

            xs_all, ys_all = [], []
            for _, row_data in sample.iterrows():
                pts = [(i, v) for i, v in enumerate(
                    [row_data["raw"], row_data["bosperrus"], row_data["SERN"]]
                ) if not np.isnan(v)]
                if len(pts) >= 2:
                    xs_all.extend([p[0] for p in pts] + [np.nan])
                    ys_all.extend([p[1] for p in pts] + [np.nan])
            if xs_all:
                ax.plot(xs_all, ys_all, color="k", alpha=0.05, lw=0.4,
                        zorder=0, solid_capstyle="round")

            # --- one-sided MWU significance brackets: is the second method's
            # correlation significantly *larger* than the first's? ---
            for a, b, x1, x2, y in SIG_PAIRS:
                va = subset[a].dropna()
                vb = subset[b].dropna()
                if len(va) > 1 and len(vb) > 1:
                    _, p = mannwhitneyu(va, vb, alternative="less")
                    _draw_bracket(ax, x1, x2, y, _get_stars(p))

    g.set(ylim=(-0.05, 1.33))
    g.figure.tight_layout()
    out = out_dir / f"correlations_violin_{corr_method}.pdf"
    g.figure.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(g.figure)


def main():
    all_dfs = load_data()
    out_dir = ROOT / "result_plots" / "figure5"
    out_dir.mkdir(parents=True, exist_ok=True)
    for corr_method in ["pearson", "spearman"]:
        make_violin_plot(all_dfs, corr_method, out_dir)


if __name__ == "__main__":
    main()
