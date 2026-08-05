"""
Figure 5 diagnostic: does restricting to border-*affected* nodes (nearest
AFFECTED_QUANTILE of crop nodes to the border, by distance_to_cap -- see
sphere.py's node_subsets in process_coords) change the raw/bosperrus/SERN
correlation picture compared to using all crop nodes?

Correlating over all crop nodes dilutes the comparison with untouched
interior nodes; "affected" isolates where border truncation actually biases
a score, using a criterion that's geometric (independent of either
correction method's fit, and of the raw/corrected values themselves) so it
can't bias the comparison toward whichever method it favors.

Reads from results/figure5_affected_check/ (a small, separate run set from
the main results/figure5/ production data, which predates this node_subset
split and doesn't have the column). Spearman only, since that's the
correlation this project settled on (Rheinwalt et al. 2012's own choice).

Run:
    cd truncated_graphs/
    pixi run python src/figure5/plot_affected_check.py
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
RESULTS_DIR = ROOT / "results" / "figure5_affected_check"
CORR_METHOD = "spearman"

METHOD_PALETTE = {"raw": "grey", "bosperrus": "#8ABBD6", "SERN": "#E59EDD"}
METHOD_ORDER = ["raw", "bosperrus", "SERN"]
GRAPH_TYPE_ORDER = ["delaunay", "knn", "rnn"]
GRAPH_TYPE_LABELS = {"delaunay": "Delaunay", "knn": "$k$-NN", "rnn": "$r$-NN"}
SUBSET_ORDER = ["all", "affected"]
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
        for f in glob.glob(str(RESULTS_DIR / "*.csv"))
    ]
    all_dfs = pd.concat(dfs)
    all_dfs = all_dfs[all_dfs["measure"].isin(
        ["betweenness", "degree", "closeness", "clustering", "pagerank"]
    )].copy()
    all_dfs = all_dfs[all_dfs["corr_method"] == CORR_METHOD]

    all_dfs = all_dfs.rename(columns={
        "original vs. on crop": "raw",
        "original vs. BOSPERRUS corrected on crop": "bosperrus",
        "original vs. SERN corrected on crop": "SERN",
    })
    for col in ("bosperrus", "SERN", "raw"):
        all_dfs[col] = np.abs(all_dfs[col])

    all_dfs["row_key"] = (
        all_dfs["graph_type"].map(GRAPH_TYPE_LABELS.get) + " (" + all_dfs["node_subset"] + ")"
    )
    all_dfs = all_dfs.reset_index(drop=True)
    all_dfs = all_dfs.iloc[all_dfs[["bosperrus", "SERN", "raw"]].dropna(how="all").index]
    return all_dfs


def plot(all_dfs, out_dir):
    row_order = [f"{GRAPH_TYPE_LABELS[gt]} ({s})" for gt in GRAPH_TYPE_ORDER for s in SUBSET_ORDER]

    df_long = all_dfs.melt(
        id_vars=["measure", "row_key"],
        value_vars=["raw", "bosperrus", "SERN"],
        var_name="method",
        value_name="correlation",
    ).dropna(subset=["correlation"])
    measure_order = sorted(df_long["measure"].unique())

    g = sns.catplot(
        data=df_long,
        x="method", y="correlation", hue="method",
        col="measure", row="row_key",
        kind="violin",
        order=METHOD_ORDER,
        row_order=row_order,
        col_order=measure_order,
        palette=METHOD_PALETTE,
        inner="box",
        cut=0,
        linewidth=0.8,
        height=1.7, aspect=0.9,
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
    for ax_row, row_key in zip(g.axes[:, 0], row_order):
        ax_row.set_ylabel(row_key, fontsize=8)
    for ax in g.axes[-1]:
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)

    # Paired lines + one-sided MWU significance brackets per facet
    rng = np.random.default_rng(42)
    for row_idx, row_key in enumerate(row_order):
        for col_idx, col_key in enumerate(measure_order):
            ax = g.axes[row_idx][col_idx]
            subset = all_dfs[
                (all_dfs["row_key"] == row_key) &
                (all_dfs["measure"] == col_key)
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
    out = out_dir / f"affected_check_{CORR_METHOD}.pdf"
    g.figure.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(g.figure)


def main():
    all_dfs = load_data()
    out_dir = ROOT / "result_plots" / "figure5"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot(all_dfs, out_dir)


if __name__ == "__main__":
    main()
