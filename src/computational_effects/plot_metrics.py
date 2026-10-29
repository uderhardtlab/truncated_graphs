"""
Figure 3: border-effect fit summary across measures, for one representative
graph type per family (Delaunay, kNN k=10, rNN with the radius matched to
kNN k=10 -- the graph types within a family are similar enough that showing
all 11 was redundant).

Columns are measures, ordered from local to global; PageRank is left out
(its effect follows degree on every graph type, and spatial omics tools don't
use it -- its fits are still in the CSVs). Rows are graph families.

Above each column: a map of one example image coloured by that measure, with
image and graph type chosen in example_scores.py. Colours are clipped to the
image's 2nd-98th percentile; the colour bar has no numbers, since only the
spatial pattern matters.

Each (measure, family) cell is a mini "jointplot":
  - main panel: every non-Constant dataset's fitted curve. y is anchored to
    observed_effect_strength (not min-max normalized), so y(0) at the
    border equals observed_effect_strength exactly and y(d_max) at the
    interior is always 0 -- sign and magnitude are both real, comparable
    quantities, not an artifact of rescaling. Constant Fit rows are flat
    lines at y=0 by definition and aren't drawn here (indistinguishable
    from the zero-reference line already shown).
  - top margin: distribution of observed_half_life, split by winning fit
    type (including Constant Fit, as a spike -- see below).
  - right margin: same idea for observed_effect_strength.

Every marginal curve/spike is scaled by that fit type's share of the
cell's total dataset count (not a bare gaussian_kde, which always
integrates to 1 regardless of sample size) -- so relative area under each
curve directly reads as "how often did this fit type win," including
Constant Fit (no border effect detected). Constant Fit's half_life and
effect_strength are identically 0 for every such row (no spread, so no
real KDE is possible) and are drawn as a narrow spike at 0 instead, with
the same area-equals-share convention.

The y-axis is shared within a column, so graph families are directly
comparable for one measure, but not across columns, since different measures
have very different natural effect magnitudes (one shared axis would let
betweenness squash degree and clustering to a line). Each column scales
symmetrically around 0 to the largest 97th-percentile |effect_strength| of its
cells (not the true max, which a handful of sign-crossing outliers could
otherwise inflate).

Run:
    cd truncated_graphs/
    pixi run python src/computational_effects/plot_metrics.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"  # keep SVG text as editable <text>, not outlined paths
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde, norm

from bosperrus.fit import PiecewiseLinearFit, ExponentialSaturationFit, MichaelisMentenFit

ROOT = Path(__file__).resolve().parent.parent.parent

MEASURES = ["degree", "clustering", "closeness", "betweenness"]

REPRESENTATIVE = {"Delaunay": "delaunay", "kNN (k=10)": "knn_k=10", "rNN (mean deg. 10)": "rnn_k=10"}
FAMILIES = list(REPRESENTATIVE.keys())
FAMILY_OF = {gt: family for family, gt in REPRESENTATIVE.items()}

CURVE_FIT_ORDER = ["Piecewise Linear Fit", "Exponential Saturation Fit", "Michaelis-Menten Fit"]

# coordinates are scaled so the longer image side is 1 before
# distance_to_convex_hull; the images are near-square (median aspect 1.001), so
# d_max is consistent across datasets (measured 0.483 +/- 0.017 over an
# 80-dataset random sample) -- used as a shared
# stand-in for each dataset's own d_max, since it isn't stored in the
# per-graph-type CSVs.
D_MAX_APPROX = 0.483
X_AXIS_MAX = 0.5  # displayed x range; curves/densities still stop at D_MAX_APPROX (no extrapolation)
_EPS = 1e-10
# narrow gaussian standing in for Constant Fit's zero-variance spike, as a
# fraction of each margin's own range
SPIKE_SIGMA_FRAC = 0.02

CURVE_FN = {
    "Piecewise Linear Fit": lambda d, r: PiecewiseLinearFit.piecewise_plateau(d, r["piecewise_linear_b"], r["piecewise_linear_m"], r["piecewise_linear_c"]),
    "Exponential Saturation Fit": lambda d, r: ExponentialSaturationFit.exp_sat(d, r["exponential_saturation_a"], r["exponential_saturation_b"], r["exponential_saturation_c"]),
    "Michaelis-Menten Fit": lambda d, r: MichaelisMentenFit.michaelis_menten(d, r["michaelis_menten_a"], r["michaelis_menten_b"], r["michaelis_menten_c"]),
}

X_GRID = np.linspace(0, D_MAX_APPROX, 60)
HALFLIFE_GRID = np.linspace(0, D_MAX_APPROX, 100)
LINE_ALPHA = 0.035


def load_data():
    """Read the per-graph-type fit_quality CSVs written by compute_fits.py for
    the representative graph types, restricted to MEASURES. Includes Constant
    Fit rows (unlike earlier revisions) -- they're needed to compute each fit
    type's true share of the cell's datasets."""
    dfs = []
    for gt in REPRESENTATIVE.values():
        df = pd.read_csv(ROOT / "results" / "computational_effects" / "mibitof_fits" / f"{gt}_graph_level_fits.csv", index_col=0)
        df["graph_type"] = gt
        df["measure"] = df.index
        dfs.append(df)
    combined = pd.concat(dfs).reset_index(drop=True)
    return combined[combined["measure"].isin(MEASURES)].copy()


def set_three_ticks(ax):
    """Exactly 3 evenly-spaced, labeled ticks per axis (min/mid/max of the
    current view limits) -- call after xlim/ylim are finalized."""
    for set_ticks, set_labels, get_lim in (
        (ax.set_xticks, ax.set_xticklabels, ax.get_xlim),
        (ax.set_yticks, ax.set_yticklabels, ax.get_ylim),
    ):
        lo, hi = get_lim()
        ticks = np.linspace(lo, hi, 3)
        set_ticks(ticks)
        set_labels([f"{t:.2g}" for t in ticks])


def kde_or_none(values, grid):
    """gaussian_kde requires >1 distinct value; returns None for degenerate
    (too few, or zero-variance) inputs rather than raising."""
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if len(values) < 5 or values.std() < 1e-8:
        return None
    return gaussian_kde(values)(grid)


def y_abs_max(sub):
    """Half-height of a cell's y-axis."""
    non_baseline = sub[sub["best_fit_type"] != "Constant Fit"]
    # 97th percentile rather than the true max: observed_effect_strength is only
    # bounded to [-1, 1] when the border and center values share a sign -- a
    # curve that crosses zero between them can spike much further out, and a
    # handful of such outliers would otherwise dictate the whole axis, squeezing
    # everything else into a fraction of the panel.
    y = np.percentile(non_baseline["observed_effect_strength"].abs(), 97) if len(non_baseline) else 0.05
    return max(y, 0.05) * 1.1


def plot_cell(ax_main, ax_top, ax_right, sub, fit_palette, y_max):
    n_total = len(sub)
    non_baseline = sub[sub["best_fit_type"] != "Constant Fit"]

    segments_by_model = {m: [] for m in CURVE_FIT_ORDER}
    for _, row in non_baseline.iterrows():
        model = row["best_fit_type"]
        S = CURVE_FN[model](X_GRID, row)
        c_border, c_center = S[0], S[-1]
        shape = (S - c_center) / (c_border - c_center + _EPS)
        segments_by_model[model].append(np.column_stack([X_GRID, shape * row["observed_effect_strength"]]))

    es_grid = np.linspace(-y_max, y_max, 100)
    spike_sigma_x = SPIKE_SIGMA_FRAC * D_MAX_APPROX
    spike_sigma_y = SPIKE_SIGMA_FRAC * y_max

    for model in CURVE_FIT_ORDER:
        segs = segments_by_model[model]
        if segs:
            color = to_rgba(fit_palette[model], alpha=LINE_ALPHA)
            lc = LineCollection(segs, colors=[color] * len(segs), linewidths=0.8)
            lc.set_rasterized(True)  # thousands of paths per panel -- keep vector output size sane
            ax_main.add_collection(lc)

        model_rows = sub[sub["best_fit_type"] == model]
        frac = len(model_rows) / n_total if n_total else 0
        if frac == 0:
            continue

        hl_density = kde_or_none(model_rows["observed_half_life"] * D_MAX_APPROX, HALFLIFE_GRID)
        if hl_density is not None:
            hl_density = hl_density * frac
            ax_top.fill_between(HALFLIFE_GRID, hl_density, color=fit_palette[model], alpha=0.35, lw=0)
            ax_top.plot(HALFLIFE_GRID, hl_density, color=fit_palette[model], lw=1)

        es_density = kde_or_none(model_rows["observed_effect_strength"], es_grid)
        if es_density is not None:
            es_density = es_density * frac
            ax_right.fill_betweenx(es_grid, es_density, color=fit_palette[model], alpha=0.35, lw=0)
            ax_right.plot(es_density, es_grid, color=fit_palette[model], lw=1)

    # Constant Fit: half_life and effect_strength are identically 0 (no spread
    # -> no real KDE possible), represented as a narrow spike whose area still
    # equals its share.
    n_const = (sub["best_fit_type"] == "Constant Fit").sum()
    frac_const = n_const / n_total if n_total else 0
    if frac_const > 0:
        spike_x = frac_const * norm(0, spike_sigma_x).pdf(HALFLIFE_GRID)
        ax_top.fill_between(HALFLIFE_GRID, spike_x, color=fit_palette["Constant Fit"], alpha=0.5, lw=0)
        ax_top.plot(HALFLIFE_GRID, spike_x, color=fit_palette["Constant Fit"], lw=1)

        spike_y = frac_const * norm(0, spike_sigma_y).pdf(es_grid)
        ax_right.fill_betweenx(es_grid, spike_y, color=fit_palette["Constant Fit"], alpha=0.5, lw=0)
        ax_right.plot(spike_y, es_grid, color=fit_palette["Constant Fit"], lw=1)

    ax_main.axhline(0, color="#898781", lw=0.7, zorder=0)
    ax_main.set_xlim(0, X_AXIS_MAX)
    ax_main.set_ylim(-y_max, y_max)
    ax_main.spines[["top", "right"]].set_visible(False)
    ax_main.tick_params(labelsize=6)
    set_three_ticks(ax_main)
    for ax in (ax_top, ax_right):
        ax.axis("off")


def plot_example(ax_map, ax_cbar, ex):
    """Map of the example image coloured by one measure, clipped to its
    2nd-98th percentile; y inverted to match image orientation."""
    vmin, vmax = np.percentile(ex["score"], [2, 98])
    sc = ax_map.scatter(ex["x"], ex["y"], c=ex["score"].clip(vmin, vmax), cmap="viridis",
                        vmin=vmin, vmax=vmax, s=0.6, lw=0)  # vector: stays sharp when zoomed
    ax_map.set_aspect("equal")
    ax_map.invert_yaxis()
    ax_map.axis("off")
    ax_map.text(0.5, -0.02, FAMILY_OF[ex["graph_type"].iloc[0]], transform=ax_map.transAxes,
                ha="center", va="top", fontsize=6.5)
    cbar = plt.colorbar(sc, cax=ax_cbar, ticks=[])
    cbar.outline.set_linewidth(0.4)
    ax_cbar.text(0.5, 1.02, "high", transform=ax_cbar.transAxes, ha="center", va="bottom", fontsize=5.5)
    ax_cbar.text(0.5, -0.02, "low", transform=ax_cbar.transAxes, ha="center", va="top", fontsize=5.5)


def plot_metrics(data, examples, fit_palette):
    n_rows, n_cols = len(FAMILIES), len(MEASURES)
    fig = plt.figure(figsize=(7.5, 6.6))
    # row 0: example maps; then a gap row; then (top margin, main) per family
    gs = GridSpec(n_rows * 2 + 2, n_cols * 2, figure=fig,
                  height_ratios=[3.2, 0.35] + [0.6, 2.6] * n_rows, width_ratios=[3.2, 0.7] * n_cols,
                  hspace=0.06, wspace=0.08)

    for ci, measure in enumerate(MEASURES):
        ax_map = fig.add_subplot(gs[0, ci * 2])
        cbar_gs = gs[0, ci * 2 + 1].subgridspec(3, 3, height_ratios=[0.15, 0.7, 0.15], width_ratios=[0.2, 0.25, 0.55])
        plot_example(ax_map, fig.add_subplot(cbar_gs[1, 1]), examples[examples["measure"] == measure])
        ax_map.set_title(measure, fontsize=9, pad=4)

        sub_col = data[data["measure"] == measure]
        y_max = max(y_abs_max(sub_col[sub_col["graph_type"] == gt]) for gt in REPRESENTATIVE.values())
        for ri, family in enumerate(FAMILIES):
            row = 2 + ri * 2  # gridspec rows shifted by the example row + gap
            ax_main = fig.add_subplot(gs[row + 1, ci * 2])
            ax_top = fig.add_subplot(gs[row, ci * 2], sharex=ax_main)
            ax_right = fig.add_subplot(gs[row + 1, ci * 2 + 1], sharey=ax_main)

            sub = sub_col[sub_col["graph_type"] == REPRESENTATIVE[family]]
            plot_cell(ax_main, ax_top, ax_right, sub, fit_palette, y_max)

            if ci == 0:
                ax_main.set_ylabel(family, fontsize=8, labelpad=4)
            if ri == n_rows - 1:
                ax_main.set_xlabel("distance to border", fontsize=6.5)
            else:
                plt.setp(ax_main.get_xticklabels(), visible=False)

    return fig


def main():
    with open(ROOT / "fit_palette.json") as f:
        fit_palette = json.load(f)

    data = load_data()
    examples = pd.read_csv(ROOT / "results" / "computational_effects" / "mibitof_fits" / "figure3_examples.csv")
    fig = plot_metrics(data, examples, fit_palette)

    out = ROOT / "result_plots" / "computational_effects" / "mibitof_fits" / "figure3"
    fig.savefig(str(out) + ".svg", format="svg", bbox_inches="tight", dpi=600)  # dpi of the rasterized curve layers
    print(f"Saved {out}.svg")
    plt.close("all")


if __name__ == "__main__":
    main()
