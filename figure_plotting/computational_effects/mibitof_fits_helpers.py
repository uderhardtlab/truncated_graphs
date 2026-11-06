"""
Plotting code for figure_plotting/computational_effects/mibitof_fits.ipynb.

Border-effect fit summary across measures, for one representative
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

Bottom row: the downstream consequence on squidpy's IMC demo FOV (data from
imc_centrality.py, squidpy's default kNN k=6 graph). Under degree, clustering
and closeness: per-cell-type mean as % deviation from the FOV mean, raw (open)
vs. BOSPERRUS-corrected (filled) -- relative to the FOV mean so that the
correction's overall level shift doesn't hide the differences between cell
types. Under betweenness (not offered by squidpy): each cell type's median
distance to the border; rows are sorted by it, border-near on top.

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

Data: results/computational_effects/mibitof_fits/ (src/computational_effects/compute_fits.py,
example_scores.py, imc_centrality.py).
"""
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bosperrus.fit import ExponentialSaturationFit, MichaelisMentenFit, PiecewiseLinearFit
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde, norm

matplotlib.rcParams["svg.fonttype"] = "none"  # keep SVG text as editable <text>, not outlined paths
ROOT = Path(__file__).resolve().parent.parent.parent

MEASURES = ["degree", "clustering", "closeness", "betweenness"]
IMC_MEASURES = ["degree", "clustering", "closeness"]  # what squidpy's centrality_scores offers

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
    "Piecewise Linear Fit": lambda d, r: PiecewiseLinearFit.piecewise_plateau(
        d, r["piecewise_linear_b"], r["piecewise_linear_m"], r["piecewise_linear_c"]),
    "Exponential Saturation Fit": lambda d, r: ExponentialSaturationFit.exp_sat(
        d, r["exponential_saturation_a"], r["exponential_saturation_b"], r["exponential_saturation_c"]),
    "Michaelis-Menten Fit": lambda d, r: MichaelisMentenFit.michaelis_menten(
        d, r["michaelis_menten_a"], r["michaelis_menten_b"], r["michaelis_menten_c"]),
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
        fits_csv = ROOT / "results" / "computational_effects" / "mibitof_fits" / f"{gt}_graph_level_fits.csv"
        df = pd.read_csv(fits_csv, index_col=0)
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


def relative_means(imc, measure, kind):
    """Per-cell-type mean as % deviation from the FOV mean."""
    col = f"{measure}_{kind}"
    return (imc.groupby("cell_type")[col].mean() / imc[col].mean() - 1) * 100


def plot_imc(fig, gs, row, imc):
    order = imc.groupby("cell_type")["d"].median().sort_values()  # border-near on top
    y = np.arange(len(order))[::-1]
    colors = imc.groupby("cell_type")["color"].first().loc[order.index].tolist()
    counts = imc["cell_type"].value_counts().loc[order.index]

    axes = []
    for ci, measure in enumerate(MEASURES):
        ax = fig.add_subplot(gs[row, ci * 2], sharey=axes[0] if axes else None)
        axes.append(ax)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=6)
        if measure not in IMC_MEASURES:
            ax.barh(y, order.values / imc["d"].max(), color=colors, height=0.7)
            ax.set_xlabel("median distance to border / d_max", fontsize=6.5)
            ax.set_title("position", fontsize=8, pad=3)
            ax.tick_params(labelleft=False)
            ax.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=6, loc="center left",
                      bbox_to_anchor=(1.0, 0.5),
                      handletextpad=0.1)
            continue
        raw = relative_means(imc, measure, "raw").loc[order.index]
        corrected = relative_means(imc, measure, "corrected").loc[order.index]
        ax.axvline(0, color="#cccccc", lw=0.6, zorder=0)
        ax.hlines(y, raw, corrected, color="#bbbbbb", lw=1, zorder=1)
        ax.scatter(raw, y, facecolor="white", edgecolor=colors, s=14, lw=1, zorder=2, label="raw")
        ax.scatter(corrected, y, color=colors, s=14, zorder=3, label="corrected")
        ax.set_xlabel(f"{measure} vs. FOV mean (%)", fontsize=6.5)
        if ci == 0:
            ax.set_yticks(y, [f"{t} ({n})" for t, n in counts.items()], fontsize=6)
            ax.set_title("squidpy IMC example (kNN, k=6): per-cell-type mean, raw vs. corrected",
                         fontsize=8, pad=3, loc="left")
        else:
            ax.tick_params(labelleft=False)


def plot_metrics(data, examples, imc, fit_palette):
    n_rows, n_cols = len(FAMILIES), len(MEASURES)
    fig = plt.figure(figsize=(7.5, 8.8))
    # row 0: example maps; then a gap row; then (top margin, main) per family;
    # then a gap row and the IMC example
    gs = GridSpec(n_rows * 2 + 4, n_cols * 2, figure=fig,
                  height_ratios=[3.2, 0.35] + [0.6, 2.6] * n_rows + [1.7, 3.4], width_ratios=[3.2, 0.7] * n_cols,
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

    plot_imc(fig, gs, n_rows * 2 + 3, imc)
    return fig


def load_all():
    """Everything plot_metrics needs: fits, example maps, IMC example and the fit palette."""
    with open(ROOT / "fit_palette.json") as f:
        fit_palette = json.load(f)
    res = ROOT / "results" / "computational_effects" / "mibitof_fits"
    return dict(data=load_data(), examples=pd.read_csv(res / "figure3_examples.csv"),
                imc=pd.read_csv(res / "imc_centrality.csv"), fit_palette=fit_palette)
