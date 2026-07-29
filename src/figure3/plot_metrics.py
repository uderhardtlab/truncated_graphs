"""
Figure 3: border-effect fit summary across measures, for one representative
graph type per family (Delaunay, kNN k=10, rNN r=0.03 -- the graph types
within a family are similar enough that showing all 8 was redundant).

Each (family, measure) cell is a mini "jointplot":
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

y-axis is not shared/fixed across panels -- each cell auto-scales
symmetrically around 0 to its own data range, since different measures
have very different natural effect magnitudes.

Run:
    cd truncated_graphs/
    pixi run python src/figure3/plot_metrics.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde, norm

from bosperrus.fit import PiecewiseLinearFit, ExponentialSaturationFit, MichaelisMentenFit
from compute_fits import MEASURES as _MEASURES

ROOT = Path(__file__).resolve().parent.parent.parent

MEASURES = sorted(_MEASURES)

REPRESENTATIVE = {"Delaunay": "delaunay", "kNN (k=10)": "knn_k=10", "rNN (r=0.03)": "rnn_r=0.03"}
FAMILIES = list(REPRESENTATIVE.keys())

CURVE_FIT_ORDER = ["Piecewise Linear Fit", "Exponential Saturation Fit", "Michaelis-Menten Fit"]
ALL_FIT_ORDER = CURVE_FIT_ORDER + ["Constant Fit"]

# coordinates are min-max normalized to the unit square before
# distance_to_convex_hull, so d_max is consistent across datasets (measured
# 0.483 +/- 0.017 over an 80-dataset random sample) -- used as a shared
# stand-in for each dataset's own d_max, since it isn't stored in the
# per-graph-type CSVs.
D_MAX_APPROX = 0.483
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
        df = pd.read_csv(ROOT / "results" / "figure3" / f"{gt}_graph_level_fits.csv", index_col=0)
        df["graph_type"] = gt
        df["measure"] = df.index
        dfs.append(df)
    combined = pd.concat(dfs).reset_index(drop=True)
    return combined[combined["measure"].isin(MEASURES)].copy()


def kde_or_none(values, grid):
    """gaussian_kde requires >1 distinct value; returns None for degenerate
    (too few, or zero-variance) inputs rather than raising."""
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if len(values) < 5 or values.std() < 1e-8:
        return None
    return gaussian_kde(values)(grid)


def plot_cell(ax_main, ax_top, ax_right, sub, fit_palette):
    n_total = len(sub)
    non_baseline = sub[sub["best_fit_type"] != "Constant Fit"]

    segments_by_model = {m: [] for m in CURVE_FIT_ORDER}
    for _, row in non_baseline.iterrows():
        model = row["best_fit_type"]
        S = CURVE_FN[model](X_GRID, row)
        c_border, c_center = S[0], S[-1]
        shape = (S - c_center) / (c_border - c_center + _EPS)
        segments_by_model[model].append(np.column_stack([X_GRID, shape * row["observed_effect_strength"]]))

    y_abs_max = non_baseline["observed_effect_strength"].abs().max() if len(non_baseline) else 0.05
    y_abs_max = max(y_abs_max, 0.05) * 1.15
    es_grid = np.linspace(-y_abs_max, y_abs_max, 100)
    spike_sigma_x = SPIKE_SIGMA_FRAC * D_MAX_APPROX
    spike_sigma_y = SPIKE_SIGMA_FRAC * y_abs_max

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
    ax_main.set_xlim(0, D_MAX_APPROX)
    ax_main.set_ylim(-y_abs_max, y_abs_max)
    ax_main.spines[["top", "right"]].set_visible(False)
    ax_main.tick_params(labelsize=6)
    for ax in (ax_top, ax_right):
        ax.axis("off")


def plot_metrics(data, fit_palette):
    n_rows, n_cols = len(FAMILIES), len(MEASURES)
    fig = plt.figure(figsize=(9, 9))
    gs = GridSpec(n_rows * 2, n_cols * 2, figure=fig,
                  height_ratios=[0.7, 3.2] * n_rows, width_ratios=[3.2, 0.7] * n_cols,
                  hspace=0.08, wspace=0.08)

    for ri, family in enumerate(FAMILIES):
        gt = REPRESENTATIVE[family]
        for ci, measure in enumerate(MEASURES):
            ax_main = fig.add_subplot(gs[ri * 2 + 1, ci * 2])
            ax_top = fig.add_subplot(gs[ri * 2, ci * 2], sharex=ax_main)
            ax_right = fig.add_subplot(gs[ri * 2 + 1, ci * 2 + 1], sharey=ax_main)

            sub = data[(data["measure"] == measure) & (data["graph_type"] == gt)]
            plot_cell(ax_main, ax_top, ax_right, sub, fit_palette)

            if ri == 0:
                ax_top.set_title(measure, fontsize=9, pad=4)
            if ci == 0:
                ax_main.set_ylabel(family, fontsize=8, labelpad=4)
            if ri == n_rows - 1:
                ax_main.set_xlabel("distance to border", fontsize=6.5)
            else:
                plt.setp(ax_main.get_xticklabels(), visible=False)

    handles = [Line2D([0], [0], color=fit_palette[m], lw=2.5, label=m) for m in ALL_FIT_ORDER]
    fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.suptitle("Border-effect fits by measure and graph type\n"
                 "(top margin: half-life distribution; right margin: effect-strength distribution; "
                 "area under each curve/spike = share of datasets won)",
                 fontsize=9.5, y=1.1)
    return fig


def main():
    with open(ROOT / "fit_palette.json") as f:
        fit_palette = json.load(f)

    data = load_data()
    fig = plot_metrics(data, fit_palette)

    out = ROOT / "result_plots" / "figure3" / "figure3"
    fig.savefig(str(out) + ".svg", format="svg", bbox_inches="tight")
    fig.savefig(str(out) + ".pdf", bbox_inches="tight")
    fig.savefig(str(out) + ".png", dpi=160, bbox_inches="tight")
    print(f"Saved {out}.svg / .pdf / .png")
    plt.close("all")


if __name__ == "__main__":
    main()
