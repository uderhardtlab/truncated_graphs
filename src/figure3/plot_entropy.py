"""
Figure 3 companion: entropy_AIC_weights distributions, same grid as
plot_metrics.py (one representative graph type per family x measure).

Each cell is a KDE of entropy_AIC_weights (model-selection ambiguity among
the three non-baseline saturation models: 0 = confident, 1 = fully
ambiguous), split by winning model type and colored via fit_palette.json.
Each curve is scaled by that model's share of the cell's total dataset count
(including Constant Fit in the denominator, not just the non-baseline rows)
so relative area still reads as "how often did this fit type win" -- the
same area-equals-share convention as plot_metrics.py's margins. Constant Fit
itself has no entropy_AIC_weights (never set for the baseline) and isn't
represented here; this plot is specifically about ambiguity among the
non-trivial models.

Run:
    cd truncated_graphs/
    pixi run python src/figure3/plot_entropy.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"  # keep SVG text as editable <text>, not outlined paths
import matplotlib.pyplot as plt
import numpy as np

from plot_metrics import REPRESENTATIVE, FAMILIES, MEASURES, load_data, kde_or_none

ROOT = Path(__file__).resolve().parent.parent.parent

CURVE_FIT_ORDER = ["Piecewise Linear Fit", "Exponential Saturation Fit", "Michaelis-Menten Fit"]
ENTROPY_GRID = np.linspace(0, 1, 200)


def plot_cell(ax, sub, fit_palette):
    n_total = len(sub)
    for model in CURVE_FIT_ORDER:
        model_rows = sub[sub["best_fit_type"] == model]
        frac = len(model_rows) / n_total if n_total else 0
        if frac == 0:
            continue
        density = kde_or_none(model_rows["entropy_AIC_weights"], ENTROPY_GRID)
        if density is None:
            continue
        density = density * frac
        ax.fill_between(ENTROPY_GRID, density, color=fit_palette[model], alpha=0.35, lw=0)
        ax.plot(ENTROPY_GRID, density, color=fit_palette[model], lw=1)

    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=6)


def plot_entropy(data, fit_palette):
    n_rows, n_cols = len(FAMILIES), len(MEASURES)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(9, 4.5), sharex=True)

    for ri, family in enumerate(FAMILIES):
        gt = REPRESENTATIVE[family]
        for ci, measure in enumerate(MEASURES):
            ax = axes[ri, ci]
            sub = data[(data["measure"] == measure) & (data["graph_type"] == gt)]
            plot_cell(ax, sub, fit_palette)

            if ri == 0:
                ax.set_title(measure, fontsize=9, pad=4)
            if ci == 0:
                ax.set_ylabel(family, fontsize=8, labelpad=4)
            if ri == n_rows - 1:
                ax.set_xlabel("entropy_AIC_weights", fontsize=6.5)
            else:
                plt.setp(ax.get_xticklabels(), visible=False)

    fig.tight_layout()
    return fig


def main():
    import json
    with open(ROOT / "fit_palette.json") as f:
        fit_palette = json.load(f)

    data = load_data()
    fig = plot_entropy(data, fit_palette)

    out = ROOT / "result_plots" / "figure3" / "figure3_entropy"
    fig.savefig(str(out) + ".svg", format="svg", bbox_inches="tight")
    print(f"Saved {out}.svg")
    plt.close("all")


if __name__ == "__main__":
    main()
