"""
Figure 3 companion: entropy_AIC_weights vs. graph size (num_nodes).

num_nodes is by far the strongest covariate of entropy_AIC_weights found
during the figure3 investigation (pooled Spearman rho=-0.512 vs. -0.453 for
edge density and +-0.3 for everything else tried), and it replicates almost
identically within a single cohort and within every measure x graph-type
cell -- a straightforward statistical-power story: more nodes means more
independent data to constrain which saturation shape fits best, so AIC
discriminates more sharply. betweenness is the one clear exception (stays
around entropy~0.6 regardless of size) -- its values are highly
interdependent across nodes (shared shortest paths), so added nodes don't
add as much independent information as they do for the per-node measures.

One row per representative graph type (same families as plot_metrics.py /
plot_entropy.py), one line per measure: median entropy_AIC_weights in bins
of log10(num_nodes), pooled across all non-Constant-Fit datasets in that
cell.

Run:
    cd truncated_graphs/
    pixi run python src/figure3/plot_entropy_vs_size.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"  # keep SVG text as editable <text>, not outlined paths
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_metrics import REPRESENTATIVE, FAMILIES, MEASURES, load_data, set_three_ticks

ROOT = Path(__file__).resolve().parent.parent.parent

# dataviz-skill categorical palette, first 5 slots, fixed order -- no
# established per-measure color convention elsewhere in this project
MEASURE_COLOR = dict(zip(MEASURES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
N_BINS = 15


def plot_cell(ax, sub):
    log_nodes = np.log10(sub["num_nodes"])
    bins = np.linspace(log_nodes.min(), log_nodes.max(), N_BINS)
    for measure in MEASURES:
        m_sub = sub[sub["measure"] == measure].copy()
        if len(m_sub) < N_BINS:
            continue
        m_sub["bin"] = pd.cut(np.log10(m_sub["num_nodes"]), bins)
        grp = m_sub.groupby("bin", observed=True)["entropy_AIC_weights"].median()
        centers = [iv.mid for iv in grp.index]
        ax.plot(centers, grp.values, "-o", color=MEASURE_COLOR[measure], ms=3, lw=1.6)

    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylim(-0.03, 1.03)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7)
    set_three_ticks(ax)


def plot_entropy_vs_size(data):
    fig, axes = plt.subplots(1, len(FAMILIES), figsize=(9, 3), sharey=True)

    for ax, family in zip(axes, FAMILIES):
        gt = REPRESENTATIVE[family]
        sub = data[(data["graph_type"] == gt) & (data["best_fit_type"] != "Constant Fit")]
        plot_cell(ax, sub)
        ax.set_title(family, fontsize=9)
        ax.set_xlabel("log10(num_nodes)", fontsize=7.5)

    axes[0].set_ylabel("median entropy_AIC_weights", fontsize=8)
    fig.tight_layout()
    return fig


def main():
    data = load_data()
    fig = plot_entropy_vs_size(data)

    out = ROOT / "result_plots" / "figure3" / "figure3_entropy_vs_nodes"
    fig.savefig(str(out) + ".svg", format="svg", bbox_inches="tight")
    print(f"Saved {out}.svg")
    plt.close("all")


if __name__ == "__main__":
    main()
