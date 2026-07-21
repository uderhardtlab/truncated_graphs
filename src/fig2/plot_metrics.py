"""
Figure 2 e-h: overview of border-effect metrics across measures and graph types.

4 rows  (rel. likelihood | H_AIC | half-life | effect strength)
5 cols  (betweenness | closeness | clustering | degree | pagerank)

Within each cell:
  - strip points colored by best-fit model type
  - box plots: black outline, no fill, drawn on top
  - shared y-axis per row

Run:
    cd truncated_graphs/
    uv run python src/fig2/plot_metrics.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent.parent

with open(ROOT / "notebooks" / "fit_palette.json") as f:
    FIT_PAL = json.load(f)

FIT_ORDER = ["Piecewise Linear Fit", "Exponential Saturation Fit", "Michaelis-Menten Fit"]
GT_SHORT  = {"delaunay": "Del.", "knn_k=10": "kNN", "rnn_r=0.03": "rNN"}
GT_ORDER  = ["delaunay", "knn_k=10", "rnn_r=0.03"]
MEASURES  = sorted(["betweenness", "closeness", "clustering", "degree", "pagerank"])

METRICS = [
    # (column,                                    ylabel,                hline0, hlines_dashed)
    ("scaled_relative_likelihood_over_baseline", "rel. likelihood",     None,   [1.1]),
    ("entropy_AIC_weights",                       r"$H_\mathrm{AIC}$", None,   []),
    ("observed_half_life",                         "half-life",         None,   []),
    ("observed_effect_strength",                   "effect strength",    0.0,   [-0.1, 0.1]),
]

TICK_FS  = 8.0
LABEL_FS = 9.5
TITLE_FS = 10.0

# ── load data ─────────────────────────────────────────────────────────────────
dfs = []
for gt in GT_ORDER:
    df = pd.read_csv(ROOT / "results" / "figure2" / f"{gt}_graph_level_fits.csv", index_col=0)
    df["graph_type"] = gt
    df["measure"]    = df.index
    dfs.append(df)
combined = pd.concat(dfs).reset_index(drop=True)
combined = combined[combined["measure"].isin(MEASURES)].copy()
data     = combined[combined["best_fit_type"] != "Constant Fit"].copy()

# ── figure ────────────────────────────────────────────────────────────────────
n_rows, n_cols = len(METRICS), len(MEASURES)
CELL_H  = 1.10
LEG_H   = 0.55
total_w = 7.5
total_h = n_rows * CELL_H + LEG_H

fig = plt.figure(figsize=(total_w, total_h), facecolor="white")
gs  = GridSpec(
    n_rows, n_cols,
    figure=fig,
    left=0.12, right=0.98,
    top=1.0 - LEG_H / total_h - 0.01,
    bottom=0.08,
    hspace=0.40, wspace=0.18,
)

rng = np.random.default_rng(42)

# 1–99th percentile clip per row (across all measures, for a consistent shared axis)
row_clips = {}
for ri, (ycol, *_) in enumerate(METRICS):
    sub = data[ycol].dropna()
    row_clips[ri] = (sub.quantile(0.01), sub.quantile(0.99))

row_axes = {}

for ri, (ycol, ylabel, hline0, hlines) in enumerate(METRICS):
    lo, hi = row_clips[ri]

    for ci, m in enumerate(MEASURES):
        sharey_ax = row_axes.get(ri)
        ax = fig.add_subplot(gs[ri, ci], sharey=sharey_ax)
        if sharey_ax is None:
            row_axes[ri] = ax

        sub = data[(data["measure"] == m) & data[ycol].between(lo, hi)]

        # strip: colored by best-fit model, drawn first (background)
        for i, gt in enumerate(GT_ORDER):
            gt_sub = sub[sub["graph_type"] == gt]
            for ft in FIT_ORDER:
                vals = gt_sub[gt_sub["best_fit_type"] == ft][ycol].dropna().values
                if not len(vals):
                    continue
                jx = i + rng.uniform(-0.18, 0.18, len(vals))
                ax.scatter(jx, vals, s=1.5, c=FIT_PAL[ft], alpha=0.35,
                           linewidths=0, zorder=2, rasterized=True)

        # box: black outline, no fill, on top
        for i, gt in enumerate(GT_ORDER):
            vals = sub[sub["graph_type"] == gt][ycol].dropna().values
            if not len(vals):
                continue
            ax.boxplot(
                vals, positions=[i], widths=0.32,
                patch_artist=True, zorder=4, showfliers=False,
                boxprops    =dict(facecolor="none", edgecolor="black", linewidth=0.8),
                medianprops =dict(color="black", linewidth=1.8),
                whiskerprops=dict(color="black", linewidth=0.8),
                capprops    =dict(color="black", linewidth=0.8),
            )

        if hline0 is not None:
            ax.axhline(hline0, lw=0.9, color="#444444", zorder=1)
        for h in hlines:
            ax.axhline(h, lw=0.8, ls="--", color="#aaaaaa", zorder=1)

        ax.set_xlim(-0.5, len(GT_ORDER) - 0.5)
        ax.set_xticks(range(len(GT_ORDER)))
        ax.set_xticklabels(
            [GT_SHORT[gt] for gt in GT_ORDER] if ri == n_rows - 1 else [""] * 3,
            fontsize=TICK_FS,
        )
        ax.tick_params(labelsize=TICK_FS, length=3, pad=2, width=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_linewidth(0.6)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(4, prune="both"))

        if ci > 0:
            plt.setp(ax.get_yticklabels(), visible=False)
        if ci == 0:
            ax.set_ylabel(ylabel, fontsize=LABEL_FS, labelpad=3)
        if ri == 0:
            ax.set_title(m, fontsize=TITLE_FS, pad=4)

# ── legend (top center, circles, fit model only) ──────────────────────────────
fit_handles = [
    Line2D([0], [0], marker="o", color="none",
           markerfacecolor=FIT_PAL[ft], markeredgewidth=0,
           markersize=7, label=ft)
    for ft in FIT_ORDER
]
ax_leg = fig.add_axes([0, 1.0 - LEG_H / total_h, 1, LEG_H / total_h])
ax_leg.axis("off")
ax_leg.legend(fit_handles, [h.get_label() for h in fit_handles],
              loc="center", ncol=3,
              fontsize=8.5, frameon=False,
              title="Best-fit model", title_fontsize=9,
              handlelength=0.5, handletextpad=0.5, columnspacing=1.2)

# ── save ──────────────────────────────────────────────────────────────────────
out = ROOT / "result_plots" / "figure2_ef"
fig.savefig(str(out) + ".svg", format="svg", bbox_inches="tight")
fig.savefig(str(out) + ".pdf",              bbox_inches="tight")
fig.savefig(str(out) + ".png", dpi=150,     bbox_inches="tight")
print(f"Saved {out}.svg / .pdf / .png  ({total_w:.1f} × {total_h:.1f} in)")
plt.close("all")
