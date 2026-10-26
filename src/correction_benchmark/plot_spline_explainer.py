"""
Explainer figure for the P-spline detrending baseline (sphere.py's pspline_gcv) on one benchmark crop:
seed 0, uniform points, Delaunay, cap radius 2 -- the first crop of sphere.py's seed-0 run.

a    the cubic B-spline basis (42 bumps on 40 equally spaced knots)
b    one bump: 4 cubic pieces joined smoothly at the knots
c/e  score vs. distance to the border, with spline fits for the smallest, the GCV-chosen and the largest
     lambda of the grid, and BOSPERRUS's AIC-best fit f(d)
d/f  GCV and effective degrees of freedom over lambda

Run on a compute node:  cd truncated_graphs/src && pixi run python correction_benchmark/plot_spline_explainer.py
Output: ../result_plots/correction_benchmark/spline_explainer.{pdf,png}
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
from scipy.stats import spearmanr

from bosperrus import Flow
from bosperrus.centrality_measures import compute_centrality_measures

from sphere import (MEASURES, crop_cap, get_edge_list, pspline_gcv, sample_uniform_on_unit_sphere,
                    sample_von_mises_fisher)

SEED = 0
PLOT_MEASURES = ["betweenness", "closeness"]
HIGHLIGHT_BUMP = 20
FIG_WIDTH = 9  # manuscript full text width, as in plot_violins.py
OUT = "../result_plots/correction_benchmark/spline_explainer"
COLORS = {"small": "#6BAF92", "gcv": "#A8842A", "large": "#D9C27E", "bosperrus": "#3A7DB0"}
PIECE_COLORS = ["#1B3F66", "#3A7DB0", "#A8842A", "#AB0C67"]


def first_crop(seed):
    """Coords, crop indices, distances and center of the uniform-Delaunay crop of sphere.py's run `seed`
    (same RNG calls in the same order as sphere.main())."""
    rng = np.random.default_rng(seed)
    coords = sample_uniform_on_unit_sphere(n=5000, rng=rng)
    sample_von_mises_fisher(n=5000, kappas=[1], rng=rng)  # drawn by main() before the first crop
    sample_von_mises_fisher(n=5000, kappas=[1, 3, 5], rng=rng)
    crop, distances, center = crop_cap(coords, 2, rng=rng, return_center=True)
    return coords, crop, distances.reset_index(drop=True), center


def plot_basis(ax, t, grid):
    n_basis = len(t) - 4
    for j in range(n_basis):
        c = np.zeros(n_basis)
        c[j] = 1
        ax.plot(grid, BSpline(t, c, 3)(grid), lw=0.6, color="#1B3F66" if j == HIGHLIGHT_BUMP else "0.7")
    ax.plot(t[3:-3], np.zeros(len(t) - 6), "|", color="k", ms=5)
    ax.set_title(f"a  cubic B-spline basis: {n_basis} bumps on {len(t) - 6} knots (|)", loc="left", fontsize=9)
    ax.set_xlabel("distance to border $d$")
    ax.set_ylabel("$B_j(d)$")


def plot_bump(ax, t):
    n_basis = len(t) - 4
    c = np.zeros(n_basis)
    c[HIGHLIGHT_BUMP] = 1
    bump = BSpline(t, c, 3)
    knots = t[HIGHLIGHT_BUMP:HIGHLIGHT_BUMP + 5]
    for i, color in enumerate(PIECE_COLORS):
        x = np.linspace(knots[i], knots[i + 1], 50)
        ax.plot(x, bump(x), lw=2, color=color, label=f"cubic piece {i + 1}")
    for k in knots:
        ax.axvline(k, ls=":", lw=0.6, color="0.5")
    pad = knots[1] - knots[0]
    x = np.r_[np.linspace(knots[0] - pad, knots[0], 10), np.linspace(knots[-1], knots[-1] + pad, 10)]
    ax.plot(x[:10], bump(x[:10]), color="0.6", lw=1)
    ax.plot(x[10:], bump(x[10:]), color="0.6", lw=1, label="zero outside")
    ax.set_title(f"b  one bump ($B_{{{HIGHLIGHT_BUMP + 1}}}$): 4 cubics, smooth at the knots (:)", loc="left", fontsize=9)
    ax.set_xlabel("distance to border $d$")
    ax.legend(fontsize=7, frameon=False)


def plot_fit(ax, d, y, spline, path, bosperrus_fit, measure, letter):
    grid = np.linspace(d.min(), d.max(), 400)
    ax.scatter(d, y, s=2, color="0.6", alpha=0.3, lw=0, rasterized=True)
    best = path["gcv"].idxmin()
    for key, row_idx, label in [("small", 0, "smallest"), ("gcv", best, "GCV choice"), ("large", len(path) - 1, "largest")]:
        row = path.loc[row_idx]
        ax.plot(grid, BSpline(spline.t, row["coef"], 3)(grid), color=COLORS[key], lw=1.6 if key == "gcv" else 1,
                label=f"spline, {label} $\\lambda$={row['lam']:.0e} (edf {row['edf']:.1f})")
    ax.plot(grid, bosperrus_fit.predict(grid), color=COLORS["bosperrus"], lw=1.4, ls="--",
            label=f"BOSPERRUS {type(bosperrus_fit).__name__}")
    hi = np.nanpercentile(y, 99.5)
    ax.set_ylim(np.nanmin(y) - 0.05 * (hi - np.nanmin(y)), hi)
    ax.set_title(f"{letter}  {measure} (y clipped at 99.5th pct)", loc="left", fontsize=9)
    ax.set_xlabel("distance to border $d$")
    ax.set_ylabel(measure)
    ax.legend(fontsize=6.5, frameon=False, loc="best")


def plot_gcv(ax, path, measure, letter):
    best = path.loc[path["gcv"].idxmin()]
    ax.semilogx(path["lam"], path["gcv"] / best["gcv"], color=COLORS["gcv"])
    ax.axvline(best["lam"], ls=":", color=COLORS["gcv"])
    ax.set_ylabel("GCV / min GCV", color=COLORS["gcv"])
    ax.set_xlabel("$\\lambda$")
    ax2 = ax.twinx()
    ax2.semilogx(path["lam"], path["edf"], color="0.3", lw=1)
    ax2.set_ylabel("effective df", color="0.3")
    ax.set_title(f"{letter}  {measure}: GCV picks $\\lambda$={best['lam']:.0e}, edf {best['edf']:.1f}",
                 loc="left", fontsize=9)


def main():
    coords, crop, d, center = first_crop(SEED)
    edges = get_edge_list(coords[crop], "delaunay", pole=-center)
    scores = pd.DataFrame(compute_centrality_measures(edges, N=len(crop), measures=MEASURES))

    # check: same crop as the benchmark (compare with sphere/correlations_seed0.csv, uniform Delaunay, Spearman, all)
    global_scores = pd.DataFrame(compute_centrality_measures(get_edge_list(coords, "delaunay"), N=len(coords),
                                                             measures=MEASURES))
    ref = pd.read_csv(f"../results/correction_benchmark/sphere/correlations_seed{SEED}.csv", index_col=0)
    ref = ref[(ref["graph_type"] == "delaunay") & (ref["coord_type"] == "uniform")
              & (ref["corr_method"] == "spearman") & (ref["node_subset"] == "all")]
    for m in MEASURES:
        rho = spearmanr(global_scores[m].to_numpy()[crop], scores[m]).statistic
        print(f"raw Spearman {m}: {rho:.4f} here, {ref.loc[m, 'original vs. on crop']:.4f} in the benchmark CSV")

    bf = Flow.from_distances_and_scores(distances=d, scores=scores)
    bf.flow(score_names=list(scores.columns))

    fits = {}
    for m in MEASURES:
        y = scores[m].to_numpy(float)
        ok = np.isfinite(y)
        fits[m] = pspline_gcv(d.to_numpy()[ok], y[ok], return_path=True)
        best = fits[m][1].loc[fits[m][1]["gcv"].idxmin()]
        print(f"{m}: GCV lambda {best['lam']:.2e}, edf {best['edf']:.1f}; BOSPERRUS {type(bf.best_fits[m]).__name__}")

    fig, axes = plt.subplots(3, 2, figsize=(FIG_WIDTH, FIG_WIDTH * 1.05))
    t = fits[PLOT_MEASURES[0]][0].t
    plot_basis(axes[0, 0], t, np.linspace(d.min(), d.max(), 800))
    plot_bump(axes[0, 1], t)
    for row, (m, letters) in enumerate(zip(PLOT_MEASURES, [("c", "d"), ("e", "f")]), start=1):
        spline, path = fits[m]
        plot_fit(axes[row, 0], d.to_numpy(), scores[m].to_numpy(float), spline, path, bf.best_fits[m], m, letters[0])
        plot_gcv(axes[row, 1], path, m, letters[1])
    fig.suptitle(f"P-spline detrending on one benchmark crop (seed {SEED}, uniform, Delaunay, cap radius 2, "
                 f"n = {len(crop)})", fontsize=10)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(f"{OUT}.{ext}", dpi=200, bbox_inches="tight")
    print(f"Saved {OUT}.pdf/.png")


if __name__ == "__main__":
    main()
