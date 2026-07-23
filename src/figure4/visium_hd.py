"""Visium HD tissue-border effect analysis.

For each 8um-binned Visium HD sample: identify tissue-border spots as grid
positions with fewer than 4 von-Neumann grid-neighbors present in the tissue
(squidpy.gr.spatial_neighbors_grid(n_rings=1, n_neighs=4)), then fit BOSPERRUS
(ConstantFit / PiecewiseLinearFit) of log1p(total_counts) against distance to
the nearest such border point.
"""
import gc
import sys
from pathlib import Path

import numpy as np
import scanpy as sc
import squidpy
from scipy.sparse.csgraph import connected_components

import bosperrus
from bosperrus.fit import ConstantFit, PiecewiseLinearFit
from bosperrus.distances import distance_to_pointset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blade import peel_sweep


def analyze_dataset(h5ad_path):
    """Load one Visium HD sample, restrict to its largest spatially-connected
    tissue component, identify tissue-border spots (grid NN<4), and fit
    BOSPERRUS of log1p(total_counts) against distance to the nearest border
    spot. Returns a dict with the fitted Flow and the raw grid coords/counts
    needed for downstream plotting. The loaded AnnData is dropped before
    returning — these files are large (4-7GB), only small derived arrays are kept.
    """
    adata = sc.read_h5ad(h5ad_path)
    adata.obs["log1p_total_counts"] = np.log1p(np.asarray(adata.X.sum(axis=1)).ravel())

    # tissue-border points: grid spots with fewer than 4 grid-neighbors
    # (von Neumann/4-connectivity, radius = 1 grid step)
    squidpy.gr.spatial_neighbors_grid(adata, n_rings=1, n_neighs=4)
    n_components, labels = connected_components(adata.obsp["spatial_connectivities"], directed=False)
    largest_component = np.argmax(np.bincount(labels))
    adata = adata[labels == largest_component].copy()

    n_neighbors = np.diff(adata.obsp["spatial_connectivities"].indptr)
    border_mask = n_neighbors < 4

    coords_grid = adata.obs[["array_row", "array_col"]].to_numpy()
    border_coords_grid = coords_grid[border_mask]

    # distance to nearest tissue-border point (0 for the border points themselves)
    dist_to_border = distance_to_pointset(coords_grid, border_coords_grid).rename("dist_to_tissue_border")

    scores = adata.obs[["log1p_total_counts"]].reset_index(drop=True)
    flow = bosperrus.Flow.from_distances_and_scores(distances=dist_to_border, scores=scores)
    flow.flow(fits=[ConstantFit, PiecewiseLinearFit])

    result = {
        "flow": flow,
        "array_row": coords_grid[:, 0],
        "array_col": coords_grid[:, 1],
        "border_mask": border_mask,
        "log1p_total_counts": adata.obs["log1p_total_counts"].to_numpy(),
    }

    del adata
    gc.collect()
    return result


def summarize_fit(name, result, bin_size_um, measure="log1p_total_counts"):
    """One summary row (best-fit type, effect strength, half-life fraction,
    elbow location) for a single analyze_dataset() result.

    Note on half_life_frac_of_dmax: it's deliberately scaled by d_max — it
    reports what *fraction* of the maximum observed distance is affected, not
    an absolute distance. That's what makes it comparable across datasets of
    different extent; elbow_um is the absolute-distance version.
    """
    flow = result["flow"]
    fit = flow.best_fits[measure]
    row = {
        "dataset": name,
        "best_fit": fit.name,
        "effect_strength": flow.fit_quality.loc["observed_effect_strength", measure],
        "half_life_frac_of_dmax": flow.fit_quality.loc["observed_half_life", measure],
        "n_bins": len(result["log1p_total_counts"]),
    }
    if isinstance(fit, PiecewiseLinearFit):
        row["elbow_grid_steps"] = fit.params["piecewise_linear_b"]
        row["elbow_um"] = fit.params["piecewise_linear_b"] * bin_size_um
        row["slope_m"] = fit.params["piecewise_linear_m"]
    return row


def plot_fit_curve(ax, fit, d, color="tab:red"):
    """Overlay the fitted curve (piecewise-linear, or a flat line if
    ConstantFit won) on a scatter axis."""
    d_line = np.linspace(0, d.max(), 200)
    if isinstance(fit, PiecewiseLinearFit):
        b = fit.params["piecewise_linear_b"]
        m = fit.params["piecewise_linear_m"]
        c = fit.params["piecewise_linear_c"]
        y_line = PiecewiseLinearFit.piecewise_plateau(d_line, b=b, m=m, c=c)
        ax.plot(d_line, y_line, color=color, lw=2, label=f"piecewise-linear (elbow={b:.2f})")
        ax.axvline(b, color=color, lw=1, ls="--", alpha=0.7)
    else:
        ax.axhline(fit.params["constant_c"], color=color, lw=2, label="constant (no effect)")
    ax.legend(fontsize=8, loc="best")


def plot_counts_vs_distance(ax, result, measure="log1p_total_counts", color="gray",
                             fit_color="tab:red", title=None,
                             scatter_size=0.3, scatter_alpha=0.1, scatter_linewidths=0):
    """Scatter of counts vs. distance to the nearest tissue-border point,
    with the fitted BOSPERRUS curve overlaid."""
    flow = result["flow"]
    d = flow.observations["dist_to_tissue_border"]
    s = flow.observations[measure]
    ax.scatter(d, s, s=scatter_size, alpha=scatter_alpha, color=color,
               linewidths=scatter_linewidths, rasterized=True)
    plot_fit_curve(ax, flow.best_fits[measure], d, color=fit_color)
    ax.set_xlabel("distance to nearest tissue-border point (grid steps)")
    ax.set_ylabel(measure)
    if title:
        ax.set_title(title)


def plot_decision_boundary(ax, result, measure="log1p_total_counts", cmap="viridis",
                            boundary_color="tab:red", boundary_alpha=0.5, title=None,
                            add_legend=False):
    """Grid-space scatter colored by counts, with the tissue-border elbow
    drawn as an exclusion boundary (bins within the elbow distance are
    highlighted directly)."""
    flow = result["flow"]
    d_border = flow.observations["dist_to_tissue_border"].to_numpy()
    counts = result["log1p_total_counts"]

    ax.scatter(result["array_col"], result["array_row"], c=counts, cmap=cmap,
               s=0.5, alpha=0.3, rasterized=True)

    fit = flow.best_fits[measure]
    if isinstance(fit, PiecewiseLinearFit):
        b = fit.params["piecewise_linear_b"]
        excluded = d_border <= b
        ax.scatter(result["array_col"][excluded], result["array_row"][excluded],
                   color=boundary_color, s=0.5, alpha=boundary_alpha, rasterized=True)

    if title:
        ax.set_title(title)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")

    if add_legend:
        ax.scatter([], [], color=boundary_color, label="excluded by tissue-border elbow")
        ax.legend(fontsize=8, loc="lower left")


def blade_comparison(result, measure="log1p_total_counts", min_group_size=30):
    """Run blade.peel_sweep comparing raw vs. BOSPERRUS-corrected `measure`
    for one analyze_dataset() result, against the same peel-layer masks.
    Returns (sweep_df, buffers) as documented in blade.peel_sweep.
    """
    flow = result["flow"]
    raw = flow.observations[measure].to_numpy()
    corrected = flow.observations[f"BOSPERRUS corrected {measure}"].to_numpy()
    return peel_sweep(
        result["array_row"], result["array_col"],
        {"raw": raw, "bosperrus_corrected": corrected},
        min_group_size=min_group_size,
    )
