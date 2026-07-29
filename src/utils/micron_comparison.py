"""MICrONS connectome border-effect analysis.

Compares BOSPERRUS's alpha-shape-boundary-distance elbow against the Reimann
lab's "outer synapse fraction" heuristic for identifying which neurons/spatial
bins in the MICrONS mm^3 connectome are affected by the edge effect (missing
synapses near the reconstructed volume's boundary).

Run standalone:
    cd truncated_graphs/
    uv run python src/figure4/micron_comparison.py
"""
import os

import alphashape
import conntility
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from shapely.geometry import Point

import bosperrus

# ── paths & constants ─────────────────────────────────────────────────────────
# FN_MAT still points at a je30bery-only path that doesn't resolve on this
# checkout — update once the actual connectome file's location here is known.
FN_MAT        = "/data/bionets/je30bery/bosperrus-experiments/reimann_data/microns_mm3_connectome.h5"
SAVE_PATH_SVG = os.path.join(os.path.dirname(__file__), "..", "..", "result_plots", "figure4", "fig_microns_buffer.svg")
SAVE_PATH_PDF = os.path.join(os.path.dirname(__file__), "..", "..", "..", "bosperrus", "figures", "figure4", "fig4_prelim_microns_buffer.pdf")

REIMANN_THRESH = 0.05
NBINS          = 51
ALPHA_SHAPE    = 0.0001

# ── colour palette ────────────────────────────────────────────────────────────
C_IND   = "#0033FF"
C_OUT   = "#00B2FF"
C_DEG   = "#000E47"
C_SAFE  = "#46E895"
C_SAFE2 = "#084023"

SCORE_COLORS = {"indegree": C_IND, "outdegree": C_OUT, "degree": C_DEG}
SCORE_LABELS = {"indegree": "Indegree", "outdegree": "Outdegree", "degree": "Degree"}


# ── data preparation ──────────────────────────────────────────────────────────
def load_connectome(fn_mat, dataset="full"):
    """Load the MICrONS connectome from an h5 file via conntility."""
    return conntility.ConnectivityMatrix.from_h5(fn_mat, dataset)


def bin_reimann(M, nbins=NBINS):
    """Bin edges/vertices into an nbins x nbins x-z grid and flag "outer" bins
    (Reimann's border-region heuristic: bin synapse count < 1000, or within 3
    bins of the top/bottom of the z-stack). Mutates M in place (adds edge/vertex
    properties). Returns the compressed per-neuron view C with an added
    "outer_syn_fraction" vertex property, plus the raw per-bin synapse-count table I.
    """
    x_col, z_col = f"x_nm_binned_{nbins}", f"z_nm_binned_{nbins}"
    for col, binned_col in [("x_nm", x_col), ("z_nm", z_col)]:
        bins = np.linspace(M.edges[col].min(), M.edges[col].max() + 1, nbins)
        M.add_edge_property(binned_col, np.digitize(M.edges[col], bins=bins))
        M.add_vertex_property(binned_col, np.digitize(M.vertices[col], bins=bins))

    I = M.edges.groupby([x_col, z_col])["id"].count().unstack(x_col)

    edge_idxx = pd.MultiIndex.from_frame(M.edges[[z_col, x_col]])
    is_outer = I.stack().rename("count").reset_index()
    is_outer["outer"] = (
        (is_outer["count"] < 1000) |
        (is_outer[z_col] <= 3) |
        (is_outer[z_col] >= (nbins - 3))
    )
    is_outer = is_outer.set_index([z_col, x_col])["outer"]
    M.add_edge_property("syn_in_outer_bin", is_outer[edge_idxx].values)

    C = M.compress({"outer_bin_count": ("syn_in_outer_bin", "sum")})
    outer_per_neuron = np.array(C.default("outer_bin_count").matrix.sum(axis=0))[0]
    C.add_vertex_property("outer_syn_fraction", outer_per_neuron / C.vertices["indegree"].values)
    return C, I


def compute_alpha_shape_distances(M, alpha=ALPHA_SHAPE):
    """Fit an alpha shape to the neurons' x-z footprint and compute each
    neuron's distance to its boundary."""
    coords_xz = M.vertices[["x_nm", "z_nm"]].values
    alpha_shape = alphashape.alphashape(coords_xz, alpha=alpha)
    alpha_distances = np.array([
        alpha_shape.boundary.distance(Point(x, z))
        for x, z in coords_xz
    ])
    return alpha_shape, alpha_distances


def fit_bosperrus(alpha_distances, scores):
    """Fit piecewise-linear (vs. constant-baseline) BOSPERRUS curves of each
    centrality score in `scores` against distance to the alpha-shape boundary."""
    flow = bosperrus.Flow.from_distances_and_scores(
        distances=pd.Series(alpha_distances, name="alpha_distance"),
        scores=scores,
    )
    flow.flow(fits=[bosperrus.PiecewiseLinearFit, bosperrus.ConstantFit],
              baseline_fit_class=bosperrus.ConstantFit)
    return flow


def compute_reimann_comparison(M, C, flow, alpha_distances, reimann_thresh=REIMANN_THRESH, nbins=NBINS):
    """Compare BOSPERRUS's elbow-based affected-neuron classification against
    Reimann's per-neuron and per-bin outer-synapse-fraction heuristics, via
    Jaccard index. Returns a dict with the comparison dataframe, Jaccard
    indices, counts, and the max/mean per-bin outer-synapse-fraction grids
    (for the boundary heatmap panel).
    """
    x_col, z_col = f"x_nm_binned_{nbins}", f"z_nm_binned_{nbins}"
    elbows = {s: flow.best_fits[s].params["piecewise_linear_b"] for s in ["indegree", "outdegree", "degree"]}
    elbow = elbows["indegree"]  # indegree elbow used for binary classification

    df = M.vertices[["x_nm", "z_nm"]].copy()
    df["outer_syn_fraction"] = C.vertices["outer_syn_fraction"].values
    df["bosperrus_distance"] = alpha_distances
    df = df.dropna(subset=["outer_syn_fraction"])
    df["reimann_affected"]   = df["outer_syn_fraction"] > reimann_thresh
    df["bosperrus_affected"] = df["bosperrus_distance"] < elbow

    def jaccard(a, b):
        return (a & b).sum() / (a | b).sum()

    j_neuron = jaccard(df["bosperrus_affected"], df["reimann_affected"])

    z_bins = M.vertices.loc[df.index, z_col]
    x_bins = M.vertices.loc[df.index, x_col]

    grp = C.vertices.groupby([x_col, z_col])["outer_syn_fraction"]
    I_frac_max = grp.max().unstack(x_col).reindex(index=range(1, nbins), columns=range(1, nbins))
    I_frac_mean = grp.mean().unstack(x_col).reindex(index=range(1, nbins), columns=range(1, nbins))

    bin_max_frac = pd.Series(
        [I_frac_max.at[zb, xb] if (zb in I_frac_max.index and xb in I_frac_max.columns) else np.nan
         for zb, xb in zip(z_bins, x_bins)],
        index=df.index,
    )
    bin_reimann_affected = bin_max_frac > reimann_thresh
    j_bin = jaccard(df["bosperrus_affected"], bin_reimann_affected)

    return {
        "df": df, "elbows": elbows, "elbow": elbow,
        "jaccard_neuron": j_neuron, "jaccard_bin": j_bin,
        "n_bosperrus": int(df["bosperrus_affected"].sum()),
        "n_reimann_neuron": int(df["reimann_affected"].sum()),
        "n_reimann_bin": int(bin_reimann_affected.sum()),
        "n_total": len(df),
        "I_frac_max": I_frac_max, "I_frac_mean": I_frac_mean,
    }


def print_comparison_summary(comparison):
    """Print the neurons-affected / Jaccard-index table (as produced by
    compute_reimann_comparison) to stdout."""
    c = comparison
    print(f"{'':30s}  {'bosperrus':>10}  {'Reimann (neuron)':>16}  {'Reimann (bin)':>13}")
    print(f"{'Neurons affected':30s}  {c['n_bosperrus']:>10,}  {c['n_reimann_neuron']:>16,}  {c['n_reimann_bin']:>13,}")
    print(f"{'  (% of total)':30s}  {c['n_bosperrus']/c['n_total']*100:>9.1f}%"
          f"  {c['n_reimann_neuron']/c['n_total']*100:>15.1f}%  {c['n_reimann_bin']/c['n_total']*100:>12.1f}%")
    print()
    print("Jaccard vs bosperrus")
    print(f"  Reimann per-neuron : {c['jaccard_neuron']:.3f}")
    print(f"  Reimann per-bin    : {c['jaccard_bin']:.3f}")


def prepare_microns_data(fn_mat, nbins=NBINS, alpha=ALPHA_SHAPE, reimann_thresh=REIMANN_THRESH):
    """End-to-end MICrONS data preparation: load the connectome, bin it for
    the Reimann heuristic, fit BOSPERRUS against alpha-shape-boundary
    distance, and compute the bosperrus-vs-Reimann comparison. Returns a dict
    bundling everything the plotting functions need.
    """
    M = load_connectome(fn_mat)
    C, _ = bin_reimann(M, nbins=nbins)

    alpha_shape, alpha_distances = compute_alpha_shape_distances(M, alpha=alpha)

    scores = M.vertices[["indegree", "outdegree"]].copy()
    scores["degree"] = scores[["indegree", "outdegree"]].sum(axis=1)

    flow = fit_bosperrus(alpha_distances, scores)
    comparison = compute_reimann_comparison(M, C, flow, alpha_distances,
                                             reimann_thresh=reimann_thresh, nbins=nbins)

    x_edges_um = np.linspace(M.edges["x_nm"].min(), M.edges["x_nm"].max() + 1, nbins) / 1e3
    z_edges_um = np.linspace(M.edges["z_nm"].min(), M.edges["z_nm"].max() + 1, nbins) / 1e3

    return {
        "M": M, "C": C, "flow": flow, "scores": scores,
        "alpha_shape": alpha_shape, "alpha_distances": alpha_distances,
        "comparison": comparison,
        "x_edges_um": x_edges_um, "z_edges_um": z_edges_um,
        "x_centers_um": (x_edges_um[:-1] + x_edges_um[1:]) / 2,
        "z_centers_um": (z_edges_um[:-1] + z_edges_um[1:]) / 2,
    }


# ── plotting ──────────────────────────────────────────────────────────────────
def plot_fits_panel(ax, flow, alpha_distances, scores, measures=("indegree", "outdegree", "degree"),
                     scatter_color=None, fit_color=None, labels=None,
                     legend_label_fn=None, ylabel="Centrality score", title=None,
                     scatter_size=0.4, scatter_alpha=0.06, scatter_linewidths=0):
    """Scatter of centrality score(s) vs. distance to the alpha-shape
    boundary, overlaid with the fitted piecewise-linear BOSPERRUS curve(s).
    Pass a single measure (e.g. ["degree"]) with a flat scatter_color/fit_color
    for a single-curve panel; omit them to fall back to the per-measure
    SCORE_COLORS palette (indegree/outdegree/degree plotted together).
    legend_label_fn(measure, b, m, c), if given, overrides the default
    SCORE_LABELS-based legend text (e.g. to show the fitted elbow value).
    """
    labels = labels or SCORE_LABELS
    d_sorted = np.sort(alpha_distances)

    for measure in measures:
        fit = flow.best_fits[measure]
        b = fit.params["piecewise_linear_b"]
        m = fit.params["piecewise_linear_m"]
        c = fit.params["piecewise_linear_c"]
        s_color = scatter_color if scatter_color is not None else SCORE_COLORS[measure]
        f_color = fit_color if fit_color is not None else SCORE_COLORS[measure]

        ax.scatter(alpha_distances / 1e3, scores[measure],
                   s=scatter_size, alpha=scatter_alpha, color=s_color,
                   linewidths=scatter_linewidths, rasterized=True)
        y_fit = bosperrus.PiecewiseLinearFit.piecewise_plateau(d_sorted, b=b, m=m, c=c)
        label = legend_label_fn(measure, b, m, c) if legend_label_fn else labels[measure]
        ax.plot(d_sorted / 1e3, y_fit, color=f_color, lw=2, label=label)
        ax.axvline(b / 1e3, color=f_color, lw=1.2, ls="--", alpha=0.85)

    ax.set_yscale("log")
    ax.set_xlabel("Distance to boundary (µm)")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8, markerscale=6, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)


def _plot_poly_boundary(ax, geom, **kwargs):
    polys = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
    for p in polys:
        if hasattr(p, "exterior"):
            coords = np.array(p.exterior.coords)
            ax.plot(coords[:, 0] / 1e3, coords[:, 1] / 1e3, **kwargs)


def plot_boundary_heatmap_panel(ax, data, reimann_thresh=REIMANN_THRESH, cmap="magma",
                                 boundary_label="tissue boundary", boundary_color="gray",
                                 reimann_neuron_label="Reimann 5% (per neuron)", reimann_neuron_color=C_SAFE,
                                 reimann_bin_label="Reimann 5% (per bin max)", reimann_bin_color=C_SAFE2,
                                 elbow_color=C_IND, legend_bbox_to_anchor=(0.5, -0.4)):
    """Max outer-synapse-fraction heatmap with Reimann contours, the
    alpha-shape tissue boundary, and the bosperrus elbow boundary overlaid.
    `data` is the dict returned by prepare_microns_data.
    """
    comparison = data["comparison"]
    I_frac_max, I_frac_mean = comparison["I_frac_max"], comparison["I_frac_mean"]

    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("0.85")
    pcm = ax.pcolormesh(data["x_edges_um"], data["z_edges_um"], I_frac_max.values,
                         cmap=cm, shading="flat", rasterized=True, vmin=0, vmax=1)
    plt.colorbar(pcm, ax=ax, label="Max outer synapse fraction", shrink=0.75, pad=0.02)

    ax.contour(data["x_centers_um"], data["z_centers_um"], I_frac_mean.values,
               levels=[reimann_thresh], colors=[reimann_neuron_color], linewidths=1.5, linestyles="-")
    ax.contour(data["x_centers_um"], data["z_centers_um"], I_frac_max.values,
               levels=[reimann_thresh], colors=[reimann_bin_color], linewidths=1.5, linestyles="-")

    _plot_poly_boundary(ax, data["alpha_shape"], color=boundary_color, lw=1, ls="-", alpha=0.7)
    elbow = comparison["elbow"]
    elbow_poly = data["alpha_shape"].buffer(-elbow)
    if not elbow_poly.is_empty:
        _plot_poly_boundary(ax, elbow_poly, color=elbow_color, lw=2)

    legend_handles = [
        Line2D([0], [0], color=boundary_color, lw=1, ls="-", alpha=0.7, label=boundary_label),
        Line2D([0], [0], color=reimann_neuron_color, lw=1.5, ls="-", label=reimann_neuron_label),
        Line2D([0], [0], color=reimann_bin_color, lw=1.5, ls="-", label=reimann_bin_label),
        Line2D([0], [0], color=elbow_color, lw=2, ls="-", label=f"bosperrus ({elbow / 1e3:.0f} µm)"),
    ]
    ax.legend(handles=legend_handles, fontsize=7, frameon=True,
              loc="upper center", bbox_to_anchor=legend_bbox_to_anchor,
              ncols=2, borderaxespad=0)
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("z (µm)")
    ax.set_aspect("equal")
    ax.spines[["top", "right"]].set_visible(False)


def main(fn_mat=FN_MAT, save_path_svg=SAVE_PATH_SVG, save_path_pdf=SAVE_PATH_PDF):
    """Reproduce the original 2-panel MICrONS figure (fits + boundary heatmap)
    and print the bosperrus-vs-Reimann comparison summary."""
    plt.rcParams["svg.fonttype"] = "none"

    data = prepare_microns_data(fn_mat)
    print_comparison_summary(data["comparison"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3), width_ratios=[1, 2])
    fig.subplots_adjust(wspace=0.2)

    plot_fits_panel(ax1, data["flow"], data["alpha_distances"], data["scores"])
    plot_boundary_heatmap_panel(ax2, data)

    for ax, label in zip([ax1, ax2], ["a.", "b."]):
        ax.text(-0.18, 1.08, label, transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top")

    os.makedirs(os.path.dirname(save_path_svg), exist_ok=True)
    fig.savefig(save_path_svg, bbox_inches="tight")
    fig.savefig(save_path_pdf, bbox_inches="tight", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
