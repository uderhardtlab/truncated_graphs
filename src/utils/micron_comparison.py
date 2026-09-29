"""MICrONS connectome border-effect analysis.

Compares BOSPERRUS's alpha-shape-boundary-distance elbow against the Reimann
lab's "outer synapse fraction" heuristic for identifying which neurons/spatial
bins in the MICrONS mm^3 connectome are affected by the edge effect (missing
synapses near the reconstructed volume's boundary).

A library of functions driven by notebooks/supplement/MICRONS_analysis_buffer.ipynb,
which owns every path/threshold/style setting; nothing here has a default.
"""
import alphashape
import conntility
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

import bosperrus


# ── data preparation ──────────────────────────────────────────────────────────
def load_connectome(fn_mat, dataset="full"):
    """Load the MICrONS connectome from an h5 file via conntility."""
    return conntility.ConnectivityMatrix.from_h5(fn_mat, dataset)


def bin_reimann(M, nbins):
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


def compute_alpha_shape_distances(M, alpha):
    """Fit an alpha shape to the neurons' x-z footprint and compute each
    neuron's distance to its boundary (via bosperrus.distance_to_alpha_shape).
    Returns the shape geometry (needed for plotting) and the distances.
    """
    coords_xz = M.vertices[["x_nm", "z_nm"]].values
    alpha_shape = alphashape.alphashape(coords_xz, alpha=alpha)
    alpha_distances = bosperrus.distance_to_alpha_shape(coords_xz, alpha=alpha).values
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


def compute_reimann_comparison(M, C, flow, alpha_distances, reimann_thresh, nbins):
    """Compare BOSPERRUS's elbow-based affected-neuron classification (using
    the indegree fit's elbow) against Reimann's per-neuron and per-bin
    outer-synapse-fraction heuristics, via Jaccard index. Returns a dict with
    the comparison dataframe, Jaccard indices, counts, and the max/mean
    per-bin outer-synapse-fraction grids (for the boundary heatmap panel).
    """
    x_col, z_col = f"x_nm_binned_{nbins}", f"z_nm_binned_{nbins}"
    elbow = flow.best_fits["indegree"].params["piecewise_linear_b"]

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
        "df": df, "elbow": elbow,
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


def prepare_microns_data(fn_mat, nbins, alpha, reimann_thresh):
    """End-to-end MICrONS data preparation: load the connectome, bin it for
    the Reimann heuristic, fit BOSPERRUS against alpha-shape-boundary
    distance (indegree and degree only), and compute the bosperrus-vs-Reimann
    comparison. Returns a dict bundling everything the plotting functions need.
    """
    M = load_connectome(fn_mat)
    C, _ = bin_reimann(M, nbins=nbins)

    alpha_shape, alpha_distances = compute_alpha_shape_distances(M, alpha=alpha)

    scores = pd.DataFrame({
        "indegree": M.vertices["indegree"],
        "degree": M.vertices["indegree"] + M.vertices["outdegree"],
    })

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
def plot_fits_panel(ax, flow, alpha_distances, scores, measure,
                     fit_color, legend_label_fn, ylabel, title):
    """2D histogram (via `bosperrus.plot_fit`) of `measure`'s centrality score
    vs. distance to the alpha-shape boundary, overlaid with the fitted
    piecewise-linear BOSPERRUS curve. `legend_label_fn(measure, b, m, c)`
    builds the legend text (e.g. to show the fitted elbow value).
    """
    fit = flow.best_fits[measure]
    b = fit.params["piecewise_linear_b"]
    m = fit.params["piecewise_linear_m"]
    c = fit.params["piecewise_linear_c"]

    # plot_fit's predict_fn is evaluated on whatever d_grid it's given -- here
    # that's in um (matching the displayed axis), so convert back to the nm
    # scale the fit itself was trained on before calling fit.predict.
    bosperrus.plot_fit(
        ax, alpha_distances / 1e3, scores[measure],
        lambda d_um: fit.predict(np.asarray(d_um) * 1e3),
        line_kwargs={"color": fit_color, "linewidth": 2, "label": legend_label_fn(measure, b, m, c)},
    )
    ax.axvline(b / 1e3, color=fit_color, lw=1.2, ls="--", alpha=0.85)

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


def plot_boundary_heatmap_panel(ax, data, reimann_thresh, cmap,
                                 boundary_label, boundary_color,
                                 reimann_neuron_label, reimann_neuron_color,
                                 reimann_bin_label, reimann_bin_color,
                                 elbow_color, legend_bbox_to_anchor):
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
