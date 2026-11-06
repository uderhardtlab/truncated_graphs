"""MICrONS connectome border-effect analysis.

Compares BOSPERRUS's alpha-shape-boundary-distance elbow against the Reimann
lab's "outer synapse fraction" heuristic for identifying which neurons/spatial
bins in the MICrONS mm^3 connectome are affected by the edge effect (missing
synapses near the reconstructed volume's boundary).

Library for microns_buffer.py, which owns every path and threshold; nothing here has a default.
"""
import alphashape
import bosperrus
import conntility
import numpy as np
import pandas as pd


# ── data preparation ──────────────────────────────────────────────────────────
def load_connectome(fn_mat, dataset="full"):
    """Load the MICrONS connectome from an h5 file via conntility."""
    return conntility.ConnectivityMatrix.from_h5(fn_mat, dataset)


def bin_reimann(M, nbins):
    """Bin edges/vertices into an nbins x nbins x-z grid and flag "outer" bins
    (Reimann's border-region heuristic: bin synapse count < 1000, or within 3
    bins of the top/bottom of the z-stack). Mutates M in place (adds edge/vertex
    properties). Returns the compressed per-neuron view C with an added
    "outer_syn_fraction" vertex property, plus the raw per-bin synapse-count table.
    """
    x_col, z_col = f"x_nm_binned_{nbins}", f"z_nm_binned_{nbins}"
    for col, binned_col in [("x_nm", x_col), ("z_nm", z_col)]:
        bins = np.linspace(M.edges[col].min(), M.edges[col].max() + 1, nbins)
        M.add_edge_property(binned_col, np.digitize(M.edges[col], bins=bins))
        M.add_vertex_property(binned_col, np.digitize(M.vertices[col], bins=bins))

    counts = M.edges.groupby([x_col, z_col])["id"].count().unstack(x_col)

    edge_idxx = pd.MultiIndex.from_frame(M.edges[[z_col, x_col]])
    is_outer = counts.stack().rename("count").reset_index()
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
    return C, counts


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
        """|a & b| / |a | b| of two boolean Series."""
        return (a & b).sum() / (a | b).sum()

    j_neuron = jaccard(df["bosperrus_affected"], df["reimann_affected"])

    z_bins = M.vertices.loc[df.index, z_col]
    x_bins = M.vertices.loc[df.index, x_col]

    grp = C.vertices.groupby([x_col, z_col])["outer_syn_fraction"]
    I_frac_max = grp.max().unstack(x_col).reindex(index=range(1, nbins), columns=range(1, nbins))
    I_frac_mean = grp.mean().unstack(x_col).reindex(index=range(1, nbins), columns=range(1, nbins))

    bin_max_frac = pd.Series(
        [I_frac_max.at[zb, xb] if (zb in I_frac_max.index and xb in I_frac_max.columns) else np.nan
         for zb, xb in zip(z_bins, x_bins, strict=True)],
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
