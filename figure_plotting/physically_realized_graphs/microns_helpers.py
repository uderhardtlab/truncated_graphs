"""Plotting code for microns.ipynb: the MICrONS degree-vs-boundary-distance fit and the per-bin maps (mean degree;
BOSPERRUS vs. Reimann analysis buffers).
Data: results/physically_realized_graphs/microns/ (src/physically_realized_graphs/microns_buffer.py)."""
import json
from pathlib import Path

import bosperrus
import numpy as np
import pandas as pd
import shapely.wkt
from matplotlib import pyplot as plt

RES = Path(__file__).resolve().parent.parent.parent / "results" / "physically_realized_graphs" / "microns"


def load():
    summary = json.loads((RES / "summary.json").read_text())
    x_edges, z_edges = np.array(summary["x_edges_um"]), np.array(summary["z_edges_um"])
    return dict(
        neurons=pd.read_csv(RES / "neurons.csv", index_col=0),
        frac_max=pd.read_csv(RES / "outer_fraction_max.csv", index_col=0),
        frac_mean=pd.read_csv(RES / "outer_fraction_mean.csv", index_col=0),
        alpha_shape=shapely.wkt.loads((RES / "alpha_shape.wkt").read_text()),
        summary=summary, x_edges_um=x_edges, z_edges_um=z_edges,
        x_centers_um=(x_edges[:-1] + x_edges[1:]) / 2, z_centers_um=(z_edges[:-1] + z_edges[1:]) / 2,
    )


def piecewise_linear(params):
    """Predictor (d in nm) of a fitted piecewise-linear model: m*d + c up to the knot b, plateau after."""
    b, m, c = (params[f"piecewise_linear_{k}"] for k in "bmc")
    return lambda d: m * np.minimum(np.asarray(d, dtype=float), b) + c


def plot_fits_panel(ax, res, measure, fit_color, legend_label_fn, ylabel, title, log_y=True):
    """Quantile bands + per-distance mean (bosperrus.plot_fit) of `measure` vs. distance to the alpha-shape
    boundary, with the fitted piecewise-linear curve and its knot."""
    params = res["summary"]["piecewise_linear_fits"][measure]
    b = params["piecewise_linear_b"]
    predict = piecewise_linear(params)
    n = res["neurons"]
    bosperrus.plot_fit(ax, n["alpha_distance_nm"] / 1e3, n[measure], lambda d_um: predict(np.asarray(d_um) * 1e3),
                       line_kwargs={"color": fit_color, "linewidth": 2, "label": legend_label_fn(b)})
    ax.axvline(b / 1e3, color=fit_color, lw=1.2, ls="--", alpha=0.85)
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("Distance to boundary (µm)")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8, markerscale=6, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)


def bin_table(res):
    """Per x-z bin (Reimann's grid) with neurons: mean degree, share of neurons inside the BOSPERRUS buffer and
    Reimann's per-bin flag (max outer synapse fraction > threshold; NaN where no neuron has incoming synapses)."""
    s, n = res["summary"], res["neurons"].copy()
    n["x_bin"] = np.digitize(n["x_nm"] / 1e3, s["x_edges_um"])
    n["z_bin"] = np.digitize(n["z_nm"] / 1e3, s["z_edges_um"])
    n["in_buffer"] = n["alpha_distance_nm"] < s["elbow_nm"]
    n_bins = len(s["x_edges_um"]) - 1
    n = n[n["x_bin"].between(1, n_bins) & n["z_bin"].between(1, n_bins)]  # the grid spans the synapses (1 neuron out)
    b = n.groupby(["z_bin", "x_bin"]).agg(degree=("degree", "mean"), buffer_share=("in_buffer", "mean"))
    frac_max = res["frac_max"].copy()
    frac_max.columns = frac_max.columns.astype(int)
    b["reimann"] = [frac_max.at[z, x] > s["reimann_thresh"] if not np.isnan(frac_max.at[z, x]) else np.nan
                    for z, x in b.index]
    return b


def _bin_image(b, res, colors):
    """RGBA image over Reimann's grid (rows z bin 1.., columns x bin 1..), transparent where a bin has no neurons."""
    nz, nx = len(res["z_edges_um"]) - 1, len(res["x_edges_um"]) - 1
    img = np.zeros((nz, nx, 4))
    for (z, x), rgba in colors.items():
        img[z - 1, x - 1] = rgba
    return img


def plot_bin_maps(ax_degree, ax_class, res, cmap, colors, aspect):
    """Left: mean degree per bin. Right: bins excluded by BOSPERRUS and Reimann (both), Reimann only, BOSPERRUS
    only (majority of the bin's neurons inside the buffer), no Reimann value; the rest by mean degree as on the
    left. colors: dict with keys both, reimann_only, bosperrus_only, no_value. Returns the degree mappable."""
    b = bin_table(res)
    norm = plt.Normalize(b["degree"].quantile(0.02), b["degree"].quantile(0.98))
    cm = plt.get_cmap(cmap)
    to_rgba = plt.matplotlib.colors.to_rgba
    degree_rgba = {i: cm(norm(v)) for i, v in b["degree"].items()}
    bsp, rei = b["buffer_share"] > 0.5, b["reimann"]
    cls = {}
    for i in b.index:
        if np.isnan(rei[i]):
            cls[i] = to_rgba(colors["no_value"])
        elif bsp[i] and rei[i]:
            cls[i] = to_rgba(colors["both"])
        elif rei[i]:
            cls[i] = to_rgba(colors["reimann_only"])
        elif bsp[i]:
            cls[i] = to_rgba(colors["bosperrus_only"])
        else:
            cls[i] = degree_rgba[i]
    extent = [res["x_edges_um"][0], res["x_edges_um"][-1], res["z_edges_um"][0], res["z_edges_um"][-1]]
    for ax, c in [(ax_degree, degree_rgba), (ax_class, cls)]:
        ax.imshow(_bin_image(b, res, c), origin="lower", extent=extent, aspect=aspect, interpolation="nearest",
                  rasterized=True)
        ax.set_xlabel("x (µm)")
        ax.spines[["top", "right"]].set_visible(False)
    ax_degree.set_ylabel("z (µm)")
    ax_class.tick_params(labelleft=False)
    counts = {
        "both": int((bsp & (rei == True)).sum()),  # noqa: E712 (NaN-aware)
        "reimann_only": int((~bsp & (rei == True)).sum()),  # noqa: E712
        "bosperrus_only": int((bsp & (rei == False)).sum()),  # noqa: E712
    }
    return plt.cm.ScalarMappable(norm=norm, cmap=cm), counts
