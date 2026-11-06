"""Plotting code for microns.ipynb: the MICrONS degree-vs-boundary-distance fit and the outer-synapse-fraction
map with the Reimann contours, the alpha shape and the BOSPERRUS buffer. Data: results/physically_realized_graphs/
microns/ (src/physically_realized_graphs/microns_buffer.py)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import shapely.wkt
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

import bosperrus

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


def plot_fits_panel(ax, res, measure, fit_color, legend_label_fn, ylabel, title):
    """Quantile bands + per-distance mean (bosperrus.plot_fit) of `measure` vs. distance to the alpha-shape
    boundary, with the fitted piecewise-linear curve and its knot."""
    params = res["summary"]["piecewise_linear_fits"][measure]
    b = params["piecewise_linear_b"]
    predict = piecewise_linear(params)
    n = res["neurons"]
    bosperrus.plot_fit(ax, n["alpha_distance_nm"] / 1e3, n[measure], lambda d_um: predict(np.asarray(d_um) * 1e3),
                       line_kwargs={"color": fit_color, "linewidth": 2, "label": legend_label_fn(b)})
    ax.axvline(b / 1e3, color=fit_color, lw=1.2, ls="--", alpha=0.85)
    ax.set_yscale("log")
    ax.set_xlabel("Distance to boundary (µm)")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8, markerscale=6, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)


def _plot_poly_boundary(ax, geom, **kwargs):
    for p in (list(geom.geoms) if hasattr(geom, "geoms") else [geom]):
        if hasattr(p, "exterior"):
            coords = np.array(p.exterior.coords)
            ax.plot(coords[:, 0] / 1e3, coords[:, 1] / 1e3, **kwargs)


def plot_boundary_heatmap_panel(ax, res, cmap, boundary_label, boundary_color,
                                reimann_neuron_label, reimann_neuron_color,
                                reimann_bin_label, reimann_bin_color, elbow_color, legend_bbox_to_anchor):
    """Max outer-synapse-fraction map with the Reimann contours (per-neuron mean, per-bin max), the alpha-shape
    boundary and the BOSPERRUS buffer (alpha shape shrunk by the indegree knot)."""
    thresh, elbow = res["summary"]["reimann_thresh"], res["summary"]["elbow_nm"]
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("0.85")
    pcm = ax.pcolormesh(res["x_edges_um"], res["z_edges_um"], res["frac_max"].values,
                        cmap=cm, shading="flat", rasterized=True, vmin=0, vmax=1)
    plt.colorbar(pcm, ax=ax, label="Max outer synapse fraction", shrink=0.75, pad=0.02)
    ax.contour(res["x_centers_um"], res["z_centers_um"], res["frac_mean"].values,
               levels=[thresh], colors=[reimann_neuron_color], linewidths=1.5, linestyles="-")
    ax.contour(res["x_centers_um"], res["z_centers_um"], res["frac_max"].values,
               levels=[thresh], colors=[reimann_bin_color], linewidths=1.5, linestyles="-")
    _plot_poly_boundary(ax, res["alpha_shape"], color=boundary_color, lw=1, ls="-", alpha=0.7)
    inner = res["alpha_shape"].buffer(-elbow)
    if not inner.is_empty:
        _plot_poly_boundary(ax, inner, color=elbow_color, lw=2)
    ax.legend(handles=[
        Line2D([0], [0], color=boundary_color, lw=1, ls="-", alpha=0.7, label=boundary_label),
        Line2D([0], [0], color=reimann_neuron_color, lw=1.5, ls="-", label=reimann_neuron_label),
        Line2D([0], [0], color=reimann_bin_color, lw=1.5, ls="-", label=reimann_bin_label),
        Line2D([0], [0], color=elbow_color, lw=2, ls="-", label=f"bosperrus ({elbow / 1e3:.0f} µm)"),
    ], fontsize=7, frameon=True, loc="upper center", bbox_to_anchor=legend_bbox_to_anchor, ncols=2, borderaxespad=0)
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("z (µm)")
    ax.set_aspect("equal")
    ax.spines[["top", "right"]].set_visible(False)
