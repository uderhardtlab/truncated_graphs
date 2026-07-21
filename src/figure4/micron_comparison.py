import os
import numpy as np
import matplotlib
matplotlib.rcParams['svg.fonttype'] = 'none'
import pandas as pd
import alphashape
from shapely.geometry import Point
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import conntility
import bosperrus

# ── paths & constants ─────────────────────────────────────────────────────────
# FN_MAT still points at a je30bery-only path that doesn't resolve on this
# checkout — update once the actual connectome file's location here is known.
FN_MAT     = "/data/bionets/je30bery/bosperrus-experiments/reimann_data/microns_mm3_connectome.h5"
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

# ── load data ─────────────────────────────────────────────────────────────────
M = conntility.ConnectivityMatrix.from_h5(FN_MAT, "full")

# ── Reimann binning ───────────────────────────────────────────────────────────
for col in ["x_nm", "z_nm"]:
    bins = np.linspace(M.edges[col].min(), M.edges[col].max() + 1, NBINS)
    M.add_edge_property(col + f"_binned_{NBINS}", np.digitize(M.edges[col], bins=bins))
    M.add_vertex_property(col + f"_binned_{NBINS}", np.digitize(M.vertices[col], bins=bins))

I = (M.edges
     .groupby(["x_nm_binned_51", "z_nm_binned_51"])["id"]
     .count()
     .unstack("x_nm_binned_51"))

edge_idxx = pd.MultiIndex.from_frame(M.edges[["z_nm_binned_51", "x_nm_binned_51"]])
is_outer = I.stack().rename("count").reset_index()
is_outer["outer"] = (
    (is_outer["count"] < 1000) |
    (is_outer["z_nm_binned_51"] <= 3) |
    (is_outer["z_nm_binned_51"] >= (NBINS - 3))
)
is_outer = is_outer.set_index(["z_nm_binned_51", "x_nm_binned_51"])["outer"]
M.add_edge_property("syn_in_outer_bin", is_outer[edge_idxx].values)

C = M.compress({"outer_bin_count": ("syn_in_outer_bin", "sum")})
outer_per_neuron = np.array(C.default("outer_bin_count").matrix.sum(axis=0))[0]
C.add_vertex_property("outer_syn_fraction",
                      outer_per_neuron / C.vertices["indegree"].values)

# ── alpha shape & distances ───────────────────────────────────────────────────
coords_xz   = M.vertices[["x_nm", "z_nm"]].values
alpha_shape = alphashape.alphashape(coords_xz, alpha=ALPHA_SHAPE)
alpha_distances = np.array([
    alpha_shape.boundary.distance(Point(x, z))
    for x, z in coords_xz
])

# ── centrality scores & bosperrus fit ─────────────────────────────────────────
scores = M.vertices[["indegree", "outdegree"]].copy()
scores["degree"] = scores[["indegree", "outdegree"]].sum(axis=1)

flow = bosperrus.Flow.from_distances_and_scores(
    distances=pd.Series(alpha_distances, name="alpha_distance"),
    scores=scores,
)
flow.flow(fits=[bosperrus.PiecewiseLinearFit, bosperrus.ConstantFit],
          baseline_fit_class=bosperrus.ConstantFit)

elbows = {s: flow.best_fits[s].params["piecewise_linear_b"]
          for s in ["indegree", "outdegree", "degree"]}
elbow = elbows["indegree"]   # indegree elbow used for binary classification

# ── per-neuron comparison dataframe ───────────────────────────────────────────
df = M.vertices[["x_nm", "z_nm"]].copy()
df["outer_syn_fraction"]  = C.vertices["outer_syn_fraction"].values
df["bosperrus_distance"]  = alpha_distances
df = df.dropna(subset=["outer_syn_fraction"])
df["reimann_affected"]    = df["outer_syn_fraction"] > REIMANN_THRESH
df["bosperrus_affected"]  = df["bosperrus_distance"]  < elbow

# ── Jaccard indices ───────────────────────────────────────────────────────────
def jaccard(a, b):
    return (a & b).sum() / (a | b).sum()

j_neuron = jaccard(df["bosperrus_affected"], df["reimann_affected"])

z_bins = M.vertices.loc[df.index, "z_nm_binned_51"]
x_bins = M.vertices.loc[df.index, "x_nm_binned_51"]

_grp = C.vertices.groupby(["x_nm_binned_51", "z_nm_binned_51"])["outer_syn_fraction"]
I_frac_max = (
    _grp.max()
    .unstack("x_nm_binned_51")
    .reindex(index=range(1, NBINS), columns=range(1, NBINS))
)
I_frac_mean = (
    _grp.mean()
    .unstack("x_nm_binned_51")
    .reindex(index=range(1, NBINS), columns=range(1, NBINS))
)

bin_max_frac = pd.Series(
    [I_frac_max.at[zb, xb]
     if (zb in I_frac_max.index and xb in I_frac_max.columns) else np.nan
     for zb, xb in zip(z_bins, x_bins)],
    index=df.index,
)
bin_reimann_affected = bin_max_frac > REIMANN_THRESH
j_bin = jaccard(df["bosperrus_affected"], bin_reimann_affected)

n_bos = int(df["bosperrus_affected"].sum())
n_rei = int(df["reimann_affected"].sum())
n_bin = int(bin_reimann_affected.sum())

print(f"{'':30s}  {'bosperrus':>10}  {'Reimann (neuron)':>16}  {'Reimann (bin)':>13}")
print(f"{'Neurons affected':30s}  {n_bos:>10,}  {n_rei:>16,}  {n_bin:>13,}")
print(f"{'  (% of total)':30s}  {n_bos/len(df)*100:>9.1f}%"
      f"  {n_rei/len(df)*100:>15.1f}%  {n_bin/len(df)*100:>12.1f}%")
print()
print(f"Jaccard vs bosperrus")
print(f"  Reimann per-neuron : {j_neuron:.3f}")
print(f"  Reimann per-bin    : {j_bin:.3f}")

# ── figure ────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3), width_ratios=[1, 2])
fig.subplots_adjust(wspace=0.2)

# Panel A – centrality scores vs distance to boundary
d_sorted = np.sort(alpha_distances)
for score in ["indegree", "outdegree", "degree"]:
    b   = elbows[score]
    m   = flow.best_fits[score].params["piecewise_linear_m"]
    c   = flow.best_fits[score].params["piecewise_linear_c"]
    col = SCORE_COLORS[score]
    ax1.scatter(alpha_distances / 1e3, scores[score],
                s=0.4, alpha=0.06, color=col, linewidths=0, rasterized=True)
    y_fit = bosperrus.PiecewiseLinearFit.piecewise_plateau(d_sorted, b=b, m=m, c=c)
    ax1.plot(d_sorted / 1e3, y_fit, color=col, lw=2, label=SCORE_LABELS[score])
    ax1.axvline(b / 1e3, color=col, lw=1.2, ls="--", alpha=0.85)
ax1.set_yscale("log")
ax1.set_xlabel("Distance to boundary (µm)")
ax1.set_ylabel("Centrality score")
ax1.legend(fontsize=8, markerscale=6, frameon=False)
ax1.spines[["top", "right"]].set_visible(False)

# Panel B – max outer-synapse fraction heatmap + contours + bosperrus polygon
x_edges_um   = np.linspace(M.edges["x_nm"].min(), M.edges["x_nm"].max() + 1, NBINS) / 1e3
z_edges_um   = np.linspace(M.edges["z_nm"].min(), M.edges["z_nm"].max() + 1, NBINS) / 1e3
x_centers_um = (x_edges_um[:-1] + x_edges_um[1:]) / 2
z_centers_um = (z_edges_um[:-1] + z_edges_um[1:]) / 2

cmap = plt.cm.magma.copy()
cmap.set_bad("0.85")
pcm = ax2.pcolormesh(x_edges_um, z_edges_um, I_frac_max.values,
                     cmap=cmap, shading="flat", rasterized=True, vmin=0, vmax=1)
plt.colorbar(pcm, ax=ax2, label="Max outer synapse fraction", shrink=0.75, pad=0.02)

ax2.contour(x_centers_um, z_centers_um, I_frac_mean.values,
            levels=[REIMANN_THRESH], colors=[C_SAFE], linewidths=1.5, linestyles="-")
ax2.contour(x_centers_um, z_centers_um, I_frac_max.values,
            levels=[REIMANN_THRESH], colors=[C_SAFE2], linewidths=1.5, linestyles="-")

def _plot_poly_boundary(ax, geom, **kwargs):
    polys = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
    for p in polys:
        if hasattr(p, "exterior"):
            coords = np.array(p.exterior.coords)
            ax.plot(coords[:, 0] / 1e3, coords[:, 1] / 1e3, **kwargs)

_plot_poly_boundary(ax2, alpha_shape, color="gray", lw=1, ls="-", alpha=0.7)
elbow_poly = alpha_shape.buffer(-elbow)
if not elbow_poly.is_empty:
    _plot_poly_boundary(ax2, elbow_poly, color=C_IND, lw=2)

legend_handles = [
    Line2D([0], [0], color="gray", lw=1,   ls="-", alpha=0.7, label="tissue boundary"),
    Line2D([0], [0], color=C_SAFE,  lw=1.5, ls="-", label="Reimann 5% (per neuron)"),
    Line2D([0], [0], color=C_SAFE2, lw=1.5, ls="-", label="Reimann 5% (per bin max)"),
    Line2D([0], [0], color=C_IND,   lw=2,   ls="-", label=f"bosperrus ({elbow / 1e3:.0f} µm)"),
]
ax2.legend(handles=legend_handles, fontsize=7, frameon=True,
           loc="upper center", bbox_to_anchor=(0.5, -0.4),
           ncols=2, borderaxespad=0)
ax2.set_xlabel("x (µm)")
ax2.set_ylabel("z (µm)")
ax2.set_aspect("equal")
ax2.spines[["top", "right"]].set_visible(False)

for ax, label in zip([ax1, ax2], ["a.", "b."]):
    ax.text(-0.18, 1.08, label, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top")

os.makedirs(os.path.dirname(SAVE_PATH_SVG), exist_ok=True)
fig.savefig(SAVE_PATH_SVG, bbox_inches="tight")
fig.savefig(SAVE_PATH_PDF, bbox_inches="tight", dpi=300)
plt.show()
