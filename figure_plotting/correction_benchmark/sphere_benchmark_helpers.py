"""
Plotting code for figure_plotting/correction_benchmark/sphere_benchmark.ipynb (sphere benchmark figure).
a-c: one example point set per pattern: uniform, von Mises-Fisher kappa = 1, mixture kappa = 1, 3, 5.
d: the cropped cap on a sphere (kept points dark, removed light).
e: violin plots comparing raw / detrended (linear, spline) / bosperrus-corrected (additive and log,
see sphere.py) / SERN-corrected centralities against the uncropped ("original") sphere, as
Spearman correlations over all crop nodes (sphere.py also writes Pearson and a border-near
quartile subset; neither is plotted). Rows: Delaunay (stereographic projection, unpruned), k-NN,
r-NN. A yellow star marks the method with the highest median correlation in each panel.
f: log BOSPERRUS minus raw, median Spearman correlation, uniform vs. clustered (kappa = 1 and
kappa = 1, 3, 5 runs pooled), one block under each violin column.

Data: results/correction_benchmark/sphere{,_detrend}/ (src/correction_benchmark/sphere.py).
"""
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.gridspec import GridSpecFromSubplotSpec
from scipy.spatial import ConvexHull
from scipy.stats import vonmises_fisher

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "correction_benchmark"))  # load_results, sphere (samplers)

from load_results import load_sphere  # noqa: E402

METHOD_PALETTE = {"raw": "grey", "linear": "#D9C27E", "spline": "#A8842A", "bosperrus": "#8ABBD6", "bosperrus_log": "#1B3F66",
                  "SERN": "#E59EDD"}
# raw first as the reference, then by overall average rank (best first)
METHOD_ORDER = ["raw", "bosperrus_log", "bosperrus", "spline", "linear", "SERN"]
METHOD_LABELS = {"raw": "raw", "linear": "linear detr.", "spline": "spline detr.", "bosperrus": "bosperrus add.", "bosperrus_log": "bosperrus log", "SERN": "SERN"}
ROW_ORDER = ["delaunay", "knn", "rnn"]
ROW_LABELS = {"delaunay": "Delaunay", "knn": "$k$-NN", "rnn": "$r$-NN"}
TEXTWIDTH = (8.5 * 2.54 - 4.0) / 2.54  # letterpaper, 2 cm margins
plt.rcParams.update({"pdf.fonttype": 42, "svg.fonttype": "none", "font.size": 6.5})
MEASURE_LABELS = {"betweenness": "betweenness", "closeness": "closeness", "clustering": "clustering",
                  "degree": "degree", "pagerank": "PageRank"}
PATTERNS = {"uniform": ["uniform"], "clustered": ["kappa=1", "kappa=1,3,5"]}
CAP_RADIUS = 2.0  # as in sphere.py
N_POINTS = 5000  # as in sphere.py
HEAT_LIM = 0.1
# von Mises-Fisher clusters, by kappa: one hue no method uses, darker = more concentrated
CLUSTER_COLORS = {1: "#A1D99B", 3: "#41AB5D", 5: "#00441B"}

# The method with the highest median correlation in each panel gets a star
# (all of them if tied), in the shared palette's yellow.
WINNER_COLOR = json.load(open(ROOT / "fit_palette.json"))["Michaelis-Menten Fit"]
METHODS = METHOD_ORDER
N_SAMPLE = 200


def load_data():
    all_dfs = load_sphere(ROOT / "results" / "correction_benchmark")
    all_dfs = all_dfs[all_dfs["measure"].isin(
        ["betweenness", "degree", "closeness", "clustering", "pagerank"]
    )].copy()

    all_dfs = all_dfs.rename(columns={
        "original vs. on crop": "raw",
        "original vs. BOSPERRUS corrected on crop": "bosperrus",
        "original vs. BOSPERRUS multiplicative on crop": "bosperrus_mult",
        "original vs. BOSPERRUS log on crop": "bosperrus_log",
        "original vs. linear detrended on crop": "linear",
        "original vs. spline detrended on crop": "spline",
        "original vs. SERN corrected on crop": "SERN",
    })
    all_dfs["radius"] = all_dfs["radius"].fillna(0)
    # r-NN r = 0.05 is excluded from figure 5 (fragmented crop graph; see sphere.py), also for runs
    # made before it was removed from sphere.py
    all_dfs = all_dfs[~((all_dfs["graph_type"] == "rnn") & np.isclose(all_dfs["radius"], 0.05))]
    all_dfs["k"] = all_dfs["k"].fillna(0)

    if "node_subset" not in all_dfs:
        all_dfs["node_subset"] = "all"
    for col in METHODS:
        all_dfs[col] = np.abs(all_dfs[col])

    all_dfs = all_dfs.reset_index(drop=True)
    all_dfs = all_dfs.iloc[all_dfs[METHODS].dropna(how="all").index]
    return all_dfs


def _basis(view):
    view = view / np.linalg.norm(view)
    e1 = np.cross(view, [0, 0, 1.0] if abs(view[2]) < 0.9 else [1.0, 0, 0])
    e1 /= np.linalg.norm(e1)
    return view, e1, np.cross(view, e1)


def plot_sphere(ax, coords, view, title, center=None, centers=()):
    """Orthographic view of the visible hemisphere. With `center`: the cap of CAP_RADIUS around it
    is drawn dark (kept by the crop), the rest light, plus the cap border."""
    view, e1, e2 = _basis(view)
    front = coords @ view > 0
    if center is None:
        layers = [(front, "#333333")]
    else:
        inside = np.arccos(np.clip(coords @ center, -1, 1)) <= CAP_RADIUS
        layers = [(front & ~inside, "#bbbbbb"), (front & inside, "#333333")]
    if centers:
        # clustered: colour each point by the cluster it was drawn from (clusters are consecutive blocks)
        labels = np.repeat(np.arange(len(centers)), len(coords) // len(centers))
        idx = np.where(front)[0]
        idx = idx[np.random.default_rng(0).permutation(len(idx))]  # no cluster systematically on top
        ax.scatter(coords[idx] @ e1, coords[idx] @ e2, s=0.8, c=[CLUSTER_COLORS[centers[l][1]] for l in labels[idx]],
                   linewidths=0, alpha=0.8, rasterized=True)
    else:
        for mask, color in layers:
            ax.scatter(coords[mask] @ e1, coords[mask] @ e2, s=0.7, color=color, linewidths=0, rasterized=True)
    t = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(t), np.sin(t), color="k", lw=0.5)
    if center is not None:
        _, u, w = _basis(center)
        border = np.cos(CAP_RADIUS) * center + np.sin(CAP_RADIUS) * (np.outer(np.cos(t), u) + np.outer(np.sin(t), w))
        vis = border @ view > 0
        ax.plot(np.where(vis, border @ e1, np.nan), np.where(vis, border @ e2, np.nan), color="k", lw=0.9)
    _mark_centers(ax, centers, view, e1, e2)
    _finish_sphere(ax, title)


def _mark_centers(ax, centers, view, e1, e2):
    """Cluster centres in their cluster colour: filled if on the visible hemisphere, open otherwise; labelled with kappa."""
    for mu, kappa in centers:
        front = mu @ view > 0
        x, y = mu @ e1, mu @ e2
        c = 0.55 * np.array(matplotlib.colors.to_rgb(CLUSTER_COLORS[kappa]))  # darker shade of the cluster colour
        ax.plot(x, y, "o", ms=6, mfc=c if front else "none", mec="white" if front else c, mew=0.9, zorder=5)
        ax.text(x + 0.09, y + 0.09, f"$\\kappa$={kappa}", fontsize=5.5, zorder=6,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))


def _finish_sphere(ax, title):
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=6.5, pad=2)


def sample_vmf_with_centers(n, kappas, rng):
    """Same draws as sphere.sample_von_mises_fisher (same rng sequence), plus the cluster centres."""
    from sphere import sample_uniform_on_unit_sphere
    per_cluster = n if len(kappas) == 1 else n // len(kappas)
    pts, centers = [], []
    for kappa in kappas:
        mu = sample_uniform_on_unit_sphere(1, rng=rng)[0]
        pts.append(vonmises_fisher(mu=mu, kappa=kappa).rvs(size=per_cluster, random_state=rng))
        centers.append((mu, kappa))
    return np.vstack(pts), centers


def plot_crop_schematic(ax, view, title, n=180, seed=3):
    """Schematic triangulation of a sparse uniform point set, with the cap border in red:
    edges with both ends in the cap (kept) dark, the rest light."""
    from sphere import sample_uniform_on_unit_sphere
    view, e1, e2 = _basis(view)
    pts = sample_uniform_on_unit_sphere(n, rng=np.random.default_rng(seed))
    # cap centre tilted so that its border runs across the visible hemisphere
    center = np.cos(CAP_RADIUS - 0.5) * view + np.sin(CAP_RADIUS - 0.5) * e2
    center /= np.linalg.norm(center)
    inside = np.arccos(np.clip(pts @ center, -1, 1)) <= CAP_RADIUS
    front = pts @ view > 0
    edges = {tuple(sorted((a, b))) for t in ConvexHull(pts).simplices for a, b in [(t[0], t[1]), (t[1], t[2]), (t[0], t[2])]}
    for a, b in edges:
        if front[a] and front[b]:
            kept = inside[a] and inside[b]
            ax.plot([pts[a] @ e1, pts[b] @ e1], [pts[a] @ e2, pts[b] @ e2], color="#333333" if kept else "#bbbbbb",
                    lw=0.4, zorder=1 if not kept else 2)
    ax.scatter(pts[front] @ e1, pts[front] @ e2, s=2.5, c=np.where(inside[front], "#333333", "#bbbbbb"), linewidths=0, zorder=3)
    t = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(t), np.sin(t), color="k", lw=0.5)
    _, u, w = _basis(center)
    border = np.cos(CAP_RADIUS) * center + np.sin(CAP_RADIUS) * (np.outer(np.cos(t), u) + np.outer(np.sin(t), w))
    vis = border @ view > 0
    ax.plot(np.where(vis, border @ e1, np.nan), np.where(vis, border @ e2, np.nan), color="#D62728", lw=1.0, zorder=4)
    _finish_sphere(ax, title)


def pattern_gain(df, graph, measure, coords):
    s = df[(df.graph_type == graph) & (df.measure == measure) & df.coord_type.isin(coords)]
    return s["bosperrus_log"].median() - s["raw"].median()


def plot_pattern_heatmaps(axes, df):
    """One 2 x 3 block per measure (rows: point pattern, columns: graph type), under its violin column."""
    measures = sorted(df["measure"].unique())
    for c, (ax, m) in enumerate(zip(axes, measures)):
        vals = np.array([[pattern_gain(df, g, m, coords) for g in ROW_ORDER] for coords in PATTERNS.values()])
        im = ax.imshow(vals, cmap="RdBu", vmin=-HEAT_LIM, vmax=HEAT_LIM, aspect="auto")
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                v = vals[i, j]
                ax.text(j, i, f"{v:.2g}" if abs(v) >= 0.001 else "0", ha="center", va="center", fontsize=5,
                        color="white" if abs(v) > 0.6 * HEAT_LIM else "black")
        ax.set_xticks(range(len(ROW_ORDER)), [ROW_LABELS[g] for g in ROW_ORDER], fontsize=5.5)
        ax.set_yticks(range(len(PATTERNS)), list(PATTERNS) if c == 0 else [], fontsize=5.5)
        ax.tick_params(length=0)
    cax = axes[-1].inset_axes([1.06, 0, 0.06, 1])
    cb = plt.colorbar(im, cax=cax, extend="both", ticks=[-HEAT_LIM, 0, HEAT_LIM])
    cb.ax.tick_params(labelsize=5.5)


def plot_violins(axes, df):
    measures = sorted(df["measure"].unique())
    rng = np.random.default_rng(42)
    for r, g in enumerate(ROW_ORDER):
        for c, m in enumerate(measures):
            ax = axes[r, c]
            subset = df[(df.graph_type == g) & (df.measure == m)][METHODS]
            long = subset.melt(var_name="method", value_name="correlation").dropna()
            sns.violinplot(data=long, x="method", y="correlation", hue="method", order=METHOD_ORDER,
                           palette=METHOD_PALETTE, inner="box", cut=0, linewidth=0.4, ax=ax, legend=False,
                           inner_kws=dict(box_width=2, whis_width=0.6))

            # paired lines (sampled subset for readability)
            s = subset.dropna(how="all")
            sample = s.iloc[rng.choice(len(s), size=min(N_SAMPLE, len(s)), replace=False)]
            xs, ys = [], []
            for _, row in sample.iterrows():
                pts = [(i, row[mm]) for i, mm in enumerate(METHODS) if not np.isnan(row[mm])]
                if len(pts) >= 2:
                    xs += [p[0] for p in pts] + [np.nan]
                    ys += [p[1] for p in pts] + [np.nan]
            ax.plot(xs, ys, color="k", alpha=0.05, lw=0.3, zorder=0, rasterized=True)

            # winner star
            med = subset.median()
            for mm in med.index[med == med.max()]:
                ax.plot(METHOD_ORDER.index(mm), 1.08, marker="*", ms=5, color=WINNER_COLOR,
                        markeredgewidth=0, clip_on=False, zorder=5)

            ax.set_ylim(-0.05, 1.12)
            ax.set_yticks([0, 0.5, 1])
            ax.set_xlabel("")
            ax.set_ylabel(f"{ROW_LABELS[g]}\nSpearman $\\rho$" if c == 0 else "")
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=5.5, length=2)
            if c > 0:
                ax.tick_params(labelleft=False)
            if r == 0:
                ax.set_title(MEASURE_LABELS[m], fontsize=6.5, pad=7)
            if r == len(ROW_ORDER) - 1:
                ax.set_xticks(range(len(METHOD_ORDER)), [METHOD_LABELS[mm] for mm in METHOD_ORDER], rotation=90, fontsize=5.5)
            else:
                ax.set_xticks(range(len(METHOD_ORDER)), [])


def make_figure(all_dfs):
    df = all_dfs[(all_dfs["corr_method"] == "spearman") & (all_dfs["node_subset"] == "all")]
    from sphere import sample_uniform_on_unit_sphere, sample_von_mises_fisher
    measures = sorted(df["measure"].unique())

    fig = plt.figure(figsize=(TEXTWIDTH, TEXTWIDTH * 0.78))
    outer = fig.add_gridspec(2, 1, height_ratios=[1, 3.3], hspace=0.08, left=0.085, right=0.9, top=0.97, bottom=0.03)

    # a-d: point patterns and the crop
    top = GridSpecFromSubplotSpec(1, 4, outer[0], wspace=0.15)
    v = np.array([1.0, -1.0, 0.6])
    ax = fig.add_subplot(top[0])
    plot_sphere(ax, sample_uniform_on_unit_sphere(N_POINTS, rng=np.random.default_rng(0)), v, "uniform")
    for k, (kappas, seed) in enumerate([([1], 1), ([1, 3, 5], 4)], start=1):
        coords, centers = sample_vmf_with_centers(N_POINTS, kappas, np.random.default_rng(seed))
        assert np.allclose(coords, sample_von_mises_fisher(N_POINTS, kappas, rng=np.random.default_rng(seed)))
        mus = np.array([mu for mu, _ in centers])
        if len(mus) == 1:  # view 40 degrees off the centre, so the density gradient runs across the disc
            view = np.cos(np.radians(40)) * mus[0] + np.sin(np.radians(40)) * _basis(mus[0])[1]
        else:  # the direction from which all centres are best visible
            cand = np.random.default_rng(0).normal(size=(4000, 3))
            cand /= np.linalg.norm(cand, axis=1)[:, None]
            view = cand[np.argmax((cand @ mus.T).min(axis=1))]
        ax = fig.add_subplot(top[k])
        plot_sphere(ax, coords, view, "clustered, " + ", ".join(f"$\\kappa$ = {kk}" if i == 0 else str(kk) for i, kk in enumerate(kappas)),
                    centers=centers)
    plot_crop_schematic(fig.add_subplot(top[3]), v, f"crop: cap of radius {CAP_RADIUS:.2g} rad")
    for k in range(4):
        fig.axes[k].text(-0.02, 1.0, "abcd"[k], transform=fig.axes[k].transAxes, fontsize=8, fontweight="bold", va="top")

    # e: violins, f: uniform vs. clustered gain under each violin column
    bottom = GridSpecFromSubplotSpec(len(ROW_ORDER) + 2, len(measures), outer[1],
                                     height_ratios=[1] * len(ROW_ORDER) + [0.95, 0.42], hspace=0.12, wspace=0.1)
    axes = np.array([[fig.add_subplot(bottom[r, c]) for c in range(len(measures))] for r in range(len(ROW_ORDER))])
    plot_violins(axes, df)
    axes[0, 0].text(-0.5, 1.1, "e", transform=axes[0, 0].transAxes, fontsize=8, fontweight="bold", va="bottom")
    heat_axes = [fig.add_subplot(bottom[len(ROW_ORDER) + 1, c]) for c in range(len(measures))]
    plot_pattern_heatmaps(heat_axes, df)
    heat_axes[0].text(-0.5, 1.08, "f", transform=heat_axes[0].transAxes, fontsize=8, fontweight="bold", va="bottom")
    heat_axes[0].text(-0.3, 1.08, "BOSPERRUS log − raw, median Spearman $\\rho$, by point pattern",
                      transform=heat_axes[0].transAxes, fontsize=6.5, va="bottom")
    return fig
