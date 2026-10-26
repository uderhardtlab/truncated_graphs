"""Illustrate the Delaunay cap artefact in the sphere benchmark.

The crop's Delaunay graph is the convex hull of the cap points (`_spherical_delaunay_edges`).
The hull also closes the cap opening with a flat "lid" of triangles, so border nodes get
edges across the cut and are not truncated.

Run from truncated_graphs/src:  python correction_benchmark/plot_delaunay_cap.py
Output: ../result_plots/correction_benchmark/delaunay_cap_artifact.{pdf,png}
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay

sys.path.insert(0, os.path.dirname(__file__))
from sphere import _spherical_delaunay_edges, crop_cap, sample_uniform_on_unit_sphere  # noqa: E402

OUT = "../result_plots/correction_benchmark/delaunay_cap_artifact"
CAP_RADIUS = 2.0  # as in the benchmark
SEED = 0


def degree(edges, n):
    d = np.zeros(n, int)
    for e in edges:
        for u in e:
            d[u] += 1
    return d


def lid_edges(coords, cap_radius, rng):
    """Crop hull edges vs. the full-sphere Delaunay restricted to the crop (both in crop indices)."""
    crop, dist = crop_cap(coords, cap_radius, rng=rng)
    to_local = {g: i for i, g in enumerate(crop)}
    local = _spherical_delaunay_edges(coords[crop])
    full_local = {frozenset(to_local[u] for u in e) for e in _spherical_delaunay_edges(coords)
                  if all(u in to_local for u in e)}
    return crop, dist.to_numpy(), local, full_local, local - full_local


def stereographic_delaunay_edges(crop_xyz, center):
    """Planar Delaunay of the crop after stereographic projection from the cap's antipode.

    Projection preserves circles, so this is the spherical Delaunay of the crop points without
    the facets that triangulate the empty hole -- the analogue of recomputing Delaunay on a planar crop.
    """
    p = rotate_to_pole(crop_xyz, center)
    xy = p[:, :2] / (1 + p[:, 2:3])
    return {frozenset((s[i], s[j])) for s in Delaunay(xy).simplices for i, j in ((0, 1), (1, 2), (2, 0))}


def rotate_to_pole(coords, center):
    """Rotation that sends the cap centre to the north pole (0, 0, 1)."""
    z = center / np.linalg.norm(center)
    a = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
    x = np.cross(a, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return coords @ np.stack([x, y, z]).T


def main():
    fig = plt.figure(figsize=(12, 7.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    # --- top row: picture with few points so the lid is visible --------------------------
    for col, (n, r) in enumerate([(600, CAP_RADIUS), (600, 1.0)]):
        rng = np.random.default_rng(SEED)
        coords = sample_uniform_on_unit_sphere(n, rng=rng)
        crop, dist, local, full_local, extra = lid_edges(coords, r, np.random.default_rng(SEED + 1))
        # recover the cap centre: the crop point with the largest distance to the border
        c_xyz = coords[crop]
        center = c_xyz[np.argmax(dist)]
        p = rotate_to_pole(c_xyz, center)
        ax = fig.add_subplot(gs[0, col], projection="3d")
        ok = [tuple(e) for e in local - extra]
        bad = [tuple(e) for e in extra]
        for es, colr, lw, lab in [(ok, "0.6", 0.4, "edges also in full-sphere Delaunay"),
                                  (bad, "#d62728", 0.9, f"lid edges, not in full graph (n={len(bad)})")]:
            for i, (u, v) in enumerate(es):
                ax.plot(*p[[u, v]].T, color=colr, lw=lw, label=lab if i == 0 else None)
        ax.scatter(*p.T, s=2, c="k")
        ax.view_init(elev=-55, azim=30)  # look up into the opening
        ax.set_box_aspect((1, 1, 1))
        ax.set_axis_off()
        ax.set_title(f"cap radius {r} rad, n={n} on sphere\n(viewed from below the cut)", fontsize=9)
        if col == 0:
            ax.legend(loc="upper left", fontsize=7, frameon=False)

    # --- top right: lid edge lengths vs. full graph (benchmark size) --------------------
    rng = np.random.default_rng(SEED)
    coords = sample_uniform_on_unit_sphere(5000, rng=rng)
    crop, dist, local, full_local, extra = lid_edges(coords, CAP_RADIUS, np.random.default_rng(SEED + 1))
    c_xyz = coords[crop]
    length = lambda es: np.array([np.linalg.norm(c_xyz[u] - c_xyz[v]) for u, v in map(tuple, es)])
    ax = fig.add_subplot(gs[0, 2])
    bins = np.linspace(0, 2, 60)
    ax.hist(length(local - extra), bins=bins, color="0.6", label="kept edges", log=True)
    ax.hist(length(extra), bins=bins, color="#d62728", label="lid edges", log=True)
    ax.set_xlabel("chord length")
    ax.set_ylabel("edges")
    ax.set_title(f"benchmark size: n=5000, cap {CAP_RADIUS} rad\n{len(extra)} lid edges of {len(local)}", fontsize=9)
    ax.legend(fontsize=7, frameon=False)

    # --- bottom row: profiles vs. distance to the cut ---------------------------------
    n_c = len(crop)
    deg_crop = degree(local, n_c)
    deg_true_trunc = degree(full_local, n_c)  # what a real truncation (full graph restricted to crop) gives
    stereo = stereographic_delaunay_edges(c_xyz, c_xyz[np.argmax(dist)])
    deg_stereo = degree(stereo, n_c)
    deg_full = degree(_spherical_delaunay_edges(coords), len(coords))[crop]
    bins = np.linspace(0, dist.max(), 25)
    mid = 0.5 * (bins[1:] + bins[:-1])
    idx = np.digitize(dist, bins) - 1
    prof = lambda x: pd.Series(x).groupby(idx).mean().reindex(range(len(mid))).to_numpy()

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(mid, prof(deg_full), "k-", label="full sphere (original)")
    ax.plot(mid, prof(deg_crop), color="#1f77b4", label="crop hull (benchmark)")
    ax.plot(mid, prof(deg_true_trunc), color="#2ca02c", ls="--", label="full graph restricted to crop")
    ax.plot(mid, prof(deg_stereo), color="#ff7f0e", ls=":", label="recomputed on crop, stereographic")
    ax.set_xlabel("distance to cap border (rad)")
    ax.set_ylabel("mean degree")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("degree", fontsize=9)

    import networkx as nx
    def closeness(edges, n):
        g = nx.Graph(); g.add_nodes_from(range(n)); g.add_edges_from(tuple(e) for e in edges)
        c = nx.closeness_centrality(g)
        return np.array([c[i] for i in range(n)])
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(mid, prof(closeness(local, n_c)), color="#1f77b4", label="crop hull (benchmark)")
    ax.plot(mid, prof(closeness(full_local, n_c)), color="#2ca02c", ls="--", label="full graph restricted to crop")
    ax.plot(mid, prof(closeness(stereo, n_c)), color="#ff7f0e", ls=":", label="recomputed on crop, stereographic")
    ax.set_xlabel("distance to cap border (rad)")
    ax.set_ylabel("mean closeness (on crop)")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("closeness", fontsize=9)

    ax = fig.add_subplot(gs[1, 2])
    lid_deg = degree(extra, n_c)
    ax.scatter(dist, lid_deg + np.random.default_rng(0).uniform(-0.15, 0.15, n_c), s=3, alpha=0.4, color="#d62728")
    ax.set_xlabel("distance to cap border (rad)")
    ax.set_ylabel("lid edges per node")
    ax.set_title("which nodes get lid edges", fontsize=9)

    fig.savefig(OUT + ".pdf")
    fig.savefig(OUT + ".png", dpi=150)
    print(f"crop n={n_c}, edges={len(local)}, lid edges={len(extra)}, "
          f"border-bin degree: full {prof(deg_full)[0]:.2f}, hull {prof(deg_crop)[0]:.2f}, "
          f"restricted {prof(deg_true_trunc)[0]:.2f}, stereo {prof(deg_stereo)[0]:.2f}; max lid chord {length(extra).max():.2f}")
    new, lost = stereo - full_local, full_local - stereo
    print(f"stereo vs restricted: {len(stereo)} edges, {len(new)} new (max chord {length(new).max():.2f}, "
          f"nodes touched max dist {max(dist[list(e)].max() for e in new):.3f}), {len(lost)} lost; "
          f"stereo vs hull: {len(local - stereo)} hull edges dropped, {len(stereo - local)} added")


if __name__ == "__main__":
    main()
