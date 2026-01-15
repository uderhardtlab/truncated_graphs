import numpy as np
import pandas as pd
import networkx as nx

from itertools import combinations
from scipy.spatial import Delaunay
from sklearn.neighbors import NearestNeighbors


def sample_points_on_square(n, xlim=1.0, ylim=1.0):
    return np.random.uniform(
        low=(-xlim, -ylim),
        high=(xlim, ylim),
        size=(n, 2),
    )


def spatial_subset(coords, xlim, ylim):
    mask = (
        (np.abs(coords[:, 0]) < xlim) &
        (np.abs(coords[:, 1]) < ylim)
    )
    return np.where(mask)[0]


def knn_edges(coords, k):
    """Directed, asymmetric kNN on full set."""
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, indices = nbrs.kneighbors(coords)

    edges = set()
    for u, neighbors in enumerate(indices):
        for v in neighbors[1:]:
            edges.add((u, v))
    return edges


def delaunay_edges(coords):
    """Undirected Delaunay on full set."""
    tri = Delaunay(coords)
    edges = set()

    for simplex in tri.simplices:
        for u, v in combinations(simplex, 2):
            edges.add(frozenset((u, v)))

    return edges


def knn_edges_on_subset(coords, subset, k):
    """Directed, asymmetric kNN recomputed on truncated domain."""
    subcoords = coords[subset]

    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(subcoords)
    _, indices = nbrs.kneighbors(subcoords)

    edges = set()
    for i, neighbors in enumerate(indices):
        u = subset[i]
        for j in neighbors[1:]:
            v = subset[j]
            edges.add((u, v))
    return edges


def delaunay_edges_on_subset(coords, subset):
    """Undirected Delaunay recomputed on truncated domain."""
    subcoords = coords[subset]
    tri = Delaunay(subcoords)

    edges = set()
    for simplex in tri.simplices:
        for u, v in combinations(simplex, 2):
            edges.add(frozenset((subset[u], subset[v])))

    return edges


# ============================================================
# Graph construction
# ============================================================

def build_graph(edges, directed):
    G = nx.DiGraph() if directed else nx.Graph()
    if directed:
        G.add_edges_from(edges)
    else:
        G.add_edges_from(map(tuple, edges))
    return G


def build_plot_graph(full_edges, subset, directed):
    subset = set(subset)
    plot_edges = set()

    if directed:
        for u, v in full_edges:
            if u in subset or v in subset:
                plot_edges.add((u, v))
    else:
        for e in full_edges:
            if e & subset:
                plot_edges.add(e)

    return build_graph(plot_edges, directed)


# ============================================================
# Statistics
# ============================================================

def edge_length_statistics(coords, trunc_edges, full_edges, directed):
    full_set = set(full_edges)

    lengths = []
    is_new = []
    symmetry = []

    for e in trunc_edges:
        if directed:
            u, v = e
            lengths.append(np.linalg.norm(coords[u] - coords[v]))
            is_new.append((u, v) not in full_set)
            symmetry.append((v, u) in trunc_edges)
        else:
            u, v = tuple(e)
            lengths.append(np.linalg.norm(coords[u] - coords[v]))
            is_new.append(e not in full_set)

    return pd.DataFrame({
        "Edge length": lengths,
        "New edge": is_new,
        "Symmetry": symmetry if directed else None,
    })


# ============================================================
# Main API
# ============================================================

def trunc_graphs(
    coords,
    method,
    xlim,
    ylim,
    k=None,
    return_graphs=False,
):
    assert method in {"kNN", "delaunay"}

    directed = method == "kNN"

    # ---- FULL GRAPH ----
    if method == "kNN":
        full_edges = knn_edges(coords, k)
    else:
        full_edges = delaunay_edges(coords)

    # ---- TRUNCATED DOMAIN ----
    subset = spatial_subset(coords, xlim, ylim)

    if method == "kNN":
        trunc_edges = knn_edges_on_subset(coords, subset, k)
    else:
        trunc_edges = delaunay_edges_on_subset(coords, subset)

    # ---- STATISTICS ----
    edge_df = edge_length_statistics(
        coords,
        trunc_edges,
        full_edges,
        directed,
    )

    if not return_graphs:
        return {"edge_df": edge_df}

    # ---- GRAPHS ----
    G = build_graph(full_edges, directed)
    truncG = build_graph(trunc_edges, directed)
    plotG = build_plot_graph(full_edges, subset, directed)
    subG = G.subgraph(subset).copy()

    vmax = max(dict(G.degree).values())

    return {
        "coords": coords,
        "subset": subset,
        "G": G,
        "subG": subG,
        "plotG": plotG,
        "truncG": truncG,
        "edge_df": edge_df,
        "vmax": vmax,
    }
