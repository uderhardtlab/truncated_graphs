import glob
import numpy as np
import os
import pandas as pd
import networkx as nx

from itertools import combinations
from scipy.spatial import Delaunay
from scipy.ndimage import center_of_mass

from sklearn.neighbors import NearestNeighbors
import squidpy as sq
import tifffile as tf
from tqdm import tqdm

def get_squidpy_visium_datasets():
    all_samples = ['V1_Breast_Cancer_Block_A_Section_1', 'V1_Breast_Cancer_Block_A_Section_2', 'V1_Human_Heart', 'V1_Human_Lymph_Node', 'V1_Mouse_Kidney', 'V1_Adult_Mouse_Brain', 'V1_Mouse_Brain_Sagittal_Posterior', 'V1_Mouse_Brain_Sagittal_Posterior_Section_2', 'V1_Mouse_Brain_Sagittal_Anterior', 'V1_Mouse_Brain_Sagittal_Anterior_Section_2', 'V1_Human_Brain_Section_1', 'V1_Human_Brain_Section_2', 'V1_Adult_Mouse_Brain_Coronal_Section_1', 'V1_Adult_Mouse_Brain_Coronal_Section_2', 'Targeted_Visium_Human_Cerebellum_Neuroscience', 'Parent_Visium_Human_Cerebellum', 'Targeted_Visium_Human_SpinalCord_Neuroscience', 'Parent_Visium_Human_SpinalCord', 'Targeted_Visium_Human_Glioblastoma_Pan_Cancer', 'Parent_Visium_Human_Glioblastoma', 'Targeted_Visium_Human_BreastCancer_Immunology', 'Parent_Visium_Human_BreastCancer', 'Targeted_Visium_Human_OvarianCancer_Pan_Cancer', 'Targeted_Visium_Human_OvarianCancer_Immunology', 'Parent_Visium_Human_OvarianCancer', 'Targeted_Visium_Human_ColorectalCancer_GeneSignature', 'Parent_Visium_Human_ColorectalCancer', 'Visium_FFPE_Mouse_Brain', 'Visium_FFPE_Mouse_Brain_IF', 'Visium_FFPE_Mouse_Kidney', 'Visium_FFPE_Human_Breast_Cancer', 'Visium_FFPE_Human_Prostate_Acinar_Cell_Carcinoma', 'Visium_FFPE_Human_Prostate_Cancer', 'Visium_FFPE_Human_Prostate_IF', 'Visium_FFPE_Human_Normal_Prostate']
    coords = dict()
    for s in all_samples:
        coords[f"sq_visium:{s}"] = sq.datasets.visium(s).obsm["spatial"].astype(np.int16)
    return coords


def get_mibitof(data_path="/data/bionets/datasets/graph_truncation/", datasets=["glioma_mibitof", "tma_controls_mibitof", "TNBC_mibitof", "tuberculosis_mibitof"]):
    coords = dict()
    for dataset in datasets:
        dataset_path = os.path.join(data_path, dataset)
        for p in tqdm(glob.glob(f"{dataset_path}/*.tif") + glob.glob(f"{dataset_path}/*.tiff")):
            segm = tf.imread(p)
            if len(segm.shape) == 3:
                assert segm.shape[0] == 1, f"Found unexpected 3D data: {segm.shape}"
                segm = segm[0]
            label_ids = np.unique(segm)
            label_ids = label_ids[label_ids != 0]  # remove background
            
            centroids = center_of_mass(
                np.ones_like(segm),
                segm,
                label_ids
            )
            name = f"{dataset}:{os.path.basename(p)}"
            coords[name] = np.array(centroids).astype(np.int16)
    return coords

def sample_points_on_square(n, xlim=1.0, ylim=1.0):
    return np.random.uniform(
        low=(-xlim, -ylim),
        high=(xlim, ylim),
        size=(n, 2),
    )


def spatial_subset(coords, xlim, ylim):
    mask = (
        (coords[:, 0] > xlim[0]) &
        (coords[:, 0] < xlim[1]) &
        (coords[:, 1] > ylim[0]) &
        (coords[:, 1] < ylim[1])
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


def rnn_edges(coords, r):
    """Directed, asymmetric kNN on full set."""
    nbrs = NearestNeighbors(radius=r).fit(coords)
    _, indices = nbrs.radius_neighbors(coords, radius=r)

    edges = set()
    for u, neighbors in enumerate(indices):
        for v in neighbors:
            if u != v:
                edges.add(frozenset((u, v)))
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


def rnn_edges_on_subset(coords, subset, r):
    """Directed, asymmetric kNN recomputed on truncated domain."""
    subcoords = coords[subset]

    nbrs = NearestNeighbors(radius=r).fit(subcoords)
    _, indices = nbrs.radius_neighbors(subcoords, radius=r)

    edges = set()
    for i, neighbors in enumerate(indices):
        u = subset[i]
        for j in neighbors:
            if i != j:
                v = subset[j]
                edges.add(frozenset((u, v)))
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
            try:
                u, v = tuple(e)
            except ValueError as err:
                print(err)
                print(e)
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
    radius=None,
    return_graphs=False,
):
    assert method in {"kNN", "delaunay", "r"}

    directed = method == "kNN"

    # ---- FULL GRAPH ----
    if method == "kNN":
        full_edges = knn_edges(coords, k)
    elif method == "r":
        full_edges = rnn_edges(coords, radius)
    else:
        full_edges = delaunay_edges(coords)

    # ---- TRUNCATED DOMAIN ----
    subset = spatial_subset(coords, xlim, ylim)

    if method == "kNN":
        trunc_edges = knn_edges_on_subset(coords, subset, k)
    elif method == "r":
        trunc_edges = rnn_edges_on_subset(coords, subset, radius)
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
        return {"edge_df": edge_df,
               "#nodes in ROI": len(subset)}

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
