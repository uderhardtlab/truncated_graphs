"""
Edge truncation: the graph built on a cropped region vs. the global graph restricted to that region.

Building a proximity graph on a crop V' gives G(V'), which differs from the global graph restricted to the crop,
G(V)|V': crop nodes near the border gain edges to other crop nodes that the global graph would have spent on
nodes outside the crop (kNN, Delaunay), or lose them (r-NN). `trunc_graphs` builds both and tabulates every crop
edge with its length, whether it is new (absent from the global graph) and, for kNN, whether it is reciprocated.

Edge representations follow bosperrus: kNN edges are directed (u, v) tuples, Delaunay and r-NN edges are
undirected frozensets. Used by edge_effects.py and figure_plotting/computational_effects/edge_effects.ipynb.
"""
import glob
import os
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd
import squidpy as sq
import tifffile as tf
from bosperrus import delaunay_edges, knn_edges, rnn_edges
from scipy.ndimage import center_of_mass
from scipy.spatial import Delaunay
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

# Original location of the MIBI-TOF segmentation TIFFs (not part of this checkout; their centroids are cached in
# mibitof_coords/coords.pickle, which is what the scripts read).
MIBITOF_TIFF_DIR = "/data/bionets/datasets/graph_truncation/"
MIBITOF_DATASETS = ("glioma_mibitof", "tma_controls_mibitof", "TNBC_mibitof", "tuberculosis_mibitof")
SQUIDPY_VISIUM_SAMPLES = (
    "V1_Breast_Cancer_Block_A_Section_1", "V1_Breast_Cancer_Block_A_Section_2", "V1_Human_Heart",
    "V1_Human_Lymph_Node", "V1_Mouse_Kidney", "V1_Adult_Mouse_Brain", "V1_Mouse_Brain_Sagittal_Posterior",
    "V1_Mouse_Brain_Sagittal_Posterior_Section_2", "V1_Mouse_Brain_Sagittal_Anterior",
    "V1_Mouse_Brain_Sagittal_Anterior_Section_2", "V1_Human_Brain_Section_1", "V1_Human_Brain_Section_2",
    "V1_Adult_Mouse_Brain_Coronal_Section_1", "V1_Adult_Mouse_Brain_Coronal_Section_2",
    "Targeted_Visium_Human_Cerebellum_Neuroscience", "Parent_Visium_Human_Cerebellum",
    "Targeted_Visium_Human_SpinalCord_Neuroscience", "Parent_Visium_Human_SpinalCord",
    "Targeted_Visium_Human_Glioblastoma_Pan_Cancer", "Parent_Visium_Human_Glioblastoma",
    "Targeted_Visium_Human_BreastCancer_Immunology", "Parent_Visium_Human_BreastCancer",
    "Targeted_Visium_Human_OvarianCancer_Pan_Cancer", "Targeted_Visium_Human_OvarianCancer_Immunology",
    "Parent_Visium_Human_OvarianCancer", "Targeted_Visium_Human_ColorectalCancer_GeneSignature",
    "Parent_Visium_Human_ColorectalCancer", "Visium_FFPE_Mouse_Brain", "Visium_FFPE_Mouse_Brain_IF",
    "Visium_FFPE_Mouse_Kidney", "Visium_FFPE_Human_Breast_Cancer",
    "Visium_FFPE_Human_Prostate_Acinar_Cell_Carcinoma", "Visium_FFPE_Human_Prostate_Cancer",
    "Visium_FFPE_Human_Prostate_IF", "Visium_FFPE_Human_Normal_Prostate",
)


# ── datasets ──────────────────────────────────────────────────────────────────
def get_squidpy_visium_datasets(samples=SQUIDPY_VISIUM_SAMPLES):
    """Spot coordinates of squidpy's Visium example datasets, keyed "sq_visium:<sample>" (downloads them)."""
    return {f"sq_visium:{s}": sq.datasets.visium(s).obsm["spatial"].astype(np.int16) for s in samples}


def get_mibitof(data_path=MIBITOF_TIFF_DIR, datasets=MIBITOF_DATASETS):
    """Cell centroids from MIBI-TOF segmentation TIFFs (label image per FOV), keyed "<dataset>:<file name>"."""
    coords = {}
    for dataset in datasets:
        dataset_path = os.path.join(data_path, dataset)
        for p in tqdm(glob.glob(f"{dataset_path}/*.tif") + glob.glob(f"{dataset_path}/*.tiff")):
            segm = tf.imread(p)
            if segm.ndim == 3:
                assert segm.shape[0] == 1, f"Found unexpected 3D data: {segm.shape}"
                segm = segm[0]
            label_ids = np.unique(segm)
            label_ids = label_ids[label_ids != 0]  # 0 is background
            centroids = center_of_mass(np.ones_like(segm), segm, label_ids)
            coords[f"{dataset}:{os.path.basename(p)}"] = np.array(centroids).astype(np.int16)
    return coords


# ── graphs on the crop ────────────────────────────────────────────────────────
def spatial_subset(coords, xlim, ylim):
    """Indices of the points strictly inside the rectangle xlim x ylim."""
    mask = (coords[:, 0] > xlim[0]) & (coords[:, 0] < xlim[1]) & (coords[:, 1] > ylim[0]) & (coords[:, 1] < ylim[1])
    return np.where(mask)[0]


def knn_edges_on_subset(coords, subset, k):
    """Directed, asymmetric kNN graph recomputed on the crop, in global node indices."""
    _, indices = NearestNeighbors(n_neighbors=k + 1).fit(coords[subset]).kneighbors(coords[subset])
    return {(subset[i], subset[j]) for i, neighbors in enumerate(indices) for j in neighbors[1:]}


def rnn_edges_on_subset(coords, subset, r):
    """Undirected radius-neighbour graph recomputed on the crop, in global node indices."""
    _, indices = NearestNeighbors(radius=r).fit(coords[subset]).radius_neighbors(coords[subset], radius=r)
    return {frozenset((subset[i], subset[j])) for i, neighbors in enumerate(indices) for j in neighbors if i != j}


def delaunay_edges_on_subset(coords, subset):
    """Undirected Delaunay graph recomputed on the crop, in global node indices."""
    tri = Delaunay(coords[subset])
    return {frozenset((subset[u], subset[v])) for simplex in tri.simplices for u, v in combinations(simplex, 2)}


def build_graph(edges, directed):
    """networkx graph from an edge set (tuples if directed, frozensets otherwise)."""
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_edges_from(edges if directed else map(tuple, edges))
    return G


def build_plot_graph(full_edges, subset, directed):
    """The global graph's edges with at least one end in the crop (for drawing the crop's surroundings)."""
    subset = set(subset)
    if directed:
        return build_graph({(u, v) for u, v in full_edges if u in subset or v in subset}, directed)
    return build_graph({e for e in full_edges if e & subset}, directed)


# ── statistics ────────────────────────────────────────────────────────────────
def edge_length_statistics(coords, trunc_edges, full_edges, directed):
    """One row per crop edge: length, whether it is absent from the global graph, and (kNN) whether reciprocated."""
    full_set = set(full_edges)
    lengths, is_new, symmetry = [], [], []
    for e in trunc_edges:
        u, v = e if directed else tuple(e)
        lengths.append(np.linalg.norm(coords[u] - coords[v]))
        is_new.append(e not in full_set)
        if directed:
            symmetry.append((v, u) in trunc_edges)
    return pd.DataFrame({"Edge length": lengths, "New edge": is_new, "Symmetry": symmetry if directed else None})


def trunc_graphs(coords, method, xlim, ylim, k=None, radius=None, return_graphs=False):
    """Global graph vs. graph rebuilt on the crop xlim x ylim, for method "kNN" (k), "r" (radius) or "delaunay".

    Returns {"edge_df", "#nodes in ROI"}; with return_graphs, also the coordinates, the crop indices and the
    graphs G (global), subG (global restricted to the crop), plotG (global edges touching the crop) and truncG
    (rebuilt on the crop).
    """
    assert method in {"kNN", "delaunay", "r"}
    directed = method == "kNN"
    subset = spatial_subset(coords, xlim, ylim)
    if method == "kNN":
        full_edges, trunc_edges = knn_edges(coords, k), knn_edges_on_subset(coords, subset, k)
    elif method == "r":
        full_edges, trunc_edges = rnn_edges(coords, radius), rnn_edges_on_subset(coords, subset, radius)
    else:
        full_edges, trunc_edges = delaunay_edges(coords), delaunay_edges_on_subset(coords, subset)

    edge_df = edge_length_statistics(coords, trunc_edges, full_edges, directed)
    if not return_graphs:
        return {"edge_df": edge_df, "#nodes in ROI": len(subset)}

    G = build_graph(full_edges, directed)
    return {
        "coords": coords,
        "subset": subset,
        "G": G,
        "subG": G.subgraph(subset).copy(),
        "plotG": build_plot_graph(full_edges, subset, directed),
        "truncG": build_graph(trunc_edges, directed),
        "edge_df": edge_df,
    }
