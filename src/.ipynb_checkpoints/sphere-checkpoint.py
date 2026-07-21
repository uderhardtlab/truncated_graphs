"""
sphere.py — Sphere benchmark using BosporusFlow

Evaluates border-effect correction methods (piecewise-linear / exp-saturation
fits and SERN) on synthetic point clouds on the unit sphere.  All graph
construction, centrality computation, distance calculation, fitting and
evaluation now delegate to the bosporus package via BosporusFlow.
"""

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.stats import pearsonr, vonmises_fisher

from time import time
from tqdm import trange

from bosporus import Flow
from bosporus.centrality_measures import compute_centrality_measures

from sern import surrogate_ensemble_gt

import os
os.environ["OMP_NUM_THREADS"] = "8"

NUMBER_OF_SERNS = 100
N_JOBS = 128
N_OF_RUNS = 100

OUTPUT_DIR = "../results/figure4"


def sample_uniform_on_unit_sphere(n, rng=None):
    """Uniform sampling on S² via cylindrical projection."""
    if rng is None:
        rng = np.random.default_rng()
    u = rng.uniform(-1, 1, n)
    theta = rng.uniform(0, 2 * np.pi, n)
    r = np.sqrt(1 - u ** 2)
    return np.column_stack((r * np.cos(theta), r * np.sin(theta), u))


def sample_von_mises_fisher(n, kappas, rng=None):
    """Clustered sampling on S² via von Mises–Fisher mixture."""
    if rng is None:
        rng = np.random.default_rng()
    if len(kappas) == 1:
        mu = sample_uniform_on_unit_sphere(1, rng=rng)[0]
        return vonmises_fisher(mu=mu, kappa=kappas[0]).rvs(size=n, random_state=rng)
    per_cluster = n // len(kappas)
    return np.vstack([
        sample_von_mises_fisher(per_cluster, [k], rng=rng) for k in kappas
    ])


def _spherical_delaunay_edges(coords):
    """Geodesic Delaunay triangulation via convex hull on unit-sphere coords."""
    hull = ConvexHull(coords)
    edges = set()
    for simplex in hull.simplices:
        edges.add(frozenset((simplex[0], simplex[1])))
        edges.add(frozenset((simplex[1], simplex[2])))
        edges.add(frozenset((simplex[2], simplex[0])))
    return edges


def get_edge_list(coords, edge_type, k=None, r=None):
    if edge_type == "delaunay":
        return _spherical_delaunay_edges(coords)
    # For knn / rnn we can reuse bosporus graph construction directly,
    # because in 3-D Euclidean space the ordering of Euclidean distances
    # equals the ordering of geodesic distances on the unit sphere.
    flow = BosporusFlow(coords)
    if edge_type == "knn":
        flow.construct_graph("knn", {"k": k})
    elif edge_type == "rnn":
        flow.construct_graph("rnn", {"r": r})
    else:
        raise ValueError(f"Unknown edge type: {edge_type}")
    return flow.edge_list


def crop_cap(coords, cap_radius):
    center = coords[np.random.choice(len(coords))]
    dot = np.clip(coords @ center, -1.0, 1.0)
    geo_dist_to_center = np.arccos(dot)
    inside = np.where(geo_dist_to_center <= cap_radius)[0]
    dist_to_border = cap_radius - geo_dist_to_center[inside]
    return inside, pd.Series(dist_to_border, index=inside, name="distance_to_cap")


def get_bosporus_corrections(crop_coords, edges, measures, distances):
    bf = BosporusFlow(crop_coords)
    bf.edge_list = edges
    bf.compute_centralities(measures=measures)
    bf.df = pd.concat([bf.df, distances.reset_index(drop=True)], axis=1)
    bf.fit_models(measures=measures, distance_key="distance_to_cap")
    bf.df["degree"] = bf.df["degree"].astype(int)
    return bf.df


def get_sern_median(crop_coords, local_edges):
    n_bins = int(np.sqrt(len(local_edges)))
    sern_median = surrogate_ensemble_gt(
        coords=crop_coords,
        edge_list=local_edges,
        n_bins=n_bins,
        n_surrogates=NUMBER_OF_SERNS,
        n_jobs=N_JOBS,
    )
    sern_median = pd.DataFrame(sern_median)
    sern_median["degree"] = sern_median["degree"].astype(int)
    return sern_median


def process_coords(coords, edge_type, cap_radii, k=None, r=None):
    N = len(coords)

    # --- global graph & centralities ---
    global_edges = get_edge_list(coords, edge_type, k=k, r=r)
    global_centralities = pd.DataFrame(
        compute_centrality_measures(global_edges, N)
    )
    measures = global_centralities.columns
    all_correlations = []

    for cap_radius in cap_radii:
        crop, distances = crop_cap(coords, cap_radius)
        crop_coords = coords[crop]


        local_edges = get_edge_list(crop_coords, edge_type, k=k, r=r)        
        BOSPORUS_results = get_bosporus_corrections(crop_coords, local_edges, measures, distances)
        sern_median = get_sern_median(crop_coords, local_edges)

        # --- assemble result frame ---
        results = pd.concat(
            {
                "original": global_centralities.loc[crop].sort_index(axis=0).sort_index(axis=1).reset_index(drop=True),
                "crop": BOSPORUS_results[measures].sort_index(axis=0).sort_index(axis=1),
                "distance": BOSPORUS_results["distance_to_cap"],
                "BOSPORUS_corrections": BOSPORUS_results.filter(like="BOSPORUS", axis=1).sort_index(axis=0).sort_index(axis=1),
                "sern": sern_median.sort_index(axis=0).sort_index(axis=1),
                "sern_corrected": (
                    BOSPORUS_results[measures].sort_index(axis=0).sort_index(axis=1)
                    - sern_median.sort_index(axis=0).sort_index(axis=1)
                ),
            },
            axis=1,
        )

        # --- correlations ---
        corrs_original_crop, corrs_original_corrected = [], []
        corrs_original_sern, corrs_crop_sern = [], []

        for m in measures:
            corrs_original_crop.append(
                pearsonr(results["original"][m], results["crop"][m]).statistic
            )
            corrs_original_corrected.append(
                pearsonr(results["original"][m], results["BOSPORUS_corrections"][f"BOSPORUS corrected {m}"]).statistic
            )
            corrs_original_sern.append(
                pearsonr(results["original"][m], results["sern_corrected"][m]).statistic
            )
            corrs_crop_sern.append(
                pearsonr(results["crop"][m], results["sern"][m]).statistic
            )

        correlations = pd.DataFrame(index=measures)
        correlations["original vs. on crop"] = corrs_original_crop
        correlations["original vs. BOSPORUS corrected on crop"] = corrs_original_corrected
        correlations["original vs. SERN corrected on crop"] = corrs_original_sern
        correlations["on crop vs. SERN values"] = corrs_crop_sern
        correlations["cap_radius"] = cap_radius
        all_correlations.append(correlations)
    return pd.concat(all_correlations)


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    n = 5000
    for _ in trange(N_OF_RUNS):
        all_correlations = []

        coord_configs = [
            ("uniform", sample_uniform_on_unit_sphere(n=n)),
            ("kappa=1", sample_von_mises_fisher(n=n, kappas=[1])),
            ("kappa=1,3,5", sample_von_mises_fisher(n=n, kappas=[1, 3, 5])),
        ]

        for coord_type, coords in coord_configs:
            for edge_type in ["delaunay", "knn", "rnn"]:
                base_kwargs = dict(
                    coords=coords,
                    edge_type=edge_type,
                    cap_radii=[1, 2],
                )

                if edge_type == "delaunay":
                    corr = process_coords(**base_kwargs)
                    corr["graph_type"] = edge_type
                    corr["coord_type"] = coord_type
                    corr["n"] = n
                    all_correlations.append(corr)

                elif edge_type == "knn":
                    for k in [5, 10, 15]:
                        corr = process_coords(**base_kwargs, k=k)
                        corr["k"] = k
                        corr["graph_type"] = edge_type
                        corr["coord_type"] = coord_type
                        corr["n"] = n
                        all_correlations.append(corr)

                elif edge_type == "rnn":
                    for r in [0.05, 0.1, 0.15]:
                        corr = process_coords(**base_kwargs, r=r)
                        corr["radius"] = r
                        corr["graph_type"] = edge_type
                        corr["coord_type"] = coord_type
                        corr["n"] = n
                        all_correlations.append(corr)

        timestamp = time()
        out_path = os.path.join(
            OUTPUT_DIR, f"correlations_{timestamp}.csv"
        )
        pd.concat(all_correlations).to_csv(out_path)
