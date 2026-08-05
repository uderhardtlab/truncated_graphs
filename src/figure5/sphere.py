"""
sphere.py — Sphere benchmark using the bosperrus Flow class

Evaluates border-effect correction methods (piecewise-linear / exp-saturation
fits and SERN) on synthetic point clouds on the unit sphere.  All graph
construction, centrality computation, distance calculation, fitting and
evaluation now delegate to the bosperrus package via the Flow class.
"""

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.stats import pearsonr, spearmanr, vonmises_fisher

from time import time
from tqdm import trange

from bosperrus import Flow, construct_graph
from bosperrus.centrality_measures import compute_centrality_measures

from sern import surrogate_ensemble_gt

import os
os.environ["OMP_NUM_THREADS"] = "8"

NUMBER_OF_SERNS = 100
N_JOBS = -1  # joblib: use all available cores on whatever machine this runs on
N_OF_RUNS = 100

# same 5 measures as figure3's compute_fits.py (excludes "harmonic")
MEASURES = ["degree", "pagerank", "betweenness", "closeness", "clustering"]
CORRELATION_METHODS = {"pearson": pearsonr, "spearman": spearmanr}

# "affected" = nearest quartile of crop nodes to the border, by distance_to_cap
# -- a purely geometric criterion (independent of any fitted correction model
# and of the raw/corrected values themselves) meant to isolate where border
# truncation actually biases a score, rather than diluting the correlation
# with the majority of untouched interior nodes.
AFFECTED_QUANTILE = 0.25

OUTPUT_DIR = "../results/figure5"


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
    # For knn / rnn we can reuse bosperrus graph construction directly,
    # because in 3-D Euclidean space the ordering of Euclidean distances
    # equals the ordering of geodesic distances on the unit sphere.
    if edge_type == "knn":
        return construct_graph(coords, "knn", k=k)
    elif edge_type == "rnn":
        return construct_graph(coords, "rnn", r=r)
    else:
        raise ValueError(f"Unknown edge type: {edge_type}")


def crop_cap(coords, cap_radius):
    center = coords[np.random.choice(len(coords))]
    dot = np.clip(coords @ center, -1.0, 1.0)
    geo_dist_to_center = np.arccos(dot)
    inside = np.where(geo_dist_to_center <= cap_radius)[0]
    dist_to_border = cap_radius - geo_dist_to_center[inside]
    return inside, pd.Series(dist_to_border, index=inside, name="distance_to_cap")


def get_bosperrus_corrections(crop_coords, edges, measures, distances):
    scores = compute_centrality_measures(edges, N=len(crop_coords), measures=list(measures))
    bf = Flow.from_distances_and_scores(
        distances=distances.reset_index(drop=True),
        scores=scores,
    )
    # A measure that succeeded on the global graph can still fail on this
    # particular local crop (e.g. pagerank on a disconnected/edgeless
    # sub-sample) -- compute_centrality_measures already warns and just omits
    # it rather than raising, so only fit whatever actually came back instead
    # of the full requested `measures`.
    bf.flow(score_names=list(scores.columns))
    bf.observations["degree"] = bf.observations["degree"].astype(int)
    return bf.observations


def get_sern_median(crop_coords, local_edges, measures):
    n_bins = int(np.sqrt(len(local_edges)))
    sern_median = surrogate_ensemble_gt(
        coords=crop_coords,
        edge_list=local_edges,
        n_bins=n_bins,
        measures=list(measures),
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
        compute_centrality_measures(global_edges, N, MEASURES)
    )
    measures = global_centralities.columns
    all_correlations = []

    for cap_radius in cap_radii:
        crop, distances = crop_cap(coords, cap_radius)
        crop_coords = coords[crop]


        local_edges = get_edge_list(crop_coords, edge_type, k=k, r=r)
        BOSPERRUS_results = get_bosperrus_corrections(crop_coords, local_edges, measures, distances)
        sern_median = get_sern_median(crop_coords, local_edges, measures)

        # A measure that succeeded on the global graph can still fail on this
        # particular local crop (e.g. pagerank on a disconnected/edgeless
        # sub-sample, or a SERN surrogate draw that happened to omit it) --
        # only carry forward whichever measures are actually present on both
        # sides for this cap_radius, rather than assuming all of `measures`.
        cap_measures = [m for m in measures if m in BOSPERRUS_results.columns and m in sern_median.columns]

        # --- assemble result frame ---
        results = pd.concat(
            {
                "original": global_centralities.loc[crop, cap_measures].sort_index(axis=0).sort_index(axis=1).reset_index(drop=True),
                "crop": BOSPERRUS_results[cap_measures].sort_index(axis=0).sort_index(axis=1),
                "distance": BOSPERRUS_results["distance_to_cap"],
                "BOSPERRUS_corrections": BOSPERRUS_results[[f"BOSPERRUS corrected {m}" for m in cap_measures]].sort_index(axis=0).sort_index(axis=1),
                "sern": sern_median[cap_measures].sort_index(axis=0).sort_index(axis=1),
                "sern_corrected": (
                    BOSPERRUS_results[cap_measures].sort_index(axis=0).sort_index(axis=1)
                    - sern_median[cap_measures].sort_index(axis=0).sort_index(axis=1)
                ),
            },
            axis=1,
        )

        # --- correlations ---
        # Both Pearson and Spearman: Rheinwalt et al. 2012 validate SERN
        # correction via Spearman's rank correlation, not Pearson, since
        # centrality distributions are typically skewed/tied rather than
        # linearly related; keeping Pearson alongside it for comparison.
        #
        # Also split by node_subset ("all" vs "affected"): correlating over
        # every crop node dilutes the comparison with untouched interior
        # nodes, especially now that only the bigger cap_radius is used, so
        # "affected" restricts to the nearest AFFECTED_QUANTILE of nodes to
        # the border -- a geometric criterion, independent of either
        # correction method's fit and of the raw/corrected values themselves,
        # so it can't bias the comparison toward whichever method it favors.
        dist = results["distance"]["distance_to_cap"]
        node_subsets = {
            "all": pd.Series(True, index=results.index),
            "affected": dist <= dist.quantile(AFFECTED_QUANTILE),
        }

        for corr_name, corr_fn in CORRELATION_METHODS.items():
            for subset_name, subset_mask in node_subsets.items():
                sub = results[subset_mask]
                corrs_original_crop, corrs_original_corrected = [], []
                corrs_original_sern, corrs_crop_sern = [], []

                for m in cap_measures:
                    corrs_original_crop.append(
                        corr_fn(sub["original"][m], sub["crop"][m]).statistic
                    )
                    corrs_original_corrected.append(
                        corr_fn(sub["original"][m], sub["BOSPERRUS_corrections"][f"BOSPERRUS corrected {m}"]).statistic
                    )
                    corrs_original_sern.append(
                        corr_fn(sub["original"][m], sub["sern_corrected"][m]).statistic
                    )
                    corrs_crop_sern.append(
                        corr_fn(sub["crop"][m], sub["sern"][m]).statistic
                    )

                correlations = pd.DataFrame(index=cap_measures)
                correlations["original vs. on crop"] = corrs_original_crop
                correlations["original vs. BOSPERRUS corrected on crop"] = corrs_original_corrected
                correlations["original vs. SERN corrected on crop"] = corrs_original_sern
                correlations["on crop vs. SERN values"] = corrs_crop_sern
                correlations["cap_radius"] = cap_radius
                correlations["corr_method"] = corr_name
                correlations["node_subset"] = subset_name
                all_correlations.append(correlations)
    return pd.concat(all_correlations)


def main(n_runs=N_OF_RUNS, n=5000, n_surrogates=NUMBER_OF_SERNS, output_dir=OUTPUT_DIR):
    global NUMBER_OF_SERNS
    NUMBER_OF_SERNS = n_surrogates
    os.makedirs(output_dir, exist_ok=True)

    for _ in trange(n_runs):
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
                    cap_radii=[2],  # bigger cap only -- see jobs/figure5_sphere.sbatch
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
            output_dir, f"correlations_{timestamp}.csv"
        )
        pd.concat(all_correlations).to_csv(out_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-runs", type=int, default=N_OF_RUNS,
                         help="Number of outer repetitions to run (default: N_OF_RUNS). "
                              "Each run writes its own timestamped CSV, so this can be "
                              "used to split the full N_OF_RUNS across a SLURM array, "
                              "one chunk per task.")
    parser.add_argument("--n", type=int, default=5000,
                         help="Points per sphere sample (default: 5000). Lower for a "
                              "cheap smoke test.")
    parser.add_argument("--n-surrogates", type=int, default=NUMBER_OF_SERNS,
                         help=f"SERN ensemble size (default: {NUMBER_OF_SERNS}). Lower "
                              "for a cheap smoke test.")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                         help=f"Where to write result CSVs (default: {OUTPUT_DIR}). "
                              "Use a separate directory for smoke tests so toy-scale "
                              "output doesn't mix into the real dataset.")
    args = parser.parse_args()
    main(n_runs=args.n_runs, n=args.n, n_surrogates=args.n_surrogates, output_dir=args.output_dir)
