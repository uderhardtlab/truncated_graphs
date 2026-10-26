"""
sphere.py — Sphere benchmark using the bosperrus Flow class

Evaluates border-effect correction methods (piecewise-linear / exp-saturation
fits and SERN) on synthetic point clouds on the unit sphere.

BOSPERRUS correction variants (all post-processing here; the package stays additive):
  additive (BOSPERRUS default)  S + P - f(d), with f Flow's AIC-best fit and P its plateau
  multiplicative                S * P / f(d)   (NaN where f(d) <= 0)
  log                           s * expm1(BOSPERRUS additive on log1p(S / s)), s = median of S
                                (median of the positive S if that is 0); always defined for S >= 0
P is recovered from BOSPERRUS's own output as P = S_add - S + f(d).

Graphs are recomputed on the crop, as when a proximity graph is built on an imaged region.
Delaunay on the crop uses a stereographic projection from the cap's antipode, i.e. the spherical
Delaunay without the hull facets that triangulate the empty hole (see plot_delaunay_cap.py).
"delaunay_pruned" additionally drops edges longer than the upper boxplot whisker of the edge
lengths, as Giotto's createSpatialNetwork(maximum_distance_delaunay="auto") does.
All methods are correlated on the same nodes (finite in every method).  All graph
construction, centrality computation, distance calculation, fitting and
evaluation now delegate to the bosperrus package via the Flow class.
"""

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, Delaunay
from scipy.stats import pearsonr, spearmanr, vonmises_fisher

from time import time
from tqdm import trange

from bosperrus import Flow, construct_graph
from bosperrus.centrality_measures import compute_centrality_measures

from sern import surrogate_ensemble_gt

import os
os.environ["OMP_NUM_THREADS"] = "8"

# Rheinwalt et al. 2012 use 1000 surrogates, but their Fig. 2 shows reliability
# barely improves after ~400 -- the cheapest size their own analysis supports
NUMBER_OF_SERNS = 400
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

OUTPUT_DIR = "../results/correction_benchmark/sphere"


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


def _stereographic_delaunay_edges(coords, pole):
    """Planar Delaunay of unit-sphere coords projected stereographically from `pole` (a point not in the set).

    Projection preserves circles, so this equals the spherical Delaunay minus the facets whose empty
    circumcircle contains `pole` -- for a crop with pole = cap antipode, the facets filling the hole.
    """
    pole = pole / np.linalg.norm(pole)
    a = np.array([1.0, 0, 0]) if abs(pole[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(pole, a); e1 /= np.linalg.norm(e1)
    e2 = np.cross(pole, e1)
    denom = 1 - coords @ pole
    xy = np.column_stack((coords @ e1, coords @ e2)) / denom[:, None]
    edges = set()
    for simplex in Delaunay(xy).simplices:
        edges.add(frozenset((simplex[0], simplex[1])))
        edges.add(frozenset((simplex[1], simplex[2])))
        edges.add(frozenset((simplex[2], simplex[0])))
    return edges


def _prune_long_edges(coords, edges):
    """Drop edges longer (geodesically) than the upper boxplot whisker, as Giotto's maximum_distance_delaunay="auto"."""
    edges = list(edges)
    u, v = np.array([tuple(e) for e in edges]).T
    length = np.arccos(np.clip(np.einsum("ij,ij->i", coords[u], coords[v]), -1.0, 1.0))
    q1, q3 = np.percentile(length, [25, 75])
    whisker = length[length <= q3 + 1.5 * (q3 - q1)].max()
    return {e for e, keep in zip(edges, length <= whisker) if keep}


def get_edge_list(coords, edge_type, k=None, r=None, pole=None):
    """`pole`: for a crop, the cap antipode (only used by Delaunay); None for the full sphere."""
    if edge_type in ("delaunay", "delaunay_pruned"):
        edges = _spherical_delaunay_edges(coords) if pole is None else _stereographic_delaunay_edges(coords, pole)
        return _prune_long_edges(coords, edges) if edge_type == "delaunay_pruned" else edges
    # For knn / rnn we can reuse bosperrus graph construction directly,
    # because in 3-D Euclidean space the ordering of Euclidean distances
    # equals the ordering of geodesic distances on the unit sphere.
    if edge_type == "knn":
        return construct_graph(coords, "knn", k=k)
    elif edge_type == "rnn":
        return construct_graph(coords, "rnn", r=r)
    else:
        raise ValueError(f"Unknown edge type: {edge_type}")


def crop_cap(coords, cap_radius, rng=None, return_center=False):
    if rng is None:
        rng = np.random.default_rng()
    center = coords[rng.choice(len(coords))]
    dot = np.clip(coords @ center, -1.0, 1.0)
    geo_dist_to_center = np.arccos(dot)
    inside = np.where(geo_dist_to_center <= cap_radius)[0]
    dist_to_border = cap_radius - geo_dist_to_center[inside]
    dist = pd.Series(dist_to_border, index=inside, name="distance_to_cap")
    return (inside, dist, center) if return_center else (inside, dist)


def median_scale(S):
    """Pseudocount scale for the log variant: median of S, or median of the positive S if that is 0."""
    s = float(np.nanmedian(S))
    return s if s > 0 else float(np.nanmedian(S[S > 0]))


LOG_SCALES = {"log": median_scale}


def log_correction(distances, S, s):
    """BOSPERRUS additive correction of y = log1p(S / s), back-transformed to s * expm1(y_corr)."""
    bf = Flow.from_distances_and_scores(distances=distances.reset_index(drop=True),
                                        scores=pd.DataFrame({"y": np.log1p(S.to_numpy(float) / s)}))
    bf.flow(score_names=["y"])
    return pd.Series(s * np.expm1(bf.observations["BOSPERRUS corrected y"].to_numpy()), index=S.index)


def get_bosperrus_corrections(crop_coords, edges, measures, distances, log_scales=None):
    """Raw scores plus additive / multiplicative / log-variant corrections (one log column per entry of log_scales)."""
    log_scales = LOG_SCALES if log_scales is None else log_scales
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
    for m in scores.columns:
        bf.observations[f"BOSPERRUS multiplicative {m}"] = multiplicative_correction(bf, m)
        S = bf.observations[m].astype(float)
        for name, scale_fn in log_scales.items():
            bf.observations[f"BOSPERRUS {name} {m}"] = log_correction(distances, S, scale_fn(S))
    bf.observations["degree"] = bf.observations["degree"].astype(int)
    return bf.observations


def multiplicative_correction(bf, m):
    """S * P / f(d) with Flow's best fit f for score m and its plateau P (see module docstring)."""
    S = bf.observations[m].astype(float)
    S_add = bf.observations[f"BOSPERRUS corrected {m}"]
    f_d = pd.Series(bf.best_fits[m].predict(bf.observations["distance_to_cap"].to_numpy()), index=S.index)
    P = float((S_add - S + f_d).median())
    return S * (P / f_d.where(f_d > 0))


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
    # no int cast on degree: the ensemble median can be x.5, and truncating it
    # would bias the boundary-effect estimate down
    return pd.DataFrame(sern_median)


def _corr(corr_fn, x, y, ok=None):
    """Correlation over the nodes in `ok` (default: where both are finite)."""
    if ok is None:
        ok = np.isfinite(x.to_numpy(float)) & np.isfinite(y.to_numpy(float))
    return corr_fn(x[ok], y[ok]).statistic if ok.sum() > 2 else np.nan


def process_coords(coords, edge_type, cap_radii, k=None, r=None, rng=None):
    N = len(coords)

    # --- global graph & centralities ---
    global_edges = get_edge_list(coords, edge_type, k=k, r=r)
    global_centralities = pd.DataFrame(
        compute_centrality_measures(global_edges, N, MEASURES)
    )
    measures = global_centralities.columns
    all_correlations = []

    for cap_radius in cap_radii:
        crop, distances, center = crop_cap(coords, cap_radius, rng=rng, return_center=True)
        crop_coords = coords[crop]

        local_edges = get_edge_list(crop_coords, edge_type, k=k, r=r, pole=-center)
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
                "BOSPERRUS_multiplicative": BOSPERRUS_results[[f"BOSPERRUS multiplicative {m}" for m in cap_measures]].sort_index(axis=0).sort_index(axis=1),
                "BOSPERRUS_log": BOSPERRUS_results[[f"BOSPERRUS log {m}" for m in cap_measures]].sort_index(axis=0).sort_index(axis=1),
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
                rows = {k: [] for k in ["crop", "corrected", "mult", "log", "sern", "crop_sern", "n_excluded"]}

                for m in cap_measures:
                    cols = {
                        "original": sub["original"][m], "crop": sub["crop"][m],
                        "corrected": sub["BOSPERRUS_corrections"][f"BOSPERRUS corrected {m}"],
                        "mult": sub["BOSPERRUS_multiplicative"][f"BOSPERRUS multiplicative {m}"],
                        "log": sub["BOSPERRUS_log"][f"BOSPERRUS log {m}"],
                        "sern": sub["sern_corrected"][m], "sern_values": sub["sern"][m],
                    }
                    # same nodes for every method: the multiplicative variant is NaN where f(d) <= 0,
                    # and closeness can be NaN for isolated nodes
                    ok = np.logical_and.reduce([np.isfinite(c.to_numpy(float)) for c in cols.values()])
                    for key in ["crop", "corrected", "mult", "log", "sern"]:
                        rows[key].append(_corr(corr_fn, cols["original"], cols[key], ok))
                    rows["crop_sern"].append(_corr(corr_fn, cols["crop"], cols["sern_values"], ok))
                    rows["n_excluded"].append(int((~ok).sum()))

                correlations = pd.DataFrame(index=cap_measures)
                correlations["original vs. on crop"] = rows["crop"]
                correlations["original vs. BOSPERRUS corrected on crop"] = rows["corrected"]
                correlations["original vs. BOSPERRUS multiplicative on crop"] = rows["mult"]
                correlations["original vs. BOSPERRUS log on crop"] = rows["log"]
                correlations["original vs. SERN corrected on crop"] = rows["sern"]
                correlations["on crop vs. SERN values"] = rows["crop_sern"]
                correlations["n_excluded"] = rows["n_excluded"]
                correlations["cap_radius"] = cap_radius
                correlations["corr_method"] = corr_name
                correlations["node_subset"] = subset_name
                correlations["n_mult_nan"] = [
                    int(sub["BOSPERRUS_multiplicative"][f"BOSPERRUS multiplicative {m}"].isna().sum()) for m in cap_measures
                ]
                all_correlations.append(correlations)
    return pd.concat(all_correlations)


def main(n_runs=N_OF_RUNS, n=5000, n_surrogates=NUMBER_OF_SERNS, output_dir=OUTPUT_DIR, seed=None):
    global NUMBER_OF_SERNS
    NUMBER_OF_SERNS = n_surrogates
    os.makedirs(output_dir, exist_ok=True)

    for run in trange(n_runs):
        # run seed = seed + run; also seeds the legacy global RNG that sern.py draws its surrogate seeds from
        run_seed = None if seed is None else seed + run
        rng = np.random.default_rng(run_seed)
        if run_seed is not None:
            np.random.seed(run_seed)
        all_correlations = []

        coord_configs = [
            ("uniform", sample_uniform_on_unit_sphere(n=n, rng=rng)),
            ("kappa=1", sample_von_mises_fisher(n=n, kappas=[1], rng=rng)),
            ("kappa=1,3,5", sample_von_mises_fisher(n=n, kappas=[1, 3, 5], rng=rng)),
        ]

        for coord_type, coords in coord_configs:
            for edge_type in ["delaunay", "delaunay_pruned", "knn", "rnn"]:
                base_kwargs = dict(
                    coords=coords,
                    edge_type=edge_type,
                    cap_radii=[2],  # bigger cap only -- see jobs/correction_benchmark_sphere.sbatch
                    rng=rng,
                )

                if edge_type in ("delaunay", "delaunay_pruned"):
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
                    # r = 0.05 dropped: the crop graph fragments, raw correlations are at the ceiling
                    # (closeness 0.98, betweenness 0.99) and many nodes are isolated -- no boundary test
                    for r in [0.1, 0.15]:
                        corr = process_coords(**base_kwargs, r=r)
                        corr["radius"] = r
                        corr["graph_type"] = edge_type
                        corr["coord_type"] = coord_type
                        corr["n"] = n
                        all_correlations.append(corr)

        out = pd.concat(all_correlations)
        out["seed"] = run_seed
        name = f"correlations_seed{run_seed}.csv" if run_seed is not None else f"correlations_{time()}.csv"
        out.to_csv(os.path.join(output_dir, name))


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
    parser.add_argument("--seed", type=int, default=None,
                         help="Base RNG seed; run i uses seed + i (default: unseeded). "
                              "The SLURM array passes its task id.")
    args = parser.parse_args()
    main(n_runs=args.n_runs, n=args.n, n_surrogates=args.n_surrogates, output_dir=args.output_dir,
         seed=args.seed)
