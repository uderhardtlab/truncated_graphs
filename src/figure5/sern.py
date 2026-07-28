import numpy as np
import os
import joblib
from scipy.spatial.distance import pdist
from joblib import Parallel, delayed
from tqdm import tqdm

import sys
sys.path.append("../../bosperrus-package/")
from bosperrus import *


def generate_sern_vectorized(pair_probs, rows, cols, rng):
    # Create the probability mask for all possible edges
    mask = rng.random(len(pair_probs)) < pair_probs

    # rows/cols are the precomputed upper-triangle (i, j) indices, in the
    # same order as pdist/pair_probs; apply the mask to get selected edges
    edges = np.column_stack((rows[mask], cols[mask]))
    return edges

def surrogate_worker_mmap(seed, prob_path, triu_path, N):
    # mmap_mode='r' ensures all workers read the same physical RAM
    pair_probs = joblib.load(prob_path, mmap_mode='r')
    rows, cols = joblib.load(triu_path, mmap_mode='r')
    rng = np.random.default_rng(seed)

    edges = generate_sern_vectorized(pair_probs, rows, cols, rng)
    return compute_centrality_measures(edges, N)

def surrogate_ensemble_gt(coords, edge_list, n_bins, n_surrogates=200, n_jobs=-1):
    # 1. Calculate probabilities (keeping your existing logic)
    p, bin_edges, pair_bins = estimate_link_probability(coords, edge_list, n_bins)
    pair_probs = build_pair_probabilities(pair_bins, p)
    N = len(coords)

    # 2. Memory-map the pair_probs array and the upper-triangle (i, j) index
    # pairs. Both only depend on N/coords, not on the surrogate, so they are
    # computed once here and shared read-only across all workers instead of
    # being recomputed (and reallocated) inside every single surrogate call.
    prob_path = 'pair_probs.mmap'
    triu_path = 'triu_indices.mmap'
    for path in (prob_path, triu_path):
        if os.path.exists(path):
            os.remove(path)
    joblib.dump(pair_probs, prob_path)
    joblib.dump(np.triu_indices(N, k=1), triu_path)

    # 3. Execution
    seeds = np.random.randint(0, 1_000_000, n_surrogates)

    try:
        results = Parallel(n_jobs=n_jobs, batch_size='auto')(
            delayed(surrogate_worker_mmap)(seed, prob_path, triu_path, N)
            for seed in tqdm(seeds, desc="SERN")
        )
    finally:
        # Clean up the temporary mmap files
        for path in (prob_path, triu_path):
            if os.path.exists(path):
                os.remove(path)

    # Compute median across all surrogates
    medians = {}
    for key in results[0].keys():
        values = np.array([np.asarray(r[key]) for r in results])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            measure_medians = np.nanmedian(values, axis=0)

        measure_medians = np.nan_to_num(measure_medians, nan=0)
        medians[key] = measure_medians
    return medians


def estimate_link_probability(coords, edge_list, n_bins):
    coords = np.asarray(coords, dtype=np.float32)
    dists = pdist(coords)
    bin_edges = np.linspace(dists.min(), dists.max(), n_bins + 1)
    pair_bins = np.digitize(dists, bin_edges) - 1
    pair_bins = np.clip(pair_bins, 0, n_bins - 1)

    A = np.bincount(pair_bins, minlength=n_bins)

    # Dedup edges to canonical (i < j) pairs, same as before, then map each
    # edge directly to its pdist condensed index (the standard i<j upper-
    # triangle index formula) and look up its bin in pair_bins. This avoids
    # an O(N^2) Python-level loop over every possible pair — cost now scales
    # with the number of edges instead of the number of nodes squared.
    N = len(coords)
    edge_arr = np.array(
        list({tuple(sorted(e)) for e in edge_list}), dtype=np.int64
    ).reshape(-1, 2)
    i, j = edge_arr[:, 0], edge_arr[:, 1]
    condensed_idx = N * i - i * (i + 1) // 2 + (j - i - 1)
    B = np.bincount(pair_bins[condensed_idx], minlength=n_bins).astype(np.float64)

    p = np.divide(B, A, out=np.zeros_like(B), where=A > 0)
    return p.astype(np.float32), bin_edges, pair_bins

def build_pair_probabilities(pair_bins, p):
    return p[pair_bins]

