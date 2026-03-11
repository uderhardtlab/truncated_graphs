import numpy as np
import os
import joblib
from scipy.spatial.distance import pdist
from graph_tool.all import Graph
from graph_tool.centrality import betweenness, pagerank, closeness
from graph_tool.clustering import local_clustering
from joblib import Parallel, delayed
from tqdm import tqdm


def generate_sern_vectorized(pair_probs, N, rng):
    """
    Vectorized edge generation to bypass slow Python loops.
    """
    # Create the probability mask for all possible edges
    mask = rng.random(len(pair_probs)) < pair_probs
    
    # Get the (i, j) indices for the upper triangle (excluding diagonal)
    # This matches the order of pdist/pair_probs
    rows, cols = np.triu_indices(N, k=1)
    
    # Apply the mask to get only the selected edges
    edges = np.column_stack((rows[mask], cols[mask]))
    return edges

def surrogate_worker_mmap(seed, prob_path, N):
    """
    Worker that reads from a shared memory-mapped file.
    """
    # mmap_mode='r' ensures all 128 cores read the same physical RAM
    pair_probs = joblib.load(prob_path, mmap_mode='r')
    rng = np.random.default_rng(seed)
    
    edges = generate_sern_vectorized(pair_probs, N, rng)
    return compute_centrality_measures(edges)

def surrogate_ensemble_gt(coords, edge_list, n_bins, n_surrogates=200, n_jobs=-1):
    # 1. Calculate probabilities (keeping your existing logic)
    p, bin_edges, pair_bins = estimate_link_probability(coords, edge_list, n_bins)
    pair_probs = build_pair_probabilities(pair_bins, p)
    N = len(coords)

    # 2. Memory-map the pair_probs array
    # This prevents pickling overhead when sending data to 128 workers
    prob_path = 'pair_probs.mmap'
    if os.path.exists(prob_path): 
        os.remove(prob_path)
    joblib.dump(pair_probs, prob_path)

    # 3. Execution
    seeds = np.random.randint(0, 1_000_000, n_surrogates)
    
    try:
        results = Parallel(n_jobs=n_jobs, batch_size='auto')(
            delayed(surrogate_worker_mmap)(seed, prob_path, N)
            for seed in tqdm(seeds, desc="SERN")
        )
    finally:
        # Clean up the temporary mmap file
        if os.path.exists(prob_path):
            os.remove(prob_path)

    return results

def estimate_link_probability(coords, edge_list, n_bins):
    coords = np.asarray(coords, dtype=np.float32)
    dists = pdist(coords)
    bin_edges = np.linspace(dists.min(), dists.max(), n_bins + 1)
    pair_bins = np.digitize(dists, bin_edges) - 1
    pair_bins = np.clip(pair_bins, 0, n_bins - 1)

    A = np.bincount(pair_bins, minlength=n_bins)
    
    # Optimization: Use a structured array or a hash for faster edge lookup
    edge_set = set(tuple(sorted(e)) for e in edge_list)
    N = len(coords)
    B = np.zeros(n_bins)

    # Note: This part remains O(N^2) but is only run once per ensemble,
    # unlike the surrogate generation which runs hundreds of times.
    idx = 0
    for i in range(N):
        for j in range(i + 1, N): # Standard upper triangle iteration
            if (i, j) in edge_set:
                b = pair_bins[idx]
                B[b] += 1
            idx += 1

    p = np.divide(B, A, out=np.zeros_like(B), where=A > 0)
    return p.astype(np.float32), bin_edges, pair_bins

def build_pair_probabilities(pair_bins, p):
    return p[pair_bins]

def compute_centrality_measures(edge_list):
    g = Graph(directed=False)
    g.add_edge_list(edge_list)
    
    # Pre-allocate results with .a.copy() to ensure data is returned 
    # as standard numpy arrays before the process terminates.
    results = {
        "degree": g.get_total_degrees(range(g.num_vertices())).copy(),
        "pagerank": pagerank(g).a.copy(),
        "betweenness": betweenness(g)[0].a.copy(),
        "closeness": closeness(g).a.copy(),
        "harmonic": closeness(g, harmonic=True).a.copy(),
        "clustering": local_clustering(g).a.copy()
    }
    return {k: list(v) for k, v in results.items()}