import numpy as np
from scipy.spatial.distance import pdist, squareform
from graph_tool.all import Graph
from graph_tool.centrality import (
    betweenness,
    pagerank,
    closeness,
)
from graph_tool.clustering import local_clustering
import numpy as np
from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm

def estimate_link_probability(coords, edge_list, n_bins):

    coords = np.asarray(coords, dtype=np.float32)
    N = len(coords)

    # full matrix (float32)
    dist_matrix = squareform(pdist(coords)).astype(np.float32)

    upper = dist_matrix[np.triu_indices(N, k=1)]
    bin_edges = np.linspace(upper.min(), upper.max(), n_bins + 1)

    edge_set = set(tuple(sorted(e)) for e in edge_list)

    A = np.zeros(n_bins, dtype=np.int64)
    B = np.zeros(n_bins, dtype=np.int64)

    for i in range(N):
        for j in range(i):
            d = dist_matrix[i, j]
            b = np.searchsorted(bin_edges, d, side="right") - 1
            if b == n_bins:
                b -= 1

            A[b] += 1
            if (j, i) in edge_set:
                B[b] += 1

    p = np.divide(B, A, out=np.zeros_like(B, dtype=np.float32), where=A > 0)

    return p, bin_edges, dist_matrix


def generate_sern(N, p, bin_edges, dist_matrix, rng):
    n_bins = len(p)
    edges = []

    for i in range(N):
        for j in range(i):
            d = dist_matrix[i, j]
            b = np.searchsorted(bin_edges, d, side="right") - 1
            if b == n_bins:
                b -= 1

            if rng.random() < p[b]:
                edges.append((j, i))

    return edges


def compute_centrality_measures(edge_list, N):

    g = Graph(directed=False)
    g.add_vertex(N)
    g.add_edge_list(edge_list)

    results = {}

    # Degree
    deg = g.get_out_degrees(g.get_vertices())
    results["degree"] = np.array(deg)

    # PageRank
    pr = pagerank(g)
    results["pagerank"] = np.array(pr.a)

    # Betweenness
    vb, _ = betweenness(g)
    results["betweenness"] = np.array(vb.a)

    # Closeness
    cl = closeness(g)
    results["closeness"] = np.array(cl.a)

    # Harmonic centrality (via closeness parameter)
    harm = closeness(g, harmonic=True)
    results["harmonic"] = np.array(harm.a)

    # Clustering
    lc = local_clustering(g)
    results["clustering"] = np.array(lc.a)

    return results


def surrogate_worker(seed, N, p, bin_edges, dist_matrix):

    rng = np.random.default_rng(seed)
    edges = generate_sern(N, p, bin_edges, dist_matrix, rng)

    return compute_centrality_measures(edges, N)


def surrogate_ensemble_gt(coords, edge_list, n_bins,
                          n_surrogates=200,
                          n_jobs=2):

    p, bin_edges, dist_matrix = estimate_link_probability(
        coords, edge_list, n_bins
    )

    N = len(coords)
    seeds = np.random.randint(0, 1_000_000, size=n_surrogates)

    results = Parallel(n_jobs=n_jobs)(
        delayed(surrogate_worker)(
            seed, N, p, bin_edges, dist_matrix
        )
        for seed in tqdm(seeds, desc="Generating surrogates")
    )

    return results