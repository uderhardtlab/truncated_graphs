# Requirements:
# pip install anndata numpy scipy networkx tqdm
import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import pdist, squareform
import networkx as nx
from tqdm import trange


def get_spatial_coords(adata):
    if 'spatial' in adata.obsm:
        coords = np.asarray(adata.obsm['spatial'])
    else:
        raise ValueError("AnnData has no adata.obsm['spatial']. Provide node coordinates there.")
    return coords
        
def build_serns_from_anndata(adata, n_surr=500, n_bins=50, distance_metric='euclidean', seed=None, return_p_of_d=False):
    """
    Build spatially-embedded random network (SERN) surrogates from an AnnData object.
    - adata.obsm['spatial'] is used for node coordinates (Nx2 or Nx3).
    - adata.obsp['connectivities'] is used as the reference adjacency (sparse matrix).
    Returns:
      surrogates: list of numpy boolean adjacency matrices (shape (n_obs,n_obs)) length n_surr
      p_bins, bin_edges: arrays describing the estimated p(d) if return_p_of_d True
    """
    rng = np.random.default_rng(seed)
    
    # 1) get coordinates
    coords = get_coords

    n = coords.shape[0]

    # 2) get pairwise distances (use provided distances if available to avoid recomputing)
    if 'distances' in adata.obsp and sp.issparse(adata.obsp['distances']):
        # If obsp distances exists as sparse full pairwise distances, convert to dense
        pair_dists = adata.obsp['distances'].toarray()
    else:
        pair_dists = squareform(pdist(coords, metric=distance_metric))

    # 3) get adjacency (binary)
    if 'connectivities' not in adata.obsp:
        raise ValueError("AnnData has no adata.obsp['connectivities'].")
    A = adata.obsp['connectivities']
    if sp.issparse(A):
        A = (A > 0).astype(int).toarray()
    else:
        A = (np.asarray(A) > 0).astype(int)

    # ensure symmetric and no self-loops
    A = np.triu(A, k=1)
    A = A + A.T

    # 4) compute empirical p(d) in distance bins:
    # For each bin, p = (# observed links with distance in bin) / (# possible pairs with distance in bin)
    d_upper = pair_dists[np.triu_indices(n, k=1)]
    a_upper = A[np.triu_indices(n, k=1)]

    # create bins
    bin_counts_total, bin_edges = np.histogram(d_upper, bins=n_bins)
    # counts of actual edges per bin
    bin_counts_links, _ = np.histogram(d_upper[a_upper == 1], bins=bin_edges)

    # avoid zero-division: where no pairs fall into a bin, set p to 0
    with np.errstate(divide='ignore', invalid='ignore'):
        p_of_bin = np.where(bin_counts_total > 0, bin_counts_links / bin_counts_total, 0.0)

    # Optionally smooth p_of_bin (simple moving average) to avoid noisy bins:
    # here small smoothing window; adjust or remove as desired
    def smooth(x, k=3):
        if k <= 1: return x
        pad = k//2
        xp = np.pad(x, pad, mode='edge')
        kernel = np.ones(k)/k
        return np.convolve(xp, kernel, mode='valid')
    p_of_bin = smooth(p_of_bin, k=3)

    # 5) precompute each pair's bin index
    # use upper triangular pairs for sampling
    # For speed, compute bin index array aligned with i<j pairs
    pair_bins = np.digitize(d_upper, bin_edges) - 1
    # clamp to valid bin indices
    pair_bins = np.clip(pair_bins, 0, len(p_of_bin)-1)

    # 6) generate surrogates
    surrogates = []
    rng_integers = rng.integers  # speed alias
    for _ in trange(n_surr, desc="Generating SERNs"):
        # sample edges for upper triangular pairs according to p_of_bin
        probs = p_of_bin[pair_bins]
        # draw bernoulli for each pair
        draws = rng.random(len(probs)) < probs
        # construct adjacency matrix
        A_surr = np.zeros((n, n), dtype=np.int8)
        iu = np.triu_indices(n, k=1)
        A_surr[iu] = draws.astype(np.int8)
        A_surr = A_surr + A_surr.T
        surrogates.append(A_surr)

    if return_p_of_d:
        return surrogates, p_of_bin, bin_edges
    return surrogates


# Example: compute degree z-scores against the SERN ensemble
def degree_zscore_from_serns(adata, surrogates):
    """
    Compute degree and z-score of degree for each node comparing original adata connectivities to SERN surrogates.
    Returns:
      deg_orig: (n,) degrees
      deg_mean: (n,) mean degree across surrogates
      deg_std: (n,) std deviation across surrogates
      z_score: (n,) (deg_orig - deg_mean) / deg_std  (where deg_std==0 -> np.nan)
    """
    import numpy as np
    if 'connectivities' in adata.obsp:
        A = adata.obsp['connectivities']
        if sp.issparse(A):
            A = (A > 0).astype(int).toarray()
        else:
            A = (np.asarray(A) > 0).astype(int)
    else:
        raise ValueError("AnnData has no adata.obsp['connectivities'].")

    n = A.shape[0]
    deg_orig = A.sum(axis=1)

    degs = np.vstack([s.sum(axis=1) for s in surrogates])  # shape (n_surr, n)
    deg_mean = degs.mean(axis=0)
    deg_std = degs.std(axis=0, ddof=1)

    z = (deg_orig - deg_mean) / (deg_std + 1e-12)
    z[deg_std == 0] = np.nan
    return deg_orig, deg_mean, deg_std, z