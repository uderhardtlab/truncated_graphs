import numpy as np
import os
import pandas as pd
from scipy.stats import pearsonr, vonmises_fisher
from time import time
from tqdm import trange

from border_effects_kNN_del import knn_edges, rnn_edges, delaunay_edges_geodesic, delaunay_edges_on_subset_geodesic, knn_edges_on_subset, rnn_edges_on_subset, crop_and_dist_cap, reindex_edges_to_crop
from fit import fit_exponential_saturation, fit_piece_wise_linear, fit_constant, piecewise_plateau, exp_sat
from evaluate_fit import log_likelihood, akaike_information_criterion
from sern import compute_centrality_measures, surrogate_ensemble_gt

os.environ["OMP_NUM_THREADS"] = "8"
NUMBER_OF_SERNS = 100
N_JOBS = 16
N_OF_RUNS = 1


def sample_uniform_on_unit_sphere(n, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    u = rng.uniform(-1, 1, n)
    theta = rng.uniform(0, 2*np.pi, n)

    r = np.sqrt(1 - u**2)

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = u

    return np.column_stack((x, y, z))


def sample_von_mises_fisher(n, kappas, rng=None):

    if rng is None:
        rng = np.random.default_rng()

    if len(kappas) == 1:
        mu = sample_uniform_on_unit_sphere(1, rng=rng)[0]
        vmf = vonmises_fisher(mu=mu, kappa=kappas[0])
        return vmf.rvs(size=n, random_state=rng)

    coords = []
    per_cluster = n // len(kappas)

    for kappa in kappas:
        coords.append(
            sample_von_mises_fisher(per_cluster, [kappa], rng=rng)
        )
    return np.vstack(coords)



def process_coords(coords, edge_type, cap_radii, k=None, radius_factor=None):
    N = len(coords)
    
    if edge_type == "delaunay":
        edges = delaunay_edges_geodesic(coords)
    elif edge_type == "knn":
        edges = knn_edges(coords, k=k)
    elif edge_type == "rnn":
        # Spherical Base Radius Calculation:
        # Area of unit sphere = 4 * pi. 
        # Average area per node = 4*pi / N.
        # Equivalent radius r = sqrt((4*pi/N) / pi) = 2 / sqrt(N)
        base_radius = 4 / np.sqrt(N)
        r = radius_factor * base_radius
        edges = rnn_edges(coords, r=r)
    else:
        raise ValueError("Invalid edge type")

    original_centralities = pd.DataFrame(
        compute_centrality_measures(edges)
    )   
    all_correlations = list()
    
    for cap_radius in cap_radii:
        crop, distances = crop_and_dist_cap(coords, radius_geodesic=cap_radius)
        if edge_type == "delaunay":
            crop_edges = delaunay_edges_on_subset_geodesic(coords, crop)
        elif edge_type == "knn":
            crop_edges = knn_edges_on_subset(coords, crop, k=k)
        elif edge_type == "rnn":
            crop_edges = rnn_edges_on_subset(coords, crop, r=r)
        else:
            print("choose proper edge type")

        reindexed = reindex_edges_to_crop(crop_edges, crop)
        crop_centralities = compute_centrality_measures(reindexed)
        crop_centralities = pd.DataFrame(crop_centralities, index=crop)

        measures = crop_centralities.columns
        distances_and_centralities = pd.concat({"distance": distances, "centrality": crop_centralities.sort_index()}, axis=1).dropna().sort_index()

        d = distances_and_centralities[("distance", "distance_to_border")].values
        methods = list()
        rel_ll = list()
        corrections = pd.DataFrame(columns=measures) 
        
        for m in measures:
            C = distances_and_centralities[("centrality", m)].sort_index()
            try:
                m_pieli, c_pieli, b_pieli, C_pieli = fit_piece_wise_linear(d, C)
            except:
                m_pieli, c_pieli, b_pieli, C_pieli = np.nan, np.nan, np.nan, np.zeros_like(C)
            ll_pieli = log_likelihood(C, C_pieli)
            aic_pieli = akaike_information_criterion(4, ll_pieli)
            
            try:
                a_exp, b_exp, c_exp, C_exp = fit_exponential_saturation(d, C)
            except:
                a_exp, b_exp, c_exp, C_exp = np.nan, np.nan, np.nan, np.zeros_like(C)
            ll_exp = log_likelihood(C, C_exp)
            aic_exp = akaike_information_criterion(4, ll_exp)
            
            const, C_const = fit_constant(C)
            ll_const = log_likelihood(C, C_const)
            aic_const = akaike_information_criterion(2, ll_const)
                        
            if aic_const == np.min([aic_const, aic_pieli, aic_exp]):
                C_corrected = C
                methods.append("original")
                rel_ll.append(1.0)
            else:
                if aic_pieli < aic_exp:
                    plateau = m_pieli * b_pieli + c_pieli
                    C_corrected = C + plateau - piecewise_plateau(d, b_pieli, m_pieli, c_pieli)
                    methods.append("pieli")
                    rel_ll.append(np.exp((aic_const - aic_pieli) / (2 * len(C)))) # Removed len(C)
                else:
                    plateau = a_exp + c_exp
                    C_corrected = C + plateau - exp_sat(d, a=a_exp, b=b_exp, c=c_exp)
                    
                    methods.append("exp")
                    rel_ll.append(np.exp((aic_const - aic_exp) / (2 * len(C)))) # Removed len(C)
        
            corrections[m] = C_corrected
        
        reindexed = reindex_edges_to_crop(crop_edges, crop)
        n_bins = np.sqrt(len(reindexed))

        median = surrogate_ensemble_gt(coords=coords[crop], edge_list=reindexed, n_bins=int(n_bins), n_surrogates=NUMBER_OF_SERNS, n_jobs=N_JOBS)
        sern_results = pd.DataFrame(median[0], index=crop)
        crop_centralities["degree"] = crop_centralities["degree"].astype(int)
        sern_results["degree"] = sern_results["degree"].astype(int)
        results = pd.concat({"original": original_centralities.sort_index(axis=0).sort_index(axis=1), "crop": crop_centralities.sort_index(axis=0).sort_index(axis=1), "distance": distances.sort_index(axis=0), "corrections": corrections.sort_index(axis=0).sort_index(axis=1), "sern": sern_results, "sern_corrected": crop_centralities.sort_index(axis=0).sort_index(axis=1)-(sern_results.sort_index(axis=0).sort_index(axis=1))}, axis=1).iloc[crop]#.dropna()
        results = results.fillna(0)

        corrs_original_crop = list()
        corrs_original_corrected = list()
        corrs_original_sern = list()
        corrs_crop_sern = list()
        
        for m in measures:
            corrs_original_crop.append(pearsonr(results["original"][m], results["crop"][m]).statistic)
            corrs_original_corrected.append(pearsonr(results["original"][m], results["corrections"][m]).statistic)
            corrs_original_sern.append(pearsonr(results["original"][m], results["sern_corrected"][m]).statistic)
            corrs_crop_sern.append(pearsonr(results["crop"][m], results["sern"][m]).statistic)

        correlations = pd.DataFrame(index=measures)
        correlations["original vs. on crop"] = corrs_original_crop
        correlations["original vs. corrected on crop"] = corrs_original_corrected
        correlations["original vs. SERN corrected on crop"] = corrs_original_sern
        correlations["on crop vs. SERN values"] = corrs_crop_sern
        
        correlations["model"] = methods
        correlations["relative_likelihood"] = rel_ll
        correlations["cap_radius"] = cap_radius
        all_correlations.append(correlations)
    return pd.concat(all_correlations)


if __name__ == "__main__":
    failures = list()
    n = 5000
    for _ in trange(N_OF_RUNS):
        all_correlations = list()

        all_coords = list()
        all_coords.append(sample_uniform_on_unit_sphere(n=n))
        all_coords.append(sample_von_mises_fisher(n=n, kappas=[1]))
        all_coords.append(sample_von_mises_fisher(n=n, kappas=[1, 3, 5]))

        for (coord_type, coords) in zip(["uniform", "kappa=1", "kappa=1,3,5"], all_coords):
            for edge_type in ["delaunay", "knn", "rnn"]:
                try:
                    if edge_type == "delaunay":
                        correlations = process_coords(coords, edge_type, cap_radii=[1, 2])
                        correlations["graph_type"] = edge_type
                        correlations["coord_type"] = coord_type
                        correlations["n"] = n
                        all_correlations.append(correlations)
                    elif edge_type == "knn":
                        for k in [5, 10, 15]:
                            correlations = process_coords(coords, edge_type, k=k, cap_radii=[1, 2])
                            correlations["k"] = k
                            correlations["graph_type"] = edge_type
                            correlations["coord_type"] = coord_type
                            correlations["n"] = n
                            all_correlations.append(correlations)
                    elif edge_type == "rnn":
                        for r in [1, 2, 3]:
                            correlations = process_coords(coords, edge_type, radius_factor=r, cap_radii=[1, 2])
                            correlations["radius_factor"] = r
                            correlations["graph_type"] = edge_type
                            correlations["coord_type"] = coord_type
                            correlations["n"] = n
                            all_correlations.append(correlations)
                except Exception as e:
                    print(e)
                    failures.append((n, coord_type, edge_type))
        print(failures)
        pd.concat(all_correlations).to_csv(f"correlations_{time()}.csv")