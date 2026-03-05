import numpy as np
from scipy.stats import pearsonr, vonmises_fisher
from tqdm import trange
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, vonmises_fisher
from tqdm import trange 

from border_effects_kNN_del import knn_edges, rnn_edges, delaunay_edges_geodesic, delaunay_edges_on_subset_geodesic, knn_edges_on_subset, rnn_edges_on_subset, crop_and_dist_cap, reindex_edges_to_crop
from fit import fit_exponential_saturation, fit_piece_wise_linear, fit_constant, piecewise_plateau, exp_sat
from evaluate_fit import log_likelihood, akaike_information_criterion
from sphere import sample_uniform_on_unit_sphere, sample_von_mises_fisher


def sample_uniform_on_unit_sphere(n):
    theta = np.random.uniform(0, 2*np.pi, size=n) 
    u = np.random.uniform(-1, 1, size=n) 
    x = np.sqrt(1-u**2)*np.cos(theta)
    y = np.sqrt(1-u**2)*np.sin(theta)
    z = u
    coords = np.array([x, y, z]).T
    return coords


def sample_von_mises_fisher(n, kappas):
    if len(kappas) == 1:
        mu = sample_uniform_on_unit_sphere(1)[0]
        vmf = vonmises_fisher(mu=mu, kappa=kappas[0])
        coords = vmf.rvs(size=n)
    else:
        n = n // len(kappas)
        coords = list()
        for kappa in kappas:
            coords.append(sample_von_mises_fisher(n, kappas=[kappa]))
        coords = np.vstack(coords)
    return coords


def process_coords(coords, edge_type, k=None, r=None):
    crop, distances = crop_and_dist_cap(coords, radius_geodesic=1)

    if edge_type == "delaunay":
        edges = delaunay_edges_geodesic(coords)
        crop_edges = delaunay_edges_on_subset_geodesic(coords, crop)
    elif edge_type == "knn":
        edges = knn_edges(coords, k=k)
        crop_edges = knn_edges_on_subset(coords, crop, k=k)
    elif edge_type == "rnn":
        edges = rnn_edges(coords, r=r)
        crop_edges = rnn_edges_on_subset(coords, crop, r=r)
    else:
        print("choose proper edge type")

    original_centralities = pd.DataFrame(compute_centrality_measures(edges, len(coords)))
    crop_centralities = pd.DataFrame(compute_centrality_measures(crop_edges, len(crop))).dropna()

    measures = crop_centralities.columns
    d = distances.sort_index()
    methods = list()
    rel_ll = list()
    corrections = pd.DataFrame(columns=measures) 
    
    for m in measures:
        C = crop_centralities[m].sort_index()
        m_pieli, c_pieli, b_pieli, C_pieli = fit_piece_wise_linear(d, C)
        ll_pieli = log_likelihood(C, C_pieli)
        aic_pieli = akaike_information_criterion(4, ll_pieli)
        
        a_exp, b_exp, c_exp, C_exp = fit_exponential_saturation(d, C)
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
    
    local_edges = reindex_edges_to_crop(crop_edges, crop)
    n_bins = np.sqrt(len(local_edges))

    median = surrogate_ensemble_gt(coords=coords[crop], edge_list=local_edges, n_bins=int(n_bins), n_surrogates=500, n_jobs=40)
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
    return correlations


if __name__ == "__main__":
    all_correlations = list()
    failures = list()
    for _ in trange(1):
        for n in [1000]:
            all_coords = list()
            all_coords.append(sample_uniform_on_unit_sphere(n=n))
            all_coords.append(sample_von_mises_fisher(n=n, kappas=[1]))
            all_coords.append(sample_von_mises_fisher(n=n, kappas=[1, 3, 5]))

            for (coord_type, coords) in zip(["uniform", "k=1", "k=1,2,3"], all_coords):
                for edge_type in ["delaunay", "knn", "rnn"]:
                    try:
                        if edge_type == "delaunay":
                            correlations = process_coords(coords, edge_type)
                            correlations["graph_type"] = edge_type
                            correlations["coord_type"] = coord_type
                            correlations["n"] = n
                            all_correlations.append(correlations)
                        elif edge_type == "knn":
                            for k in [5, 10, 15]:
                                correlations = process_coords(coords, edge_type, k=k)
                                correlations["k"] = k
                                correlations["graph_type"] = edge_type
                                correlations["coord_type"] = coord_type
                                correlations["n"] = n
                                all_correlations.append(correlations)
                        elif edge_type == "rnn":
                            for r in [1, 2, 3]:
                                correlations = process_coords(coords, edge_type, r=r)
                                correlations["r"] = r
                                correlations["graph_type"] = edge_type
                                correlations["coord_type"] = coord_type
                                correlations["n"] = n
                                all_correlations.append(correlations)
                    except:
                        failures.append(n, coord_type, edge_type)
    pd.concat(all_correlations).to_csv("correlations.csv")