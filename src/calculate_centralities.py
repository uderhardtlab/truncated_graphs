from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import sys
from tqdm import tqdm
plt.rcParams['svg.fonttype'] = 'none'

from border_effects_kNN_del import delaunay_edges, knn_edges, rnn_edges
from truncated_graphs import distance_to_border, compute_centrality_measures
from fit import fit_piece_wise_linear, fit_log
from evaluate_fit import log_likelihood, relative_likelihood


def process_dataset_del(dataset):
    coords = datasets[dataset]
    distances = distance_to_border(coords)
    edges = delaunay_edges(coords)
    df = compute_centrality_measures(coords, edges)
    df["dataset"] = dataset
    return pd.concat([distances, df], axis=1)


def process_dataset_knn(dataset, k):
    coords = datasets[dataset]
    distances = distance_to_border(coords)
    try:
        edges = knn_edges(coords, k=k)
        df = compute_centrality_measures(coords, edges)
    except Exception:
        df = pd.DataFrame()
    df["dataset"] = dataset
    df["k"] = k
    return pd.concat([distances, df], axis=1)


def process_dataset_rnn(dataset, radius_factor):
    coords = datasets[dataset]
    distances = distance_to_border(coords)
    base_radius = np.max(coords) / np.sqrt(len(coords))
    r = radius_factor * base_radius
    edges = rnn_edges(coords, r=r)
    df = compute_centrality_measures(coords, edges)
    df["dataset"] = dataset
    df["radius"] = r
    df["radius_factor"] = radius_factor
    return pd.concat([distances, df], axis=1)


if __name__ == "__main__":
    with open("../mibitof_coords/coords.pickle", "rb") as f:
        datasets = pickle.load(f)
    
    if 0:
        dfs = []
        for k in [5, 10, 15]:
            with ProcessPoolExecutor() as executor:
                futures = [executor.submit(process_dataset_knn, dataset, k) for dataset in datasets]

                for future in tqdm(as_completed(futures), total=len(futures)):
                    dfs.append(future.result())

            delaunay_dfs = pd.concat(dfs)
            delaunay_dfs.to_csv(f"../results/knn_centralities_k={k}.csv")

    else:
        dfs = []
        for radius_factor in [1, 2, 3]:
            with ProcessPoolExecutor() as executor:
                futures = [executor.submit(process_dataset_rnn, dataset, radius_factor) for dataset in datasets]

                for future in tqdm(as_completed(futures), total=len(futures)):
                    dfs.append(future.result())

            delaunay_dfs = pd.concat(dfs)
            delaunay_dfs.to_csv(f"../results/rnn_centralities_r={radius_factor}.csv")
