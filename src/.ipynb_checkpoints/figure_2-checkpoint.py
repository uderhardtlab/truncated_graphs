from itertools import combinations
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import networkx as nx
import numpy as np
import pandas as pd
import pickle
from scipy.spatial import ConvexHull, Delaunay
from scipy.stats import mannwhitneyu
import numpy as np
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from statsmodels.stats.multitest import multipletests
import sys
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
plt.rcParams['svg.fonttype'] = 'none'

sys.path.append("../../bosporus-package/")
from bosporus import *


def get_fits(dataset, coordinates, graph_type):

    params = dict()
    graph_type_1 = graph_type.split("_")[0]
    if graph_type_1 != "delaunay":
        param = graph_type.split("_")[1][0]
        value = float(graph_type.split("_")[1].split("=")[-1])
        if param == "r":
            mins = coordinates.min(axis=0)
            maxs = coordinates.max(axis=0)
            coordinates = (coordinates - mins) / (maxs - mins)
        else:
            value = int(value)
        params[param] = value
    else:
        params = None
    bf = BosporusFlow(coordinates=coordinates)
    bf.run_all(graph_type_1, params, bf.compute_distance_to_convex_hull)

    result = bf.fit_quality
    result["dataset"] = dataset
    result["num_edges"] = len(bf.edge_list)
    result["num_nodes"] = len(coordinates)
    return result



def get_fit_data(graph_type):
    print("calculating fits and their quality...")
    try:
        pd.read_csv(f"../results/{graph_type}_graph_level_fits.csv")
    except:
        dfs = list()
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(get_fits, dataset, datasets[dataset], graph_type) for dataset in datasets]
            for future in tqdm(as_completed(futures), total=len(futures)):
                dfs.append(future.result())
        conc = pd.concat(dfs)
        conc.to_csv(f"../results/{graph_type}_graph_level_fits.csv")
    return conc


def main(graph_types = ["delaunay", "knn_k=5", "knn_k=10", "knn_k=15", "rnn_r=0.03", "rnn_r=0.04", "rnn_r=0.05"], pickle_f="/data/bionets/je30bery/truncated_graphs/mibitof_coords/coords.pickle"):
    with open(pickle_f, "rb") as f:
        datasets = pickle.load(f)
        del datasets["glioma_mibitof:CHOP_907_R1C6_whole_cell.tiff"] # this guy only has 3 cells
    
    conconc = list() # ha ha 
    for graph_type in graph_types:
        conc = get_fit_data(graph_type)
        conc["graph_type"] = graph_type
        conconc.append(conc)
        
    conconc = pd.concat(conconc)
    return conconc


if __name__ == "__main__":    
    main()