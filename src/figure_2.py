import matplotlib.pyplot as plt
import pandas as pd
import pickle
import sys
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.append("../../bosperrus-package/")
from bosperrus import *
from bosperrus.distances import distance_to_convex_hull


def get_fits(dataset, coordinates, graph_type):
    params = dict()
    graph_type_1 = graph_type.split("_")[0]
    
    mins = coordinates.min(axis=0)
    maxs = coordinates.max(axis=0)
    coordinates = (coordinates - mins) / (maxs - mins)
            
    if graph_type_1 != "delaunay":
        param = graph_type.split("_")[1][0]
        if param == "r":
            value = float(graph_type.split("_")[1].split("=")[-1])
        else:
            value = int(graph_type.split("_")[1].split("=")[-1])
        params[param] = value
    else:
        params = None
    try:
        bf = Flow.from_coords(coordinates=coordinates, distance_fn=distance_to_convex_hull, measures=["degree", "pagerank", "betweenness", "closeness", "clustering"], graph_type=graph_type_1, distance_kwargs=None, graph_kwargs=params)
        bf.flow()
    except ValueError as e:
        print(f"Error processing {dataset} with graph type {graph_type}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error

    result = bf.fit_quality.T
    result["dataset"] = dataset
    result["num_edges"] = len(bf._edge_list)
    result["num_nodes"] = len(coordinates)
    return result



def get_fit_data(datasets, graph_type):
    print("calculating fits and their quality...")
    try:
        conc = pd.read_csv(f"../results/figure2/{graph_type}_graph_level_fits.csv")
    except:
        dfs = list()
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(get_fits, dataset, datasets[dataset], graph_type) for dataset in datasets]
            for future in tqdm(as_completed(futures), total=len(futures)):
                dfs.append(future.result())
        conc = pd.concat(dfs)
        conc.to_csv(f"../results/figure2/{graph_type}_graph_level_fits.csv")
    return conc


def main(graph_types = ["delaunay", "knn_k=5", "knn_k=10", "knn_k=15", "rnn_r=0.02", "rnn_r=0.03", "rnn_r=0.04", "rnn_r=0.05"], pickle_f="/data/bionets/je30bery/truncated_graphs/mibitof_coords/coords.pickle"):
    with open(pickle_f, "rb") as f:
        datasets = pickle.load(f)
        del datasets["glioma_mibitof:CHOP_907_R1C6_whole_cell.tiff"] # this guy only has 3 cells
    
    #test_key = list(datasets.keys())[0]
    #print(get_fits(test_key, datasets[test_key], "knn_k=5"))

    conconc = list() # ha ha 
    for graph_type in graph_types:
        conc = get_fit_data(datasets, graph_type)
        conc["graph_type"] = graph_type
        conconc.append(conc)
        
    conconc = pd.concat(conconc)
    return conconc


if __name__ == "__main__":    
    main()