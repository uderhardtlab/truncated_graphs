import numpy as np
import anndata as ad
import squidpy as sq
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics import mean_absolute_error
import pandas as pd
from scipy import stats
from scipy.spatial import distance


def distance_to_border(coords):
    if coords.shape[1] != 2:
        raise ValueError("Spatial coordinates must be Nx2.")
    x = coords[:, 0]
    y = coords[:, 1]

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    # distances to each of the four borders
    d_left   = x - xmin
    d_right  = xmax - x
    d_bottom = y - ymin
    d_top    = ymax - y

    # distance to the rectangle boundary = smallest distance to any border
    d_border = np.vstack([d_left, d_right, d_bottom, d_top]).min(axis=0)

    return pd.Series(d_border, name="distance_to_border")
    

def compute_centrality_measures(coords, edges):
    graph = nx.from_edgelist(edges)

    degree_centrality = nx.degree_centrality(graph)
    closeness_centrality = nx.closeness_centrality(graph)
    betweenness_centrality = nx.betweenness_centrality(graph)
    harmonic_centrality = nx.harmonic_centrality(graph)
    clustering = nx.clustering(graph)
    pagerank = nx.pagerank(graph)
    df = pd.concat([pd.Series(degree_centrality, name="degree"), 
                    pd.Series(closeness_centrality, name="closeness"),
                    pd.Series(betweenness_centrality, name="betweenness"), 
                    pd.Series(harmonic_centrality, name="harmonic"),
                    pd.Series(clustering, name="clustering"),
                    pd.Series(pagerank, name="pagerank")
                   ], axis=1)
    return df
    
