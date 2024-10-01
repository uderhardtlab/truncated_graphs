import numpy as np
import anndata as ad
import squidpy as sq
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics import mean_absolute_error
import pandas as pd

def generate_coordinates(n, bounds, type):
    """
    Generate random 2D spatial coordinates.
    :param n: Number of points
    :param bounds: Tuple specifying the (min, max) bounds for the coordinates
    :return: 2D array of coordinates
    """
    assert type in ["regular", "normal"]
    if type == "normal":
        return np.random.uniform(bounds[0] + np.finfo(float).eps, bounds[1] - np.finfo(float).eps, size=(n, 2))
    elif type == "regular": 
        grid_size = int(np.ceil(np.sqrt(n)))

        # add/subtract small positive constant so that the behavior of border=0 is as expected
        x = np.linspace(bounds[0] + 1e-5, bounds[1] - 1e-5, grid_size)
        y = np.linspace(bounds[0] + 1e-5, bounds[1] - 1e-5, grid_size)
        xx, yy = np.meshgrid(x, y)
        grid_coords = np.vstack([xx.ravel(), yy.ravel()]).T
        return grid_coords[:n]
    else:
        return None


def create_anndata(coordinates, n_neighs, bounds):
    """
    Create AnnData object and compute spatial neighbors.
    :param coordinates: 2D array of coordinates
    :param n_neighs: Number of nearest neighbors to compute
    :return: AnnData object with computed neighbors
    """
    adata = ad.AnnData(X=np.zeros((coordinates.shape[0], 1)))  # Empty X matrix
    adata.obsm['spatial'] = coordinates

    distances_to_border = np.minimum(
        np.minimum(coordinates[:, 0], bounds[1] - coordinates[:, 0]),
        np.minimum(coordinates[:, 1], bounds[1] - coordinates[:, 1])
    )

    adata.obs["distance_to_border"] = distances_to_border

    sorted_adata = adata[np.argsort(distances_to_border)].copy()
    sorted_adata.obs_names = np.array(range(len(adata))).astype(str)
    
    sq.gr.spatial_neighbors(sorted_adata, coord_type="grid", n_neighs=n_neighs)
    return sorted_adata


def filter_border_nodes(adata, border, bounds):
    """
    Filter nodes that lie within a border of the image and return the filtered AnnData with original indices.
    :param adata: AnnData object containing spatial coordinates
    :param border: Number of pixels to filter at the borders
    :param bounds: Tuple specifying the (min, max) bounds for the coordinates
    :return: Filtered AnnData object
    """
    coordinates = adata.obsm['spatial']
    mask = np.all((coordinates > bounds[0] + border) & (coordinates < bounds[1] - border), axis=1)
    return adata[mask].copy()


def normalize_dict(d):
    max = np.max(list(d.values()))
    return [(v / max) for k, v in d.items()]


def compute_centrality_measures(adata):
    """
    Compute centrality measures on the graph built from AnnData's spatial connectivity.
    :param adata: AnnData object with computed spatial neighbors
    :return: Dictionary of centrality measures
    """
    connectivity_matrix = adata.obsp['spatial_connectivities']
    graph = nx.Graph(connectivity_matrix)

    degree_centrality = nx.degree_centrality(graph)
    closeness_centrality = nx.closeness_centrality(graph)
    betweenness_centrality = nx.betweenness_centrality(graph)
    
    adata.obs['degree'] = normalize_dict(degree_centrality)
    adata.obs['closeness'] = normalize_dict(closeness_centrality)
    adata.obs['betweenness'] = normalize_dict(betweenness_centrality)
    return
    

def compute_node_errors_and_distances(adata_original, adata_truncated, bounds):    
    original_coords = adata_original.obsm['spatial']
    adata_truncated.obs["distance_to_border"] = adata_original.obs['distance_to_border'][np.array(adata_truncated.obs_names).astype(int)]

    for measure in ["degree", "closeness", "betweenness"]:
        adata_truncated.obs[f"{measure} error"] = adata_original.obs[measure].iloc[np.array(adata_truncated.obs_names).astype(int)] - adata_truncated.obs[measure]
    return 


def plot_graphs_with_errors(adata_original, adata_truncated, border, measure, bounds):
    """
    Plot original and truncated graphs with node sizes based on centrality and colors based on errors.
    :param adata_original: AnnData object of the original graph
    :param centrality_original: Centrality measures from the original graph
    :param adatas_truncated: Dictionary of truncated AnnData objects
    :param centralities_truncated: Dictionary of centrality measures for truncated graphs
    :param border: Border size used for truncation
    :param measure: Centrality measure to use for plotting (degree, closeness, or betweenness)
    :param bounds: Tuple specifying the (min, max) bounds for the coordinates
    """
    # Extract original and truncated coordinates
    original_coords = adata_original.obsm['spatial']
    truncated_coords = adata_truncated.obsm['spatial']

    original_sizes = adata_original.obs[measure].values * 100  # Scale sizes for visibility
    truncated_sizes = adata_truncated.obs[measure].values * 100  # Scale sizes

    error_colors = adata_truncated.obs[f'{measure} error'].values 
    
    plt.figure(figsize=(10, 8))
    plt.scatter(original_coords[:, 0], original_coords[:, 1], s=original_sizes, c='grey', alpha=0.6, label='Original Graph', edgecolors='k')
    scatter = plt.scatter(truncated_coords[:, 0], truncated_coords[:, 1], s=truncated_sizes, c=error_colors, cmap='YlOrRd', alpha=0.8, label='Truncated Graph', edgecolors='none')

    cbar = plt.colorbar(scatter)
    cbar.set_label('Error')
    
    # Draw rectangles for borders
    plt.gca().add_patch(plt.Rectangle((bounds[0] + border, bounds[0] + border), bounds[1] - 2 * border, bounds[1] - 2 * border, linewidth=2, edgecolor='none', facecolor='none', label='Borders'))

    # Set axis limits and labels
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f'Original and Truncated Graphs with {measure.capitalize()} Centrality and Error')
    plt.legend()
    plt.grid(True)
    
    plt.show()
