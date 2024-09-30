import numpy as np
import anndata as ad
import squidpy as sq
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics import mean_absolute_error
import pandas as pd

def generate_coordinates(n, bounds=(0, 100)):
    """
    Generate random 2D spatial coordinates.
    :param n: Number of points
    :param bounds: Tuple specifying the (min, max) bounds for the coordinates
    :return: 2D array of coordinates
    """
    return np.random.uniform(bounds[0], bounds[1], size=(n, 2))

def create_anndata(coordinates, n_neighs=6):
    """
    Create AnnData object and compute spatial neighbors.
    :param coordinates: 2D array of coordinates
    :param n_neighs: Number of nearest neighbors to compute
    :return: AnnData object with computed neighbors
    """
    adata = ad.AnnData(X=np.zeros((coordinates.shape[0], 1)))  # Empty X matrix
    adata.obsm['spatial'] = coordinates
    sq.gr.spatial_neighbors(adata, coord_type="grid", n_neighs=n_neighs)
    return adata

def filter_border_nodes(adata, border=5, bounds=(0, 100)):
    """
    Filter nodes that lie within a border of the image.
    :param adata: AnnData object containing spatial coordinates
    :param border: Number of pixels to filter at the borders
    :param bounds: Tuple specifying the (min, max) bounds for the coordinates
    :return: Filtered AnnData object
    """
    coordinates = adata.obsm['spatial']
    mask = np.all((coordinates > bounds[0] + border) & (coordinates < bounds[1] - border), axis=1)
    return adata[mask].copy()

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

    return {
        'degree': degree_centrality,
        'closeness': closeness_centrality,
        'betweenness': betweenness_centrality}


def compute_error(original_measures, truncated_measures, adata_original, adata_truncated):
    """
    Compute error between centrality measures of original and truncated graphs.
    Only compares centrality measures for common nodes (i.e., nodes that exist in both graphs).
    :param original_measures: Centrality measures from the original graph
    :param truncated_measures: Centrality measures from the truncated graph
    :param adata_original: AnnData object of the original graph
    :param adata_truncated: AnnData object of the truncated graph
    :return: Dictionary of mean absolute errors for each centrality measure
    """
    # Get the indices of common nodes between original and truncated graphs
    original_coords = adata_original.obsm['spatial']
    truncated_coords = adata_truncated.obsm['spatial']

    # Find common node indices
    common_node_mask = np.isin(original_coords, truncated_coords).all(axis=1)
    
    errors = {}
    for key in original_measures:
        # Get centrality values for common nodes only
        original_common = np.array([original_measures[key][i] for i in range(len(original_measures[key])) if common_node_mask[i]])
        truncated_common = np.array(list(truncated_measures[key].values()))

        # Compute mean absolute error
        errors[key] = mean_absolute_error(original_common, truncated_common)
    return errors


def visualize_centrality(adata, centrality, title="Centrality Visualization"):
    """
    Visualize centrality measures on a scatter plot.
    :param adata: AnnData object with spatial coordinates
    :param centrality: Centrality measure to visualize (e.g., 'degree')
    :param title: Plot title
    """
    coordinates = adata.obsm['spatial']
    centrality_values = list(centrality.values())
    node_size = [1000 * c for c in centrality_values]  # Scale node size by centrality

    plt.figure(figsize=(8, 8))
    plt.scatter(coordinates[:, 0], coordinates[:, 1], s=node_size, c=centrality_values, cmap='coolwarm')
    
    # Draw edges based on neighbors
    connectivity_matrix = adata.obsp['spatial_connectivities']
    graph = nx.Graph(connectivity_matrix)
    for i, (x, y) in enumerate(coordinates):
        neighbors = graph.neighbors(i)
        for neighbor in neighbors:
            x_neigh, y_neigh = coordinates[neighbor]
            plt.plot([x, x_neigh], [y, y_neigh], color="gray", lw=0.5)

    plt.colorbar(label=title)
    plt.title(title)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.show()


def compute_node_errors_and_distances(original_measures, truncated_measures, adata_original, adata_truncated, measure, bounds=(0, 100)):
    """
    Compute error per node and distance to the closest border for the original and truncated graphs.
    :param original_measures: Centrality measures from the original graph
    :param truncated_measures: Centrality measures from the truncated graph
    :param adata_original: AnnData object of the original graph
    :param adata_truncated: AnnData object of the truncated graph
    :param bounds: Tuple specifying the (min, max) bounds for the coordinates
    :return: DataFrame containing errors and distances for each node
    """

    assert measure in ["degree", "closeness", "betweenness"], "Please choose between degree, closeness, or betweenness"
    # Get coordinates for original and truncated graphs
    original_coords = adata_original.obsm['spatial']
    truncated_coords = adata_truncated.obsm['spatial']

    # Calculate distances to the closest border for original coordinates
    distances_to_borders = np.minimum(
        np.minimum(original_coords[:, 0], bounds[1] - original_coords[:, 0]),
        np.minimum(original_coords[:, 1], bounds[1] - original_coords[:, 1])
    )

    # Prepare a list to store errors per node
    node_errors = []

    # Iterate through original nodes and calculate errors for common nodes
    for i in range(len(original_measures[measure])):
        if i < len(truncated_coords):  # Check if the node is in truncated data
            error = abs(original_measures[measure][i] - truncated_measures[measure].get(i, 0))
            node_errors.append((i, error, distances_to_borders[i]))

    return node_errors



def prepare_3d_scatter_data(error_dfs):
    """
    Prepare data for 3D scatter plot.
    :param error_dfs: Dictionary of error DataFrames for different borders
    :return: Combined DataFrame suitable for 3D plotting
    """
    combined_data = []
    for border, df in error_dfs.items():
        for _, row in df.iterrows():
            combined_data.append((row['Distance to Closest Border'], row['Error'], border))
    
    return pd.DataFrame(combined_data, columns=['Distance to Closest Border', 'Error', 'Border'])
