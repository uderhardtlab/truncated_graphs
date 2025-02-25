import numpy as np
import anndata as ad
import squidpy as sq
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics import mean_absolute_error
import pandas as pd
from scipy import stats
from scipy.spatial import distance


def get_mibitof(path):    
    real_data = sq.datasets.mibitof(path=path)
    bounds = (0, np.max(real_data.obsm["spatial"][:, 0]))

    real_data.obsp["spatial_connectivities"] = real_data.obsp["connectivities"].toarray()
    del real_data.obsp["connectivities"]
    del real_data.var
    del real_data.obsm["X_scanorama"]
    del real_data.obsm["X_umap"]
    del real_data.obs
    del real_data.uns
    bounds = (0, np.max(real_data.obsm["spatial"][:, 0]))
    calculate_distance_to_border(real_data, bounds)
    return real_data, bounds



def fully_process(adata_original, bounds, borders):
    compute_centrality_measures(adata_original)
    
    error_dfs = dict()
    adatas_truncated = dict()
    centralities_truncated = dict()
    
    for border in borders:
        adatas_truncated[border] = filter_border_nodes(adata_original, border=border, bounds=bounds)
        compute_centrality_measures(adatas_truncated[border])
        compute_node_distances(adata_original, 
                                          adata_truncated=adatas_truncated[border])
        compute_errors(adata_original=adata_original, 
                                          adata_truncated=adatas_truncated[border])
    
        compute_pearson_correlation(adata_original=adata_original, adata_truncated=adatas_truncated[border])
        compute_kendalls_tau(adata_original=adata_original, adata_truncated=adatas_truncated[border])
        compute_cosine_similarity(adata_original=adata_original, adata_truncated=adatas_truncated[border])

    return adata_original, adatas_truncated
    

def generate_coordinates(n, bounds, type, hex_size=1):
    """
    Generate random or structured 2D spatial coordinates.
    :param n: Number of points
    :param bounds: Tuple specifying the (min, max) bounds for the coordinates
    :param type: Type of grid - "regular", "normal", or "hexagonal"
    :param hex_size: Size of the hexagons (only used if type is "hexagonal")
    :return: 2D array of coordinates
    """
    assert type in ["regular", "normal", "hexagonal"], "type must be 'regular', 'normal', or 'hexagonal'"

    if type == "normal":
        # Randomly generate points with a uniform distribution within bounds
        return np.random.uniform(bounds[0] + np.finfo(float).eps, bounds[1] - np.finfo(float).eps, size=(n, 2))

    elif type == "regular":
        # Generate a regular grid
        grid_size = int(np.ceil(np.sqrt(n)))
        x = np.linspace(bounds[0] + 1e-5, bounds[1] - 1e-5, grid_size)
        y = np.linspace(bounds[0] + 1e-5, bounds[1] - 1e-5, grid_size)
        xx, yy = np.meshgrid(x, y)
        grid_coords = np.vstack([xx.ravel(), yy.ravel()]).T
        return grid_coords[:n]
    
    elif type == "hexagonal":
        # Generate a hexagonal grid
        coords = []
        dx = hex_size * 3/2  # horizontal distance between hexagons
        dy = hex_size * np.sqrt(3)  # vertical distance between hexagons
        cols = int(np.ceil((bounds[1] - bounds[0]) / dx))
        rows = int(np.ceil((bounds[1] - bounds[0]) / dy))

        for row in range(rows):
            for col in range(cols):
                x = bounds[0] + col * dx
                y = bounds[0] + row * dy

                # Offset every other row by half the horizontal distance (to stagger them)
                if col % 2 == 1:
                    y += dy / 2

                # Ensure the coordinates are within bounds
                if bounds[0] <= x <= bounds[1] and bounds[0] <= y <= bounds[1]:
                    coords.append([x, y])

                if len(coords) >= n:
                    return np.array(coords)

        return np.array(coords[:n])

    else:
        return None


def calculate_distance_to_border(adata, bounds):
    coordinates = adata.obsm["spatial"]
    distances_to_border = np.minimum(
        np.minimum(coordinates[:, 0], bounds[1] - coordinates[:, 0]),
        np.minimum(coordinates[:, 1], bounds[1] - coordinates[:, 1])
    )
    adata.obs["distance_to_border"] = distances_to_border


def create_anndata(coordinates, n_neighs, bounds):
    """
    Create AnnData object and compute spatial neighbors.
    :param coordinates: 2D array of coordinates
    :param n_neighs: Number of nearest neighbors to compute
    :return: AnnData object with computed neighbors
    """
    adata = ad.AnnData(X=np.zeros((coordinates.shape[0], 1)))  # Empty X matrix
    adata.obsm['spatial'] = coordinates

    calculate_distance_to_border(adata, bounds)
    distances_to_border = adata.obs["distance_to_border"] 

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
    harmonic_centrality = nx.harmonic_centrality(graph)
    
    adata.obs['degree'] = normalize_dict(degree_centrality) #list(degree_centrality.values()) 
    adata.obs['closeness'] = normalize_dict(closeness_centrality) #list(closeness_centrality.values())
    adata.obs['betweenness'] = normalize_dict(betweenness_centrality) #list(betweenness_centrality.values())
    adata.obs['harmonic'] = normalize_dict(harmonic_centrality) 
    return
    

def compute_node_distances(adata_original, adata_truncated):    
    original_coords = adata_original.obsm['spatial']
    adata_truncated.obs["distance_to_border"] = adata_original.obs['distance_to_border'][adata_truncated.obs_names]


def compute_errors(adata_original, adata_truncated, measure=None):    
    if measure:
        measures = [measure]
    else:
        measures = ["degree", "closeness", "betweenness", "harmonic"]
    
    for measure in measures:
        adata_truncated.obs[f"{measure} error"] = adata_original.obs[measure][adata_truncated.obs_names] - adata_truncated.obs[measure]


def compute_pearson_correlation(adata_original, adata_truncated, measure=None):    
    if measure:
        measures = [measure]
    else:
        measures = ["degree", "closeness", "betweenness", "harmonic"]
    
    for measure in measures:
            adata_truncated.uns[f"{measure} pearson correlation"] = stats.pearsonr(adata_original.obs[measure][adata_truncated.obs_names], adata_truncated.obs[measure]).statistic


def compute_kendalls_tau(adata_original, adata_truncated, measure=None):
    if measure:
        measures = [measure]
    else:
        measures = ["degree", "closeness", "betweenness", "harmonic"]
    
    for measure in measures:
            adata_truncated.uns[f"{measure} kendall's tau"] = stats.kendalltau(adata_original.obs[measure][adata_truncated.obs_names], adata_truncated.obs[measure]).statistic


def compute_jaccard_similarity(adata_original, adata_truncated, measure=None):
    if measure:
        measures = [measure]
    else:
        measures = ["degree", "closeness", "betweenness", "harmonic"]
    
    for measure in measures:
            adata_truncated.uns[f"{measure} jaccard distance"] = distance.jaccard(adata_original.obs[measure][adata_truncated.obs_names], adata_truncated.obs[measure])


def compute_cosine_similarity(adata_original, adata_truncated, measure=None):
    if measure:
        measures = [measure]
    else:
        measures = ["degree", "closeness", "betweenness", "harmonic"]
    
    for measure in measures:
            adata_truncated.uns[f"{measure} cosine distance"] = distance.cosine(adata_original.obs[measure][adata_truncated.obs_names], adata_truncated.obs[measure])


def draw_edges(spatial_connectivities, coords, color, alpha, label):
    graph = nx.Graph(spatial_connectivities)
    labelled = False
    for i, (x, y) in enumerate(coords):
        neighbors = graph.neighbors(i)
        for neighbor in neighbors:
            x_neigh, y_neigh = coords[neighbor]
            if not labelled:
                plt.plot([x, x_neigh], [y, y_neigh], color=color, lw=0.5, alpha=alpha, zorder=-1, label=label)
                labelled = True
            else:
                plt.plot([x, x_neigh], [y, y_neigh], color=color, lw=0.5, alpha=alpha, zorder=-1)


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
    draw_edges(spatial_connectivities=adata_original.obsp['spatial_connectivities'], coords=original_coords, color="grey", alpha=1, label="Original edges")
    draw_edges(spatial_connectivities=adata_truncated.obsp['spatial_connectivities'], coords=truncated_coords, color="black", alpha=0.7, label="Trunacted edges")
    
    plt.scatter(original_coords[:, 0], original_coords[:, 1], 
                s=original_sizes, c='grey', label=f'Original graph, size indicates {measure} centrality', 
                edgecolors='none', zorder=1)
    
    scatter = plt.scatter(truncated_coords[:, 0], truncated_coords[:, 1], 
                          s=truncated_sizes, c=error_colors, 
                          cmap='YlOrRd', 
                          label=f'Truncated graph, size indicates {measure} centrality, color indicates error', edgecolors='none', zorder=1)

    cbar = plt.colorbar(scatter)
    cbar.set_label('Error')


    plt.gca().add_patch(plt.Rectangle((bounds[0] + border, bounds[0] + border), bounds[1] - 2 * border, bounds[1] - 2 * border, linewidth=2, edgecolor='red', facecolor='none', label='Borders'))

    # Set axis limits and labels
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f'Original and Truncated Graphs with {measure.capitalize()} Centrality and Error')
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05))
    plt.tight_layout()