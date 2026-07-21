import numpy as np
from scipy.spatial import ConvexHull
from statsmodels.stats.proportion import proportion_confint
import networkx as nx
import pandas as pd


def point_to_segment_dist(p, a, b):
    """
    Euclidean distance from point p to line segment ab.
    p, a, b are NumPy arrays of shape (2,)
    """
    ap = p - a
    ab = b - a
    denom = np.dot(ab, ab)
    if denom == 0:
        return np.linalg.norm(ap)
    t = np.clip(np.dot(ap, ab) / denom, 0, 1)
    projection = a + t * ab
    return np.linalg.norm(p - projection)


def distance_to_convex_hull(G):
    positions = np.array([G.nodes[n]["pos"] for n in G.nodes()])
    node_list = list(G.nodes())
    
    hull = ConvexHull(positions)
    hull_points = positions[hull.vertices]
    
    distances = {}
    
    for node, pos in zip(node_list, positions):
        d = float("inf")
        for i in range(len(hull_points)):
            a = hull_points[i]
            b = hull_points[(i+1) % len(hull_points)]
            d = min(d, point_to_segment_dist(pos, a, b))
        distances[node] = d
    
    for node in G.nodes():
        G.nodes[node]["dist_to_border"] = distances[node]


def distance_to_image_border(G):
    positions = np.array([G.nodes[n]["pos"] for n in G.nodes()])
    node_list = list(G.nodes())

    x = positions[:, 0]
    y = positions[:, 1]

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    for node, (xi, yi) in zip(node_list, positions):
        d = min(
            xi - x_min,   # left border
            x_max - xi,   # right border
            yi - y_min,   # bottom border
            y_max - yi    # top border
        )
        G.nodes[node]["dist_to_border"] = d


def remove_lines(G):
    line_nodes = [n for n, d in G.degree if d == 2]
    lines = G.subgraph(line_nodes).copy()
    
    ccs = nx.connected_components(lines)
    line_degrees = lines.degree
    
    newG = nx.Graph()
    for cc in ccs:
        connect = list()
        for n in cc:
            if line_degrees[n] == 1: # nodes who had a neighbor that was not a line
                neighbors = nx.neighbors(G, n)
                for n in neighbors:
                    if not n in cc:
                        connect.append(n)
                        newG.add_node(n, **G.nodes[n])
        if len(connect) == 2:
            if connect[0] != connect[1]:
                newG.add_edge(*connect, length=len(cc))    
    return newG


def assign_avg_lengths(G):
    avg_edge_length_per_node = {}
    
    for node in G.nodes:
        lengths = [
            data["length"]
            for _, _, data in G.edges(node, data=True)
            if "length" in data
        ]
        
        avg_edge_length_per_node[node] = (
            sum(lengths) / len(lengths) if lengths else np.nan
        )
    nx.set_node_attributes(G, avg_edge_length_per_node, "avg_edge_length")
    return avg_edge_length_per_node


def get_distance_df(G):
    # Ensure avg_edge_length is computed
    if "avg_edge_length" not in next(iter(G.nodes(data=True)))[1]:
        assign_avg_lengths(G)

    df = pd.DataFrame(
        nx.get_node_attributes(G, "pos"),
        index=["x", "y"]
    ).T

    df["degree"] = pd.Series(dict(G.degree))
    df["distance_to_border"] = pd.Series(
        nx.get_node_attributes(G, "dist_to_border")
    )
    df["avg_edge_length"] = pd.Series(
        nx.get_node_attributes(G, "avg_edge_length")
    )

    return df


def bin_quantiles(df, num_quantiles=21):
    quantiles = np.linspace(0, 1, num_quantiles) 
    edges = df["distance_to_border"].quantile(quantiles).values 
    df["dist_bin"] = pd.cut( df["distance_to_border"], bins=edges, include_lowest=True, duplicates="drop" ) 
    df["bin_id"] = df["dist_bin"].cat.codes 
    return df


def bin_linspace(df, num_bins=20):
    d = df["distance_to_border"]

    if d.nunique() == 1:
        df["bin_id"] = 0
        df["dist_bin"] = pd.IntervalIndex.from_tuples([(d.iloc[0], d.iloc[0])])
        return df

    edges = np.linspace(d.min(), d.max(), num_bins + 1)

    df["dist_bin"] = pd.cut(
        d,
        bins=np.unique(edges),
        include_lowest=True
    )
    df["bin_id"] = df["dist_bin"].cat.codes

    return df

    
def get_bep_densities(df): 
    df["is_endpoint"] = (df["degree"] == 1).astype(int) 
    df["is_branch"] = (df["degree"] >= 3).astype(int) 

    summary = (
    df.groupby("bin_id")
          .agg(
              n_nodes=("degree", "size"),
              endpoint_frac=("is_endpoint", "mean"),
              branch_frac=("is_branch", "mean"),
              mean_distance=("distance_to_border", "mean"),
              mean_edge_length=("avg_edge_length", "mean"),
          )
          .reset_index()
    )
    
    """
    ep_ci = [proportion_confint(count=int(round(f * n)), nobs=n, method="wilson") 
             for f, n in zip(summary["endpoint_frac"], summary["n_nodes"])] 
    summary[["endpoint_ci_low", "endpoint_ci_high"]] = ep_ci 
    
    ep_ci = [ proportion_confint( count=int(round(f * n)), nobs=n, method="wilson" ) 
              for f, n in zip( summary["branch_frac"], summary["n_nodes"] ) ] 
    summary[["branchpoint_ci_low", "branchpoint_ci_high"]] = ep_ci 
    """
    return summary


def add_2d_coords(G):
    for n in G.nodes:
        G.nodes[n]["pos"] = G.nodes[n]["v_coords"][1:]
    return G