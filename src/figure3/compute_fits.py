"""
Figure 3 data pipeline: run BOSPERRUS fits on all MIBI-TOF datasets.

For each graph type, fits piecewise-linear / exponential-saturation /
Michaelis-Menten curves of centrality vs. border distance and writes
per-dataset fit-quality metrics to results/figure3/{graph_type}_graph_level_fits.csv.
Results are cached; delete a CSV to force recomputation.

Run:
    cd truncated_graphs/
    uv run python src/figure3/compute_fits.py
"""
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bosperrus import Flow
from bosperrus.distances import distance_to_convex_hull

MEASURES    = ["degree", "pagerank", "betweenness", "closeness", "clustering"]
GRAPH_TYPES = [
    "delaunay",
    "knn_k=5", "knn_k=10", "knn_k=15",
    "rnn_r=0.02", "rnn_r=0.03", "rnn_r=0.04", "rnn_r=0.05",
]
COORDS_PICKLE = ROOT / "mibitof_coords" / "coords.pickle"
RESULTS_DIR   = ROOT / "results" / "figure3"


def _graph_kwargs(graph_type):
    parts = graph_type.split("_")
    if parts[0] == "delaunay":
        return None
    key, val = parts[1][0], parts[1].split("=")[-1]
    return {key: int(val) if key != "r" else float(val)}


def fit_dataset(dataset, coordinates, graph_type):
    coords = coordinates.copy()
    lo, hi = coords.min(axis=0), coords.max(axis=0)
    coords = (coords - lo) / (hi - lo)
    try:
        flow = Flow.from_coords(
            coordinates=coords,
            distance_fn=distance_to_convex_hull,
            measures=MEASURES,
            graph_type=graph_type.split("_")[0],
            graph_kwargs=_graph_kwargs(graph_type),
        )
        flow.flow()
    except ValueError as e:
        print(f"  skip {dataset}: {e}")
        return pd.DataFrame()
    result = flow.fit_quality.T
    result["dataset"]   = dataset
    result["num_edges"] = len(flow._edge_list)
    result["num_nodes"] = len(coordinates)
    return result


def compute_fits(datasets, graph_type):
    out_csv = RESULTS_DIR / f"{graph_type}_graph_level_fits.csv"
    if out_csv.exists():
        return pd.read_csv(out_csv)
    print(f"computing fits: {graph_type} …")
    dfs = []
    with ProcessPoolExecutor() as ex:
        futures = {ex.submit(fit_dataset, ds, datasets[ds], graph_type): ds
                   for ds in datasets}
        for fut in tqdm(as_completed(futures), total=len(futures)):
            dfs.append(fut.result())
    result = pd.concat(dfs)
    result.to_csv(out_csv)
    return result


def main(graph_types=GRAPH_TYPES, coords_pickle=COORDS_PICKLE):
    with open(coords_pickle, "rb") as f:
        datasets = pickle.load(f)
    datasets.pop("glioma_mibitof:CHOP_907_R1C6_whole_cell.tiff", None)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for gt in graph_types:
        df = compute_fits(datasets, gt)
        df["graph_type"] = gt
        frames.append(df)
    return pd.concat(frames)


if __name__ == "__main__":
    main()
