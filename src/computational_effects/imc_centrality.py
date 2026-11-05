"""
Figure 3 bottom row: per-cell-type centrality on squidpy's IMC demo FOV
(Jackson et al. 2020; sq.datasets.imc, the example of Palla et al. 2022,
Suppl. Fig. 2), raw vs. BOSPERRUS-corrected.

Graph: squidpy's default `sq.gr.spatial_neighbors` (kNN, k=6 on generic
coordinates). squidpy's centrality_scores builds `nx.Graph(A)`, which keeps an
edge if either direction exists, so the per-cell scores use the same undirected
graph. squidpy's degree and closeness are group measures that can't be
corrected per cell, so the plot compares per-cell means per cell type instead
(squidpy's average_clustering is exactly that mean).

Writes results/computational_effects/mibitof_fits/imc_centrality.csv
(one row per cell: cell_type, its squidpy colour, x, y, d = distance to convex
hull, and raw and corrected degree, clustering and closeness).

Run:
    cd truncated_graphs/
    pixi run python src/computational_effects/imc_centrality.py
"""
from pathlib import Path

import pandas as pd
import scanpy as sc
import squidpy as sq
from scipy.sparse import triu

from bosperrus import Flow
from bosperrus.centrality_measures import compute_centrality_measures
from bosperrus.distances import distance_to_convex_hull

ROOT = Path(__file__).resolve().parent.parent.parent
IMC_H5AD = ROOT / "data" / "anndata" / "imc.h5ad"  # sq.datasets.imc(path=...)
OUT_CSV = ROOT / "results" / "computational_effects" / "mibitof_fits" / "imc_centrality.csv"
MEASURES = ["degree", "clustering", "closeness"]


def main():
    adata = sc.read_h5ad(IMC_H5AD)
    sq.gr.spatial_neighbors(adata)  # squidpy default, as in the tutorial
    A = adata.obsp["spatial_connectivities"]
    rows, cols = triu(A + A.T, 1).nonzero()  # undirected, as squidpy's nx.Graph(A)
    edges = {frozenset((int(i), int(j))) for i, j in zip(rows, cols)}
    scores = compute_centrality_measures(edges, N=adata.n_obs, measures=MEASURES)
    coords = adata.obsm["spatial"]
    d = distance_to_convex_hull(coords)
    flow = Flow.from_distances_and_scores(distances=d, scores=scores)
    flow.flow()

    colors = dict(zip(adata.obs["cell type"].cat.categories, adata.uns["cell type_colors"]))
    out = pd.DataFrame({"cell_type": adata.obs["cell type"].astype(str).values,
                        "color": adata.obs["cell type"].map(colors).astype(str).values,
                        "x": coords[:, 0], "y": coords[:, 1], "d": d.values})
    for m in MEASURES:
        out[f"{m}_raw"] = flow.observations[m].values
        out[f"{m}_corrected"] = flow.observations[f"BOSPERRUS corrected {m}"].values
    out.to_csv(OUT_CSV, index=False)
    print(f"Saved {OUT_CSV}: " + ", ".join(f"{m} {flow.best_fits[m].name} "
                                           f"{flow.best_fits[m].observed_effect_strength:+.2f}" for m in MEASURES))


if __name__ == "__main__":
    main()
