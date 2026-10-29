"""
Figure 3 column examples: per-cell scores of one MIBI-TOF image per measure,
for the maps drawn above each measure column of plot_metrics.py.

The graph type per measure follows the text (NARRATIVE): the usual drop at the
border (degree, rNN), the denser one-sided neighbourhoods at a fixed out-degree
(clustering, kNN), the reversed effect of long Delaunay hull edges (closeness),
and the deep reach of a global measure (betweenness, rNN).

The image per measure shows a strong but typical effect: among images with at
least MIN_CELLS cells whose best fit is not Constant and whose effect has the
sign of the cell median, the one closest to the EFFECT_QUANTILE of
|observed_effect_strength| (read from the CSVs written by compute_fits.py).

Writes results/computational_effects/mibitof_fits/figure3_examples.csv
(long format: dataset, node, x, y, measure, graph_type, score).

Run:
    cd truncated_graphs/
    pixi run python src/computational_effects/example_scores.py
"""
import numpy as np
import pandas as pd

from bosperrus import Flow
from bosperrus.distances import distance_to_convex_hull
from compute_fits import RESULTS_DIR, _graph_kwargs, load_datasets, normalise_coords

NARRATIVE = {
    "degree": "rnn_k=10",
    "betweenness": "rnn_k=10",
    "closeness": "delaunay",
    "clustering": "knn_k=10",
}
MIN_CELLS = 2000  # sparser images make poor maps
EFFECT_QUANTILE = 0.9
OUT_CSV = RESULTS_DIR / "figure3_examples.csv"


def pick_example(measure, graph_type):
    df = pd.read_csv(RESULTS_DIR / f"{graph_type}_graph_level_fits.csv", index_col=0).loc[measure].set_index("dataset")
    effect = df["observed_effect_strength"].where(df["best_fit_type"] != "Constant Fit", 0)
    sign = np.sign(effect.median())
    pool = df[(df["num_nodes"] >= MIN_CELLS) & (np.sign(effect) == sign)]
    target = effect.abs().quantile(EFFECT_QUANTILE)
    return (pool["observed_effect_strength"].abs() - target).abs().idxmin()


def main():
    datasets = load_datasets()
    frames = []
    for measure, gt in NARRATIVE.items():
        dataset = pick_example(measure, gt)
        coords = normalise_coords(datasets[dataset])
        flow = Flow.from_coords(
            coordinates=coords,
            distance_fn=distance_to_convex_hull,
            measures=[measure],
            graph_type=gt.split("_")[0],
            graph_kwargs=_graph_kwargs(gt, coords),
        )
        frames.append(pd.DataFrame({"dataset": dataset, "node": range(len(coords)), "x": coords[:, 0], "y": coords[:, 1],
                                    "measure": measure, "graph_type": gt, "score": flow.observations[measure].values}))
        print(f"{measure}: {gt}, {dataset}")
    pd.concat(frames).to_csv(OUT_CSV, index=False)
    print(f"Saved {OUT_CSV}")


if __name__ == "__main__":
    main()
