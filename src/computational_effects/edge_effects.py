"""
Edge truncation effects for Delaunay and kNN graphs (computational effects section).

Computes, for every MIBI-TOF / Visium dataset, a crop to the central 3/4 of the image and compares the graph
built on the crop with the global graph restricted to it:
  - Delaunay: % new edges, Mann-Whitney p of original vs. new edge lengths (BH-adjusted)
  - kNN (k = 5, 10, 15): % new edges, Mann-Whitney and Fisher's exact (edge symmetry) p-values (BH-adjusted)

Writes results/computational_effects/edge_effects/{del_trunc,kNN_trunc}.csv; an existing CSV is reused, so
delete it to recompute. Plotted by figure_plotting/computational_effects/edge_effects.ipynb.

Run:
    cd truncated_graphs/
    pixi run python src/computational_effects/edge_effects.py
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))  # border_effects_kNN_del.py lives next to this script

from border_effects_kNN_del import get_mibitof, get_squidpy_visium_datasets, trunc_graphs  # noqa: E402

COORDS_PICKLE = ROOT / "mibitof_coords" / "coords.pickle"
DEL_CSV       = ROOT / "results" / "computational_effects" / "edge_effects" / "del_trunc.csv"
KNN_CSV       = ROOT / "results" / "computational_effects" / "edge_effects" / "kNN_trunc.csv"
DEL_CSV.parent.mkdir(parents=True, exist_ok=True)

np.random.seed(41)


# ── datasets ───────────────────────────────────────────────────────────────────
try:
    with open(COORDS_PICKLE, "rb") as f:
        datasets = pickle.load(f)
except FileNotFoundError:
    datasets = get_mibitof()
    datasets.update(get_squidpy_visium_datasets())
    with open(COORDS_PICKLE, "wb") as f:
        pickle.dump(datasets, f)


def _roi_limits(coords, square_limits):
    w = np.max(coords[:, 0]) - np.min(coords[:, 0])
    h = np.max(coords[:, 1]) - np.min(coords[:, 1])
    xlim = (np.min(coords[:, 0]) + w * square_limits[0],
            np.min(coords[:, 0]) + w * square_limits[1])
    ylim = (np.min(coords[:, 1]) + h * square_limits[0],
            np.min(coords[:, 1]) + h * square_limits[1])
    return xlim, ylim, w, h


# ── Delaunay edge statistics ───────────────────────────────────────────────────
try:
    all_results_del = pd.read_csv(DEL_CSV)
except FileNotFoundError:
    sq = (1/8, 7/8)
    df = pd.DataFrame(columns=["data_source", "node density in ROI", "% new edges", "P", "border_size"])
    for dataset in tqdm(datasets, desc="Delaunay"):
        coords = datasets[dataset]
        xlim, ylim, w, h = _roi_limits(coords, sq)
        try:
            out     = trunc_graphs(coords=coords, return_graphs=False,
                                   method="delaunay", xlim=xlim, ylim=ylim)
            edge_df = out["edge_df"]
            old = edge_df.loc[~edge_df["New edge"], "Edge length"]
            new = edge_df.loc[ edge_df["New edge"], "Edge length"]
            p   = mannwhitneyu(old, new, alternative="two-sided").pvalue
            df.loc[dataset] = [dataset.split(":")[0],
                               out["#nodes in ROI"] / (h * w),
                               len(new) / (len(new) + len(old)), p, sq[0]]
        except Exception:
            continue
    df["% new edges"] *= 100
    _, df["P_adj"], _, _ = multipletests(df["P"], alpha=0.05, method="fdr_bh")
    df.to_csv(DEL_CSV)
    all_results_del = df


# ── kNN edge statistics ────────────────────────────────────────────────────────
try:
    all_results_knn = pd.read_csv(KNN_CSV)
except FileNotFoundError:
    sq   = (1/8, 7/8)
    dfs  = []
    for k in [5, 10, 15]:
        df = pd.DataFrame(columns=["data_source", "#nodes in ROI", "node density in ROI",
                                    "% new edges", "P_MWU", "P_fisher",
                                    "border_size", "k", "img_size"])
        for dataset in tqdm(datasets, desc=f"kNN k={k}"):
            coords = datasets[dataset]
            xlim, ylim, w, h = _roi_limits(coords, sq)
            try:
                out     = trunc_graphs(coords=coords, return_graphs=False,
                                       method="kNN", xlim=xlim, ylim=ylim, k=k)
                edge_df = out["edge_df"]
                old = edge_df.loc[~edge_df["New edge"], "Edge length"]
                new = edge_df.loc[ edge_df["New edge"], "Edge length"]
                p_mwu  = mannwhitneyu(old, new, alternative="two-sided").pvalue
                cont   = pd.crosstab(edge_df["New edge"], edge_df["Symmetry"])
                vals   = np.array([[cont.loc[True,  False], cont.loc[True,  True]],
                                   [cont.loc[False, False], cont.loc[False, True]]])
                _, p_fisher = fisher_exact(vals, alternative="greater")
                df.loc[dataset] = [dataset.split(":")[0], out["#nodes in ROI"],
                                   out["#nodes in ROI"] / (h * w),
                                   len(new) / (len(new) + len(old)),
                                   p_mwu, p_fisher, sq[0], k, (h, w)]
            except Exception:
                continue
        dfs.append(df)
    all_results_knn = pd.concat(dfs)
    all_results_knn["% new edges"] *= 100
    _, all_results_knn["P_MWU_adj"],    _, _ = multipletests(
        all_results_knn["P_MWU"],    alpha=0.05, method="fdr_bh")
    _, all_results_knn["P_fisher_adj"], _, _ = multipletests(
        all_results_knn["P_fisher"], alpha=0.05, method="fdr_bh")
    all_results_knn.to_csv(KNN_CSV)


print(f"Delaunay % new edges median: {all_results_del['% new edges'].median():.4f}")
for k, sub in all_results_knn.groupby("k"):
    print(f"kNN k={k}: max={sub['% new edges'].max():.4f}, median={sub['% new edges'].median():.4f}")
