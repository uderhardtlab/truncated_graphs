"""
Figure 2 a-c: edge truncation effects for Delaunay and kNN graphs.

Computes (with CSV caching):
  - % new edges and MWU p-values for Delaunay graphs
  - % new edges, MWU and Fisher's exact p-values for kNN (k=5,10,15)

Saves to result_plots/fig2/:
  fig2_delaunay_trunc_1.svg  — example crop + truncation edges
  fig2_delaunay_trunc_2.svg  — edge length violin for one dataset
  fig2_edge_effects.svg      — main 5-panel summary figure
"""
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import fisher_exact, mannwhitneyu
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

plt.rcParams["svg.fonttype"] = "none"

from border_effects_kNN_del import get_mibitof, get_squidpy_visium_datasets, trunc_graphs

COORDS_PICKLE = ROOT / "mibitof_coords" / "coords.pickle"
DEL_CSV       = ROOT / "results" / "figure1" / "del_trunc.csv"
KNN_CSV       = ROOT / "results" / "figure1" / "kNN_trunc.csv"
OUT_DIR       = ROOT / "result_plots" / "fig2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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


# ── example crop visualization ─────────────────────────────────────────────────
EXAMPLE = "glioma_mibitof:Brainiaqc_R1C1_whole_cell.tiff"
sq      = (1/8, 7/8)
coords_ex       = datasets[EXAMPLE]
xlim, ylim, *_  = _roi_limits(coords_ex, sq)
out             = trunc_graphs(coords=coords_ex, return_graphs=True,
                               method="delaunay", xlim=xlim, ylim=ylim)
coords_ex = out["coords"]
G, subG, plotG, truncG = out["G"], out["subG"], out["plotG"], out["truncG"]
edge_df = out["edge_df"]

f, axs = plt.subplots(1, 2, figsize=(6, 3))
nx.draw(G,    pos=coords_ex, edge_color=(0.5, 0.5, 0.5, 0.5), node_size=1, node_color="gray", ax=axs[0])
nx.draw(subG, pos=coords_ex, node_size=1, node_color="blue",  ax=axs[0])
nx.draw_networkx_edges(plotG,  pos=coords_ex, ax=axs[1], edge_color="gray")
nx.draw(subG,                  pos=coords_ex, ax=axs[1], node_size=1, node_color="blue")
edge_colors = [e in G.edges for e in truncG.edges]
nx.draw_networkx_edges(truncG, pos=coords_ex, ax=axs[1],
                       edge_color=np.where(edge_colors, "black", "red"))
for ax in axs:
    ax.set_aspect("equal")
plt.tight_layout()
plt.savefig(OUT_DIR / "fig2_delaunay_trunc_1.svg")
plt.close()

sns.violinplot(edge_df, y="Edge length", hue="New edge", cut=0, fill=False, split=True,
               palette={False: "black", True: "red"}, hue_order=[True, False], inner=None)
plt.savefig(OUT_DIR / "fig2_delaunay_trunc_2.svg")
plt.close()


# ── summary figure ─────────────────────────────────────────────────────────────
f, axs = plt.subplots(1, 5, figsize=(10, 2))
sns.violinplot(all_results_del, y="% new edges",    ax=axs[0], legend=False, fill=False, color="purple")
sns.violinplot(all_results_del, y="P_adj",          ax=axs[1], legend=False, fill=False, color="purple")
sns.violinplot(all_results_knn, hue="k", y="% new edges",    ax=axs[2], legend=False, fill=False, color="green")
sns.violinplot(all_results_knn, hue="k", y="P_MWU_adj",      ax=axs[3], legend=False, fill=False, color="green")
sns.violinplot(all_results_knn, hue="k", y="P_fisher_adj",   ax=axs[4], legend=True,  fill=False, color="green")
axs[0].set_title("$\\mathbf{d}$ % of edges that\nwere not contained\nin global graph")
axs[1].set_title("$\\mathbf{e}$ Adjusted MWU P-values\nof original vs. new\nedge length")
axs[2].set_title("$\\mathbf{f}$ % of edges that\nwere not contained\nin global graph")
axs[3].set_title("$\\mathbf{g}$ Adjusted MWU P-values\nof original vs. new\nedge length")
axs[4].set_title("$\\mathbf{h}$ Adjusted Fisher's exact\nP-values of original vs.\nnew edge symmetry")
plt.tight_layout()
plt.savefig(OUT_DIR / "fig2_edge_effects.svg", bbox_inches="tight")
plt.close()

print(f"Delaunay % new edges median: {all_results_del['% new edges'].median():.4f}")
for k, sub in all_results_knn.groupby("k"):
    print(f"kNN k={k}: max={sub['% new edges'].max():.4f}, median={sub['% new edges'].median():.4f}")
