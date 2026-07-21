import osmnx as ox
import geopandas as gpd
import numpy as np
import pandas as pd
import bosperrus
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "result_plots"
OUT.mkdir(exist_ok=True)

# 1. Download Isle of Wight drive network (cache to graphml to avoid re-download)
cache = ROOT / "notebooks" / "cache" / "iow_drive.graphml"
if cache.exists():
    G = ox.load_graphml(cache)
else:
    G = ox.graph_from_place("Isle of Wight, United Kingdom", network_type="drive")
    ox.save_graphml(G, cache)

nodes, edges = ox.graph_to_gdfs(G)

# 2. Get coastline polygon and project both to British National Grid (EPSG:27700)
iow_poly = ox.geocode_to_gdf("Isle of Wight, United Kingdom")
iow_proj   = iow_poly.to_crs("EPSG:27700")
nodes_proj = nodes.to_crs("EPSG:27700")
coastline  = iow_proj.geometry.iloc[0].boundary   # exterior ring = coastline

# 3. Distance from each node to coastline (in metres)
dist_m = nodes_proj.geometry.distance(coastline)
dist_series = pd.Series(dist_m.values, index=nodes.index, name="dist_coast_m")

# 4. Degree in undirected graph
G_u = G.to_undirected()
degree = pd.Series(dict(G_u.degree()), name="degree")

# 5. Align
common = dist_series.index.intersection(degree.index)
dist_series = dist_series.loc[common]
degree      = degree.loc[common]
scores_df   = pd.DataFrame({"degree": degree})

print(f"nodes: {len(common)}, dist range: {dist_series.min():.0f}–{dist_series.max():.0f} m")

# 6. BOSPERRUS
flow = bosperrus.Flow.from_distances_and_scores(distances=dist_series, scores=scores_df)
flow.flow(measures=["degree"])
fq = flow.fit_quality["degree"]
print(flow.fit_quality)

# 7. Scatter plot: dist_coast vs degree + fitted model
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(dist_series, degree, s=1, alpha=0.15, rasterized=True, label="nodes")
# sort for model line
sort_idx = dist_series.argsort()
ax.plot(dist_series.iloc[sort_idx].values,
        flow.best_fits["degree"].S_model[sort_idx],
        color="red", lw=1.5, label=f"fit: {fq['best_fit_type']}")
ax.set_xlabel("distance to coast (m)")
ax.set_ylabel("degree")
ax.set_title(f"Isle of Wight road network\nrel_ll={fq['scaled_relative_likelihood_over_baseline']:.4f}  effect={fq['observed_effect_strength']:.4f}")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "iow_coast_scatter.png", dpi=130, bbox_inches="tight")
plt.close()
print("saved iow_coast_scatter.png")
