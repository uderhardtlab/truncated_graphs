import osmnx as ox
import geopandas as gpd
import numpy as np
import pandas as pd
import bosperrus
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from shapely.ops import unary_union
import networkx as nx

ROOT = Path(__file__).resolve().parent.parent.parent
OUT  = ROOT / "result_plots"
OUT.mkdir(exist_ok=True)
CACHE = ROOT / "notebooks" / "cache"
CACHE.mkdir(exist_ok=True)

# 1. Download road networks for Northumberland (English side) and
#    Scottish Borders (Scottish side) — cache each
def load_or_download(place, fname, network_type="drive"):
    p = CACHE / fname
    if p.exists():
        print(f"Loading cached {fname}")
        return ox.load_graphml(p)
    print(f"Downloading graph for {place} ...")
    G = ox.graph_from_place(place, network_type=network_type)
    ox.save_graphml(G, p)
    return G

G_nb = load_or_download("Northumberland, United Kingdom", "northumberland_drive.graphml")
G_sb = load_or_download("Scottish Borders, United Kingdom", "scottish_borders_drive.graphml")

# 2. Merge graphs
G = nx.compose(G_nb, G_sb)
# re-get nodes in WGS84 for geocoding then project
nodes_wgs, edges_wgs = ox.graph_to_gdfs(G)
nodes_proj = nodes_wgs.to_crs("EPSG:27700")
print(f"Total nodes before alignment: {len(nodes_proj)}")

# 3. Get England/Scotland border line
#    Use the shared boundary between the two admin polygons
# Try primary names first, fall back to variants
for nb_name in ["Northumberland, United Kingdom", "Northumberland, England, United Kingdom"]:
    try:
        nb_poly = ox.geocode_to_gdf(nb_name).to_crs("EPSG:27700")
        print(f"Geocoded Northumberland as: {nb_name}")
        break
    except Exception as e:
        print(f"Failed '{nb_name}': {e}")

for sb_name in ["Scottish Borders, United Kingdom",
                "Scottish Borders Council, United Kingdom",
                "Scottish Borders Council Area, United Kingdom"]:
    try:
        sb_poly = ox.geocode_to_gdf(sb_name).to_crs("EPSG:27700")
        print(f"Geocoded Scottish Borders as: {sb_name}")
        break
    except Exception as e:
        print(f"Failed '{sb_name}': {e}")

border_line = nb_poly.geometry.iloc[0].boundary.intersection(
              sb_poly.geometry.iloc[0].boundary)
print("border_line type:", border_line.geom_type)
print("border_line is empty:", border_line.is_empty)

# If intersection is empty or a point, fall back to unary_union of boundaries
# and take shared region via buffer approach
if border_line.is_empty or border_line.geom_type == "Point":
    print("WARNING: border_line intersection is empty/point — using buffer overlap fallback")
    nb_bound = nb_poly.geometry.iloc[0].boundary
    sb_bound = sb_poly.geometry.iloc[0].boundary
    # Buffer each boundary by 100 m, intersect, then take midline
    nb_buf = nb_bound.buffer(100)
    sb_buf = sb_bound.buffer(100)
    border_line = nb_buf.intersection(sb_buf)
    print("Fallback border_line type:", border_line.geom_type)

# 4. Distance from each node to the border line (metres, BNG)
dist_m = nodes_proj.geometry.distance(border_line)
dist_series = pd.Series(dist_m.values, index=nodes_wgs.index, name="dist_border_m")

# 5. Degree
G_u = G.to_undirected()
degree = pd.Series(dict(G_u.degree()), name="degree")

# 6. Align
common = dist_series.index.intersection(degree.index)
dist_series = dist_series.loc[common]
degree      = degree.loc[common]
scores_df   = pd.DataFrame({"degree": degree})
print(f"nodes: {len(common)}, dist range: {dist_series.min()/1000:.2f}–{dist_series.max()/1000:.2f} km")

# 7. BOSPERRUS
flow = bosperrus.Flow.from_distances_and_scores(distances=dist_series, scores=scores_df)
flow.flow(measures=["degree"])
fq = flow.fit_quality["degree"]
print("=== fit_quality ===")
print(flow.fit_quality)

# 8. Scatter + fitted model
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(dist_series / 1000, degree, s=1, alpha=0.15, rasterized=True, label="nodes")
sort_idx = dist_series.argsort()
ax.plot(dist_series.iloc[sort_idx].values / 1000,
        flow.best_fits["degree"].S_model[sort_idx],
        color="red", lw=1.5, label=f"fit: {fq['best_fit_type']}")
ax.set_xlabel("distance to E/S border (km)")
ax.set_ylabel("degree")
ax.set_title(f"England/Scotland border road network\nrel_ll={fq['scaled_relative_likelihood_over_baseline']:.4f}  effect={fq['observed_effect_strength']:.4f}")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "es_border_scatter.png", dpi=130, bbox_inches="tight")
plt.close()
print("saved es_border_scatter.png")
