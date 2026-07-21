#!/usr/bin/env python3
"""
Full BOSPERRUS analysis of the Great Britain rail network.

Downloads the entire GB rail network via a single bounding-box query, then:
  - simplifies, makes undirected, 0-indexes nodes
  - computes global centrality (degree/betweenness/closeness/pagerank)
  - assigns nodes to England / Scotland / Wales via spatial join
  - computes dist_coast and dist_political for every node
  - runs BOSPERRUS for country-level and coastal-county-level sub-networks
  - saves two figures to result_plots/

Outputs
-------
result_plots/uk_rail_country_results.png
result_plots/uk_rail_county_coast_results.png
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import osmnx as ox
import geopandas as gpd
from shapely.ops import unary_union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import bosperrus
from bosperrus.centrality_measures import compute_centrality_measures

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT  = Path(__file__).resolve().parent.parent.parent
CACHE = ROOT / "notebooks" / "cache"
OUT   = ROOT / "result_plots"
CACHE.mkdir(parents=True, exist_ok=True)
OUT.mkdir(exist_ok=True)

MEASURES = ["degree", "betweenness", "closeness", "pagerank"]

ox.settings.requests_timeout = 600
# Increase max query area so OSMnx sends fewer sub-queries for the GB bbox.
# Default is 2.5e9 m² (~2,500 km²); set to 1e12 m² so GB fits in one chunk.
# Railway data is sparse, so a single large Overpass query is fine.
ox.settings.max_query_area_size = 1_000_000_000_000
ox.settings.overpass_settings = "[out:json][timeout:600][maxsize:2147483648]"


# ══════════════════════════════════════════════════════════════════════════════
# 1. Download / load GB rail network (single bounding-box query)
# ══════════════════════════════════════════════════════════════════════════════

cache_path = CACHE / "gb_rail.graphml"

if cache_path.exists():
    print(f"Loading cached GB rail graph from {cache_path} ...", flush=True)
    G = ox.load_graphml(cache_path)
else:
    print("Downloading GB rail network via bounding box ...", flush=True)
    # OSMnx 2.x: bbox = (left, bottom, right, top) = (west, south, east, north)
    G = ox.graph_from_bbox(
        (-8.2, 49.9, 1.9, 60.9),
        custom_filter='["railway"="rail"]',
        retain_all=True,
    )
    ox.save_graphml(G, cache_path)
    print(f"Saved to {cache_path}")

print(f"Raw graph : {G.number_of_nodes():,} nodes / {G.number_of_edges():,} edges", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Simplify → undirected → reindex
# ══════════════════════════════════════════════════════════════════════════════

print("Simplifying graph ...", flush=True)
try:
    G = ox.simplify_graph(G)
    print(f"Simplified : {G.number_of_nodes():,} nodes / {G.number_of_edges():,} edges")
except Exception as exc:
    warnings.warn(f"simplify_graph failed ({exc}); continuing without simplification.", RuntimeWarning)

G_undirected = G.to_undirected()
print(f"Undirected : {G_undirected.number_of_nodes():,} nodes / {G_undirected.number_of_edges():,} edges")

# 0-indexed node mapping
sorted_osm_ids = sorted(G_undirected.nodes())
node_idx = {osm_id: i for i, osm_id in enumerate(sorted_osm_ids)}
idx_node = {i: osm_id for i, osm_id in enumerate(sorted_osm_ids)}
N = len(sorted_osm_ids)
print(f"{N:,} nodes reindexed 0 … {N-1}")

edge_list = [(node_idx[u], node_idx[v]) for u, v in G_undirected.edges()]
print(f"{len(edge_list):,} edges in edge list")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Centrality measures (cached)
# ══════════════════════════════════════════════════════════════════════════════

scores_cache = CACHE / "uk_rail_scores.csv"

if scores_cache.exists():
    print(f"\nLoading cached centrality scores from {scores_cache} ...", flush=True)
    global_scores_osm = pd.read_csv(scores_cache, index_col=0)
    global_scores_osm.index = global_scores_osm.index.astype(int)
else:
    print(f"\nComputing centrality for {N:,} nodes ...", flush=True)
    try:
        centrality_df = compute_centrality_measures(
            edge_list=edge_list,
            N=N,
            measures=MEASURES,
        )
        print("compute_centrality_measures done.")
    except Exception as exc:
        warnings.warn(
            f"compute_centrality_measures raised: {exc}\n"
            "Falling back to raw networkx calls.",
            RuntimeWarning,
        )
        G_nx = nx.Graph()
        G_nx.add_nodes_from(range(N))
        G_nx.add_edges_from(edge_list)

        print("  degree ...", flush=True)
        deg = dict(G_nx.degree())
        print("  betweenness (slow) ...", flush=True)
        bc = nx.betweenness_centrality(G_nx)
        print("  closeness ...", flush=True)
        cc = nx.closeness_centrality(G_nx)
        print("  pagerank ...", flush=True)
        pr = nx.pagerank(G_nx)

        centrality_df = pd.DataFrame({
            "degree":      [float(deg.get(i, 0)) for i in range(N)],
            "betweenness": [bc.get(i, 0.0)        for i in range(N)],
            "closeness":   [cc.get(i, 0.0)         for i in range(N)],
            "pagerank":    [pr.get(i, 0.0)          for i in range(N)],
        })

    global_scores_osm = pd.DataFrame(
        centrality_df.values,
        index=[idx_node[i] for i in range(N)],
        columns=centrality_df.columns,
    )
    global_scores_osm.index.name = "osmid"
    global_scores_osm.to_csv(scores_cache)
    print(f"Saved centrality scores → {scores_cache}")

print("\nCentrality summary:")
print(global_scores_osm[MEASURES].describe().to_string())


# ══════════════════════════════════════════════════════════════════════════════
# 4. Node GeoDataFrame projected to EPSG:27700
# ══════════════════════════════════════════════════════════════════════════════

print("\nProjecting nodes to EPSG:27700 ...", flush=True)
nodes_wgs, _ = ox.graph_to_gdfs(G_undirected)
nodes_proj   = nodes_wgs.to_crs("EPSG:27700")
print(f"{len(nodes_proj):,} nodes projected")

# Country polygons
print("Geocoding country polygons ...", flush=True)
eng = ox.geocode_to_gdf("England, United Kingdom").to_crs("EPSG:27700")
sco = ox.geocode_to_gdf("Scotland, United Kingdom").to_crs("EPSG:27700")
wal = ox.geocode_to_gdf("Wales, United Kingdom").to_crs("EPSG:27700")

# Assign each node to a country via sjoin
countries_union = gpd.GeoDataFrame(
    pd.concat([
        eng[["geometry"]].assign(country="England"),
        sco[["geometry"]].assign(country="Scotland"),
        wal[["geometry"]].assign(country="Wales"),
    ]).reset_index(drop=True),
    crs="EPSG:27700",
)

joined = gpd.sjoin(
    nodes_proj[["geometry"]],
    countries_union,
    how="left",
    predicate="within",
)
joined = joined[~joined.index.duplicated(keep="first")]


# ══════════════════════════════════════════════════════════════════════════════
# 5. Coastline and distances
# ══════════════════════════════════════════════════════════════════════════════

print("Computing distances ...", flush=True)

coastline = unary_union([
    eng.geometry.iloc[0],
    sco.geometry.iloc[0],
    wal.geometry.iloc[0],
]).boundary

print("  dist_coast ...", flush=True)
dist_coast = nodes_proj.geometry.distance(coastline)

# Political borders
es_border = eng.geometry.iloc[0].boundary.intersection(sco.geometry.iloc[0].boundary)
ew_border = eng.geometry.iloc[0].boundary.intersection(wal.geometry.iloc[0].boundary)

print(f"  E/S border: {es_border.geom_type}  E/W border: {ew_border.geom_type}")

def _border_fallback(poly_a, poly_b, label):
    buf = poly_a.geometry.iloc[0].boundary.buffer(200).intersection(
          poly_b.geometry.iloc[0].boundary.buffer(200))
    print(f"  Fallback {label} border: {buf.geom_type}")
    return buf

if es_border.is_empty or es_border.geom_type == "Point":
    warnings.warn("E/S border empty/point — using 200 m buffer fallback", RuntimeWarning)
    es_border = _border_fallback(eng, sco, "E/S")

if ew_border.is_empty or ew_border.geom_type == "Point":
    warnings.warn("E/W border empty/point — using 200 m buffer fallback", RuntimeWarning)
    ew_border = _border_fallback(eng, wal, "E/W")

print("  dist_es ...", flush=True)
dist_es = nodes_proj.geometry.distance(es_border)
print("  dist_ew ...", flush=True)
dist_ew = nodes_proj.geometry.distance(ew_border)


# ══════════════════════════════════════════════════════════════════════════════
# 6. Build main analysis DataFrame
# ══════════════════════════════════════════════════════════════════════════════

nodes_df = pd.DataFrame(index=nodes_proj.index)
nodes_df["dist_coast"]    = dist_coast
nodes_df["dist_es"]       = dist_es
nodes_df["dist_ew"]       = dist_ew
nodes_df["country"]       = joined.reindex(nodes_proj.index)["country"]

# dist_political per country
nodes_df["dist_political"] = np.nan
eng_m = nodes_df["country"] == "England"
sco_m = nodes_df["country"] == "Scotland"
wal_m = nodes_df["country"] == "Wales"

nodes_df.loc[eng_m, "dist_political"] = nodes_df.loc[eng_m, ["dist_es", "dist_ew"]].min(axis=1)
nodes_df.loc[sco_m, "dist_political"] = nodes_df.loc[sco_m, "dist_es"]
nodes_df.loc[wal_m, "dist_political"] = nodes_df.loc[wal_m, "dist_ew"]

# Drop nodes outside all three countries
n_before = len(nodes_df)
nodes_df = nodes_df.dropna(subset=["country"])
print(f"\nCountry-assigned: {len(nodes_df):,}/{n_before:,} nodes")
print(nodes_df["country"].value_counts().to_string())

# Join centrality scores
nodes_df = nodes_df.join(global_scores_osm[MEASURES], how="inner")
print(f"After joining centrality scores: {len(nodes_df):,} nodes")


# ══════════════════════════════════════════════════════════════════════════════
# 7. Country-level BOSPERRUS fits
# ══════════════════════════════════════════════════════════════════════════════

print("\n=== Country-level BOSPERRUS fits ===", flush=True)
country_results: list[dict] = []

for country in ["England", "Scotland", "Wales"]:
    sub = nodes_df[nodes_df["country"] == country]
    if len(sub) < 10:
        print(f"  {country}: only {len(sub)} nodes — skipping")
        continue

    for dist_type in ["coast", "political"]:
        dist_col   = f"dist_{dist_type}"
        dist       = sub[dist_col].dropna().rename(dist_col)
        valid_idx  = dist.index
        scores_sub = global_scores_osm.reindex(valid_idx)[MEASURES].dropna()
        dist       = dist.reindex(scores_sub.index)

        if len(dist) < 10:
            print(f"  {country}/{dist_type}: {len(dist)} valid nodes — skipping")
            continue

        print(f"  {country} / {dist_type} : {len(dist):,} nodes  "
              f"[{dist.min()/1e3:.1f}–{dist.max()/1e3:.1f} km]", flush=True)

        try:
            flow = bosperrus.Flow.from_distances_and_scores(
                distances=dist,
                scores=scores_sub,
            )
            flow.flow(measures=MEASURES)
            fq = flow.fit_quality   # rows = param names, cols = measures

            for m in MEASURES:
                rel_ll   = fq.loc["scaled_relative_likelihood_over_baseline", m]
                effect   = fq.loc["observed_effect_strength", m]
                fit_type = fq.loc["best_fit_type", m]
                country_results.append({
                    "country":         country,
                    "dist_type":       dist_type,
                    "measure":         m,
                    "n_nodes":         int(len(dist)),
                    "rel_ll":          float(rel_ll),
                    "effect_strength": float(effect) if effect is not None else np.nan,
                    "best_fit_type":   fit_type,
                })
                eff_str = f"{effect:.4f}" if effect is not None else "N/A"
                print(f"    {m:12s}  rel_ll={rel_ll:.4f}  effect={eff_str}  fit={fit_type}")
        except Exception as exc:
            warnings.warn(f"BOSPERRUS failed for {country}/{dist_type}: {exc}", RuntimeWarning)

country_results_df = pd.DataFrame(country_results)


# ══════════════════════════════════════════════════════════════════════════════
# 8. Coastal county BOSPERRUS fits
# ══════════════════════════════════════════════════════════════════════════════

county_results: list[dict] = []

print("\n=== Coastal county BOSPERRUS fits ===", flush=True)
try:
    counties_raw = ox.features_from_place(
        "Great Britain",
        tags={"boundary": "administrative", "admin_level": "6"},
    )
    counties_gdf = (
        counties_raw[counties_raw.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
        .to_crs("EPSG:27700")
        .copy()
    )
    print(f"Got {len(counties_gdf)} county polygon features")
except Exception as exc:
    warnings.warn(f"County download failed ({exc}); skipping county analysis.", RuntimeWarning)
    counties_gdf = None

if counties_gdf is not None and len(counties_gdf) > 0:
    # Detect coastal counties: distance to GB coastline < 1000 m
    coastal_mask = counties_gdf.geometry.apply(lambda g: g.distance(coastline) < 1000)
    coastal_gdf  = counties_gdf[coastal_mask].copy()
    print(f"Coastal counties (dist < 1 km to coastline): {len(coastal_gdf)}")

    # Identify name column
    county_name_col = next(
        (c for c in ("name", "NAME", "admin_name", "official_name")
         if c in coastal_gdf.columns),
        None,
    )
    if county_name_col is None:
        non_geo = [c for c in coastal_gdf.columns if c != "geometry"]
        county_name_col = non_geo[0] if non_geo else None

    if county_name_col is None:
        warnings.warn("No name column in counties_gdf; skipping county analysis.", RuntimeWarning)
    else:
        print(f"County name column: '{county_name_col}'")

        # Assign nodes to coastal counties
        try:
            county_sjoin = gpd.sjoin(
                nodes_proj[["geometry"]],
                coastal_gdf[["geometry", county_name_col]],
                how="left",
                predicate="within",
            )
            county_sjoin = county_sjoin[~county_sjoin.index.duplicated(keep="first")]
        except Exception as exc:
            warnings.warn(f"County sjoin failed ({exc}); skipping.", RuntimeWarning)
            county_sjoin = None

        if county_sjoin is not None:
            nodes_df["county"] = county_sjoin.reindex(nodes_df.index)[county_name_col]

            for county_name, group in nodes_df[nodes_df["county"].notna()].groupby("county"):
                if len(group) <= 30:
                    continue

                dist       = group["dist_coast"].dropna().rename("dist_coast")
                scores_sub = global_scores_osm.reindex(dist.index)[MEASURES].dropna()
                dist       = dist.reindex(scores_sub.index)

                if len(dist) <= 30:
                    continue

                print(f"  {county_name} : {len(dist):,} nodes  "
                      f"[{dist.min()/1e3:.1f}–{dist.max()/1e3:.1f} km]", flush=True)

                try:
                    flow = bosperrus.Flow.from_distances_and_scores(
                        distances=dist,
                        scores=scores_sub,
                    )
                    flow.flow(measures=MEASURES)
                    fq = flow.fit_quality

                    for m in MEASURES:
                        rel_ll   = fq.loc["scaled_relative_likelihood_over_baseline", m]
                        effect   = fq.loc["observed_effect_strength", m]
                        fit_type = fq.loc["best_fit_type", m]
                        county_results.append({
                            "county":          county_name,
                            "dist_type":       "coast",
                            "measure":         m,
                            "n_nodes":         int(len(dist)),
                            "rel_ll":          float(rel_ll),
                            "effect_strength": float(effect) if effect is not None else np.nan,
                            "best_fit_type":   fit_type,
                        })
                        eff_str = f"{effect:.4f}" if effect is not None else "N/A"
                        print(f"    {m:12s}  rel_ll={rel_ll:.4f}  effect={eff_str}  fit={fit_type}")
                except Exception as exc:
                    warnings.warn(f"BOSPERRUS failed for county {county_name}: {exc}", RuntimeWarning)

county_results_df = pd.DataFrame(county_results) if county_results else pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# 9. Figure 1 — Country-level grouped bar charts (2 rows × 4 cols)
# ══════════════════════════════════════════════════════════════════════════════

print("\n=== Generating Figure 1 ===", flush=True)

fig1, axes = plt.subplots(2, 4, figsize=(16, 8))
dist_types = ["coast", "political"]
countries  = ["England", "Scotland", "Wales"]
pal        = {"England": "#4c78a8", "Scotland": "#54a24b", "Wales": "#e45756"}
x_pos      = np.arange(len(countries))
bar_w      = 0.55

for ri, dist_type in enumerate(dist_types):
    for ci, measure in enumerate(MEASURES):
        ax = axes[ri, ci]
        for xi, country in enumerate(countries):
            val = 0.0
            if len(country_results_df) > 0:
                mask = (
                    (country_results_df["dist_type"] == dist_type) &
                    (country_results_df["measure"]   == measure)   &
                    (country_results_df["country"]   == country)
                )
                rows = country_results_df[mask]
                if len(rows) > 0:
                    val = float(rows["rel_ll"].iloc[0])
            ax.bar(xi, val, bar_w, color=pal[country], alpha=0.85)

        ax.axhline(1.0, color="black", linestyle="--", linewidth=0.9, alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(countries, fontsize=8, rotation=20)
        ax.set_title(measure, fontsize=10, fontweight="bold")
        if ci == 0:
            label = "coast" if dist_type == "coast" else "political border"
            ax.set_ylabel(f"{label}\nrel. log-likelihood", fontsize=8)
        ax.tick_params(axis="both", labelsize=8)

handles = [plt.Rectangle((0, 0), 1, 1, color=pal[c], alpha=0.85) for c in countries]
fig1.legend(handles, countries, loc="lower center", ncol=3, fontsize=9,
            bbox_to_anchor=(0.5, -0.01))
fig1.suptitle("BOSPERRUS: Great Britain Rail Network — Country Level",
              fontsize=13, fontweight="bold")
fig1.tight_layout(rect=[0, 0.05, 1, 1])
fig1.savefig(OUT / "uk_rail_country_results.png", dpi=130, bbox_inches="tight")
plt.close(fig1)
print("Saved result_plots/uk_rail_country_results.png")


# ══════════════════════════════════════════════════════════════════════════════
# 10. Figure 2 — Coastal county bar charts (1 row × 4 cols)
# ══════════════════════════════════════════════════════════════════════════════

if len(county_results_df) > 0:
    print("=== Generating Figure 2 ===", flush=True)
    fig2, axes2 = plt.subplots(1, 4, figsize=(20, 6))
    for ci, measure in enumerate(MEASURES):
        ax = axes2[ci]
        sub = (county_results_df[county_results_df["measure"] == measure]
               .sort_values("rel_ll", ascending=False)
               .reset_index(drop=True))
        if len(sub) == 0:
            ax.set_title(measure, fontsize=10, fontweight="bold")
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        ax.bar(range(len(sub)), sub["rel_ll"].values, color="#4c78a8", alpha=0.85)
        ax.axhline(1.0, color="black", linestyle="--", linewidth=0.9, alpha=0.7)
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(sub["county"].values, rotation=70, ha="right", fontsize=6)
        ax.set_title(measure, fontsize=10, fontweight="bold")
        ax.set_ylabel("rel. log-likelihood", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)

    fig2.suptitle(
        "BOSPERRUS: GB Rail — Coastal Counties (dist. to coast, >30 rail nodes)",
        fontsize=11, fontweight="bold",
    )
    fig2.tight_layout()
    fig2.savefig(OUT / "uk_rail_county_coast_results.png", dpi=130, bbox_inches="tight")
    plt.close(fig2)
    print("Saved result_plots/uk_rail_county_coast_results.png")
else:
    print("No county results — Figure 2 skipped.")


# ══════════════════════════════════════════════════════════════════════════════
# 11. Summary table
# ══════════════════════════════════════════════════════════════════════════════

SEP = "=" * 90
print(f"\n{SEP}")
print("BOSPERRUS SUMMARY — Great Britain Rail Network")
print(SEP)

if len(country_results_df) > 0:
    for dist_type in ["coast", "political"]:
        label = "Distance to coast" if dist_type == "coast" else "Distance to political border"
        print(f"\n--- COUNTRY LEVEL | {label} ---")
        sub = (country_results_df[country_results_df["dist_type"] == dist_type]
               [["country", "measure", "n_nodes", "rel_ll", "effect_strength", "best_fit_type"]]
               .sort_values(["country", "measure"]))
        print(sub.to_string(index=False))
else:
    print("No country-level results.")

if len(county_results_df) > 0:
    print("\n--- COASTAL COUNTY LEVEL | Distance to coast ---")
    for measure in MEASURES:
        sub = (county_results_df[county_results_df["measure"] == measure]
               .sort_values("rel_ll", ascending=False)
               [["county", "n_nodes", "rel_ll", "effect_strength", "best_fit_type"]])
        print(f"\n  Measure: {measure}")
        print(sub.to_string(index=False))
else:
    print("\nNo coastal-county results.")

print(f"\n{SEP}")
print("Done.")
