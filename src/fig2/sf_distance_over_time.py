"""
Evolutionary test: does SF-distance predict degree centrality,
and does this effect decay as the Brightkite network matures?

For each quarter (2008-Q2 → 2010-Q3), builds the induced subgraph on all
users whose first check-in fell at or before that quarter end, computes
in-snapshot degree centrality, then runs BOSPERRUS with SF great-circle
distance as the distance metric.

Saves: result_plots/sf_distance_over_time.{svg,png}
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import bosperrus

SF_LAT, SF_LON = 37.7749, -122.4194


def haversine_km(lat, lon, lat2=SF_LAT, lon2=SF_LON):
    R = 6371.0
    phi1, phi2 = np.radians(lat), np.radians(lat2)
    a = (np.sin(np.radians(lat2 - lat) / 2) ** 2
         + np.cos(phi1) * np.cos(phi2) * np.sin(np.radians(lon2 - lon) / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(a))


# ── load data ─────────────────────────────────────────────────────────────────
print("loading …")
logs = pd.read_csv(ROOT / "SNAP_data" / "loc-brightkite_totalCheckins.txt",
                   delimiter="\t", header=None,
                   names=["user", "time", "lat", "lon", "loc_id"])
logs["time"] = pd.to_datetime(logs["time"])

edges = pd.read_csv(ROOT / "SNAP_data" / "loc-brightkite_edges.txt",
                    delimiter="\t", header=None, names=["u", "v"])

median_locs = (pd.read_csv(ROOT / "notebooks" / "geo_data" /
                            "median_location_per_user_brightkite.csv")
               .set_index("user"))
median_locs["dist_sf"] = haversine_km(median_locs["latitude"],
                                       median_locs["longitude"])

first_checkin = logs.groupby("user")["time"].min()

# ── quarterly snapshots: Q2-2008 … Q3-2010 ───────────────────────────────────
quarters = pd.date_range("2008-06-30", "2010-09-30", freq="QE", tz="UTC")

records = []
for t in quarters:
    active = set(first_checkin[first_checkin <= t].index)
    active_with_loc = active & set(median_locs.index)

    # degree in the induced subgraph (edges where BOTH endpoints are active)
    mask   = edges["u"].isin(active_with_loc) & edges["v"].isin(active_with_loc)
    counts = pd.concat([edges.loc[mask, "u"],
                        edges.loc[mask, "v"]]).value_counts()
    users  = sorted(active_with_loc)
    degree = counts.reindex(users, fill_value=0)

    dist   = median_locs.loc[users, "dist_sf"]

    flow = bosperrus.Flow.from_distances_and_scores(
        distances=dist,
        scores=pd.DataFrame({"degree": degree}),
    )
    flow.flow(measures=["degree"])

    fq       = flow.fit_quality["degree"]
    rel_ll   = fq["scaled_relative_likelihood_over_baseline"]
    effect   = fq["observed_effect_strength"]
    best_fit = fq["best_fit_type"]

    print(f"  {t.date()}  n={len(users):6d}  "
          f"rel_ll={rel_ll:.4f}  effect={effect:.4f}  [{best_fit}]")
    records.append(dict(date=t, n_users=len(users),
                        rel_ll=rel_ll, effect=effect, best_fit=best_fit))

results = pd.DataFrame(records)

# ── plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(7, 6), sharex=True)

axes[0].plot(results["date"], results["rel_ll"], "o-", color="steelblue")
axes[0].axhline(1.0, lw=0.8, ls="--", color="#aaaaaa")
axes[0].set_ylabel("rel. likelihood\nvs. constant")
axes[0].set_title("SF-distance → degree centrality over Brightkite's lifetime")

axes[1].plot(results["date"], results["effect"].abs(), "o-", color="salmon")
axes[1].axhline(0.0, lw=0.8, ls="--", color="#aaaaaa")
axes[1].set_ylabel("|effect strength|")

axes[2].bar(results["date"], results["n_users"], width=60,
            color="steelblue", alpha=0.4)
axes[2].set_ylabel("users in snapshot")
axes[2].set_xlabel("quarter end")

fig.tight_layout()
out = ROOT / "result_plots" / "sf_distance_over_time"
fig.savefig(str(out) + ".svg", bbox_inches="tight")
fig.savefig(str(out) + ".png", dpi=150, bbox_inches="tight")
print(f"\nsaved {out}.svg / .png")
plt.close("all")
