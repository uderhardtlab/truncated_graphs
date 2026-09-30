"""How much does bosperrus' switch from single-start curve_fit (+ local DE for
PiecewiseLinearFit) to global profile fits change figure 3 (MIBI-TOF)?

For a random subset of datasets x graph types: compute distances + centralities
once (bosperrus.Flow.from_coords, as src/figure3/compute_fits.py does), then run
the same AIC model selection (Constant vs PiecewiseLinear / ExponentialSaturation /
MichaelisMenten) with the OLD fit classes (git HEAD of bosperrus/fit.py, copied to
old_fit.py) and the NEW ones on identical scores.

Usage (cwd = truncated_graphs/): pixi run python src/exploratory/fit_optimizer_check/figure3_subset.py
Writes results/exploratory/fit_optimizer_check/figure3_subset.csv
"""
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

import bosperrus
from bosperrus.distances import distance_to_convex_hull

sys.path.insert(0, str(Path(__file__).resolve().parent))
import old_fit  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "exploratory" / "fit_optimizer_check" / "figure3_subset.csv"
MEASURES = ["degree", "pagerank", "betweenness", "closeness", "clustering"]
GRAPH_TYPES = {"delaunay": None, "knn_k=10": {"k": 10}, "rnn_r=0.03": {"r": 0.03}}
N_DATASETS = 60

NEW = [bosperrus.PiecewiseLinearFit, bosperrus.ExponentialSaturationFit, bosperrus.MichaelisMentenFit]
OLD = [old_fit.PiecewiseLinearFit, old_fit.ExponentialSaturationFit, old_fit.MichaelisMentenFit]


def select(S, d, classes, constant_cls):
    base = constant_cls(S, d); base.fit()
    out = {"Constant Fit": (base.AIC, dict(base.params))}
    for cls in classes:
        f = cls(S, d); f.fit()
        out[f.name] = (f.AIC, dict(f.params))
    best = min(out, key=lambda k: out[k][0])
    return best, out


def run(dataset, coordinates, graph_type):
    lo, hi = coordinates.min(axis=0), coordinates.max(axis=0)
    coords = (coordinates - lo) / (hi - lo)
    try:
        flow = bosperrus.Flow.from_coords(coordinates=coords, distance_fn=distance_to_convex_hull, measures=MEASURES,
                                          graph_type=graph_type.split("_")[0], graph_kwargs=GRAPH_TYPES[graph_type])
    except ValueError as e:
        return []
    obs = flow.observations
    d = obs[flow._distance_key]
    rows = []
    for m in MEASURES:
        S = obs[m]
        best_old, fits_old = select(S, d, OLD, old_fit.ConstantFit)
        best_new, fits_new = select(S, d, NEW, bosperrus.ConstantFit)
        row = {"dataset": dataset, "graph_type": graph_type, "measure": m, "n": len(S),
               "best_old": best_old, "best_new": best_new}
        for name in fits_new:
            row[f"AIC_old:{name}"] = fits_old[name][0]
            row[f"AIC_new:{name}"] = fits_new[name][0]
        for key in ["piecewise_linear_b", "exponential_saturation_b", "michaelis_menten_b"]:
            for tag, fits in (("old", fits_old), ("new", fits_new)):
                row[f"{key}_{tag}"] = next((p[key] for _, p in fits.values() if key in p), np.nan)
        rows.append(row)
    return rows


if __name__ == "__main__":
    with open(ROOT / "mibitof_coords" / "coords.pickle", "rb") as f:
        datasets = pickle.load(f)
    datasets.pop("glioma_mibitof:CHOP_907_R1C6_whole_cell.tiff", None)
    datasets = {k: v for k, v in datasets.items() if not k.startswith("sq_visium:")}
    names = sorted(datasets)
    pick = list(np.random.default_rng(0).choice(names, size=min(N_DATASETS, len(names)), replace=False))
    print(f"{len(names)} MIBI-TOF datasets, checking {len(pick)} x {len(GRAPH_TYPES)} graph types", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=len(os.sched_getaffinity(0))) as ex:
        futs = [ex.submit(run, ds, datasets[ds], gt) for ds in pick for gt in GRAPH_TYPES]
        for i, fut in enumerate(as_completed(futs)):
            rows.extend(fut.result())
            if i % 20 == 0:
                print(f"  {i + 1}/{len(futs)}", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print("wrote", OUT)
