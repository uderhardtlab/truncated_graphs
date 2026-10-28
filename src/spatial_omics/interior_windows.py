"""Null distribution for the border-effect criterion: square windows
placed well inside the tissue (no physical border), fit exactly like the real
tissue edges, so their fits show what tissue structure alone produces.

Per sample (Visium HD demo set), windows of WINDOW_SIZES_UM are placed at
random positions where every bin of the window is
- inside the capture area and inside the image mask, and
- at least MARGIN_UM from the edge of the whole image mask (so no real border
  effect reaches in; real-edge elbows are <= ~300 um except false ones).
Each window's bins go through bosperrus.identify_analysis_buffer_from_filtered
(n_counts > 0, hole filling, components, ConstantFit vs RisingPiecewiseLinearFit, m > 0
rule) with the distance to the window's own edge supplied as distance_to_border, fitted
within TISSUE_FIT_MAX_DISTANCE_UM like the tissue edges.
Effect strength and half-life come from border_criterion.criterion_metrics, the
same function used for the real tissue-edge fits; their distribution over the
windows is the null of border_criterion.window_p_value.

Needs the main st_precompute.py run ({sample}_diffusion_8um.npz).

Usage (cwd = truncated_graphs/):
    pixi run python src/spatial_omics/interior_windows.py [sample ...]   # default: all visium_demo samples; paired samples allowed
    pixi run python src/spatial_omics/interior_windows.py --resolution 2 [sample ...]   # setup of st_precompute.py --tissue-edge

Writes results/spatial_omics/st_reroll/interior_windows/{sample}_windows.csv, one row per window.
"""
import argparse
import sys
import time
from pathlib import Path

import anndata
import numpy as np
import pandas as pd

import bosperrus

sys.path.insert(0, str(Path(__file__).resolve().parent))
from border_criterion import criterion_metrics  # noqa: E402
from st_precompute import (  # noqa: E402
    MAX_HOLE_AREA_UM2, MIN_COMPONENT_SPOTS, OUT_DIR, RESOLUTION_UM, SAMPLES, TISSUE_FIT_MAX_DISTANCE_UM,
    distance_to_tissue_edge, load_raw,
    min_component_bins,
)

WINDOW_SIZES_UM = [500.0, 1000.0, 2000.0]
MARGIN_UM = 500.0
N_WINDOWS = 20  # per size and sample (fewer if fewer valid positions)
SEED = 0
WINDOW_DIR = OUT_DIR / "interior_windows"


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def valid_window_corners(ok, size_bins):
    """(row, col) of every top-left corner whose size_bins x size_bins window
    lies entirely on True cells of the dense grid `ok` (integral image)."""
    s = np.zeros((ok.shape[0] + 1, ok.shape[1] + 1), dtype=np.int64)
    s[1:, 1:] = ok.cumsum(0).cumsum(1)
    k = size_bins
    full = s[k:, k:] - s[:-k, k:] - s[k:, :-k] + s[:-k, :-k]
    return np.argwhere(full == k * k)


def fit_window(row, col, n_counts, r0, c0, k, resolution=RESOLUTION_UM, keep_zeros=False):
    """identify_analysis_buffer_from_filtered on one window, distance = to the window edge.
    keep_zeros: keep zero-count bins (as st_precompute.py --tissue-edge)."""
    in_win = (row >= r0) & (row < r0 + k) & (col >= c0) & (col < c0 + k)
    wr, wc = row[in_win], col[in_win]
    obs = pd.DataFrame({"array_row": wr, "array_col": wc, "n_counts": n_counts[in_win].astype(np.float64)})
    dist = np.minimum.reduce([wr - r0, r0 + k - 1 - wr, wc - c0, c0 + k - 1 - wc]) * resolution
    obs["distance_to_border"] = np.where(dist <= TISSUE_FIT_MAX_DISTANCE_UM, dist, np.nan)
    obs.index = obs["array_row"].astype(str) + "_" + obs["array_col"].astype(str)
    adata = anndata.AnnData(obs=obs)
    adata.uns["bosperrus"] = {"bin_size_um": resolution}
    if keep_zeros:
        bosperrus.identify_analysis_buffer_from_filtered(
            adata, n_counts_key=None, min_component_size=min_component_bins(resolution), max_hole_area_um2=None,
            rising_only=True,
        )
    else:
        bosperrus.identify_analysis_buffer_from_filtered(
            adata, min_component_size=MIN_COMPONENT_SPOTS, max_hole_area_um2=MAX_HOLE_AREA_UM2,
            rising_only=True,
        )
    comp = adata.obs["components"].to_numpy()
    dist = adata.obs["distance_to_border"].to_numpy()
    rows = []
    for label, info in adata.uns["analysis_buffer_fit"]["per_component"].items():
        member = (comp == label) & np.isfinite(dist)
        eff, half = criterion_metrics(info, dist[member].max())
        rows.append({
            "component": int(label), "n_bins": info["n_spots"], "best_fit_type": info["best_fit_type"],
            "border_effect": info["border_effect"], "elbow_um": info["elbow_um"],
            "effect_strength": eff, "half_life_um": half,
            **{k: v for k, v in info["params"].items()},
        })
    return rows


def run_sample(sample, rng, resolution=None):
    """resolution=None: the 8um main-run setup (n_counts > 0); a number: the --tissue-edge setup at that
    bin size (zero-count bins kept), written as {sample}_windows_tissueedge_{res}um.csv."""
    keep_zeros = resolution is not None
    res = RESOLUTION_UM if resolution is None else resolution
    raw = load_raw(sample, res)
    inside = raw["distance_um"] == 0
    to_tissue = np.full(len(inside), -np.inf)
    to_tissue[inside] = distance_to_tissue_edge(raw, inside)
    row, col = raw["array_row"].astype(np.int64), raw["array_col"].astype(np.int64)
    row, col = row - row.min(), col - col.min()  # raw bins cover the full capture-area grid
    ok = np.zeros((row.max() + 1, col.max() + 1), dtype=bool)
    ok[row, col] = inside & (to_tissue >= MARGIN_UM)
    n_counts = raw["n_counts"]

    out = []
    for size_um in WINDOW_SIZES_UM:
        k = int(round(size_um / res))
        corners = valid_window_corners(ok, k)
        n = min(N_WINDOWS, len(corners))
        _log(f"{sample}: {size_um:g} um windows: {len(corners):,} valid positions, fitting {n}")
        for i in rng.choice(len(corners), size=n, replace=False) if n else []:
            r0, c0 = corners[i]
            for rec in fit_window(row, col, n_counts, r0, c0, k, res, keep_zeros):
                out.append({"sample": sample, "window_um": size_um, "row0": int(r0), "col0": int(c0), **rec})
    WINDOW_DIR.mkdir(parents=True, exist_ok=True)
    name = f"_tissueedge_{res:g}um" if keep_zeros else ""
    pd.DataFrame(out).to_csv(WINDOW_DIR / f"{sample}_windows{name}.csv", index=False)
    _log(f"{sample}: wrote {len(out)} window fits")


if __name__ == "__main__":
    demo = list(SAMPLES.index[SAMPLES["dataset_type"] == "visium_demo"])
    paired = list(SAMPLES.index[SAMPLES["dataset_type"].isin(["visium_custom", "stomics_custom"])])
    parser = argparse.ArgumentParser()
    parser.add_argument("samples", nargs="*", help="visium_demo or paired (visium_custom/stomics_custom) samples "
                                                   "(default: all visium_demo)")
    parser.add_argument("--resolution", type=float, default=None,
                        help="the st_precompute.py --tissue-edge setup at this bin size (zero-count bins kept); "
                             "writes {sample}_windows_tissueedge_{res}um.csv")
    args = parser.parse_args()
    unknown = set(args.samples) - set(demo) - set(paired)
    if unknown:
        parser.error(f"not visium_demo or paired samples: {sorted(unknown)}")
    for sample in args.samples or demo:
        # same windows whatever subset is run; paired samples get their own seed range
        seed_index = demo.index(sample) if sample in demo else 1000 + paired.index(sample)
        run_sample(sample, np.random.default_rng([SEED, seed_index]), args.resolution)
    _log("done")
