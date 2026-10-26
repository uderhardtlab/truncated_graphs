"""Load the sphere benchmark CSVs: sphere/ (all methods incl. SERN) merged with sphere_detrend/ (the
detrending baselines from a --no-sern run of sphere.py with the same seeds, i.e. the same crops).

The detrending columns are only taken from sphere_detrend/ if sphere/ lacks them. The merge checks that
raw and BOSPERRUS correlations agree between both runs. Rows may differ only where the node masks differ:
sphere/ also excludes nodes that are NaN in every SERN surrogate, the --no-sern run cannot.
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd

DETREND_COLS = ["original vs. linear detrended on crop", "original vs. spline detrended on crop"]
CHECK_COLS = ["original vs. on crop", "original vs. BOSPERRUS corrected on crop",
              "original vs. BOSPERRUS multiplicative on crop", "original vs. BOSPERRUS log on crop"]
KEYS = ["seed", "coord_type", "graph_type", "k", "radius", "cap_radius", "corr_method", "node_subset", "measure"]


def _read(directory):
    files = glob.glob(str(Path(directory) / "*.csv"))
    df = pd.concat(pd.read_csv(f).rename(columns={"Unnamed: 0": "measure"}) for f in files)
    df[["k", "radius"]] = df[["k", "radius"]].fillna(0)
    return df.reset_index(drop=True)


def load_sphere(results_dir, verbose=True):
    """All sphere rows with every method's correlation column (see module docstring)."""
    df = _read(Path(results_dir) / "sphere")
    if all(c in df for c in DETREND_COLS):
        return df
    det = _read(Path(results_dir) / "sphere_detrend")
    merged = df.merge(det[KEYS + CHECK_COLS + DETREND_COLS + ["n_excluded"]], on=KEYS, how="left",
                      suffixes=("", "_detrend"), validate="one_to_one", indicator=True)
    missing = (merged.pop("_merge") != "both").sum()
    if missing:
        raise ValueError(f"{missing} sphere/ rows have no match in sphere_detrend/")

    # atol 1e-4: graph-tool's multithreaded betweenness differs in the last digits between runs with different
    # core counts (up to 3.6e-5 in the 2026-10-26 merge); a different crop would differ by far more
    same = np.logical_and.reduce([np.isclose(merged[c], merged[f"{c}_detrend"], rtol=0, atol=1e-4, equal_nan=True)
                                  for c in CHECK_COLS])
    same_mask = merged["n_excluded"] == merged["n_excluded_detrend"]
    if (~same & same_mask).any():
        raise ValueError(f"{(~same & same_mask).sum()} rows differ between sphere/ and sphere_detrend/ "
                         "although their node masks are the same size -- not the same crops?")
    if verbose:
        print(f"sphere_detrend merge: {len(merged)} rows; node mask differs (SERN-NaN nodes) in "
              f"{(~same_mask).sum()}, raw/BOSPERRUS values differ in {(~same).sum()}")
    return merged.drop(columns=[f"{c}_detrend" for c in CHECK_COLS] + ["n_excluded_detrend"])
