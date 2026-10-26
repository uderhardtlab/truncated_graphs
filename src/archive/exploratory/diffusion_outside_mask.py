"""Quantify RNA diffusion outside the image-based tissue mask: fit
ExponentialSaturationFit (vs. ConstantFit baseline) of log1p(total_counts)
against distance *outside* the mask, for bins that fall outside it.

This is the outward-facing mirror of the usual inside-tissue border-effect
story (score recovers with distance *into* the tissue): here, score is
expected to *decay* with distance *away* from the tissue edge, back toward
a background/noise floor, if RNA is genuinely diffusing/leaking beyond the
true tissue boundary. ExponentialSaturationFit's `a` parameter sign tells
the two cases apart: a>0 is the usual "deficit at the border, recovers
outward" shape; a<0 here would mean "elevated near the border, decays
outward" -- the diffusion signature. No constraint is imposed either way;
b>0 (monotonic) is bosperrus's only built-in constraint, so the fit is free
to come out either way.

One sample per invocation (--index into src/figure4/sample_manifest.csv),
for a SLURM array job -- computing per-bin total counts for the biggest
Visium samples needs the full (sparse) expression matrix in memory
(Visium's h5ad has no precomputed total_counts obs column, unlike STOmics'
-- summed here directly from the CSR matrix via h5py + scipy.sparse rather
than a full sc.read_h5ad, for the same reason load_visium_tissue_mask
avoids it).

Image mask: Visium via the embedded H&E image; STOmics via the final
image-only masks (segment_stomics_tissue_final.py's output), NOT
tissue_cut.tif -- see spatial_data/README.md's stomics section for why.
Distance is via bosperrus.distance_to_mask (0 for bins inside the mask,
true distance for bins outside), computed in the mask's own pixel space
then divided by pixel_scale to get back to the *native* pixel grid bins
were originally in.

Unit caveat, deliberately NOT resolved here: reported distances/decay
lengths are in *native pixel* units, not microns. sample_manifest.csv's
bin_size_um is the physical size of one array_row/array_col *grid step*,
not one raw obsm['spatial'] pixel -- converting this analysis's native-
pixel distances to microns would need the actual raw-pixel-to-micron scale
(derivable from the known affine relationship between obsm['spatial'] and
array_row/array_col, not yet verified), so no micron conversion is applied
here to avoid silently reporting a wrong-by-some-factor number.

Writes results/exploratory/diffusion_outside_mask/{name}.json and
{name}.png (scatter + fitted curve) per sample.
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

import bosperrus

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
import visium_hd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MANIFEST_PATH = REPO_ROOT / "src/figure4/sample_manifest.csv"
STOMICS_DATA_DIR = Path("/home/woody/iwbn/iwbn007h/spatial_data/stomics")
OUT_DIR = REPO_ROOT / "results/exploratory/diffusion_outside_mask"


def load_log1p_total_counts(row):
    if row["loader"] == "visium":
        with h5py.File(row["h5ad_path"], "r") as f:
            X = csr_matrix(
                (f["X"]["data"][:], f["X"]["indices"][:], f["X"]["indptr"][:]),
                shape=tuple(f["X"].attrs["shape"]),
            )
        total_counts = np.asarray(X.sum(axis=1)).ravel()
    else:
        with h5py.File(row["h5ad_path"], "r") as f:
            total_counts = f["obs"]["total_counts"][:]
    return np.log1p(total_counts)


def load_spot_coords(h5ad_path):
    with h5py.File(h5ad_path, "r") as f:
        return f["obsm"]["spatial"][:]


def load_image_mask(row):
    if row["loader"] == "visium":
        return visium_hd.load_visium_tissue_mask(row["h5ad_path"], row["name"])
    npz = np.load(STOMICS_DATA_DIR / f"{row['name']}_ssDNA_tissue_mask_final_native.npz")
    return npz["mask"], 1.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, required=True)
    args = parser.parse_args()

    row = pd.read_csv(MANIFEST_PATH).iloc[args.index]
    name = row["name"]

    coords = load_spot_coords(row["h5ad_path"])
    log1p_counts = load_log1p_total_counts(row)
    mask, pixel_scale = load_image_mask(row)

    coords_rc_mask_space = np.stack([coords[:, 1] * pixel_scale, coords[:, 0] * pixel_scale], axis=1)
    dist_mask_space = bosperrus.distance_to_mask(coords_rc_mask_space, mask).to_numpy()

    outside = dist_mask_space > 0
    n_total, n_outside = len(dist_mask_space), int(outside.sum())
    d_outside = dist_mask_space[outside] / pixel_scale  # back to native-pixel units

    scores = pd.Series(log1p_counts[outside], name="log1p_total_counts")
    d = pd.Series(d_outside, name="dist_outside_mask_native_px")

    baseline = bosperrus.ConstantFit(scores, d)
    baseline.fit()
    exp_sat = bosperrus.ExponentialSaturationFit(scores, d)
    exp_sat.fit()

    result = {
        "name": name, "dataset_type": row["dataset_type"],
        "n_total": n_total, "n_outside": n_outside, "frac_outside": n_outside / n_total,
        "aic_constant": baseline.AIC, "aic_exp_sat": exp_sat.AIC,
        "exp_sat_wins": bool(exp_sat.AIC < baseline.AIC),
        "converged": bool(exp_sat._converged),
    }
    if exp_sat._converged:
        a, b, c = (exp_sat.params["exponential_saturation_a"],
                   exp_sat.params["exponential_saturation_b"],
                   exp_sat.params["exponential_saturation_c"])
        result.update({
            "a": a, "b": b, "c": c,
            "decay_length_native_px": 1 / b,
            "direction": "decay away from tissue (diffusion-like)" if a < 0 else "increase away from tissue",
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / f"{name}.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(d_outside, scores, s=1, alpha=0.1, color="gray", rasterized=True)
    if exp_sat._converged:
        d_sorted = np.sort(d_outside)
        y_fit = bosperrus.ExponentialSaturationFit.exp_sat(d_sorted, a, b, c)
        ax.plot(d_sorted, y_fit, color="crimson", lw=2,
                label=f"exp-sat (decay length={1/b:.1f} native px, a={a:.2f})")
        ax.legend(fontsize=8)
    ax.set_xlabel("distance outside image mask (native px)")
    ax.set_ylabel("log1p(total counts)")
    ax.set_title(f"{name}: {n_outside}/{n_total} bins outside mask ({n_outside/n_total*100:.1f}%)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{name}.png", dpi=100)
    print(f"saved {OUT_DIR / (name + '.png')}")
