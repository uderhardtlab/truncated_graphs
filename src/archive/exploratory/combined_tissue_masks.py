"""Combined figure: image-based tissue mask vs. spot-based "tissue mask"
(every analyzed bin's raw position) for all 12 samples in
src/figure4/sample_manifest.csv.

Image-based mask:
- Visium (demo + custom): visium_hd.load_visium_tissue_mask -- segments the
  embedded hires H&E/CytAssist image, counts-independent.
- STOmics: the final image-only masks from segment_stomics_tissue_final.py
  (spatial_data/stomics/{name}_ssDNA_tissue_mask_final_native.npz),
  NOT tissue_cut.tif -- see spatial_data/README.md's stomics section for why
  tissue_cut.tif isn't actually counts-independent.

Spot-based mask: every bin present in the sample's h5ad (obsm["spatial"]),
read directly via h5py rather than a full sc.read_h5ad (only a small Nx2
array is needed, not the expression matrix -- same reasoning as
load_visium_tissue_mask's docstring).

Uses visium_hd.plot_mask_overlap_transparent for the overlay rendering --
NOT plot_image_tissue_vs_counts's alpha-blended scatter (saturates to solid
red once bin density is high, hiding the mask-vs-spot distinction), and not
a hard-categorical raster either (looked like illegible "braille" once a
huge sparse array got downsampled for display). This version renders both
masks as independent translucent color layers, pre-pooled down to a common
modest resolution so nothing gets aliased away -- overlap shows as a
blended color, each mask alone stays in its own color.

Also computes, per sample, the fraction of spots falling outside the image
mask (bosperrus.distance_to_mask > 0) -- the first concrete number behind
the original "are there spots with signal outside the image-based tissue
boundary" question, not just a qualitative picture.

Writes result_plots/exploratory/combined_tissue_masks.png and prints the
per-sample outside-fraction table.
"""
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import bosperrus

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
import visium_hd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MANIFEST_PATH = REPO_ROOT / "src/figure4/sample_manifest.csv"
STOMICS_DATA_DIR = Path("/home/woody/iwbn/iwbn007h/spatial_data/stomics")
OUT_PATH = REPO_ROOT / "result_plots/exploratory/combined_tissue_masks.png"


def load_spot_coords(h5ad_path):
    """Raw obsm['spatial'] pixel coordinates, read directly via h5py (no
    full AnnData load -- these files can hold a >1M-bin counts matrix, none
    of which this needs)."""
    with h5py.File(h5ad_path, "r") as f:
        return f["obsm"]["spatial"][:]


def load_image_mask(row):
    """(mask, pixel_scale) -- Visium via the embedded H&E image (unchanged,
    already counts-independent); STOmics via the final image-only masks
    (NOT tissue_cut.tif, which isn't actually counts-independent -- see
    spatial_data/README.md's stomics section)."""
    if row["loader"] == "visium":
        return visium_hd.load_visium_tissue_mask(row["h5ad_path"], row["name"])
    elif row["loader"] == "stomics":
        npz = np.load(STOMICS_DATA_DIR / f"{row['name']}_ssDNA_tissue_mask_final_native.npz")
        return npz["mask"], 1.0
    raise ValueError(f"unknown loader {row['loader']!r}")


if __name__ == "__main__":
    samples = pd.read_csv(MANIFEST_PATH)

    ncols = 4
    nrows = -(-len(samples) // ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axs = axs.ravel()

    outside_fractions = []
    for i, (_, row) in enumerate(samples.iterrows()):
        coords = load_spot_coords(row["h5ad_path"])
        spatial_x, spatial_y = coords[:, 0], coords[:, 1]
        mask, pixel_scale = load_image_mask(row)

        # distance_to_mask indexes mask as [row, col] = [y, x]
        coords_rc = np.stack([spatial_y * pixel_scale, spatial_x * pixel_scale], axis=1)
        dist = bosperrus.distance_to_mask(coords_rc, mask).to_numpy()
        frac_outside = (dist > 0).mean()
        outside_fractions.append({"name": row["name"], "dataset_type": row["dataset_type"],
                                   "n_spots": len(coords), "frac_outside_image_mask": frac_outside})

        results = [{"spatial_x": spatial_x, "spatial_y": spatial_y}]
        visium_hd.plot_mask_overlap_transparent(
            axs[i], results, mask, pixel_scale,
            title=f"{row['name']}\n{frac_outside * 100:.1f}% of spots outside image mask",
        )
        print(f"{row['name']}: n_spots={len(coords)}, frac_outside_image_mask={frac_outside * 100:.2f}%")

    for ax in axs[len(samples):]:
        ax.axis("off")

    legend_handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#1f77b4", alpha=0.55, markersize=10,
               label="image mask"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#d62728", alpha=0.55, markersize=10,
               label="bins"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#7b3f9e", markersize=10,
               label="both (blended)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncols=3, fontsize=10, bbox_to_anchor=(0.5, 0))

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=100)
    print(f"saved {OUT_PATH}")

    summary_path = REPO_ROOT / "result_plots/exploratory/combined_tissue_masks_summary.csv"
    pd.DataFrame(outside_fractions).to_csv(summary_path, index=False)
    print(f"saved {summary_path}")
