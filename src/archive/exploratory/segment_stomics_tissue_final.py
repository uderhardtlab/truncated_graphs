"""Final STOmics ssDNA tissue segmentation -- an image-only tissue mask,
independent of BGI's expression-informed tissue_cut.tif (see
visium_hd.load_stomics_tissue_mask's docstring). See
spatial_data/stomics/README.md for the full story of how this config was
picked; the short version:

- A02991D1: multi-Otsu (classes=3), keep the top class
- A02994C6, Y01087DD, Y01084J8: single Otsu (one threshold)
- All 4: binary_closing with disk=6 at the downsampled ("small") working
  resolution, then binary_fill_holes

A single rule doesn't work for all 4 samples -- they come from two different
SAW pipeline runs/dates (see spatial_data/README.md's stomics/ table) with
visibly different tissue-vs-background intensity separation. Multi-Otsu's
middle class is background for A02991D1 but is real tissue for the other
three (confirmed by directly visualizing the 3 classes) -- see this
directory's git history for that diagnostic if you need to revisit it.

Known limitation, left as-is by choice: Y01087DD's two small satellite
fragments stay somewhat speckled/holey at disk=6 (they didn't fully
solidify anywhere in the swept range up to disk=8); its main fragment and
all of the other 3 samples come out solid and correctly non-merged.

Run (no SLURM needed -- each sample takes a few seconds once past package
import; only the *interactive login node* was unreliable this session, not
the computation):
    cd src/exploratory && pixi run python segment_stomics_tissue_final.py
or via jobs/stomics_tissue_final.sbatch.

Writes, per sample, to spatial_data/stomics/:
    {name}_ssDNA_tissue_mask_final_small.npz   (downsampled resolution)
    {name}_ssDNA_tissue_mask_final_native.npz  (upsampled to native
                                                 resolution, nearest-neighbor,
                                                 exact integer factor)
Both store the boolean mask under the key "mask". Also writes a summary
diagnostic, spatial_data/stomics/final_masks.png.
"""
from pathlib import Path

import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes
from skimage.filters import threshold_multiotsu, threshold_otsu
from skimage.morphology import binary_closing, disk
from skimage.transform import downscale_local_mean

DATA_DIR = Path("/home/woody/iwbn/iwbn007h/spatial_data/stomics")
TARGET_SIZE = 2000  # downsampled long-edge size, px
DISK_SIZE_SMALL = 6

METHODS = {
    "A02991D1": "multiotsu_top",
    "A02994C6": "single_otsu",
    "Y01087DD": "single_otsu",
    "Y01084J8": "single_otsu",
}


def segment(small, method):
    if method == "multiotsu_top":
        thresh = threshold_multiotsu(small, classes=3)
        return small > thresh[-1]
    return small > threshold_otsu(small)


if __name__ == "__main__":
    fig, axs = plt.subplots(2, len(METHODS), figsize=(4 * len(METHODS), 8))

    for i, (name, method) in enumerate(METHODS.items()):
        img = tifffile.imread(DATA_DIR / f"{name}_ssDNA_regist.tif")
        native_shape = img.shape
        factor = max(1, round(img.shape[0] / TARGET_SIZE))
        small = downscale_local_mean(img, (factor, factor)).astype(float)

        seed = segment(small, method)
        closed = binary_closing(seed, disk(DISK_SIZE_SMALL))
        filled = binary_fill_holes(closed)

        filled_native = np.repeat(np.repeat(filled, factor, axis=0), factor, axis=1)
        assert filled_native.shape == native_shape, (filled_native.shape, native_shape)

        small_path = DATA_DIR / f"{name}_ssDNA_tissue_mask_final_small.npz"
        native_path = DATA_DIR / f"{name}_ssDNA_tissue_mask_final_native.npz"
        np.savez_compressed(small_path, mask=filled)
        np.savez_compressed(native_path, mask=filled_native)

        axs[0, i].imshow(np.log1p(small), cmap="gray")
        axs[0, i].set_title(name)
        axs[0, i].axis("off")
        axs[1, i].imshow(filled, cmap="gray")
        axs[1, i].set_title(f"{method}, disk={DISK_SIZE_SMALL}@small\ncov={filled.mean() * 100:.1f}%")
        axs[1, i].axis("off")

        print(f"{name}: method={method}, coverage={filled.mean() * 100:.2f}% "
              f"-> {small_path.name}, {native_path.name}")

    plt.tight_layout()
    out_path = DATA_DIR / "final_masks.png"
    plt.savefig(out_path, dpi=100)
    print(f"saved {out_path}")
