"""Visium HD / STOmics tissue-border effect analysis.

For each binned sample: identify tissue-border spots as grid positions with
fewer than 4 von-Neumann grid-neighbors present in the tissue
(squidpy.gr.spatial_neighbors_grid(n_rings=1, n_neighs=4)), split the tissue
into its spatially-connected components (e.g. a TMA's individual cores, or
any other disconnected fragments), and independently fit BOTH
PiecewiseLinearFit (a discrete buffer/exclusion zone) and
ExponentialSaturationFit (a smooth, diffusion-interpretable correction) of
log1p(total_counts) against distance to the nearest border spot *within each
component* -- independently of bosperrus.Flow.flow()'s single-AIC-winner
selection, so both are always available regardless of which one AIC prefers.
Components with too few bins to fit meaningfully are dropped; see
_border_fit_from_adata's docstring for why distances are computed
per-component rather than pooled across all tissue.
"""
import gc
import sys
from pathlib import Path

import h5py
import numpy as np
import scanpy as sc
import squidpy
import tifffile
import zarr
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_rgb
from matplotlib.lines import Line2D
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from skimage.color import rgb2gray
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import binary_closing, disk, remove_small_holes, remove_small_objects
from skimage.segmentation import clear_border

import bosperrus
from bosperrus.fit import ConstantFit, PiecewiseLinearFit, ExponentialSaturationFit
from bosperrus.pipeline import Flow
from bosperrus.evaluate_fit import relative_likelihood, scaled_relative_likelihood
from bosperrus.distances import distance_to_pointset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blade import peel_sweep

STOMICS_DATA_DIR = Path("/home/woody/iwbn/iwbn007h/spatial_data/stomics")
_MASK_CACHE_DIR = Path(__file__).resolve().parents[2] / "results" / "exploratory" / "visium_tissue_masks"
_MASK_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def read_h5ad(h5ad_path):
    adata = sc.read_h5ad(h5ad_path)
    adata.obs["log1p_total_counts"] = np.log1p(np.asarray(adata.X.sum(axis=1)).ravel())
    return adata


def _fit_border_models(scores, d):
    """Fit ConstantFit (baseline), PiecewiseLinearFit, and
    ExponentialSaturationFit independently on the same (scores, d) data.

    Unlike bosperrus.Flow.flow() -- which fits every candidate model but
    only keeps the single AIC-best one, discarding the rest -- this keeps
    both non-baseline fits so callers can use PiecewiseLinearFit's elbow
    (discrete buffer zone) and ExponentialSaturationFit's smooth correction
    side by side, regardless of which one AIC actually prefers. Reproduces
    Flow.flow()'s AIC-weight/relative-likelihood bookkeeping by hand
    (Flow._set_entropy_weights, relative_likelihood, scaled_relative_likelihood)
    so Fit.params_summary() comes back fully populated for both, exactly as
    it would if either had won inside Flow.flow().
    """
    baseline = ConstantFit(scores, d)
    baseline.fit()

    piecewise_linear = PiecewiseLinearFit(scores, d)
    piecewise_linear.fit_correct()

    exponential_saturation = ExponentialSaturationFit(scores, d)
    exponential_saturation.fit_correct()

    for fit_instance in (piecewise_linear, exponential_saturation):
        fit_instance.relative_likelihood_over_baseline = relative_likelihood(fit_instance.AIC, baseline.AIC)
        fit_instance.scaled_relative_loglikelihood_over_baseline = scaled_relative_likelihood(
            fit_instance.AIC, baseline.AIC, len(d)
        )
    Flow._set_entropy_weights([baseline, piecewise_linear, exponential_saturation], baseline_fit=baseline)

    return {
        "baseline": baseline,
        "piecewise_linear": piecewise_linear,
        "exponential_saturation": exponential_saturation,
    }


def exp_sat_diffusion_params(fit, bin_size_um):
    """Reparametrize ExponentialSaturationFit's native S(d) = a*(1 - exp(-b*d)) + c
    into the diffusion-interpretable form S(d) = gamma*(1 - beta*exp(-lambda*d)):

        gamma  = a + c        asymptotic plateau (same units/value as the piecewise-
                               linear plateau)
        beta   = a / (a + c)  fractional deficit at the border: S(0) = c = gamma*(1-beta)
        lambda = b             decay rate -- already in the right form, no conversion needed

    This is an exact algebraic identity (same curve, same AIC, same
    .correct()), not a refit. Also reports the "decay length" 1/lambda (the
    exp-sat analogue of the piecewise-linear elbow) in grid-steps and um.
    """
    a = fit.params["exponential_saturation_a"]
    b = fit.params["exponential_saturation_b"]
    c = fit.params["exponential_saturation_c"]
    gamma = a + c
    beta = a / gamma
    decay_length_gridstep = 1 / b
    return {
        "gamma": gamma,
        "beta": beta,
        "lambda_gridstep": b,
        "lambda_per_um": b / bin_size_um,
        "decay_length_gridstep": decay_length_gridstep,
        "decay_length_um": decay_length_gridstep * bin_size_um,
    }


def rasterize_grid(array_row, array_col, values, fill=np.nan):
    """Place per-bin `values` onto a dense 2D array over the (array_row,
    array_col) bounding box (fill elsewhere, default NaN so ax.contour()
    doesn't draw spurious lines through gaps/holes in the tissue footprint).
    Returns (grid, row_offset, col_offset) so callers can align the grid back
    to (array_row, array_col) coordinates, e.g. for ax.contour(x, y, grid, ...).
    """
    row_offset = int(np.min(array_row))
    col_offset = int(np.min(array_col))
    n_rows = int(np.max(array_row)) - row_offset + 1
    n_cols = int(np.max(array_col)) - col_offset + 1
    grid = np.full((n_rows, n_cols), fill, dtype=float)
    grid[np.asarray(array_row) - row_offset, np.asarray(array_col) - col_offset] = values
    return grid, row_offset, col_offset


def sequential_colormap_from(hex_color, light_frac=0.85, n=256):
    """Build a sequential single-hue colormap: a light tint of `hex_color`
    (blended `light_frac` of the way to white, so near-zero values stay
    visible against a white figure background) through to the full color."""
    color = np.array(to_rgb(hex_color))
    light = color + (1 - color) * light_frac
    return LinearSegmentedColormap.from_list(f"seq_{hex_color}", [light, color], N=n)


def _border_fit_from_adata(adata, min_component_size=100):
    """Shared core of analyze_dataset()/analyze_stomics_dataset(): given an
    AnnData whose .obs already has array_row/array_col and log1p_total_counts
    (Visium's native columns; STOmics needs them derived first, see
    read_stomics_h5ad), identify tissue-border spots (grid NN<4), split the
    tissue into its spatially-connected components, and independently fit
    both PiecewiseLinearFit and ExponentialSaturationFit of
    log1p(total_counts) against distance to the nearest border spot *within
    each component*.

    Splitting matters not just for interpretability (e.g. a TMA's individual
    cores are separate biological samples, not one pooled blob) but for
    correctness: pooling distances across disconnected fragments (as an
    earlier version of this pipeline did) lets a bin in one fragment end up
    "nearest" to a border point belonging to a completely different,
    physically disconnected fragment that just happens to sit close by in
    raw grid coordinates -- not a meaningful distance for border-effect
    modeling. Computing distance_to_pointset separately per component avoids
    that; border detection itself (grid NN<4) is already a per-bin local
    property and doesn't need splitting.

    Components with <= min_component_size bins are dropped (too few points
    to fit a 3-parameter curve meaningfully). Returns a list of result dicts,
    one per surviving component (may be empty), each shaped like the
    previous single-result dict plus component_id/component_size. The
    AnnData is dropped before returning -- these files can be large, only
    small derived arrays are kept.

    Also carries along each bin's raw obsm["spatial"] pixel coordinate
    (spatial_x/spatial_y, both loaders populate this: full-res CytAssist
    pixel space for Visium, raw ssDNA/DNB pixel space for STOmics) --
    unused by the border-fit/BLADE/CSV pipeline itself, but this is the
    coordinate space get_sample_tissue_mask's image-derived masks are in,
    so plot_image_tissue_vs_counts can align the two without a second,
    separate load of the same AnnData.
    """
    # tissue-border points: grid spots with fewer than 4 grid-neighbors
    # (von Neumann/4-connectivity, radius = 1 grid step). A per-bin local
    # property, so computing it before splitting into components is fine --
    # it's only the *distance to* those border points that must be computed
    # per component (see docstring above).
    squidpy.gr.spatial_neighbors_grid(adata, n_rings=1, n_neighs=4)
    n_neighbors = np.diff(adata.obsp["spatial_connectivities"].indptr)
    border_mask_all = n_neighbors < 4

    _, labels = connected_components(adata.obsp["spatial_connectivities"], directed=False)
    component_sizes = np.bincount(labels)

    coords_grid_all = adata.obs[["array_row", "array_col"]].to_numpy()
    coords_pixel_all = adata.obsm["spatial"]
    scores_all = adata.obs["log1p_total_counts"]

    results = []
    for component_id in np.argsort(-component_sizes):
        size = int(component_sizes[component_id])
        if size <= min_component_size:
            continue
        member_mask = labels == component_id

        coords_grid = coords_grid_all[member_mask]
        coords_pixel = coords_pixel_all[member_mask]
        border_mask = border_mask_all[member_mask]
        border_coords_grid = coords_grid[border_mask]

        # distance to nearest tissue-border point *within this component only*
        dist_to_border = distance_to_pointset(coords_grid, border_coords_grid).rename("dist_to_tissue_border")
        scores = scores_all[member_mask].reset_index(drop=True)
        dist_to_border = dist_to_border.reset_index(drop=True)
        fits = _fit_border_models(scores, dist_to_border)

        results.append({
            "fits": fits,
            "array_row": coords_grid[:, 0],
            "array_col": coords_grid[:, 1],
            "spatial_x": coords_pixel[:, 0],
            "spatial_y": coords_pixel[:, 1],
            "border_mask": border_mask,
            "dist_to_border": dist_to_border.to_numpy(),
            "log1p_total_counts": scores.to_numpy(),
            "component_id": int(component_id),
            "component_size": size,
        })

    del adata
    gc.collect()
    return results


def analyze_dataset(h5ad_path, min_component_size=100):
    """Load one Visium HD sample (10x-hosted demo or in-house spaceranger
    run — both share the same square_{bin}um loader output) and run the
    shared border/BOSPERRUS pipeline per connected component; see
    _border_fit_from_adata. Returns a list of per-component result dicts."""
    return _border_fit_from_adata(read_h5ad(h5ad_path), min_component_size=min_component_size)


def read_stomics_h5ad(h5ad_path, bin_size=20):
    """Load a STOmics (Stereo-seq) bin*.h5ad and adapt it to the same
    array_row/array_col/log1p_total_counts shape Visium's loader produces.

    Unlike Visium HD, .X here already holds normalized/scaled values (not
    raw counts) — obs already has a precomputed raw `total_counts` QC column
    from the original pipeline, so that's log1p'd instead of summing X.
    Grid coordinates come from obsm["spatial"] (integer Stereo-seq DNB
    coordinates on a `bin_size`-unit lattice — nominal DNB pitch is 0.5um,
    so bin_size=20 -> 10um bins); dividing by bin_size gives the same
    small-integer array-row/col convention Visium uses, so
    _border_fit_from_adata needs no dataset-type-specific logic.
    """
    adata = sc.read_h5ad(h5ad_path)
    adata.obs["log1p_total_counts"] = np.log1p(adata.obs["total_counts"].to_numpy())
    grid = np.round(adata.obsm["spatial"] / bin_size).astype(int)
    adata.obs["array_col"] = grid[:, 0]
    adata.obs["array_row"] = grid[:, 1]
    return adata


def analyze_stomics_dataset(h5ad_path, bin_size=20, min_component_size=100):
    """STOmics counterpart of analyze_dataset — same border/BOSPERRUS
    pipeline, different loader (see read_stomics_h5ad). Returns a list of
    per-component result dicts."""
    return _border_fit_from_adata(read_stomics_h5ad(h5ad_path, bin_size=bin_size), min_component_size=min_component_size)


def load_counts_and_coords(loader_type, h5ad_path, bin_size=20):
    """h5py-only load of total counts, array_row/array_col and raw
    obsm["spatial"] -- skips building a full AnnData (PCA/neighbors/log1p
    layer/clustering all get pulled in by plain read_h5ad, unused here;
    confirmed via profiling: full load for the largest STOmics sample peaks
    ~19GB RSS vs. <1GB for this targeted read)."""
    with h5py.File(h5ad_path, "r") as f:
        spatial = f["obsm"]["spatial"][:]
        if loader_type == "visium":
            X = csr_matrix((f["X"]["data"][:], f["X"]["indices"][:], f["X"]["indptr"][:]),
                            shape=tuple(f["X"].attrs["shape"]))
            n_counts = np.asarray(X.sum(axis=1)).ravel()
            array_row, array_col = f["obs"]["array_row"][:], f["obs"]["array_col"][:]
        else:
            n_counts = f["obs"]["total_counts"][:]
            grid = np.round(spatial / bin_size).astype(int)
            array_col, array_row = grid[:, 0], grid[:, 1]
    return n_counts, array_row, array_col, spatial


def segment_tissue_from_rgb(image, sigma=8, close_radius=10, min_hole_area=50000, min_object_area=3000):
    """Simple, uniform tissue-vs-background segmentation for an H&E/CytAssist
    RGB image: grayscale -> heavy Gaussian blur -> Otsu threshold (tissue is
    darker than the white/light slide background) -> drop anything touching
    the image border -> morphological closing + small-hole-filling +
    small-object removal, to turn the raw threshold into a handful of solid
    tissue blobs instead of a speckled "nuclei only" mask (a single global
    Otsu on the *unblurred* grayscale image picks out only the darkest
    nuclei-dense foci, not the bulk tissue outline, since H&E has a lot of
    internal texture -- the blur washes that out first).

    clear_border matters for real samples, not just a defensive extra: Pat3's
    hires image has a faint scan-boundary artifact running almost the entire
    image perimeter (confirmed: rows/columns 0-15ish read as ~90%+ "tissue"
    before this step, dropping to baseline by row/col ~20) that Otsu alone
    reads as tissue -- too large in area for min_object_area to catch (a
    thin ring spanning a 6000x5200 image easily exceeds a few thousand
    pixels) but disconnected from the real tissue blobs, so clear_border
    removes it cleanly without touching them. None of this pipeline's real
    tissue blobs happen to touch the image edge in the 8 Visium samples this
    was checked against, so nothing else is lost by this step.

    One fixed parameter set for all samples (not tuned per-sample) --
    confirmed visually on both a single bulk tissue piece (breast_cancer,
    solid blob outline recovered cleanly) and ~100 small, closely-spaced TMA
    cores (breast_cancer_tma, cores stay distinct, none merged or dropped).
    """
    gray = rgb2gray(image)
    blurred = gaussian(gray, sigma=sigma)
    mask = blurred < threshold_otsu(blurred)
    mask = clear_border(mask)
    mask = binary_closing(mask, disk(close_radius))
    mask = remove_small_holes(mask, area_threshold=min_hole_area)
    mask = remove_small_objects(mask, min_size=min_object_area)
    return mask


def load_visium_hires_image(h5ad_path, library_id):
    """Read a Visium sample's own embedded hires CytAssist/H&E image (RGB
    array) straight from uns/spatial via h5py, rather than sc.read_h5ad --
    these files can hold a >1M-bin counts matrix, none of which this needs.
    Already a modest-sized "hires" image (Space Ranger's own downsample),
    safe to load fully into memory unlike STOmics' native ssDNA TIFFs.

    Returns (image, pixel_scale): pixel_scale converts a bin's raw
    obsm["spatial"] (full-res pixel) coordinate into this image's own
    (hires-image) pixel coordinate via hires_xy = obsm_spatial_xy *
    pixel_scale -- i.e. scalefactors["tissue_hires_scalef"], the same
    factor Space Ranger itself uses to align spots to the hires image.
    """
    with h5py.File(h5ad_path, "r") as f:
        spatial_group = f["uns"]["spatial"][library_id]
        image = spatial_group["images"]["hires"][:]
        pixel_scale = float(spatial_group["scalefactors"]["tissue_hires_scalef"][()])
    return image, pixel_scale


def load_visium_tissue_mask(h5ad_path, library_id):
    """Segment a Visium sample's own embedded hires image (see
    segment_tissue_from_rgb) as an image-only, counts-independent tissue
    estimate. Returns (mask, pixel_scale) -- see load_visium_hires_image's
    docstring for pixel_scale's meaning (mask and image share the same
    pixel grid, so the same pixel_scale applies to both).

    Cached to _MASK_CACHE_DIR/{library_id}.npz, keyed by h5ad_path --
    segment_tissue_from_rgb (Gaussian blur + Otsu + morphology) is expensive
    enough that ST.ipynb and border_sanity_checks.ipynb would otherwise both
    re-segment the same sample from scratch every run. Delete the cache file
    (or this whole directory) if segment_tissue_from_rgb's parameters change."""
    cache_path = _MASK_CACHE_DIR / f"{library_id}.npz"
    if cache_path.exists():
        cached = np.load(cache_path)
        if cached["h5ad_path"].item() == str(h5ad_path):
            return cached["mask"], cached["pixel_scale"].item()

    image, pixel_scale = load_visium_hires_image(h5ad_path, library_id)
    mask = segment_tissue_from_rgb(image)
    np.savez_compressed(cache_path, mask=mask, pixel_scale=pixel_scale, h5ad_path=str(h5ad_path))
    return mask, pixel_scale


def load_stomics_tissue_mask(tissue_mask_path):
    """Load BGI's own precomputed bulk tissue-region mask (already binary --
    no segmentation needed; see spatial_data/README.md's stomics/ section for
    provenance and why this is `*_ssDNA_tissue_cut.tif` specifically, not the
    same-shaped but visually-distinct `*_ssDNA_mask.tif`, which is a
    per-nucleus/cell segmentation, not a tissue-region mask).

    Same raw-pixel coordinate space as obsm["spatial"] -- confirmed directly
    (97-99% of every STOmics sample's bins land on a tissue_cut==1 pixel) --
    so pixel_scale is 1.0, unlike the Visium case.

    NOT counts-independent, unlike the Visium H&E case: the SAW pipeline's
    `tissuecut` step (see pipeline-logs/stereo_log) takes the raw expression
    matrix and per-spot read counts as direct inputs alongside the ssDNA
    image, not just the image. Using this mask as an independent reference
    for e.g. quantifying RNA-diffusion-driven signal outside the tissue
    boundary would be partly circular. A genuinely image-only alternative,
    segmented from the pre-tissuecut `*_ssDNA_regist.tif`, is precomputed at
    spatial_data/stomics/{name}_ssDNA_tissue_mask_final_{small,native}.npz --
    see that directory's README.md for how those were made and
    src/exploratory/segment_stomics_tissue_final.py to reproduce them.
    """
    mask = tifffile.imread(tissue_mask_path) > 0
    return mask, 1.0


def get_sample_tissue_mask(row):
    """Dispatch an image-derived tissue mask for one src/figure4/sample_manifest.csv
    row: Visium samples (loader == "visium") segment their own embedded
    image (load_visium_tissue_mask); STOmics samples (loader == "stomics")
    load BGI's precomputed tissue_cut mask from the row's tissue_mask_path
    column (STOmics h5ad files carry no embedded image at all -- see
    spatial_data/README.md). Returns (mask, pixel_scale) as documented in
    those two loaders.
    """
    if row["loader"] == "visium":
        return load_visium_tissue_mask(row["h5ad_path"], row["name"])
    elif row["loader"] == "stomics":
        return load_stomics_tissue_mask(row["tissue_mask_path"])
    raise ValueError(f"unknown loader {row['loader']!r}")


def load_image_mask(loader_type, h5ad_path, name):
    """Image-only tissue mask for a border-effect *reference* (not
    get_sample_tissue_mask/load_stomics_tissue_mask's tissue_cut: that mask
    is derived partly from counts, which would be circular here). STOmics
    reads the counts-independent alternative segmented from the pre-tissuecut
    ssDNA_regist.tif -- see load_stomics_tissue_mask's docstring."""
    if loader_type == "visium":
        return load_visium_tissue_mask(h5ad_path, name)
    npz = np.load(STOMICS_DATA_DIR / f"{name}_ssDNA_tissue_mask_final_native.npz")
    return npz["mask"], 1.0


def load_cropped_image(loader_type, h5ad_path, sample, row0, row1, col0, col1):
    """Visium's hires image is small -- load fully, then crop. STOmics' native
    ssDNA TIFF (~1-2GB decompressed) is read via zarr, decoding only the
    row-strips inside the crop (each row is its own compressed strip)."""
    if loader_type == "visium":
        image, _ = load_visium_hires_image(h5ad_path, sample)
        return image[row0:row1, col0:col1]
    store = tifffile.imread(STOMICS_DATA_DIR / f"{sample}_ssDNA_regist.tif", aszarr=True)
    z = zarr.open(store, mode="r")
    crop = np.asarray(z[row0:row1, col0:col1])
    store.close()
    return crop


def plot_image_tissue_vs_counts(ax, results, mask, pixel_scale, title=None,
                                 tissue_color="#a6bddb", counts_color="#d62728",
                                 scatter_size=0.5, scatter_alpha=0.25, margin_frac=0.05):
    """A second, image-only approach to where the tissue border sits: overlay
    the segmented/precomputed tissue mask (get_sample_tissue_mask -- derived
    purely from the H&E/ssDNA image, no counts involved) with every analyzed
    bin from all of a sample's connected components (the counts-derived
    tissue footprint the rest of this notebook works with), in the same
    pixel space.

    This is a purely visual comparison, cropped to the bins' own bounding
    box (+ margin) rather than the mask's full native canvas -- STOmics
    tissue_cut masks in particular span a much larger raw chip than the
    tissue region actually profiled, so showing the whole canvas would be
    mostly empty. Counts-bins landing outside the tissue-colored region
    (visible as red-on-white) show where RNA was captured beyond the image's
    own tissue call; tissue-colored regions with no red bins on them show
    the reverse gap. No distance/quantification is computed here -- see this
    function's caller for follow-up ideas.
    """
    spatial_x = np.concatenate([r["spatial_x"] for r in results]) * pixel_scale
    spatial_y = np.concatenate([r["spatial_y"] for r in results]) * pixel_scale

    margin_x = (spatial_x.max() - spatial_x.min()) * margin_frac
    margin_y = (spatial_y.max() - spatial_y.min()) * margin_frac
    row0 = max(0, int(spatial_y.min() - margin_y))
    row1 = min(mask.shape[0], int(spatial_y.max() + margin_y))
    col0 = max(0, int(spatial_x.min() - margin_x))
    col1 = min(mask.shape[1], int(spatial_x.max() + margin_x))
    cropped_mask = mask[row0:row1, col0:col1]

    tissue_cmap = ListedColormap(["white", tissue_color])
    ax.imshow(cropped_mask, cmap=tissue_cmap, vmin=0, vmax=1, extent=[col0, col1, row1, row0])
    ax.scatter(spatial_x, spatial_y, s=scatter_size, alpha=scatter_alpha, color=counts_color,
               linewidths=0, rasterized=True)

    if title:
        ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.axis("off")


def block_or_pool(arr, factor):
    """OR-reduce a boolean array in factor x factor blocks (cropping any
    remainder rows/cols that don't fill a full block). factor=1 returns arr
    unchanged. Used to downsample a boolean mask/occupancy layer to a modest
    resolution before contour/imshow -- see plot_mask_overlap_transparent's
    docstring for why this matters (naive downsampling aliases sparse
    boolean layers into illegible speckle)."""
    if factor <= 1:
        return arr
    h, w = arr.shape
    h2, w2 = (h // factor) * factor, (w // factor) * factor
    return arr[:h2, :w2].reshape(h2 // factor, factor, w2 // factor, factor).any(axis=(1, 3))


def plot_mask_overlap_transparent(ax, results, mask, pixel_scale, title=None,
                                   mask_color="#1f77b4", spot_color="#d62728",
                                   alpha=0.55, target_size=800, margin_frac=0.05):
    """Overlay the image mask and a rasterized bin mask as two independent
    translucent color layers, so overlap reads as a blended color while
    each mask alone stays in its own color.

    A first attempt at this (plot_mask_overlap_categorical, since removed)
    colored every pixel by one of 4 hard categories at the mask's *native*
    resolution -- looked like illegible "braille" once matplotlib had to
    downsample a huge sparse array into a small subplot, since isolated
    single-pixel categories (e.g. a lone bin-outside-mask pixel) get
    aliased away or reduced to scattered specks by that resize. Fixed here
    by explicitly max-pooling (OR-reducing) both boolean layers down to a
    common, modest resolution *before* plotting, so every rendered pixel
    faithfully means "was any of this true in this block" rather than an
    arbitrary decimated sample -- what's plotted is what's actually there.

    Returns (cropped_mask, cropped_spots), the two boolean layers actually
    rendered (post-crop, post-pooling), in case a caller wants them.
    """
    spatial_x = np.concatenate([r["spatial_x"] for r in results]) * pixel_scale
    spatial_y = np.concatenate([r["spatial_y"] for r in results]) * pixel_scale

    margin_x = (spatial_x.max() - spatial_x.min()) * margin_frac
    margin_y = (spatial_y.max() - spatial_y.min()) * margin_frac
    row0 = max(0, int(spatial_y.min() - margin_y))
    row1 = min(mask.shape[0], int(spatial_y.max() + margin_y))
    col0 = max(0, int(spatial_x.min() - margin_x))
    col1 = min(mask.shape[1], int(spatial_x.max() + margin_x))

    spot_raster = np.zeros(mask.shape, dtype=bool)
    rows = np.clip(np.round(spatial_y).astype(int), 0, mask.shape[0] - 1)
    cols = np.clip(np.round(spatial_x).astype(int), 0, mask.shape[1] - 1)
    spot_raster[rows, cols] = True

    cropped_mask = mask[row0:row1, col0:col1]
    cropped_spots = spot_raster[row0:row1, col0:col1]

    factor = max(1, round(cropped_mask.shape[0] / target_size))
    cropped_mask = block_or_pool(cropped_mask, factor)
    cropped_spots = block_or_pool(cropped_spots, factor)

    def transparent_cmap(hex_color):
        return ListedColormap([(0, 0, 0, 0), (*to_rgb(hex_color), alpha)])

    ax.imshow(cropped_mask, cmap=transparent_cmap(mask_color), vmin=0, vmax=1,
              extent=[col0, col1, row1, row0], interpolation="nearest")
    ax.imshow(cropped_spots, cmap=transparent_cmap(spot_color), vmin=0, vmax=1,
              extent=[col0, col1, row1, row0], interpolation="nearest")

    if title:
        ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.axis("off")
    return cropped_mask, cropped_spots


def blade_comparison(result, min_group_size=30):
    """Run blade.peel_sweep comparing raw vs. exponential-saturation-corrected
    log1p_total_counts for one analyze_dataset() result, against the same
    peel-layer masks. Corrects with the exp-sat fit specifically (not
    whichever fit AIC preferred): exp-sat's correction is the smooth,
    closed-form a*exp(-b*d) term (see exp_sat_diffusion_params's docstring
    for the a/b -> gamma/beta/lambda correspondence), added back onto raw
    counts. Returns (sweep_df, buffers) as documented in blade.peel_sweep.
    """
    raw = result["log1p_total_counts"]
    exp = result["fits"]["exponential_saturation"]
    a = exp.params["exponential_saturation_a"]
    b = exp.params["exponential_saturation_b"]
    correction = a * np.exp(-b * result["dist_to_border"])
    corrected = raw + correction
    return peel_sweep(
        result["array_row"], result["array_col"],
        {"raw": raw, "exp_sat_corrected": corrected},
        min_group_size=min_group_size,
    )


def _flatten_params_summary(prefix, fit):
    """Fit.params_summary() -> {f"{prefix}_{key}": value}, with spaces in
    bosperrus's own key names (e.g. "affected samples") turned into
    underscores for clean DataFrame/CSV column names. Adds AIC explicitly
    (params_summary() doesn't include it, only the AIC-derived weights)."""
    flat = {f"{prefix}_{key.replace(' ', '_')}": value for key, value in fit.params_summary().items()}
    flat[f"{prefix}_AIC"] = fit.AIC
    return flat


def summarize_all(name, dataset_type, result, bin_size_um, blade_min_group_size=30):
    """Comprehensive per-component summary row for the master results table:
    flattens Fit.params_summary() for both piecewise-linear and
    exponential-saturation fits (prefixed pl_/exp_), adds the diffusion-form
    exp-sat params, elbow/decay-length in both grid-steps and um, an
    AIC-based winner across all 3 models, BLADE buffers (raw and
    exp-sat-corrected, grid-steps and um), and basic component metadata
    (component_id/component_size, from analyze_dataset's per-component
    splitting). Every bosperrus-native quantity is kept alongside its
    um-converted counterpart.

    Note: params_summary()'s own "best_fit_type" key is just each fit's own
    name (e.g. always "Piecewise Linear Fit" for the pl_ columns) -- it does
    NOT mean "this was the AIC winner" here, since both fits are always
    computed regardless of AIC. `aic_best_model` below is the real winner.
    """
    fits = result["fits"]
    baseline, pl, exp = fits["baseline"], fits["piecewise_linear"], fits["exponential_saturation"]

    row = {
        "name": name,
        "dataset_type": dataset_type,
        "bin_size_um": bin_size_um,
        "component_id": result["component_id"],
        "component_size": result["component_size"],
        "n_bins": len(result["log1p_total_counts"]),
        "n_border_bins": int(result["border_mask"].sum()),
        "const_c": baseline.params["constant_c"],
        "const_AIC": baseline.AIC,
    }

    row.update(_flatten_params_summary("pl", pl))
    row["pl_affected_pct"] = row["pl_affected_samples"] * 100
    row["pl_elbow_grid_steps"] = pl.params["piecewise_linear_b"]
    row["pl_elbow_um"] = pl.params["piecewise_linear_b"] * bin_size_um

    row.update(_flatten_params_summary("exp", exp))
    row["exp_affected_pct"] = row["exp_affected_samples"] * 100
    for key, value in exp_sat_diffusion_params(exp, bin_size_um).items():
        row[f"exp_{key}"] = value

    row["aic_best_model"] = min(
        [("Constant", baseline.AIC), ("PiecewiseLinear", pl.AIC), ("ExponentialSaturation", exp.AIC)],
        key=lambda pair: pair[1],
    )[0]

    sweep_df, buffers = blade_comparison(result, min_group_size=blade_min_group_size)
    row["blade_buffer_raw_grid_steps"] = buffers.get("raw", np.nan)
    row["blade_buffer_raw_um"] = buffers.get("raw", np.nan) * bin_size_um
    row["blade_buffer_exp_sat_corrected_grid_steps"] = buffers.get("exp_sat_corrected", np.nan)
    row["blade_buffer_exp_sat_corrected_um"] = buffers.get("exp_sat_corrected", np.nan) * bin_size_um

    return row


def plot_counts_vs_distance_both_fits(ax, result, pl_color, exp_color, scatter_color="gray",
                                       title=None, ylabel=None,
                                       scatter_size=0.3, scatter_alpha=0.1, scatter_linewidths=0):
    """Scatter of counts vs. distance to the nearest tissue-border point, with
    BOTH the piecewise-linear (discrete buffer zone) and exponential-
    saturation (smooth, diffusion-interpretable) fits overlaid, regardless of
    which one AIC prefers. `ylabel` defaults to "log1p_total_counts"; pass ""
    to suppress it (e.g. for non-leftmost columns of a sharey row)."""
    d = result["dist_to_border"]
    s = result["log1p_total_counts"]
    ax.scatter(d, s, s=scatter_size, alpha=scatter_alpha, color=scatter_color,
               linewidths=scatter_linewidths, rasterized=True)

    d_line = np.linspace(0, d.max(), 200)

    pl = result["fits"]["piecewise_linear"]
    b, m, c = pl.params["piecewise_linear_b"], pl.params["piecewise_linear_m"], pl.params["piecewise_linear_c"]
    ax.plot(d_line, PiecewiseLinearFit.piecewise_plateau(d_line, b=b, m=m, c=c),
            color=pl_color, lw=2, label=f"piecewise-linear (elbow={b:.2f})")
    ax.axvline(b, color=pl_color, lw=1, ls="--", alpha=0.7)

    exp = result["fits"]["exponential_saturation"]
    ea, eb, ec = (exp.params["exponential_saturation_a"], exp.params["exponential_saturation_b"],
                  exp.params["exponential_saturation_c"])
    decay_length = 1 / eb
    ax.plot(d_line, ExponentialSaturationFit.exp_sat(d_line, ea, eb, ec),
            color=exp_color, lw=2, label=f"exp. saturation (decay length={decay_length:.2f})")

    ax.set_xlabel("distance to nearest tissue-border point (grid steps)")
    ax.set_ylabel("log1p_total_counts" if ylabel is None else ylabel)
    if title:
        ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="best")


def _clip_correction_to_data_range(correction, counts):
    """Clip one component's correction to its own observed count range. A
    correction bigger in magnitude than the entire observed dynamic range of
    the data it's supposedly correcting is definitionally a degenerate fit
    (e.g. a small/noisy component's exp-sat fit landing on a huge |a|), not a
    meaningful signal -- confirmed empirically (breast_cancer_tma_c4, 9220
    bins, fit a=-1268 against log1p_total_counts that only spans a handful
    of units; 3 of its parent sample's 44 components were similarly
    degenerate, together >1% of the sample's pooled bins, meaning
    percentile-based color clipping alone wasn't tight enough for every
    sample -- this per-component clip is the principled fix, done before
    pooling across components, rather than tuning the percentile threshold
    further per pathological case). Bounds magnitude without assuming sign.
    """
    max_reasonable = float(counts.max() - counts.min())
    if max_reasonable <= 0:
        return np.zeros_like(correction)
    return np.clip(correction, -max_reasonable, max_reasonable)


def _correction_color_range(correction, low_pct=1, high_pct=99):
    """Robust vmin/vmax for the correction colormap: percentile-based rather
    than raw min/max. A single degenerate component (e.g. a tiny, noisy
    fragment whose exp-sat fit blew up to a huge |a|) is a small fraction of
    the pooled bin count, so percentile clipping keeps it from hijacking the
    shared color/alpha scale for an entire panel of otherwise well-behaved
    components -- confirmed empirically (colon_cancer_ff's 113-bin
    component c6 fit a=-459, swamping its 1.72M-bin main component c0's
    a=1.01 on a raw min/max scale)."""
    vmin = min(0.0, float(np.percentile(correction, low_pct)))
    vmax = float(np.percentile(correction, high_pct))
    if vmax <= 0:
        vmax = max(float(correction.max()), 1e-12)
    return vmin, vmax


def _scale_alpha_by_correction(correction, vmax, max_alpha):
    """Per-point alpha proportional to correction magnitude (relative to
    `vmax`, see _correction_color_range), so bins with ~no correction fade to
    fully transparent (revealing the counts base layer, see
    plot_correction_map) instead of showing a visible tint -- only
    genuinely-corrected, near-border bins should read as colored at all."""
    if vmax <= 0:
        return np.zeros_like(correction)
    return np.clip(correction / vmax, 0, 1) * max_alpha


def plot_correction_map(ax, result, exp_color, boundary_color, title=None, add_legend=False,
                         scatter_size=0.5, scatter_alpha=0.3, counts_cmap="gray", counts_alpha=0.5,
                         cbar_label="exp. sat. correction"):
    """Grid-space scatter with the actual log1p(total_counts) as a base layer
    (so real tissue/count structure stays visible everywhere, not just where
    correction applies), overlaid with how much the exponential-saturation
    model would correct each bin (closed-form a*exp(-b*d) -- see
    exp_sat_diffusion_params's docstring for the a/b -> gamma/beta/lambda
    correspondence), using a sequential colormap built from shades of
    `exp_color`, with per-point alpha also scaled by correction magnitude
    (see _scale_alpha_by_correction) so uncorrected interior bins let the
    counts layer show through instead of a visible tint. The piecewise-
    linear elbow is drawn as an unfilled contour outline (not a filled
    highlight) around the bins within that distance -- mirrors row 1's own
    elbow-as-outline styling (micron_comparison._plot_poly_boundary),
    discrete-grid-appropriate implementation via rasterize_grid + ax.contour.
    """
    pl = result["fits"]["piecewise_linear"]
    exp = result["fits"]["exponential_saturation"]
    d_border = result["dist_to_border"]
    counts = result["log1p_total_counts"]
    a = exp.params["exponential_saturation_a"]
    b = exp.params["exponential_saturation_b"]
    correction = a * np.exp(-b * d_border)
    correction = _clip_correction_to_data_range(correction, counts)

    # base layer: actual counts, so tissue structure is visible everywhere,
    # not just where the correction overlay has something to show.
    ax.scatter(result["array_col"], result["array_row"], c=counts, cmap=counts_cmap,
               s=scatter_size, alpha=counts_alpha, rasterized=True)

    cmap = sequential_colormap_from(exp_color)
    vmin, vmax = _correction_color_range(correction)
    point_alpha = _scale_alpha_by_correction(correction, vmax, scatter_alpha)
    sca = ax.scatter(result["array_col"], result["array_row"], c=correction, cmap=cmap,
                      s=scatter_size, alpha=point_alpha, rasterized=True, vmin=vmin, vmax=vmax)
    ax.figure.colorbar(sca, ax=ax, label=cbar_label, shrink=0.75, pad=0.02)

    elbow = pl.params["piecewise_linear_b"]
    excluded = (d_border <= elbow).astype(float)
    grid, row_offset, col_offset = rasterize_grid(result["array_row"], result["array_col"], excluded)
    x_coords = col_offset + np.arange(grid.shape[1])
    y_coords = row_offset + np.arange(grid.shape[0])
    ax.contour(x_coords, y_coords, grid, levels=[0.5], colors=[boundary_color], linewidths=2)

    if title:
        ax.set_title(title)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")

    if add_legend:
        ax.legend(handles=[Line2D([0], [0], color=boundary_color, lw=2,
                                   label="piecewise-linear elbow (buffer zone)")],
                  fontsize=8, loc="lower left")


def plot_correction_map_multi(ax, results, exp_color, boundary_color, title=None, add_legend=False,
                               scatter_size=0.5, scatter_alpha=0.3, counts_cmap="gray", counts_alpha=0.5,
                               cbar_label="exp. sat. correction"):
    """Like plot_correction_map, but for a whole sample's worth of components
    at once (the list analyze_dataset()/analyze_stomics_dataset() returns) --
    e.g. all of a TMA's cores on one set of axes. Each component is colored
    by its *own* component-local exp-sat correction and gets its *own*
    piecewise-linear elbow outline (components were fit independently, see
    _border_fit_from_adata's docstring for why), but all components share
    one counts base layer, one color scale/colorbar, and one alpha scale.
    Each component's correction is first clipped to its own data range (see
    _clip_correction_to_data_range -- necessary, not just percentile
    clipping: a sample can have enough degenerate components that their
    pooled bins exceed the percentile cutoff), then the shared scale is
    percentile-robust on top of that (_correction_color_range/
    _scale_alpha_by_correction) so they're visually comparable within the
    panel.
    """
    all_col, all_row, all_counts, all_correction = [], [], [], []
    for result in results:
        exp = result["fits"]["exponential_saturation"]
        a = exp.params["exponential_saturation_a"]
        b = exp.params["exponential_saturation_b"]
        counts = result["log1p_total_counts"]
        correction = _clip_correction_to_data_range(a * np.exp(-b * result["dist_to_border"]), counts)
        all_col.append(result["array_col"])
        all_row.append(result["array_row"])
        all_counts.append(counts)
        all_correction.append(correction)
    all_col = np.concatenate(all_col)
    all_row = np.concatenate(all_row)
    all_counts = np.concatenate(all_counts)
    all_correction = np.concatenate(all_correction)

    # base layer: actual counts, so tissue structure is visible everywhere,
    # not just where the correction overlay has something to show.
    ax.scatter(all_col, all_row, c=all_counts, cmap=counts_cmap,
               s=scatter_size, alpha=counts_alpha, rasterized=True)

    cmap = sequential_colormap_from(exp_color)
    vmin, vmax = _correction_color_range(all_correction)
    point_alpha = _scale_alpha_by_correction(all_correction, vmax, scatter_alpha)
    sca = ax.scatter(all_col, all_row, c=all_correction, cmap=cmap,
                      s=scatter_size, alpha=point_alpha, rasterized=True, vmin=vmin, vmax=vmax)
    ax.figure.colorbar(sca, ax=ax, label=cbar_label, shrink=0.75, pad=0.02)

    for result in results:
        pl = result["fits"]["piecewise_linear"]
        d_border = result["dist_to_border"]
        elbow = pl.params["piecewise_linear_b"]
        excluded = (d_border <= elbow).astype(float)
        grid, row_offset, col_offset = rasterize_grid(result["array_row"], result["array_col"], excluded)
        x_coords = col_offset + np.arange(grid.shape[1])
        y_coords = row_offset + np.arange(grid.shape[0])
        ax.contour(x_coords, y_coords, grid, levels=[0.5], colors=[boundary_color], linewidths=1)

    if title:
        ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")

    if add_legend:
        ax.legend(handles=[Line2D([0], [0], color=boundary_color, lw=2,
                                   label="piecewise-linear elbow (buffer zone, per component)")],
                  fontsize=8, loc="lower left")
