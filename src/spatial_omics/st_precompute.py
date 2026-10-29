"""Per-sample precomputation for notebooks/spatial_omics/ST_{demo,paired}.ipynb and
border_sanity_checks_{demo,paired}.ipynb, run on a compute node (see
st_precompute.sbatch). Everything here goes through the two bosperrus entry
points -- nothing is re-implemented locally except BLADE (blade.py, not part
of bosperrus):

- border effect: bosperrus.load_filtered + identify_analysis_buffer_from_filtered
  on the pipeline's own tissue-filtered bins (n_counts > 0, per connected
  component), plus BLADE.
- diffusion: bosperrus.quantify_diffusion_from_raw on the raw, whole-capture-
  area bins vs. an image-only tissue mask (segmentation parameters per sample
  from st_samples.csv).

All cross-technology comparisons run at RESOLUTION_UM = 8 (Visium HD's
native 8um bins; Stereo-seq bin1 pooled 16x16).

Usage (cwd = truncated_graphs/):
    pixi run python src/spatial_omics/st_precompute.py <sample>              # 8um: buffer + diffusion
    pixi run python src/spatial_omics/st_precompute.py <sample> --ablation   # diffusion only, all ABLATION_RESOLUTIONS
    pixi run python src/spatial_omics/st_precompute.py <sample> --image-mask # border analysis: raw bins inside the image mask,
                                             # elbow vs. distance to the tissue edge (main figures)
    pixi run python src/spatial_omics/st_precompute.py <sample> --in-tissue  # same on Space Ranger's in_tissue bins (supplement)
    pixi run python src/spatial_omics/st_precompute.py <sample> --blade-image-border
                                             # spots discarded by BLADE (in_tissue) vs. bosperrus (image border), Jaccard
                                             # (needs --image-mask, --in-tissue and the interior windows)
    pixi run python src/spatial_omics/st_precompute.py <sample> --tissue-edge --resolution 2
                                             # elbow only (no BLADE) at any bin size, zero-count bins kept (resolution check)

Every elbow fit is rising-only (bosperrus.RisingPiecewiseLinearFit) and uses only bins within
TISSUE_FIT_MAX_DISTANCE_UM of the border.

Writes to results/spatial_omics/st_reroll/:
    {sample}_filtered_bins.parquet  one row per filtered bin (+ virtual=True rows for filled
                                    hole positions without a bin): array_row, array_col,
                                    n_counts, components, virtual, border, distance_to_border (um),
                                    analysis_buffer, blade_layer, in_blade_buffer,
                                    mask_row, mask_col
    {sample}_buffer.json            per-component elbow + BLADE thresholds
    {sample}_diffusion_{res}um.json quantify_diffusion_from_raw's result dict
    {sample}_diffusion_8um.npz      mask, display image, and every raw bin's
                                    array_row/array_col/mask_row/mask_col/distance_um/n_counts
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

import bosperrus

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blade import peel_sweep  # noqa: E402

TRUNCATED_GRAPHS = Path(__file__).resolve().parents[2]
SAMPLES = pd.read_csv(Path(__file__).resolve().parent / "st_samples.csv").set_index("name")
OUT_DIR = TRUNCATED_GRAPHS / "results" / "spatial_omics" / "st_reroll"

RESOLUTION_UM = 8.0
MIN_COMPONENT_SPOTS = 1000
MAX_HOLE_AREA_UM2 = 1024.0  # fill enclosed empty holes up to 16 bins at 8um (see identify_analysis_buffer_from_filtered)
# the elbow is fitted only on bins within this distance of the border: ~10x the observed elbows, short
# enough that a narrow edge dip is not swamped by tissue-scale gradients, and the depth the smallest
# interior windows (500 um, see interior_windows.py) reach, so edges and windows share one fit range
TISSUE_FIT_MAX_DISTANCE_UM = 250.0
ABLATION_RESOLUTIONS = {
    "visium_hd": [2.0, 8.0, 16.0],
    "stereo-seq": [2.0, 4.0, 8.0, 10.0, 16.0, 25.0],
}
DISPLAY_LONG_EDGE = 2000  # display image saved to the npz is block-averaged down to about this size


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _jsonify(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


BLADE_CONNECTIVITY = 8  # BLADE's Visium HD neighbourhood (sides + diagonals)


def _blade_layer(array_row, array_col):
    """Peel layer of every position of one component (1 = outermost),
    exactly blade.peel_sweep's layers: iterated 8-connectivity
    binary_erosion == chessboard distance to the nearest non-member position."""
    array_row, array_col = np.asarray(array_row, dtype=np.int64), np.asarray(array_col, dtype=np.int64)
    r0, c0 = array_row.min() - 1, array_col.min() - 1
    grid = np.zeros((array_row.max() - r0 + 2, array_col.max() - c0 + 2), dtype=bool)
    grid[array_row - r0, array_col - c0] = True
    depth = ndimage.distance_transform_cdt(grid, metric="chessboard")
    return depth[array_row - r0, array_col - c0]


def _nearest_value(row, col, values, query_row, query_col):
    """Value of the nearest (row, col) node for each query position -- used to
    give virtual hole positions (no bin) a display value for the rasters."""
    if len(query_row) == 0:
        return np.array([], dtype=values.dtype)
    from scipy.spatial import cKDTree
    _, idx = cKDTree(np.stack([row, col], axis=1)).query(np.stack([query_row, query_col], axis=1))
    return values[idx]


def run_buffer(sample, technology, path, adata=None, footprint="pipeline"):
    """identify_analysis_buffer_from_filtered + BLADE per component, both on the
    same tissue footprint: n_counts > 0 bins with small enclosed holes
    (<= MAX_HOLE_AREA_UM2) filled.

    Returns the filtered-bins DataFrame -- one row per filtered bin plus one
    row per filled hole position without a bin (virtual=True, n_counts NaN;
    they shape borders/layers but are never fit or tested). Mask coordinates
    are filled in later by run_main.

    adata: the bins to analyse; by default the pipeline's own tissue-filtered
    bins (bosperrus.load_filtered: Space Ranger's in_tissue / SAW's
    tissue.gef). run_image_mask passes the raw bins inside the image mask
    instead (footprint="image_mask")."""
    if adata is None:
        _log(f"load_filtered({technology}, {RESOLUTION_UM}um)")
        adata = bosperrus.load_filtered(path, technology, RESOLUTION_UM)
    _log(f"  {adata.n_obs:,} input bins ({footprint}); identify_analysis_buffer_from_filtered ...")
    bosperrus.identify_analysis_buffer_from_filtered(
        adata, min_component_size=MIN_COMPONENT_SPOTS, max_hole_area_um2=MAX_HOLE_AREA_UM2,
        rising_only=True,
    )
    fit_info = adata.uns["analysis_buffer_fit"]
    virtual = fit_info["filled_hole_positions"]
    v_row, v_col = np.asarray(virtual["array_row"]), np.asarray(virtual["array_col"])
    v_comp = np.asarray(virtual["components"])
    _log(f"  filled holes: {fit_info['n_filled_hole_bins']:,} zero-count bins kept, "
         f"{len(v_row):,} positions without a bin")

    obs = adata.obs
    array_row, array_col = obs["array_row"].to_numpy(), obs["array_col"].to_numpy()
    n_counts, components = obs["n_counts"].to_numpy(), obs["components"].to_numpy()
    distance = obs["distance_to_border"].to_numpy()
    buffer = obs["analysis_buffer"].to_numpy()

    blade_layer = np.zeros(len(obs), dtype=np.int32)
    v_layer = np.zeros(len(v_row), dtype=np.int32)
    in_blade = np.zeros(len(obs), dtype=bool)
    v_in_blade = np.zeros(len(v_row), dtype=bool)
    per_component = {}
    for label, info in fit_info["per_component"].items():
        member, v_member = components == label, v_comp == label
        _log(f"  component {label}: {member.sum():,} bins (+{v_member.sum()} hole positions), "
             f"{info['best_fit_type']}; BLADE ...")
        peel = dict(connectivity=BLADE_CONNECTIVITY, extra_row=v_row[v_member], extra_col=v_col[v_member])
        sweep, blade = peel_sweep(array_row[member], array_col[member], {"n_counts": n_counts[member]}, **peel)
        # BLADE buffer depth: number of layers removed (p < 0.05); NaN if the sweep never reached p >= 0.05
        n_removed = blade["n_counts"]

        layer = _blade_layer(np.concatenate([array_row[member], v_row[v_member]]),
                             np.concatenate([array_col[member], v_col[v_member]]))
        n_real = int(member.sum())
        blade_layer[member], v_layer[v_member] = layer[:n_real], layer[n_real:]
        flag = (layer <= n_removed) if np.isfinite(n_removed) else np.zeros(len(layer), bool)
        in_blade[member], v_in_blade[v_member] = flag[:n_real], flag[n_real:]

        bsp_thresh_um = info["elbow_um"]  # None unless a piecewise fit with m > 0 won
        per_component[int(label)] = {
            **info,
            "bsp_thresh_um": bsp_thresh_um,
            "pct_affected": float(buffer[member].mean() * 100),
            "n_filled_hole_bins": int((member & (n_counts <= 0)).sum()),
            "n_hole_positions_without_bin": int(v_member.sum()),
            "blade_thresh": n_removed,  # layers removed by BLADE (8-connectivity peel layers)
            "blade_thresh_um": n_removed * RESOLUTION_UM,
            "max_distance_um": float(np.nanmax(distance[member])),  # distance range the elbow fit saw
            "blade_sweep": sweep.to_dict(orient="list"),
        }

    buffer_json = {
        "sample": sample, "technology": technology, "path": str(path), "resolution_um": RESOLUTION_UM,
        "footprint": footprint, "min_component_size": MIN_COMPONENT_SPOTS, "max_hole_area_um2": MAX_HOLE_AREA_UM2,
        "blade_connectivity": BLADE_CONNECTIVITY, "n_filtered_bins": int(adata.n_obs),
        "n_filled_hole_bins": fit_info["n_filled_hole_bins"], "n_hole_positions_without_bin": int(len(v_row)),
        "per_component": per_component,
    }
    filtered = pd.DataFrame({
        "array_row": array_row.astype(np.int32), "array_col": array_col.astype(np.int32),
        "n_counts": n_counts.astype(np.float32), "components": components.astype(np.int32),
        "virtual": False,
        "border": obs["border"].to_numpy(), "distance_to_border": distance.astype(np.float64),
        "analysis_buffer": buffer, "blade_layer": blade_layer,
        "in_blade_buffer": in_blade,
    })
    if len(v_row):
        real = components >= 0
        virtual_df = pd.DataFrame({
            "array_row": v_row.astype(np.int32), "array_col": v_col.astype(np.int32),
            "n_counts": np.float32(np.nan), "components": v_comp.astype(np.int32), "virtual": True,
            "border": False, "distance_to_border": np.nan,
            # display only (virtual positions are never fit): the nearest real bin's buffer flag
            "analysis_buffer": _nearest_value(array_row[real], array_col[real], buffer[real], v_row, v_col),
            "blade_layer": v_layer, "in_blade_buffer": v_in_blade,
        })
        filtered = pd.concat([filtered, virtual_df], ignore_index=True)
    return filtered, buffer_json


def run_diffusion(technology, path, mask_kwargs, resolution, return_data):
    _log(f"quantify_diffusion_from_raw({technology}, {resolution}um, mask_kwargs={mask_kwargs})")
    out = bosperrus.quantify_diffusion_from_raw(path, technology, resolution, mask_kwargs=mask_kwargs,
                                                return_data=return_data)
    result = out[0] if return_data else out
    _log(f"  {result['best_fit_type']}: alpha={result['alpha']}, beta={result['beta']}, "
         f"perc_counts_outside={result['perc_counts_outside']:.3f}, n_bins={result['n_bins_total']:,}")
    return out


def _display_image(image):
    """Block-average to about DISPLAY_LONG_EDGE px; uint8. Returns (image, factor)."""
    from skimage.transform import downscale_local_mean
    factor = max(1, int(round(max(image.shape[:2]) / DISPLAY_LONG_EDGE)))
    block = (factor, factor) + ((1,) if image.ndim == 3 else ())
    small = downscale_local_mean(np.asarray(image, dtype=np.float32), block)
    if image.ndim == 3:  # Visium hires RGB, float in [0, 1]
        small = np.clip(small * 255, 0, 255)
    else:  # ssDNA intensity: stretch to the 99.5th percentile
        small = np.clip(small / np.percentile(small, 99.5) * 255, 0, 255)
    return small.astype(np.uint8), factor


def run_main(sample):
    row = SAMPLES.loc[sample]
    technology, path, mask_kwargs = row["technology"], row["path"], json.loads(row["mask_kwargs"])

    filtered, buffer_json = run_buffer(sample, technology, path)
    result, data = run_diffusion(technology, path, mask_kwargs, RESOLUTION_UM, return_data=True)

    # filtered bins sit on the same 8um grid as the raw bins -> take their mask
    # coordinates straight from the raw data (exact, no coordinate conversion)
    raw_index = pd.Series(np.arange(len(data["array_row"])),
                          index=pd.MultiIndex.from_arrays([data["array_row"], data["array_col"]]))
    idx = raw_index.reindex(pd.MultiIndex.from_arrays([filtered["array_row"], filtered["array_col"]])).to_numpy()
    if np.isnan(idx).any():
        raise ValueError(f"{np.isnan(idx).sum()} filtered bins not found on the raw grid")
    idx = idx.astype(np.int64)
    filtered["mask_row"] = data["mask_row"][idx].astype(np.float32)
    filtered["mask_col"] = data["mask_col"][idx].astype(np.float32)
    filtered["distance_outside_mask_um"] = data["distance_um"][idx].astype(np.float32)

    image, display_factor = _display_image(data["image"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    filtered.to_parquet(OUT_DIR / f"{sample}_filtered_bins.parquet")
    (OUT_DIR / f"{sample}_buffer.json").write_text(json.dumps(_jsonify(buffer_json), indent=1))
    result = {"sample": sample, "dataset_type": row["dataset_type"], **result}
    (OUT_DIR / f"{sample}_diffusion_{RESOLUTION_UM:g}um.json").write_text(json.dumps(_jsonify(result), indent=1))
    np.savez_compressed(
        OUT_DIR / f"{sample}_diffusion_{RESOLUTION_UM:g}um.npz",
        mask=data["mask"], mask_pixel_size_um=data["mask_pixel_size_um"],
        image=image, image_factor=display_factor,  # image pixel (i, j) covers mask pixels [i*f, (i+1)*f)
        array_row=data["array_row"].astype(np.int32), array_col=data["array_col"].astype(np.int32),
        mask_row=data["mask_row"].astype(np.float32), mask_col=data["mask_col"].astype(np.float32),
        distance_um=data["distance_um"].astype(np.float32), n_counts=data["n_counts"].astype(np.float32),
    )
    _log(f"wrote {sample} outputs to {OUT_DIR}")


def distance_to_tissue_edge(raw, inside):
    """Distance (um) of every bin inside the image mask to the edge of the whole image mask
    (tissue beyond the capture area still counts as tissue; the mask never touches the image
    edge, see bosperrus.image_masks.segment_tissue_from_rgb)."""
    mask = raw["mask"].astype(bool)
    inside_px = ndimage.distance_transform_edt(mask) * float(raw["mask_pixel_size_um"])
    r = np.clip(np.rint(raw["mask_row"][inside]).astype(np.int64), 0, mask.shape[0] - 1)
    c = np.clip(np.rint(raw["mask_col"][inside]).astype(np.int64), 0, mask.shape[1] - 1)
    return inside_px[r, c]


def run_image_mask(sample):
    """Border analysis of the main figures: run_buffer on the raw 8um bins inside the
    image-only tissue mask (distance to the mask == 0, exactly the mask used by
    quantify_diffusion_from_raw; read from the main run's {sample}_diffusion_8um.npz)
    instead of the pipeline's own tissue call. The elbow is fitted against the
    distance to the edge of the whole image mask (cuts by the capture area are not a
    border), on bins within TISSUE_FIT_MAX_DISTANCE_UM. n_counts > 0, hole filling,
    components and BLADE as in run_buffer.
    Writes {sample}_buffer_imagemask.json and {sample}_filtered_bins_imagemask.parquet."""
    import anndata
    row = SAMPLES.loc[sample]
    npz_path = OUT_DIR / f"{sample}_diffusion_{RESOLUTION_UM:g}um.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"{npz_path} missing -- run the main job for {sample} first")
    raw = np.load(npz_path)
    inside = raw["distance_um"] == 0
    obs = pd.DataFrame({
        "array_row": raw["array_row"][inside].astype(np.int64),
        "array_col": raw["array_col"][inside].astype(np.int64),
        "n_counts": raw["n_counts"][inside].astype(np.float64),
    })
    obs.index = obs["array_row"].astype(str) + "_" + obs["array_col"].astype(str)
    # an externally supplied distance column is reused as-is by identify_analysis_buffer_from_filtered
    full = distance_to_tissue_edge(raw, inside)
    obs["distance_to_border"] = np.where(full <= TISSUE_FIT_MAX_DISTANCE_UM, full, np.nan)
    adata = anndata.AnnData(obs=obs)
    adata.uns["bosperrus"] = {"technology": row["technology"], "bin_size_um": RESOLUTION_UM, "path": str(row["path"])}
    _log(f"{int(inside.sum()):,} raw bins inside the image mask "
         f"({int((obs['n_counts'] > 0).sum()):,} with counts)")

    filtered, buffer_json = run_buffer(sample, row["technology"], row["path"], adata=adata,
                                     footprint="image_mask (distance to tissue edge, fit within 250 um)")
    # same 8um grid as the npz -> mask coordinates by (array_row, array_col) lookup
    raw_index = pd.Series(np.arange(len(raw["array_row"])),
                          index=pd.MultiIndex.from_arrays([raw["array_row"], raw["array_col"]]))
    idx = raw_index.reindex(pd.MultiIndex.from_arrays([filtered["array_row"], filtered["array_col"]])).to_numpy()
    idx = idx.astype(np.int64)
    filtered["mask_row"] = raw["mask_row"][idx].astype(np.float32)
    filtered["mask_col"] = raw["mask_col"][idx].astype(np.float32)
    filtered.to_parquet(OUT_DIR / f"{sample}_filtered_bins_imagemask.parquet")
    (OUT_DIR / f"{sample}_buffer_imagemask.json").write_text(json.dumps(_jsonify(buffer_json), indent=1))
    _log(f"wrote {sample} image-mask outputs to {OUT_DIR}")


def run_in_tissue(sample):
    """run_buffer on Space Ranger's in_tissue bins (as the main run), but the
    elbow is fitted only on bins within TISSUE_FIT_MAX_DISTANCE_UM of the
    footprint edge (same fit range as --image-mask).
    Writes {sample}_buffer_intissue.json and {sample}_filtered_bins_intissue.parquet."""
    row = SAMPLES.loc[sample]
    adata = bosperrus.load_filtered(row["path"], row["technology"], RESOLUTION_UM)
    # first pass only computes components + footprint distance (reused below with the same parameters)
    bosperrus.identify_analysis_buffer_from_filtered(
        adata, min_component_size=MIN_COMPONENT_SPOTS, max_hole_area_um2=MAX_HOLE_AREA_UM2,
    )
    full = adata.obs["distance_to_border"].to_numpy()
    adata.obs["distance_to_border"] = np.where(full <= TISSUE_FIT_MAX_DISTANCE_UM, full, np.nan)
    filtered, buffer_json = run_buffer(sample, row["technology"], row["path"], adata=adata,
                                       footprint="in_tissue (fit within 250 um)")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    filtered.to_parquet(OUT_DIR / f"{sample}_filtered_bins_intissue.parquet")
    (OUT_DIR / f"{sample}_buffer_intissue.json").write_text(json.dumps(_jsonify(buffer_json), indent=1))
    _log(f"wrote {sample} in_tissue outputs to {OUT_DIR}")


def min_component_bins(resolution):
    """MIN_COMPONENT_SPOTS (set at RESOLUTION_UM) as the same area at another bin size."""
    return int(round(MIN_COMPONENT_SPOTS * (RESOLUTION_UM / resolution) ** 2))


def load_raw(sample, resolution):
    """Raw bins of the whole capture area at `resolution` with their image-mask coordinates
    ({sample}_diffusion_{res}um.npz; written here if missing -- the 8um one by the main run)."""
    npz_path = OUT_DIR / f"{sample}_diffusion_{resolution:g}um.npz"
    if not npz_path.exists():
        row = SAMPLES.loc[sample]
        _, data = run_diffusion(row["technology"], row["path"], json.loads(row["mask_kwargs"]), resolution,
                                return_data=True)
        np.savez_compressed(
            npz_path, mask=data["mask"], mask_pixel_size_um=data["mask_pixel_size_um"],
            array_row=data["array_row"].astype(np.int32), array_col=data["array_col"].astype(np.int32),
            mask_row=data["mask_row"].astype(np.float32), mask_col=data["mask_col"].astype(np.float32),
            distance_um=data["distance_um"].astype(np.float32), n_counts=data["n_counts"].astype(np.float32),
        )
    return dict(np.load(npz_path))


def run_tissue_edge(sample, resolution):
    """Panel a only, at any bin size: every raw bin inside the image mask (zero-count bins
    kept -- at 2um many tissue bins have no counts, and dropping them would bias the profile),
    elbow vs. distance to the whole image mask's edge within TISSUE_FIT_MAX_DISTANCE_UM,
    no BLADE. Run at 8 and 2um with identical settings for the resolution check.
    Writes {sample}_buffer_tissueedge_{res}um.json and {sample}_filtered_bins_tissueedge_{res}um.parquet."""
    import anndata
    raw = load_raw(sample, resolution)
    inside = raw["distance_um"] == 0
    full = distance_to_tissue_edge(raw, inside)
    obs = pd.DataFrame({
        "array_row": raw["array_row"][inside].astype(np.int64), "array_col": raw["array_col"][inside].astype(np.int64),
        "n_counts": raw["n_counts"][inside].astype(np.float64),
        "distance_to_border": np.where(full <= TISSUE_FIT_MAX_DISTANCE_UM, full, np.nan),
    })
    obs.index = obs["array_row"].astype(str) + "_" + obs["array_col"].astype(str)
    adata = anndata.AnnData(obs=obs)
    adata.uns["bosperrus"] = {"bin_size_um": resolution}
    min_bins = min_component_bins(resolution)
    _log(f"{int(inside.sum()):,} bins inside the image mask at {resolution:g}um "
         f"({int((obs['n_counts'] > 0).sum()):,} with counts); min component {min_bins:,} bins")
    bosperrus.identify_analysis_buffer_from_filtered(
        adata, n_counts_key=None, min_component_size=min_bins, max_hole_area_um2=None, rising_only=True,
    )
    components = adata.obs["components"].to_numpy()
    buffer = adata.obs["analysis_buffer"].to_numpy()
    per_component = {}
    for label, info in adata.uns["analysis_buffer_fit"]["per_component"].items():
        member = components == label
        per_component[int(label)] = {
            **info, "pct_affected": float(buffer[member].mean() * 100), "n_bins": int(member.sum()),
            "max_distance_um": float(np.nanmax(obs["distance_to_border"].to_numpy()[member])),
        }
        _log(f"  component {label}: {member.sum():,} bins, {info['best_fit_type']}, elbow {info['elbow_um']}")
    suffix = f"_tissueedge_{resolution:g}um"
    pd.DataFrame({
        "array_row": obs["array_row"].astype(np.int32), "array_col": obs["array_col"].astype(np.int32),
        "n_counts": obs["n_counts"].astype(np.float32), "components": components.astype(np.int32),
        "distance_to_border": obs["distance_to_border"].astype(np.float32), "distance_full_um": full.astype(np.float32),
        "analysis_buffer": buffer,
    }).to_parquet(OUT_DIR / f"{sample}_filtered_bins{suffix}.parquet")
    (OUT_DIR / f"{sample}_buffer{suffix}.json").write_text(json.dumps(_jsonify({
        "sample": sample, "resolution_um": resolution, "min_component_size": min_bins,
        "zero_count_bins_kept": True, "rising_only": True, "fit_max_distance_um": TISSUE_FIT_MAX_DISTANCE_UM,
        "per_component": per_component,
    }), indent=1))
    _log(f"wrote {sample}{suffix} outputs to {OUT_DIR}")


def run_blade_image_border(sample):
    """Spots discarded by BLADE as published vs. by bosperrus, both on Space Ranger's in_tissue bins (the
    --in-tissue run's footprint, components and BLADE buffer). bosperrus discards the in_tissue bins outside
    the image mask, and those closer to the image tissue edge than the elbow of their image-mask component
    (--image-mask run; nearest component) when that elbow has p <= ALPHA against the interior windows
    (border_criterion). Jaccard index of the two discarded sets.
    Writes {sample}_blade_imageborder.json and {sample}_bins_blade_imageborder.parquet."""
    from scipy.spatial import cKDTree
    from border_criterion import criterion_metrics, load_window_null, passes, window_p_value
    demo = SAMPLES.index[SAMPLES["dataset_type"] == "visium_demo"]
    null = load_window_null(OUT_DIR / "interior_windows", demo)

    raw = load_raw(sample, RESOLUTION_UM)
    inside = raw["distance_um"] == 0
    to_edge = np.zeros(len(inside))
    to_edge[inside] = distance_to_tissue_edge(raw, inside)
    raw_index = pd.Series(np.arange(len(inside)), index=pd.MultiIndex.from_arrays([raw["array_row"], raw["array_col"]]))

    # image-mask components: p-value and elbow
    img = pd.read_parquet(OUT_DIR / f"{sample}_filtered_bins_imagemask.parquet")
    img = img[~img["virtual"] & (img["components"] >= 0)]
    img_info = json.loads((OUT_DIR / f"{sample}_buffer_imagemask.json").read_text())["per_component"]
    comp_fit = {}
    for label, g in img.groupby("components"):
        info = img_info[str(label)]
        fitted = g[np.isfinite(g["distance_to_border"])]
        p = window_p_value(*criterion_metrics(info, fitted["distance_to_border"].max()), null)
        comp_fit[int(label)] = {"p_value": p, "passes": passes(p), "elbow_um": info["elbow_um"]}

    # in_tissue bins (BLADE buffer from the --in-tissue run)
    it = pd.read_parquet(OUT_DIR / f"{sample}_filtered_bins_intissue.parquet")
    real = it[~it["virtual"] & (it["components"] >= 0)].copy()
    idx = raw_index.reindex(pd.MultiIndex.from_arrays([real["array_row"], real["array_col"]])).to_numpy().astype(np.int64)
    real["inside_image_mask"] = inside[idx]
    real["distance_image_edge_um"] = to_edge[idx]
    _, nearest = cKDTree(img[["array_row", "array_col"]].to_numpy()).query(real[["array_row", "array_col"]].to_numpy())
    real["image_component"] = img["components"].to_numpy()[nearest]

    bsp_discard = ~real["inside_image_mask"].to_numpy()
    for label, entry in comp_fit.items():
        if entry["passes"]:
            member = (real["image_component"] == label).to_numpy()
            bsp_discard[member] |= real["distance_image_edge_um"].to_numpy()[member] < entry["elbow_um"]
    real["bosperrus_discard"] = bsp_discard

    blade_discard = real["in_blade_buffer"].to_numpy()
    union = int((blade_discard | bsp_discard).sum())
    jaccard = int((blade_discard & bsp_discard).sum()) / union if union else np.nan
    out = {"sample": sample, "alpha_null": "8 um demo interior windows", "jaccard_discarded": jaccard,
           "n_in_tissue": int(len(real)), "n_discard_blade": int(blade_discard.sum()),
           "n_discard_bosperrus": int(bsp_discard.sum()), "n_discard_both": int((blade_discard & bsp_discard).sum()),
           "image_components": comp_fit}
    (OUT_DIR / f"{sample}_blade_imageborder.json").write_text(json.dumps(_jsonify(out), indent=1))
    real[["array_row", "array_col", "components", "image_component", "inside_image_mask", "distance_image_edge_um",
          "in_blade_buffer", "bosperrus_discard"]].to_parquet(OUT_DIR / f"{sample}_bins_blade_imageborder.parquet")
    _log(f"wrote {sample}: Jaccard (discarded) {jaccard:.3f}")


def run_ablation(sample):
    row = SAMPLES.loc[sample]
    technology, path, mask_kwargs = row["technology"], row["path"], json.loads(row["mask_kwargs"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for resolution in ABLATION_RESOLUTIONS[technology]:
        out_path = OUT_DIR / f"{sample}_diffusion_{resolution:g}um.json"
        if resolution == RESOLUTION_UM and out_path.exists():
            _log(f"{out_path.name} exists (main run), skipping")
            continue
        result = run_diffusion(technology, path, mask_kwargs, resolution, return_data=False)
        result = {"sample": sample, "dataset_type": row["dataset_type"], **result}
        out_path.write_text(json.dumps(_jsonify(result), indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sample", choices=list(SAMPLES.index))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--ablation", action="store_true")
    mode.add_argument("--image-mask", action="store_true",
                      help="border analysis on the raw bins inside the image mask, distance to the tissue edge "
                           "(needs the main run's npz)")
    mode.add_argument("--in-tissue", action="store_true",
                      help="border analysis on Space Ranger's in_tissue bins, distance to their footprint edge")
    mode.add_argument("--blade-image-border", action="store_true",
                      help="spots discarded by BLADE (in_tissue) vs. bosperrus (image border), Jaccard index")
    mode.add_argument("--tissue-edge", action="store_true",
                      help="tissue-edge elbow only (zero-count bins kept, no BLADE) at --resolution")
    parser.add_argument("--resolution", type=float, default=RESOLUTION_UM, help="with --tissue-edge: bin size (um)")
    args = parser.parse_args()
    name = ("ablation" if args.ablation else "image-mask" if args.image_mask
            else "in-tissue" if args.in_tissue else "tissue-edge" if args.tissue_edge
            else "blade-image-border" if args.blade_image_border else "main")
    _log(f"=== {args.sample} ({name}) ===")
    if name == "image-mask":
        run_image_mask(args.sample)
    elif name == "tissue-edge":
        run_tissue_edge(args.sample, args.resolution)
    else:
        {"ablation": run_ablation, "in-tissue": run_in_tissue, "blade-image-border": run_blade_image_border,
         "main": run_main}[name](args.sample)
    _log("done")
