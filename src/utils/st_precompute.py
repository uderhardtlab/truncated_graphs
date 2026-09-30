"""Per-sample precomputation for notebooks/ST.ipynb and
notebooks/border_sanity_checks.ipynb, run on a compute node (see
st_precompute.sbatch). Everything here goes through the two bosperrus entry
points -- nothing is re-implemented locally except BLADE (blade.py, not part
of bosperrus):

- border effect: bosperrus.load_filtered + identify_analysis_buffer_from_filtered
  on the pipeline's own tissue-filtered bins (n_counts > 0, per connected
  component), plus BLADE on raw and ExponentialSaturationFit-corrected counts.
- diffusion: bosperrus.quantify_diffusion_from_raw on the raw, whole-capture-
  area bins vs. an image-only tissue mask (segmentation parameters per sample
  from st_samples.csv).

All cross-technology comparisons run at RESOLUTION_UM = 8 (Visium HD's
native 8um bins; Stereo-seq bin1 pooled 16x16).

Usage (cwd = truncated_graphs/):
    pixi run python src/utils/st_precompute.py <sample>              # 8um: buffer + diffusion
    pixi run python src/utils/st_precompute.py <sample> --ablation   # diffusion only, all ABLATION_RESOLUTIONS

Writes to results/exploratory/st_reroll/:
    {sample}_filtered_bins.parquet  one row per filtered bin (+ virtual=True rows for filled
                                    hole positions without a bin): array_row, array_col,
                                    n_counts, components, virtual, border, distance_to_border (um),
                                    analysis_buffer, blade_layer, in_blade_buffer,
                                    in_blade_buffer_corrected, mask_row, mask_col
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
OUT_DIR = TRUNCATED_GRAPHS / "results" / "exploratory" / "st_reroll"

RESOLUTION_UM = 8.0
MIN_COMPONENT_SPOTS = 1000
MAX_HOLE_AREA_UM2 = 1024.0  # fill enclosed empty holes up to 16 bins at 8um (see identify_analysis_buffer_from_filtered)
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


def run_buffer(sample, technology, path):
    """identify_analysis_buffer_from_filtered + BLADE (raw and exp-sat
    corrected) per component, both on the same tissue footprint: n_counts > 0
    bins with small enclosed holes (<= MAX_HOLE_AREA_UM2) filled.

    Returns the filtered-bins DataFrame -- one row per filtered bin plus one
    row per filled hole position without a bin (virtual=True, n_counts NaN;
    they shape borders/layers but are never fit or tested). Mask coordinates
    are filled in later by run_main."""
    _log(f"load_filtered({technology}, {RESOLUTION_UM}um)")
    adata = bosperrus.load_filtered(path, technology, RESOLUTION_UM)
    _log(f"  {adata.n_obs:,} filtered bins; identify_analysis_buffer_from_filtered ...")
    bosperrus.identify_analysis_buffer_from_filtered(
        adata, min_component_size=MIN_COMPONENT_SPOTS, max_hole_area_um2=MAX_HOLE_AREA_UM2,
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
    in_blade_corr = np.zeros(len(obs), dtype=bool)
    v_in_blade = np.zeros(len(v_row), dtype=bool)
    v_in_blade_corr = np.zeros(len(v_row), dtype=bool)
    per_component = {}
    for label, info in fit_info["per_component"].items():
        member, v_member = components == label, v_comp == label
        _log(f"  component {label}: {member.sum():,} bins (+{v_member.sum()} hole positions), "
             f"{info['best_fit_type']}; BLADE ...")
        peel = dict(connectivity=BLADE_CONNECTIVITY, extra_row=v_row[v_member], extra_col=v_col[v_member])
        sweep, blade = peel_sweep(array_row[member], array_col[member], {"n_counts": n_counts[member]}, **peel)
        exp_sat = bosperrus.ExponentialSaturationFit(n_counts[member], distance[member])
        n_counts_corrected = np.asarray(exp_sat.fit_correct())
        sweep_corr, blade_corr = peel_sweep(array_row[member], array_col[member],
                                            {"n_counts_corrected": n_counts_corrected}, **peel)
        # BLADE buffer depth: number of layers removed (p < 0.05); NaN if the sweep never reached p >= 0.05
        n_removed, n_removed_corr = blade["n_counts"], blade_corr["n_counts_corrected"]

        layer = _blade_layer(np.concatenate([array_row[member], v_row[v_member]]),
                             np.concatenate([array_col[member], v_col[v_member]]))
        n_real = int(member.sum())
        blade_layer[member], v_layer[v_member] = layer[:n_real], layer[n_real:]
        flag = (layer <= n_removed) if np.isfinite(n_removed) else np.zeros(len(layer), bool)
        flag_corr = (layer <= n_removed_corr) if np.isfinite(n_removed_corr) else np.zeros(len(layer), bool)
        in_blade[member], v_in_blade[v_member] = flag[:n_real], flag[n_real:]
        in_blade_corr[member], v_in_blade_corr[v_member] = flag_corr[:n_real], flag_corr[n_real:]

        bsp_thresh_um = info["elbow_um"]  # None unless a piecewise fit with m > 0 won
        per_component[int(label)] = {
            **info,
            "bsp_thresh_um": bsp_thresh_um,
            "pct_affected": float(buffer[member].mean() * 100),
            "n_filled_hole_bins": int((member & (n_counts <= 0)).sum()),
            "n_hole_positions_without_bin": int(v_member.sum()),
            "blade_thresh": n_removed,  # layers removed by BLADE (8-connectivity peel layers)
            "blade_thresh_corrected": n_removed_corr,
            "blade_thresh_um": n_removed * RESOLUTION_UM,
            "blade_thresh_corrected_um": n_removed_corr * RESOLUTION_UM,
            "exp_sat_converged": bool(exp_sat._converged),
            "blade_sweep": sweep.to_dict(orient="list"),
            "blade_sweep_corrected": sweep_corr.to_dict(orient="list"),
        }

    buffer_json = {
        "sample": sample, "technology": technology, "path": str(path), "resolution_um": RESOLUTION_UM,
        "min_component_size": MIN_COMPONENT_SPOTS, "max_hole_area_um2": MAX_HOLE_AREA_UM2,
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
        "in_blade_buffer": in_blade, "in_blade_buffer_corrected": in_blade_corr,
    })
    if len(v_row):
        real = components >= 0
        virtual_df = pd.DataFrame({
            "array_row": v_row.astype(np.int32), "array_col": v_col.astype(np.int32),
            "n_counts": np.float32(np.nan), "components": v_comp.astype(np.int32), "virtual": True,
            "border": False, "distance_to_border": np.nan,
            # display only (virtual positions are never fit): the nearest real bin's buffer flag
            "analysis_buffer": _nearest_value(array_row[real], array_col[real], buffer[real], v_row, v_col),
            "blade_layer": v_layer, "in_blade_buffer": v_in_blade, "in_blade_buffer_corrected": v_in_blade_corr,
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
    parser.add_argument("--ablation", action="store_true")
    args = parser.parse_args()
    _log(f"=== {args.sample} ({'ablation' if args.ablation else 'main'}) ===")
    (run_ablation if args.ablation else run_main)(args.sample)
    _log("done")
