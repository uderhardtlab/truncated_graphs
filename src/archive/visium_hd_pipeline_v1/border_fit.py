"""Cache-backed BLADE + bosperrus border-effect fitting, one spatially-
connected grid component at a time. Shared by notebooks/ST.ipynb and
notebooks/border_sanity_checks.ipynb."""
import json
import sys
from datetime import datetime, timezone
from importlib.metadata import version as pkg_version
from pathlib import Path

import numpy as np
import pandas as pd

import bosperrus

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blade import peel_sweep

TRUNCATED_GRAPHS_DIR = Path(__file__).resolve().parents[2]
MANIFEST_PATH = TRUNCATED_GRAPHS_DIR / "src" / "figure4" / "sample_manifest.csv"
CACHE_DIR = TRUNCATED_GRAPHS_DIR / "results" / "exploratory" / "st_border_comparison"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

ALL_SAMPLES = pd.read_csv(MANIFEST_PATH)
FIT_PALETTE = bosperrus.FIT_PALETTE
BOSPERRUS_VERSION = pkg_version("bosperrus")
MIN_COMPONENT_SPOTS = 1000


def _jsonify(obj):
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def get_grid_components(array_row, array_col, min_size=MIN_COMPONENT_SPOTS):
    """Split a sample into its spatially-connected fragments on the
    (array_row, array_col) grid (von Neumann/NN<4 adjacency) -- e.g. a TMA's
    individual cores, kept separate so border-distance/BLADE stats never
    pool across physically disconnected tissue. Fragments with <= min_size
    spots are dropped.

    Thin wrapper around bosperrus.split_into_connected_components, adapting
    its per-node label array into this module's own [(component_rank,
    member_mask, size), ...] shape (largest first) -- kept for backward
    compatibility with existing callers (ST.ipynb, border_sanity_checks.ipynb,
    notebooks/tutorials/bosperrus_on_TMA.ipynb).
    """
    labels = bosperrus.split_into_connected_components(array_row, array_col, grid_type="rect", min_size=min_size)
    components = []
    for rank in sorted(r for r in np.unique(labels) if r >= 0):
        member_mask = labels == rank
        components.append((int(rank), member_mask, int(member_mask.sum())))
    return components


def native_pixel_size_um(array_row, array_col, spatial, bin_size_um, max_edges=2000):
    """um per native obsm['spatial'] pixel, measured empirically from the
    known physical grid pitch (bin_size_um) vs. the pixel distance between
    grid-adjacent bins -- unlike grid_border_distance's grid-step distance,
    a mask-based distance_to_mask fit runs in this native pixel space, which
    has no fixed physical size (varies per Visium scan; confirmed: two
    Visium samples here measured 0.27 and 0.18 um/px). Cast to float64
    first -- STOmics stores obsm['spatial'] as uint32, and a plain
    difference silently wraps around instead of going negative."""
    spatial = spatial.astype(np.float64)
    edges = list(bosperrus.grid_edges(array_row, array_col, grid_type="rect"))
    if len(edges) > max_edges:
        edges = [edges[i] for i in np.random.default_rng(0).choice(len(edges), max_edges, replace=False)]
    u, v = zip(*edges)
    pixel_pitch = np.median(np.linalg.norm(spatial[list(u)] - spatial[list(v)], axis=1))
    return bin_size_um / pixel_pitch


def grid_border_distance(array_row, array_col):
    """Per-spot Euclidean distance (in grid steps) to the nearest border spot
    (grid degree < 4, von Neumann adjacency), computed within each
    spatially-connected component separately.

    Thin wrapper around bosperrus.distance_to_grid_border -- bin_size_um=1.0
    keeps this function's original grid-step units (callers multiply by the
    real bin_size_um themselves), and for "rect" that unit choice is an exact
    isotropic scaling (see distance_to_grid_border's docstring), so this is
    numerically identical to the old hand-rolled version, just no longer
    duplicating its own copy of the connected-components/border-detection
    logic.
    """
    return bosperrus.distance_to_grid_border(
        array_row, array_col, bin_size_um=1.0, grid_type="rect",
    ).to_numpy()


def _fit_component(array_row, array_col, n_counts, bin_size_um):
    _, blade_result = peel_sweep(array_row=array_row, array_col=array_col, counts_by_label={"n_counts": n_counts})
    blade_thresh = blade_result["n_counts"]

    dist_to_border = grid_border_distance(array_row, array_col)
    flow = bosperrus.Flow.from_distances_and_scores(
        distances=pd.Series(dist_to_border, name="distance_to_border"),
        scores=pd.DataFrame({"n_counts": n_counts}),
    )
    flow.flow(fits=[bosperrus.ConstantFit, bosperrus.PiecewiseLinearFit])
    bsp_fit = flow.best_fits["n_counts"]
    bsp_thresh = bsp_fit.params["piecewise_linear_b"]

    exp_sat = bosperrus.ExponentialSaturationFit(n_counts, dist_to_border)
    n_counts_corrected = np.asarray(exp_sat.fit_correct())
    _, blade_result_corrected = peel_sweep(
        array_row=array_row, array_col=array_col, counts_by_label={"n_counts_corrected": n_counts_corrected}
    )
    blade_thresh_corrected = blade_result_corrected["n_counts_corrected"]

    return {
        "blade_thresh": blade_thresh,
        "blade_thresh_um": blade_thresh * bin_size_um,
        "bsp_fit": {
            "best_fit_type": bsp_fit.name,
            "params": dict(bsp_fit.params),
            "observed_effect_strength": bsp_fit.observed_effect_strength,
            "observed_half_life": bsp_fit.observed_half_life,
            "affected_fraction": bsp_fit.fraction_not_converged,
            "grid_type": "rect",
        },
        "bsp_thresh": bsp_thresh,
        "bsp_thresh_um": bsp_thresh * bin_size_um,
        "correction_fit": {
            "model": "exponential_saturation",
            "params": exp_sat.params,
            "AIC": exp_sat.AIC,
            "observed_effect_strength": exp_sat.observed_effect_strength,
            "observed_half_life": exp_sat.observed_half_life,
            "fraction_not_converged": exp_sat.fraction_not_converged,
        },
        "blade_thresh_corrected": blade_thresh_corrected,
        "blade_thresh_corrected_um": blade_thresh_corrected * bin_size_um,
    }


def get_or_compute_component_fit(sample, component_rank, component_size, h5ad_path, bin_size_um,
                                  array_row, array_col, n_counts):
    """BLADE + bosperrus thresholds for one connected component, cached to a
    JSON file (keyed by sample + component) shared across notebooks."""
    cache_path = CACHE_DIR / f"{sample}_{component_rank}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        if (cached["bosperrus_version"], cached["h5ad_path"], cached["component_size"]) == \
           (BOSPERRUS_VERSION, h5ad_path, component_size):
            return cached

    result = _fit_component(array_row, array_col, n_counts, bin_size_um)
    result.update(
        sample=sample, component_rank=component_rank, component_size=component_size,
        h5ad_path=h5ad_path, bin_size_um=bin_size_um, bosperrus_version=BOSPERRUS_VERSION,
        computed_at=datetime.now(timezone.utc).isoformat(),
    )
    cache_path.write_text(json.dumps(_jsonify(result), indent=2))
    return result
