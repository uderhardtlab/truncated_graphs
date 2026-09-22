"""Cache-backed BLADE + bosperrus border-effect fitting, one spatially-
connected grid component at a time. Shared by notebooks/ST.ipynb and
notebooks/supplement/border_sanity_checks.ipynb."""
import json
import sys
from datetime import datetime, timezone
from importlib.metadata import version as pkg_version
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

import bosperrus

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blade import peel_sweep

TRUNCATED_GRAPHS_DIR = Path(__file__).resolve().parents[2]
MANIFEST_PATH = TRUNCATED_GRAPHS_DIR / "src" / "figure4" / "sample_manifest.csv"
CACHE_DIR = TRUNCATED_GRAPHS_DIR / "results" / "exploratory" / "st_border_comparison"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

ALL_SAMPLES = pd.read_csv(MANIFEST_PATH)
FIT_PALETTE = json.loads((TRUNCATED_GRAPHS_DIR / "fit_palette.json").read_text())
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
    (array_row, array_col) grid (von Neumann/NN<4 adjacency, via
    bosperrus.grid_edges) -- e.g. a TMA's individual cores, kept separate so
    border-distance/BLADE stats never pool across physically disconnected
    tissue. Fragments with <= min_size spots are dropped.

    Returns [(component_rank, member_mask, size), ...], largest first.
    """
    edges = bosperrus.grid_edges(array_row, array_col, grid_type="rect")
    n = len(array_row)
    rows, cols = zip(*edges) if edges else ((), ())
    adjacency = csr_matrix((np.ones(2 * len(rows)), (rows + cols, cols + rows)), shape=(n, n))
    _, labels = connected_components(adjacency, directed=False)
    sizes = np.bincount(labels)

    components, rank = [], 0
    for label in np.argsort(-sizes):
        size = int(sizes[label])
        if size <= min_size:
            continue
        components.append((rank, labels == label, size))
        rank += 1
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
    """Per-spot Euclidean distance to the nearest border spot (grid degree <
    4, von Neumann adjacency)."""
    edges = bosperrus.grid_edges(array_row, array_col, grid_type="rect")
    degree = np.zeros(len(array_row), dtype=int)
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1
    coords = np.column_stack([array_row, array_col])
    return bosperrus.distance_to_pointset(coords, coords[degree < 4]).to_numpy()


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
