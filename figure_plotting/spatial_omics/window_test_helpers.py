"""Per-component interior-window test (bosperrus.interior_window_test, run by src/spatial_omics/st_precompute.py)
as read by the ST notebooks: test columns from a buffer-JSON component entry and the three-state scatter.

Status of a component:
    significant      rising elbow, p <= ALPHA against interior windows of the same component (buffer kept)
    not significant  rising elbow, p > ALPHA (buffer dropped)
    untested         rising elbow, too few interior windows fit (component < ~0.8 mm across at the 100 um fit depth; buffer kept as fitted)
    none             no rising elbow
"""
import numpy as np
import seaborn as sns

ALPHA = 0.05


def _nan(v):
    return np.nan if v is None else v


def status(info):
    if not info["border_effect"]:
        return "none"
    if info.get("p_value") is None:
        return "untested"
    return "significant" if info["significant"] else "not significant"


def test_columns(info):
    """Effect strength, half-life (um), steepness, p-value, number of windows, status and whether the buffer is
    kept (significant or untested) of one per_component entry."""
    st = status(info)
    effect = st != "none"
    return {
        "effect_strength": _nan(info["observed_effect_strength"]) if effect else np.nan,
        "half_life_um": _nan(info["half_life_um"]) if effect else np.nan,
        "steepness": _nan(info.get("steepness")),
        "p_value": _nan(info.get("p_value")),
        "n_windows": info.get("n_windows", 0),
        "status": st,
        "buffer_kept": st in ("significant", "untested"),
    }


def scatter_by_status(ax, df, linewidth=0.8, **kw):
    """sns.scatterplot of df (kw: x, y, hue, size, style, markers, ...) in three layers: significant filled,
    not significant open, untested as a filled X (style/markers ignored). Components without an elbow are drawn
    like 'not significant' (open), at whatever stand-in position the caller gives them."""
    kw = dict(kw, ax=ax, legend=False)
    sig = df[df["status"] == "significant"]
    if len(sig):
        sns.scatterplot(sig, **kw)
    open_ = df[df["status"].isin(["not significant", "none"])]
    if len(open_):
        sns.scatterplot(open_, **kw)
        coll = ax.collections[-1]
        coll.set_edgecolors(coll.get_facecolors())
        coll.set_facecolors("none")
        coll.set_linewidths(linewidth)
    untested = df[df["status"] == "untested"]
    if len(untested):
        kx = {k: v for k, v in kw.items() if k not in ("style", "markers")}
        sns.scatterplot(untested, marker="X", linewidth=0, **kx)
