"""Border-effect criterion: an empirical p-value against interior windows.

A rising piecewise-linear elbow (AIC vs. constant, m > 0) is described by
bosperrus' own fit descriptors:
    effect strength  observed_effect_strength: (fit at border - fit at d_max) / their sum
                     (negative for a dip at the border)
    half-life        um from the border to the midpoint of the rise
                     (observed_half_life x d_max; the package normalises by d_max, the
                     component's deepest fitted distance, which varies with section/core size)
Interior windows (interior_windows.py) -- squares well inside the tissue, fitted exactly
like the tissue edges -- are the null: no physical border, so any elbow there comes from
tissue structure.

Test statistic: the number of windows at least as extreme in both descriptors (at least
as deep AND at least as narrow) -- small = a deep, narrow dip. Its null distribution is the
same count for every window against the other windows (leave-one-out), and a component's
p-value is its rank in that distribution:
    p = (1 + #windows with count <= the component's count) / (1 + #windows)
(Using the raw fraction of more extreme windows as p is not calibrated in two dimensions:
it calls ~23% of the windows themselves significant at 0.05.) Windows without a rising
elbow have the largest possible count. A component is a border effect if p <= ALPHA,
the only parameter; nothing is tuned on the tissue edges.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ALPHA = 0.05


def criterion_metrics(info, d_max):
    """(effect strength, half-life in um) of one component's fit (an entry of
    identify_analysis_buffer_from_filtered's per_component); NaNs unless the
    winning fit is a rising elbow (border_effect)."""
    if not info["border_effect"]:
        return np.nan, np.nan
    return info["observed_effect_strength"], info["observed_half_life"] * d_max


def _n_as_extreme(effect_strength, half_life_um, null_eff, null_half):
    """Number of null windows at least as deep and at least as narrow (inf without a rising elbow)."""
    if not np.isfinite(effect_strength):
        return np.inf
    return int(((null_eff <= effect_strength) & (null_half <= half_life_um)).sum())  # NaN windows compare False


def load_window_null(window_dir, samples, suffix=""):
    """Window null from {sample}_windows{suffix}.csv in window_dir: every window's effect
    strength and half-life (NaN without a rising elbow) and its test statistic against
    the other windows."""
    frames = [pd.read_csv(f) for f in (Path(window_dir) / f"{s}_windows{suffix}.csv" for s in samples)
              if f.exists() and f.stat().st_size > 1]
    w = pd.concat(frames)
    eff, half = w["effect_strength"].to_numpy(float), w["half_life_um"].to_numpy(float)
    count = np.array([_n_as_extreme(e, h, eff, half) - 1 for e, h in zip(eff, half)])  # -1: itself
    return {"effect_strength": eff, "half_life_um": half, "count": count}


def window_p_value(effect_strength, half_life_um, null):
    """Empirical p-value of one elbow against the window null (NaN if no rising elbow)."""
    if not np.isfinite(effect_strength):
        return np.nan
    count = _n_as_extreme(effect_strength, half_life_um, null["effect_strength"], null["half_life_um"])
    return (1 + int((null["count"] <= count).sum())) / (1 + len(null["count"]))


def passes(p_value):
    """Border effect: p <= ALPHA (False for NaN, i.e. no rising elbow)."""
    return bool(p_value <= ALPHA)


def false_positive_rate(null, windows=None):
    """Fraction of windows called a border effect. windows=None: the null's own windows,
    each against the others (leave-one-out) -- the criterion's false-positive rate. Else
    another window set (e.g. load_window_null of other samples) against `null`."""
    if windows is None:
        count = null["count"]
        p = np.array([(1 + int((np.delete(count, i) <= c).sum())) / len(count) for i, c in enumerate(count)])
        p[~np.isfinite(count)] = np.nan
    else:
        p = np.array([window_p_value(e, h, null) for e, h in zip(windows["effect_strength"], windows["half_life_um"])])
    return float(np.mean(p <= ALPHA))
