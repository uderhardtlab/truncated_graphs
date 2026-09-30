"""BLADE-style border peeling, following KummerfeldLab/BLADE
(https://github.com/KummerfeldLab/BLADE, run_iterative_edge_cleanup):
Welch-t-test the outermost layer of tissue against everything deeper; while
p < 0.05, remove that layer and repeat; stop at the first layer with
p >= 0.05, which is kept. The buffer is the removed layers.

Layers are peeled with scipy.ndimage.binary_erosion. connectivity=8 (the
default) matches BLADE's Visium HD class (Artifact_remove_HD: k=8 nearest
bins, i.e. sides + diagonals); connectivity=4 peels through shared sides only.
"""
import numpy as np
import pandas as pd
from scipy import ndimage, stats


def peel_sweep(array_row, array_col, counts_by_label, min_group_size=30, connectivity=8,
               extra_row=None, extra_col=None):
    """Peel border layers off a (array_row, array_col) grid one at a time,
    Welch-t-testing each peeled layer against the remaining interior for
    every count array in counts_by_label (so e.g. raw vs. BOSPERRUS-corrected
    counts are compared against the same peel-layer masks).

    extra_row/extra_col: grid positions that belong to the tissue for peeling
    (e.g. filled hole positions without a bin) but have no counts -- they
    shape the layers and are skipped by the t-tests.

    Returns
    -------
    sweep_df : long-form DataFrame with columns
        counts, layer, p_value, n_border, n_interior, mean_border, mean_interior
        (n_* count bins with counts only)
    buffers : dict mapping each counts_by_label key to BLADE's buffer depth:
        the number of layers removed, i.e. (first layer with p_value >= 0.05) - 1.
        NaN if p_value never reaches 0.05 before the sweep stops.
    """
    array_row = np.asarray(array_row, dtype=np.int64)
    array_col = np.asarray(array_col, dtype=np.int64)
    extra_row = np.asarray([] if extra_row is None else extra_row, dtype=np.int64)
    extra_col = np.asarray([] if extra_col is None else extra_col, dtype=np.int64)
    all_row, all_col = np.concatenate([array_row, extra_row]), np.concatenate([array_col, extra_col])

    row0, col0 = all_row.min(), all_col.min()
    n_rows, n_cols = all_row.max() - row0 + 1, all_col.max() - col0 + 1
    ridx, cidx = array_row - row0, array_col - col0

    mask = np.zeros((n_rows, n_cols), dtype=bool)
    mask[all_row - row0, all_col - col0] = True
    has_counts = np.zeros((n_rows, n_cols), dtype=bool)
    has_counts[ridx, cidx] = True

    grids = {}
    for label, counts in counts_by_label.items():
        grid = np.full((n_rows, n_cols), np.nan)
        grid[ridx, cidx] = counts
        grids[label] = grid

    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")
    structure = ndimage.generate_binary_structure(2, 1 if connectivity == 4 else 2)
    rows = []
    layer = 0
    while mask.any():
        layer += 1
        eroded = ndimage.binary_erosion(mask, structure=structure, border_value=0)
        border_mask = mask & ~eroded & has_counts
        interior_mask = eroded & has_counts
        n_border, n_interior = int(border_mask.sum()), int(interior_mask.sum())
        if n_border < min_group_size or n_interior < min_group_size:
            break

        for label, grid in grids.items():
            border_vals = grid[border_mask]
            interior_vals = grid[interior_mask]
            p = stats.ttest_ind(border_vals, interior_vals, equal_var=False).pvalue
            rows.append({
                "counts": label, "layer": layer, "p_value": p,
                "n_border": n_border, "n_interior": n_interior,
                "mean_border": border_vals.mean(), "mean_interior": interior_vals.mean(),
            })
        mask = eroded

    sweep_df = pd.DataFrame(rows, columns=["counts", "layer", "p_value", "n_border", "n_interior",
                                           "mean_border", "mean_interior"])
    buffers = {}
    for label in counts_by_label:
        sub = sweep_df[sweep_df["counts"] == label]
        not_sig = sub["p_value"] >= 0.05
        buffers[label] = sub.loc[not_sig, "layer"].min() - 1 if not_sig.any() else np.nan
    return sweep_df, buffers


def plot_peel_sweep(ax, sweep_df, counts_labels, colors,
                     xlabel="peel iteration (topological layers from border)",
                     ylabel=None, sig_threshold=0.05, sig_line_color="gray",
                     ylim_bottom=None):
    """Plot p-value vs. peel layer for each label in counts_labels, from a
    long-form sweep_df (columns: 'counts', 'layer', 'p_value') as returned by
    peel_sweep. Draws a dashed vertical line, in each label's own color, at
    the first peel layer where that label's p-value stops being significant
    (BLADE keeps that layer; its buffer is the layers before it).

    ylim_bottom clips the (log-scale) y-axis at this value — Welch t-tests on
    large samples routinely produce p-values many orders of magnitude smaller
    than sig_threshold, which otherwise stretch the axis and squash the
    interesting region near sig_threshold. Points below it are simply out of
    view, not removed from sweep_df.
    """
    for label in counts_labels:
        sub = sweep_df[sweep_df["counts"] == label]
        if not len(sub):
            continue
        ax.plot(sub["layer"], sub["p_value"], marker="o", ms=3,
                color=colors[label], label=label)
        sig = sub["p_value"] >= sig_threshold
        if sig.any():
            buffer = sub.loc[sig, "layer"].min()
            ax.axvline(buffer, color=colors[label], lw=1, ls="--", alpha=0.8)
    ax.axhline(sig_threshold, color=sig_line_color, ls="--", lw=1)
    ax.set_yscale("log")
    if ylim_bottom is not None:
        ax.set_ylim(bottom=ylim_bottom)
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.legend(fontsize=7)
