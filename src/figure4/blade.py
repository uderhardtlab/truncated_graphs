"""BLADE-style border peeling.

Iteratively identifies border points as grid positions with fewer than 4
active von-Neumann grid-neighbors (squidpy's spatial_neighbors_grid(n_rings=1,
n_neighs=4) NN<4 definition), Welch-t-tests each peeled layer against the
remaining interior, then strips ("peels") that layer and repeats on the
shrunken mask until it empties or a layer/interior group gets too small.
Implemented via scipy.ndimage.binary_erosion with a cross-shaped
(4-connectivity) structuring element rather than re-invoking squidpy every
iteration — mathematically identical to the NN<4 definition, but avoids
rebuilding a spatial graph on a shrinking AnnData every iteration.
"""
import numpy as np
import pandas as pd
from scipy import ndimage, stats


def peel_sweep(array_row, array_col, counts_by_label, min_group_size=30):
    """Peel border layers off a (array_row, array_col) grid one at a time,
    Welch-t-testing each peeled layer against the remaining interior for
    every count array in counts_by_label (so e.g. raw vs. BOSPERRUS-corrected
    counts are compared against the same peel-layer masks).

    Returns
    -------
    sweep_df : long-form DataFrame with columns
        counts, layer, p_value, n_border, n_interior, mean_border, mean_interior
    buffers : dict mapping each counts_by_label key to the smallest layer
        where p_value >= 0.05 (the BLADE-style buffer), or NaN if never reached.
    """
    array_row = np.asarray(array_row)
    array_col = np.asarray(array_col)

    row0, col0 = array_row.min(), array_col.min()
    n_rows, n_cols = array_row.max() - row0 + 1, array_col.max() - col0 + 1
    ridx, cidx = array_row - row0, array_col - col0

    mask = np.zeros((n_rows, n_cols), dtype=bool)
    mask[ridx, cidx] = True

    grids = {}
    for label, counts in counts_by_label.items():
        grid = np.full((n_rows, n_cols), np.nan)
        grid[ridx, cidx] = counts
        grids[label] = grid

    structure = ndimage.generate_binary_structure(2, 1)  # von Neumann / 4-connectivity
    rows = []
    layer = 0
    while mask.any():
        layer += 1
        eroded = ndimage.binary_erosion(mask, structure=structure, border_value=0)
        border_mask = mask & ~eroded
        n_border, n_interior = int(border_mask.sum()), int(eroded.sum())
        if n_border < min_group_size or n_interior < min_group_size:
            break

        for label, grid in grids.items():
            border_vals = grid[border_mask]
            interior_vals = grid[eroded]
            p = stats.ttest_ind(border_vals, interior_vals, equal_var=False).pvalue
            rows.append({
                "counts": label, "layer": layer, "p_value": p,
                "n_border": n_border, "n_interior": n_interior,
                "mean_border": border_vals.mean(), "mean_interior": interior_vals.mean(),
            })
        mask = eroded

    sweep_df = pd.DataFrame(rows)
    buffers = {}
    for label in counts_by_label:
        sub = sweep_df[sweep_df["counts"] == label]
        sig = sub["p_value"] >= 0.05
        buffers[label] = sub.loc[sig, "layer"].min() if sig.any() else np.nan
    return sweep_df, buffers


def plot_peel_sweep(ax, sweep_df, counts_labels, colors,
                     xlabel="peel iteration (topological layers from border)",
                     ylabel=None, sig_threshold=0.05, sig_line_color="gray"):
    """Plot p-value vs. peel layer for each label in counts_labels, from a
    long-form sweep_df (columns: 'counts', 'layer', 'p_value') as returned by
    peel_sweep. Draws a dashed vertical line, in each label's own color, at
    the first peel layer where that label's p-value stops being significant
    (i.e. its BLADE-style buffer).
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
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.legend(fontsize=7)
