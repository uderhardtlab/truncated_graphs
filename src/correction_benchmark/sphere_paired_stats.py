"""Paired statistics behind the sphere-benchmark violins (supplementary figure/table).

Same data as the violins: Spearman correlation between the original (whole-sphere) and the cropped
(corrected) scores, all nodes, graph types delaunay / knn / rnn, |rho|.

Independent unit: one point set = (point pattern, seed), 3 patterns x 100 seeds = 300 per panel. The k-NN and
r-NN graphs (k = 5, 10, 15; r-NN radius = mean distance to the k-th neighbour) are built on the same point set,
so their per-graph differences are averaged within the point set before testing (the violins pool them).

For BOSPERRUS (additive; the log variant is not benchmarked) against each of raw, spline, linear and SERN,
per graph type x measure, with delta = rho_BOSPERRUS - rho_other per point set:
  median_delta, ci_low/ci_high  median and 95% percentile-bootstrap CI over point sets (BOOTSTRAP resamples)
  win_rate                      fraction of point sets with delta > 0 (ties count half)
  p_better                      one-sided Wilcoxon signed-rank, delta > 0
  p_noninferior                 one-sided Wilcoxon signed-rank, delta > -MARGIN
  p_worse                       one-sided Wilcoxon signed-rank, delta < -MARGIN
Each p-value family is Holm-corrected over all its tests (columns *_holm). Verdict (ALPHA, MARGIN):
  better          p_better_holm <= ALPHA and median_delta > MARGIN
  non-inferior    otherwise, p_noninferior_holm <= ALPHA
  worse           p_worse_holm <= ALPHA
  inconclusive    none of these
Second table: per panel, the method with the best median against the runner-up (same paired test,
two-sided Wilcoxon, Holm over panels) -- whether the violin figure's winner star is a clear win.

Run from truncated_graphs/src:
    pixi run python correction_benchmark/sphere_paired_stats.py
Writes ../results/correction_benchmark/sphere_paired_stats.csv and sphere_winner_vs_runner_up.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from load_results import load_sphere
from scipy.stats import wilcoxon

RESULTS = Path(__file__).resolve().parents[2] / "results" / "correction_benchmark"
GRAPHS = ["delaunay", "knn", "rnn"]
MEASURES = ["betweenness", "closeness", "clustering", "degree", "pagerank"]
COLUMNS = {"original vs. on crop": "raw", "original vs. BOSPERRUS corrected on crop": "bosperrus",
           "original vs. linear detrended on crop": "linear",
           "original vs. spline detrended on crop": "spline", "original vs. SERN corrected on crop": "SERN"}
METHODS = list(COLUMNS.values())
BOSPERRUS = ["bosperrus"]
COMPETITORS = ["raw", "spline", "linear", "SERN"]
UNIT = ["coord_type", "seed"]  # one point set
MARGIN = 0.01  # smallest difference in rho that counts
ALPHA = 0.05
BOOTSTRAP = 2000
SEED = 0


def load():
    """The violins' data: Spearman, all nodes, |rho| (as figure_plotting/correction_benchmark)."""
    df = load_sphere(RESULTS).rename(columns=COLUMNS)
    df = df[(df["corr_method"] == "spearman") & df["graph_type"].isin(GRAPHS) & df["measure"].isin(MEASURES)]
    if "node_subset" in df:
        df = df[df["node_subset"].fillna("all") == "all"]
    df = df.copy()
    df[METHODS] = df[METHODS].abs()
    return df


def unit_deltas(panel, a, b):
    """delta = rho_a - rho_b per point set, averaged over the graphs (k or r) built on it."""
    d = (panel[a] - panel[b]).rename("delta")
    return pd.concat([panel[UNIT], d], axis=1).dropna().groupby(UNIT)["delta"].mean().to_numpy()


def _wilcoxon(x, alternative):
    return float(wilcoxon(x, alternative=alternative).pvalue) if np.any(x != 0) else 1.0


def holm(p):
    """Holm-adjusted p-values."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    out = np.empty_like(p)
    out[order] = np.minimum(adj, 1.0)
    return out


def compare(delta, rng):
    """Median, bootstrap CI, win rate and the three one-sided tests of one set of paired differences."""
    boot = np.median(rng.choice(delta, size=(BOOTSTRAP, len(delta))), axis=1)
    return {"n": len(delta), "median_delta": float(np.median(delta)),
            "ci_low": float(np.percentile(boot, 2.5)), "ci_high": float(np.percentile(boot, 97.5)),
            "win_rate": float(np.mean(delta > 0) + 0.5 * np.mean(delta == 0)),
            "p_better": _wilcoxon(delta, "greater"),
            "p_noninferior": _wilcoxon(delta + MARGIN, "greater"),
            "p_worse": _wilcoxon(delta + MARGIN, "less")}


def verdict(r):
    if r["p_better_holm"] <= ALPHA and r["median_delta"] > MARGIN:
        return "better"
    if r["p_noninferior_holm"] <= ALPHA:
        return "non-inferior"
    if r["p_worse_holm"] <= ALPHA:
        return "worse"
    return "inconclusive"


def main():
    """Both tables -> RESULTS."""
    df = load()
    rng = np.random.default_rng(SEED)
    rows, winners = [], []
    for g in GRAPHS:
        for m in MEASURES:
            panel = df[(df["graph_type"] == g) & (df["measure"] == m)]
            for a in BOSPERRUS:
                for b in COMPETITORS:
                    rows.append({"graph_type": g, "measure": m, "bosperrus": a, "competitor": b,
                                 **compare(unit_deltas(panel, a, b), rng)})
            med = panel[METHODS].median().sort_values(ascending=False)
            delta = unit_deltas(panel, med.index[0], med.index[1])
            winners.append({"graph_type": g, "measure": m, "winner": med.index[0], "runner_up": med.index[1],
                            "n": len(delta), "median_delta": float(np.median(delta)),
                            "win_rate": float(np.mean(delta > 0) + 0.5 * np.mean(delta == 0)),
                            "p": _wilcoxon(delta, "two-sided")})
    out = pd.DataFrame(rows)
    for col in ["p_better", "p_noninferior", "p_worse"]:
        out[f"{col}_holm"] = holm(out[col])
    out["verdict"] = out.apply(verdict, axis=1)
    out.to_csv(RESULTS / "sphere_paired_stats.csv", index=False)

    win = pd.DataFrame(winners)
    win["p_holm"] = holm(win["p"])
    win["clear_win"] = (win["p_holm"] <= ALPHA) & (win["median_delta"] > MARGIN)
    win.to_csv(RESULTS / "sphere_winner_vs_runner_up.csv", index=False)
    print(out.groupby(["bosperrus", "competitor"])["verdict"].value_counts().unstack(fill_value=0))
    print(win.to_string(index=False))


if __name__ == "__main__":
    main()
