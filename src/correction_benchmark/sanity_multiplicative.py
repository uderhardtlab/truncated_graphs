"""Sanity check of the BOSPERRUS correction variants in sphere.py before the full figure 5 rerun.

One seeded sphere per coordinate config (uniform, kappa=1), cap radius 2, graph types delaunay /
delaunay_pruned / knn k=10 / rnn r=0.1, all five measures. Log variant with two scales s: median of S
("log_med") and 10th percentile of the positive S ("log_p10"). Location-scale variant ("locscale"): mean and spread corrected separately, spread from a
BOSPERRUS fit of |S - f(d)| vs distance.
Per (config, graph): crop Delaunay via stereographic projection == hull minus facets whose empty
circumcircle contains the cap antipode (edge sets must be identical).
Per (config, graph, measure):
  - best fit, plateau P, max deviation of the recovered P_i = S_add - S + f(d) from P (must be ~0),
  - min f(d), number of NaN (f <= 0) and of large factors P / f(d) > 3,
  - identity: if the best fit is Constant, multiplicative == additive == raw,
  - fraction of zeros, scale s (and whether the positive-median fallback was used), negative corrected values,
  - Pearson/Spearman of original (global graph) vs every variant on the same nodes, all and affected
    (nearest quartile to the border) nodes.
Profiles per distance bin: median per method, and SD(method) / SD(original) -- whether a variant restores the
spread that truncation compresses near the border (the reason for the multiplicative model).
Figures: profiles.pdf (medians), sd_ratio.pdf (SD ratios).

Run from truncated_graphs/src:  python correction_benchmark/sanity_multiplicative.py   (tinyfat: jobs/correction_benchmark_sanity.sbatch)
Outputs: ../results/correction_benchmark/sanity/{edges.csv, checks.csv, correlations.csv, profiles.csv, *.pdf}
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sphere as S
S_mod = S
from bosperrus.centrality_measures import compute_centrality_measures

SEED, N, N_SERN, CAP = 20261015, 5000, 20, 2
OUT = "../results/correction_benchmark/sanity"
GRAPHS = [("delaunay", {}), ("delaunay_pruned", {}), ("knn", {"k": 10}), ("rnn", {"r": 0.1})]
LOG_SCALES = {"log_med": S.median_scale, "log_p10": lambda x: float(np.percentile(x[x > 0], 10))}
METHODS = ["raw", "additive", "multiplicative", "log_med", "log_p10", "locscale", "SERN"]


def locscale_correction(d, S, f_d, P):
    """Location-scale variant: P + (S - f(d)) * g_P / g(d), with g the BOSPERRUS fit of |S - f(d)| vs distance.

    Corrects mean and spread separately; NaN where g(d) <= 0. Returns (corrected, best fit name of g).
    """
    absres = (S - f_d).abs()
    ok = np.isfinite(absres.to_numpy())  # closeness can be NaN for isolated nodes
    bf = S_mod.Flow.from_distances_and_scores(distances=d[ok].reset_index(drop=True),
                                              scores=pd.DataFrame({"absres": absres[ok].to_numpy()}))
    bf.flow(score_names=["absres"])
    g = bf.best_fits["absres"]
    g_d = pd.Series(g.predict(d.to_numpy()), index=S.index)
    g_P = float(np.median(bf.observations["BOSPERRUS corrected absres"].to_numpy() - absres[ok].to_numpy()
                          + g_d[ok].to_numpy()))
    return P + (S - f_d) * (g_P / g_d.where(g_d > 0)), type(g).__name__


def circumcircle_filter_edges(coords, pole):
    """Hull facets of the crop minus those whose empty circumcircle (cap n.x > c) contains `pole`."""
    hull = S.ConvexHull(coords)
    n, c = hull.equations[:, :3], -hull.equations[:, 3]
    keep = n @ pole <= c
    return {frozenset((a, b)) for t in hull.simplices[keep] for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0]))}


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    np.random.seed(SEED)
    S.NUMBER_OF_SERNS = N_SERN
    configs = [("uniform", S.sample_uniform_on_unit_sphere(N, rng=rng)),
               ("kappa=1", S.sample_von_mises_fisher(N, [1], rng=rng))]
    edge_checks, checks, corrs, profiles = [], [], [], []
    for cname, coords in configs:
        for gname, kw in GRAPHS:
            g_edges = S.get_edge_list(coords, gname, **kw)
            glob = pd.DataFrame(compute_centrality_measures(g_edges, len(coords), S.MEASURES))
            crop, dist, center = S.crop_cap(coords, CAP, rng=rng, return_center=True)
            cc = coords[crop]
            l_edges = S.get_edge_list(cc, gname, pole=-center, **kw)
            if gname == "delaunay":
                hull, filt = S._spherical_delaunay_edges(cc), circumcircle_filter_edges(cc, -center)
                edge_checks.append({"coords": cname, "n_stereo": len(l_edges), "n_circumcircle": len(filt),
                                    "identical": l_edges == filt, "n_hull": len(hull),
                                    "hull_edges_dropped": len(hull - l_edges), "edges_added": len(l_edges - hull)})
            obs = S.get_bosperrus_corrections(cc, l_edges, glob.columns, dist, log_scales=LOG_SCALES)
            sern = S.get_sern_median(cc, l_edges, glob.columns)
            d = obs["distance_to_cap"]
            affected = d <= d.quantile(S.AFFECTED_QUANTILE)
            bf = S.Flow.from_distances_and_scores(distances=dist.reset_index(drop=True),
                                                  scores=obs[[m for m in glob.columns if m in obs.columns]])
            bf.flow()
            for m in [c for c in glob.columns if c in obs.columns]:
                raw = obs[m].astype(float)
                v = {"raw": raw, "additive": obs[f"BOSPERRUS corrected {m}"],
                     "multiplicative": obs[f"BOSPERRUS multiplicative {m}"],
                     "log_med": obs[f"BOSPERRUS log_med {m}"], "log_p10": obs[f"BOSPERRUS log_p10 {m}"],
                     "SERN": raw - sern[m]}
                fit = bf.best_fits[m]
                f_d = pd.Series(fit.predict(d.to_numpy()), index=raw.index)
                P_i = v["additive"] - raw + f_d
                P = float(P_i.median())
                orig = glob.loc[crop, m].reset_index(drop=True)
                v["locscale"], g_name = locscale_correction(d, raw, f_d, P)
                v = {k: v[k] for k in METHODS}  # column order
                const = type(fit).__name__ == "ConstantFit"
                checks.append({
                    "coords": cname, "graph": gname, "measure": m, "n_crop": len(raw),
                    "n_edges": len(l_edges), "best_fit": type(fit).__name__, "plateau": P,
                    "plateau_max_dev": float((P_i - P).abs().max()),
                    "f_min": float(f_d.min()), "n_mult_nan": int(v["multiplicative"].isna().sum()),
                    "n_factor_gt3": int((P / f_d.where(f_d > 0) > 3).sum()),
                    "constant_identity_ok": bool(np.allclose(v["multiplicative"], raw)) if const else None,
                    "frac_zero": float((raw == 0).mean()),
                    "median_is_zero": bool(np.nanmedian(raw) == 0),
                    "s_med": S.median_scale(raw), "s_p10": LOG_SCALES["log_p10"](raw),
                    **{f"n_neg_{k}": int((v[k] < 0).sum()) for k in ["additive", "multiplicative", "log_med", "log_p10", "locscale"]},
                    "locscale_g_fit": g_name, "n_nan_locscale": int(v["locscale"].isna().sum()),
                    "n_nan_orig_or_raw": int((~np.isfinite(orig) | ~np.isfinite(raw)).sum()),
                })
                ok_all = np.logical_and.reduce([np.isfinite(x.to_numpy(float)) for x in [orig, *v.values()]])
                for sub, mask in [("all", pd.Series(True, index=raw.index)), ("affected", affected)]:
                    ok = ok_all & mask.to_numpy()
                    for cm, fn in S.CORRELATION_METHODS.items():
                        row = {"coords": cname, "graph": gname, "measure": m, "subset": sub, "corr": cm,
                               "n_excluded": int((mask.to_numpy() & ~ok_all).sum())}
                        for k, x in v.items():
                            row[k] = S._corr(fn, orig, x, ok)
                        corrs.append(row)
                bins = pd.qcut(d, 15, duplicates="drop")
                frame = pd.DataFrame({"original": orig, **v, "d": d})
                g = frame.groupby(bins, observed=True)
                prof = g.median()
                sd = g.std()
                for k in METHODS:
                    prof[f"sd_ratio_{k}"] = sd[k] / sd["original"]
                prof["coords"], prof["graph"], prof["measure"] = cname, gname, m
                profiles.append(prof.reset_index(drop=True))
    edge_checks, checks, corrs, profiles = map(pd.DataFrame, [edge_checks, checks, corrs, pd.concat(profiles)])
    for name, df in [("edges", edge_checks), ("checks", checks), ("correlations", corrs), ("profiles", profiles)]:
        df.to_csv(f"{OUT}/{name}.csv", index=False)
    plot(profiles, METHODS[:-1] + ["original"], "", "profiles.pdf")
    plot(profiles, METHODS[:-1], "sd_ratio_", "sd_ratio.pdf", hline=1)
    with pd.option_context("display.width", 300, "display.max_rows", 300, "display.max_columns", 40):
        print(edge_checks.to_string())
        print(checks.round(4).to_string())
        print(corrs[corrs["corr"] == "spearman"].round(3).to_string())


def plot(profiles, methods, prefix, fname, hline=None):
    colors = {"original": "#1a1a1a", "raw": "#9a9a9a", "additive": "#2a78b5", "multiplicative": "#d1752a",
              "log_med": "#3b9c4a", "log_p10": "#8e5bb5", "locscale": "#c23b7a"}
    measures = sorted(profiles["measure"].unique())
    panels = profiles[["coords", "graph"]].drop_duplicates().values.tolist()
    fig, axs = plt.subplots(len(panels), len(measures), figsize=(2.6 * len(measures), 2.1 * len(panels)),
                            squeeze=False, constrained_layout=True)
    for i, (c, g) in enumerate(panels):
        for j, m in enumerate(measures):
            ax = axs[i, j]
            p = profiles[(profiles.coords == c) & (profiles.graph == g) & (profiles.measure == m)]
            for k in methods:
                ax.plot(p["d"], p[prefix + k], color=colors[k], lw=1.3, label=k)
            if hline is not None:
                ax.axhline(hline, color="k", lw=0.5, ls=":")
            ax.set_title(f"{m}\n{c}, {g}", fontsize=7)
            ax.tick_params(labelsize=6)
            if i == len(panels) - 1:
                ax.set_xlabel("distance to cap border (rad)", fontsize=7)
    axs[0, 0].legend(fontsize=6, frameon=False)
    fig.savefig(f"{OUT}/{fname}")


if __name__ == "__main__":
    main()
