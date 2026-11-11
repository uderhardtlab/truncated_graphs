"""MICrONS buffer on log-transformed degrees (completeness check of microns_buffer.py).

Same piecewise-linear vs. constant BOSPERRUS fit against distance to the alpha-shape boundary as microns_buffer.py,
on indegree and degree raw and as log10(x + 1) (490 / 615 neurons have degree / indegree 0). For each knot: the share
of neurons inside the buffer and the Jaccard index with the Reimann heuristic, per neuron (own outer synapse fraction
> REIMANN_THRESH) and per x-z bin (the bin's max outer synapse fraction > REIMANN_THRESH), on the neurons with an
outer synapse fraction, as in micron_comparison.compute_reimann_comparison.

Reads results/physically_realized_graphs/microns/ (microns_buffer.py); writes log_fits.json there.
Run (light, seconds):
    cd truncated_graphs/src && pixi run python physically_realized_graphs/microns_log_fit.py
"""
import json
from pathlib import Path

import bosperrus
import numpy as np
import pandas as pd

RES = Path(__file__).resolve().parents[2] / "results" / "physically_realized_graphs" / "microns"
MEASURES = ["indegree", "degree"]


def main():
    """Raw and log fits, buffer sizes and Jaccard indices -> RES / log_fits.json."""
    n = pd.read_csv(RES / "neurons.csv", index_col=0)
    s = json.loads((RES / "summary.json").read_text())
    scores = pd.DataFrame({**{m: n[m] for m in MEASURES}, **{f"log10_{m}": np.log10(n[m] + 1) for m in MEASURES}})
    flow = bosperrus.Flow.from_distances_and_scores(
        distances=pd.Series(n["alpha_distance_nm"].to_numpy(), name="alpha_distance"),
        scores=scores.reset_index(drop=True))
    flow.flow(fits=[bosperrus.PiecewiseLinearFit, bosperrus.ConstantFit], baseline_fit_class=bosperrus.ConstantFit)

    # Reimann classifications on the neurons with an outer synapse fraction
    rated = n.dropna(subset=["outer_syn_fraction"])
    frac_max = pd.read_csv(RES / "outer_fraction_max.csv", index_col=0)
    frac_max.columns = frac_max.columns.astype(int)
    x_bin = np.digitize(rated["x_nm"] / 1e3, s["x_edges_um"])
    z_bin = np.digitize(rated["z_nm"] / 1e3, s["z_edges_um"])
    per_bin = np.array([frac_max.at[z, x] > s["reimann_thresh"] if z in frac_max.index and x in frac_max.columns
                        else False for z, x in zip(z_bin, x_bin, strict=True)])
    per_neuron = rated["outer_syn_fraction"].to_numpy() > s["reimann_thresh"]

    def jaccard(a, b):
        return float((a & b).sum() / (a | b).sum())

    out = {"reimann_excluded_per_bin": float(per_bin.mean()), "reimann_excluded_per_neuron": float(per_neuron.mean())}
    for m in scores:
        fit = flow.best_fits[m]
        rec = {"best_fit": type(fit).__name__, "params": {k: float(v) for k, v in fit.params.items()}}
        if "piecewise_linear_b" in fit.params:
            knot = float(fit.params["piecewise_linear_b"])
            inside = rated["alpha_distance_nm"].to_numpy() < knot
            rec.update(knot_um=knot / 1e3, excluded=float(inside.mean()),
                       jaccard_per_bin=jaccard(inside, per_bin), jaccard_per_neuron=jaccard(inside, per_neuron))
        out[m] = rec
        print(m, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items() if k != "params"})
    (RES / "log_fits.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
