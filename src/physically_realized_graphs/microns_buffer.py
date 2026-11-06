"""
MICrONS mm^3 connectome: BOSPERRUS analysis buffer vs. the Reimann lab's outer-synapse-fraction heuristic.

Fits a piecewise-linear BOSPERRUS curve of (in)degree against each neuron's x-z distance to the alpha shape of
the neurons' footprint, takes the indegree knot as the buffer, and compares the neurons inside it with the
neurons the Reimann heuristic flags (per neuron and per x-z bin; Jaccard index). Library code:
micron_comparison.py.

Writes results/physically_realized_graphs/microns/:
  neurons.csv                 one row per neuron: x_nm, z_nm, indegree, degree, alpha_distance_nm,
                              outer_syn_fraction, reimann_affected, bosperrus_affected
  outer_fraction_{max,mean}.csv  per x-z bin max / mean outer synapse fraction (rows z bin, columns x bin)
  alpha_shape.wkt             the alpha shape (nm)
  summary.json                parameters, piecewise-linear fit parameters, bin edges, counts, Jaccard indices
Plotted by figure_plotting/physically_realized_graphs/microns.ipynb.

Run (needs ~40 GB memory; data/reimann_data/microns_mm3_connectome.h5):
    cd truncated_graphs/
    sbatch --clusters=tinyfat jobs/physically_realized_graphs_microns.sbatch
"""
import json
from pathlib import Path

import micron_comparison as mc

ROOT = Path(__file__).resolve().parent.parent.parent
FN_MAT = ROOT / "data" / "reimann_data" / "microns_mm3_connectome.h5"
OUT = ROOT / "results" / "physically_realized_graphs" / "microns"
NBINS = 51            # x-z grid resolution for the Reimann outer-synapse-fraction heuristic
ALPHA_SHAPE = 0.0001  # concaveness of the alpha shape fit to the neuron x-z footprint
REIMANN_THRESH = 0.05  # outer-synapse-fraction cutoff for Reimann's "affected" classification


def main():
    """Run the MICrONS comparison and write its tables to OUT."""
    data = mc.prepare_microns_data(FN_MAT, nbins=NBINS, alpha=ALPHA_SHAPE, reimann_thresh=REIMANN_THRESH)
    comp = data["comparison"]
    mc.print_comparison_summary(comp)
    OUT.mkdir(parents=True, exist_ok=True)

    M = data["M"]
    neurons = M.vertices[["x_nm", "z_nm"]].copy()
    neurons["indegree"] = data["scores"]["indegree"].values
    neurons["degree"] = data["scores"]["degree"].values
    neurons["alpha_distance_nm"] = data["alpha_distances"]
    neurons = neurons.join(comp["df"][["outer_syn_fraction", "reimann_affected", "bosperrus_affected"]])
    neurons.to_csv(OUT / "neurons.csv")
    comp["I_frac_max"].to_csv(OUT / "outer_fraction_max.csv")
    comp["I_frac_mean"].to_csv(OUT / "outer_fraction_mean.csv")
    (OUT / "alpha_shape.wkt").write_text(data["alpha_shape"].wkt)

    fits = {m: {k: float(v) for k, v in data["flow"].best_fits[m].params.items()} for m in ["indegree", "degree"]}
    summary = dict(nbins=NBINS, alpha_shape=ALPHA_SHAPE, reimann_thresh=REIMANN_THRESH,
                   elbow_nm=float(comp["elbow"]), piecewise_linear_fits=fits,
                   x_edges_um=data["x_edges_um"].tolist(), z_edges_um=data["z_edges_um"].tolist(),
                   jaccard_neuron=float(comp["jaccard_neuron"]), jaccard_bin=float(comp["jaccard_bin"]),
                   **{k: int(comp[k]) for k in ["n_bosperrus", "n_reimann_neuron", "n_reimann_bin", "n_total"]})
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
