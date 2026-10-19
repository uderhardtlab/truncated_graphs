"""Figure 5 summary table: median |correlation| with the uncropped graph per graph type x measure x method,
and paired comparisons over runs/configs of multiplicative and log vs. additive (fraction better, Wilcoxon p).

Run from truncated_graphs/src:  python figure5/summarize.py
Output: ../results/figure5_summary.csv (printed as well)
"""
import glob

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

COLS = {"original vs. on crop": "raw", "original vs. BOSPERRUS corrected on crop": "additive",
        "original vs. BOSPERRUS multiplicative on crop": "multiplicative", "original vs. BOSPERRUS log on crop": "log",
        "original vs. SERN corrected on crop": "SERN"}


def main():
    df = pd.concat(pd.read_csv(f).rename(columns={"Unnamed: 0": "measure"}) for f in glob.glob("../results/figure5/*.csv"))
    df = df.rename(columns=COLS)
    # r-NN r = 0.05 excluded, as in plot_violins.py
    df = df[~((df["graph_type"] == "rnn") & np.isclose(df["radius"].fillna(0), 0.05))]
    for c in COLS.values():
        df[c] = df[c].abs()
    rows = []
    for (cm, sub, g, m), x in df.groupby(["corr_method", "node_subset", "graph_type", "measure"]):
        row = {"corr": cm, "subset": sub, "graph": g, "measure": m, "n": len(x),
               **{f"median_{c}": x[c].median() for c in COLS.values()},
               "n_excluded_nodes": int(x["n_excluded"].sum())}
        for v in ["multiplicative", "log"]:
            diff = (x[v] - x["additive"]).dropna()
            row[f"frac_{v}_gt_add"] = float((diff > 0).mean())
            row[f"wilcoxon_p_{v}_vs_add"] = wilcoxon(diff).pvalue if (diff != 0).any() else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv("../results/figure5_summary.csv", index=False)
    print("seeds:", df["seed"].nunique())
    with pd.option_context("display.width", 250, "display.max_rows", 200):
        print(out.round(3).to_string())


if __name__ == "__main__":
    main()
