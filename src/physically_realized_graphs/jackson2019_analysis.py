"""
Border effects in the Arabidopsis shoot apical meristem cell-contact networks (Jackson et al. 2019, Cell Systems
8:53-65; data https://osf.io/mxgrh/, WT).

Per meristem (4 replicates x 3 time points): cell-contact network (walls >= 0.01 um^2, as in the paper),
betweenness, degree and random-walk (current-flow) centrality on the full segmented network, and the lateral
(x-y) distance of each centroid to the convex hull of all centroids (the field-of-view edge). Then:
  - per-layer BOSPERRUS correction of degree, log10 betweenness and RWC against that distance (L4+ pooled,
    layers with < 30 cells skipped);
  - data-driven buffer: piecewise-linear knot of degree and of log10 betweenness vs. distance, per meristem on
    L1-L3 (cells beyond the knot are kept);
  - betweenness-division association (logit, central cells): raw vs. corrected, per buffer;
  - crop experiment (field of view reduced by 5/10/15 um), depth test (L4 removed);
  - Fig. 3A fold changes (non-dividing / dividing / divided cells), raw vs. corrected.

Writes results/physically_realized_graphs/jackson2019/:
  cells.csv                    one row per cell and meristem (rep, time, label): geometry, distance d, layer,
                               degree / log_bc / rwc raw and *_corr, central (evaluated in the paper), published
                               columns, beyond_knot_degree / beyond_knot_log_bc (kept by the buffer)
  layer_fits.csv, knots.csv, buffer_association.csv, division_association.csv, crops.csv,
  crop_association.csv, depth.csv, depth_association.csv, fold_changes.csv, fold_change_tests.csv, summary.json
Cell centroids come from jackson_centroids.py (results/.../centroids.csv). Plotted by
figure_plotting/physically_realized_graphs/meristem.ipynb.

Run (about 5 min, 64 GB):
    cd truncated_graphs/
    sbatch --clusters=tinyfat jobs/physically_realized_graphs_jackson2019.sbatch
"""
import json
import zipfile
from io import BytesIO
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from bosperrus import Flow
from bosperrus.fit import PiecewiseLinearFit
from scipy.spatial import ConvexHull
from scipy.stats import mannwhitneyu, spearmanr, ttest_ind
from shapely.geometry import Point, Polygon

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data" / "jackson2019"
OUT = ROOT / "results" / "physically_realized_graphs" / "jackson2019"
CENTROIDS = OUT / "centroids.csv"
GENOTYPE, ZIP = "WT", DATA_DIR / "WT_Arabidopsis.zip"
MIN_LAYER_CELLS = 30
PUBLISHED_COLS = ["Division", "Volume", "Layer", "Log_Betweenness_Centrality", "Random_Walk_Centrality", "Area",
                  "Anisotropy"]
STRUCTURAL = ["degree", "log_bc", "rwc"]  # network properties; cell geometry is not a graph-construction artefact
FC_PROPS = ["area", "volume", "anisotropy", "degree", "bc", "rwc"]


# ── data ──────────────────────────────────────────────────────────────────────
def read_csv_from_zip(z, name):
    """A CSV member of an open zip file as a DataFrame."""
    return pd.read_csv(BytesIO(z.read(name)), skipinitialspace=True)


def load_samples(zip_path, genotype):
    """Per (rep, time): contact network, published per-cell values (central cells only) and centroids."""
    centroids = pd.read_csv(CENTROIDS)
    samples = {}
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        for net in sorted(n for n in names if n.endswith("_network.csv")):
            _, rep, time, _ = net.split("/")
            edges = read_csv_from_zip(z, net)
            edges = edges[edges.iloc[:, 2] >= 0.01]  # the paper omits walls < 0.01 um^2
            G = nx.Graph()
            G.add_edges_from(zip(edges.iloc[:, 0], edges.iloc[:, 1], strict=True))
            topo_name = [n for n in names if n.startswith(f"{net.rsplit('/', 1)[0]}/") and n.endswith("_topology.csv")]
            topo = read_csv_from_zip(z, topo_name[0]).set_index("Label") if topo_name else None
            this = (centroids.genotype == genotype) & (centroids.rep == rep) & (centroids.time == time)
            pos = centroids[this].set_index("label")
            samples[(rep, time)] = dict(G=G, topo=topo, pos=pos.loc[list(G.nodes)])
    return samples


def load_parents(zip_path):
    """T0 -> T1 lineage table per replicate (column order not yet oriented)."""
    out = {}
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if name.endswith("_T0-T1_parents.csv"):
                lineage = read_csv_from_zip(z, name)
                lineage.columns = ["a", "b"] + list(lineage.columns[2:])
                out[name.split("/")[1]] = lineage
    return out


def lateral_distance(xy):
    """Distance of each x-y centroid to the convex hull of all centroids (the field-of-view edge)."""
    hull = Polygon(xy[ConvexHull(xy).vertices])
    return np.array([hull.exterior.distance(Point(p)) for p in xy])


def cell_table(G, pos, topo=None):
    """Per cell: geometry, distance d to the field-of-view edge, log10 betweenness, degree, RWC and the published
    values where they exist (`central` marks the cells the paper evaluated)."""
    nodes = list(G.nodes)
    df = pd.DataFrame(index=pd.Index(nodes, name="label"))
    cols = ["x", "y", "z", "area", "volume", "anisotropy", "layer_from_start", "layer_from_end"]
    df[cols] = pos.loc[nodes, cols].values
    df["d"] = lateral_distance(df[["x", "y"]].values)
    df["log_bc"] = np.log10(pd.Series(nx.betweenness_centrality(G)).loc[nodes].values + 1e-9)
    df["degree"] = pd.Series(dict(G.degree())).loc[nodes].values
    # the paper's "random walk centrality" is Newman's random-walk (current-flow) betweenness
    giant = G.subgraph(max(nx.connected_components(G), key=len))  # current-flow betweenness needs a connected graph
    df["rwc"] = pd.Series(nx.current_flow_betweenness_centrality(giant)).reindex(nodes).values
    df["central"] = False
    if topo is not None:
        common = df.index.intersection(topo.index)
        df.loc[common, "central"] = True
        for col in PUBLISHED_COLS:
            df.loc[common, col] = topo.loc[common, col]
    return df


def layer_group(layer):
    """Layer with L4 and deeper pooled into 4."""
    return np.minimum(np.asarray(layer, dtype=float), 4)  # L4 and deeper column counts pooled


def correct_by_layer(df, cols, d_col="d"):
    """BOSPERRUS correction of `cols` against `d_col`, one fit per layer (L4+ pooled)."""
    out, fits = df.copy(), {}
    out[cols] = out[cols].astype(float)  # degree is int; corrected values are not
    for layer, part in df.groupby(layer_group(df["layer"])):
        if len(part) < MIN_LAYER_CELLS:
            continue
        for col in cols:
            flow = Flow.from_distances_and_scores(distances=part[d_col].reset_index(drop=True),
                                                  scores=part[[col]].reset_index(drop=True))
            flow.flow()
            out.loc[part.index, col] = flow.observations[f"BOSPERRUS corrected {col}"].values
            fits[(int(layer), col)] = flow.best_fits[col]
    return out, fits


def pl_knot(df, col):
    """Knot (um) of a piecewise-linear fit of `col` against d."""
    fit = PiecewiseLinearFit(df[col].reset_index(drop=True), df["d"].reset_index(drop=True))
    fit.fit()
    return fit.params["piecewise_linear_b"]


def logit(formula, data, score):
    """Logistic regression; coefficient, 95 % CI and p of `score`."""
    r = smf.logit(formula, data).fit(disp=0)
    lo, hi = r.conf_int().loc[score]
    return dict(coef=r.params[score], ci_low=lo, ci_high=hi, p=r.pvalues[score])


def pooled(tables, mask=lambda t: slice(None)):
    """All meristems' cell tables (rows selected by `mask`) stacked, with rep and time columns."""
    return pd.concat([t.loc[mask(t)].assign(rep=k[0], time=k[1]) for k, t in tables.items()])


# ── analysis ──────────────────────────────────────────────────────────────────
def main():
    """Run all analyses and write the tables to OUT."""
    OUT.mkdir(parents=True, exist_ok=True)
    samples = load_samples(ZIP, GENOTYPE)
    tables = {key: cell_table(s["G"], s["pos"], s["topo"]) for key, s in samples.items()}
    summary = {}

    # layers: which end of the stack is the surface (the count that reproduces the published layers); the
    # column count is approximate on the dome's flanks, so published layers are used for the central cells
    check = pd.concat(tables.values()).dropna(subset=["Log_Betweenness_Centrality"])
    match = {c: float((check[c] == check.Layer).mean()) for c in ["layer_from_start", "layer_from_end"]}
    layer_col = max(match, key=match.get)
    for t in tables.values():
        t["layer"] = t["Layer"].fillna(t[layer_col]) if "Layer" in t else t[layer_col]
    summary["layer_agreement"] = match
    summary["spearman_with_published"] = {
        name: float(spearmanr(check[theirs], check[mine])[0])
        for name, theirs, mine in [("log_bc", "Log_Betweenness_Centrality", "log_bc"),
                                   ("rwc", "Random_Walk_Centrality", "rwc"), ("area", "Area", "area"),
                                   ("volume", "Volume", "volume"), ("anisotropy", "Anisotropy", "anisotropy")]}

    # per-layer correction of degree, log10 BC and RWC
    fit_rows = []
    for key, t in tables.items():
        corr, layer_fits = correct_by_layer(t, STRUCTURAL)
        for col in STRUCTURAL:
            t[f"{col}_corr"] = corr[col]
        for (layer, col), best in layer_fits.items():
            fit_rows.append(dict(rep=key[0], time=key[1], layer=layer, measure=col, fit=best.name,
                                 effect=best.observed_effect_strength, half_life=best.observed_half_life))
    pd.DataFrame(fit_rows).to_csv(OUT / "layer_fits.csv", index=False)

    # data-driven buffer: piecewise-linear knot per meristem on L1-L3
    knot_rows, buffers = [], {}
    for key, t in tables.items():
        for col in ["degree", "log_bc"]:
            t[f"beyond_knot_{col}"] = False
        if not t.central.any():
            continue
        cells = t[t.layer.between(1, 3)]
        central = set(cells.index[cells.central])
        for col in ["degree", "log_bc"]:
            knot = pl_knot(cells, col)
            inside = set(cells.index[cells.d > knot])
            buffers[(key, col)] = inside
            t.loc[list(inside), f"beyond_knot_{col}"] = True
            knot_rows.append(dict(rep=key[0], time=key[1], measure=col, knot_um=knot, cells=len(cells),
                                  kept=len(inside), paper_central=len(central),
                                  paper_edge_um=cells.loc[list(central), "d"].quantile(0.05),
                                  jaccard=len(inside & central) / len(inside | central),
                                  paper_cells_inside=len(inside & central) / len(central)))
    knots = pd.DataFrame(knot_rows)
    knots.to_csv(OUT / "knots.csv", index=False)

    # betweenness-division association within each buffer (labels exist for central cells only)
    labelled = pooled(tables, lambda t: t.central & t.layer.between(1, 3))
    labelled["Division"] = labelled.Division.astype(int)
    rows = []
    for name, col in [("paper (central cells)", None), ("PL knot, degree", "degree"), ("PL knot, log10 BC", "log_bc")]:
        sub = labelled if col is None else labelled[labelled[f"beyond_knot_{col}"]]
        rows.append(dict(buffer=name, cells=len(sub), divisions=int(sub.Division.sum()),
                         **logit("Division ~ log_bc + np.log(Volume) + C(rep) + C(time)", sub, "log_bc")))
    pd.DataFrame(rows).to_csv(OUT / "buffer_association.csv", index=False)

    # where dividing cells sit, and the association raw vs. corrected (central cells)
    cent = pooled(tables, lambda t: t.central)
    cent["Division"] = cent.Division.astype(int)
    div, non = cent[cent.Division == 1].d, cent[cent.Division == 0].d
    summary["division_distance"] = dict(dividing_median_um=float(div.median()), n_dividing=len(div),
                                         non_dividing_median_um=float(non.median()), n_non_dividing=len(non),
                                         mannwhitney_p=float(mannwhitneyu(div, non).pvalue))
    rows = []
    for score in ["log_bc", "log_bc_corr"]:
        for extra in ["", " + np.log(Volume)", " + np.log(Volume) + C(Layer)"]:
            rows.append(dict(score="raw" if score == "log_bc" else "BOSPERRUS", covariates=extra.strip(" +") or "-",
                             **logit(f"Division ~ {score}{extra} + C(rep) + C(time)", cent, score)))
    pd.DataFrame(rows).to_csv(OUT / "division_association.csv", index=False)

    # crop experiment: field of view reduced by delta um, betweenness recomputed, raw vs. corrected
    crop_rows, cropped = [], {}
    for key, t in tables.items():
        if not t.central.any():
            continue
        G = samples[key]["G"]
        for delta in [5, 10, 15]:
            H = G.subgraph(t.index[t.d > delta]).copy()
            H = H.subgraph(max(nx.connected_components(H), key=len)).copy()
            sub = t.loc[list(H.nodes)].copy()
            sub["d_crop"] = lateral_distance(sub[["x", "y"]].values)
            sub["crop_raw"] = np.log10(pd.Series(nx.betweenness_centrality(H)).loc[sub.index].values + 1e-9)
            sub["crop_corr"] = correct_by_layer(sub, ["crop_raw"], d_col="d_crop")[0]["crop_raw"]
            c = sub[sub.central]
            crop_rows.append(dict(rep=key[0], time=key[1], delta=delta, central_kept=len(c),
                                  rho_raw=spearmanr(c.log_bc, c.crop_raw)[0],
                                  rho_corr=spearmanr(c.log_bc, c.crop_corr)[0]))
            cropped[(key, delta)] = sub
    pd.DataFrame(crop_rows).to_csv(OUT / "crops.csv", index=False)
    rows = []
    for delta in [5, 10, 15]:
        sub = pd.concat([s.assign(rep=k[0], time=k[1]) for (k, d), s in cropped.items() if d == delta])
        sub = sub[sub.central].copy()
        sub["Division"] = sub.Division.astype(int)
        for score, label in [("log_bc", "full network (reference)"), ("crop_raw", "cropped, raw"),
                             ("crop_corr", "cropped, BOSPERRUS")]:
            rows.append(dict(delta=delta, score=label, n=len(sub),
                             **logit(f"Division ~ {score} + np.log(Volume) + C(rep) + C(time)", sub, score)))
    pd.DataFrame(rows).to_csv(OUT / "crop_association.csv", index=False)

    # depth: remove L4, recompute on L1-L3
    depth_rows, depth_scores = [], {}
    for key, t in tables.items():
        if not t.central.any():
            continue
        G = samples[key]["G"]
        H = G.subgraph(t.index[layer_group(t.layer) <= 3])
        H = H.subgraph(max(nx.connected_components(H), key=len)).copy()
        nodes = list(H.nodes)
        sub = t.loc[nodes].copy()
        sub["degree_noL4"] = pd.Series(dict(H.degree())).loc[nodes].values
        sub["log_bc_noL4"] = np.log10(pd.Series(nx.betweenness_centrality(H)).loc[nodes].values + 1e-9)
        sub["rwc_noL4"] = pd.Series(nx.current_flow_betweenness_centrality(H)).loc[nodes].values
        depth_scores[key] = sub
        for layer in [1, 2, 3]:
            for subset in ["all cells", "central cells"]:
                m = (sub.layer == layer) & (sub.central if subset == "central cells" else True)
                if m.sum() < 10:
                    continue
                s = sub[m]
                depth_rows.append(dict(rep=key[0], time=key[1], layer=layer, subset=subset, n=int(m.sum()),
                                       degree_lost=(s.degree - s.degree_noL4).mean(),
                                       rho_bc=spearmanr(s.log_bc, s.log_bc_noL4)[0],
                                       rho_rwc=spearmanr(s.rwc, s.rwc_noL4)[0]))
    pd.DataFrame(depth_rows).to_csv(OUT / "depth.csv", index=False)
    sub = pd.concat([s.assign(rep=k[0], time=k[1]) for k, s in depth_scores.items()])
    sub = sub[sub.central].copy()
    sub["Division"] = sub.Division.astype(int)
    rows = [dict(network=label, n=len(sub),
                 **logit(f"Division ~ {score} + np.log(Volume) + C(rep) + C(time)", sub, score))
            for score, label in [("log_bc", "full network (with L4)"), ("log_bc_noL4", "without L4")]]
    pd.DataFrame(rows).to_csv(OUT / "depth_association.csv", index=False)

    # Fig. 3A: fold changes of non-dividing / dividing (T0) / divided (T1) central cells, raw vs. corrected
    parents = {}
    for rep, lineage in load_parents(ZIP).items():
        t0, t1 = set(tables[(rep, "T0")].index), set(tables[(rep, "T1")].index)
        a_in_t1 = np.mean([v in t1 for v in lineage.a]) > np.mean([v in t0 for v in lineage.a])  # orient the columns
        child, parent = (lineage.a, lineage.b) if a_in_t1 else (lineage.b, lineage.a)
        parents[rep] = pd.DataFrame({"child": child, "parent": parent})

    def scenario_cells(key, scenario):
        """Central L1-L3 cells of one meristem, network properties raw or corrected, plus BC = 10^log_bc."""
        t = tables[key][tables[key].layer.between(1, 3) & tables[key].central].copy()
        if scenario == "central cells, BOSPERRUS":
            for col in STRUCTURAL:
                t[col] = t[f"{col}_corr"]
        t["bc"] = 10 ** t["log_bc"]
        return t

    fc_rows, test_rows = [], []
    for scenario in ["central cells, raw (paper)", "central cells, BOSPERRUS"]:
        group_cells = {g: [] for g in ["non-dividing (T0)", "dividing (T0)", "divided (T1)"]}
        for rep in sorted(parents):
            t0, t1 = scenario_cells((rep, "T0"), scenario), scenario_cells((rep, "T1"), scenario)
            dividing = set(t0.index[t0.Division == 1])
            daughters = set(parents[rep].child[parents[rep].parent.isin(dividing)]) & set(t1.index)
            for group, cells, ref in [("non-dividing (T0)", t0.drop(index=list(dividing)), t0),
                                      ("dividing (T0)", t0.loc[list(dividing)], t0),
                                      ("divided (T1)", t1.loc[list(daughters)], t1)]:
                fc = cells[FC_PROPS].mean() / ref[FC_PROPS].mean()
                fc_rows.append(dict(scenario=scenario, rep=rep, group=group, n=len(cells), **fc.to_dict()))
                group_cells[group].append(cells[FC_PROPS])
        ref_cells = pd.concat(group_cells["non-dividing (T0)"])
        for group in ["dividing (T0)", "divided (T1)"]:
            grp = pd.concat(group_cells[group])
            for prop in FC_PROPS:
                test_rows.append(dict(scenario=scenario, group=group, prop=prop,
                                      welch_p=ttest_ind(grp[prop], ref_cells[prop], equal_var=False).pvalue))
    pd.DataFrame(fc_rows).to_csv(OUT / "fold_changes.csv", index=False)
    pd.DataFrame(test_rows).to_csv(OUT / "fold_change_tests.csv", index=False)

    cells = pooled(tables).reset_index().set_index(["rep", "time", "label"])
    cells.to_csv(OUT / "cells.csv")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"Wrote {OUT}")
    print(knots.groupby("measure")[["knot_um", "kept", "cells"]].median())


if __name__ == "__main__":
    main()
