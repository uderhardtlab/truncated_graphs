#!/usr/bin/env python3
"""Process one sample from sample_manifest.csv: run the border/BOSPERRUS
analysis (visium_hd.analyze_dataset or analyze_stomics_dataset, per the
manifest's `loader` column -- returns one result per spatially-connected
component with >100 bins, e.g. one per TMA core), summarize each component
(visium_hd.summarize_all, which includes the BLADE peel-sweep), and write
the list of per-component summaries to <outdir>/<name>.json (a JSON array,
one object per component -- not a single object, since a sample can yield
anywhere from one to dozens of components).

This is the single source of truth for the analysis logic -- invoked
identically whether run locally or via the SLURM array job
(jobs/figure4_border_analysis.sbatch), on this cluster or another one. Nothing
here is duplicated in the notebook, which only ever reads these JSON outputs.

Usage (select by manifest row index, e.g. $SLURM_ARRAY_TASK_ID):
    pixi run python src/figure4/process_sample.py \\
        --manifest src/figure4/sample_manifest.csv --index 0 \\
        --outdir results/figure4/per_sample

--name is a convenience alternative to --index for manual/interactive use;
`pixi run` has been observed to drop a bare `--name VALUE` pair passed through
to the wrapped command in some environments (root cause not identified,
seems specific to this flag) -- if that happens, either run inside `pixi
shell` and call this script with plain `python`, or just use --index.

Run from the `src/` directory (see truncated_graphs/CLAUDE.md's run
convention) so the default --manifest/--outdir relative paths resolve; or
pass absolute paths from anywhere.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str((Path(__file__).resolve().parent / ".." / "utils").resolve()))
import visium_hd

LOADERS = {
    "visium": visium_hd.analyze_dataset,
    "stomics": visium_hd.analyze_stomics_dataset,
}


def _to_jsonable(value):
    """numpy scalars (int64/float64/bool_) aren't JSON-serializable -- unwrap
    via .item() where present; everything else passes through unchanged."""
    return value.item() if hasattr(value, "item") else value


def process_one(row):
    t0 = time.time()
    print(f"[{row['name']}] loading + analyzing ({row['dataset_type']}, "
          f"{row['bin_size_um']}um, loader={row['loader']})...", flush=True)
    loader = LOADERS[row["loader"]]
    component_results = loader(row["h5ad_path"])
    print(f"[{row['name']}] found {len(component_results)} component(s) with >100 bins, "
          f"running summarize_all (incl. BLADE) for each...", flush=True)

    summaries = []
    for result in component_results:
        component_name = f"{row['name']}_c{result['component_id']}"
        summary = visium_hd.summarize_all(component_name, row["dataset_type"], result, row["bin_size_um"])
        summary["parent_sample"] = row["name"]
        summaries.append({k: _to_jsonable(v) for k, v in summary.items()})

    print(f"[{row['name']}] done in {time.time() - t0:.1f}s, {len(summaries)} component row(s)", flush=True)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True,
                         help="CSV with columns: name, dataset_type, loader, h5ad_path, bin_size_um")
    parser.add_argument("--index", type=int, help="row index into the manifest (e.g. $SLURM_ARRAY_TASK_ID)")
    parser.add_argument("--name", help="alternative to --index: select by the manifest's `name` column")
    parser.add_argument("--outdir", required=True, help="output directory; writes <outdir>/<name>.json")
    parser.add_argument("--force", action="store_true", help="recompute even if <outdir>/<name>.json already exists")
    args = parser.parse_args()

    if (args.index is None) == (args.name is None):
        parser.error("specify exactly one of --index or --name")

    manifest = pd.read_csv(args.manifest)
    if args.index is not None:
        if not (0 <= args.index < len(manifest)):
            parser.error(f"--index {args.index} out of range for manifest with {len(manifest)} rows")
        row = manifest.iloc[args.index].to_dict()
    else:
        matches = manifest[manifest["name"] == args.name]
        if matches.empty:
            parser.error(f"no manifest row with name={args.name!r}")
        row = matches.iloc[0].to_dict()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"{row['name']}.json"
    if outpath.exists() and not args.force:
        print(f"[{row['name']}] {outpath} already exists, skipping (use --force to recompute)")
        return

    summary = process_one(row)
    with open(outpath, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[{row['name']}] wrote {outpath}")


if __name__ == "__main__":
    main()
