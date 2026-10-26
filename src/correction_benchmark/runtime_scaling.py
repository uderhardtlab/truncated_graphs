"""Runtime and peak-memory scaling of BOSPERRUS vs. SERN with the number of nodes (supplementary figure).

One run per method x graph type x size, on an exclusive node (jobs/correction_benchmark_runtime.sbatch).
Points: uniform in a square of side sqrt(n) (unit density), so edge lengths do not change with n.
Graph construction is shared by both methods and timed separately; it is not counted for either method.

  bosperrus  centralities once on the observed graph + one Flow fit per measure (all five measures).
             graph-tool may use every core (OMP_NUM_THREADS = cores).
  sern       surrogate_ensemble_gt (src/correction_benchmark/sern.py): link probability over all node pairs + 400
             surrogates, all five measures per surrogate; joblib n_jobs = cores, one thread per worker.

Peak memory, two measures (the plot uses the larger):
  peak_node_delta_bytes  peak node-wide used memory minus the value at start, sampled every 0.2 s; covers the
                         SERN worker processes and is valid because the node is exclusive
  peak_main_bytes        peak RSS of the main process (ru_maxrss) minus its RSS at start; exact for BOSPERRUS,
                         which runs in one process, and for runs shorter than the sampling interval A record is written before the computation starts and completed afterwards, so runs
killed by the OOM killer or the time limit are still visible (status "started").

Run from truncated_graphs/src:
  python correction_benchmark/runtime_scaling.py --method bosperrus --graph knn --n 10000
Output: ../results/correction_benchmark/runtime_scaling/{method}_{graph}_{n}.json
"""
import argparse
import json
import os
import platform
import sys
import threading
import time

P = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
P.add_argument("--method", choices=["bosperrus", "sern"], required=True)
P.add_argument("--graph", choices=["knn", "delaunay"], required=True)
P.add_argument("--n", type=int, required=True)
P.add_argument("--k", type=int, default=10)
P.add_argument("--n-surrogates", type=int, default=400)
P.add_argument("--seed", type=int, default=0)
P.add_argument("--out-dir", default="../results/correction_benchmark/runtime_scaling")
ARGS = P.parse_args()

CORES = int(os.environ.get("SLURM_CPUS_ON_NODE", os.cpu_count()))
# must be set before graph-tool is imported
os.environ["OMP_NUM_THREADS"] = str(CORES if ARGS.method == "bosperrus" else 1)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import psutil  # noqa: E402
import resource  # noqa: E402
from bosperrus import Flow, construct_graph, distance_to_rectangular_border  # noqa: E402
from bosperrus.centrality_measures import compute_centrality_measures  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sern import surrogate_ensemble_gt  # noqa: E402

MEASURES = ["degree", "pagerank", "betweenness", "closeness", "clustering"]


class PeakMemory:
    """Peak node-wide used memory above the value at start, sampled in a background thread."""

    def __init__(self, interval=0.2):
        self.interval, self.base, self.peak = interval, psutil.virtual_memory().used, 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, psutil.virtual_memory().used - self.base)
            time.sleep(self.interval)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()
        self.peak = max(self.peak, psutil.virtual_memory().used - self.base)


def write(record, path):
    with open(path + ".tmp", "w") as f:
        json.dump(record, f, indent=1)
    os.replace(path + ".tmp", path)


def main():
    os.makedirs(ARGS.out_dir, exist_ok=True)
    path = os.path.join(ARGS.out_dir, f"{ARGS.method}_{ARGS.graph}_{ARGS.n}.json")
    record = {**vars(ARGS), "cores": CORES, "host": platform.node(), "cpu": platform.processor(),
              "slurm_job": os.environ.get("SLURM_JOB_ID"), "status": "started"}
    write(record, path)

    rng = np.random.default_rng(ARGS.seed)
    coords = rng.uniform(0, np.sqrt(ARGS.n), size=(ARGS.n, 2))
    t0 = time.perf_counter()
    edges = construct_graph(coords, ARGS.graph, k=ARGS.k) if ARGS.graph == "knn" else construct_graph(coords, "delaunay")
    record.update(t_graph=time.perf_counter() - t0, n_edges=len(edges))
    write(record, path)

    rss_start = psutil.Process().memory_info().rss
    with PeakMemory() as mem:
        t_start = time.perf_counter()
        if ARGS.method == "bosperrus":
            dist = distance_to_rectangular_border(coords)
            t_centrality, t_fit, best = {}, {}, {}
            for m in MEASURES:
                t = time.perf_counter()
                scores = pd.DataFrame(compute_centrality_measures(edges, ARGS.n, measures=[m]))
                t_centrality[m] = time.perf_counter() - t
                t = time.perf_counter()
                flow = Flow.from_distances_and_scores(distances=dist, scores=scores)
                flow.flow()
                t_fit[m] = time.perf_counter() - t
                best[m] = type(flow.best_fits[m]).__name__
            record.update(t_centrality=t_centrality, t_fit=t_fit, best_fit=best)
        else:
            np.random.seed(ARGS.seed)  # sern.py draws its surrogate seeds from the global RNG
            surrogate_ensemble_gt(coords, edges, n_bins=int(np.sqrt(len(edges))), measures=MEASURES,
                                  n_surrogates=ARGS.n_surrogates, n_jobs=CORES)
        record["t_total"] = time.perf_counter() - t_start
    peak_main = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 - rss_start  # ru_maxrss is in KiB
    record.update(peak_node_delta_bytes=int(mem.peak), peak_main_bytes=int(peak_main),
                  peak_mem_bytes=int(max(mem.peak, peak_main)), status="done")
    write(record, path)
    print(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
