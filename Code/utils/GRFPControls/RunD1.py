"""
Diagnostic 1: rerun the September paper-protocol comparison with the full-pool
error on not-yet-labelled points logged alongside the two existing metrics.

Same experiment as the September `paper_protocol` (test RMSE) and
`paper_metric_fullpool` (full-pool) sweeps: RunExpA's nine selectors, the
manuscript's 10/5/85 split, standardized features, seeds 0-99, the 14 datasets
of the full-pool sweep. Every run records three RMSE curves from the same
split: `full_pool` (the paper's hybrid metric), `full_pool_excl` (candidate
points only) and `test` (held-out). AnalyzeD1.py checks the first and last
against the September curves bit-for-bit.

One pickle per (dataset, selector, seed) under Results/diagnostics_v3/D1_raw/.

Usage (from the Code/ directory):
    # one SLURM array task (see Cluster/8_RunD1.sbatch)
    python -m utils.GRFPControls.RunD1 --TaskID $SLURM_ARRAY_TASK_ID
    # locally
    python -m utils.GRFPControls.RunD1 --local --workers 7
"""

from utils.GRFPControls import _bootstrap  # noqa: F401

import argparse
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from utils.GRFPControls.ControlLoop import RunControlIteration
from utils.GRFPControls.RunExpA import SELECTORS

DATASETS = ["beer", "bodyfat", "concrete_cs", "concrete_flow", "concrete_slump",
            "dgp_three_regime", "dgp_two_regime", "dgp_two_regime_gmm", "housing",
            "mpg", "no2", "pm10", "qsar", "yacht"]
N_SEEDS = 100
SEEDS_PER_TASK = 10
METRICS = ("full_pool", "full_pool_excl", "test")
RUN_KWARGS = dict(Protocol="paper", Standardize=True, LogFullPoolExclLabeled=True)


def default_out():
    return os.path.join(os.path.dirname(os.getcwd()), "Results", "diagnostics_v3", "D1_raw")


def out_path(out_root, dataset, label, seed):
    safe = label.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
    return os.path.join(out_root, dataset, f"{safe}__seed{seed:03d}.pkl")


def run_one(dataset, label, seed, out_root):
    path = out_path(out_root, dataset, label, seed)
    if os.path.exists(path):
        return path, None
    sel_type, params = SELECTORS[label]
    t0 = time.time()
    out = RunControlIteration(DataFileInput=dataset, Seed=seed, SelectorType=sel_type,
                              selector_params=params, **RUN_KWARGS)
    rec = {
        "dataset": dataset, "selector": label, "seed": seed,
        "curves": {m: np.asarray(out["Curves"][m]["RMSE"], dtype=float) for m in METRICS},
        "n_initial_train": out["n_initial_train"],
        "elapsed": time.time() - t0,
        "run_kwargs": dict(RUN_KWARGS, selector_params=params),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(rec, fh, protocol=4)
    os.replace(tmp, path)
    return path, rec["elapsed"]


def tasks():
    """(dataset, label, seeds) per array task, in a fixed order."""
    n_chunks = (N_SEEDS + SEEDS_PER_TASK - 1) // SEEDS_PER_TASK
    return [(d, lab, list(range(c * SEEDS_PER_TASK, min((c + 1) * SEEDS_PER_TASK, N_SEEDS))))
            for d in DATASETS for lab in SELECTORS for c in range(n_chunks)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--TaskID", type=int, help="1-based SLURM array index.")
    ap.add_argument("--local", action="store_true")
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--seeds", type=int, default=None, help="With --local, only seeds 0..n-1.")
    ap.add_argument("--n-tasks", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_root = args.out or default_out()
    T = tasks()

    if args.n_tasks:
        print(len(T))
        return

    if args.TaskID is not None:
        if not 1 <= args.TaskID <= len(T):
            raise SystemExit(f"TaskID {args.TaskID} outside 1..{len(T)}")
        dataset, label, seeds = T[args.TaskID - 1]
        print(f"--- {dataset} | {label} | seeds {seeds[0]}-{seeds[-1]} ---", flush=True)
        for s in seeds:
            _, el = run_one(dataset, label, s, out_root)
            print(f"    seed {s}: " + ("exists" if el is None else f"{el/60:.1f} min"), flush=True)
        return

    if not args.local:
        raise SystemExit("pass --TaskID (cluster) or --local")
    jobs = [(d, lab, s) for d, lab, seeds in T for s in seeds
            if (args.datasets is None or d in args.datasets)
            and (args.seeds is None or s < args.seeds)]
    print(f"{len(jobs)} runs, {args.workers} workers -> {out_root}", flush=True)
    t0, done = time.time(), 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for f in as_completed([ex.submit(run_one, d, lab, s, out_root) for d, lab, s in jobs]):
            f.result()
            done += 1
            if done % 100 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)}  {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
