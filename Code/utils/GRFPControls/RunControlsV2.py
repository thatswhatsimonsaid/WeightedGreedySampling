"""
Step 5: rerun the random-arm control with the rescaled bandit.

Same protocol as the September controls (RunExpA.py defaults): ridge
(alpha=0.01), repo split with CandidateProportion=0.99 and a 10% held-out test
set, raw features, held-out test RMSE, seeds 0-99, the same 13 datasets. Only
the two reward-driven selectors are rerun:

    MAB (std)        WiGS_MAB_Selector, c=2.0, reward_scaling="std"
    RandomArm (v2)   RandomArmSelector

both with cv_fallback_to_test=False. Passive and the fixed weights ignore the
reward, so their September curves are reused by AnalyzeControlsV2.py.

One pickle per (dataset, selector, seed) under Results/controls_v2/raw/, holding
both metric curves and the selector's state history in columnar form.

Usage (from the Code/ directory):
    # one SLURM array task (see Cluster/6_RunControlsV2.sbatch)
    python -m utils.GRFPControls.RunControlsV2 --TaskID $SLURM_ARRAY_TASK_ID
    # everything, locally
    python -m utils.GRFPControls.RunControlsV2 --local --workers 7
"""

from utils.GRFPControls import _bootstrap  # noqa: F401

import argparse
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from utils.GRFPControls.ControlLoop import RunControlIteration

DATASETS = ["beer", "bodyfat", "concrete_cs", "concrete_flow", "concrete_slump",
            "dgp_three_regime", "dgp_two_regime", "housing", "mpg", "no2",
            "pm10", "qsar", "yacht"]
SELECTORS = {
    "MAB (std)":      ("WiGS_MAB_Selector", {"mab_c": 2.0, "reward_scaling": "std"}),
    "RandomArm (v2)": ("RandomArmSelector", {}),
}
N_SEEDS = 100
SEEDS_PER_TASK = 10
RUN_KWARGS = dict(Protocol="repo", Standardize=False, Metric="test",
                  cv_fallback_to_test=False)


def default_out():
    return os.path.join(os.path.dirname(os.getcwd()), "Results", "controls_v2", "raw")


def out_path(out_root, dataset, label, seed):
    safe = label.replace(" ", "_").replace("(", "").replace(")", "")
    return os.path.join(out_root, dataset, f"{safe}__seed{seed:03d}.pkl")


def columnar(state_history):
    """state_history as arrays: a list of dicts is ~10x larger on disk."""
    if not state_history:
        return {}
    keys = state_history[0].keys()
    return {k: np.asarray([s[k] for s in state_history]) for k in keys}


def run_one(dataset, label, seed, out_root):
    path = out_path(out_root, dataset, label, seed)
    if os.path.exists(path):
        return path, None
    sel_type, params = SELECTORS[label]
    t0 = time.time()
    out = RunControlIteration(DataFileInput=dataset, Seed=seed,
                              SelectorType=sel_type, selector_params=params,
                              **RUN_KWARGS)
    rec = {
        "dataset": dataset, "selector": label, "seed": seed,
        "curves": {m: np.asarray(v["RMSE"], dtype=float) for m, v in out["Curves"].items()},
        "state": columnar(out["SelectorStateHistory"]),
        "n_initial_train": out["n_initial_train"],
        "n_rounds": out["n_rounds"],
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
    ap.add_argument("--local", action="store_true", help="Run every task here.")
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="With --local, restrict to these datasets.")
    ap.add_argument("--n-tasks", action="store_true", help="Print the array size and exit.")
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
            p, el = run_one(dataset, label, s, out_root)
            print(f"    seed {s}: " + ("exists" if el is None else f"{el/60:.1f} min"), flush=True)
        return

    if not args.local:
        raise SystemExit("pass --TaskID (cluster) or --local")

    jobs = [(d, lab, s) for d, lab, seeds in T for s in seeds
            if args.datasets is None or d in args.datasets]
    # longest datasets first, so the pool does not end on one slow straggler
    order = {d: i for i, d in enumerate(["dgp_three_regime", "dgp_two_regime", "qsar",
                                         "housing", "no2", "pm10", "mpg", "beer"])}
    jobs.sort(key=lambda j: order.get(j[0], 99))
    print(f"{len(jobs)} runs, {args.workers} workers -> {out_root}", flush=True)
    t0, done = time.time(), 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_one, d, lab, s, out_root) for d, lab, s in jobs]
        for f in as_completed(futs):
            f.result()
            done += 1
            if done % 100 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)}  {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
