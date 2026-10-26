"""
Cluster entry point: one (dataset, seed, selector) per SLURM array task.

Mirrors `Code/RunSimulation.py` in shape, but splits the job at the selector
level rather than running all strategies in one task. `wine_white` takes about
80 minutes for a single cheap selector and roughly twice that for the ones that
cross-validate, so a task covering all 18 would not fit a 24-hour limit.

Each task writes one small pickle; `AggregateControls.py` merges them into the
shard format the analysis already reads.

Usage (from the Code/ directory):
    python -m utils.GRFPControls.RunControlsCluster \\
        --Data wine_white --TaskID $SLURM_ARRAY_TASK_ID --NReplications 100
"""

from utils.GRFPControls import _bootstrap  # noqa: F401

import argparse
import os
import pickle
import time

import numpy as np

from utils.GRFPControls.ControlLoop import RunControlIteration
from utils.GRFPControls.FullSuite import STRATEGIES, ARM_LOGGING, params_for


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Data", type=str, required=True)
    ap.add_argument("--TaskID", type=int, required=True, help="1-based SLURM array index.")
    ap.add_argument("--NReplications", type=int, required=True)
    ap.add_argument("--Protocol", choices=["repo", "paper"], default="paper")
    ap.add_argument("--Standardize", type=int, default=1)
    ap.add_argument("--SeedsPerTask", type=int, default=1,
                    help="Seeds handled by one array task. >1 shrinks the array, "
                         "which matters under a QOSMaxSubmitJobPerUser cap: array "
                         "elements count individually, so 18x100 per dataset is "
                         "1800 submitted jobs.")
    ap.add_argument("--OutRoot", type=str, default=None)
    args = ap.parse_args()

    labels = list(STRATEGIES)
    spt = max(1, args.SeedsPerTask)
    n_chunks = (args.NReplications + spt - 1) // spt
    idx = args.TaskID - 1
    sel_index, chunk = divmod(idx, n_chunks)
    if sel_index >= len(labels):
        raise SystemExit(f"TaskID {args.TaskID} exceeds "
                         f"{len(labels) * n_chunks} tasks for this dataset.")
    label = labels[sel_index]
    seeds = list(range(chunk * spt, min((chunk + 1) * spt, args.NReplications)))
    if not seeds:
        raise SystemExit(f"TaskID {args.TaskID} maps to an empty seed range.")

    out_root = args.OutRoot or os.path.join(os.path.dirname(os.getcwd()),
                                            "results_grfp_controls", "cluster")
    out_dir = os.path.join(out_root, args.Data)
    os.makedirs(out_dir, exist_ok=True)
    safe = label.replace("/", "_").replace(" ", "_")

    print(f"--- {args.Data} | {label} | seeds {seeds[0]}-{seeds[-1]} "
          f"| protocol={args.Protocol} standardize={bool(args.Standardize)} ---", flush=True)
    for seed in seeds:
        _run_one(args, label, safe, seed, out_dir)


def _run_one(args, label, safe, seed, out_dir):
    out_path = os.path.join(out_dir, f"{safe}__seed{seed:03d}.pkl")
    if os.path.exists(out_path):
        print(f"[skip] seed {seed} exists", flush=True)
        return

    sel_type, params = params_for(label, seed=seed)
    t0 = time.time()
    out = RunControlIteration(DataFileInput=args.Data,
                              Seed=seed,
                              SelectorType=sel_type,
                              selector_params=params,
                              Protocol=args.Protocol,
                              Standardize=bool(args.Standardize))
    elapsed = time.time() - t0

    # Keep both metric curves; drop the heavy per-round structures for
    # selectors whose arms are not of interest.
    rec = {
        "dataset": args.Data, "selector": label, "seed": seed,
        "curves": {m: np.asarray(v["RMSE"], dtype=np.float32)
                   for m, v in out["Curves"].items()},
        "curves_other": {m: {k: np.asarray(v[k], dtype=np.float32)
                             for k in ("MAE", "R2", "CC")}
                         for m, v in out["Curves"].items()},
        "n_rounds": out["n_rounds"],
        "n_initial_train": out["n_initial_train"],
        "n_test": out["n_test"],
        "elapsed": elapsed,
        "protocol": args.Protocol,
        "standardize": bool(args.Standardize),
    }
    if label in ARM_LOGGING:
        rec["arms"] = [{"round": r["round"], "w_x": r["w_x"],
                        "arm_index": r["arm_index"],
                        "reward": r["reward_derived"],
                        "cv_rmse": r["cv_rmse"]}
                       for r in out["ArmLog"]]

    with open(out_path, "wb") as fh:
        pickle.dump(rec, fh, protocol=4)
    print(f"    seed {seed}: {elapsed/60:.1f} min -> {os.path.basename(out_path)}", flush=True)


if __name__ == "__main__":
    main()
