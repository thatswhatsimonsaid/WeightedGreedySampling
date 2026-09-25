"""
Experiment A -- random-arm control.

Runs, per (dataset, seed): the UCB1 bandit, the RandomArm control, each fixed
weight in {0, 0.25, 0.5, 0.75, 1}, iGS, and passive learning. Every selector
sees the identical test set, initial labeled set and candidate pool for a given
seed, which is what makes the comparison paired.

Note on the fixed-weight endpoints: w=0 and w=1 are run through
WeightedGreedySamplingSelector for uniformity across the sweep. They are
*equivalent* to GSy/GSx: min-max normalization is a single global affine map
with positive scale, so it commutes with the min over the training set and the
argmax over the pool (verified 500/500 in verification/check_endpoints.py).

Usage (from the Code/ directory):
    python -m utils.GRFPControls.RunExpA --datasets dgp_two_regime --seeds 100 --workers 7
"""

from utils.GRFPControls import _bootstrap  # noqa: F401

import argparse
import gzip
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from utils.GRFPControls.ControlLoop import RunControlIteration
from utils.GRFPControls.ControlProtocol import LearningCurveArea

### The comparison set ###
SELECTORS = {
    "RandomArm":        ("RandomArmSelector", {}),
    "MAB (c=2.0)":      ("WiGS_MAB_Selector", {"mab_c": 2.0}),
    "Fixed w=0.00":     ("WeightedGreedySamplingSelector", {"weight_strategy": "static", "w_x": 0.0}),
    "Fixed w=0.25":     ("WeightedGreedySamplingSelector", {"weight_strategy": "static", "w_x": 0.25}),
    "Fixed w=0.50":     ("WeightedGreedySamplingSelector", {"weight_strategy": "static", "w_x": 0.50}),
    "Fixed w=0.75":     ("WeightedGreedySamplingSelector", {"weight_strategy": "static", "w_x": 0.75}),
    "Fixed w=1.00":     ("WeightedGreedySamplingSelector", {"weight_strategy": "static", "w_x": 1.0}),
    "iGS":              ("GreedySamplingSelector", {"strategy": "iGS"}),
    "Passive":          ("PassiveLearningSelector", {}),
}

ARM_LOGGING = {"RandomArm", "MAB (c=2.0)"}


def _one(job):
    """Worker: run a single (dataset, selector, seed) configuration."""
    dataset, label, seed, max_iter, protocol, standardize = job
    sel_type, params = SELECTORS[label]
    t0 = time.time()
    out = RunControlIteration(DataFileInput=dataset,
                              Seed=seed,
                              SelectorType=sel_type,
                              selector_params=params,
                              MaxIterations=max_iter,
                              Protocol=protocol,
                              Standardize=standardize)
    curve = out["ErrorVecs"]["RMSE"]
    rec = {
        "dataset": dataset,
        "selector": label,
        "seed": seed,
        "area": LearningCurveArea(curve),
        "final_rmse": float(curve[-1]),
        "n_rounds": len(curve),
        "elapsed": time.time() - t0,
    }
    arms = None
    if label in ARM_LOGGING:
        arms = []
        for r in out["ArmLog"]:
            row = {"dataset": dataset, "selector": label, "seed": seed,
                   "round": r["round"], "w_x": r["w_x"],
                   "arm_index": int(r["arm_index"]) if not pd.isna(r["arm_index"]) else -1,
                   "reward": r["reward_derived"],
                   "cv_rmse": r["cv_rmse"], "test_rmse": r["test_rmse"]}
            # One numeric column per arm. Writing the arrays themselves would
            # serialise as repr strings ("[np.float64(0.0), ...]"), which is both
            # unreadable and an order of magnitude larger on disk.
            for k, v in enumerate(r["arm_values"] if r["arm_values"] is not None else []):
                row[f"arm_value_{k}"] = float(v)
            for k, v in enumerate(r["arm_counts"] if r["arm_counts"] is not None else []):
                row[f"arm_count_{k}"] = float(v)
            arms.append(row)
    return rec, curve, arms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--max-iter", type=int, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--protocol", choices=["repo", "paper"], default="repo",
                    help="'paper' uses the manuscript's stated split: 10%% test, "
                         "5%% initial train, 85%% candidate. 'repo' uses "
                         "CandidateProportion=0.99 from the cluster config, which "
                         "gives a 0.9%% initial train instead.")
    ap.add_argument("--resume", action="store_true",
                    help="Skip datasets that already have a shard on disk.")
    ap.add_argument("--standardize", action="store_true",
                    help="Rescale features to zero mean / unit variance, the "
                         "manuscript's main-text preprocessing. Data/processed was "
                         "built with RobustScaler.")
    args = ap.parse_args()

    out_dir = args.out or os.path.join(os.path.dirname(os.getcwd()), "results_grfp_controls")
    os.makedirs(out_dir, exist_ok=True)

    jobs = [(d, lab, s, args.max_iter, args.protocol, args.standardize)
            for d in args.datasets
            for lab in SELECTORS
            for s in range(args.seeds)]
    print(f"Experiment A [protocol={args.protocol}, standardize={args.standardize}]: {len(jobs)} runs "
          f"({len(args.datasets)} datasets x {len(SELECTORS)} selectors x {args.seeds} seeds), "
          f"{args.workers} workers")

    # Checkpoint per dataset. A full sweep is several hours, and writing only at
    # the very end means an interruption loses everything -- which is exactly
    # what happened once. Each dataset's results are written as a shard as soon
    # as it finishes, and the combined files are rebuilt from the shards, so the
    # top-level outputs are always complete-as-of-the-last-dataset.
    shard_dir = os.path.join(out_dir, "shards")
    os.makedirs(shard_dir, exist_ok=True)

    t_start = time.time()
    done = 0
    for dataset in args.datasets:
        shard = os.path.join(shard_dir, f"{dataset}.pkl")
        ds_jobs = [j for j in jobs if j[0] == dataset]
        if args.resume and os.path.exists(shard):
            done += len(ds_jobs)
            print(f"  [skip] {dataset} already done ({done}/{len(jobs)})", flush=True)
            continue

        recs, curv, arms_ds = [], {}, []
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_one, j) for j in ds_jobs]
            for f in as_completed(futs):
                rec, curve, arms = f.result()
                recs.append(rec)
                curv[(rec["dataset"], rec["selector"], rec["seed"])] = np.asarray(curve, dtype=float)
                if arms:
                    arms_ds.extend(arms)
                done += 1
                if done % 50 == 0:
                    el = time.time() - t_start
                    print(f"  {done}/{len(jobs)}  elapsed={el/60:.1f}m", flush=True)

        with open(shard, "wb") as fh:
            pickle.dump({"records": recs, "curves": curv, "arms": arms_ds}, fh, protocol=4)
        print(f"  [done] {dataset}: {len(recs)} runs -> {os.path.basename(shard)} "
              f"({(time.time()-t_start)/60:.1f}m elapsed)", flush=True)
        _rebuild(out_dir, shard_dir)

    _rebuild(out_dir, shard_dir)
    print(f"\nTotal wall time: {(time.time()-t_start)/60:.1f} min")


def _rebuild(out_dir, shard_dir):
    """Rebuild the combined outputs from whatever shards exist."""
    records, curves, arm_rows = [], {}, []
    for fn in sorted(os.listdir(shard_dir)):
        if not fn.endswith(".pkl"):
            continue
        with open(os.path.join(shard_dir, fn), "rb") as fh:
            d = pickle.load(fh)
        records.extend(d["records"])
        curves.update(d["curves"])
        arm_rows.extend(d["arms"])
    if not records:
        return
    pd.DataFrame(records).sort_values(["dataset", "selector", "seed"]).to_csv(
        os.path.join(out_dir, "exp_a_runs.csv"), index=False)
    with gzip.open(os.path.join(out_dir, "exp_a_curves.pkl.gz"), "wb", compresslevel=6) as fh:
        pickle.dump(curves, fh, protocol=4)
    if arm_rows:
        pd.DataFrame(arm_rows).to_csv(os.path.join(out_dir, "exp_a_arms.csv.gz"),
                                      index=False, compression="gzip", float_format="%.6g")


if __name__ == "__main__":
    main()
