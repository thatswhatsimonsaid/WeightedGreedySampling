"""
Merge the per-task cluster pickles into the format the analysis already reads.

`RunControlsCluster.py` writes one pickle per (dataset, seed, selector). This
turns a directory of those into `exp_a_runs.csv` and `exp_a_curves.pkl.gz` per
metric, which `RelativeToIGS.py`, `WinLossTie.py`, `BudgetSensitivity.py` and
`Analyze.py` all consume unchanged.

Usage (from the Code/ directory):
    python -m utils.GRFPControls.AggregateControls \\
        --in results_grfp_controls/cluster --out results_grfp_controls/full_suite
"""

from utils.GRFPControls import _bootstrap  # noqa: F401

import argparse
import glob
import gzip
import os
import pickle

import numpy as np
import pandas as pd

from utils.GRFPControls.ControlProtocol import LearningCurveArea


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--metrics", nargs="+", default=["full_pool", "test"])
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.src, "**", "*__seed*.pkl"), recursive=True))
    if not files:
        raise SystemExit(f"no per-task pickles under {args.src}")
    print(f"{len(files)} task files")

    for metric in args.metrics:
        out_dir = os.path.join(args.dst, metric)
        os.makedirs(out_dir, exist_ok=True)
        recs, curves, arm_rows, missing = [], {}, [], 0
        for p in files:
            with open(p, "rb") as fh:
                r = pickle.load(fh)
            if metric not in r["curves"]:
                missing += 1
                continue
            c = np.asarray(r["curves"][metric], dtype=float)
            key = (r["dataset"], r["selector"], r["seed"])
            curves[key] = c
            recs.append({"dataset": r["dataset"], "selector": r["selector"],
                         "seed": r["seed"], "area": LearningCurveArea(c),
                         "final_rmse": float(c[-1]), "n_rounds": len(c),
                         "elapsed": r.get("elapsed", np.nan)})
            for a in r.get("arms", []):
                arm_rows.append({"dataset": r["dataset"], "selector": r["selector"],
                                 "seed": r["seed"], **a})

        if not recs:
            print(f"[{metric}] nothing to write"); continue
        df = pd.DataFrame(recs).sort_values(["dataset", "selector", "seed"])
        df.to_csv(os.path.join(out_dir, "exp_a_runs.csv"), index=False)
        with gzip.open(os.path.join(out_dir, "exp_a_curves.pkl.gz"), "wb", compresslevel=6) as fh:
            pickle.dump(curves, fh, protocol=4)
        if arm_rows:
            pd.DataFrame(arm_rows).to_csv(os.path.join(out_dir, "exp_a_arms.csv.gz"),
                                          index=False, compression="gzip",
                                          float_format="%.6g")

        ### Report completeness, since a job array can come back partial ###
        grid = df.groupby("dataset").agg(selectors=("selector", "nunique"),
                                         seeds=("seed", "nunique"),
                                         runs=("seed", "size"))
        print(f"\n[{metric}] -> {out_dir}  ({len(df)} runs"
              + (f", {missing} task files lacked this metric)" if missing else ")"))
        print(grid.to_string())
        short = grid[grid.runs < grid.selectors * grid.seeds]
        if len(short):
            print("  incomplete (selectors x seeds not fully populated):",
                  ", ".join(short.index))


if __name__ == "__main__":
    main()
