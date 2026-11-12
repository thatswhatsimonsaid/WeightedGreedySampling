"""
Trace plots for the non-WiGS baselines only, on RMSE and CC, for sharing with
the authors of another paper.

Reads the per-task pickles of the full control sweep (cluster-only;
results_grfp_controls/cluster/ is not tracked), keeps the five baselines, and
writes them in the aggregated layout the repository's plotting code reads. The
figures are then drawn by `GeneratePlots.generate_all_plots` unmodified, so
colours, linestyles, labels and axes match the manuscript. Only the five
strategies are exported, which is what restricts the plots to them.

    <out>/<metric>/aggregated/<dataset>/full_pool_metrics/{RMSE,CC}.pkl
    <out>/<metric>/images/trace_plots/{RMSE,CC}/{trace,trace_relative_iGS}/...

`<metric>` is `full_pool` or `test` (held-out); generate_all_plots hard-codes
the folder name `full_pool_metrics`, so each is exported to its own directory.

Usage (from the Code/ directory):
    python -m utils.GRFPControls.BaselineTraces \\
        --in ../results_grfp_controls/cluster \\
        --out ../results_grfp_controls/baseline_traces
"""

from utils.GRFPControls import _bootstrap  # noqa: F401

import argparse
import glob
import os
import pickle

import numpy as np
import pandas as pd

from utils.GRFPControls.ExportToAggregated import reset_style

BASELINES = ["Passive Learning", "QBC", "GSx", "GSy", "iGS"]
MEASURES = ["RMSE", "CC"]


def _safe(label):
    # same file naming as RunControlsCluster
    return label.replace("/", "_").replace(" ", "_")


def _series(rec, metric, measure):
    if measure == "RMSE":
        return rec["curves"][metric]
    return rec["curves_other"][metric][measure]


def export(src, out, metrics):
    datasets = sorted(d for d in os.listdir(src) if os.path.isdir(os.path.join(src, d)))
    report = []
    for ds in datasets:
        recs = {}
        for lab in BASELINES:
            for p in sorted(glob.glob(os.path.join(src, ds, f"{_safe(lab)}__seed*.pkl"))):
                with open(p, "rb") as fh:
                    r = pickle.load(fh)
                recs.setdefault(lab, []).append(r)
        if not recs:
            continue
        n0 = next(iter(recs.values()))[0]["n_initial_train"]
        for metric in metrics:
            d = os.path.join(out, metric, "aggregated", ds, "full_pool_metrics")
            os.makedirs(d, exist_ok=True)
            for measure in MEASURES:
                per_sel = {}
                for lab, rs in recs.items():
                    cols = {f"Sim_{r['seed']}": np.asarray(_series(r, metric, measure), dtype=float)
                            for r in sorted(rs, key=lambda r: r["seed"])}
                    n = min(len(v) for v in cols.values())
                    per_sel[lab] = pd.DataFrame({k: v[:n] for k, v in cols.items()})
                with open(os.path.join(d, f"{measure}.pkl"), "wb") as fh:
                    pickle.dump(per_sel, fh, protocol=4)
                if metric == metrics[0]:
                    nan = sum(int(df.isna().values.sum()) for df in per_sel.values())
                    report.append({"dataset": ds, "measure": measure, "nan_cells": nan,
                                   **{lab: len(rs) for lab, rs in recs.items()}})
            # InitialIndices.csv: only its row count is read, for the x axis.
            pd.DataFrame({"index": range(n0)}).to_csv(
                os.path.join(out, metric, "aggregated", ds, "InitialIndices.csv"), index=False)
    rep = pd.DataFrame(report)
    print("Seeds per strategy (and NaN cells per measure):")
    print(rep.to_string(index=False))
    missing = [lab for lab in BASELINES if lab not in rep]
    if missing:
        print(f"WARNING: no runs found for {missing}")
    short = rep[(rep[[c for c in BASELINES if c in rep]] < rep[[c for c in BASELINES if c in rep]].max().max()).any(axis=1)]
    if len(short):
        print("WARNING: incomplete seeds on:", ", ".join(sorted(set(short.dataset))))
    return datasets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True, help="results_grfp_controls/cluster")
    ap.add_argument("--out", required=True)
    ap.add_argument("--metrics", nargs="+", default=["full_pool", "test"])
    args = ap.parse_args()

    export(args.src, args.out, args.metrics)

    from utils.Auxiliary.GeneratePlots import generate_all_plots
    for metric in args.metrics:
        print(f"\n=== {metric} ===")
        reset_style()
        generate_all_plots(aggregated_results_dir=os.path.join(args.out, metric, "aggregated"),
                           image_dir=os.path.join(args.out, metric, "images"),
                           show_legend=True)
    n = len(glob.glob(os.path.join(args.out, "*", "images", "**", "*TracePlot.png"), recursive=True))
    print(f"\n{n} trace plots under {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
