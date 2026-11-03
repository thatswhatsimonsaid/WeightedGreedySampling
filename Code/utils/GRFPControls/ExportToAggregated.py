"""
Export the control sweep into the manuscript's aggregated layout, then drive
the manuscript's own plotting code over it.

Everything here reuses `utils/Auxiliary` rather than reimplementing it, so the
figures come out with the repository's colours, linestyles, legend mapping,
axis labels ("Percent of Learning Pool Labeled") and one file per dataset:

    GeneratePlots.generate_all_plots        trace + trace_relative_iGS
    GenerateAUCTable                        relative-AUC heatmaps
    GenerateLabellingComplexity             N_rel boxplot

The layout those functions expect is

    <dir>/<dataset>/full_pool_metrics/RMSE.pkl   {selector: DataFrame(round x seed)}
    <dir>/<dataset>/InitialIndices.csv           one row per initial labelled point

`eval_types` is hard-coded to 'full_pool' inside generate_all_plots, so each
metric is exported to its own directory under that same folder name and plotted
separately — the directory, not the folder name, is what distinguishes them.

Usage (from the Code/ directory):
    python -m utils.GRFPControls.ExportToAggregated \\
        --suite ../results_grfp_controls/full_suite \\
        --out   ../results_grfp_controls/manuscript_style
"""

from utils.GRFPControls import _bootstrap  # noqa: F401

import argparse
import gzip
import os
import pickle

import numpy as np
import pandas as pd

TRAIN_PROPORTION = 0.05   # the manuscript's initial labelled share


def _load_curves(metric_dir):
    for name in ("exp_a_curves.pkl.gz", "exp_a_curves.pkl"):
        p = os.path.join(metric_dir, name)
        if os.path.exists(p):
            op = gzip.open if p.endswith(".gz") else open
            with op(p, "rb") as fh:
                return pickle.load(fh)
    raise SystemExit(f"no curves under {metric_dir}")


def initial_train_size(dataset):
    """
    Rows in the initial labelled set, as PaperTrainCandidateSplit computes it.

    Derived from the dataset rather than stored, so the export does not need to
    re-read 36,000 task pickles just to recover one integer per dataset.
    """
    from utils.GRFPControls.ControlLoop import _load_dataset
    n = len(_load_dataset(dataset))
    return max(2, int(round(n * TRAIN_PROPORTION)))


def export(metric_dir, out_dir, metric_name="RMSE", budget=1.0):
    """
    Write the aggregated layout the Auxiliary plotting code reads.

    `budget` < 1 truncates every curve to that leading fraction of its rounds.
    GenerateAUCTable integrates whatever curve it is given over the full length
    and has no truncation option, so truncating here is how a budget-limited
    relative AUC is obtained from the repository's own code.
    """
    curves = _load_curves(metric_dir)
    datasets = sorted({k[0] for k in curves})
    for ds in datasets:
        sels = sorted({k[1] for k in curves if k[0] == ds})
        per_sel = {}
        for sel in sels:
            seeds = sorted(k[2] for k in curves if k[0] == ds and k[1] == sel)
            cols = {f"Sim_{s}": np.asarray(curves[(ds, sel, s)], dtype=float)
                    for s in seeds}
            n = min(len(v) for v in cols.values())
            if budget < 1.0:
                n = max(2, int(round(n * budget)))
            per_sel[sel] = pd.DataFrame({k: v[:n] for k, v in cols.items()})

        d = os.path.join(out_dir, ds, "full_pool_metrics")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"{metric_name}.pkl"), "wb") as fh:
            pickle.dump(per_sel, fh, protocol=4)

        # InitialIndices.csv: only its row count is read, for the x axis.
        n0 = initial_train_size(ds)
        pd.DataFrame({"index": range(n0)}).to_csv(
            os.path.join(out_dir, ds, "InitialIndices.csv"), index=False)
    print(f"  exported {len(datasets)} datasets -> {out_dir}")
    return datasets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True,
                    help="results_grfp_controls/full_suite")
    ap.add_argument("--out", required=True,
                    help="Where to write the aggregated layout and figures.")
    ap.add_argument("--metrics", nargs="+", default=["full_pool", "test"])
    ap.add_argument("--budget", type=float, default=0.70,
                    help="Leading fraction of each curve used for the AUC "
                         "heatmap. Trace plots and label efficiency always use "
                         "the full curve.")
    ap.add_argument("--no-plots", action="store_true",
                    help="Export only; do not call the Auxiliary plotting code.")
    args = ap.parse_args()

    for metric in args.metrics:
        src = os.path.join(args.suite, metric)
        if not os.path.isdir(src):
            print(f"[skip] {src} missing"); continue

        agg = os.path.join(args.out, metric, "aggregated")
        img = os.path.join(args.out, metric, "images")
        os.makedirs(img, exist_ok=True)
        print(f"\n=== {metric} ===")
        export(src, agg)

        if args.no_plots:
            continue

        ### 1. Trace plots, one file per dataset ###
        from utils.Auxiliary.GeneratePlots import generate_all_plots
        generate_all_plots(aggregated_results_dir=agg, image_dir=img,
                           show_legend=True)

        ### 2. Relative-AUC heatmaps, over --budget of the curve ###
        from utils.Auxiliary import GenerateAUCTable as AUC
        if args.budget < 1.0:
            agg_b = os.path.join(args.out, metric, f"aggregated_{int(args.budget*100)}pct")
            print(f"  truncating to {int(args.budget*100)}% for the AUC heatmap")
            export(src, agg_b, budget=args.budget)
        else:
            agg_b = agg
        auc_df = AUC.load_and_calculate_auc_from_dirs(agg_b)
        auc_out = os.path.join(img, f"AUC_Tables_{int(args.budget*100)}pct")
        os.makedirs(auc_out, exist_ok=True)
        for baseline in AUC.TARGET_BASELINES:
            try:
                AUC.generate_heatmap(auc_df, auc_out, baseline)
            except Exception as e:                      # a baseline may be absent
                print(f"    [skip heatmap vs {baseline}: {type(e).__name__}]")

        ### 3. Label efficiency, on the full curve ###
        # plot_efficiency_boxplot saves to the module-level OUTPUT_DIR and
        # returns None; that default is the relative "Results/images/manuscript/",
        # which would land inside Code/ and write into a Results tree. Point it
        # at our own directory before calling, and do not expect a figure back.
        # The 70%/80% here are shares of the achievable gain, unrelated to the
        # AUC budget, so this uses the untruncated export.
        from utils.Auxiliary import GenerateLabellingComplexity as LC
        LC.OUTPUT_DIR = img
        ratios = LC.calculate_n_rel(agg)
        if ratios is not None and len(ratios):
            LC.plot_efficiency_boxplot(ratios)
            ratios.to_csv(os.path.join(img, "label_efficiency.csv"), index=False)
            print(f"  label efficiency: {len(ratios)} points, "
                  f"{ratios['Method'].nunique()} methods, "
                  f"targets {sorted(ratios['Target'].unique())}")
        else:
            print("  [label efficiency produced no rows]")

    print(f"\nDone. Figures under {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
