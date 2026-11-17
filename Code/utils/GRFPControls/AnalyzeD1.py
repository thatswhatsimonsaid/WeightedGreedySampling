"""
Diagnostic 1 summary: full-pool error with labelled points excluded.

Reads Results/diagnostics_v3/D1_raw/ (RunD1.py) and writes
Results/diagnostics_v3/D1_full_pool_excl_labeled.md, plus per-run areas and
per-dataset comparisons as CSV.

Scoring follows SUMMARY.md: trapezoidal area under the RMSE curve over the full
budget and the first 25% of rounds (25% of the shortest curve on the dataset);
"vs passive" is 100 * mean(area - area_passive) / mean(area_passive), paired on
seed, averaged over datasets; paired comparisons use the percentile bootstrap
(10,000 resamples, seed 17); relative AUC is mean(area) / mean(area_iGS).
`full_pool_excl` is undefined once the candidate set is empty, so its area is
taken over the rounds where it is defined.

Before scoring, every rerun curve is compared with the September sweeps:
`full_pool` against paper_metric_fullpool, `test` against paper_protocol.

Usage (from the Code/ directory):
    python -m utils.GRFPControls.AnalyzeD1
"""

from utils.GRFPControls import _bootstrap  # noqa: F401

import argparse
import glob
import gzip
import os
import pickle

import numpy as np
import pandas as pd

from utils.GRFPControls.Analyze import EARLY_FRAC, paired_bootstrap_ci
from utils.GRFPControls.ControlProtocol import LearningCurveArea
from utils.GRFPControls.RunD1 import DATASETS, METRICS, N_SEEDS, SELECTORS, default_out

MAB, RANDOM, IGS, PASSIVE = "MAB (c=2.0)", "RandomArm", "iGS", "Passive"
STATIC_025 = "Fixed w=0.25"
TWO_REGIME = ["dgp_two_regime", "dgp_two_regime_gmm"]
SEPT = {"full_pool": "paper_metric_fullpool", "test": "paper_protocol"}
METRIC_NAME = {"full_pool": "Full-pool (paper)", "full_pool_excl": "Full-pool excl. labelled",
               "test": "Test RMSE"}
BUDGETS = ("25", "100")


def load_runs(raw):
    runs = {}
    for p in glob.glob(os.path.join(raw, "*", "*.pkl")):
        with open(p, "rb") as fh:
            r = pickle.load(fh)
        runs[(r["dataset"], r["selector"], r["seed"])] = r["curves"]
    return runs


def reproduction(runs, root):
    rows = []
    for metric, sweep in SEPT.items():
        with gzip.open(os.path.join(root, "results_grfp_controls", sweep, "exp_a_curves.pkl.gz"), "rb") as fh:
            sept = pickle.load(fh)
        for (d, sel, s), c in runs.items():
            if (d, sel, s) in sept:
                rows.append({"metric": metric, "selector": sel,
                             "exact": bool(np.array_equal(c[metric], np.asarray(sept[(d, sel, s)], float)))})
    df = pd.DataFrame(rows)
    return df.groupby(["metric", "selector"]).exact.agg(["sum", "size"]).reset_index()


def areas(runs):
    rows = []
    for d in DATASETS:
        keys = [k for k in runs if k[0] == d]
        if not keys:
            continue
        for m in METRICS:
            n_early = max(2, int(round(min(len(runs[k][m]) for k in keys) * EARLY_FRAC)))
            for k in keys:
                c = runs[k][m]
                c = c[np.isfinite(c)] if m == "full_pool_excl" else c
                rows.append({"dataset": d, "selector": k[1], "seed": k[2], "metric": m,
                             "area_100": LearningCurveArea(c),
                             "area_25": LearningCurveArea(c[:n_early])})
    return pd.DataFrame(rows)


def paired(wide, a, b):
    p = wide[[a, b]].dropna()
    m, lo, hi = paired_bootstrap_ci((p[a] - p[b]).values, n_boot=10000, seed=17)
    base = p[b].mean()
    return {"n": len(p), "diff": m, "lo": lo, "hi": hi, "pct": 100 * m / base,
            "pct_lo": 100 * lo / base, "pct_hi": 100 * hi / base,
            "verdict": "better" if hi < 0 else "worse" if lo > 0 else "tie"}


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(os.getcwd()))
    ap.add_argument("--raw", default=None)
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()
    raw = args.raw or default_out()
    out_dir = os.path.dirname(raw)

    runs = load_runs(raw)
    present = pd.DataFrame(list(runs), columns=["dataset", "selector", "seed"])
    grid = present.pivot_table(index="dataset", columns="selector", values="seed", aggfunc="nunique")
    if (grid.reindex(index=DATASETS, columns=list(SELECTORS)).fillna(0) < N_SEEDS).any().any():
        print(grid.to_string())
        if not args.allow_partial:
            raise SystemExit("D1 runs incomplete; resubmit missing tasks or pass --allow-partial")

    rep = reproduction(runs, args.root)
    print(rep.to_string(index=False))

    A = areas(runs)
    sections = []
    sections.append("## Reproduction of the September curves (exact match / runs compared)\n\n" + md_table(
        ["Selector"] + [f"{METRIC_NAME[m]} vs `{SEPT[m]}`" for m in SEPT],
        [[sel] + [(lambda r: f"{int(r['sum'].iloc[0])} / {int(r['size'].iloc[0])}" if len(r) else "—")(
            rep[(rep.metric == m) & (rep.selector == sel)]) for m in SEPT] for sel in SELECTORS]))

    ### 1. Change in area vs passive ###
    vs_passive = []
    for (d, m), g in A.groupby(["dataset", "metric"]):
        for b in BUDGETS:
            wide = g.pivot_table(index="seed", columns="selector", values=f"area_{b}")
            for sel in SELECTORS:
                if sel != PASSIVE and sel in wide:
                    vs_passive.append({"dataset": d, "metric": m, "budget": b, "selector": sel,
                                       **paired(wide, sel, PASSIVE)})
    vs_passive = pd.DataFrame(vs_passive)
    n_ds = A.dataset.nunique()
    rows = []
    for sel in SELECTORS:
        if sel == PASSIVE:
            continue
        row = [sel]
        for m in METRICS:
            for b in BUDGETS:
                g = vs_passive[(vs_passive.selector == sel) & (vs_passive.metric == m) & (vs_passive.budget == b)]
                row.append(f"{g.pct.mean():+.2f}%")
        rows.append(row)
    sections.append(f"## Change in area vs passive, mean over {n_ds} datasets\n\n" + md_table(
        ["Selector"] + [f"{METRIC_NAME[m]}, {'first 25%' if b == '25' else 'full'}"
                        for m in METRICS for b in BUDGETS], rows))

    ### 2. Paired counts: MAB vs iGS and vs random arm, both full-pool variants ###
    pair_rows, detail = [], []
    for ref in (IGS, RANDOM):
        for m in ("full_pool", "full_pool_excl"):
            for b in BUDGETS:
                counts = {"better": 0, "worse": 0, "tie": 0}
                for d, g in A[A.metric == m].groupby("dataset"):
                    wide = g.pivot_table(index="seed", columns="selector", values=f"area_{b}")
                    r = paired(wide, MAB, ref)
                    counts[r["verdict"]] += 1
                    detail.append({"comparison": f"{MAB} vs {ref}", "metric": m, "budget": b,
                                   "dataset": d, **r})
                pair_rows.append([f"{MAB} vs {ref}", METRIC_NAME[m],
                                  "first 25%" if b == "25" else "full",
                                  counts["better"], counts["worse"], counts["tie"]])
    sections.append("## Paired counts per dataset, 95% CI over seeds (better = lower area)\n\n"
                    + md_table(["Comparison", "Metric", "Budget", "better", "worse", "tie"], pair_rows))

    ### 3. Two-Regime ###
    rows = []
    for d in TWO_REGIME:
        for sel in (MAB, STATIC_025):
            row = [d, sel]
            for m in METRICS:
                g = A[(A.dataset == d) & (A.metric == m)]
                if g.empty:
                    row += ["—"] * 4
                    continue
                for b in BUDGETS:
                    wide = g.pivot_table(index="seed", columns="selector", values=f"area_{b}")
                    row.append(f"{wide[sel].mean() / wide[IGS].mean():.4f}")
                    row.append(f"{paired(wide, sel, PASSIVE)['pct']:+.2f}%")
            rows.append(row)
    sections.append("## Two-Regime: relative AUC vs iGS (mean area / mean iGS area) and change vs passive\n\n"
                    + md_table(["Dataset", "Selector"]
                               + [f"{METRIC_NAME[m]}, {'25%' if b == '25' else 'full'}, {k}"
                                  for m in METRICS for b in BUDGETS for k in ("rel. iGS", "vs passive")],
                               rows))

    md = "# Diagnostic 1: full-pool error with labelled points excluded\n\n" + "\n\n".join(sections) + "\n"
    with open(os.path.join(out_dir, "D1_full_pool_excl_labeled.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    A.to_csv(os.path.join(out_dir, "D1_areas.csv.gz"), index=False, compression="gzip")
    vs_passive.to_csv(os.path.join(out_dir, "D1_vs_passive.csv"), index=False)
    pd.DataFrame(detail).to_csv(os.path.join(out_dir, "D1_paired.csv"), index=False)
    print(md)
    print(f"-> {os.path.join(out_dir, 'D1_full_pool_excl_labeled.md')}")


if __name__ == "__main__":
    main()
