"""
Diagnostic 2: how often the three arms choose the same point.

Reads the v2 control pickles (Results/controls_v2/raw/<dataset>/*.pkl). Each
holds the selector state per round in columnar form under `state`, including
`candidates_by_arm` (round x 3, the point each of w=0.25/0.5/0.75 would query)
and `arm_pulled`. Per round:

    n_distinct     distinct candidates among the three arms (1, 2 or 3)
    pulled_unique  1 if the pulled arm's point was chosen by no other arm

Rows are per (dataset, selector): the MAB (std) and the random-arm runs follow
different trajectories, so they are kept apart. Fractions are per run, then the
median over seeds.

Outputs, under Results/diagnostics_v3/:
    D2_agreement.csv            one row per dataset x selector
    D2_agreement_by_decile.csv  one row per dataset x selector x round decile

Usage (from the repository root):
    python scripts/d2_agreement.py
"""

import argparse
import glob
import os
import pickle

import numpy as np
import pandas as pd

EARLY_FRAC = 0.25
FIXED_HI, FIXED_LO = "Fixed w=0.75", "Fixed w=0.25"


def per_run(state):
    c = np.asarray(state["candidates_by_arm"])
    pulled = np.asarray(state["arm_pulled"], dtype=int)
    T = len(c)
    n_distinct = np.array([len(set(row)) for row in c])
    mine = c[np.arange(T), pulled]
    others = np.array([[c[t, k] for k in range(c.shape[1]) if k != pulled[t]] for t in range(T)])
    pulled_unique = (others != mine[:, None]).all(axis=1).astype(int)
    return n_distinct, pulled_unique


def fractions(n_distinct, prefix=""):
    return {f"{prefix}frac_distinct_{k}": float(np.mean(n_distinct == k)) for k in (1, 2, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join("Results", "controls_v2", "raw"))
    ap.add_argument("--areas", default=os.path.join("Results", "controls_v2", "v2_areas.csv"))
    ap.add_argument("--out", default=os.path.join("Results", "diagnostics_v3"))
    args = ap.parse_args()

    runs, deciles = [], []
    for p in sorted(glob.glob(os.path.join(args.raw, "*", "*.pkl"))):
        with open(p, "rb") as fh:
            r = pickle.load(fh)
        nd, pu = per_run(r["state"])
        T = len(nd)
        n_early = max(2, int(round(T * EARLY_FRAC)))
        key = {"dataset": r["dataset"], "selector": r["selector"], "seed": r["seed"]}
        runs.append({**key, "n_rounds": T,
                     **fractions(nd), **fractions(nd[:n_early], "early_"),
                     "pulled_unique_rate": float(pu.mean())})
        dec = np.minimum((10 * np.arange(T)) // T, 9)
        for k in range(10):
            m = dec == k
            deciles.append({**key, "decile": k + 1, **fractions(nd[m]),
                            "pulled_unique_rate": float(pu[m].mean())})

    runs = pd.DataFrame(runs)
    cols = [c for c in runs if c.startswith(("frac_", "early_", "pulled_"))]
    summary = runs.groupby(["dataset", "selector"]).agg(
        n_seeds=("seed", "size"), n_rounds=("n_rounds", "median"),
        **{c: (c, "median") for c in cols}).reset_index()
    summary["disagreement_rate"] = 1.0 - summary["frac_distinct_1"]

    ### Cross-check: fixed-weight area gap, w=0.75 minus w=0.25, test RMSE ###
    A = pd.read_csv(args.areas)
    gap = []
    for d, g in A[A.selector.isin([FIXED_HI, FIXED_LO])].groupby("dataset"):
        row = {"dataset": d}
        for b in ("25", "100"):
            w = g.pivot_table(index="seed", columns="selector", values=f"area_{b}")
            diff = (w[FIXED_HI] - w[FIXED_LO]).mean()
            row[f"area_w075_minus_w025_{b}"] = diff
            row[f"area_w075_minus_w025_{b}_pct"] = 100 * diff / w[FIXED_LO].mean()
        gap.append(row)
    summary = summary.merge(pd.DataFrame(gap), on="dataset", how="left")

    dec = pd.DataFrame(deciles)
    dcols = [c for c in dec if c.startswith(("frac_", "pulled_"))]
    by_dec = dec.groupby(["dataset", "selector", "decile"]).agg(
        n_seeds=("seed", "size"), **{c: (c, "median") for c in dcols}).reset_index()

    os.makedirs(args.out, exist_ok=True)
    summary.to_csv(os.path.join(args.out, "D2_agreement.csv"), index=False)
    by_dec.to_csv(os.path.join(args.out, "D2_agreement_by_decile.csv"), index=False)
    with pd.option_context("display.width", 250, "display.float_format", "{:.3g}".format):
        print(summary[["dataset", "selector", "n_seeds", "frac_distinct_1", "frac_distinct_2",
                       "frac_distinct_3", "early_frac_distinct_1", "pulled_unique_rate",
                       "area_w075_minus_w025_100_pct"]].to_string(index=False))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
