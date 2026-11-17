"""
Diagnostic 3 summary: the ORACLE bandit (reward = drop in held-out test RMSE).

Reads the ORACLE runs (Results/diagnostics_v3/ORACLE_raw/, RunControlsV2
--suite oracle) next to the v2 runs and the reused September Passive and
fixed-weight curves, scored exactly as in SUMMARY_v2.md (AnalyzeControlsV2's
loader, areas and paired bootstrap). Writes
Results/diagnostics_v3/D3_ORACLE_SUMMARY.md.

The ORACLE bandit uses the test set to choose its arm. It is a diagnostic
bound, not a method, and is labelled ORACLE everywhere it appears.

Usage (from the Code/ directory):
    python -m utils.GRFPControls.AnalyzeD3Oracle
"""

from utils.GRFPControls import _bootstrap  # noqa: F401

import argparse
import glob
import os
import pickle

import numpy as np
import pandas as pd

from utils.GRFPControls.AnalyzeControlsV2 import (ARMS, FIXED, STATE_ROUND, areas, fmt_pct, load,
                                                  md_table, paired)
from utils.GRFPControls.RunControlsV2 import DATASETS, N_SEEDS, ORACLE_SELECTORS, default_out

ORACLE = next(iter(ORACLE_SELECTORS))
V2_MAB, RANDOM, PASSIVE = "MAB (std)", "RandomArm (v2)", "Passive"
BUDGETS = ("25", "100")


def load_oracle(raw):
    curves, states = {}, {}
    for p in glob.glob(os.path.join(raw, "*", "*ORACLE*.pkl")):
        with open(p, "rb") as fh:
            r = pickle.load(fh)
        assert r.get("oracle") is True, p
        curves[(r["dataset"], r["selector"], r["seed"])] = r["curves"]["test"]
        states[(r["dataset"], r["selector"], r["seed"])] = r["state"]
    return curves, states


def best_fixed(wide):
    return min(FIXED, key=lambda f: wide[f].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(os.getcwd()))
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()
    oracle_raw = default_out("oracle")
    out_dir = os.path.dirname(oracle_raw)

    curves, _ = load(args.root, default_out("v2"))
    o_curves, o_states = load_oracle(oracle_raw)
    n_per_ds = pd.Series([k[0] for k in o_curves]).value_counts()
    short = [d for d in DATASETS if n_per_ds.get(d, 0) < N_SEEDS]
    if short:
        print("incomplete:", {d: int(n_per_ds.get(d, 0)) for d in short})
        if not args.allow_partial:
            raise SystemExit("ORACLE runs incomplete; pass --allow-partial to score what exists")
    present = [d for d in DATASETS if n_per_ds.get(d, 0) > 0]
    curves.update(o_curves)
    A = areas({k: v for k, v in curves.items() if k[0] in present})
    sections = []

    ### 1. Mean change in area vs passive ###
    rows = {ORACLE: [], V2_MAB: [], RANDOM: [], "Best fixed weight": []}
    for b in BUDGETS:
        acc = {k: [] for k in rows}
        for d, g in A.groupby("dataset"):
            wide = g.pivot_table(index="seed", columns="selector", values=f"area_{b}")
            for sel in (ORACLE, V2_MAB, RANDOM):
                acc[sel].append(paired(wide, sel, PASSIVE)["pct"])
            acc["Best fixed weight"].append(paired(wide, best_fixed(wide), PASSIVE)["pct"])
        for k in rows:
            rows[k].append(fmt_pct(np.mean(acc[k])))
    sections.append(f"## Change in area vs passive, mean over {len(present)} datasets\n\n" + md_table(
        ["Selector", "First 25%", "Full budget"], [[k] + v for k, v in rows.items()]))

    ### 2. Paired counts ###
    count_rows, detail = [], []
    for name, ref_of in ((f"{ORACLE} vs {RANDOM}", lambda w: RANDOM),
                         (f"{ORACLE} vs best fixed weight", best_fixed)):
        counts = {b: {"better": 0, "worse": 0, "tie": 0} for b in BUDGETS}
        for b in BUDGETS:
            for d, g in A.groupby("dataset"):
                wide = g.pivot_table(index="seed", columns="selector", values=f"area_{b}")
                ref = ref_of(wide)
                r = paired(wide, ORACLE, ref)
                counts[b][r["verdict"]] += 1
                detail.append({"comparison": name, "budget": b, "dataset": d, "reference": ref, **r})
        for v in ("better", "worse", "tie"):
            count_rows.append([name, v] + [counts[b][v] for b in BUDGETS])
    sections.append("## Paired counts per dataset, 95% CI over seeds (better = lower area)\n\n"
                    + md_table(["Comparison", "", "First 25%", "Full budget"], count_rows))

    ### 3. Selector state ###
    rows = []
    for d in present:
        st = [v for (dd, _, _), v in o_states.items() if dd == d]
        med = np.median(np.array([s["arm_counts"][-1] for s in st]), axis=0)
        g = A[A.dataset == d]
        bf = [best_fixed(g.pivot_table(index="seed", columns="selector", values=f"area_{b}"))
              for b in BUDGETS]
        rows.append([d, len(st)] + [f"{m:g}" for m in med]
                    + [f"w={ARMS[int(np.argmax(med))]}"] + bf)
    sections.append(f"## {ORACLE}: selector state, median over seeds\n\n" + md_table(
        ["Dataset", "Seeds"] + [f"Final pulls, w={w}" for w in ARMS]
        + ["Most-pulled arm", "Best fixed (25%)", "Best fixed (full)"], rows))

    md = "# Diagnostic 3: ORACLE-reward bandit\n\n" + "\n\n".join(sections) + "\n"
    with open(os.path.join(out_dir, "D3_ORACLE_SUMMARY.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    pd.DataFrame(detail).to_csv(os.path.join(out_dir, "D3_ORACLE_paired.csv"), index=False)
    print(md)
    print(f"-> {os.path.join(out_dir, 'D3_ORACLE_SUMMARY.md')}")


if __name__ == "__main__":
    main()
