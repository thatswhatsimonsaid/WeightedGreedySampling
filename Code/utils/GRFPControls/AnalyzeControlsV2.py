"""
Step 5 summary: score the v2 control rerun and write Results/controls_v2/SUMMARY_v2.md.

Reads the v2 per-run pickles written by RunControlsV2.py and, for the selectors
that ignore the reward (Passive and the five fixed weights), the September
curves for the same datasets and seeds. The September MAB and random arm are
included as reference rows.

Areas are trapezoidal under the test-RMSE curve, over the full budget and over
the first 25% of rounds, exactly as Analyze.py computes them. "vs passive" is
100 * mean(area - area_passive) / mean(area_passive), paired on seed, per
dataset; the cross-dataset figure is the mean of those. Paired comparisons use
the same percentile bootstrap (10,000 resamples, seed 17) as SUMMARY.md.

Usage (from the Code/ directory):
    python -m utils.GRFPControls.AnalyzeControlsV2
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
from utils.GRFPControls.RunControlsV2 import DATASETS, N_SEEDS, SELECTORS, default_out

NEW_MAB, NEW_RANDOM = "MAB (std)", "RandomArm (v2)"
FIXED = ["Fixed w=0.00", "Fixed w=0.25", "Fixed w=0.50", "Fixed w=0.75", "Fixed w=1.00"]
REUSED = ["Passive"] + FIXED
SEPT_REF = {"MAB (c=2.0)": "MAB (c=2.0), September", "RandomArm": "RandomArm, September"}
ARMS = [0.25, 0.5, 0.75]
STATE_ROUND = 25
SEPT_SOURCES = ["results_grfp_controls", os.path.join("results_grfp_controls", "benchmarks")]


def load(root, raw_dir):
    curves, states = {}, {}
    for src in SEPT_SOURCES:
        with gzip.open(os.path.join(root, src, "exp_a_curves.pkl.gz"), "rb") as fh:
            for (d, sel, s), c in pickle.load(fh).items():
                if d in DATASETS and (sel in REUSED or sel in SEPT_REF):
                    curves[(d, SEPT_REF.get(sel, sel), s)] = np.asarray(c, dtype=float)
    for p in glob.glob(os.path.join(raw_dir, "*", "*.pkl")):
        with open(p, "rb") as fh:
            r = pickle.load(fh)
        curves[(r["dataset"], r["selector"], r["seed"])] = r["curves"]["test"]
        states[(r["dataset"], r["selector"], r["seed"])] = r["state"]
    return curves, states


def areas(curves):
    rows = []
    for d in DATASETS:
        lens = [len(c) for (dd, _, _), c in curves.items() if dd == d]
        n_early = max(2, int(round(min(lens) * EARLY_FRAC)))
        for (dd, sel, s), c in curves.items():
            if dd == d:
                rows.append({"dataset": d, "selector": sel, "seed": s,
                             "area_100": LearningCurveArea(c),
                             "area_25": LearningCurveArea(c[:n_early])})
    return pd.DataFrame(rows)


def paired(wide, a, b):
    """mean(a - b) with CI, and as % of mean(b); lower area is better."""
    p = wide[[a, b]].dropna()
    m, lo, hi = paired_bootstrap_ci((p[a] - p[b]).values, n_boot=10000, seed=17)
    verdict = "better" if hi < 0 else "worse" if lo > 0 else "tie"
    base = p[b].mean()
    return {"n": len(p), "diff": m, "lo": lo, "hi": hi, "pct": 100 * m / base,
            "pct_lo": 100 * lo / base, "pct_hi": 100 * hi / base, "verdict": verdict}


def fmt_pct(x):
    return f"{x:+.2f}%"


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(os.getcwd()))
    ap.add_argument("--raw", default=None)
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()
    raw_dir = args.raw or default_out()
    out_dir = os.path.dirname(raw_dir)

    curves, states = load(args.root, raw_dir)
    A = areas(curves)

    ### Completeness ###
    grid = A.pivot_table(index="dataset", columns="selector", values="seed", aggfunc="nunique")
    short = grid[(grid[list(SELECTORS)] < N_SEEDS).any(axis=1)] if set(SELECTORS) <= set(grid) else grid
    if len(short) or not set(SELECTORS) <= set(grid):
        print(grid.to_string())
        if not args.allow_partial:
            raise SystemExit("v2 runs incomplete; rerun missing tasks or pass --allow-partial")

    # With --allow-partial, score only datasets that have v2 runs for both selectors.
    present = [d for d in DATASETS
               if all((d, sel) in set(zip(A.dataset, A.selector)) for sel in SELECTORS)]
    A = A[A.dataset.isin(present)]
    states = {k: v for k, v in states.items() if k[0] in present}
    n_ds = len(present)

    ### The random arm never reads its reward, so v2 must equal September ###
    same = [np.array_equal(curves[(d, NEW_RANDOM, s)], curves[(d, SEPT_REF["RandomArm"], s)])
            for (d, sel, s) in curves if sel == NEW_RANDOM and (d, SEPT_REF["RandomArm"], s) in curves]
    print(f"RandomArm (v2) identical to September on {sum(same)}/{len(same)} runs")

    sections = []
    selectors = [NEW_MAB, NEW_RANDOM] + FIXED + list(SEPT_REF.values())

    ### 1. Mean change in area vs passive ###
    per_ds = []
    for budget in ("25", "100"):
        for d, g in A.groupby("dataset"):
            wide = g.pivot_table(index="seed", columns="selector", values=f"area_{budget}")
            for sel in selectors:
                if sel in wide:
                    r = paired(wide, sel, "Passive")
                    per_ds.append({"budget": budget, "dataset": d, "selector": sel, **r})
    per_ds = pd.DataFrame(per_ds)
    rows = []
    for sel in selectors:
        g = per_ds[per_ds.selector == sel]
        rows.append([sel] + [fmt_pct(g[g.budget == b]["pct"].mean()) for b in ("25", "100")]
                    + [f"{(g[g.budget == b].verdict == 'better').sum()} / "
                       f"{(g[g.budget == b].verdict == 'worse').sum()}" for b in ("25", "100")])
    sections.append(f"## Change in area vs passive, mean over {n_ds} datasets\n\n" + md_table(
        ["Selector", "First 25%", "Full budget",
         "better / worse than passive (25%)", "better / worse than passive (full)"], rows))

    ### 2 and 3. Paired comparisons per dataset ###
    def comparison(title, ref_of):
        detail, counts = [], {}
        for budget in ("25", "100"):
            for d, g in A.groupby("dataset"):
                wide = g.pivot_table(index="seed", columns="selector", values=f"area_{budget}")
                ref = ref_of(wide)
                r = paired(wide, NEW_MAB, ref)
                detail.append((budget, d, ref, r))
                counts.setdefault(budget, {"better": 0, "worse": 0, "tie": 0})[r["verdict"]] += 1
        txt = [f"## {title}\n"]
        txt.append(md_table(["", "First 25%", "Full budget"],
                            [[k] + [counts[b][k] for b in ("25", "100")]
                             for k in ("better", "worse", "tie")]))
        rows = []
        for d in present:
            row = [d]
            for b in ("25", "100"):
                budget, _, ref, r = next(x for x in detail if x[0] == b and x[1] == d)
                row += ([ref] if ref_of is not _random_ref else []) + [
                    f"{fmt_pct(r['pct'])} [{fmt_pct(r['pct_lo'])}, {fmt_pct(r['pct_hi'])}]",
                    r["verdict"]]
            rows.append(row)
        if ref_of is _random_ref:
            head = ["Dataset", "First 25%", "", "Full budget", ""]
        else:
            head = ["Dataset", "Best fixed (25%)", "First 25%", "", "Best fixed (full)", "Full budget", ""]
        txt.append("\n" + md_table(head, rows))
        return "\n".join(txt)

    sections.append(comparison(f"{NEW_MAB} vs {NEW_RANDOM}, paired per dataset, 95% CI over seeds "
                               "(% of reference area; negative = MAB lower)", _random_ref))
    sections.append(comparison(f"{NEW_MAB} vs best fixed weight, paired per dataset, 95% CI over seeds "
                               "(best = lowest mean area among the five at that budget; "
                               "% of reference area; negative = MAB lower)", _best_fixed_ref))

    ### 4. Selector state ###
    rows = []
    for d in present:
        st = [v for (dd, sel, _), v in states.items() if dd == d and sel == NEW_MAB]
        final = np.array([s["arm_counts"][-1] for s in st])
        ratio = []
        for s in st:
            if len(s["bonus"]) > STATE_ROUND:
                spread = np.ptp(s["arm_values"][STATE_ROUND])
                ratio.append(np.mean(s["bonus"][STATE_ROUND]) / spread if spread > 0 else np.inf)
        med = np.median(final, axis=0)
        rows.append([d, len(st)] + [f"{m:g}" for m in med] + [f"{np.median(ratio):.3g}"])
    sections.append(f"## {NEW_MAB}: selector state, median over seeds\n\n" + md_table(
        ["Dataset", "Seeds"] + [f"Final pulls, w={w}" for w in ARMS]
        + [f"Bonus / arm-value spread at round {STATE_ROUND}"], rows))

    md = "# Controls v2\n\n" + "\n\n".join(sections) + "\n"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "SUMMARY_v2.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    A.to_csv(os.path.join(out_dir, "v2_areas.csv"), index=False)
    per_ds.to_csv(os.path.join(out_dir, "v2_vs_passive.csv"), index=False)
    print(md)
    print(f"-> {os.path.join(out_dir, 'SUMMARY_v2.md')}")


def _random_ref(wide):
    return NEW_RANDOM


def _best_fixed_ref(wide):
    return min(FIXED, key=lambda f: wide[f].mean())


if __name__ == "__main__":
    main()
