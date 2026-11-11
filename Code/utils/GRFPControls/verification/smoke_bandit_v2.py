"""
Step 4 smoke test for the rescaled bandit, before any cluster run.

Runs WiGS_MAB_Selector with reward_scaling="std" and cv_fallback_to_test=False
on one small dataset, under the September controls' protocol, and checks from
SelectorStateHistory that:

  1. after round 25 the exploration bonus and the spread of arm_values are
     within a factor of 10 of each other;
  2. cv_was_nan is True exactly on the rounds whose labelled set has fewer than
     10 points, and reward is NaN exactly on those rounds, round 0, and the
     first round after them (whose previous RMSE is NaN);
  3. max_count_gap (as in scripts/diagnose_bandit.py) exceeds 1 on at least
     some seeds.

Exits non-zero if any check fails.

Usage (from the Code/ directory):
    python -m utils.GRFPControls.verification.smoke_bandit_v2 --data concrete_slump --seeds 5
"""

import argparse
import sys
import time

import numpy as np

from utils.GRFPControls import _bootstrap  # noqa: F401
from utils.GRFPControls.ControlLoop import RunControlIteration

CV_MIN_N = 10
FACTOR = 10.0
FROM_ROUND = 25


def max_count_gap(arms, n_arms=3):
    counts = np.zeros(n_arms)
    gaps = []
    for t, a in enumerate(arms):
        counts[a] += 1
        if t >= n_arms:
            gaps.append(counts.max() - counts.min())
    return max(gaps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="concrete_slump")
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    failures = []
    gaps = []
    for seed in range(args.seeds):
        t0 = time.time()
        out = RunControlIteration(DataFileInput=args.data, Seed=seed,
                                  SelectorType="WiGS_MAB_Selector",
                                  selector_params={"mab_c": 2.0, "reward_scaling": "std"},
                                  Protocol="repo", Standardize=False, Metric="test",
                                  cv_fallback_to_test=False)
        sh = out["SelectorStateHistory"]
        n0 = out["n_initial_train"]
        T = len(sh)

        ### 1. bonus vs arm-value spread after round 25 ###
        ratios = []
        for s in sh[FROM_ROUND:]:
            spread = np.ptp(s["arm_values"])
            ratios.append(np.mean(s["bonus"]) / spread if spread > 0 else np.inf)
        ratios = np.array(ratios)
        within = (ratios >= 1 / FACTOR) & (ratios <= FACTOR)
        med = float(np.median(ratios))
        ok1 = 1 / FACTOR <= med <= FACTOR

        ### 2. NaN bookkeeping ###
        cv_nan = np.array([s["cv_was_nan"] for s in sh])
        expect_cv_nan = (n0 + np.arange(T)) < CV_MIN_N
        rew_nan = np.array([np.isnan(s["reward"]) for s in sh])
        prev_nan = np.r_[True, cv_nan[:-1]]           # round 0 has no previous RMSE
        expect_rew_nan = cv_nan | prev_nan
        ok2 = bool((cv_nan == expect_cv_nan).all() and (rew_nan == expect_rew_nan).all())

        ### 3. round-robin? ###
        gap = max_count_gap([s["arm_pulled"] for s in sh])
        gaps.append(gap)

        print(f"seed {seed}: n0={n0} rounds={T} | "
              f"bonus/spread after r{FROM_ROUND}: median {med:.3g}, "
              f"range [{ratios.min():.3g}, {ratios.max():.3g}], "
              f"{within.mean():.0%} of rounds within x{FACTOR:g} | "
              f"cv_nan rounds {int(cv_nan.sum())} (expect {int(expect_cv_nan.sum())}), "
              f"reward_nan rounds {int(rew_nan.sum())} (expect {int(expect_rew_nan.sum())}) | "
              f"max_count_gap {gap:g} | final counts {sh[-1]['arm_counts'].astype(int).tolist()} | "
              f"{time.time() - t0:.1f}s")
        if not ok1:
            failures.append(f"seed {seed}: median bonus/spread {med:.3g} outside x{FACTOR:g}")
        if not ok2:
            failures.append(f"seed {seed}: NaN bookkeeping does not match the labelled-set size")

    if not any(g > 1 for g in gaps):
        failures.append("max_count_gap == 1 on every seed (still round-robin)")

    if failures:
        print("\nFAIL")
        for f in failures:
            print("  " + f)
        sys.exit(1)
    print("\nOK: all three checks pass")


if __name__ == "__main__":
    main()
