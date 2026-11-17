"""
Diagnostic 3 smoke test, before the oracle sweep.

On one small dataset, runs the ORACLE bandit (reward_source="test_oracle") and
checks from its state history that:

  1. reward at round t equals test_rmse[t-1] - test_rmse[t] (NaN at round 0);
  2. the bonus / arm-value spread at round 25 is within a factor of 10.

Also reruns the v2 MAB (reward_source left at its default) for the same seeds
and requires its curve and state to match the stored v2 pickle exactly, so
adding the flag is shown not to move the v2 results.

Exits non-zero on any failure.

Usage (from the Code/ directory):
    python -m utils.GRFPControls.verification.smoke_oracle_d3 --data concrete_slump --seeds 5
"""

import argparse
import os
import pickle
import sys
import tempfile

import numpy as np

from utils.GRFPControls import _bootstrap  # noqa: F401
from utils.GRFPControls import RunControlsV2 as R

FACTOR = 10.0
ROUND = 25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="concrete_slump")
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    failures = []
    tmp = tempfile.mkdtemp()
    for seed in range(args.seeds):
        label = next(iter(R.ORACLE_SELECTORS))
        path, _ = R.run_one(args.data, label, seed, tmp, suite="oracle")
        with open(path, "rb") as fh:
            r = pickle.load(fh)
        st, curve = r["state"], r["curves"]["test"]
        T = len(st["reward"])

        ### 1. reward is the test-RMSE drop ###
        expect = np.r_[np.nan, curve[:T - 1] - curve[1:T]]
        ok1 = np.allclose(st["reward"], expect, equal_nan=True, rtol=0, atol=1e-12)

        ### 2. bonus vs spread at round 25 ###
        spread = np.ptp(st["arm_values"][ROUND])
        ratio = np.mean(st["bonus"][ROUND]) / spread if spread > 0 else np.inf
        ok2 = 1 / FACTOR <= ratio <= FACTOR

        ### 3. v2 default unchanged ###
        v2_path = R.out_path(R.default_out("v2"), args.data, "MAB (std)", seed)
        with open(v2_path, "rb") as fh:
            v2 = pickle.load(fh)
        re_path, _ = R.run_one(args.data, "MAB (std)", seed, tmp, suite="v2")
        with open(re_path, "rb") as fh:
            re = pickle.load(fh)
        ok3 = (np.array_equal(re["curves"]["test"], v2["curves"]["test"])
               and np.array_equal(re["state"]["arm_pulled"], v2["state"]["arm_pulled"]))

        print(f"seed {seed}: oracle={r['oracle']} file={os.path.basename(path)} | "
              f"reward == test drop: {ok1} | bonus/spread at r{ROUND}: {ratio:.3g} | "
              f"final counts {st['arm_counts'][-1].astype(int).tolist()} | "
              f"v2 MAB reproduced: {ok3}")
        if not r["oracle"] or "ORACLE" not in os.path.basename(path):
            failures.append(f"seed {seed}: output not marked ORACLE")
        if not ok1:
            failures.append(f"seed {seed}: reward is not the test-RMSE drop")
        if not ok2:
            failures.append(f"seed {seed}: bonus/spread {ratio:.3g} outside x{FACTOR:g}")
        if not ok3:
            failures.append(f"seed {seed}: v2 MAB no longer reproduces")

    if failures:
        print("\nFAIL")
        for f in failures:
            print("  " + f)
        sys.exit(1)
    print("\nOK: all checks pass")


if __name__ == "__main__":
    main()
