"""
Win / loss / tie record of every selector against a reference.

A win or loss requires the paired 95% bootstrap CI on the per-seed difference
to exclude zero; anything else is a tie. The tie column is the point of this
table: a headline Relative AUC can rest on datasets that are individually
indistinguishable, and that only shows up here.

Usage (from the Code/ directory):
    python -m utils.GRFPControls.WinLossTie --dirs ../results_grfp_controls/full_suite/full_pool
"""

from utils.GRFPControls import _bootstrap  # noqa: F401

import argparse
import os

from utils.GRFPControls.RelativeToIGS import table, results_root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="*", default=None)
    ap.add_argument("--ref", default="iGS")
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()

    if args.dirs:
        for d in args.dirs:
            table([d], os.path.basename(d.rstrip("/\\")), ref=args.ref,
                  n_boot=args.n_boot)
        return

    R = results_root()
    for sub, lab in (("full_suite/full_pool", "FULL SUITE + FULL-POOL"),
                     ("full_suite/test", "FULL SUITE + TEST RMSE"),
                     ("paper_metric_fullpool", "PAPER PROTOCOL + FULL-POOL"),
                     ("paper_protocol", "PAPER PROTOCOL + TEST RMSE")):
        p = os.path.join(R, sub)
        if os.path.exists(os.path.join(p, "exp_a_runs.csv")):
            table([p], lab, ref=args.ref, n_boot=args.n_boot)


if __name__ == "__main__":
    main()
