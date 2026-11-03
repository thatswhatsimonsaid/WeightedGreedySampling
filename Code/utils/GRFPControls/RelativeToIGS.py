"""
Selector ranking as Relative AUC against a reference selector.

Rel. AUC = AUC_method / AUC_reference, so < 1.000 beats the reference. iGS is
the default because it is the manuscript's own normaliser; --ref takes any
selector present in the results, e.g. the bandit.

Usage (from the Code/ directory):
    python -m utils.GRFPControls.RelativeToIGS --dirs ../results_grfp_controls/full_suite/full_pool
    python -m utils.GRFPControls.RelativeToIGS --dirs <dir> --ref "WiGS (MAB-UCB1, c=2.0)"

With no --dirs it falls back to whichever of the local sweep directories exist,
so the earlier 13/14-dataset runs can still be reproduced.
"""

from utils.GRFPControls import _bootstrap  # noqa: F401

import argparse
import os

import numpy as np
import pandas as pd

from utils.GRFPControls.Analyze import paired_bootstrap_ci

### Manuscript display order; unknown names are appended ###
ORDER = ['Passive Learning', 'GSx', 'GSy', 'iGS', 'Uncertainty Sampling', 'EGAL',
         'EMCM', 'QBC', 'WiGS (Static w_x=0.25)', 'WiGS (Static w_x=0.5)',
         'WiGS (Static w_x=0.75)', 'WiGS (Time-Decay, Linear)',
         'WiGS (Time-Decay, Exponential)', 'WiGS (MAB-UCB1, c=0.5)',
         'WiGS (MAB-UCB1, c=2.0)', 'WiGS (MAB-UCB1, c=5.0)', 'WiGS (SAC)',
         'WiGS (Random Arm)',
         # names used by the earlier local sweeps
         'Fixed w=0.00', 'Fixed w=0.25', 'Fixed w=0.50', 'Fixed w=0.75',
         'Fixed w=1.00', 'MAB (c=2.0)', 'RandomArm', 'Passive']


def results_root():
    """<repo>/results_grfp_controls, derived from the working directory."""
    return os.path.join(os.path.dirname(os.getcwd()), "results_grfp_controls")


def load(paths):
    frames = []
    for p in paths:
        f = p if p.endswith(".csv") else os.path.join(p, "exp_a_runs.csv")
        if os.path.exists(f):
            frames.append(pd.read_csv(f))
        else:
            print(f"  [skip] no exp_a_runs.csv at {p}")
    if not frames:
        raise SystemExit("no results found; pass --dirs")
    return pd.concat(frames, ignore_index=True)


def table(paths, label, ref="iGS", subset=None, n_boot=10000):
    r = load(paths)
    if subset:
        r = r[r.dataset.isin(subset)]
    dss = sorted(r.dataset.unique())
    sels = sorted(r.selector.unique())
    sels = [s for s in ORDER if s in sels] + [s for s in sels if s not in ORDER]
    if ref not in sels:
        raise SystemExit(f"reference {ref!r} not in results; have: {sels}")

    rows = []
    for sel in sels:
        rels, better, worse, ties = [], 0, 0, 0
        for ds in dss:
            w = r[r.dataset == ds].pivot_table(index="seed", columns="selector", values="area")
            if sel not in w.columns or ref not in w.columns:
                continue
            pr = w[[sel, ref]].dropna()
            rels.append(pr[sel].mean() / pr[ref].mean())
            if sel == ref:
                continue
            _, lo, hi = paired_bootstrap_ci((pr[sel] - pr[ref]).values,
                                            n_boot=n_boot, seed=17)
            if hi < 0:
                better += 1
            elif lo > 0:
                worse += 1
            else:
                ties += 1
        if rels:
            # Named Win/Loss/Tie, not W/L/T: `row.T` resolves to the DataFrame
            # transpose rather than a column named "T".
            rows.append({"selector": sel, "rel": float(np.mean(rels)),
                         "Win": better, "Loss": worse, "Tie": ties})

    d = pd.DataFrame(rows).sort_values("rel")
    print(f"\n=== {label} ({len(dss)} datasets x 100 seeds) ===")
    print(f"  Rel. AUC vs {ref}; < 1.000 = better. "
          f"beats/loses/ties = datasets where the paired 95% CI excludes 0")
    print(f"  {'selector':<32}{'Rel.AUC':>9}{'vs ref':>9}   {'beats/loses/ties':>18}")
    print("  " + "-" * 70)
    for _, x in d.iterrows():
        if x.selector == ref:
            print(f"  {x.selector:<32}{x.rel:>9.4f}{'—':>9}   {'(reference)':>18}")
        else:
            wlt = f"{int(x.Win)} / {int(x.Loss)} / {int(x.Tie)}"
            print(f"  {x.selector:<32}{x.rel:>9.4f}{100*(x.rel-1):>+8.2f}%   {wlt:>18}")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="*", default=None,
                    help="Result directories (each holding exp_a_runs.csv).")
    ap.add_argument("--ref", default="iGS")
    ap.add_argument("--label", default=None)
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()

    if args.dirs:
        table(args.dirs, args.label or " + ".join(os.path.basename(d.rstrip("/\\"))
                                                  for d in args.dirs),
              ref=args.ref, n_boot=args.n_boot)
        return

    ### No --dirs: fall back to whichever local sweeps are present ###
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
