"""
Regenerate the manuscript's figures, relative to iGS, with the random-arm
control added.

Three figures, matching the parameters in the repo's own plotting code:

  AUC heatmap        GenerateAUCTable.py     Rel. AUC = AUC_method / AUC_iGS,
                                             trapezoidal, diverging palette
                                             centred at 1.0. Integrated over
                                             --budget of the curve (default
                                             0.70) rather than all of it.
  Trace plots        GeneratePlots.py        dRMSE = RMSE_method - RMSE_iGS,
                                             mean over seeds with a 95% band,
                                             one panel per dataset. Full curve.
  Label efficiency   GenerateLabellingComplexity.py
                                             N_rel = labels to reach a given
                                             share of the achievable gain,
                                             divided by iGS's. Milestones 70%
                                             and 80%, baseline iGS.

Every figure is produced for both metrics, because the manuscript's quoted
numbers come from full-pool while its Evaluation Metrics section names held-out
test error.

Usage (from the Code/ directory):
    python -m utils.GRFPControls.ManuscriptFigures --dirs <result dirs> --out <dir>
"""

from utils.GRFPControls import _bootstrap  # noqa: F401

import argparse
import glob
import gzip
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

BASELINE = "iGS"
MILESTONES = (0.70, 0.80)          # GenerateLabellingComplexity.TARGET_PERCENTAGES
CONTROL = "WiGS (Random Arm)"

### Manuscript display order; anything unrecognised is appended ###
ORDER = ['Passive Learning', 'GSx', 'GSy', 'iGS', 'Uncertainty Sampling', 'EGAL',
         'EMCM', 'QBC', 'WiGS (Static w_x=0.25)', 'WiGS (Static w_x=0.5)',
         'WiGS (Static w_x=0.75)', 'WiGS (Time-Decay, Linear)',
         'WiGS (Time-Decay, Exponential)', 'WiGS (MAB-UCB1, c=0.5)',
         'WiGS (MAB-UCB1, c=2.0)', 'WiGS (MAB-UCB1, c=5.0)', 'WiGS (SAC)', CONTROL]

C_INK, C_INK2, C_GRID, C_SURFACE = "#0b0b0b", "#52514e", "#e3e2dd", "#fcfcfb"
HILITE = {CONTROL: "#eb6834", 'WiGS (MAB-UCB1, c=2.0)': "#2a78d6",
          'WiGS (Static w_x=0.75)': "#1baf7a"}

plt.rcParams.update({
    "figure.facecolor": C_SURFACE, "axes.facecolor": C_SURFACE,
    "savefig.facecolor": C_SURFACE, "axes.edgecolor": C_GRID,
    "axes.labelcolor": C_INK2, "text.color": C_INK,
    "xtick.color": C_INK2, "ytick.color": C_INK2,
    "axes.grid": True, "grid.color": C_GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "legend.frameon": False,
})


### Loading ###
def load_curves(dirs, metric):
    """
    (dataset, selector, seed) -> RMSE curve, from either storage layout.

    `exp_a_curves.pkl[.gz]` holds a single metric, written before both were
    recorded; the per-task cluster pickles hold both and are keyed by metric.
    """
    curves = {}
    for d in dirs:
        for pat in ("exp_a_curves.pkl.gz", "exp_a_curves.pkl"):
            p = os.path.join(d, pat)
            if os.path.exists(p):
                op = gzip.open if p.endswith(".gz") else open
                with op(p, "rb") as fh:
                    curves.update(pickle.load(fh))
        for p in glob.glob(os.path.join(d, "**", "*__seed*.pkl"), recursive=True):
            with open(p, "rb") as fh:
                r = pickle.load(fh)
            curves[(r["dataset"], r["selector"], r["seed"])] = \
                np.asarray(r["curves"][metric], dtype=float)
    return curves


def _frame(curves):
    return sorted({k[0] for k in curves}), sorted({k[1] for k in curves})


def _ordered(sels):
    return [s for s in ORDER if s in sels] + [s for s in sels if s not in ORDER]


def _area(curve, frac):
    y = np.asarray(curve, dtype=float)
    n = max(2, int(round(len(y) * frac)))
    y = y[:n]
    if np.isnan(y).any():
        y = pd.Series(y).interpolate().bfill().ffill().values
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz(y, np.arange(len(y))))


### Figure 1: relative-AUC heatmap ###
def fig_auc_heatmap(curves, out_dir, metric, budget=0.70):
    datasets, sels = _frame(curves)
    sels = _ordered(sels)
    M = np.full((len(sels), len(datasets)), np.nan)
    for j, ds in enumerate(datasets):
        base = [_area(v, budget) for (d, s, _), v in curves.items()
                if d == ds and s == BASELINE]
        if not base:
            continue
        b = np.mean(base)
        for i, sel in enumerate(sels):
            a = [_area(v, budget) for (d, s, _), v in curves.items()
                 if d == ds and s == sel]
            if a:
                M[i, j] = np.mean(a) / b

    fig, ax = plt.subplots(figsize=(1.0 + 0.62 * len(datasets), 0.42 * len(sels) + 2.2))
    lo, hi = np.nanmin(M), np.nanmax(M)
    norm = TwoSlopeNorm(vmin=min(lo, 0.98), vcenter=1.0, vmax=max(hi, 1.02))
    im = ax.imshow(M, cmap="RdBu_r", norm=norm, aspect="auto")
    for i in range(len(sels)):
        for j in range(len(datasets)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=6.5,
                        color=C_INK if 0.985 < M[i, j] < 1.015 else "white")
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(sels)))
    ax.set_yticklabels(sels, fontsize=8)
    for lbl in ax.get_yticklabels():
        if lbl.get_text() in HILITE:
            lbl.set_color(HILITE[lbl.get_text()]); lbl.set_fontweight("bold")
    ax.grid(False)
    ax.set_title(f"Relative AUC vs {BASELINE} — {metric}, first {int(budget*100)}% of budget\n"
                 f"blue < 1.000 = lower cumulative error than {BASELINE}",
                 loc="left", fontsize=11, color=C_INK, pad=10)
    fig.colorbar(im, ax=ax, shrink=0.75, pad=0.015, label=f"AUC / AUC$_{{{BASELINE}}}$")
    fig.tight_layout()
    p = os.path.join(out_dir, f"fig_auc_heatmap_vs_iGS_{metric}_{int(budget*100)}pct.png")
    fig.savefig(p, dpi=200); plt.close(fig)
    return p


### Figure 2: trace plots relative to iGS ###
def fig_traces(curves, out_dir, metric, ncol=4):
    datasets, sels = _frame(curves)
    sels = [s for s in _ordered(sels) if s != BASELINE]
    nrow = int(np.ceil(len(datasets) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 3.1 * nrow), squeeze=False)

    for k, ds in enumerate(datasets):
        ax = axes[k // ncol][k % ncol]
        base = [v for (d, s, _), v in curves.items() if d == ds and s == BASELINE]
        if not base:
            ax.axis("off"); continue
        n = min(len(v) for v in base)
        B = np.vstack([v[:n] for v in base]).mean(axis=0)
        ax.axhline(0, color="#d14b4b", lw=1.4, zorder=3)
        for sel in sels:
            cs = [v for (d, s, _), v in curves.items() if d == ds and s == sel]
            if not cs:
                continue
            m = min(n, min(len(v) for v in cs))
            A = np.vstack([v[:m] for v in cs])
            delta = A.mean(axis=0) - B[:m]
            hi = sel in HILITE
            ax.plot(delta, lw=2.0 if hi else 0.9,
                    color=HILITE.get(sel, "#9a9992"), alpha=1.0 if hi else 0.55,
                    zorder=4 if hi else 2, label=sel if hi else None)
            if hi:
                se = A.std(axis=0, ddof=1) / np.sqrt(A.shape[0])
                ax.fill_between(np.arange(m), delta - 1.96 * se, delta + 1.96 * se,
                                color=HILITE[sel], alpha=0.15, lw=0, zorder=3)
        ax.set_title(ds, loc="left", fontsize=10, color=C_INK)
        ax.set_xlabel("Round"); ax.set_ylabel(f"$\\Delta$ {metric} vs {BASELINE}")
        if k == 0:
            ax.legend(fontsize=7, loc="upper right")
    for k in range(len(datasets), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle(f"Trace relative to {BASELINE} — {metric}. Below 0 = lower error than {BASELINE}. "
                 f"Grey = other strategies.", x=0.01, ha="left", fontsize=12, color=C_INK)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    p = os.path.join(out_dir, f"fig_traces_vs_iGS_{metric}.png")
    fig.savefig(p, dpi=170); plt.close(fig)
    return p


### Figure 3: label efficiency ###
def label_efficiency(curves, metric):
    """
    N_rel per (dataset, selector, milestone), following
    GenerateLabellingComplexity: the target is a share of the gain between the
    first-round error and the best error any strategy reaches on that dataset,
    and N_rel divides a strategy's rounds-to-target by iGS's.
    """
    datasets, sels = _frame(curves)
    rows = []
    for ds in datasets:
        per = {}
        for sel in sels:
            cs = [v for (d, s, _), v in curves.items() if d == ds and s == sel]
            if cs:
                n = min(len(v) for v in cs)
                per[sel] = np.vstack([v[:n] for v in cs]).mean(axis=0)
        if BASELINE not in per:
            continue
        start = np.mean([c[0] for c in per.values()])
        floor = min(c.min() for c in per.values())
        for frac in MILESTONES:
            target = start - frac * (start - floor)

            def reach(c):
                idx = np.flatnonzero(c <= target)
                return float(idx[0]) if idx.size else float(len(c))

            nb = max(1.0, reach(per[BASELINE]))
            for sel, c in per.items():
                rows.append({"dataset": ds, "selector": sel,
                             "milestone": f"{int(frac*100)}%",
                             "N_rel": max(1.0, reach(c)) / nb})
    return pd.DataFrame(rows)


def fig_label_efficiency(curves, out_dir, metric):
    df = label_efficiency(curves, metric)
    if df.empty:
        return None
    sels = [s for s in _ordered(sorted(df.selector.unique()))]
    fig, ax = plt.subplots(figsize=(8.6, 0.42 * len(sels) + 2.4))
    offs = {m: o for m, o in zip(sorted(df.milestone.unique()), (-0.18, +0.18))}
    cols = {sorted(df.milestone.unique())[0]: "#2a78d6",
            sorted(df.milestone.unique())[-1]: "#1baf7a"}
    ax.axvline(1.0, color="#d14b4b", lw=1.4, ls="--", zorder=1)
    for i, sel in enumerate(sels):
        for ms, off in offs.items():
            v = df[(df.selector == sel) & (df.milestone == ms)].N_rel.values
            if not len(v):
                continue
            bp = ax.boxplot(v, positions=[i + off], vert=False, widths=0.3,
                            patch_artist=True, showfliers=False, zorder=2)
            for b in bp["boxes"]:
                b.set(facecolor=cols[ms], alpha=0.55, lw=0)
            for part in ("whiskers", "caps", "medians"):
                for a in bp[part]:
                    a.set(color=C_INK2, lw=1.0)
    ax.set_yticks(range(len(sels)))
    ax.set_yticklabels(sels, fontsize=8)
    for lbl in ax.get_yticklabels():
        if lbl.get_text() in HILITE:
            lbl.set_color(HILITE[lbl.get_text()]); lbl.set_fontweight("bold")
    ax.set_ylim(-0.7, len(sels) - 0.3)
    ax.set_xlabel(f"Labels needed relative to {BASELINE} ($N_{{rel}}$) — {metric}")
    ax.set_title(f"Label efficiency vs {BASELINE}; < 1.0 = fewer labels to the milestone",
                 loc="left", fontsize=11, color=C_INK, pad=10)
    handles = [plt.Line2D([], [], marker="s", ls="", ms=9, color=cols[m],
                          alpha=0.65, label=f"{m} of achievable gain")
               for m in sorted(offs)]
    ax.legend(handles=handles, fontsize=8, loc="lower right")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    p = os.path.join(out_dir, f"fig_label_efficiency_vs_iGS_{metric}.png")
    fig.savefig(p, dpi=200); plt.close(fig)
    df.to_csv(os.path.join(out_dir, f"label_efficiency_{metric}.csv"), index=False)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--metrics", nargs="+", default=["full_pool", "test"])
    ap.add_argument("--budget", type=float, default=0.70)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for metric in args.metrics:
        curves = load_curves(args.dirs, metric)
        if not curves:
            print(f"[{metric}] no curves found"); continue
        ds, sel = _frame(curves)
        print(f"\n[{metric}] {len(ds)} datasets x {len(sel)} selectors "
              f"x {len({k[2] for k in curves})} seeds")
        for p in (fig_auc_heatmap(curves, args.out, metric, args.budget),
                  fig_traces(curves, args.out, metric),
                  fig_label_efficiency(curves, args.out, metric)):
            if p:
                print("  wrote", os.path.basename(p))


if __name__ == "__main__":
    main()
