"""
Analysis and figures for the GRFP control experiments.

Produces:
  exp_a_summary.csv       per dataset x selector: mean/SE area and final error,
                          plus the paired bandit-minus-random-arm difference
                          with a bootstrap 95% CI
  exp_a_reward_gap.csv    between-arm reward gap vs within-arm reward noise
  fig_learning_curves_*.png
  fig_paired_difference.png
  fig_oracle_path.png

Colour use follows the reference categorical palette in its documented slot
order, restricted to slots 1-3 (blue / orange / aqua), the subset certified for
all-pairs CVD separation. Every other series is deliberately neutral: the
comparison being made is bandit vs random-arm, and the remaining selectors are
context, not competing identities.

Usage (from the Code/ directory):
    python -m utils.GRFPControls.Analyze
"""

from utils.GRFPControls import _bootstrap  # noqa: F401

import argparse
import ast
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

### Palette (reference instance, slots 1-3 + neutrals) ###
C_BANDIT = "#2a78d6"   # slot 1, blue
C_RANDOM = "#eb6834"   # slot 2, orange
C_IGS = "#1baf7a"      # slot 3, aqua
C_MUTED = "#9a9992"    # fixed-weight family / passive
C_INK = "#0b0b0b"
C_INK2 = "#52514e"
C_GRID = "#e3e2dd"
C_SURFACE = "#fcfcfb"

BANDIT = "MAB (c=2.0)"
RANDOM = "RandomArm"
FIXED = ["Fixed w=0.00", "Fixed w=0.25", "Fixed w=0.50", "Fixed w=0.75", "Fixed w=1.00"]

plt.rcParams.update({
    "figure.facecolor": C_SURFACE, "axes.facecolor": C_SURFACE,
    "savefig.facecolor": C_SURFACE,
    "axes.edgecolor": C_GRID, "axes.labelcolor": C_INK2,
    "text.color": C_INK, "xtick.color": C_INK2, "ytick.color": C_INK2,
    "axes.grid": True, "grid.color": C_GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "legend.frameon": False,
})


### Paired bootstrap ###
def paired_bootstrap_ci(diffs, n_boot=10000, seed=0, alpha=0.05):
    """Percentile bootstrap CI for the mean of paired differences."""
    d = np.asarray(diffs, dtype=float)
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(d.mean()), float(lo), float(hi)


### Summary table ###
def build_summary(runs, out_dir, n_boot=10000):
    rows = []
    for dataset, g in runs.groupby("dataset"):
        wide_area = g.pivot_table(index="seed", columns="selector", values="area")
        wide_final = g.pivot_table(index="seed", columns="selector", values="final_rmse")
        for selector in sorted(g["selector"].unique()):
            a = wide_area[selector].dropna()
            f = wide_final[selector].dropna()
            row = {
                "dataset": dataset, "selector": selector, "n_seeds": int(len(a)),
                "area_mean": a.mean(), "area_se": a.std(ddof=1) / np.sqrt(len(a)),
                "final_mean": f.mean(), "final_se": f.std(ddof=1) / np.sqrt(len(f)),
            }
            # Paired difference against the random-arm control #
            if RANDOM in wide_area.columns and selector != RANDOM:
                paired = wide_area[[selector, RANDOM]].dropna()
                d = (paired[selector] - paired[RANDOM]).values
                m, lo, hi = paired_bootstrap_ci(d, n_boot=n_boot, seed=17)
                row.update({
                    "diff_vs_randomarm_mean": m,
                    "diff_ci_lo": lo, "diff_ci_hi": hi,
                    "diff_excludes_zero": bool(lo > 0 or hi < 0),
                    # area is RMSE-based: lower is better #
                    "favors": ("selector" if hi < 0 else
                               "RandomArm" if lo > 0 else "neither"),
                })
            rows.append(row)
    df = pd.DataFrame(rows)
    path = os.path.join(out_dir, "exp_a_summary.csv")
    df.to_csv(path, index=False)
    return df, path


### Reward gap vs reward noise ###
def build_reward_gap(arms, out_dir):
    """
    Between-arm reward gap over within-arm reward noise.

    Computed per (dataset, selector). The RandomArm rows are the cleaner
    estimate: under a uniform policy each arm's reward sample is unconfounded
    by the adaptive allocation that UCB1 imposes.
    """
    rows = []
    for (dataset, selector), g in arms.groupby(["dataset", "selector"]):
        g = g[np.isfinite(g["reward"])]
        if g.empty:
            continue
        per_arm = g.groupby("w_x")["reward"].agg(["count", "mean", "std"])
        gap = float(per_arm["mean"].max() - per_arm["mean"].min())
        # pooled within-arm SD #
        noise = float(np.sqrt(np.average(per_arm["std"].fillna(0) ** 2,
                                         weights=per_arm["count"])))
        for w, r in per_arm.iterrows():
            rows.append({"dataset": dataset, "selector": selector, "arm_w": w,
                         "n_pulls": int(r["count"]), "reward_mean": r["mean"],
                         "reward_sd": r["std"],
                         "between_arm_gap": gap, "within_arm_noise": noise,
                         "gap_over_noise": gap / noise if noise > 0 else np.nan})
    df = pd.DataFrame(rows)
    path = os.path.join(out_dir, "exp_a_reward_gap.csv")
    df.to_csv(path, index=False)
    return df, path


### Figure 1: learning curves ###
def fig_learning_curves(curves, dataset, out_dir):
    def mean_curve(sel):
        cs = [v for (d, s, _), v in curves.items() if d == dataset and s == sel]
        if not cs:
            return None, None
        n = min(len(c) for c in cs)
        M = np.vstack([c[:n] for c in cs])
        return M.mean(axis=0), M.std(axis=0, ddof=1) / np.sqrt(M.shape[0])

    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    for i, sel in enumerate(FIXED):
        m, _ = mean_curve(sel)
        if m is not None:
            ax.plot(m, color=C_MUTED, lw=1.0, alpha=0.55,
                    label="Fixed weights (w=0…1)" if i == 0 else None, zorder=1)
    m, _ = mean_curve("Passive")
    if m is not None:
        ax.plot(m, color=C_MUTED, lw=1.2, ls=":", label="Passive", zorder=1)
    m, _ = mean_curve("iGS")
    if m is not None:
        ax.plot(m, color=C_IGS, lw=2.0, label="iGS", zorder=2)

    for sel, col in [(RANDOM, C_RANDOM), (BANDIT, C_BANDIT)]:
        m, se = mean_curve(sel)
        if m is None:
            continue
        x = np.arange(len(m))
        ax.fill_between(x, m - 1.96 * se, m + 1.96 * se, color=col, alpha=0.18, lw=0, zorder=3)
        ax.plot(x, m, color=col, lw=2.0,
                label="Bandit (UCB1)" if sel == BANDIT else "Random arm", zorder=4)

    ax.set_xlabel("Round"); ax.set_ylabel("Test RMSE (mean over seeds)")
    ax.set_title(f"{dataset} — learning curves, random-arm control added",
                 loc="left", color=C_INK, fontsize=11, pad=10)
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    fig.tight_layout()
    p = os.path.join(out_dir, f"fig_learning_curves_{dataset}.png")
    fig.savefig(p, dpi=200); plt.close(fig)
    return p


### Figure 2: paired difference ###
def fig_paired_difference(summary, out_dir):
    d = summary[(summary["selector"] == BANDIT) &
                summary["diff_vs_randomarm_mean"].notna()].copy()
    if d.empty:
        return None
    d = d.sort_values("dataset")
    fig, ax = plt.subplots(figsize=(7.0, 0.9 * len(d) + 2.2))
    y = np.arange(len(d))

    ax.axvline(0, color=C_INK2, lw=1.2, zorder=1)
    for yi, (_, r) in zip(y, d.iterrows()):
        excl = r["diff_excludes_zero"]
        col = C_BANDIT if (excl and r["diff_vs_randomarm_mean"] < 0) else (
              C_RANDOM if excl else C_MUTED)
        ax.plot([r["diff_ci_lo"], r["diff_ci_hi"]], [yi, yi], color=col, lw=2.4,
                solid_capstyle="round", zorder=2)
        ax.plot([r["diff_vs_randomarm_mean"]], [yi], "o", ms=9, color=col,
                mec=C_SURFACE, mew=2, zorder=3)
        ax.annotate(f"{r['diff_vs_randomarm_mean']:+.2f}  [{r['diff_ci_lo']:+.2f}, {r['diff_ci_hi']:+.2f}]",
                    (max(r["diff_ci_hi"], 0), yi), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=8, color=C_INK2)

    ax.set_yticks(y); ax.set_yticklabels(d["dataset"])
    ax.set_xlabel("Paired difference in learning-curve area  (bandit − random arm)")
    ax.set_title("Bandit minus random arm, paired across seeds (95% bootstrap CI)",
                 loc="left", color=C_INK, fontsize=11, pad=10)
    ax.annotate("← bandit better", (0, -0.6), xytext=(-8, 0), textcoords="offset points",
                ha="right", fontsize=8, color=C_INK2, annotation_clip=False)
    ax.annotate("random arm better →", (0, -0.6), xytext=(8, 0), textcoords="offset points",
                ha="left", fontsize=8, color=C_INK2, annotation_clip=False)
    ax.margins(x=0.28)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    p = os.path.join(out_dir, "fig_paired_difference.png")
    fig.savefig(p, dpi=200); plt.close(fig)
    return p


### Figure 3: oracle path ###
def fig_oracle_path(oracle, out_dir):
    datasets = sorted(oracle["dataset"].unique())
    fig, axes = plt.subplots(3, len(datasets), figsize=(6.2 * len(datasets), 9.0),
                             squeeze=False)
    for j, ds in enumerate(datasets):
        g = oracle[oracle["dataset"] == ds]
        piv = g.pivot_table(index="round", columns="seed", values="oracle_w")
        m = piv.mean(axis=1); sd = piv.std(axis=1, ddof=1)
        x = piv.index.values

        ax = axes[0][j]
        ax.fill_between(x, m - sd, m + sd, color=C_BANDIT, alpha=0.18, lw=0)
        ax.plot(x, m, color=C_BANDIT, lw=1.6)
        ax.axhline(0.5, color=C_MUTED, lw=1.0, ls="--")
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel("Oracle w"); ax.set_title(f"{ds} — oracle weight vs round",
                                                loc="left", fontsize=11, color=C_INK)

        ax = axes[1][j]
        gp = g.pivot_table(index="round", columns="seed", values="gap")
        gm = gp.mean(axis=1)
        ax.plot(gp.index.values, gm, color=C_RANDOM, lw=1.6)
        ax.set_ylabel("Reward gap\n(max−min RMSE over w)")
        ax.set_title("Spread across the weight grid", loc="left", fontsize=11, color=C_INK)

        ax = axes[2][j]
        comp = (g.pivot_table(index="round", columns="region", values="seed",
                              aggfunc="count").fillna(0))
        comp = comp.div(comp.sum(axis=1), axis=0)
        # bucket rounds for legibility #
        bucket = (comp.index // max(1, len(comp) // 60))
        comp = comp.groupby(bucket).mean()
        bottom = np.zeros(len(comp))
        region_colors = {"sine_lownoise": C_BANDIT, "linear_lownoise": C_IGS,
                         "noise_trap": C_RANDOM, "cosine_lownoise": C_MUTED}
        for region in comp.columns:
            ax.bar(comp.index * max(1, len(g["round"].unique()) // 60), comp[region],
                   bottom=bottom, width=max(1, len(g["round"].unique()) // 60),
                   color=region_colors.get(region, C_MUTED), label=region, lw=0)
            bottom += comp[region].values
        ax.set_ylabel("Region share"); ax.set_xlabel("Round")
        ax.set_title("Region of the committed point", loc="left", fontsize=11, color=C_INK)
        ax.legend(fontsize=8, ncol=2, loc="upper right")
    fig.tight_layout()
    p = os.path.join(out_dir, "fig_oracle_path.png")
    fig.savefig(p, dpi=200); plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default=None)
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()
    out_dir = args.dir or os.path.join(os.path.dirname(os.getcwd()), "results_grfp_controls")

    written = []
    runs_p = os.path.join(out_dir, "exp_a_runs.csv")
    if os.path.exists(runs_p):
        runs = pd.read_csv(runs_p)
        summary, p = build_summary(runs, out_dir, n_boot=args.n_boot); written.append(p)

        curves_p = os.path.join(out_dir, "exp_a_curves.pkl")
        if os.path.exists(curves_p):
            with open(curves_p, "rb") as fh:
                curves = pickle.load(fh)
            for ds in sorted(runs["dataset"].unique()):
                written.append(fig_learning_curves(curves, ds, out_dir))
        written.append(fig_paired_difference(summary, out_dir))

        arms_p = os.path.join(out_dir, "exp_a_arms.csv")
        if os.path.exists(arms_p):
            arms = pd.read_csv(arms_p)
            _, p = build_reward_gap(arms, out_dir); written.append(p)

    oracle_p = os.path.join(out_dir, "exp_b_oracle_path.csv")
    if os.path.exists(oracle_p):
        oracle = pd.read_csv(oracle_p)
        written.append(fig_oracle_path(oracle, out_dir))

    for w in written:
        if w:
            print("wrote", w)


if __name__ == "__main__":
    main()
