"""
Step 0 of the bandit-scale diagnosis: read the September control runs and
report, per (dataset, seed), whether UCB1 is behaving as round-robin, how the
exploration bonus compares with the reward scale, how many rounds used the
CV fallback, and how early the fixed-weight curves become indistinguishable.

No reruns. The September controls (repo protocol, CandidateProportion=0.99,
held-out test RMSE, 13 datasets x 100 seeds) were run by
Code/utils/GRFPControls/RunExpA.py, not through LearningProcedure, so they are
read from that sweep's outputs rather than Results/simulation_results/raw:

    results_grfp_controls/            dgp_two_regime, dgp_three_regime
    results_grfp_controls/benchmarks/ the 11 benchmark datasets

`exp_a_arms.csv.gz` logs the arm pulled every round (the `WeightHistory`
equivalent) along with the running arm values and counts; `exp_a_curves.pkl.gz`
holds the per-round test-RMSE curve for every (dataset, selector, seed).

Usage (from the repository root):
    python scripts/diagnose_bandit.py
"""

import argparse
import gzip
import math
import os
import pickle

import numpy as np
import pandas as pd

ARMS = [0.25, 0.5, 0.75]
MAB = "MAB (c=2.0)"
FIXED = {0.0: "Fixed w=0.00", 0.25: "Fixed w=0.25", 0.5: "Fixed w=0.50",
         0.75: "Fixed w=0.75", 1.0: "Fixed w=1.00"}
BONUS_T = [10, 25, 50, 100]
MAB_C = 2.0
CV_MIN_N = 10                         # get_cv_rmse returns NaN below 2k = 10
CANDIDATE_PROPORTION = 0.99           # the September runs' repo protocol
TEST_PROPORTION = 0.10
SOURCES = ["results_grfp_controls", os.path.join("results_grfp_controls", "benchmarks")]


def load_sources(root):
    arms, curves = [], {}
    for src in SOURCES:
        d = os.path.join(root, src)
        arms.append(pd.read_csv(os.path.join(d, "exp_a_arms.csv.gz")))
        with gzip.open(os.path.join(d, "exp_a_curves.pkl.gz"), "rb") as fh:
            curves.update(pickle.load(fh))
    arms = pd.concat(arms, ignore_index=True)
    return arms[arms["selector"] == MAB], curves


def initial_train_size(root, dataset):
    """n0 as ControlTrainCandidateSplit computes it under the repo protocol."""
    df = pd.read_pickle(os.path.join(root, "Data", "processed", f"{dataset}.pkl")).dropna()
    n = len(df)
    n_learning = n - math.ceil(n * TEST_PROPORTION)     # sklearn rounds test up
    return max(2, int(n_learning * (1.0 - CANDIDATE_PROPORTION))), n_learning


def bonus(t):
    """UCB1 bonus at round t when the three arms have been pulled evenly."""
    return MAB_C * math.sqrt(math.log(t) / (t / len(ARMS)))


### 0a. Round-robin test ###
def round_robin(seq):
    counts = np.zeros(len(ARMS))
    gaps, least = [], []
    for t, a in enumerate(seq):
        if t >= len(ARMS):
            least.append(counts[a] == counts.min())
        counts[a] += 1
        if t >= len(ARMS):
            gaps.append(counts.max() - counts.min())
    return (float(max(gaps)) if gaps else np.nan,
            float(np.mean(least)) if least else np.nan)


### Bonus against the spread of the arm values, at round 25 ###
def arm_state_at(seq, rewards, t):
    """
    Arm counts and values the UCB1 decision at round t used.

    Rebuilt from (arm pulled, reward) rather than read from the logged
    arm_value_*/arm_count_* columns: those are corrupt in the top-level (DGP)
    log, which carries six of each with values like 64. The reward logged at
    round s is the one credited to the arm pulled at round s, observed at s+1,
    so the state at round t uses rewards from rounds < t.
    """
    counts = np.zeros(len(ARMS))
    sums = np.zeros(len(ARMS))
    for s in range(t):
        counts[seq[s]] += 1
        if np.isfinite(rewards[s]):
            sums[seq[s]] += rewards[s]
    # the selector divides by the pull count, rewarded or not
    values = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    return counts, values


def ratio_at_25(seq, rewards):
    """
    The brief's 0b proxy assumes arm_values were not logged; this sweep logged
    the rewards, so the same bonus/value comparison is also made on the
    bandit's own state.
    """
    if len(seq) <= 25:
        return np.nan, np.nan
    counts, values = arm_state_at(seq, rewards, 25)
    b = MAB_C * np.sqrt(np.log(25) / counts)
    spread = values.max() - values.min()
    return float(spread), (float(b.mean() / spread) if spread > 0 else np.inf)


### 0d. Window: where the w=0.75 vs w=0 separation accumulates ###
WINDOW_PAIR = ("Fixed w=0.75", "Fixed w=0.00")
WINDOW_SHARE = 0.90


def window_90(curves, dataset):
    """
    Round by which 90% of the accumulated gap between the mean test-RMSE curves
    for w=0.75 and w=0 is reached.

    The gap is |mean_0.75(t) - mean_0(t)| per round, accumulated from round 0;
    window_90 is the first round whose running sum reaches 90% of the final sum.
    Replaces the earlier "all five weights within one paired SE for the rest of
    the run" rule, which held only once the pool was exhausted.
    """
    a, b = WINDOW_PAIR
    seeds = sorted(set(s for (d, sel, s) in curves if d == dataset and sel == a)
                   & set(s for (d, sel, s) in curves if d == dataset and sel == b))
    ma = np.mean([curves[(dataset, a, s)] for s in seeds], axis=0)
    mb = np.mean([curves[(dataset, b, s)] for s in seeds], axis=0)
    cum = np.cumsum(np.abs(ma - mb))
    return int(np.argmax(cum >= WINDOW_SHARE * cum[-1])), len(cum)


### Switch-round artifact ###
def switch_arm(seq, rewards, n0):
    """
    The arm credited with the reward that spans the train-RMSE -> CV-RMSE switch.

    CV is first defined at round t_s = 10 - n0. The reward computed there is
    train_rmse(n=9) - cv_rmse(n=10) and is credited to the arm pulled at round
    t_s - 1, which is returned. The arm pulled at t_s itself is returned too.
    None when n0 >= 10 (no fallback rounds, no switch).
    """
    t_s = CV_MIN_N - n0
    if t_s <= 0 or t_s >= len(seq):
        return None
    return seq[t_s - 1], float(rewards[t_s - 1]), seq[t_s]


def same_as_least(arm, seq):
    """1/k if `arm` is among the k arms tied for the lowest final count, else 0."""
    counts = np.bincount(seq, minlength=len(ARMS))
    least = np.flatnonzero(counts == counts.min())
    return (1.0 / len(least) if arm in least else 0.0), len(least)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--out", default=os.path.join("Results", "diagnostics", "bandit_scale.csv"))
    args = ap.parse_args()

    arms, curves = load_sources(args.root)
    arm_of = {w: k for k, w in enumerate(ARMS)}
    rows = []
    for dataset, gd in arms.groupby("dataset"):
        n0, n_learning = initial_train_size(args.root, dataset)
        w90, n_rounds_fixed = window_90(curves, dataset)
        for seed, g in gd.groupby("seed"):
            g = g.sort_values("round")
            seq = [arm_of[float(w)] for w in g["w_x"]]
            gap, frac = round_robin(seq)

            curve = np.asarray(curves[(dataset, MAB, seed)], dtype=float)
            d = curve[:-1] - curve[1:]
            scale = float(np.std(d))
            last = len(seq)
            # n_learning - n0 selections, plus the final post-exhaustion point
            assert len(curve) == n_learning - n0 + 1, (dataset, seed, len(curve), n_learning, n0)

            row = {"dataset": dataset, "seed": int(seed),
                   "n_rounds": last,
                   "max_count_gap": gap,
                   "frac_rounds_least_pulled": frac,
                   "reward_scale": scale,
                   "reward_mean_abs": float(np.mean(np.abs(d)))}
            for t, tag in [(t, str(t)) for t in BONUS_T] + [(last, "last")]:
                b = bonus(t) if 2 <= t <= last else np.nan
                row[f"bonus_at_{tag}"] = b
                row[f"ratio_at_{tag}"] = b / scale if scale > 0 else np.nan

            # The same scale on the reward the bandit actually received.
            r = g["reward"].to_numpy(dtype=float)
            post = (n0 + g["round"].to_numpy()) >= CV_MIN_N
            row["cv_reward_scale"] = float(np.nanstd(r))
            row["cv_reward_scale_cv_rounds"] = float(np.nanstd(r[post])) if post.any() else np.nan
            row["ratio_cv_at_25"] = (bonus(25) / row["cv_reward_scale_cv_rounds"]
                                     if last >= 25 else np.nan)
            spread, ratio = ratio_at_25(seq, r)
            row["arm_value_spread_at_25"] = spread
            row["bonus_over_value_spread_at_25"] = ratio

            row["n0"] = n0
            row["fallback_rounds"] = max(0, CV_MIN_N - n0)
            row["window_90"] = w90
            row["window_90_frac"] = w90 / n_rounds_fixed

            sw = switch_arm(seq, r, n0)
            if sw is None:
                row.update(switch_arm=np.nan, switch_reward=np.nan, switch_is_least=np.nan,
                           least_ties=np.nan, next_arm_is_least=np.nan)
            else:
                arm, rew, nxt = sw
                row["switch_arm"] = arm
                row["switch_reward"] = rew
                row["switch_is_least"], row["least_ties"] = same_as_least(arm, seq)
                row["next_arm_is_least"], _ = same_as_least(nxt, seq)
            # median |reward| over the CV-defined rounds, to put switch_reward on scale
            row["median_abs_cv_reward"] = float(np.nanmedian(np.abs(r[post]))) if post.any() else np.nan
            rows.append(row)

    df = pd.DataFrame(rows).sort_values(["dataset", "seed"])
    out = os.path.join(args.root, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)

    summary = df.groupby("dataset").agg(
        n_seeds=("seed", "size"),
        n_rounds=("n_rounds", "median"),
        max_count_gap=("max_count_gap", "median"),
        frac_least_pulled=("frac_rounds_least_pulled", "median"),
        ratio_at_25=("ratio_at_25", "median"),
        bonus_over_value_spread_25=("bonus_over_value_spread_at_25", "median"),
        fallback_rounds=("fallback_rounds", "first"),
        window_90=("window_90", "first"),
        window_90_frac=("window_90_frac", "first"),
    )
    with pd.option_context("display.width", 200, "display.float_format", "{:.3g}".format):
        print(summary.to_string())
    print(f"\nmax_count_gap == 1 on {int((df.max_count_gap == 1).sum())}/{len(df)} runs; "
          f"frac_rounds_least_pulled == 1 on {int((df.frac_rounds_least_pulled == 1).sum())}/{len(df)}")
    print(f"ratio_at_t >= 10 at every t on "
          f"{int((df[[c for c in df if c.startswith('ratio_at_')]].fillna(np.inf) >= 10).all(axis=1).sum())}"
          f"/{len(df)} runs")

    # Switch-round artifact, benchmarks only (the two DGPs have n0 = 13 and 9).
    bench = df[~df.dataset.str.startswith("dgp_")]
    sw = bench.groupby("dataset").agg(
        n_seeds=("seed", "size"),
        fallback_rounds=("fallback_rounds", "first"),
        frac_switch_arm_is_least=("switch_is_least", "mean"),
        frac_next_arm_is_least=("next_arm_is_least", "mean"),
        seeds_with_tied_least=("least_ties", lambda x: int((x > 1).sum())),
        median_switch_reward=("switch_reward", "median"),
        median_abs_cv_reward=("median_abs_cv_reward", "median"),
    )
    print("\nSwitch-round arm vs least-pulled arm (null ~ 1/3; ties credited 1/k):")
    with pd.option_context("display.width", 200, "display.float_format", "{:.3g}".format):
        print(sw.to_string())
    print(f"pooled over {len(bench)} benchmark runs: switch arm {bench.switch_is_least.mean():.3f}, "
          f"next arm {bench.next_arm_is_least.mean():.3f}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
