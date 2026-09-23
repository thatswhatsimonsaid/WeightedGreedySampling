# GRFP control experiments — summary

Branch: `troy-branch`. Nothing here modifies the paper's selectors, evaluation
code, reward code, config or results directories; all new code lives in
`Code/utils/GRFPControls/` and all output in `results_grfp_controls/`.

Model Ridge (`alpha=0.01`), seeds 0–99, `CandidateProportion=0.99` (the pool is
run to exhaustion), 10% held-out test split, reward = per-step drop in 5-fold CV
RMSE, metric = test RMSE. Bandit arm set `{0.25, 0.5, 0.75}` (the repo default,
never overridden in the paper's config); UCB1 `c=2.0`.

**Scope: 13 datasets × 9 selectors × 100 seeds = 11,700 runs** (2 synthetic +
11 benchmarks), plus 445,600 oracle-lookahead rounds in Experiment B.

---

## Headline: branch (b), on all 13 datasets

The bandit does not beat a random-arm policy anywhere. Paired on 100 seeds per
dataset, the 95% bootstrap CI for *bandit − random arm* **includes zero on
13 of 13 datasets** at full budget. In the early-budget window it excludes zero
once, and that one favours the **random arm**, not the bandit. Effect sizes are
under 0.4% of the metric everywhere.

| | Full budget | First 25% |
|---|---|---|
| CI excludes zero | **0 / 13** | 1 / 13 |
| …favouring the bandit | 0 | **0** |
| …favouring the random arm | 0 | 1 |

**The mechanism.** The between-arm reward gap is a small fraction of the
within-arm reward noise — and on the real benchmarks it is far *smaller* than on
the synthetic sets:

| Dataset group | gap / noise |
|---|---|
| 11 benchmarks | **0.004 – 0.034** |
| dgp_three_regime | 0.117 |
| dgp_two_regime | 0.140 |

The reward the bandit sees cannot separate its arms, so UCB1 explores them
near-uniformly (44758 / 44714 / 44228 on dgp_three_regime) and *is*, in effect,
the random-arm policy it is being compared against. Blocking on round leaves the
ratio unchanged, so this is not an artefact of rewards shrinking over a run.

**Confirmed independently by lookahead.** Along the bandit's own realised
trajectory (222,800 rounds, free labels), its chosen arm agrees with the
oracle-best arm at **0.3349 and 0.3302 against a chance rate of 0.3333** — at
chance, with one marginally below. Its arm choice carries no information about
which arm is better. This never touches the learning-curve area, so it does not
depend on the metric at all.

## The sharper result: a fixed weight beats the bandit

The adaptivity is not merely neutral against a random arm — it loses to simply
picking one good weight and keeping it. Mean change in area vs passive across
the 11 benchmarks (more negative is better):

| Selector | vs passive |
|---|---|
| **Fixed w=0.75** | **−3.45%** |
| Fixed w=0.50 | −3.07% |
| MAB (c=2.0) | −2.87% |
| RandomArm | −2.81% |
| Fixed w=0.25 | −2.74% |
| iGS | −2.72% |
| Fixed w=1.00 | −2.49% |
| Fixed w=0.00 | −1.59% |

The bandit lands between the fixed weights, which is where averaging over its
arm set puts you. Paired head-to-head, **`Fixed w=0.75` is significantly better
than the bandit on 6 of 11 benchmarks and significantly worse on 1** (beer,
−0.52%); four are ties.

**Against the best fixed weight chosen in hindsight per dataset** — the exact
comparison the manuscript claims WiGS matches — the bandit never wins:

| Budget | Fixed better | MAB better | Statistical match |
|---|---|---|---|
| Full | **8 / 13** | 0 / 13 | 5 / 13 |
| First 25% | **11 / 13** | 0 / 13 | 2 / 13 |

So under held-out test error the claim that WiGS *matches the best fixed weight
in hindsight* does not hold, and it fails harder at the shorter budget — which
is the budget at which weighting has any leverage at all. The shortfall is
+0.3% to +3.6% of area at full budget and up to +7.7% early (concrete_slump).

## What does work: the weighted score function

This is the correction to the picture the synthetic sets alone suggested. On the
11 benchmarks, **active selection clearly beats passive**: the bandit improves on
passive on 8 of 11 (CIs excluding zero) — bodyfat −7.2%, concrete_flow −4.5%,
concrete_slump −3.8%, beer −3.6%, concrete_cs −3.3%, mpg −2.9%, qsar −2.0%,
pm10 −1.0% — and is never significantly worse. *Every* weighted variant beats
passive by 1.6–3.4% on average.

The two synthetic datasets were the only ones where passive won, and they are
the two where the model is degenerate (below). **The earlier reading that
"passive beats active" was an artefact of those two datasets and does not
survive contact with the benchmarks.** The paper's case for greedy-sampling
selection as such is not threatened by anything here.

## Model capability — the caveat that the benchmarks resolved

On the two synthetic DGPs a ridge model cannot fit the signal, so no selection
rule could matter much there and "flat in `w`" was close to guaranteed:

| Dataset | R² (5-fold CV) | | Dataset | R² (5-fold CV) |
|---|---|---|---|---|
| concrete_cs | 0.822 | | qsar | 0.365 |
| beer | 0.654 | | housing | 0.354 |
| yacht | 0.650 | | concrete_flow | 0.325 |
| bodyfat | 0.645 | | pm10 | 0.114 |
| mpg | 0.594 | | concrete_slump | 0.093 |
| no2 | 0.488 | | **dgp_two_regime** | **0.131** |
| | | | **dgp_three_regime** | **−0.004** |

The benchmarks span capable fits (concrete_cs 0.82, beer 0.65, yacht 0.65,
bodyfat 0.65, mpg 0.59) and the result is unchanged on every one of them. **The
model-degeneracy explanation is therefore ruled out**: branch (b) is a property
of the weight-adaptation layer, not of the two synthetic datasets.

## Experiment B — oracle weight path

100 seeds × 2 synthetic datasets, weight grid {0, 0.1, …, 1} plus the bandit's
own arms, each round scored on a fresh 20,000-point test set drawn from the
generator and mapped through the scaler fitted on the original sample
(`exp_b_oracle_path.csv.gz`, 222,800 rounds).

**No sustained trend in the oracle weight.** Spearman(round, oracle w) is −0.098
and −0.031. Mean oracle w is 0.21 and 0.28 with a standard deviation across
seeds of ~0.37 — the ±1 sd band covers most of [0, 1], so the per-seed oracle
arm is close to unidentifiable. There *is* a real but short-lived early drift:
the mean starts near 0.4–0.5 and falls to ~0.2 by round ~50, flat thereafter. So
Aim 2's separation hypothesis gets weak, qualified support confined to the
opening rounds, not the sustained drift it would want.

**The objective goes flat fast.** The spread between best and worst weight falls
more than two orders of magnitude over a run:

| Rounds | Gap, full grid | as % of error | Gap, bandit's arms | as % of error |
|---|---|---|---|---|
| 0–24 | 1.0–1.8e−2 | 1.8–2.0% | 5.1–8.7e−3 | 0.90–0.98% |
| 25–99 | 1.0–2.3e−3 | 0.18–0.27% | 3.6e−4–1.0e−3 | 0.06–0.12% |
| 100–249 | 3.3–4.8e−4 | 0.055–0.060% | 1.1–1.9e−4 | 0.02% |
| 250+ | 5e−5–2.3e−4 | 0.006–0.041% | ~1e−5–8e−5 | 0.001–0.015% |

The weight has leverage for roughly the first 25 rounds and essentially none
after ~100. Under the paper's budget — the pool run to exhaustion, 891 and 1338
rounds — **97%+ of rounds are spent where `w` provably cannot matter**.

**Region composition** is roughly stable. The high-noise trap is 5.4% and 7.3% of
committed points; the oracle neither chases nor avoids it. There is no
dense-noisy / sparse-clean switch to point to.

---

## What this supports in the plan

Three independent lines — the reward the bandit sees, the learning curves, and
free-label lookahead — agree, so the conclusion does not rest on any one of them.

- **Aim 1 should be framed as branch (b), and it can now be stated generally**
  rather than for synthetic data only. One line: *the between-arm reward gap is a
  few percent of the within-arm reward noise, so UCB1 explores near-uniformly,
  never identifies an arm, and performs exactly like drawing the weight at
  random — which on most benchmarks is worse than just fixing w=0.75.*
- **"Never settles on one arm" is not evidence of useful adaptation.** It is what
  a bandit does when its arms are indistinguishable. The lookahead makes this
  concrete: arm choice agrees with the oracle at chance across 222,800 rounds.
- **Separate the two claims the method bundles together.** The weighted score
  function earns its keep (1.6–3.4% over passive on benchmarks); the online
  adaptation of `w` does not. That is a cleaner and more defensible Aim 1 than
  either "WiGS works" or "WiGS doesn't".
- **The interesting regime is small budgets.** Experiment B locates the leverage
  precisely: roughly the first 25 rounds, under 3% of the paper's budget. That is
  a concrete, testable claim for Aim 2, and it suggests the pool-exhaustion budget
  is the wrong place to look for weighting effects.
- **A better target than adapting `w` online: choosing `w` well once.** The spread
  between the best and worst fixed weight (−3.45% vs −1.59% against passive) is
  much larger than anything the adaptation recovers.

## What this does not establish

1. **Eight of the paper's 21 datasets have not been run** — `wine_white`,
   `wine_red`, `concrete_4`, `cps_wage`, `dgp_new` and the three `burbidge`
   variants. These were skipped on cost: with the pool run to exhaustion, cost
   scales as roughly `T³·d`, and `wine_white` alone is ~1300× `dgp_two_regime`.
   They need the cluster. The 13 run here are consistent enough (0/13) that I
   would be surprised by a reversal, but that is an expectation, not a result.
2. **Only `c=2.0` was run.** The paper also reports `c=0.5` and `c=5.0`. A smaller
   exploration constant would concentrate pulls sooner — on an arm chosen largely
   by noise, given the gap/noise ratios — but that is an inference.
3. **Ridge only.** The paper's config also lists Gaussian-process and
   random-forest predictors, both commented out. A model whose error responds more
   sharply to *which* points are labelled could have a larger between-arm gap.
   This is the most plausible route by which branch (a) could still hold somewhere.
4. **The fixed-w=0.75 result is an in-sample winner.** It is the best of five
   fixed weights *chosen on these same benchmarks*, so its margin over the bandit
   is optimistic as a forward-looking recommendation. The bandit-vs-random-arm
   result does not have this problem.

**Recommended next step:** the remaining 8 datasets on the cluster, and — more
informative per unit compute — a non-linear model (the GP or RF already in the
config) on the current 13, which is the one condition under which the reward
might separate the arms.

Per the brief's pitfall note, a branch-(b) outcome bears on the paper's framing
of the bandit and should go to Simon and Tyler. **Nothing has been sent, pushed,
or posted.** I would send it with the "what does work" section attached, since
the headline result is about the weight-adaptation layer specifically and not
about WiGS selection as a whole.

---

## Deliverables

| File | Contents |
|---|---|
| `exp_a_summary.csv` | per dataset × selector: mean/SE area (full and early), final error, paired diff vs random arm with bootstrap CI |
| `exp_a_vs_randomarm.csv` / `exp_a_vs_passive.csv` | every selector against each reference, both windows |
| `exp_a_reward_gap.csv` / `exp_a_reward_gap_blocked.csv` | between-arm gap vs within-arm noise, pooled and round-blocked |
| `exp_a_arms.csv.gz` | per-round arm, reward, CV and test RMSE, running arm means and counts |
| `exp_a_runs.csv`, `exp_a_curves.pkl` | per-run areas; every per-seed test-RMSE learning curve |
| `benchmarks/` | the same set for the 11 benchmark datasets (kept separate so neither run overwrites the other; `Analyze --also` merges them) |
| `exp_b_oracle_path.csv.gz` | per-round oracle weight, per-weight errors, gap, region |
| `exp_b_bandit_trajectory.csv.gz` | add-on: per-w lookahead along the bandit's own trajectory, chosen arm vs best arm |
| `fig_paired_difference.png` | the decision figure, all 13 datasets, both budget windows |
| `fig_learning_curves_*.png`, `fig_oracle_path.png` | learning curves with the control added; oracle path |

Reproduce with, from `Code/`:

```
python -m utils.GRFPControls.RunExpA --datasets <names> --seeds 100 --workers 7
python -m utils.GRFPControls.RunExpB --datasets dgp_two_regime dgp_three_regime \
    --seeds 100 --workers 7 --policy oracle
python -m utils.GRFPControls.Analyze --also results_grfp_controls/benchmarks
```

---

## Deviations from the paper's protocol, and why

1. **Test-set evaluation is new code on this branch.** troy-branch evaluates on
   the full pool (train + candidate) and has no test split; the brief requires
   the paper's test-error curve. Rather than edit the evaluation code, which the
   brief forbids, `ControlProtocol.py` reproduces test-set-eval's 3-way split and
   test scoring verbatim as new functions. All selectors are byte-identical
   between the two branches, so the selector/score/reward path is the paper's
   unchanged.
2. **Per-seed area, not area-of-the-mean.** `GenerateAUCTable.py` averages curves
   over seeds and then integrates. Pairing requires each seed's own area, so the
   integration is per seed. Integration and averaging commute, so the means agree;
   only the per-seed spread is newly available.
3. **An early-budget window is reported alongside the full budget.** The curves
   converge well before the pool is exhausted, so the full-budget area is
   dominated by a flat tail common to every selector. The 25% fraction was fixed
   before looking at results and both windows are always reported.
4. **Final error at T is degenerate and is not used.** Running the pool to
   exhaustion means every selector finishes having labelled the same points, so
   the final model — and final test error — is identical across selectors by
   construction. The brief asks for it; the honest answer is that it carries no
   signal under this budget.
5. **Fixed-weight endpoints come from `WeightedGreedySamplingSelector` rather
   than GSx/GSy — but the two are equivalent.** GSx/GSy score on *unnormalized*
   distances while WiGS min-max normalizes, so I initially treated them as
   different selectors. They are not: min-max normalization is a single global
   affine map with positive scale, so it commutes with the `min` over the
   training set and the `argmax` over the pool. Verified 500/500 identical
   selections across four datasets (`verification/check_endpoints.py`). Using
   one selector for the whole sweep is still the tidier choice, but nothing
   turned on it, and `w=1` / `w=0` really are the pure diversity and pure
   uncertainty selectors.
6. **CV is skipped for selectors that never read it.** Only `WiGS_MAB_Selector`
   consumes `current_rmse`; the rest swallow it in `**kwargs`, and `get_cv_rmse`
   uses a fixed `KFold(random_state=42)` that draws nothing from the global RNG.
   Verified bit-identical curves and selections either way; 2.4× faster.
7. **The bandit's reward is reconstructed, not intercepted.** `WiGS_MAB_Selector`
   keeps its reward internally and does not return it, and may not be edited. It
   is recomputed from the CV trace as `cv[t] − cv[t+1]`, matching the control's
   own logged reward exactly (max abs difference 0.0).
8. **The three large logs are gzipped** — arms 120 MB → 9.7 MB (also one numeric
   column per arm rather than repr-serialised arrays), oracle path 99 MB →
   11.8 MB, bandit trajectory 101 MB → 14.4 MB. The brief names them without the
   extension; `pd.read_csv` opens them directly.
9. **Not done: the `aaai2027-submission` tag.** The repo has no tags and it is
   ambiguous which commit produced the submitted numbers; tagging a shared repo's
   submitted state is the authors' call.
10. **8 of 21 datasets not run** — see "What this does not establish".

## Correctness checks

`Code/utils/GRFPControls/verification/` — all passing:

- `RandomArmSelector` and `WiGS_MAB_Selector` restricted to a single arm select
  the *same* point for w ∈ {0, 0.25, 0.5, 0.75, 1} — so the control differs from
  the bandit only in the arm rule.
- All five selectors see identical splits for a given seed (same initial train
  indices, same test set), which is what makes the pairing valid.
- Derived reward == the control's internally computed reward, exactly.
- CV-skip produces bit-identical curves and selection histories.
- Experiment B's fast path reproduces the reference pandas implementation exactly
  (max diff 0.000e+00) at 2.8×.
- The synthetic `.pkl`s are reproduced exactly by generator + RobustScaler, which
  is what lets Experiment B draw a fresh large test set into the same feature
  space.
