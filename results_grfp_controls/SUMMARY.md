# GRFP control experiments — summary

Branch: `troy-branch`. Nothing here modifies the paper's selectors, evaluation
code, reward code, config or results directories; all new code lives in
`Code/utils/GRFPControls/` and all output in `results_grfp_controls/`.

Model Ridge (`alpha=0.01`), seeds 0–99, `CandidateProportion=0.99` (the pool is
run to exhaustion), 10% held-out test split, reward = per-step drop in 5-fold CV
RMSE, metric = test RMSE. Bandit arm set `{0.25, 0.5, 0.75}` (the repo default,
never overridden in the paper's config); UCB1 `c=2.0`.

---

## Headline: branch (b), on these two datasets, with a large caveat

The bandit does not beat a random-arm policy. Paired on seeds, in both budget
windows, on both synthetic datasets, the 95% bootstrap CI includes zero:

| Dataset | Window | Bandit − random arm | 95% CI | Excludes 0 |
|---|---|---|---|---|
| dgp_two_regime | full | −0.079 | [−0.731, +0.577] | no |
| dgp_two_regime | first 25% | −0.057 | [−0.572, +0.447] | no |
| dgp_three_regime | full | +0.240 | [−1.323, +1.753] | no |
| dgp_three_regime | first 25% | +0.430 | [−1.085, +1.905] | no |

(n = 100 seeds each; area is of the test-RMSE curve, lower is better. Area
levels are ~485 and ~1150, so these differences are ~0.02–0.04% of the metric.)

**The caveat that governs everything below: on these two DGPs a ridge model is
degenerate.** The best possible linear fit gives R² = 0.138 on dgp_two_regime
and **R² = 0.0012** on dgp_three_regime — on the latter, predicting the mean is
as good as the model can do. Every selector converges to that floor almost
immediately:

| Dataset | std(Y) | Best linear fit RMSE | R² | Converged RMSE |
|---|---|---|---|---|
| dgp_two_regime | 0.5768 | 0.5356 | 0.138 | 0.558 |
| dgp_three_regime | 0.8529 | 0.8524 | 0.0012 | 0.859 |

So the objective's flatness in `w` here is substantially a model–DGP mismatch,
not by itself evidence that weighting never matters. **Branch (b) is established
for these two synthetic datasets; it is not yet established for the paper's
benchmarks**, where ridge is a sensible model. That distinction matters for how
Aim 1 is framed and is the single most important open item.

## Mechanism: the reward cannot separate the arms

From the logged per-round rewards (`exp_a_reward_gap.csv`,
`exp_a_reward_gap_blocked.csv`):

| Dataset | Selector | Between-arm gap | Within-arm noise | Ratio | Gap in per-run SEs |
|---|---|---|---|---|---|
| dgp_two_regime | MAB (c=2.0) | 0.00139 | 0.01121 | **0.124** | 2.14 |
| dgp_two_regime | RandomArm | 0.00145 | 0.01038 | **0.140** | 2.43 |
| dgp_three_regime | MAB (c=2.0) | 0.00151 | 0.01466 | **0.103** | 2.20 |
| dgp_three_regime | RandomArm | 0.00173 | 0.01471 | **0.117** | 2.52 |

The gap between the best and worst arm is about **one tenth** of the
round-to-round noise in the reward. This is not an artefact of rewards shrinking
over the run: blocking on round and re-centring within (seed, block) leaves the
ratio unchanged at 0.104–0.141.

Expressed per run, the best-vs-worst arm gap is only ~2.1–2.5 standard errors of
an arm's mean *after the full run* (297 pulls per arm on dgp_two_regime, 446 on
dgp_three_regime). The bandit has to choose from round one, when it has a
handful of pulls and the separation is far below noise.

The behaviour follows directly. UCB1 at `c=2.0` pulls its three arms
44758 / 44714 / 44228 times on dgp_three_regime and 29803 / 29767 / 29530 on
dgp_two_regime — near-perfectly uniform. It never settles because it cannot;
and a policy that pulls uniformly *is* a random-arm policy, which is exactly why
the control matches it. The RandomArm rows above are the cleaner estimate of the
arm-reward distributions, being unconfounded by UCB1's adaptive allocation.

## Every selector, against passive learning

`exp_a_vs_passive.csv`. Paired on seeds, full budget, 95% bootstrap CI. Positive
= worse than passive (higher error area):

| Dataset | Selector | Diff | 95% CI | % of passive | Favours |
|---|---|---|---|---|---|
| dgp_three_regime | RandomArm | +5.48 | [+3.68, +7.34] | +0.48% | Passive |
| dgp_three_regime | MAB (c=2.0) | +5.72 | [+4.12, +7.34] | +0.50% | Passive |
| dgp_three_regime | iGS | +13.35 | [+9.76, +16.95] | +1.17% | Passive |
| dgp_two_regime | Fixed w=1.00 | **−0.75** | [−1.40, −0.14] | −0.16% | **selector** |
| dgp_two_regime | iGS | +0.22 | [−0.55, +0.96] | +0.05% | neither |
| dgp_two_regime | MAB (c=2.0) | +1.07 | [+0.18, +1.95] | +0.22% | Passive |

On dgp_three_regime **all eight** active selectors are significantly worse than
passive random sampling. On dgp_two_regime only `Fixed w=1.00` beats it.

This should be read narrowly. It is two synthetic datasets on which the model
explains ~0–14% of the variance, so "active selection does not help" is close to
vacuous there. It is also a different metric from the paper's published numbers:
under the paper-era full-pool metric, WiGS-MAB beat passive by 27–30% on these
same two datasets, versus +0.2–0.5% *worse* under held-out test error. That
reversal is a metric effect, and is presumably why the `test-set-eval` branch
exists. It is not a like-for-like refutation of anything in the paper, and it
should not be treated as one without the benchmark runs.

## Experiment B — oracle weight path

100 seeds × 2 datasets, weight grid {0, 0.1, …, 1} plus the bandit's own arms
{0.25, 0.75}, each round scored on a fresh 20,000-point test set drawn from the
generator and mapped through the scaler fitted on the original sample.
222,800 rounds total (`exp_b_oracle_path.csv.gz`).

**There is no sustained trend in the oracle weight.** Spearman(round, oracle w)
is −0.098 on dgp_three_regime and −0.031 on dgp_two_regime. Mean oracle w over a
whole run is 0.21 and 0.28, with a standard deviation across seeds of ~0.37 —
the ±1 sd band covers most of [0, 1], so the per-seed oracle arm is close to
unidentifiable.

**There is a real but short-lived early drift.** The mean oracle w starts near
0.4–0.5, falls to ~0.2 by round ~50, and is flat and noisy thereafter. So the
separation hypothesis gets weak, qualified support confined to the opening
rounds, not the sustained drift Aim 2 would want.

**The decisive number is how fast the objective goes flat.** The spread between
the best and worst weight falls by more than two orders of magnitude over a run:

| Rounds | Gap, full grid | as % of error | Gap, bandit's arms | as % of error |
|---|---|---|---|---|
| 0–24 | 1.0–1.8e−2 | 1.8–2.0% | 5.1–8.7e−3 | 0.90–0.98% |
| 25–99 | 1.0–2.3e−3 | 0.18–0.27% | 3.6e−4–1.0e−3 | 0.06–0.12% |
| 100–249 | 3.3–4.8e−4 | 0.055–0.060% | 1.1–1.9e−4 | 0.02% |
| 250+ | 5e−5–2.3e−4 | 0.006–0.041% | ~1e−5–8e−5 | 0.001–0.015% |

The weight has leverage for roughly the first 25 rounds and essentially none
after ~100. Under the paper's budget — the pool run to exhaustion, 891 and 1338
rounds — **97%+ of rounds are spent in a regime where `w` provably cannot
matter**, and on 99–100% of rounds the gap across the bandit's own arms is under
0.1% of the error level. That is the same conclusion Experiment A reaches from
the reward side, measured independently with free labels.

### Add-on: does the bandit's arm track the oracle arm?

The brief's optional extension, run in full (`exp_b_bandit_trajectory.csv.gz`,
100 seeds, 222,800 rounds). Along the bandit's *own* realised trajectory, every
weight is evaluated by lookahead, and the bandit's chosen arm is compared with
the best arm restricted to its own arm set {0.25, 0.5, 0.75}.

**The bandit picks the best arm at chance.**

| Dataset | Rounds | Agreement with oracle arm | Chance | z | Mean regret vs best arm |
|---|---|---|---|---|---|
| dgp_three_regime | 133,700 | 0.3349 | 0.3333 | +1.2 | 0.069% of error |
| dgp_two_regime | 89,100 | 0.3302 | 0.3333 | −2.0 | 0.082% of error |

Neither is meaningfully above chance, and dgp_two_regime is marginally *below*
it. With >89,000 rounds per dataset these are precise estimates, not
underpowered ones. Even over rounds 0–24, where the spread across weights is at
its largest, agreement only reaches 0.350 and 0.340.

This is the most direct statement of branch (b) available, because it does not
go through the learning-curve area at all: the bandit's arm choice carries no
information about which arm is actually better. The consolation is that it costs
almost nothing — per-round regret against the best arm is under 0.1% of the
error level, which is just the flat objective seen from the other side.

**Region composition** is roughly stable over the run. The high-noise trap is
5.4% (dgp_two_regime) and 7.3% (dgp_three_regime) of committed points — it is a
small region and the oracle neither chases nor avoids it strongly. There is no
dense-noisy / sparse-clean switch to point to.

---

## What this supports in the plan

Both experiments reach the same place from independent directions — Experiment A
from the reward the bandit actually sees, Experiment B from free labels — so the
conclusion does not rest on either alone.

- **Aim 1 should be framed as branch (b) for these datasets.** The adaptivity is
  not doing work here: the bandit matches a random-arm policy because it cannot
  tell its arms apart. The one-line mechanism: *the between-arm reward gap is
  about one tenth of the within-arm reward noise, so UCB1 pulls all three arms
  near-uniformly and is, in effect, the random-arm policy.*
- **"Never settles on one arm" is not evidence of useful adaptation.** It is what
  a bandit does when the arms are indistinguishable. The add-on makes this
  concrete: across 222,800 rounds the bandit's chosen arm agrees with the
  oracle-best arm at chance (0.330-0.335 against 0.3333). Any framing that reads
  continued exploration as the method responding to a changing optimum is not
  supported by this data.
- **The more useful question is the one branch (b) opens: when does weighting
  matter at all?** Experiment B answers it sharply here — only while the labelled
  set is very small (roughly the first 25 rounds, under 3% of the budget). That
  is a concrete, testable claim to build Aim 2 on, and it suggests the interesting
  regime is small budgets, not the pool-exhaustion budget the paper uses.
- **Aim 2's separation hypothesis gets weak, qualified support.** There is an
  early drift in the oracle weight (≈0.5 → ≈0.2 over ~50 rounds), but no
  sustained trend and no regime switch in which points get chosen.

## What this does not establish

Stated plainly, because it bounds every claim above:

1. **Only two datasets, and both are ones where the model is degenerate.** Ridge
   explains 14% of the variance on dgp_two_regime and ~0% on dgp_three_regime.
   On a dataset where the model cannot fit the signal, no selection rule can
   matter much, so "the objective is flat in `w`" is close to guaranteed rather
   than discovered. **The benchmarks have not been run**, and they are where
   ridge is a sensible model. Until they are, branch (b) is established for the
   synthetic sets only, and the plan should not claim it generally.
2. **The passive-learning result is not a refutation of the paper.** It is a
   different metric (held-out test error vs the paper-era full-pool error), on
   two degenerate-model datasets. The sign of the WiGS-vs-passive comparison flips
   between the two metrics on this same data, which is a metric effect and
   presumably why `test-set-eval` exists.
3. **Only `c=2.0` was run.** The paper also reports `c=0.5` and `c=5.0`. A smaller
   exploration constant would concentrate pulls more; given a gap one tenth the
   noise it would concentrate on an arm chosen largely by chance, but that is an
   inference, not a measurement.
4. **The full protocol needs the cluster.** Cost scales roughly as `T³·d` with the
   pool run to exhaustion; `wine_white` alone is ~1300× the cost of
   `dgp_two_regime`. The remaining datasets are not a local overnight job.

**Recommended next step, in priority order:** run Experiment A on the benchmark
datasets — `mpg`, `qsar`, `concrete_cs/flow/slump`, `housing`, `bodyfat` (largest
paper-era WiGS gains) and `no2`, `yacht`, `beer`, `cps_wage` (smallest or
negative) — since that is what decides whether branch (b) is a property of these
two synthetic sets or of the method. That subset is ~5 h locally; the full 21 is
a cluster job.

Per the brief's pitfall note, a branch-(b) outcome bears on the paper's central
claim and should go to Simon and Tyler the same day. **Nothing has been sent, and
nothing has been pushed or posted** — that is the authors' call, and I would send
it with caveat 1 attached rather than as a bare headline.

---

## Deliverables

| File | Contents |
|---|---|
| `exp_a_summary.csv` | per dataset × selector: mean/SE area (full and early), final error, paired diff vs random arm with bootstrap CI |
| `exp_a_vs_randomarm.csv` / `exp_a_vs_passive.csv` | every selector against each reference, both windows |
| `exp_a_reward_gap.csv` / `exp_a_reward_gap_blocked.csv` | between-arm gap vs within-arm noise, pooled and round-blocked |
| `exp_a_arms.csv.gz` | per-round arm, reward, CV and test RMSE, running arm means and counts |
| `exp_a_curves.pkl` | every per-seed test-RMSE learning curve |
| `exp_b_oracle_path.csv.gz` | per-round oracle weight, per-weight errors, gap, region |
| `exp_b_bandit_trajectory.csv.gz` | the add-on: per-w lookahead along the bandit's own trajectory, with its chosen arm vs the best arm |
| `fig_learning_curves_*.png`, `fig_paired_difference.png`, `fig_oracle_path.png` | figures |

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
   converge after roughly a quarter of the rounds, so the full-budget area is
   dominated by a flat tail common to every selector. The 25% fraction was fixed
   before looking at results and both windows are always reported.
4. **Final error at T is degenerate and is not used.** Running the pool to
   exhaustion means every selector finishes having labelled the same points, so
   the final model — and final test error — is identical across selectors by
   construction (verified: 0.558034 on dgp_two_regime for both). The brief asks
   for it; the honest answer is that it carries no signal under this budget.
5. **Fixed-weight endpoints come from `WeightedGreedySamplingSelector`, not
   GSx/GSy.** GSx and GSy score on *unnormalized* distances, so they are not the
   `w=1` and `w=0` ends of the weighted sweep; using them would have varied the
   normalization along the axis under study.
6. **CV is skipped for selectors that never read it.** Only `WiGS_MAB_Selector`
   consumes `current_rmse`; the rest swallow it in `**kwargs`, and `get_cv_rmse`
   uses a fixed `KFold(random_state=42)` that draws nothing from the global RNG.
   Verified bit-identical curves and selections either way; 2.4× faster.
7. **The bandit's reward is reconstructed, not intercepted.** `WiGS_MAB_Selector`
   keeps its reward internally and does not return it, and may not be edited. It
   is recomputed from the CV trace as `cv[t] − cv[t+1]`, which matches the
   control's own logged reward exactly (max abs difference 0.0).
8. **The two large logs are gzipped.** `exp_a_arms.csv.gz` is 9.7 MB rather than
   120 MB (it also stores one numeric column per arm instead of repr-serialised
   arrays), and `exp_b_oracle_path.csv.gz` is 11.8 MB rather than 99 MB. The
   brief names them without the extension; they are otherwise unchanged and
   `pd.read_csv` opens them directly.
9. **Not done: the `aaai2027-submission` tag.** The repo has no tags and it is
   ambiguous which commit produced the submitted numbers; tagging a shared repo's
   submitted state is the authors' call.

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
