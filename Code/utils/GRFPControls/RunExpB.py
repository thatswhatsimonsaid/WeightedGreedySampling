"""
Experiment B -- oracle weight path on the synthetic data.

Labels are free on the synthetic generator, so at each round the best weight can
be computed rather than estimated. For every w on a grid finer than the paper's
arm set, we take the point that w would select, add it with its true label,
refit, and score on a LARGE test set drawn fresh from the generator (not CV, and
not the 10% split). The oracle arm is the argmin; that choice is committed and
the process continues.

Held-out set construction. The processed .pkl is
`_preprocess_dataframe(generator(seed=42))`, i.e. features passed through a
RobustScaler fitted on those samples (Y is left unscaled). The oracle test set
is therefore drawn from the generator with a different seed and pushed through
the scaler fitted on the ORIGINAL sample -- never refitted on the new draw --
so the two live in the same feature space. `verify_expb_scaling()` asserts that
this reconstruction reproduces the stored .pkl exactly.

Usage (from the Code/ directory):
    python -m utils.GRFPControls.RunExpB --datasets dgp_two_regime --seeds 100 --workers 7
"""

from utils.GRFPControls import _bootstrap  # noqa: F401

import argparse
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error

from utils.Auxiliary.LoadDataSet import LoadData
from utils.Auxiliary.DataFrameUtils import get_features_and_target
from utils.Auxiliary.PreprocessData import (
    generate_two_regime_data,
    generate_three_regime_data,
)
from utils.Prediction.RidgeRegressionPredictor import RidgeRegressionPredictor
from utils.Prediction.CrossValidation import get_cv_rmse
from utils.Selector.WiGS_MAB import WiGS_MAB_Selector
from utils.GRFPControls.ControlProtocol import ControlTrainCandidateSplit

### Generator registry: (function, n in the stored pkl) ###
GENERATORS = {
    "dgp_two_regime": (generate_two_regime_data, 1000),
    "dgp_three_regime": (generate_three_regime_data, 1500),
}

ORIGINAL_SEED = 42          # seed used to build the stored .pkl
ORACLE_TEST_SEED = 987_654  # dedicated, disjoint from the replication seeds
WEIGHT_GRID = [float(round(w, 2)) for w in np.arange(0.0, 1.01, 0.1)]

### The bandit's own arm set (WiGS_MAB_Selector's default) ###
# The grid above does not contain 0.25 or 0.75, and the oracle's argmin often
# sits at endpoints the bandit structurally cannot reach. Comparing the two
# directly would therefore understate the bandit. These arms are scored as well,
# so the oracle can also be reported restricted to the arms actually available.
MAB_ARMS = [0.25, 0.5, 0.75]
EVAL_WEIGHTS = sorted(set(WEIGHT_GRID) | set(MAB_ARMS))


### Region labels, defined on the RAW (unscaled) x, straight from the generator ###
def region_of(dataset, x_raw):
    """Generator regime containing a raw x value."""
    if dataset == "dgp_two_regime":
        if x_raw < 0.5:
            return "sine_lownoise"      # complex signal, sigma=0.1
        if 0.8 < x_raw < 0.9:
            return "noise_trap"         # sigma=1.0, in the denser half
        return "linear_lownoise"        # simple signal, sigma=0.1
    if dataset == "dgp_three_regime":
        if x_raw < 0.4:
            return "sine_lownoise"      # sigma=0.1
        if 0.6 < x_raw < 0.65:
            return "noise_trap"         # sigma=1.5
        if x_raw < 0.7:
            return "linear_lownoise"    # sigma=0.1
        return "cosine_lownoise"        # sigma=0.15
    return "unknown"


def build_oracle_test_set(dataset, n_oracle_test):
    """
    Large held-out set drawn from the generator, mapped into the stored
    dataset's feature space with the scaler fitted on the ORIGINAL sample.

    Returns (X_test_scaled, y_test, scaler).
    """
    gen, n_orig = GENERATORS[dataset]
    raw_orig = gen(n_samples=n_orig, seed=ORIGINAL_SEED)
    scaler = RobustScaler().fit(raw_orig[["X1"]])

    raw_test = gen(n_samples=n_oracle_test, seed=ORACLE_TEST_SEED)
    # Keep it a named-column frame: the models are fitted on DataFrames, and
    # predicting on a bare array otherwise trips a feature-name warning.
    X_test = pd.DataFrame(scaler.transform(raw_test[["X1"]]), columns=["X1"])
    y_test = raw_test["Y"].values
    return X_test, y_test, scaler


def verify_expb_scaling():
    """Assert the generator + scaler reconstruction reproduces the stored pkl."""
    ok = {}
    for name, (gen, n_orig) in GENERATORS.items():
        pkl = LoadData(name)
        raw = gen(n_samples=n_orig, seed=ORIGINAL_SEED)
        sc = RobustScaler().fit(raw[["X1"]])
        ok[name] = (np.allclose(pkl["Y"].values, raw["Y"].values) and
                    np.allclose(pkl["X1"].values, sc.transform(raw[["X1"]]).ravel()))
    return ok


def run_seed(dataset, seed, n_oracle_test=20000, max_iter=None,
             CandidateProportion=0.99, regularization=0.01, policy="oracle"):
    """
    One seed of the weight path.

    policy="oracle" (Experiment B proper): commit the argmin weight each round.

    policy="bandit" (the brief's optional add-on): record the same per-w
    lookahead, but commit the point the UCB1 bandit actually chooses, so the
    realized arm can be compared against the oracle arm on the same trajectory.
    The bandit is driven exactly as in the main pipeline -- its reward is the
    per-step drop in 5-fold CV RMSE, and the selector object is used unmodified.

    Returns a list of per-round records.
    """
    ### Oracle test set FIRST, so the split's RNG stream is untouched ###
    X_oracle, y_oracle, scaler = build_oracle_test_set(dataset, n_oracle_test)

    ### Seed + split, identical order and values to Experiment A ###
    random.seed(seed)
    np.random.seed(seed)
    df = LoadData(dataset)
    df_Train, df_Candidate, _df_Test = ControlTrainCandidateSplit(df, CandidateProportion)

    model = RidgeRegressionPredictor(regularization=regularization)

    bandit = None
    if policy == "bandit":
        bandit = WiGS_MAB_Selector(mab_c=2.0)

    rows = []
    t = 0
    while len(df_Candidate) > 0:
        if max_iter is not None and t >= max_iter:
            break

        ## Fit on the current labeled set ##
        X_train_df, y_train_series = get_features_and_target(df=df_Train, target_column_name="Y")
        model.fit(X_train_df=X_train_df, y_train_series=y_train_series)

        ## Distances + normalization: computed ONCE, shared by every w ##
        X_Candidate, _ = get_features_and_target(df=df_Candidate, target_column_name="Y")
        d_nmX = cdist(X_Candidate.values, X_train_df.values, metric='euclidean')
        Predictions = model.predict(X_Candidate)
        d_nmY = cdist(Predictions.reshape(-1, 1), y_train_series.values.reshape(-1, 1),
                      metric='euclidean')
        epsilon = 1e-8
        dX = (d_nmX - d_nmX.min()) / (d_nmX.max() - d_nmX.min() + epsilon)
        dY = (d_nmY - d_nmY.min()) / (d_nmY.max() - d_nmY.min() + epsilon)

        ## For each w: which point, and what error would follow ##
        picks, errs = {}, {}
        cache = {}  # point index -> post-add error (many w pick the same point)
        for w in EVAL_WEIGHTS:
            scores = ((w * dX) + ((1.0 - w) * dY)).min(axis=1)
            iloc = int(np.argmax(scores))
            idx = df_Candidate.index[iloc]
            picks[w] = idx
            if idx not in cache:
                trial = pd.concat([df_Train, df_Candidate.loc[[idx]]])
                Xt, yt = get_features_and_target(df=trial, target_column_name="Y")
                trial_model = RidgeRegressionPredictor(regularization=regularization)
                trial_model.fit(X_train_df=Xt, y_train_series=yt)
                pred = trial_model.model.predict(X_oracle)
                cache[idx] = float(np.sqrt(mean_squared_error(y_oracle, pred)))
            errs[w] = cache[idx]

        ## Oracle arm = argmin; gap = spread across w ##
        err_arr = np.array([errs[w] for w in WEIGHT_GRID])
        best_i = int(np.argmin(err_arr))
        oracle_w = WEIGHT_GRID[best_i]
        gap = float(err_arr.max() - err_arr.min())

        # Same thing restricted to the arms the bandit can actually pull #
        arm_err = np.array([errs[w] for w in MAB_ARMS])
        oracle_w_armset = MAB_ARMS[int(np.argmin(arm_err))]
        gap_armset = float(arm_err.max() - arm_err.min())

        ## Which point gets committed depends on the policy ##
        bandit_w = np.nan
        if policy == "bandit":
            sk = model.model
            cv = get_cv_rmse(sk, X_train_df, y_train_series, k=5) if sk is not None else np.nan
            if np.isnan(cv):
                cv = float(np.sqrt(mean_squared_error(
                    y_train_series, model.predict(X_train_df))))
            sel_out = bandit.select(df_Candidate=df_Candidate, df_Train=df_Train,
                                    Model=model, current_rmse=cv)
            chosen_idx = sel_out["IndexRecommendation"][0]
            bandit_w = sel_out.get("w_x", np.nan)
        else:
            chosen_idx = picks[oracle_w]

        x_raw = float(scaler.inverse_transform(
            df_Candidate.loc[[chosen_idx], ["X1"]].values)[0, 0])

        row = {"dataset": dataset, "seed": seed, "round": t, "policy": policy,
               "oracle_w": oracle_w, "bandit_w": bandit_w,
               "oracle_w_armset": oracle_w_armset, "gap": gap, "gap_armset": gap_armset,
               "err_min": float(err_arr.min()), "err_max": float(err_arr.max()),
               "err_mean": float(err_arr.mean()), "err_sd": float(err_arr.std(ddof=0)),
               "n_distinct_points": len(cache),
               "region": region_of(dataset, x_raw), "x_raw": x_raw,
               "n_train": len(df_Train)}
        for w in EVAL_WEIGHTS:
            row[f"err_w{w:.2f}"] = errs[w]
        rows.append(row)

        ## Commit the oracle choice and continue ##
        df_Train = pd.concat([df_Train, df_Candidate.loc[[chosen_idx]]])
        df_Candidate = df_Candidate.drop([chosen_idx])
        t += 1

    return rows


def _one(job):
    dataset, seed, n_oracle_test, max_iter, policy = job
    t0 = time.time()
    rows = run_seed(dataset, seed, n_oracle_test=n_oracle_test, max_iter=max_iter,
                    policy=policy)
    return rows, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["dgp_two_regime", "dgp_three_regime"])
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--oracle-test-n", type=int, default=20000)
    ap.add_argument("--max-iter", type=int, default=None)
    ap.add_argument("--policy", choices=["oracle", "bandit"], default="oracle",
                    help="oracle = Experiment B proper; bandit = the optional "
                         "add-on that scores the same grid along the bandit's "
                         "own realized trajectory")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--filename", type=str, default=None)
    args = ap.parse_args()

    checks = verify_expb_scaling()
    print("Scaling reconstruction check:", checks)
    if not all(checks.values()):
        raise SystemExit("Oracle test set would not share the dataset's feature space. Aborting.")

    out_dir = args.out or os.path.join(os.path.dirname(os.getcwd()), "results_grfp_controls")
    os.makedirs(out_dir, exist_ok=True)

    jobs = [(d, s, args.oracle_test_n, args.max_iter, args.policy)
            for d in args.datasets for s in range(args.seeds)]
    print(f"Experiment B [{args.policy}]: {len(jobs)} seed-runs, grid={WEIGHT_GRID}, "
          f"oracle test n={args.oracle_test_n}, {args.workers} workers")

    all_rows = []
    t_start = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_one, j) for j in jobs]
        for f in as_completed(futs):
            rows, el = f.result()
            all_rows.extend(rows)
            done += 1
            if done % 5 == 0 or done == len(jobs):
                tot = time.time() - t_start
                print(f"  {done}/{len(jobs)}  elapsed={tot/60:.1f}m "
                      f"eta={(len(jobs)-done)/(done/tot)/60:.1f}m", flush=True)

    df_out = pd.DataFrame(all_rows).sort_values(["dataset", "seed", "round"])
    default_name = ("exp_b_oracle_path.csv" if args.policy == "oracle"
                    else "exp_b_bandit_trajectory.csv")
    path = os.path.join(out_dir, args.filename or default_name)
    df_out.to_csv(path, index=False)
    print(f"\nWrote {path} ({len(df_out)} rows)")
    print(f"Total wall time: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
