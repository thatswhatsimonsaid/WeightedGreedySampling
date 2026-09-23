"""
Protocol pieces for the GRFP control experiments.

troy-branch evaluates on the full pool (train + candidate). The GRFP brief
requires the paper's *test-error* learning curve, which lives on the
`test-set-eval` branch. Rather than edit troy-branch's evaluation code (which
the brief forbids), the two needed pieces are reproduced here verbatim as new
code: the 3-way split and the held-out test-set scoring.

Both functions are byte-equivalent in behaviour to
`Code/utils/Main/TrainCandidateSplit.py` and
`Code/utils/Prediction/TestSetError.py` as of test-set-eval @ 478e2648.
"""

### Libraries ###
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from utils.Auxiliary.DataFrameUtils import get_features_and_target


### Three-way split (test / train / candidate) ###
def ControlTrainCandidateSplit(df, CandidateProportion, TestProportion=0.10):
    """
    Splits df into THREE sets: test, initial training, and candidate.

    Reproduces test-set-eval's TrainCandidateSplit exactly. Like the original it
    passes no `random_state`, so it draws from the global NumPy stream that the
    caller seeds -- which is what makes the split identical across selectors for
    a given seed, and hence what makes the paired comparison valid.

    Args:
        df: The original dataframe.
        CandidateProportion: Proportion of the LEARNING data held as candidates.
        TestProportion: Proportion of the TOTAL data held out for testing.

    Returns:
        (df_Train, df_Candidate, df_Test)
    """

    # 1. Hold-out Test Split FIRST #
    df_LearningPool, df_Test = train_test_split(df, test_size=TestProportion)

    # 2. Calculate Safe Train Size #
    n_learning_samples = len(df_LearningPool)
    n_train_samples = max(2, int(n_learning_samples * (1.0 - CandidateProportion)))

    # 3. Train/Candidate Split on the REMAINING data #
    X_Train, X_Candidate, y_Train, y_Candidate = train_test_split(
        df_LearningPool.loc[:, df_LearningPool.columns != "Y"],
        df_LearningPool["Y"],
        train_size=n_train_samples
    )

    # Column names #
    df_Train = X_Train.copy()
    df_Train.insert(0, 'Y', y_Train)

    df_Candidate = X_Candidate.copy()
    df_Candidate.insert(0, 'Y', y_Candidate)

    return df_Train, df_Candidate, df_Test


### The manuscript's stated split ###
def PaperTrainCandidateSplit(df, TrainProportion=0.05, TestProportion=0.10):
    """
    Split as the manuscript's Experimental Protocol describes it: a held-out test
    set (10% of the data), an initial training set (5% of the data), and a
    candidate pool (the remaining 85%).

    This is NOT what `CandidateProportion=0.99` in the repo's cluster config
    produces. That setting makes the initial training set 1% of the *learning
    pool* — 9 points for N=1000, against the 50 the manuscript specifies. The
    equivalent CandidateProportion for the manuscript's split is 0.9444.

    Like the other split functions it passes no `random_state`, drawing from the
    global NumPy stream the caller seeds, so a given seed yields the identical
    split for every selector.
    """
    df_LearningPool, df_Test = train_test_split(df, test_size=TestProportion)
    n_train = max(2, int(round(len(df) * TrainProportion)))
    n_train = min(n_train, len(df_LearningPool) - 1)

    X_Train, X_Candidate, y_Train, y_Candidate = train_test_split(
        df_LearningPool.loc[:, df_LearningPool.columns != "Y"],
        df_LearningPool["Y"],
        train_size=n_train
    )

    df_Train = X_Train.copy()
    df_Train.insert(0, 'Y', y_Train)
    df_Candidate = X_Candidate.copy()
    df_Candidate.insert(0, 'Y', y_Candidate)
    return df_Train, df_Candidate, df_Test


### The manuscript's stated feature scaling ###
def standardize_features(df):
    """
    Scale features to zero mean and unit variance, as the manuscript's main
    results specify.

    `Data/processed` was built with RobustScaler (StandardScaler is commented out
    in PreprocessData.py), so the stored features are not the manuscript's main
    preprocessing. Re-standardizing recovers it exactly: both scalers are
    per-feature affine maps, and standardizing removes any prior affine scaling,
    so StandardScaler(RobustScaler(x)) == StandardScaler(x). Verified to 1e-12.
    """
    from sklearn.preprocessing import StandardScaler
    out = df.copy()
    feats = [c for c in out.columns if c != "Y"]
    out[feats] = StandardScaler().fit_transform(out[feats])
    return out


### Held-out test set scoring ###
def ControlTestSetError(InputModel, df_Test: pd.DataFrame) -> dict:
    """
    Calculates performance metrics on the held-out test set.

    Reproduces test-set-eval's TestSetErrorFunction exactly.

    Returns:
        dict: {'RMSE', 'MAE', 'R2', 'CC'}
    """
    X_test, y_true = get_features_and_target(df_Test, "Y")
    y_pred = InputModel.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    if np.std(y_pred) > 0 and np.std(y_true) > 0:
        cc = np.corrcoef(y_true, y_pred)[0, 1]
    else:
        cc = 1.0

    return {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'CC': cc}


### Learning-curve area ###
def LearningCurveArea(curve):
    """
    Trapezoidal area under a learning curve, integrated over round index.

    Matches GenerateAUCTable.py's `np.trapezoid(y, x)` with x = 0..T-1. The one
    difference is scope, and it is deliberate: the paper's table averages over
    seeds and then integrates, whereas the paired analysis here integrates each
    seed's own curve so that seeds can be paired. Integration and averaging
    commute for the mean, so the reported means agree; only the per-seed spread
    is newly available.
    """
    y = np.asarray(curve, dtype=float)
    if np.isnan(y).any():
        y = pd.Series(y).interpolate().bfill().ffill().values
    x = np.arange(len(y))
    trapz = getattr(np, 'trapezoid', None) or np.trapz
    return float(trapz(y, x))
