"""
Instrumented learning loop for the GRFP control experiments (Experiment A).

Mirrors `Code/utils/Main/LearningProcedure.py` step for step, with two
additions and no substitutions:

  * the per-round metric is test RMSE on a held-out set (the paper's metric),
    instead of troy-branch's full-pool error;
  * the arm index, the per-round reward, and each arm's running empirical mean
    are logged every round.

The reward handed to the selector is unchanged: the per-step drop in 5-fold CV
RMSE on the labeled set, from the repo's own `get_cv_rmse`. troy-branch's
fallback when CV is undefined (fewer than 2k labeled points) is kept as-is --
it uses *train* RMSE, so no test information ever reaches the reward.
"""

### Import functions ###
from utils.GRFPControls import _bootstrap  # noqa: F401  (installs optional-dep stubs)

import inspect
import random

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from utils.Auxiliary.LoadDataSet import LoadData
from utils.Auxiliary.DataFrameUtils import get_features_and_target
from utils.Prediction.CrossValidation import get_cv_rmse
from utils.Prediction.RidgeRegressionPredictor import RidgeRegressionPredictor
from utils.Selector.GreedySamplingSelector import GreedySamplingSelector
from utils.Selector.PassiveLearningSelector import PassiveLearningSelector
from utils.Selector.WeightedGreedySamplingSelector import WeightedGreedySamplingSelector
from utils.Selector.WiGS_MAB import WiGS_MAB_Selector

from utils.GRFPControls.RandomArmSelector import RandomArmSelector
from utils.GRFPControls.ControlProtocol import (
    ControlTrainCandidateSplit,
    PaperTrainCandidateSplit,
    ControlTestSetError,
    standardize_features,
)

### Registry ###
_MODELS = {"RidgeRegressionPredictor": RidgeRegressionPredictor}
_SELECTORS = {
    "GreedySamplingSelector": GreedySamplingSelector,
    "PassiveLearningSelector": PassiveLearningSelector,
    "WeightedGreedySamplingSelector": WeightedGreedySamplingSelector,
    "WiGS_MAB_Selector": WiGS_MAB_Selector,
    "RandomArmSelector": RandomArmSelector,
}

### Selectors that actually consume the reward signal ###
# Only WiGS_MAB_Selector (and its RandomArm control) read `current_rmse`; every
# other selector swallows it in **kwargs. `get_cv_rmse` uses a fixed
# KFold(random_state=42) and so draws nothing from the global RNG stream, which
# means skipping it for the reward-free selectors cannot change their splits,
# their choices or their curves -- it only avoids 5 wasted refits per round.
# `verify_cv_skip.py` asserts the curves are bit-identical either way.
_REWARD_CONSUMERS = {"WiGS_MAB_Selector", "RandomArmSelector"}


def RunControlIteration(DataFileInput,
                        Seed,
                        SelectorType,
                        selector_params=None,
                        CandidateProportion=0.99,
                        ModelType="RidgeRegressionPredictor",
                        regularization=0.01,
                        MaxIterations=None,
                        df=None,
                        ComputeCV=None,
                        Protocol="repo",
                        TrainProportion=0.05,
                        Standardize=False):
    """
    Runs one (dataset, seed, selector) control run and returns its logs.

    The seeding and split order replicate OneIterationFunction exactly --
    `random.seed` / `np.random.seed`, then load, then split -- so that for a
    given seed every selector sees the identical test set, initial labeled set
    and candidate pool.

    Args:
        DataFileInput (str): Dataset name under Data/processed.
        Seed (int): Replication seed.
        SelectorType (str): Key into the selector registry.
        selector_params (dict, optional): Extra selector kwargs.
        CandidateProportion (float): Paper's value is 0.99.
        ModelType (str): Paper's value is RidgeRegressionPredictor.
        regularization (float): Paper's value is 0.01.
        MaxIterations (int, optional): Cap on rounds. None = run the pool to
            exhaustion, which is the paper's budget.
        df (pd.DataFrame, optional): Pre-loaded data, used by Experiment B.

    Returns:
        dict with the test-metric curves and the per-round arm/reward logs.
    """
    selector_params = dict(selector_params or {})

    ### Set Up (identical order to OneIterationFunction) ###
    random.seed(Seed)
    np.random.seed(Seed)

    ### Load Data ###
    if df is None:
        df = LoadData(DataFileInput)

    ### Feature scaling ###
    # Deterministic, so it does not disturb the RNG stream the split consumes.
    if Standardize:
        df = standardize_features(df)

    ### Three-way split ###
    if Protocol == "paper":
        df_Train, df_Candidate, df_Test = PaperTrainCandidateSplit(
            df, TrainProportion=TrainProportion)
    else:
        df_Train, df_Candidate, df_Test = ControlTrainCandidateSplit(df, CandidateProportion)
    InitialTrainIndices = list(df_Train.index)

    ### Initialize Model ###
    ModelClass = _MODELS[ModelType]
    model_cfg = {"regularization": regularization}
    model_init_args = {k: v for k, v in model_cfg.items()
                       if k in inspect.signature(ModelClass.__init__).parameters}
    predictor_model = ModelClass(**model_init_args)

    ### Initialize Selector ###
    SelectorClass = _SELECTORS[SelectorType]
    sel_cfg = dict(selector_params)
    sel_cfg.setdefault("Seed", Seed)
    sel_cfg.setdefault("initial_candidate_size", len(df_Candidate))
    sig = inspect.signature(SelectorClass.__init__)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        selector_init_args = sel_cfg
    else:
        selector_init_args = {k: v for k, v in sel_cfg.items() if k in sig.parameters}
    selector_model = SelectorClass(**selector_init_args)
    needs_reward = SelectorType in _REWARD_CONSUMERS if ComputeCV is None else ComputeCV

    ### Logs ###
    ErrorVecs = {'RMSE': [], 'MAE': [], 'R2': [], 'CC': []}
    ArmLog = []
    SelectedObservationHistory = []
    cv_history = []

    ### Algorithm ###
    i = 0
    while True:

        ## 1. Features / target of current training set ##
        X_train_df, y_train_series = get_features_and_target(df=df_Train, target_column_name="Y")

        ## 2. Fit ##
        predictor_model.fit(X_train_df=X_train_df, y_train_series=y_train_series)

        ## 3. Test-set error (the paper's metric) ##
        TestOutputs = ControlTestSetError(InputModel=predictor_model, df_Test=df_Test)
        for metric_name, value in TestOutputs.items():
            ErrorVecs[metric_name].append(value)

        ## 4. CV error (the bandit's reward signal -- unchanged) ##
        if needs_reward:
            sklearn_model = predictor_model.model
            if sklearn_model is not None:
                current_cv_rmse = get_cv_rmse(sklearn_model, X_train_df, y_train_series, k=5)
            else:
                current_cv_rmse = np.nan
            if np.isnan(current_cv_rmse):
                # troy-branch fallback: train RMSE, never test #
                train_predictions = predictor_model.predict(X_train_df)
                current_cv_rmse = np.sqrt(mean_squared_error(y_train_series, train_predictions))
        else:
            current_cv_rmse = np.nan
        cv_history.append(current_cv_rmse)

        ## 5. Break conditions ##
        if len(df_Candidate) == 0:
            break
        if MaxIterations is not None and i >= MaxIterations:
            break

        ## 6. Sampling ##
        SelectorFuncOutput = selector_model.select(df_Candidate=df_Candidate,
                                                   df_Train=df_Train,
                                                   Model=predictor_model,
                                                   current_rmse=current_cv_rmse)

        ## 7. Query ##
        QueryObservationIndex = SelectorFuncOutput["IndexRecommendation"]
        if len(QueryObservationIndex) == 0:
            break
        QueryObservation = df_Candidate.loc[QueryObservationIndex]
        SelectedObservationHistory.append(QueryObservationIndex[0])

        ## 8. Arm / reward instrumentation ##
        arm_values = getattr(selector_model, "arm_values", None)
        arm_counts = getattr(selector_model, "arm_counts", None)
        ArmLog.append({
            "round": i,
            "w_x": SelectorFuncOutput.get("w_x", np.nan),
            "arm_index": SelectorFuncOutput.get("arm_index",
                                                getattr(selector_model, "last_arm_pulled", np.nan)),
            "reward": SelectorFuncOutput.get("reward", np.nan),
            "cv_rmse": current_cv_rmse,
            "test_rmse": TestOutputs["RMSE"],
            "arm_values": None if arm_values is None else arm_values.copy(),
            "arm_counts": None if arm_counts is None else arm_counts.copy(),
        })

        ## 9. Update sets ##
        df_Train = pd.concat([df_Train, QueryObservation])
        df_Candidate = df_Candidate.drop(QueryObservationIndex)

        ## 10. Next round ##
        i += 1

    ### Derive the realized reward for every logged round ###
    # WiGS_MAB_Selector keeps its reward internally and does not return it, and
    # the brief forbids editing that file. The reward it uses is exactly
    # `last_cv_rmse - current_cv_rmse`, so it is reconstructed here from the CV
    # trace and attributed to the arm that was pulled that round. For
    # RandomArmSelector, which does return its reward, the two agree exactly --
    # asserted in the verification script.
    for t, row in enumerate(ArmLog):
        if t + 1 < len(cv_history):
            row["reward_derived"] = float(cv_history[t] - cv_history[t + 1])
        else:
            row["reward_derived"] = np.nan

    return {
        "ErrorVecs": ErrorVecs,
        "ArmLog": ArmLog,
        "cv_history": cv_history,
        "SelectedObservationHistory": SelectedObservationHistory,
        "InitialTrainIndices": InitialTrainIndices,
        "n_test": len(df_Test),
        "n_initial_train": len(InitialTrainIndices),
        "n_rounds": len(ErrorVecs["RMSE"]),
    }
