### Libraries ###
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from utils.Auxiliary.DataFrameUtils import get_features_and_target


class RandomArmSelector:
    """
    Random-arm control for the WiGS MAB selector (GRFP Experiment A).

    This is a deliberate, line-for-line control on `WiGS_MAB_Selector`: the
    distance computation, the min-max normalization, the weighted score
    `s_n = min_m ( w * phi(d_x) + (1 - w) * phi(d_y) )` and the argmax over the
    candidate pool are all identical. The ONLY difference is the arm-selection
    rule -- UCB1 is replaced by a uniform draw over the same arm set.

    Reward bookkeeping is kept (and logged) even though it does not influence
    the choice of arm. Under the uniform policy the per-arm reward samples are
    an unconfounded estimate of each arm's reward distribution, which is what
    the between-arm gap / within-arm noise ratio is computed from.

    Attributes:
        arms (list): The list of possible `w_x` values, same set as the bandit.
        rng (np.random.Generator): Seeded generator for the arm draws.
        arm_counts (np.ndarray): Number of times each arm has been pulled.
        arm_values (np.ndarray): Running average reward for each arm.
        reward_log (list): Per-round (arm_index, reward) pairs.
    """

    ### Initialize ###
    def __init__(self,
                 mab_arms: list = None,
                 Seed: int = None,
                 **kwargs):
        """
        Initializes the RandomArmSelector.

        Args:
            mab_arms (list, optional): Candidate `w_x` values. Defaults to the
                bandit's own default arm set, [0.25, 0.5, 0.75].
            Seed (int, optional): Seed for the arm draws. A dedicated Generator
                is used so the draws never perturb the global RNG stream that
                the train/candidate/test split consumes.
            **kwargs: Accepts and ignores additional keyword arguments for consistency.
        """

        ## Same arm set and default as WiGS_MAB_Selector ##
        self.arms = mab_arms if mab_arms is not None else [0.25, 0.5, 0.75]

        ## Dedicated stream: offset so it is not correlated with the split draw ##
        self.rng = np.random.default_rng(None if Seed is None else Seed + 100_000)

        # State variables (mirrors the bandit; used for logging only) #
        self.arm_counts = np.zeros(len(self.arms))
        self.arm_values = np.zeros(len(self.arms))
        self.iteration = 0
        self.last_arm_pulled = None
        self.last_rmse = None
        self.reward_log = []

    ### Select Observation ###
    def select(self,
               df_Candidate: pd.DataFrame,
               Model=None,
               df_Train: pd.DataFrame = None,
               current_rmse: float = None) -> dict:
        """
        Selects a point after drawing the weight `w_x` uniformly at random.

        Args:
            df_Candidate (pd.DataFrame): The pool of unlabeled data points.
            Model (object): A trained model with a `.predict()` method.
            df_Train (pd.DataFrame): The current set of labeled training data.
            current_rmse (float): The CV RMSE of the model *before* this selection.

        Returns:
            dict: `{'IndexRecommendation': [index], 'w_x': w, 'arm_index': i, 'reward': r}`.
        """

        ### Update values based on the reward from the last iteration ###
        reward = np.nan
        if self.last_arm_pulled is not None and self.last_rmse is not None:
            if current_rmse is None:
                raise ValueError("RandomArm control requires 'current_rmse' to be passed.")
            reward = self.last_rmse - current_rmse

            last_arm_idx = self.last_arm_pulled
            old_value = self.arm_values[last_arm_idx]
            new_count = self.arm_counts[last_arm_idx]
            self.arm_values[last_arm_idx] = old_value + (reward - old_value) / new_count
            self.reward_log.append((int(last_arm_idx), float(reward)))
        self.last_rmse = current_rmse

        ### Calculate distances and normalize ###
        if df_Candidate.empty:
            return {"IndexRecommendation": []}

        X_Candidate, _ = get_features_and_target(df=df_Candidate, target_column_name="Y")
        X_Train, y_Train = get_features_and_target(df=df_Train, target_column_name="Y")
        d_nmX = cdist(X_Candidate.values, X_Train.values, metric='euclidean')
        Predictions = Model.predict(X_Candidate)
        d_nmY = cdist(Predictions.reshape(-1, 1), y_Train.values.reshape(-1, 1), metric='euclidean')

        epsilon = 1e-8
        d_prime_nmX = (d_nmX - d_nmX.min()) / (d_nmX.max() - d_nmX.min() + epsilon)
        d_prime_nmY = (d_nmY - d_nmY.min()) / (d_nmY.max() - d_nmY.min() + epsilon)

        ### Select an arm uniformly at random (the ONLY departure from UCB1) ###
        arm_to_pull = int(self.rng.integers(0, len(self.arms)))

        w_x = self.arms[arm_to_pull]
        w_y = 1.0 - w_x

        self.last_arm_pulled = arm_to_pull
        self.arm_counts[arm_to_pull] += 1

        ## Final Score and Selection ##
        score_matrix = (w_x * d_prime_nmX) + (w_y * d_prime_nmY)
        final_scores = score_matrix.min(axis=1)
        best_candidate_iloc = np.argmax(final_scores)
        self.iteration += 1

        IndexRecommendation = df_Candidate.iloc[[best_candidate_iloc]].index[0]
        return {"IndexRecommendation": [float(IndexRecommendation)],
                "w_x": w_x,
                "arm_index": arm_to_pull,
                "reward": float(reward)}
