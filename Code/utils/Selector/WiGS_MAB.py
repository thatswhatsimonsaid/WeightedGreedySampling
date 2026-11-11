### Libraries ###
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from utils.Auxiliary.DataFrameUtils import get_features_and_target

class WiGS_MAB_Selector:
    """
    Implements a WiGS selector using a Multi-Armed Bandit (MAB) for weight selection. 🤖

    Attributes:
        arms (list): The list of possible `w_x` values that can be chosen.
        mab_c (float): The exploration parameter for the UCB1 algorithm.
        arm_counts (np.ndarray): The number of times each arm has been pulled.
        arm_values (np.ndarray): The running average reward for each arm.
        iteration (int): The current selection step.
        last_arm_pulled (int): The index of the arm chosen in the previous step.
        last_rmse (float): The RMSE recorded from the previous step.
        reward_scaling (str): "none" (UCB1 bonus scaled by `mab_c`) or "std"
            (bonus scaled by the running standard deviation of the rewards).
        reward_history (list): Every reward computed so far, across all arms.
        state_history (list): One dict per `select` call with the bandit's state.
    """

    ### Initialize ###
    def __init__(self,
                 mab_arms: list = None,
                 mab_c: float = 2.0,
                 reward_scaling: str = "none",
                 **kwargs):
        
        """
        Initializes the WiGS_MAB_Selector.

        Args:
            mab_arms (list, optional): A list of floats between 0 and 1 representing the possible `w_x` values 
                (ie. the "arms" of the bandit).
            mab_c (float, optional): The exploration parameter `c` for the UCB1 algorithm.
                 Higher values encourage more exploration of less-tried arms.
                 Used only when `reward_scaling="none"`.
            reward_scaling (str, optional): "none" keeps the original bonus
                `mab_c * sqrt(ln t / n_k)`, which assumes rewards in [0, 1].
                "std" replaces it with `sigma_hat * sqrt(2 ln t / n_k)`, where
                `sigma_hat` is the standard deviation of all rewards so far, so
                the bonus is on the same scale as the rewards (RMSE drops in the
                units of y). Until three rewards exist the original bonus is used.
            **kwargs: Accepts and ignores additional keyword arguments for consistency.
        """

        ## Set up MAB ##
        self.arms = mab_arms if mab_arms is not None else [0.25, 0.5, 0.75]
        self.mab_c = mab_c 
        if reward_scaling not in ("none", "std"):
            raise ValueError(f"reward_scaling must be 'none' or 'std', got {reward_scaling!r}")
        self.reward_scaling = reward_scaling
        
        # MAB state variables
        self.arm_counts = np.zeros(len(self.arms))
        self.arm_values = np.zeros(len(self.arms))
        self.iteration = 0
        self.last_arm_pulled = None
        self.last_rmse = None
        self.reward_history = []
        self.state_history = []
        
    ### Select Observation ###
    def select(self,
               df_Candidate: pd.DataFrame, 
               Model=None, 
               df_Train: pd.DataFrame = None,  
               current_rmse: float = None) -> dict:
        """
        Selects a point by first choosing a weight `w_x` via the MAB algorithm.

        Args:
            df_Candidate (pd.DataFrame): The pool of unlabeled data points.
            Model (object): A trained model with a `.predict()` method.
            df_Train (pd.DataFrame): The current set of labeled training data.
            auxiliary_columns (list, optional): Columns to exclude from features.
            current_rmse (float): The RMSE of the model *before* the current selection is made. 
                NaN means no reward is available this round (the CV RMSE was
                undefined and the loop did not substitute a fallback); the arm
                values are then left unchanged.

        Returns:
            dict: A dictionary containing the recommended point's index, in the
                format `{'IndexRecommendation': [index]}`.

        """
        
        ### Update values based on the reward from the last iteration ####
        reward = np.nan
        if self.last_arm_pulled is not None and self.last_rmse is not None:
            if current_rmse is None:
                raise ValueError("MAB strategy requires 'current_rmse' to be passed.")
            # A NaN on either side means no reward this round. The pull has
            # already been counted, so that arm's value is averaged over a count
            # that includes the unrewarded pull.
            if not (np.isnan(current_rmse) or np.isnan(self.last_rmse)):
                reward = self.last_rmse - current_rmse
                self.reward_history.append(reward)

                last_arm_idx = self.last_arm_pulled
                old_value = self.arm_values[last_arm_idx]
                new_count = self.arm_counts[last_arm_idx]
                self.arm_values[last_arm_idx] = old_value + (reward - old_value) / new_count
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

        ###Select an arm for the current step using UCB1 ###
        arm_to_pull = -1
        exploration_bonus = np.full(len(self.arms), np.nan)
        sigma_hat = np.nan
        if self.iteration < len(self.arms):
            arm_to_pull = self.iteration
        else:
            safe_arm_counts = self.arm_counts + epsilon
            if self.reward_scaling == "std" and len(self.reward_history) >= 3:
                sigma_hat = float(np.std(self.reward_history))
                exploration_bonus = sigma_hat * np.sqrt(2 * np.log(self.iteration) / safe_arm_counts)
            else:
                exploration_bonus = self.mab_c * np.sqrt(np.log(self.iteration) / safe_arm_counts)
            ucb_scores = self.arm_values + exploration_bonus
            arm_to_pull = np.argmax(ucb_scores)
        
        w_x = self.arms[arm_to_pull]
        w_y = 1.0 - w_x
        
        self.last_arm_pulled = arm_to_pull
        self.arm_counts[arm_to_pull] += 1

        ## Final Score and Selection ##
        # Every arm's candidate is computed, not only the pulled one's, so the
        # state log records what each arm would have queried this round.
        candidate_ilocs = []
        for arm_w_x in self.arms:
            score_matrix = (arm_w_x * d_prime_nmX) + ((1.0 - arm_w_x) * d_prime_nmY)
            final_scores = score_matrix.min(axis=1)
            candidate_ilocs.append(np.argmax(final_scores))
        best_candidate_iloc = candidate_ilocs[arm_to_pull]

        ## Log state ##
        self.state_history.append({
            "iteration": self.iteration,
            "arm_pulled": int(arm_to_pull),
            "w_x": w_x,
            "reward": reward,
            "current_rmse": current_rmse,
            "cv_was_nan": bool(current_rmse is not None and np.isnan(current_rmse)),
            "arm_counts": self.arm_counts.copy(),
            "arm_values": self.arm_values.copy(),
            "bonus": np.asarray(exploration_bonus, dtype=float).copy(),
            "sigma_hat": sigma_hat,
            "candidates_by_arm": [df_Candidate.index[i] for i in candidate_ilocs],
        })
        self.iteration += 1

        IndexRecommendation = df_Candidate.iloc[[best_candidate_iloc]].index[0]
        return {"IndexRecommendation": [float(IndexRecommendation)], "w_x": w_x}