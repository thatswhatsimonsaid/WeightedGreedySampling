QBCSelector.py
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.base import clone
from sklearn.utils import resample
from utils.Auxiliary.DataFrameUtils import get_features_and_target

class QBCSelector:
    """
    Implements Homogeneous Query-By-Committee (QBC) with Bagging.
    
    The 'committee' consists of Ridge Regression models trained on 
    BOOTSTRAP samples of the training data.
    """

    def __init__(self, n_members=10, Seed=None, **kwargs):
        """
        Args:
            n_members (int): Number of committee members (bootstrap samples).
            Seed (int): Random seed for reproducibility.
            **kwargs: Ignored arguments.
        """
        # 1. Capture 'Seed' (Capital S) to match LearningProcedure config
        self.seed = Seed
        self.rng = np.random.default_rng(Seed)
        self.n_members = n_members
        
        # 2. Define Base Model (Ridge)
        # We use Ridge because a linear model on non-linear data (like dgp_three_regime)
        # is "misspecified", which helps it lose to WiGS as intended.
        self.base_model = Ridge(alpha=1.0, random_state=Seed)

    def select(self, df_Candidate: pd.DataFrame, df_Train: pd.DataFrame, Model=None, **kwargs) -> dict:
        """
        Selects the candidate with the highest prediction variance across 
        the homogeneous committee.
        
        Args:
            df_Candidate: Pool of unlabeled data.
            df_Train: Labeled training data.
            Model: The main predictor model (Ignored here, but accepted for consistency).
        """
        if df_Candidate.empty:
            return {"IndexRecommendation": []}

        # 1. Prepare Data
        X_train_full, y_train_full = get_features_and_target(df_Train, "Y")
        X_cand, _ = get_features_and_target(df_Candidate, "Y")
        
        # 2. Train Committee on Bootstrap Samples
        predictions = [] # Shape: (n_members, n_candidates)
        
        for i in range(self.n_members):
            # Create a clean instance of Ridge
            member_model = clone(self.base_model)
            
            # Create Bootstrap Sample (Sample with replacement)
            # We vary the seed by 'i' so every member sees different data
            X_boot, y_boot = resample(
                X_train_full, 
                y_train_full, 
                replace=True, 
                random_state=self.seed + i if self.seed else None
            )
            
            # Train and Predict
            member_model.fit(X_boot, y_boot)
            preds = member_model.predict(X_cand)
            predictions.append(preds)

        # 3. Calculate Variance across Committee
        committee_preds = np.vstack(predictions)
        
        # Variance across the different bootstrap models (axis=0 is variance across models)
        prediction_variance = np.var(committee_preds, axis=0)
        
        # 4. Select Max Variance
        best_idx_loc = np.argmax(prediction_variance)
        IndexRecommendation = df_Candidate.iloc[[best_idx_loc]].index[0]

        return {"IndexRecommendation": [float(IndexRecommendation)]}