"""Are WiGS w=1 / w=0 the same selectors as GSx / GSy?"""
import os, sys, random
sys.path.insert(0, os.getcwd())
from utils.GRFPControls import _bootstrap  # noqa
import numpy as np
from utils.Auxiliary.LoadDataSet import LoadData
from utils.Auxiliary.DataFrameUtils import get_features_and_target
from utils.Prediction.RidgeRegressionPredictor import RidgeRegressionPredictor
from utils.GRFPControls.ControlProtocol import ControlTrainCandidateSplit
from utils.Selector.GreedySamplingSelector import GreedySamplingSelector
from utils.Selector.WeightedGreedySamplingSelector import WeightedGreedySamplingSelector

print("=== Does WiGS(w=1) == GSx and WiGS(w=0) == GSy ? ===")
for ds in ["dgp_two_regime", "housing", "mpg", "concrete_cs"]:
    agree_x = agree_y = n = 0
    for seed in range(5):
        random.seed(seed); np.random.seed(seed)
        df = LoadData(ds)
        tr, ca, te = ControlTrainCandidateSplit(df, 0.99)
        m = RidgeRegressionPredictor(regularization=0.01)
        for step in range(25):
            X, y = get_features_and_target(df=tr, target_column_name="Y")
            m.fit(X_train_df=X, y_train_series=y)
            gx = GreedySamplingSelector(strategy="GSx").select(df_Candidate=ca, df_Train=tr, Model=m)
            gy = GreedySamplingSelector(strategy="GSy").select(df_Candidate=ca, df_Train=tr, Model=m)
            w1 = WeightedGreedySamplingSelector(weight_strategy="static", w_x=1.0).select(df_Candidate=ca, df_Train=tr, Model=m)
            w0 = WeightedGreedySamplingSelector(weight_strategy="static", w_x=0.0).select(df_Candidate=ca, df_Train=tr, Model=m)
            agree_x += int(float(gx["IndexRecommendation"][0]) == float(w1["IndexRecommendation"][0]))
            agree_y += int(float(gy["IndexRecommendation"][0]) == float(w0["IndexRecommendation"][0]))
            n += 1
            idx = w1["IndexRecommendation"]
            import pandas as pd
            tr = pd.concat([tr, ca.loc[idx]]); ca = ca.drop(idx)
    print(f"  {ds:<16} GSx==w1: {agree_x}/{n}   GSy==w0: {agree_y}/{n}")
