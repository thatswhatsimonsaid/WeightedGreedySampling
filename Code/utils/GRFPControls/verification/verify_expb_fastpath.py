"""Reference (slow, pandas) implementation vs the optimized fast path."""
import os, sys, time, random
sys.path.insert(0, os.getcwd())
from utils.GRFPControls import _bootstrap  # noqa
import numpy as np, pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import mean_squared_error
from utils.Auxiliary.LoadDataSet import LoadData
from utils.Auxiliary.DataFrameUtils import get_features_and_target
from utils.Prediction.RidgeRegressionPredictor import RidgeRegressionPredictor
from utils.GRFPControls.ControlProtocol import ControlTrainCandidateSplit
from utils.GRFPControls.RunExpB import build_oracle_test_set, EVAL_WEIGHTS, WEIGHT_GRID, run_seed

def reference(dataset, seed, n_oracle_test=20000, max_iter=40, reg=0.01):
    X_oracle, y_oracle, scaler = build_oracle_test_set(dataset, n_oracle_test)
    random.seed(seed); np.random.seed(seed)
    df = LoadData(dataset)
    df_Train, df_Candidate, _ = ControlTrainCandidateSplit(df, 0.99)
    model = RidgeRegressionPredictor(regularization=reg)
    out = []
    for t in range(max_iter):
        Xtr, ytr = get_features_and_target(df=df_Train, target_column_name="Y")
        model.fit(X_train_df=Xtr, y_train_series=ytr)
        Xc, _ = get_features_and_target(df=df_Candidate, target_column_name="Y")
        dX_ = cdist(Xc.values, Xtr.values); pr = model.predict(Xc)
        dY_ = cdist(pr.reshape(-1,1), ytr.values.reshape(-1,1))
        eps=1e-8
        dX=(dX_-dX_.min())/(dX_.max()-dX_.min()+eps); dY=(dY_-dY_.min())/(dY_.max()-dY_.min()+eps)
        errs={}; cache={}; picks={}
        for w in EVAL_WEIGHTS:
            sc=((w*dX)+((1.0-w)*dY)).min(axis=1); il=int(np.argmax(sc))
            idx=df_Candidate.index[il]; picks[w]=idx
            if idx not in cache:
                trial=pd.concat([df_Train, df_Candidate.loc[[idx]]])
                Xt,yt=get_features_and_target(df=trial, target_column_name="Y")
                tm=RidgeRegressionPredictor(regularization=reg); tm.fit(X_train_df=Xt,y_train_series=yt)
                cache[idx]=float(np.sqrt(mean_squared_error(y_oracle, tm.model.predict(X_oracle))))
            errs[w]=cache[idx]
        ea=np.array([errs[w] for w in WEIGHT_GRID])
        ow=WEIGHT_GRID[int(np.argmin(ea))]
        out.append({"round":t,"oracle_w":ow,"gap":float(ea.max()-ea.min()),
                    "err_min":float(ea.min()),"err_mean":float(ea.mean())})
        ci=picks[ow]
        df_Train=pd.concat([df_Train, df_Candidate.loc[[ci]]]); df_Candidate=df_Candidate.drop([ci])
    return pd.DataFrame(out)

t0=time.time(); ref=reference("dgp_two_regime",3); t_ref=time.time()-t0
t0=time.time(); fast=pd.DataFrame(run_seed("dgp_two_regime",3,max_iter=40)); t_fast=time.time()-t0

print("=== fast path vs reference (40 rounds) ===")
for c in ["oracle_w","gap","err_min","err_mean"]:
    a,b=ref[c].values, fast[c].values
    print(f"  {c:<10} identical={np.array_equal(a,b) if c=='oracle_w' else np.allclose(a,b,atol=0,rtol=0)}  maxdiff={np.abs(a-b).max():.3e}")
print(f"  reference {t_ref:.1f}s | fast {t_fast:.1f}s | speedup {t_ref/t_fast:.1f}x")
