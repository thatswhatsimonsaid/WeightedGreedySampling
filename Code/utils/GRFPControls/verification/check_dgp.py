import os, sys
sys.path.insert(0, os.getcwd())
from utils.GRFPControls import _bootstrap  # noqa
import numpy as np, pandas as pd
from sklearn.preprocessing import RobustScaler
from utils.Auxiliary.PreprocessData import generate_two_regime_data, generate_three_regime_data
from utils.Auxiliary.LoadDataSet import LoadData

for name, gen, n in [("dgp_two_regime", generate_two_regime_data, 1000),
                     ("dgp_three_regime", generate_three_regime_data, 1500)]:
    pkl = LoadData(name)
    raw = gen(n_samples=n, seed=42)
    sc = RobustScaler().fit(raw[["X1"]])
    recon_X = sc.transform(raw[["X1"]]).ravel()
    print(f"\n{name}: pkl{pkl.shape} raw{raw.shape}")
    print("  Y identical:      ", np.allclose(pkl['Y'].values, raw['Y'].values))
    print("  X1 reconstructed: ", np.allclose(pkl['X1'].values, recon_X))
    print("  scaler center/scale:", sc.center_, sc.scale_)
