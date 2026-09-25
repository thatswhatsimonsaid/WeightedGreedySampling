"""
Build the Two-Regime DGP exactly as the manuscript describes it.

The manuscript's Section 5.1 specifies

    x ~ 0.4 N(0.2, 0.07^2) + 0.3 N(0.5, 0.1^2) + 0.3 N(0.85, 0.05^2)

and rests the density-veto demonstration on the third component overlapping the
high-noise band at 0.8 < x < 0.9.

`PreprocessData.generate_gmm_x_samples` implements exactly that mixture -- same
weights, means and standard deviations -- but nothing in the repository calls
it. `generate_two_regime_data` and `generate_three_regime_data` both call
`generate_normal_x_samples`, a single N(0.5, 0.25) clipped to [0, 1]. So the
shipped `dgp_two_regime.pkl` puts the noise trap in a sparse tail (5.4% of the
mass) rather than on a density peak, and the conflict the experiment is built to
create does not exist in the data.

This module rebuilds the dataset with the mixture the manuscript specifies,
reusing the repo's own `generate_gmm_x_samples` and the same response and noise
definitions as `generate_two_regime_data`. Output goes to a GRFP-controls data
directory; nothing under `Data/` is touched.

Usage (from the Code/ directory):
    python -m utils.GRFPControls.BuildPaperDGP
"""

from utils.GRFPControls import _bootstrap  # noqa: F401

import os
import pickle

import numpy as np
import pandas as pd

from utils.Auxiliary.PreprocessData import generate_gmm_x_samples

### Where GRFP-controls-only datasets live ###
def controls_data_dir():
    root = os.path.dirname(os.getcwd())          # repo root, when run from Code/
    d = os.path.join(root, "results_grfp_controls", "data")
    os.makedirs(d, exist_ok=True)
    return d


def generate_two_regime_gmm(n_samples=1000, seed=None):
    """
    The manuscript's Two-Regime DGP: GMM inputs, piecewise response, and a
    high-noise band at 0.8 < x < 0.9.

    The response and noise are copied from `generate_two_regime_data`; the only
    change is drawing x from `generate_gmm_x_samples` rather than
    `generate_normal_x_samples`. Note the manuscript writes the trap noise as
    N(0, 1), i.e. sd 1.0, which is what the repo's two-regime generator uses too.
    """
    if seed is not None:
        np.random.seed(seed)

    x = generate_gmm_x_samples(n_samples=n_samples, seed=seed)
    y = np.zeros(n_samples)
    noise = np.zeros(n_samples)

    ## Regime 1: complex sine ##
    m1 = x < 0.5
    y[m1] = np.sin(x[m1] * 10 * np.pi)
    noise[m1] = np.random.normal(0, 0.1, size=np.sum(m1))

    ## Regime 2: linear ##
    m2 = x >= 0.5
    y[m2] = 2 * x[m2] - 1
    noise[m2] = np.random.normal(0, 0.1, size=np.sum(m2))

    ## The noise trap, overlapping the densest mixture component ##
    trap = (x > 0.8) & (x < 0.9)
    noise[trap] = np.random.normal(0, 1.0, size=np.sum(trap))

    return pd.DataFrame({'X1': x, 'Y': y + noise})


def main():
    from sklearn.preprocessing import RobustScaler

    out_dir = controls_data_dir()
    raw = generate_two_regime_gmm(n_samples=1000, seed=42)

    # Match the stored datasets' pipeline: RobustScaler on features, Y untouched.
    # Runs use --standardize on top, which recovers the manuscript's zero-mean
    # unit-variance preprocessing regardless of which scaler was applied first.
    df = raw.copy()
    df[["X1"]] = RobustScaler().fit_transform(df[["X1"]])
    df = df[["Y", "X1"]]

    path = os.path.join(out_dir, "dgp_two_regime_gmm.pkl")
    with open(path, "wb") as fh:
        pickle.dump(df, fh)

    ### Report the density structure against the manuscript's claim ###
    x = raw["X1"].values
    print(f"wrote {path}  n={len(df)}")
    print("\nmanuscript: 0.4 N(0.2,0.07^2) + 0.3 N(0.5,0.1^2) + 0.3 N(0.85,0.05^2)")
    print("mass by region (this dataset vs the shipped dgp_two_regime):")
    from utils.Auxiliary.LoadDataSet import LoadData
    from utils.Auxiliary.PreprocessData import generate_two_regime_data
    old = generate_two_regime_data(n_samples=1000, seed=42)["X1"].values
    for lo, hi, lab in [(0.10, 0.30, "component 1 (~0.2)"),
                        (0.40, 0.60, "component 2 (~0.5)"),
                        (0.80, 0.90, "trap / component 3 (~0.85)")]:
        a = 100 * ((x > lo) & (x < hi)).mean()
        b = 100 * ((old > lo) & (old < hi)).mean()
        print(f"  {lab:<28} GMM {a:5.1f}%   shipped {b:5.1f}%")


if __name__ == "__main__":
    main()
