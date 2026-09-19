"""Verification that RandomArmSelector is a true control on WiGS_MAB_Selector."""
import os, sys
sys.path.insert(0, os.getcwd())
from utils.GRFPControls import _bootstrap  # noqa

import numpy as np
from utils.GRFPControls.RandomArmSelector import RandomArmSelector
from utils.GRFPControls.ControlLoop import RunControlIteration
from utils.Selector.WiGS_MAB import WiGS_MAB_Selector
from utils.Auxiliary.LoadDataSet import LoadData
from utils.Prediction.RidgeRegressionPredictor import RidgeRegressionPredictor
from utils.GRFPControls.ControlProtocol import ControlTrainCandidateSplit
from utils.Auxiliary.DataFrameUtils import get_features_and_target
import random

# ---- Test 1: score path identity (single-arm forces both to the same w) ----
print("=== Test 1: score-path identity vs WiGS_MAB (single-arm) ===")
df = LoadData("dgp_two_regime")
random.seed(0); np.random.seed(0)
df_Train, df_Candidate, df_Test = ControlTrainCandidateSplit(df, 0.99)
X_tr, y_tr = get_features_and_target(df=df_Train, target_column_name="Y")
m = RidgeRegressionPredictor(regularization=0.01)
m.fit(X_train_df=X_tr, y_train_series=y_tr)

all_match = True
for w in [0.0, 0.25, 0.5, 0.75, 1.0]:
    mab = WiGS_MAB_Selector(mab_arms=[w], mab_c=2.0)
    rnd = RandomArmSelector(mab_arms=[w], Seed=0)
    a = mab.select(df_Candidate=df_Candidate, df_Train=df_Train, Model=m, current_rmse=1.0)
    b = rnd.select(df_Candidate=df_Candidate, df_Train=df_Train, Model=m, current_rmse=1.0)
    ok = a["IndexRecommendation"] == b["IndexRecommendation"] and a["w_x"] == b["w_x"]
    all_match &= ok
    print(f"  w={w:<5} mab_idx={a['IndexRecommendation']} rand_idx={b['IndexRecommendation']}  {'MATCH' if ok else 'MISMATCH'}")
print("  -> score path identical:", all_match)

# ---- Test 2: arm draws are uniform and seeded ----
print("\n=== Test 2: arm draw uniformity / reproducibility ===")
r1 = RandomArmSelector(mab_arms=[0.25, 0.5, 0.75], Seed=7)
r2 = RandomArmSelector(mab_arms=[0.25, 0.5, 0.75], Seed=7)
d1 = [int(r1.rng.integers(0, 3)) for _ in range(3000)]
d2 = [int(r2.rng.integers(0, 3)) for _ in range(3000)]
print("  same seed reproducible:", d1 == d2)
print("  arm frequencies:", np.bincount(d1) / len(d1))
r3 = RandomArmSelector(mab_arms=[0.25, 0.5, 0.75], Seed=8)
print("  different seed differs:", d1 != [int(r3.rng.integers(0, 3)) for _ in range(3000)])

# ---- Test 3: splits identical across selectors for a given seed ----
print("\n=== Test 3: split identity across selectors (paired comparison validity) ===")
specs = [
    ("WiGS_MAB_Selector", {"mab_c": 2.0}),
    ("RandomArmSelector", {}),
    ("WeightedGreedySamplingSelector", {"weight_strategy": "static", "w_x": 0.5}),
    ("GreedySamplingSelector", {"strategy": "iGS"}),
    ("PassiveLearningSelector", {}),
]
sigs = {}
for name, params in specs:
    out = RunControlIteration("dgp_two_regime", Seed=3, SelectorType=name,
                              selector_params=params, MaxIterations=5)
    sigs[name] = (tuple(sorted(out["InitialTrainIndices"])), out["n_test"])
    print(f"  {name:<32} n_train0={out['n_initial_train']} n_test={out['n_test']} rounds={out['n_rounds']}")
ref = list(sigs.values())[0]
print("  -> all splits identical:", all(v == ref for v in sigs.values()))

# ---- Test 4: arm/reward logging populated ----
print("\n=== Test 4: instrumentation ===")
out = RunControlIteration("dgp_two_regime", Seed=3, SelectorType="WiGS_MAB_Selector",
                          selector_params={"mab_c": 2.0}, MaxIterations=12)
log = out["ArmLog"]
print("  rounds logged:", len(log))
print("  arms seen:", sorted(set(r["w_x"] for r in log)))
print("  rewards non-nan after warmup:", sum(1 for r in log if not np.isnan(r["reward"])))
print("  sample arm_values:", log[-1]["arm_values"], "counts:", log[-1]["arm_counts"])
outr = RunControlIteration("dgp_two_regime", Seed=3, SelectorType="RandomArmSelector",
                           MaxIterations=12)
print("  randomarm rewards non-nan:", sum(1 for r in outr["ArmLog"] if not np.isnan(r["reward"])))
print("  randomarm arms seen:", sorted(set(r["w_x"] for r in outr["ArmLog"])))
