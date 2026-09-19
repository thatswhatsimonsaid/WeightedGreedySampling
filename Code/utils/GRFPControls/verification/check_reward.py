import os, sys
sys.path.insert(0, os.getcwd())
from utils.GRFPControls import _bootstrap  # noqa
import numpy as np
from utils.GRFPControls.ControlLoop import RunControlIteration

out = RunControlIteration("dgp_two_regime", Seed=3, SelectorType="RandomArmSelector", MaxIterations=25)
log = out["ArmLog"]
# selector logs reward for arm pulled at t-1 in row t; derived attributes to row t
diffs = []
for t in range(len(log) - 1):
    a = log[t]["reward_derived"]
    b = log[t + 1]["reward"]
    if not (np.isnan(a) or np.isnan(b)):
        diffs.append(abs(a - b))
print("compared:", len(diffs), "max abs diff:", max(diffs) if diffs else "n/a")
print("derived == selector-internal reward:", bool(diffs) and max(diffs) < 1e-12)

mab = RunControlIteration("dgp_two_regime", Seed=3, SelectorType="WiGS_MAB_Selector",
                          selector_params={"mab_c": 2.0}, MaxIterations=25)
ml = mab["ArmLog"]
print("MAB reward_derived non-nan:", sum(1 for r in ml if not np.isnan(r["reward_derived"])), "/", len(ml))
print("MAB arm_index sample:", [r["arm_index"] for r in ml[:10]])
print("MAB final arm_values:", ml[-1]["arm_values"], "counts:", ml[-1]["arm_counts"])
