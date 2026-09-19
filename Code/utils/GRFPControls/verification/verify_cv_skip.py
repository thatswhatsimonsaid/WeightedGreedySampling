import os, sys, time
sys.path.insert(0, os.getcwd())
from utils.GRFPControls import _bootstrap  # noqa
import numpy as np
from utils.GRFPControls.ControlLoop import RunControlIteration

print("=== CV-skip equivalence (reward-free selectors) ===")
specs = [("WeightedGreedySamplingSelector", {"weight_strategy":"static","w_x":0.25}),
         ("WeightedGreedySamplingSelector", {"weight_strategy":"static","w_x":1.0}),
         ("GreedySamplingSelector", {"strategy":"iGS"}),
         ("PassiveLearningSelector", {})]
allok = True
for name, p in specs:
    a = RunControlIteration("dgp_two_regime", Seed=5, SelectorType=name, selector_params=p,
                            MaxIterations=120, ComputeCV=True)
    b = RunControlIteration("dgp_two_regime", Seed=5, SelectorType=name, selector_params=p,
                            MaxIterations=120, ComputeCV=False)
    same_curve = np.array_equal(np.array(a["ErrorVecs"]["RMSE"]), np.array(b["ErrorVecs"]["RMSE"]))
    same_sel = a["SelectedObservationHistory"] == b["SelectedObservationHistory"]
    allok &= (same_curve and same_sel)
    lbl = f"{name}{p.get('w_x', p.get('strategy',''))}"
    print(f"  {lbl:<45} curve_identical={same_curve} selections_identical={same_sel}")
print("  -> CV skip is a no-op:", allok)

print("\n=== speed ===")
for cv in (True, False):
    t=time.time()
    RunControlIteration("dgp_two_regime", Seed=5, SelectorType="GreedySamplingSelector",
                        selector_params={"strategy":"iGS"}, ComputeCV=cv)
    print(f"  full run ComputeCV={cv}: {time.time()-t:.1f}s")
