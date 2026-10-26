"""
The manuscript's full strategy set, plus the random-arm control.

Names are copied verbatim from `Code/utils/Main/RunSimulationFunction.py` so
that the regenerated figures inherit the manuscript's colour and linestyle maps
in `GeneratePlots.py` and its label mapping in `GenerateLabellingComplexity.py`.
The only addition is `WiGS (Random Arm)`.

Two notes on fidelity:

* `GSx` and `GSy` are the w=1 and w=0 ends of the weighted sweep. Verified
  identical to `WeightedGreedySamplingSelector` at those weights (500/500), so
  the manuscript's own selectors are used here rather than substitutes.
* `QBCSelector.__init__` takes `seed` (lower case) while the config supplies
  `Seed`, so the committee bootstrap is unseeded in the manuscript's pipeline
  too. That is reproduced rather than silently fixed; `QBC_SEEDED` below flips
  it if a seeded committee is ever wanted.
"""

QBC_SEEDED = False

### The manuscript's 17 strategies, verbatim, plus the control ###
STRATEGIES = {
    'Passive Learning':               ('PassiveLearningSelector', {}),
    'GSx':                            ('GreedySamplingSelector', {'strategy': 'GSx'}),
    'GSy':                            ('GreedySamplingSelector', {'strategy': 'GSy'}),
    'iGS':                            ('GreedySamplingSelector', {'strategy': 'iGS'}),
    'Uncertainty Sampling':           ('UncertaintySamplingSelector', {}),
    'EGAL':                           ('EGALSelector', {}),
    'EMCM':                           ('EMCMSelector', {}),
    'WiGS (Static w_x=0.25)':         ('WeightedGreedySamplingSelector',
                                       {'weight_strategy': 'static', 'w_x': 0.25}),
    'WiGS (Static w_x=0.5)':          ('WeightedGreedySamplingSelector',
                                       {'weight_strategy': 'static', 'w_x': 0.5}),
    'WiGS (Static w_x=0.75)':         ('WeightedGreedySamplingSelector',
                                       {'weight_strategy': 'static', 'w_x': 0.75}),
    'WiGS (Time-Decay, Linear)':      ('WeightedGreedySamplingSelector',
                                       {'weight_strategy': 'time_decay',
                                        'decay_type': 'linear'}),
    'WiGS (Time-Decay, Exponential)': ('WeightedGreedySamplingSelector',
                                       {'weight_strategy': 'time_decay',
                                        'decay_type': 'exponential',
                                        'decay_constant': 5.0}),
    'WiGS (MAB-UCB1, c=0.5)':         ('WiGS_MAB_Selector', {'mab_c': 0.5}),
    'WiGS (MAB-UCB1, c=2.0)':         ('WiGS_MAB_Selector', {'mab_c': 2.0}),
    'WiGS (MAB-UCB1, c=5.0)':         ('WiGS_MAB_Selector', {'mab_c': 5.0}),
    'WiGS (SAC)':                     ('WiGS_SAC_Selector', {}),
    'QBC':                            ('QBCSelector', {'n_committee': 5}),
    ### the control this project adds ###
    'WiGS (Random Arm)':              ('RandomArmSelector', {}),
}

### Selectors whose per-round arm and reward are worth logging ###
ARM_LOGGING = {'WiGS (MAB-UCB1, c=0.5)', 'WiGS (MAB-UCB1, c=2.0)',
               'WiGS (MAB-UCB1, c=5.0)', 'WiGS (Random Arm)'}

### Needs torch; skipped automatically where unavailable ###
TORCH_SELECTORS = {'WiGS (SAC)'}


def available(include_sac=True):
    """Strategy names runnable in this environment, in manuscript order."""
    from utils.GRFPControls.ControlLoop import _register_sac
    ok = _register_sac() if include_sac else False
    return [k for k in STRATEGIES if ok or k not in TORCH_SELECTORS]


def params_for(label, seed=None):
    """Selector type and kwargs for one strategy."""
    sel_type, params = STRATEGIES[label]
    params = dict(params)
    if sel_type == 'QBCSelector' and QBC_SEEDED and seed is not None:
        params['seed'] = seed
    return sel_type, params
