import os, sys
sys.path.insert(0, os.getcwd())
from utils.GRFPControls import _bootstrap  # noqa
import numpy as np, pandas as pd
from utils.GRFPControls.Analyze import paired_bootstrap_ci

R = 'C:/vscode/WeightedGreedySampling_Test_Split/results_grfp_controls/'
ORDER = ['Fixed w=0.00','Fixed w=0.25','Fixed w=0.50','Fixed w=0.75','Fixed w=1.00',
         'MAB (c=2.0)','RandomArm','iGS','Passive']
BENCH = ['concrete_cs','concrete_flow','concrete_slump','yacht','bodyfat','beer',
         'mpg','no2','pm10','qsar','housing']

def table(paths, label, subset=None):
    r = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    if subset: r = r[r.dataset.isin(subset)]
    dss = sorted(r.dataset.unique())
    rows = []
    for sel in ORDER:
        rels, better, worse = [], 0, 0
        for ds in dss:
            w = r[r.dataset == ds].pivot_table(index='seed', columns='selector', values='area')
            if sel not in w.columns: continue
            pr = w[[sel, 'iGS']].dropna()
            rels.append(pr[sel].mean() / pr['iGS'].mean())
            if sel != 'iGS':
                m, lo, hi = paired_bootstrap_ci((pr[sel] - pr['iGS']).values, n_boot=10000, seed=17)
                better += hi < 0; worse += lo > 0
        if not rels: continue
        rows.append({'selector': sel, 'RelAUC': np.mean(rels),
                     'pct_vs_iGS': 100*(np.mean(rels)-1),
                     'better': better, 'worse': worse, 'n': len(dss)})
    d = pd.DataFrame(rows).sort_values('RelAUC')
    print(f'\n=== {label}  ({len(dss)} datasets, 100 seeds) ===')
    print('  Rel. AUC vs iGS  (<1.000 = better than iGS);  better/worse = datasets where paired 95% CI excludes 0')
    for _, x in d.iterrows():
        tag = '' if x.selector == 'iGS' else f"{int(x.better)}/{int(x.n)} better, {int(x.worse)} worse"
        print(f"  {x.selector:<14} {x.RelAUC:.4f}   {x.pct_vs_iGS:+6.2f}%   {tag}")
    return d

table([R+'exp_a_runs.csv', R+'benchmarks/exp_a_runs.csv'], 'REPO PROTOCOL + TEST RMSE, 11 benchmarks', BENCH)
table([R+'exp_a_runs.csv', R+'benchmarks/exp_a_runs.csv'], 'REPO PROTOCOL + TEST RMSE, all 13')
table([R+'paper_protocol/exp_a_runs.csv'], 'PAPER PROTOCOL + TEST RMSE, all 13')
table([R+'paper_metric_fullpool/exp_a_runs.csv'], "PAPER PROTOCOL + FULL-POOL (the paper's own metric), 14")
