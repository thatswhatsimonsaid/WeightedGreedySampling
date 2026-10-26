import os, sys, gzip, pickle
sys.path.insert(0, os.getcwd())
from utils.GRFPControls import _bootstrap  # noqa
import numpy as np, pandas as pd
from utils.GRFPControls.Analyze import paired_bootstrap_ci
from utils.GRFPControls.ControlProtocol import LearningCurveArea

R='C:/vscode/WeightedGreedySampling_Test_Split/results_grfp_controls/'
ORDER=['Fixed w=0.00','Fixed w=0.25','Fixed w=0.50','Fixed w=0.75','Fixed w=1.00',
       'MAB (c=2.0)','RandomArm','iGS','Passive']
FIXED=['Fixed w=0.00','Fixed w=0.25','Fixed w=0.50','Fixed w=0.75','Fixed w=1.00']

def load(paths):
    c={}
    for p in paths:
        with gzip.open(p,'rb') as fh: c.update(pickle.load(fh))
    return c

def areas(curves, frac):
    rows=[]
    lens={}
    for (ds,sel,seed),v in curves.items(): lens.setdefault(ds,[]).append(len(v))
    cut={ds:max(2,int(round(min(L)*frac))) for ds,L in lens.items()}
    for (ds,sel,seed),v in curves.items():
        rows.append({'dataset':ds,'selector':sel,'seed':seed,'area':LearningCurveArea(v[:cut[ds]])})
    return pd.DataFrame(rows), cut

def report(curves, label, fracs=(1.0,0.70)):
    print(f'\n######## {label} ########')
    for frac in fracs:
        df,cut = areas(curves, frac)
        dss=sorted(df.dataset.unique())
        rel={}; nb=nr=0; fb=fm=0
        for sel in ORDER: rel[sel]=[]
        for ds in dss:
            w=df[df.dataset==ds].pivot_table(index='seed',columns='selector',values='area')
            for sel in ORDER:
                if sel in w: rel[sel].append(w[sel].mean()/w['iGS'].mean())
            pr=w[['MAB (c=2.0)','RandomArm']].dropna()
            m,lo,hi=paired_bootstrap_ci((pr['MAB (c=2.0)']-pr['RandomArm']).values,n_boot=10000,seed=17)
            nb+= hi<0; nr+= lo>0
            best=w[FIXED].mean().idxmin()
            pr2=w[['MAB (c=2.0)',best]].dropna()
            m2,lo2,hi2=paired_bootstrap_ci((pr2['MAB (c=2.0)']-pr2[best]).values,n_boot=10000,seed=17)
            fb+= lo2>0; fm+= hi2<0
        print(f'\n  --- budget {int(frac*100)}%  (e.g. dgp_two_regime {cut.get("dgp_two_regime","-")} rounds) ---')
        for sel in sorted(ORDER,key=lambda s: np.mean(rel[s]) if rel[s] else 9):
            if rel[sel]: print(f'    {sel:<14} {np.mean(rel[sel]):.4f}  ({100*(np.mean(rel[sel])-1):+6.2f}% vs iGS)')
        print(f'    bandit vs random arm : bandit better {nb}/{len(dss)} | random better {nr}/{len(dss)}')
        print(f'    bandit vs best fixed : fixed better {fb}/{len(dss)} | bandit better {fm}/{len(dss)}')

report(load([R+'paper_metric_fullpool/exp_a_curves.pkl.gz']), "FULL-POOL (paper's metric), 14 datasets")
report(load([R+'paper_protocol/exp_a_curves.pkl.gz']), "TEST RMSE (paper protocol), 13 datasets")
