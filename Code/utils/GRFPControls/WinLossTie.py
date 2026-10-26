import os, sys
sys.path.insert(0, os.getcwd())
from utils.GRFPControls import _bootstrap  # noqa
import numpy as np, pandas as pd
from utils.GRFPControls.Analyze import paired_bootstrap_ci

R='C:/vscode/WeightedGreedySampling_Test_Split/results_grfp_controls/'
ORDER=['Fixed w=0.00','Fixed w=0.25','Fixed w=0.50','Fixed w=0.75','Fixed w=1.00',
       'MAB (c=2.0)','RandomArm','iGS','Passive']

def tbl(paths,label):
    r=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True)
    dss=sorted(r.dataset.unique()); n=len(dss)
    rows=[]
    for sel in ORDER:
        rels=[];b=l=t=0
        for ds in dss:
            w=r[r.dataset==ds].pivot_table(index='seed',columns='selector',values='area')
            if sel not in w.columns: continue
            pr=w[[sel,'iGS']].dropna()
            rels.append(pr[sel].mean()/pr['iGS'].mean())
            if sel=='iGS': continue
            m,lo,hi=paired_bootstrap_ci((pr[sel]-pr['iGS']).values,n_boot=10000,seed=17)
            if hi<0: b+=1
            elif lo>0: l+=1
            else: t+=1
        if not rels: continue
        rows.append({'Selector':sel,'RelAUC':np.mean(rels),'pct':100*(np.mean(rels)-1),
                     'Win':b,'Loss':l,'Tie':t,'n':n})
    d=pd.DataFrame(rows).sort_values('RelAUC')
    print(f'\n=== {label}  ({n} datasets x 100 seeds) ===')
    print(f'{"Selector":<14} {"Rel.AUC":>8} {"vs iGS":>8}   {"beats / loses / ties":>22}')
    print('-'*58)
    for _,x in d.iterrows():
        if x.Selector=='iGS':
            print(f'{x.Selector:<14} {x.RelAUC:>8.4f} {"—":>8}   {"(reference)":>22}')
        else:
            wlt=f'{int(x.Win)} / {int(x.Loss)} / {int(x.Tie)}'
            print(f'{x.Selector:<14} {x.RelAUC:>8.4f} {x.pct:>+7.2f}%   {wlt:>22}')

tbl([R+'paper_metric_fullpool/exp_a_runs.csv'], "FULL-POOL — the paper's own metric")
tbl([R+'paper_protocol/exp_a_runs.csv'], "TEST RMSE — paper protocol")
tbl([R+'exp_a_runs.csv',R+'benchmarks/exp_a_runs.csv'], "TEST RMSE — repo config")
