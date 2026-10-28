"""Read already fitted functions to describe added-field use; no additional fit."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from crforest import CompetingRiskForest
from experiments.post_st_prognosis_v1.common import arrays
from .analyze import outputs


def audit(run):
    rows=[];blocks=[]
    for year,variant,model,seed,folder in outputs(run):
        if model!='xai_q2':continue
        plan=json.loads((run/'input/folds'/str(year)/variant/'plan.json').read_text())
        names=plan['feature_names'];tr=arrays(run/'input/folds'/str(year)/variant/'train_PRIVATE.npz')
        ex=arrays(folder/'components_PRIVATE.npz');keys=[tuple(map(int,k.split(','))) for k in ex['keys']]
        n=len(tr['X']);ti=np.searchsorted(ex['times'],tr['h24'])
        terms=ex['terms'][:n][np.arange(n)[:,None],np.arange(len(keys))[None,:],ti[:,None],:]
        f=CompetingRiskForest.load(folder/'forest_PRIVATE.json')
        for cause in range(2):
            importance=np.mean(abs(terms[:,:,cause]),axis=0);den=importance.sum()
            added=np.array([any(j>=75 for j in key) for key in keys])
            blocks.append(dict(test_year=year,variant=variant,seed=seed,cause=['termination','recovery'][cause],
                added_field_term_absolute_share=float(importance[added].sum()/den) if den else 0.,
                main_absolute_share=float(importance[[len(k)==1 for k in keys]].sum()/den) if den else 0.,
                reference='fixed fold training companies, their actual 24-calendar-month horizons',
                allocation='absolute contribution mass; not causal or fraction of predictive improvement'))
            for j,name in enumerate(names):
                main=next((importance[i] for i,k in enumerate(keys) if k==(j,)),0.)
                allocated=sum(importance[i]/len(k) for i,k in enumerate(keys) if j in k)
                trees=sum(any(z.get('feature')==j for z in tree) for tree in f.trees_)
                rows.append(dict(test_year=year,variant=variant,seed=seed,cause=['termination','recovery'][cause],
                    feature=name,added_field=j>=75,main_mean_absolute_CIF_contribution=main,
                    allocated_mean_absolute_CIF_contribution=allocated,
                    trees_splitting_feature=trees,trees=32,
                    attribution='pair absolute contributions split half to each input, descriptive only'))
    pd.DataFrame(rows).to_csv(run/'evaluation/field_usage.csv',index=False)
    pd.DataFrame(blocks).to_csv(run/'evaluation/announcement_block_contribution.csv',index=False)
    d=pd.DataFrame(rows);v=['main_mean_absolute_CIF_contribution','allocated_mean_absolute_CIF_contribution','trees_splitting_feature']
    a=d.groupby(['test_year','variant','cause','feature','added_field'])[v].agg(['mean','min','max'])
    a.columns=['_'.join(k) for k in a.columns];a.reset_index().to_csv(run/'evaluation/field_usage_seed_summary.csv',index=False)
    print(pd.DataFrame(blocks).query("variant=='announcement_fields' and cause=='termination'").to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);audit(p.parse_args().run)
