"""Independent sklearn and direct three-seed prediction checks; no fitting."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from crforest import CompetingRiskForest
from experiments.post_st_prognosis_v1.common import horizon_predictions
from .common import *


def validate(run):
    CompetingRiskForest.fit=prohibit_fit
    effects=pd.read_csv(run/'reports/permutation_by_cohort.csv',dtype={'key':str})
    checked=0;maximum=0.;ensemble_roundoff=0.
    for v in range(2019,2024):
        folder=run/f'input/folds/{v}';X=arrays(folder/'eval_PRIVATE.npz')['X'];base=arrays(folder/'baseline_PRIVATE.npz')['pred']
        fs=[CompetingRiskForest.load(folder/f'models/{s}/forest_PRIVATE.json') for s in SEEDS]
        mean=np.mean([f.predict(X,TIMES) for f in fs],axis=0);ensemble_roundoff=max(ensemble_roundoff,float(abs(base-mean).max()))
        np.testing.assert_allclose(base,mean,atol=1e-13,rtol=0)
        perm=arrays(folder/'draws_PRIVATE.npz')['permutations']
        # Repeat all128 draws for one ordinary field and the logical audit block
        # in every cohort. sklearn is independent of the vectorized scorer.
        for j in [48,76]:
            moved=logical_closure([j]);xx=np.broadcast_to(X,(len(perm),*X.shape)).copy()
            for k in moved:xx[:,:,k]=X[perm,k]
            preds=np.mean([f.predict(xx.reshape(-1,83),TIMES) for f in fs],axis=0).reshape(len(perm),len(X),len(TIMES),2)
            for outer in [2023,2024]:
                if v>=outer:continue
                labels=arrays(folder/f'labels_{outer}_PRIVATE.npz')
                for h in [12,24]:
                    idx=np.searchsorted(TIMES,labels[f'h{h}'])
                    for c in [1,2]:
                        y,w,known,supported,_=evaluation_context(labels,h,c)
                        if not supported:continue
                        active=w>0;orig=mean[np.arange(len(X)),idx,c-1];p=preds[:,np.arange(len(X)),idx,c-1]
                        vals=[]
                        for risk in [orig,*p]:
                            auc=roc_auc_score(y[active],risk[active],sample_weight=w[active]) if len(np.unique(y[active]))==2 else np.nan
                            ap=average_precision_score(y[active],risk[active],sample_weight=w[active]) if y[active].any() else np.nan
                            vals.append([auc,ap,np.sum(w*(risk-y)**2)/len(y)])
                        vals=np.asarray(vals);expected=vals[0]-np.mean(vals[1:],axis=0);expected[2]*=-1
                        row=effects[(effects.cohort==v)&(effects.outer_cutoff==outer)&(effects.horizon==h)&(effects.cause==c)&(effects.kind=='single')&(effects.key==str(j))]
                        assert len(row)==1
                        actual=row.iloc[0][['delta_auc','delta_ap','delta_brier']].to_numpy(float)
                        np.testing.assert_allclose(expected,actual,atol=1e-12,rtol=0,equal_nan=True)
                        maximum=max(maximum,float(np.nanmax(abs(expected-actual))));checked+=1
    # Unsupported rows must never appear as zero error / negative outcomes.
    assert effects.loc[~effects.supported,['delta_auc','delta_ap','delta_brier']].isna().all().all()
    for outer in [2023,2024]:
        d=pd.read_csv(run/'reports/feature_classification.csv');d=d[d.outer_cutoff==outer]
        assert len(d)==83 and d.feature.nunique()==83
    result=dict(status='PASS',independent_effect_rows=checked,all128_draws_per_checked_item=True,
        maximum_metric_difference=maximum,maximum_ensemble_roundoff=ensemble_roundoff,new_fits=0,
        checked_columns=['log_price_last','linked_audit_block'],unsupported_metrics_are_NA=True)
    write(run/'audit/independent_validation.json',result);print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();validate(a.run)
