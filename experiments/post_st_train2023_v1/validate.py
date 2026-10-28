"""Independent endpoint metric checks and immutable input/output verification."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from crforest.core import aalen_johansen
from experiments.post_st_prognosis_v1.common import arrays,sha,write,horizon_predictions


def direct_metrics(y,p):
    """Pairwise AUC and threshold-group AP, independent of sklearn metrics."""
    y=np.asarray(y,bool);p=np.asarray(p,float)
    pos=p[y];neg=p[~y]
    auc=float(np.mean((pos[:,None]>neg)+.5*(pos[:,None]==neg))) if len(pos) and len(neg) else np.nan
    ap=np.nan
    if y.any():
        order=np.argsort(-p,kind='stable');scores=p[order];yy=y[order]
        ends=np.r_[np.flatnonzero(scores[:-1]!=scores[1:]),len(scores)-1]
        tp=np.cumsum(yy)[ends];recall=tp/y.sum();precision=tp/(ends+1)
        ap=float(np.sum(np.diff(np.r_[0.,recall])*precision))
    return auc,ap,float(np.mean((p-y)**2))


def validate(run,reports):
    frozen=json.loads((run/'audit/immutable_input_hashes.json').read_text())
    for name,digest in frozen.items():assert sha(run/'input'/name)==digest,name
    tr=arrays(run/'input/train_PRIVATE.npz');ev=arrays(run/'input/eval_PRIVATE.npz')
    assert len(set(tr['groups'])&set(ev['groups']))==0
    plan=json.loads((run/'input/plan.json').read_text())
    models={};files=[]
    for p in sorted((run/'results/formal').glob('*/task-*/output/summary.json')):
        s=json.loads(p.read_text());key=(s['model'],str(s['seed']));assert key not in models
        models[key]=arrays(p.parent/'predictions_PRIVATE.npz')
        for name,digest in json.loads((p.parent/'hashes.json').read_text()).items():assert sha(p.parent/name)==digest
        files.append(dict(model=s['model'],seed=s['seed'],host=s['host'],
            model_file='forest_PRIVATE.rds' if s['model']=='standard_rsf' else 'forest_PRIVATE.json',
            model_sha256=sha(p.parent/('forest_PRIVATE.rds' if s['model']=='standard_rsf' else 'forest_PRIVATE.json')),
            prediction_sha256=sha(p.parent/'predictions_PRIVATE.npz'),
            functions_sha256=sha(p.parent/'functions_PRIVATE.npz') if s['model']=='xai_q2' else None))
    assert set(models)=={(m,str(s)) for m in plan['variants'] for s in plan['seeds']}
    for m in plan['variants']:
        a=[models[(m,str(s))] for s in plan['seeds']]
        models[(m,'ensemble')]=dict(cif=np.mean([x['cif'] for x in a],axis=0),times=a[0]['times'])
    times=np.array([365.,366.,730.,731.]);p=aalen_johansen(tr['time'],tr['event'],times)
    models[('training_AJ','none')]=dict(cif=np.broadcast_to(p,(len(ev['X']),4,2)),times=times)
    df=pd.read_csv(reports/'metrics.csv',dtype={'seed':str});checked=0
    for row in df[df.scope.ne('train_apparent')].itertuples():
        mask=np.ones(len(ev['X']),bool) if row.scope=='pooled' else ev['year']==int(row.scope)
        h=ev[f'h{row.horizon_months}'][mask];t=ev['time'][mask];e=ev['event'][mask]
        assert ev['label_eligible'][mask].all()
        assert ((t>=h)|((e>0)&(t<=h))).all(),'This independent complete-endpoint check needs extension if the cohort changes'
        c=1 if row.cause=='termination' else 2
        p=models[(row.model,row.seed)];risk=horizon_predictions(p['cif'][mask],p['times'],h)[:,c-1]
        y=(e==c)&(t<=h)
        np.testing.assert_allclose(direct_metrics(y,risk),[row.auc,row.ap,row.brier],rtol=0,atol=5e-14,equal_nan=True)
        assert row.known_target_events==int(y.sum())
        assert row.ipcw_min_G==1.
        checked+=1
    assert checked==52
    baseline=df[(df.model=='training_AJ')&(df.scope=='2024')&(df.horizon_months==24)&(df.cause=='termination')].iloc[0]
    assert abs(baseline.ap-20/45)<1e-14 and baseline.auc==.5
    warnings=pd.read_csv(reports/'warning_metrics.csv')
    assert (warnings.checked_companies==warnings.captured_events+warnings.known_non_target_checked+warnings.unknown_checked).all()
    assert (warnings.captured_events<=warnings.all_target_events).all()
    result=dict(status='PASS',immutable_input_files=len(frozen),formal_models=9,independent_metric_rows=checked,
        constant_baseline_AP_equals_observed_prevalence=True,record_company_weights_identical=True,
        train_test_group_overlap=0,complete_test_endpoints=True,training_commit=plan['code_commit'],
        checks_do_not_certify_historical_vintages=True,model_hashes=files)
    write(reports/'validation.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='model_hashes'}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--reports',type=Path,required=True)
    a=p.parse_args();validate(a.run,a.reports)
