"""Verify every published replicate and report paired conditional intervals."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from crforest.transparent import file_sha256 as sha,write_json

def main(a):
    run=a.run;names=None;values=[];ids=[];cost=[];point=None
    report=json.loads((run/f'bootstrap.jsonl.results-{a.job}.json').read_text())
    assert report['expected']==10 and len(report['completed'])==10 and not report['pending']
    assert {r['id'] for r in report['completed']}==set(range(10))
    for r in report['completed']:assert r['returncode']==0 and r['compute_completed'] and r['publication_completed']
    for batch in range(10):
        root=run/'results/bootstrap'/str(a.job)/f'task-{batch}';out=root/'output';rec=json.loads((root/'complete.json').read_text())
        for entry in rec['manifest']:
            if entry[1]=='directory':continue
            n,size,h=entry;assert (out/n).stat().st_size==size and sha(out/n)==h
        with np.load(out/'bootstrap.npz') as z:
            if names is None:names=z['names'];point=z['point']
            else:np.testing.assert_array_equal(names,z['names']);np.testing.assert_allclose(point,z['point'],atol=1e-13,rtol=0)
            np.testing.assert_array_equal(z['replicate_ids'],np.arange(batch*100,(batch+1)*100));values.append(z['replicates']);ids.extend(z['replicate_ids'].tolist())
        cost.append(json.loads((out/'summary.json').read_text()))
    assert ids==list(range(1000));v=np.concatenate(values)
    with (run/'bootstrap_replicates_PRIVATE.npz').open('xb') as f:np.savez_compressed(f,names=names,point=point,replicates=v,replicate_ids=np.array(ids))
    mapped={tuple(n.split('|')):(point[i],v[:,i]) for i,n in enumerate(names)}
    # Average metrics across the three fixed seeds, distinct from probability ensemble.
    for scope,model,seed,weighting,metric in list(mapped):
        if seed!='42':continue
        series=[mapped[scope,model,str(s),weighting,metric] for s in [42,101,202]]
        mapped[scope,model,'mean_seed_metrics',weighting,metric]=(float(np.mean([t[0] for t in series])),np.mean([t[1] for t in series],axis=0))
    def row(p,arr):
        ok=np.isfinite(arr);lo,hi=np.quantile(arr[ok],[.025,.975]) if ok.any() else (np.nan,np.nan)
        return dict(point=p,ci_low=lo,ci_high=hi,valid_replicates=int(ok.sum()),requested_replicates=1000)
    interval=[]
    for (scope,model,seed,weighting,metric),(p,arr) in sorted(mapped.items()):interval.append(dict(scope=scope,model=model,seed=seed,weighting=weighting,metric=metric,**row(p,arr)))
    comparisons=[('base_weighted','base_equal'),('ordinary_extra_weighted','ordinary_extra_equal'),('pair_weighted','pair_equal'),('ordinary_extra_equal','base_equal'),('pair_equal','ordinary_extra_equal'),('pair_weighted','ordinary_extra_weighted'),('pair_weighted','base_equal'),('pair_weighted','ebm_pair0'),('pair_weighted','q2_fulltrain_reference'),('pair_weighted','M2_allsector_reference')]
    diffs=[]
    for a1,b1 in comparisons:
        for (scope,model,seed,weighting,metric),(pa,va) in sorted(mapped.items()):
            if model!=a1:continue
            pb,vb=mapped[scope,b1,seed,weighting,metric]
            diffs.append(dict(scope=scope,model_a=a1,model_b=b1,contrast='A minus B',seed=seed,weighting=weighting,metric=metric,role_matched=b1 not in ['q2_fulltrain_reference','M2_allsector_reference'],**row(pa-pb,va-vb)))
    pd.DataFrame(interval).to_csv(run/'reports/bootstrap_intervals.csv',index=False);pd.DataFrame(diffs).to_csv(run/'reports/bootstrap_paired_differences.csv',index=False);pd.DataFrame(cost).to_csv(run/'reports/bootstrap_costs.csv',index=False)
    write_json(run/'audit/bootstrap_receipt.json',dict(status='PASS',job=a.job,batches=10,replicates=1000,replicate_ids_unique=True,common_draws=True,scope='conditional on fixed predictions, alert lists and IPCW nuisance; no model refits',point_validation_sha256=sha(run/'audit/bootstrap_validation.json'),replicate_file_sha256=sha(run/'bootstrap_replicates_PRIVATE.npz'),minimum_valid_replicates=min(r['valid_replicates'] for r in interval),multiple_comparisons_adjusted=False,task_hosts={str(x['batch']):x['host'] for x in cost}))
    print('Verified 1000 paired company draws and published all intervals/differences.')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--job',type=int,required=True);main(p.parse_args())
