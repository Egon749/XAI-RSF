"""Compare fresh computation against historical files without refitting or editing."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from experiments.post_st_prognosis_v1.common import arrays,sha,write
from experiments.post_st_next_open_info_v1.analyze import outputs


def compare_npz(a,b):
    aa,bb=arrays(a),arrays(b)
    if set(aa)!=set(bb):return False,None
    exact=True;maximum=0.
    for key in aa:
        x,y=aa[key],bb[key]
        if x.shape!=y.shape or x.dtype!=y.dtype:return False,None
        if x.dtype.kind in 'fc':
            exact &= bool(np.array_equal(x,y,equal_nan=True))
            if not np.array_equal(np.isfinite(x),np.isfinite(y)):return False,None
            good=np.isfinite(x)&np.isfinite(y)
            if good.any():maximum=max(maximum,float(abs(x[good]-y[good]).max()))
        else:exact &= bool(np.array_equal(x,y))
    return bool(exact),maximum


def evaluate(run):
    refs={(r['year'],r['model'],str(r['seed'])):Path(r['path'])
          for r in json.loads((run/'audit/reference_models.json').read_text())}
    rows=[]
    for year,variant,model,seed,new in outputs(run):
        old=refs[(year,model,seed)];assert new.resolve()!=old.resolve()
        for p in [new,old]:
            for name,h in json.loads((p/'hashes.json').read_text()).items():assert sha(p/name)==h
        a=json.loads((new/'summary.json').read_text());b=json.loads((old/'summary.json').read_text())
        plan=json.loads((run/'input/folds'/str(year)/variant/'plan.json').read_text())
        assert a['code_commit']==plan['code_commit'] and a['seconds']>0
        eq,err=compare_npz(new/'predictions_PRIVATE.npz',old/'predictions_PRIVATE.npz')
        model_file='forest_PRIVATE.json' if model=='xai_q2' else 'forest_PRIVATE.rds'
        row=dict(test_year=year,model=model,seed=seed,fresh_fit_seconds=a['seconds'],
            predictions_array_exact=eq,max_prediction_difference=err,
            model_bytes_identical=sha(new/model_file)==sha(old/model_file),
            predictions_bytes_identical=sha(new/'predictions_PRIVATE.npz')==sha(old/'predictions_PRIVATE.npz'),
            fresh_model_sha256=sha(new/model_file),original_model_sha256=sha(old/model_file))
        for name,key in [('functions_PRIVATE.npz','functions'),('native_curves_PRIVATE.npz','native_curves'),
                         ('components_PRIVATE.npz','components')]:
            if (new/name).exists():
                eq,err=compare_npz(new/name,old/name)
                row[key+'_array_exact']=eq;row[key+'_max_difference']=err
        rows.append(row)
    assert len(rows)==12
    df=pd.DataFrame(rows);df.to_csv(run/'evaluation/reproduction_models.csv',index=False)
    metrics=pd.read_csv(run/'evaluation/paired_changes.csv')
    differences=metrics[[c for c in metrics if c.endswith('_change')]].abs().max().to_dict()
    prediction_exact=bool(df.predictions_array_exact.all())
    functions_exact=bool(df[df.model=='xai_q2'].functions_array_exact.all())
    all_metrics_exact=all(v==0 for v in differences.values())
    result=dict(status='EXACT_REPRODUCTION' if prediction_exact and functions_exact and all_metrics_exact else 'DIFFERENCES_RETAINED',
        new_formal_fits=12,reused_reference_models=12,new_pilots=0,new_fits_in_this_audit=0,
        exact_prediction_models=int(df.predictions_array_exact.sum()),
        exact_model_bytes=int(df.model_bytes_identical.sum()),
        full_functions_array_exact=functions_exact,maximum_prediction_difference=float(df.max_prediction_difference.max()),
        maximum_metric_differences=differences,all_metrics_exact=all_metrics_exact,
        evidence='Fresh Slurm completion/publication, new fitting commit and elapsed times; comparison is against read-only historical files',
        no_new_independent_validation=True)
    write(run/'evaluation/reproduction_summary.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);evaluate(p.parse_args().run)
