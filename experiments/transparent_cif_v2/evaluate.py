"""Evaluate already frozen predictions using the SAME legacy source helpers.

No training, calibration, threshold tuning or test-selected tree weights.
Point estimates only; reuse the existing company-bootstrap pipeline separately.
"""
import argparse
import importlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from crforest.transparent import file_sha256,write_json
from .run import read_metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-input',type=Path,required=True)
    parser.add_argument('--predictions',nargs='+',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--metadata-format',choices=['parquet','csv'],default='parquet')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    legacy=(args.source_input/'legacy').resolve()
    for name in ['common.py','endpoint_guard.py','evaluate.py']:
        if not (legacy/name).is_file():raise ValueError('Frozen legacy helper missing: '+name)
    sys.path.insert(0,str(legacy))
    common=importlib.import_module('common');guard=importlib.import_module('endpoint_guard');old=importlib.import_module('evaluate')
    for module in [common,guard,old]:
        if Path(module.__file__).resolve().parent!=legacy:raise RuntimeError('Incorrect helper import resolution')
    data=read_metadata(args.source_input,'eval',args.metadata_format)
    predictions={};source_hashes={}
    for folder in args.predictions:
        receipt=json.loads((folder/'prediction_receipt.json').read_text())
        source_hashes[str(folder/'prediction_receipt.json')]=file_sha256(folder/'prediction_receipt.json')
        for item in receipt['models']:
            path=folder/f'{item["model"]}_PRIVATE.npz'
            if file_sha256(path)!=item['prediction_sha256']:raise ValueError('Saved prediction hash mismatch')
            with np.load(path,allow_pickle=False) as z:
                np.testing.assert_array_equal(z['sample_ids'].astype(str),data.sample_id.astype(str))
                key=(item['model'],item['seed'])
                if key in predictions:raise ValueError('Duplicate model/seed')
                predictions[key]=z['risk'].copy()
    # Frozen references are descriptive, not matched tree-training-role controls.
    m2=args.source_input/'M2_PRIVATE.npz'
    if m2.exists():
        with np.load(m2,allow_pickle=False) as z:
            np.testing.assert_array_equal(z['sample_ids'].astype(str),data.sample_id.astype(str))
            for seed,risk in zip(z['seeds'],z['probabilities']):predictions['M2_allsector_reference',int(seed)]=risk.copy()
        source_hashes[str(m2)]=file_sha256(m2)
    for path in args.source_input.glob('reference_q2_*_PRIVATE.npz'):
        seed=int(path.name.split('_')[2])
        with np.load(path,allow_pickle=False) as z:
            np.testing.assert_array_equal(z['sample_ids'].astype(str),data.sample_id.astype(str))
            predictions['q2_fulltrain_reference',seed]=z['cif12'][:,0].copy()
    for model in sorted({m for m,s in predictions}):
        arr=[v for (m,s),v in predictions.items() if m==model]
        predictions[model,-1]=np.mean(arr,axis=0)
    metrics=[];warning=[]
    scopes=[(str(y),np.flatnonzero(data.year.to_numpy()==y)) for y in sorted(data.year.unique())]+[('pooled',np.arange(len(data)))]
    for scope,ix in scopes:
        d=data.iloc[ix].reset_index(drop=True)
        w,wa=guard.source_consistent_endpoint_weights(d,common.endpoint_weights)
        y=(d.event.to_numpy()==1).astype(int)
        for (model,seed),allp in sorted(predictions.items()):
            p=allp[ix]
            if not np.isfinite(p).all() or p.min() < -1e-10 or p.max()>1+1e-10:raise ValueError('Illegal prediction')
            for weighting in ['record_equal','company_equal']:
                base=np.ones(len(d)) if weighting=='record_equal' else 1/d.groupby('group').group.transform('size').to_numpy()
                ww=w*base;keep=ww>0;both=len(np.unique(y[keep]))==2
                metrics.append(dict(scope=scope,model=model,seed=seed,
                    aggregation='probability_ensemble' if seed==-1 else 'single_seed',weighting=weighting,
                    auc=float(roc_auc_score(y[keep],p[keep],sample_weight=ww[keep])) if both else None,
                    ap=float(average_precision_score(y[keep],p[keep],sample_weight=ww[keep])) if both else None,
                    ipcw_brier=float(np.sum(ww*(y-p)**2)/base.sum()),
                    prediction_mean=float(np.average(p,weights=base)),
                    observed_ipcw_incidence=float(np.sum(ww*y)/base.sum()),records=len(d)))
            for fraction in [.2,.25]:
                flags=old.top_flags(d,p,fraction)
                values=old.warning_metrics(d,flags);lead=old.lead_summary(old.event_leads(d,flags))
                group_count=values.get('alert_company_groups',0)
                values['confirmed_events_per_alert_company_group']=values['detected_events']/group_count if group_count else None
                warning.append(dict(scope=scope,model=model,seed=seed,top_fraction=fraction,**values,**lead))
    pd.DataFrame(metrics).to_csv(args.output/'metrics.csv',index=False)
    pd.DataFrame(warning).to_csv(args.output/'warning_metrics.csv',index=False)
    write_json(args.output/'evaluation_receipt.json',{'status':'POST_EXPOSURE_EXPLORATORY_PROVISIONAL',
        'new_fits':0,'data_records':len(data),'source_hashes':source_hashes,
        'helper_hashes':{str(legacy/name):file_sha256(legacy/name) for name in ['common.py','endpoint_guard.py','evaluate.py']},
        'bootstrap_performed':False,'model_selection_performed':False,
        'caveats':['Company-group confirmed hit fraction is not record precision or mature-only PPV',
                   'Unknown outcomes retained in workload, not silently labeled negative',
                   'M2 and old q2 use different training scopes; matched controls are six new arms',
                   'Seed mean metrics and probability-ensemble metrics must be kept separate']})

if __name__=='__main__':main()
