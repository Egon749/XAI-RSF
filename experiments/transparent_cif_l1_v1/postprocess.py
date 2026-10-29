"""Evaluate once after every seed is frozen; expose versioned references explicitly."""
from pathlib import Path
import sys,json,subprocess,shutil
import numpy as np,pandas as pd
from crforest.transparent import file_sha256 as sha,write_json
from experiments.transparent_cif_v2.evaluate import load_versioned_references
from experiments.transparent_cif_l1_v1.orchestrate import gate
SOURCE=Path('/home/oliver/MainDiskData/label_revision_sensitivity_v1_20261028T151804Z/handoff_L1')
def main(run):
 outputs=gate(run,'predict',3);dirs=[p/'predict' for p in outputs];code=run/'input/code'
 subprocess.run([sys.executable,'-m','experiments.transparent_cif_v2.evaluate','--source-input',str(SOURCE),'--reference-manifest',str(run/'input/reference_manifest.json'),'--predictions',*[str(p) for p in dirs],'--output',str(run/'evaluation')],cwd=code,check=True)
 d=pd.read_parquet(SOURCE/'final_eval_metadata_PRIVATE.parquet');assert len(d)==23799 and d.loc[d.event==1,'event_id'].nunique()==79
 preds={};provenance=[]
 for folder in dirs:
  receipt=json.loads((folder/'prediction_receipt.json').read_text())
  for row in receipt['models']:
   p=folder/f"{row['model']}_PRIVATE.npz";assert sha(p)==row['prediction_sha256'];z=np.load(p);np.testing.assert_array_equal(z['sample_ids'],d.sample_id);np.testing.assert_array_equal(z['groups'],d.group.astype(str));np.testing.assert_array_equal(z['horizon'],(d.horizon-d.origin).dt.days)
   preds[row['model'],row['seed']]=z['risk'];provenance.append(dict(model=row['model'],seed=row['seed'],reference=False,prediction_sha256=sha(p),path=str(p)))
 refs,_=load_versioned_references(run/'input/reference_manifest.json',d);preds.update(refs)
 for row in json.loads((run/'input/reference_manifest.json').read_text()):provenance.append(dict(model=row['model'],seed=row['seed'],reference=True,prediction_sha256=row['sha256'],path=row['path']))
 with np.load(SOURCE/'M2_PRIVATE.npz') as z:
  np.testing.assert_array_equal(z['sample_ids'],d.sample_id)
  for seed,p in zip(z['seeds'],z['probabilities']):preds['M2_allsector_reference',int(seed)]=p
 for model in sorted({m for m,s in preds}):preds[model,-1]=np.mean([preds[model,s] for s in [42,101,202]],axis=0)
 assert len(preds)==44
 bs=run/'bootstrap_input';bs.mkdir();d.to_parquet(bs/'metadata.parquet',index=False);np.savez_compressed(bs/'predictions.npz',**{f'{m}|{s}':v for (m,s),v in preds.items()});shutil.copytree(SOURCE/'legacy',bs/'legacy')
 sys.path.insert(0,str(SOURCE/'legacy'));from evaluate import top_flags,warning_metrics,event_leads,lead_summary
 from common import endpoint_weights
 from endpoint_guard import source_consistent_endpoint_weights
 warning=[];cal=[]
 scopes=[('pooled',np.arange(len(d)))]+[(str(y),np.asarray(ix)) for y,ix in d.groupby('year').indices.items()]+[(str(q),np.asarray(ix)) for q,ix in d.groupby('quarter').indices.items()]
 for scope,ix in scopes:
  dd=d.iloc[ix].reset_index(drop=True)
  for (model,seed),allp in preds.items():
   p=allp[ix]
   for frac in [.2,.25]:
    flags=top_flags(dd,p,frac);wm=warning_metrics(dd,flags);lf=event_leads(dd,flags);hit=flags&(dd.event.to_numpy()==1);unknown=flags&~(dd.label_eligible.to_numpy(bool)|(dd.event.to_numpy()==2))
    ag=set(dd.loc[flags,'group']);hg=set(dd.loc[hit,'group']);ug=set(dd.loc[unknown,'group'])-hg
    # Verify this cohort's one-first-event-per-group correspondence before event/company ratios.
    assert dd.loc[hit].groupby('group').event_id.nunique().max()<=1 if hit.any() else True
    warning.append(dict(scope=scope,model=model,seed=seed,top_fraction=frac,**wm,**(lead_summary(lf) if len(lf) else {}),hit_company_groups=len(hg),company_hit_fraction=len(hg)/len(ag),no_hit_with_unknown_groups=len(ug),no_hit_known_only_groups=len(ag-hg-ug)))
   if scope in ['2023','2024','pooled']:
    w,_=source_consistent_endpoint_weights(dd,endpoint_weights);y=dd.event.to_numpy()==1
    for weighting in ['record_equal','company_equal']:
     base=np.ones(len(dd)) if weighting=='record_equal' else 1/dd.groupby('group').group.transform('size').to_numpy();ww=w*base
     for b,ii in enumerate(np.array_split(np.lexsort((dd.sample_id.to_numpy(),p)),10)):
      cal.append(dict(scope=scope,model=model,seed=seed,weighting=weighting,bin=b+1,records=len(ii),prediction_mean=float(np.average(p[ii],weights=base[ii])),observed_ipcw_incidence=float(np.sum(ww[ii]*y[ii])/base[ii].sum())))
 out=run/'reports';m=pd.read_csv(run/'evaluation/metrics.csv');m.to_csv(out/'metrics_by_year_seed.csv',index=False)
 pd.DataFrame(warning).to_csv(out/'warning_utility.csv',index=False);pd.DataFrame(cal).to_csv(out/'calibration.csv',index=False);pd.DataFrame(provenance).to_csv(out/'model_provenance.csv',index=False)
 m[m.seed!=-1].groupby(['scope','model','weighting'])[['auc','ap','ipcw_brier','prediction_mean']].agg(['mean','min','max']).to_csv(out/'seed_ranges.csv')
 write_json(run/'audit/predictions_frozen.json',dict(data_records=len(d),target_events=79,prediction_sets=len(preds),predictions_sha256=sha(bs/'predictions.npz'),metadata_sha256=sha(bs/'metadata.parquet'),all_fits_before_evaluation=True,fit_receipt_sha256=sha(run/'audit/fits_frozen.json')))
 print(m[(m.seed==-1)&(m.weighting=='record_equal')][['scope','model','auc','ap','ipcw_brier']].to_string(index=False))
if __name__=='__main__':main(Path(sys.argv[1]))
