"""Frozen endpoint, list and calendar-month definitions; no fitting or selection."""
import sys,json,shutil,argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from support import arrays,write,sha

def main(run):
 inp=run/'input';out=run/'reports';sys.path.insert(0,str(inp/'legacy'))
 from common import endpoint_weights
 from endpoint_guard import source_consistent_endpoint_weights
 from evaluate import top_flags,warning_metrics,event_leads,lead_summary
 d=pd.read_parquet(inp/'L0_ORIGINAL/final_eval_metadata_PRIVATE.parquet');preds={};index=json.loads((run/'audit/models_index.json').read_text());provenance=[]
 for key,folder in index.items():
  version,model,seed=key.split('/');p=arrays(Path(folder)/'predictions_PRIVATE.npz');assert np.array_equal(p['sample_ids'],d.sample_id) and np.array_equal(p['groups'],d.group.astype(str));assert np.array_equal(p['horizon_days'],(d.horizon-d.origin).dt.days)
  preds[(version+'_'+model,int(seed))]=p['cif12'][:,0]
  provenance.append(dict(version=version,model=model,seed=int(seed),formal_fit='new_L1' if version=='L1' else 'reused_L0',model_sha256=sha(Path(folder)/'forest_PRIVATE.json'),prediction_sha256=sha(Path(folder)/'predictions_PRIVATE.npz'),training_code='206e414d05e8cd15e13429637bc6efc7f0abfcd4',folder=folder))
 for version in ['L0','L1']:
  for model in ['q2','unrestricted']:
   label=version+'_'+model;preds[label,-1]=np.mean([preds[label,s] for s in [42,101,202]],axis=0)
 m2=arrays(inp/'M2_PRIVATE.npz');assert np.array_equal(m2['sample_ids'],d.sample_id)
 preds['M2_allsector_reference',-1]=m2['ensemble']
 for i,s in enumerate(m2['seeds']):preds['M2_allsector_reference',int(s)]=m2['probabilities'][i]
 metrics=[];warnings=[];calibration=[];transitions=[];changes=[]
 scopes=[('2023',np.flatnonzero(d.year==2023)),('2024',np.flatnonzero(d.year==2024)),('pooled',np.arange(len(d)))]
 for scope,ix in scopes:
  dd=d.iloc[ix].reset_index(drop=True);y=(dd.event.to_numpy()==1).astype(int);w,wa=source_consistent_endpoint_weights(dd,endpoint_weights)
  for (model,seed),p0 in sorted(preds.items()):
   p=p0[ix];assert np.isfinite(p).all() and p.min()>=0 and p.max()<=1
   for weighting in ['record_equal','company_equal']:
    base=np.ones(len(ix)) if weighting=='record_equal' else 1/dd.groupby('group').group.transform('size').to_numpy();ww=w*base;ok=ww>0
    meta=dict(scope=scope,model=model,seed=seed,aggregation='seed_ensemble' if seed==-1 else 'single_seed',weighting=weighting)
    metrics.append(dict(**meta,records=len(dd),groups=dd.group.nunique(),independent_events=dd.loc[dd.event==1,'event_id'].nunique(),target_records=int(y.sum()),auc=roc_auc_score(y[ok],p[ok],sample_weight=ww[ok]),ap=average_precision_score(y[ok],p[ok],sample_weight=ww[ok]),ipcw_brier=np.sum(ww*(y-p)**2)/base.sum(),prediction_mean=np.average(p,weights=base),observed_ipcw_incidence=np.sum(ww*y)/base.sum(),**wa))
    for b,ii in enumerate(np.array_split(np.lexsort((dd.sample_id.to_numpy(),p)),10)):
     calibration.append(dict(**meta,bin=b+1,records=len(ii),prediction=np.average(p[ii],weights=base[ii]),observed=np.sum(ww[ii]*y[ii])/base[ii].sum()))
   for fraction in [.2,.25]:
    flags=top_flags(dd,p,fraction)
    slices=[(scope,np.arange(len(dd)))]
    if scope!='pooled':slices += [(str(q),np.asarray(ii)) for q,ii in dd.groupby('quarter').indices.items()]
    for sc,ii in slices:
     sub=dd.iloc[ii].reset_index(drop=True);ff=flags[ii];r=warning_metrics(sub,ff);lf=event_leads(sub,ff)
     hg=sub.loc[ff&(sub.event==1),'group'].nunique();unknown=ff&~(sub.label_eligible.to_numpy(bool)|(sub.event.to_numpy()==2))
     extra=lead_summary(lf) if len(lf) else {}
     warnings.append(dict(scope=sc,model=model,seed=seed,top_fraction=fraction,**r,**extra,confirmed_hit_company_groups=hg,company_hit_fraction=hg/r['alert_company_groups'],unknown_company_groups=sub.loc[unknown,'group'].nunique(),known_target_free_company_groups=sub.loc[ff&(sub.event==0)&sub.label_eligible,'group'].nunique(),competing_company_groups=sub.loc[ff&(sub.event==2),'group'].nunique()))
  for model in ['q2','unrestricted']:
   for seed in [42,101,202,-1]:
    p0=preds['L0_'+model,seed][ix];p1=preds['L1_'+model,seed][ix];delta=p1-p0
    changes.append(dict(scope=scope,model=model,seed=seed,changed_records=int((delta!=0).sum()),mean_absolute_change=np.mean(abs(delta)),max_absolute_change=np.max(abs(delta)),mean_signed_change=np.mean(delta),p95_absolute_change=np.quantile(abs(delta),.95)))
    for fraction in [.2,.25]:
     f0=top_flags(dd,p0,fraction);f1=top_flags(dd,p1,fraction);assert f0.sum()==f1.sum()
     ag0=set(dd.loc[f0,'group']);ag1=set(dd.loc[f1,'group']);all_events=set(dd.loc[dd.event==1,'event_id'])
     lead0=event_leads(dd,f0);lead1=event_leads(dd,f1)
     for endpoint in ['window','lead3m','lead6m']:
      if endpoint=='window':e0=set(dd.loc[f0&(dd.event==1),'event_id']);e1=set(dd.loc[f1&(dd.event==1),'event_id']);universe=all_events
      else:
       mo=int(endpoint[4]);col=f'announcement_detected_{mo}m';e0=set(lead0.loc[lead0[col],'event_id']);e1=set(lead1.loc[lead1[col],'event_id']);universe=set(lead0.loc[lead0[f'announcement_opportunity_{mo}m'],'event_id'])
      transitions.append(dict(scope=scope,model=model,seed=seed,top_fraction=fraction,endpoint=endpoint,common_captured=len(e0&e1),new_captured=len(e1-e0),lost_captured=len(e0-e1),common_missed=len(universe-(e0|e1)),opportunity_events=len(universe),companies_entering=len(ag1-ag0),companies_exiting=len(ag0-ag1),companies_shared=len(ag0&ag1),records_entering=int((f1&~f0).sum()),records_exiting=int((f0&~f1).sum())))
 m=pd.DataFrame(metrics);ws=pd.DataFrame(warnings)
 assert np.array_equal(ws.alert_records,ws.hit_records+ws.known_no_target_alerts+ws.unknown_outcome_alerts)
 assert set(m.loc[m.scope=='pooled','independent_events'])=={79} and set(m.loc[m.scope=='pooled','records'])=={23799}
 for file,rows in [('metrics_by_year_seed',m),('warning_utility',ws),('calibration',calibration),('alert_transition_summary',transitions),('prediction_change_summary',changes),('model_provenance',provenance)]:pd.DataFrame(rows).to_csv(out/f'{file}.csv',index=False)
 cols=['auc','ap','ipcw_brier','prediction_mean']
 m[m.seed!=-1].groupby(['scope','model','weighting'])[cols].agg(['mean','min','max']).to_csv(out/'seed_metric_ranges.csv')
 b=run/'bootstrap_input';b.mkdir(exist_ok=False);shutil.copytree(inp/'legacy',b/'legacy');d.to_parquet(b/'metadata.parquet',index=False)
 # All seed predictions share the same company resampling; primary inference is on ensembles.
 np.savez_compressed(b/'predictions.npz',**{f'{m}|{s}':p for (m,s),p in preds.items()})
 write(b/'manifest.json',dict(metadata_sha256=sha(b/'metadata.parquet'),predictions_sha256=sha(b/'predictions.npz'),groups=d.group.nunique(),replicates=1000,stream='SeedSequence(20261025, spawn_key=(rep,))',lists='frozen full-riskset quarterly lists',nuisance='frozen scope-specific original endpoint weights'))
 print(m[(m.seed==-1)&(m.weighting=='record_equal')][['scope','model','auc','ap','ipcw_brier']].to_string(index=False))
if __name__=='__main__':main(Path(sys.argv[1]))
