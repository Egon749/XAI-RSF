"""Reconstruct full-risk-set frozen flags, then subset; no fitting or reranking."""
from pathlib import Path
import json,sys
import pandas as pd,numpy as np
from review_common import PRIOR,BASE,write

def main(run):
 run=Path(run);sys.path.insert(0,str(BASE/'input/legacy'))
 from evaluate import top_flags,event_leads,warning_metrics
 d=pd.read_parquet(BASE/'input/final_eval_metadata_PRIVATE.parquet');pred={}
 for model in ['q2','unrestricted']:
  parts=[]
  for seed in [42,101,202]:
   z=np.load(BASE/f'input/reference_{model}_{seed}_PRIVATE.npz');np.testing.assert_array_equal(z['sample_ids'],d.sample_id);parts.append(z['cif12'][:,0])
  pred[model]=np.mean(parts,axis=0)
 z=np.load(BASE/'input/M2_PRIVATE.npz');np.testing.assert_array_equal(z['sample_ids'],d.sample_id);pred['M2_allsector']=np.mean(z['probabilities'],axis=0)
 priorflags=np.load(PRIOR/'private/reproduced_flags_PRIVATE.npz');np.testing.assert_array_equal(priorflags['sample_ids'],d.sample_id)
 e=pd.read_parquet(PRIOR/'private/event_rule_evidence.parquet');c=e[(e.role=='eval')&(e.frozen_event_type==1)&(e.verification_status=='VERIFIED')&e.verified_families.str.contains('INTERNAL_CONTROL')];assert len(c)==18
 write(run/'private/cohort.json',c[['event_id','code','group','announcement_date','event_date','verified_families']].to_dict('records'))
 captures=[];flags={};old=pd.read_parquet(PRIOR/'private/event_leads_PRIVATE.parquet')
 for model,p in pred.items():
  for frac in [.2,.25]:
   f=top_flags(d,p,frac);key=model+'_top'+str(int(frac*100));np.testing.assert_array_equal(f,priorflags[key]);flags[key]=f
   leads=event_leads(d,f);prev=old[(old.model==model)&old.top_fraction.eq(frac)].sort_values('event_id').reset_index(drop=True)
   for col in ['event_id','first_warning','announcement_opportunity_3m','announcement_detected_3m','announcement_opportunity_6m','announcement_detected_6m','implementation_opportunity_3m','implementation_detected_3m','implementation_opportunity_6m','implementation_detected_6m']:
    pd.testing.assert_series_equal(leads.sort_values('event_id').reset_index(drop=True)[col],prev[col],check_names=False,check_dtype=False)
   sub=leads[leads.event_id.isin(c.event_id)].copy();sub['model']=model;sub['top_fraction']=frac;captures.append(sub)
 cap=pd.concat(captures,ignore_index=True);cap.to_parquet(run/'private/event_capture_comparison_PRIVATE.parquet',index=False)
 orig=d[d.event_id.isin(c.event_id)&d.event.eq(1)].copy();assert (orig.origin<orig.event_date).all() and (orig.event_date<=orig.horizon).all()
 for key,f in flags.items():orig[key]=f[orig.index]
 orig.to_parquet(run/'private/frozen_audit_origins_PRIVATE.parquet',index=False)
 cap.assign(window_captured=cap.first_warning.notna()).groupby(['model','top_fraction']).agg(events=('event_id','nunique'),window_captured=('window_captured','sum'),opportunity_3m=('announcement_opportunity_3m','sum'),captured_3m=('announcement_detected_3m','sum'),opportunity_6m=('announcement_opportunity_6m','sum'),captured_6m=('announcement_detected_6m','sum')).reset_index().to_csv(run/'results/frozen_capture_summary.csv',index=False)
 write(run/'audit/capture_reproduction.json',{'full_riskset_records':len(d),'events':len(c),'event_origin_rows':len(orig),'all_six_full_riskset_lists_identical':True,'all_original_event_leads_identical':True,'subset_reranking':False,'new_fits':0})
 print((run/'results/frozen_capture_summary.csv').read_text())
if __name__=='__main__':
 import sys;main(sys.argv[1])
