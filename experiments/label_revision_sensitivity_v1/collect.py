"""Verify every bootstrap draw and calculate all five prespecified paired contrasts."""
import sys,json
from pathlib import Path
import numpy as np
import pandas as pd
from support import write,sha,arrays
from orchestrate import outputs
CONTRASTS={
'q2_L1_minus_L0':{'L1_q2':1,'L0_q2':-1},
'unrestricted_L1_minus_L0':{'L1_unrestricted':1,'L0_unrestricted':-1},
'L0_q2_minus_unrestricted':{'L0_q2':1,'L0_unrestricted':-1},
'L1_q2_minus_unrestricted':{'L1_q2':1,'L1_unrestricted':-1},
'gap_change':{'L1_q2':1,'L1_unrestricted':-1,'L0_q2':-1,'L0_unrestricted':1}}
def main(run):
 chunks=[arrays(p/'bootstrap.npz') for p in outputs(run,'bootstrap',10)];first=chunks[0]
 for z in chunks:assert np.array_equal(z['names'],first['names']) and np.allclose(z['point'],first['point'],rtol=0,atol=0,equal_nan=True)
 ids=np.concatenate([z['replicate_ids'] for z in chunks]);assert np.array_equal(np.sort(ids),np.arange(1000))
 reps=np.concatenate([z['replicates'] for z in chunks])[np.argsort(ids)];point=dict(zip(first['names'],first['point']));idx={k:i for i,k in enumerate(first['names'])}
 # Independent sklearn-backed point evaluation and all alert/lead denominators.
 m=pd.read_csv(run/'reports/metrics_by_year_seed.csv');w=pd.read_csv(run/'reports/warning_utility.csv');errs=[]
 for row in m.to_dict('records'):
  for metric in ['auc','ap','ipcw_brier']:
   key=f"{row['scope']}|{row['model']}|{row['seed']}|{row['weighting']}|{metric}";errs.append(abs(point[key]-row[metric]))
 for row in w[w.scope.isin(['2023','2024','pooled'])].to_dict('records'):
  prefix=f"{row['scope']}|{row['model']}|{row['seed']}|Top{int(row['top_fraction']*100)}|"
  mapping={'company_hit_fraction':'company_hit_fraction','event_coverage':'event_coverage','detected_events':'detected_events','independent_events':'independent_events','alert_companies':'alert_company_groups','alert_records':'alert_records'}
  for mo in [3,6]:
   for name,field in [('detected','detected'),('opportunity','opportunity'),('coverage','coverage')]:mapping[f'lead{mo}m_{name}']=f'announcement_{field}_{mo}m'
  for metric,col in mapping.items():errs.append(abs(point[prefix+metric]-row[col]))
 assert max(errs)<1e-10,max(errs)
 rows=[]
 for name,weights in CONTRASTS.items():
  for key in first['names']:
   scope,model,seed,weighting,metric=key.split('|')
   if model!='L0_q2':continue
   names={mod:'|'.join([scope,mod,seed,weighting,metric]) for mod in weights}
   delta=sum(wt*reps[:,idx[names[mod]]] for mod,wt in weights.items());pp=sum(wt*point[names[mod]] for mod,wt in weights.items());ok=np.isfinite(delta);lo,hi=np.quantile(delta[ok],[.025,.975])
   rows.append(dict(scope=scope,seed=int(seed),weighting_or_budget=weighting,metric=metric,contrast=name,point_difference=pp,conditional_ci_lower=lo,conditional_ci_upper=hi,valid_replicates=int(ok.sum()),replicates=1000))
 pd.DataFrame(rows).to_csv(run/'reports/paired_metric_differences.csv',index=False)
 np.savez_compressed(run/'private/paired_bootstrap_PRIVATE.npz',names=first['names'],replicate_ids=np.arange(1000),replicates=reps,point=first['point'])
 cc=[]
 for p in outputs(run,'explain',8):
  s=json.loads((p/'summary.json').read_text());cc.append(s)
 pd.DataFrame(cc).to_csv(run/'reports/native_complexity.csv',index=False)
 write(run/'audit/bootstrap_verification.json',dict(replicates=1000,unique_ids=1000,paired_draws_all_models_seeds_scopes=True,point_max_error=max(errs),min_valid_replicates=min(r['valid_replicates'] for r in rows),fits=0))
 print(pd.DataFrame(rows).query("scope=='pooled' and seed==-1 and metric in ['ap','ipcw_brier','company_hit_fraction','event_coverage'] and weighting_or_budget in ['record_equal','Top20']").to_string(index=False))
if __name__=='__main__':main(Path(sys.argv[1]))
