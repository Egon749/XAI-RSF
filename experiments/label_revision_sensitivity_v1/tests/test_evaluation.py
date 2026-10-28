"""No-fit checks of real-data bootstrap estimands and transparent handoff."""
import os,sys,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
R=Path(os.environ['REVISION_RUN']);sys.path.insert(0,str(R/'input/legacy'))
from bootstrap import rank_setup,weighted_rank
from common import endpoint_weights
from endpoint_guard import source_consistent_endpoint_weights
from evaluate import event_leads,top_flags

def test_nonuniform_company_draw_matches_independent_sklearn():
 d=pd.read_parquet(R/'bootstrap_input/metadata.parquet');p=np.load(R/'bootstrap_input/predictions.npz')['L1_q2|-1'];g,gi=np.unique(d.group.astype(str),return_inverse=True)
 rng=np.random.default_rng(np.random.SeedSequence(20261025,spawn_key=(7,)));counts=np.bincount(rng.integers(0,len(g),len(g)),minlength=len(g));y=(d.event.to_numpy()==1).astype(float);ipcw,_=source_consistent_endpoint_weights(d,endpoint_weights)
 for base in [np.ones(len(d)),1/d.groupby('group').group.transform('size').to_numpy()]:
  w=counts[gi]*ipcw*base;ok=w>0;a,b=weighted_rank(y,w,rank_setup(y,p,ipcw>0))
  np.testing.assert_allclose([a,b],[roc_auc_score(y[ok],p[ok],sample_weight=w[ok]),average_precision_score(y[ok],p[ok],sample_weight=w[ok])],atol=1e-12,rtol=0)

def test_fixed_lists_include_unknown_and_have_same_budget():
 d=pd.read_parquet(R/'bootstrap_input/metadata.parquet');z=np.load(R/'bootstrap_input/predictions.npz');unknown=~(d.label_eligible.to_numpy(bool)|(d.event.to_numpy()==2))
 for fraction in [.2,.25]:
  n=None
  for model in ['L0_q2','L1_q2','L0_unrestricted','L1_unrestricted']:
   f=top_flags(d,z[model+'|-1'],fraction);assert (f&unknown).sum()>0
   if n is None:n=f.sum()
   assert f.sum()==n

def test_calendar_month_not_days_or_reverse_addition():
 d=pd.DataFrame(dict(event=[1],event_id=['e'],group=['g'],origin=pd.to_datetime(['2023-02-28']),announcement_date=pd.to_datetime(['2023-05-30']),event_date=pd.to_datetime(['2023-05-31'])))
 lf=event_leads(d,np.array([True]));assert bool(lf.announcement_detected_3m.iloc[0]);assert not bool(lf.announcement_detected_6m.iloc[0])

def test_handoff_hashes_and_no_stale_source_plan():
 h=R/'handoff_L1';m=json.loads((h/'handoff_manifest.json').read_text());import hashlib
 for f,digest in m['files'].items():assert hashlib.sha256((h/f).read_bytes()).hexdigest()==digest
 p=json.loads((h/'plan.json').read_text());assert p['source_counts']['final_train']['target_events']==174
 for f,digest in p['source_hashes'].items():assert hashlib.sha256((h/f).read_bytes()).hexdigest()==digest
 assert m['training_contract']['training_contract']=='PASS' and not m['transparent_training_executed']

def test_complete_functions_and_ensemble_not_single_seed():
 c=pd.read_csv(R/'reports/native_complexity.csv');assert len(c)==8 and c.missing_cells_retained.all();assert c.reconstruction_error.max()<1e-10 and c.function_reload_error.max()==0
 for version in ['L0','L1']:
  x=c[c.version==version];ensemble=x[x.seed==-1].iloc[0];assert ensemble.main_functions>=x[x.seed!=-1].main_functions.max();assert ensemble.interaction_functions>=x[x.seed!=-1].interaction_functions.max()
