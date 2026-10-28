"""Read-only freeze of existing labels, models and probabilities. No model fitting."""
from pathlib import Path
import hashlib,json,shutil
import numpy as np
import pandas as pd
BASE=Path('/home/oliver/MainDiskData/xai_rsf_interpretable_baselines_v1_20261027_hw9gdrol')
EVENT=Path('/home/oliver/MainDiskData/first_st_followup_audit_20261025_v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(run):
 run=Path(run);inp=run/'input';hashes={};expected=json.loads((BASE/'audit/input_hashes.json').read_text())
 for name,digest in expected.items():
  p=BASE/'input'/name;assert sha(p)==digest;hashes[str(p)]=digest
 for role in ['train','eval']:
  for n in [f'final_{role}_metadata_PRIVATE.parquet',f'final_{role}_PRIVATE.npz']:shutil.copy2(BASE/'input'/n,inp/n)
 for n in ['plan.json','M2_PRIVATE.npz',*[f'reference_{m}_{s}_PRIVATE.npz' for m in ['q2','unrestricted'] for s in [42,101,202]]]:shutil.copy2(BASE/'input'/n,inp/n)
 shutil.copytree(BASE/'input/legacy',inp/'legacy')
 plan=json.loads((BASE/'input/plan.json').read_text());parent=Path(plan['parent'])
 for p in (parent/'results/assembled').glob('*/task-*/output/*'):
  if p.is_file() and p.name in ['forest_PRIVATE.json','predictions_PRIVATE.npz','functions_PRIVATE.npz']:hashes[str(p)]=sha(p)
 for n in ['input/quarterly_first_financial_audit_12m.json','input/review160_event_decisions.json','assembly/merged_events.parquet','assembly/itemchanges.parquet','assembly/master_snapshot.parquet','audit_v4_observed_first/event_decisions.json']:
  p=EVENT/n;hashes[str(p)]=sha(p)
 frames=[]
 for role in ['train','eval']:
  d=pd.read_parquet(inp/f'final_{role}_metadata_PRIVATE.parquet');d['role']=role;frames.append(d)
 d=pd.concat(frames,ignore_index=True);ev=d[d.event>0];assert not ev.groupby('event_id').event.nunique().gt(1).any()
 e=ev.groupby('event_id',sort=True).agg(code=('code','first'),group=('group','first'),frozen_event_type=('event','first'),announcement_date=('announcement_date','first'),event_date=('event_date','first'),role=('role','first'),window_rows=('sample_id','size')).reset_index()
 base=json.loads((EVENT/'input/quarterly_first_financial_audit_12m.json').read_text());events={x['group_key']+'_'+x['date']:x for x in base['events']};decisions={x['event_id']:x for x in json.loads((EVENT/'input/review160_event_decisions.json').read_text())['decisions']}
 items=pd.read_parquet(EVENT/'assembly/itemchanges.parquet');item_lookup={k:v for k,v in items.groupby('Symbol')};merged=pd.read_parquet(EVENT/'assembly/merged_events.parquet');ml={k:v for k,v in merged.groupby('Symbol')};rows=[]
 for rr in e.to_dict('records'):
  prior=events.get(rr['event_id'],{});manual=decisions.get(rr['event_id'],{});matches=item_lookup.get(rr['code'],pd.DataFrame());ann=pd.Timestamp(rr['announcement_date']);dt=pd.Timestamp(rr['event_date'])
  if len(matches):matches=matches[(pd.to_datetime(matches.ChangeDate)==dt)|(pd.to_datetime(matches.DeclareDate)==ann)]
  z=ml.get(rr['code'],pd.DataFrame());z=z[(pd.to_datetime(z.Execudt)==dt)] if len(z) else z
  rr.update(frozen_category=prior.get('category'),frozen_reason_codes=prior.get('codes'),frozen_reason=prior.get('reason'),frozen_manual_subtype=manual.get('subtype'),manual_evidence=manual,archived_announcements=matches.to_dict('records'),change_records=z.to_dict('records'))
  rows.append(rr)
 (run/'private/event_inventory_PRIVATE.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2,default=str))
 e.to_parquet(run/'private/event_inventory_PRIVATE.parquet',index=False)
 for n in ['financial.parquet','financial_facts.parquet','financial_manifest.json']:
  p=Path('/home/oliver/MainDiskData/early_warning_12m_server_isolated_20261025/prepared')/n;hashes[str(p)]=sha(p)
 (run/'audit/source_hashes_before.json').write_text(json.dumps(hashes,indent=2))
 counts={}
 for role,g in d.groupby('role'):counts[role]=dict(rows=len(g),groups=g.group.nunique(),target_events=g.loc[g.event==1,'event_id'].nunique(),competing_events=g.loc[g.event==2,'event_id'].nunique(),unknown_rows=int(((g.event==0)&~g.label_eligible).sum()))
 (run/'audit/freeze.json').write_text(json.dumps(dict(source_root=str(BASE),counts=counts,unique_events=len(e),archived_text_events=sum(bool(x['archived_announcements']) for x in rows),new_fits=0,models=['q2','unrestricted','M2_allsector'],seeds=[42,101,202],aggregation='mean probabilities; not mean metrics',budgets=[.2,.25],strict_asof='published_at < origin, never same day without verified intraday time',status=['PROVISIONAL','POST_EXPOSURE_EXPLORATORY']),indent=2))
 print(json.dumps(counts),len(e),'events',sum(bool(x['archived_announcements']) for x in rows),'with archived announcement candidates')
if __name__=='__main__':
 import sys;main(sys.argv[1])
