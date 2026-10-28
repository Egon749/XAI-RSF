"""Validate a private proposed patch against originals without writing modified inputs."""
from pathlib import Path
import pandas as pd,numpy as np,json
from review_common import sha,write,BASE

def validate(patch,original,metadata_hash,expected_hash):
 if metadata_hash!=expected_hash:raise ValueError('source hash mismatch')
 if patch.sample_id.duplicated().any():raise ValueError('duplicate patch windows')
 if not patch.old_event.eq(1).all() or not patch.new_event.eq(2).all():raise ValueError('only target to competing allowed; never normal')
 for eid,g in patch.groupby('event_id'):
  source=original[original.event_id.eq(eid)]
  if set(g.sample_id)!=set(source.sample_id):raise ValueError('incomplete event-window linkage')
  joined=g.merge(source,on='sample_id',suffixes=('_patch','_source'),validate='one_to_one')
  for f in ['event_id','event_date','announcement_date','outcome_available','stop_date','duration','origin','horizon','group']:
   if not joined[f+'_patch'].equals(joined[f+'_source']):raise ValueError('changed event contract: '+f)
  if not source.event.eq(1).all():raise ValueError('old label mismatch')
 return True

def main(run):
 run=Path(run);p=pd.read_parquet(run/'private/label_patch_PRIVATE.parquet');source=BASE/'input/final_train_metadata_PRIVATE.parquet';d=pd.read_parquet(source);arr=BASE/'input/final_train_PRIVATE.npz';z=np.load(arr);m=json.loads((run/'private/label_patch_manifest.json').read_text())
 assert sha(run/'private/label_patch_PRIVATE.parquet')==m['patch_sha256'];assert sha(arr)==m['array_sha256'];validate(p,d,sha(source),m['metadata_sha256']);np.testing.assert_array_equal(z['sample_ids'],d.sample_id)
 assert (z['event'][p.array_row.to_numpy()]==p.old_event.to_numpy()).all()
 proposed=d.event.copy();proposed.loc[p.array_row]=p.new_event.to_numpy();assert (proposed!=d.event).sum()==len(p)
 receipt={'validated':True,'patch_applied':False,'events':p.event_id.nunique(),'groups':p.group.nunique(),'quarter_records':len(p),'original_target_events':d.loc[d.event.eq(1),'event_id'].nunique(),'proposed_target_events':d.loc[proposed.eq(1),'event_id'].nunique(),'original_competing_events':d.loc[d.event.eq(2),'event_id'].nunique(),'proposed_competing_events':d.loc[proposed.eq(2),'event_id'].nunique(),'original_target_rows':int(d.event.eq(1).sum()),'proposed_target_rows':int(proposed.eq(1).sum()),'original_competing_rows':int(d.event.eq(2).sum()),'proposed_competing_rows':int(proposed.eq(2).sum()),'riskset_and_dates_unchanged':True,'future_dependency':'Rebuild derived status/previous_label/label_observed_first consistently in a new version; do not just edit metadata or NPZ alone; do not add later events.'}
 write(run/'audit/label_patch_validation.json',receipt);print(json.dumps(receipt),flush=True)
if __name__=='__main__':
 import sys;main(sys.argv[1])
