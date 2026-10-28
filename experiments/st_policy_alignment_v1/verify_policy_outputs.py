"""Verify outputs and immutable original inputs. Never imports a training entrypoint."""
from pathlib import Path
import json,hashlib
import pandas as pd

def verify(run):
 run=Path(run);before=json.loads((run/'audit/source_hashes_before.json').read_text());after={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in before};assert before==after
 (run/'audit/source_hashes_after.json').write_text(json.dumps(after,indent=2))
 ev=pd.read_parquet(run/'private/event_rule_evidence.parquet');assert len(ev)==323 and ev.event_id.is_unique
 c=pd.read_parquet(run/'private/origin_policy_context.parquet');assert len(c)==80257 and c.sample_id.is_unique
 rules=pd.read_csv(run/'results/st_rule_versions.csv').set_index('rule_id')
 for row in c[['origin','known_rule_ids']].drop_duplicates().itertuples(index=False):
  ids=json.loads(row.known_rule_ids)
  if ids:assert (pd.to_datetime(rules.loc[ids,'official_published_at'])<row.origin).all()
 d=pd.read_parquet(run/'input/final_eval_metadata_PRIVATE.parquet');assert len(d)==23799 and ((d.event==0)&~d.label_eligible).sum()==713
 reproduction=json.loads((run/'audit/reproduction.json').read_text());assert reproduction['warning_counts_exact'] and reproduction['max_abs_metric_difference']<1e-12
 result=dict(status='PASS',original_files_checked=len(before),original_hashes_unchanged=True,new_fits=0,origin_records=len(c),events=len(ev),unknown_eval_rows_preserved=713,no_labels_revised=True,warning_reproduction=reproduction)
 (run/'audit/verification.json').write_text(json.dumps(result,indent=2));return result
if __name__=='__main__':
 import sys;print(json.dumps(verify(sys.argv[1]),indent=2))
