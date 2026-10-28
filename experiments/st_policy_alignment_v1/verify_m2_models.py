"""Supplementary byte checks of original M2 RDS; never load/refit an R model."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys

def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main(run,stage):
 run=Path(run);src=run/'input/plan.json';parent=Path(json.loads(src.read_text())['parent']);prov=json.loads((parent/'audit/M2_provenance.json').read_text());rows=[]
 for r in prov:
  p=Path(r['path']);assert digest(p)==r['sha256']
  for name in ['forest.rds','predictions.npz','preprocessing.pkl']:
   file=p.parent/name;rows.append(dict(seed=r['seed'],path=str(file),bytes=file.stat().st_size,sha256=digest(file)))
 receipt=dict(stage=stage,timestamp_utc=datetime.now(timezone.utc).isoformat(),note='Supplementary check before/after final diagnostic recomputation; not retrospectively claimed as initial session hash',files=rows)
 if stage=='after':
  before=json.loads((run/'audit/m2_models_before_final_evaluation.json').read_text());assert before['files']==rows;receipt['unchanged']=True
 (run/f'audit/m2_models_{stage}_final_evaluation.json').write_text(json.dumps(receipt,indent=2));print(stage,len(rows),'M2 original artifacts checked',flush=True)
if __name__=='__main__':main(sys.argv[1],sys.argv[2])
