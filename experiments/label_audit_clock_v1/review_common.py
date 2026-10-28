"""Private evidence paths and streaming immutable-source checks; no fitting code."""
from pathlib import Path
import hashlib,json
PRIOR=Path('/home/oliver/MainDiskData/st_policy_alignment_v1_20261028T110421Z')
BASE=Path('/home/oliver/MainDiskData/xai_rsf_interpretable_baselines_v1_20261027_hw9gdrol')
EVENT=Path('/home/oliver/MainDiskData/first_st_followup_audit_20261025_v1')
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str))
def register(run,paths):
 p=Path(run)/'audit/sources_before.json';d=json.loads(p.read_text()) if p.exists() else {}
 for path in paths:
  path=Path(path).resolve();key=str(path)
  if key not in d:d[key]={'sha256':sha(path),'bytes':path.stat().st_size}
 write(p,d);return d

def initialize(run):
 run=Path(run);old=json.loads((PRIOR/'audit/source_hashes_before.json').read_text());paths=[]
 for name in old:
  p=Path(name)
  if (p.parent==BASE/'input' and ('reference_q2' in p.name or 'reference_unrestricted' in p.name or p.name in ['plan.json','M2_PRIVATE.npz','final_train_metadata_PRIVATE.parquet','final_eval_metadata_PRIVATE.parquet','final_train_PRIVATE.npz','final_eval_PRIVATE.npz'])) or (p.parent==BASE/'input/legacy') or (p.name=='forest_PRIVATE.json'):paths.append(p)
 for p in (PRIOR/'private').glob('*'):
  if p.is_file():paths.append(p)
 paths+=list((PRIOR/'audit').glob('*.json'))+list((PRIOR/'results').glob('*.csv'))
 paths+=[EVENT/'input'/n for n in ['save_audit_metadata.py','rebuild_candidate_audit.py','review160_event_decisions.json','quarterly_first_financial_audit_12m.json']]
 paths+=[EVENT/'assembly'/n for n in ['merged_events.parquet','itemchanges.parquet']]
 paths+=[Path(x['path']) for x in json.loads((PRIOR/'audit/m2_models_after_final_evaluation.json').read_text())['files']]
 d=register(run,paths)
 for name,v in d.items():
  if name in old:assert v['sha256']==old[name],name
 write(run/'audit/initial_freeze.json',{'files':len(d),'unchanged_from_prior':True,'fits':0})
 print('Frozen sources',len(d),flush=True)
if __name__=='__main__':
 import sys;initialize(sys.argv[1])
