"""Recover already-computed numeric draws after a wrapper-only summary collision.
Never invokes fitting, prediction, bootstrap evaluation or random resampling.
"""
from pathlib import Path
import json,sys,os,shutil,hashlib,subprocess,shlex
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def prepare(run):
 src=run/'recovery_input';src.mkdir();report=json.loads((run/'audit/bootstrap.jsonl.results-2565.json').read_text());assert len(report['completed'])==10
 receipts=[];allids=[];first=None
 for row in sorted(report['completed'],key=lambda x:x['id']):
  assert row['returncode']==1 and 'FileExistsError' in row['tail'] and 'summary.json' in row['tail']
  source=Path(row['storage']['output']);assert '/.cache/heteropool-work/2565/' in str(source);dest=src/str(row['id']);digests={}
  for name in ['bootstrap.npz','summary.json','calls.json','FAILED.json']:
   p=source/name
   digests[name]=subprocess.check_output(['ssh','lemon','sha256sum -- '+shlex.quote(str(p))],text=True).split()[0] if row['host']=='lemon' else sha(p)
  if row['host']=='lemon':dest.mkdir();subprocess.run(['rsync','-a',f'lemon:{source}/',str(dest)+'/'],check=True)
  else:shutil.copytree(source,dest)
  for name,h in digests.items():assert sha(dest/name)==h
  z=dict(np.load(dest/'bootstrap.npz'));summary=json.loads((dest/'summary.json').read_text());assert summary['status']=='COMPLETE' and summary['batch']==row['id'] and summary['replicates']==100
  assert np.array_equal(z['replicate_ids'],np.arange(row['id']*100,(row['id']+1)*100));assert len(z['replicates'])==100 and len(z['resampling_hashes'])==100
  assert np.isfinite(z['point']).all() and np.isfinite(z['replicates']).all()
  if first is None:first=z
  else:assert np.array_equal(z['names'],first['names']) and np.array_equal(z['point'],first['point'])
  allids.extend(z['replicate_ids'].tolist());receipts.append(dict(task=row['id'],original_node=row['host'],original_output=str(source),source_hashes=digests,numeric_complete=True,reason='backend completed; outer summary exclusive-create collision',numeric_recomputed=False))
 assert sorted(allids)==list(range(1000))
 (src/'identity_verification.json').write_text(json.dumps(receipts,indent=2));(run/'audit/bootstrap_recovery_plan.json').write_text(json.dumps(dict(original_job=2565,failed_wrapper_tasks=10,complete_numeric_replicates=1000,new_fits=0,new_bootstrap_draws=0,identity_verified=True),indent=2))
 # The original failed job and scratch remain untouched.
 script=run/'audit/recover_bootstrap.py';shutil.copy2(__file__,script)
 launch=run/'audit/recovery_launch.py';launch.write_text("import os,sys,socket\nfrom pathlib import Path\np='/home/oliver/.venvs/xai-baselines-20261027/bin/python' if socket.gethostname().startswith('oliver') else '/home/lemon/.venvs/xai-baselines-py313-20261027/bin/python'\nos.execv(p,[p,str(Path(__file__).with_name('recover_bootstrap.py')),'publish',*sys.argv[1:]])\n")
 tasks=[dict(kind='cpu',argv=['/usr/bin/python3',str(launch),str(i)],timeout=600,storage=dict(source=str(src),mode='nfs',results=str(run/'results/bootstrap_recovery'))) for i in range(10)]
 (run/'audit/bootstrap_recovery.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in tasks))

def publish(i):
 src=Path(os.environ['DATA_DIR']);out=Path(os.environ['OUTPUT_DIR']);r=next(x for x in json.loads((src/'identity_verification.json').read_text()) if x['task']==i)
 for name,digest in r['source_hashes'].items():
  p=src/str(i)/name;assert sha(p)==digest;shutil.copy2(p,out/('original_failure.json' if name=='FAILED.json' else name))
 (out/'recovery_receipt.json').write_text(json.dumps(r,indent=2));(out/'hashes.json').write_text(json.dumps({p.name:sha(p) for p in out.iterdir() if p.is_file()},indent=2))
if __name__=='__main__':prepare(Path(sys.argv[2])) if sys.argv[1]=='prepare' else publish(int(sys.argv[2]))
