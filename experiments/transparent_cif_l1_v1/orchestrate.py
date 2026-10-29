"""Manifest construction and strict successful-publication gates."""
from pathlib import Path
import json,sys,subprocess,tarfile
from crforest.transparent import file_sha256 as sha,write_json
MODELS=['base_equal','base_weighted','ordinary_extra_equal','ordinary_extra_weighted','pair_equal','pair_weighted']
def manifest(run,stage,commands,source=None):
 rows=[dict(kind='cpu',argv=['/usr/bin/python3',str(run/'input/code/experiments/transparent_cif_l1_v1/launch.py'),*args],cwd=str(run/'input/code'),timeout=14400,storage=dict(source=str(source or run/'input'),mode='nfs',results=str(run/'results'/stage))) for args in commands]
 (run/'audit'/f'{stage}.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
 write_json(run/'audit'/f'{stage}.py.hetero.json',dict(independent=True,manifest=f'{stage}.jsonl'))
 (run/'audit'/f'{stage}.py').write_text('# Independent Slurm task manifest.\n')
def gate(run,stage,n):
 report=next((run/'audit').glob(stage+'.jsonl.results-*.json'));r=json.loads(report.read_text());assert r['expected']==n and len(r['completed'])==n and not r['pending']
 assert {x['id'] for x in r['completed']}==set(range(n));out=[]
 for item in sorted(r['completed'],key=lambda x:x['id']):
  assert item['returncode']==0 and item['compute_completed'] and item['publication_completed'],item
  p=Path(item['storage']['result'])
  for name,digest in json.loads((p/'hashes.json').read_text()).items():assert sha(p/name)==digest
  out.append(p)
 return out

def main(run,stage):
 if stage=='freeze':
  repo=Path.cwd();assert not subprocess.check_output(['git','status','--porcelain'],text=True)
  commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
  with (run/'audit/code.tar').open('xb') as f:subprocess.run(['git','archive',commit],stdout=f,check=True)
  with tarfile.open(run/'audit/code.tar') as f:f.extractall(run/'input/code',filter='data')
  write_json(run/'audit/code_freeze.json',dict(training_sha=commit,code_archive_sha256=sha(run/'audit/code.tar'),config_sha256=sha(run/'input/frozen_config.json'),inference_sha256=sha(run/'input/inference_protocol.json')))
  manifest(run,'demo',[['demo']])
 elif stage=='train':
  gate(run,'demo',1)
  manifest(run,'train_oliver',[['train','--seed',str(s)] for s in [42,101]])
  manifest(run,'train_lemon',[['train','--seed','202']])
 elif stage=='predict':
  outputs=gate(run,'train_oliver',2)+gate(run,'train_lemon',1);paths={};hashes={}
  for p in outputs:
   s=json.loads((p/'summary.json').read_text());seed=str(s['seed']);fit=p/'train';r=json.loads((fit/'execution_receipt.json').read_text());assert r['base_forest_fits']==1 and r['weight_qp_solves']==3 and r['ebm_candidate_fits']==6
   paths[seed]=str(fit);hashes[seed]={str(f.relative_to(fit)):sha(f) for f in fit.rglob('*') if f.is_file()}
  write_json(run/'audit/fits_frozen.json',dict(seed_paths=paths,file_hashes=hashes,all_three_complete_before_predict=True,all_eighteen_models_frozen_before_evaluation=True))
  manifest(run,'predict',[['predict','--seed',str(s)] for s in [42,101,202]])
 elif stage=='audit':manifest(run,'native_audit',[['audit','--seed',str(s),'--model',m] for m in MODELS for s in [42,101,202,-1]])
 elif stage=='bootstrap':manifest(run,'bootstrap',[['bootstrap','--batch',str(i)] for i in range(10)],run/'bootstrap_input')
if __name__=='__main__':main(Path(sys.argv[1]),sys.argv[2])
