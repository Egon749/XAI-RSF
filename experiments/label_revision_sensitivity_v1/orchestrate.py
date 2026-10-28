"""Build finite independent manifests and verify pool publication, not just job exit."""
import argparse,json,shutil
from pathlib import Path
from support import write,sha

def manifest(run,stage,commands,source=None):
 p=run/'audit'/f'{stage}.jsonl'
 tasks=[dict(kind='cpu',argv=['/usr/bin/python3',str(run/'driver/launch.py'),*args],cwd=str(run/'input/frozen_code'),timeout=7200,storage=dict(source=str(source or run/'input'),mode='nfs',results=str(run/'results'/stage))) for args in commands]
 p.write_text(''.join(json.dumps(t)+'\n' for t in tasks));return p

def outputs(run,stage,expected):
 folders=sorted((run/'results'/stage).glob('*/task-*/output'));assert len(folders)==expected,(stage,len(folders),expected)
 for p in folders:
  h=json.loads((p/'output_hashes.json').read_text())
  for name,digest in h.items():assert sha(p/name)==digest,(p,name)
 return folders

def main(run,stage):
 run=Path(run)
 if stage=='prepare':
  (run/'driver').mkdir(exist_ok=True)
  for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,run/'driver'/p.name)
  plan=json.loads((run/'input/plan.json').read_text());counts=json.loads((run/'input/patch_application_receipt.json').read_text())
  plan['label_version']='L1_ADJUDICATED';plan['source_counts']['final_train']={'records':56458,'groups':2577,'target_events':174,'competing_events':34,'target_rows':642,'competing_rows':129}
  write(run/'input/L1_ADJUDICATED/plan.json',plan)
  manifest(run,'control',[['--mode','control','--seed',str(s),'--variant',v] for s in [42,101,202] for v in ['coverage','unrestricted']])
 elif stage=='train':
  folders=outputs(run,'control',6)
  assert all(json.loads((p/'summary.json').read_text())['reload_error']==0 for p in folders)
  manifest(run,'train',[['--mode','shard','--seed',str(s),'--variant',v,'--tree-start',str(i)] for s in [42,101,202] for v in ['coverage','unrestricted'] for i in range(0,32,4)])
 elif stage=='assemble':
  folders=outputs(run,'train',48);index={}
  for p in folders:
   d=json.loads((p/'shard_summary.json').read_text());key=f"final/{d['seed']}/{d['variant']}";index.setdefault(key,[]).append(dict(path=str(p),sha256=d['model_sha256']))
  write(run/'audit/shard_index.json',index)
  manifest(run,'assemble',[['--mode','assemble','--seed',str(s),'--variant',v,'--index',str(run/'audit/shard_index.json')] for s in [42,101,202] for v in ['coverage','unrestricted']])
 elif stage=='explain':
  index={}
  for label,st in [('L0','control'),('L1','assemble')]:
   for p in outputs(run,st,6):
    d=json.loads((p/'summary.json').read_text());m='q2' if d['variant']=='coverage' else 'unrestricted';index[f"{label}/{m}/{d['seed']}"]=str(p)
  write(run/'audit/models_index.json',index)
  manifest(run,'explain',[['--mode','explain','--seed',str(s),'--version',v] for v in ['L0','L1'] for s in [42,101,202,-1]])
 elif stage=='bootstrap':manifest(run,'bootstrap',[['--mode','bootstrap','--batch',str(i)] for i in range(10)],run/'bootstrap_input')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('stage');a=p.parse_args();main(a.run,a.stage)
