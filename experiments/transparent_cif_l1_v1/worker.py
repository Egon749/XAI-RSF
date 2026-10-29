"""Call existing entrypoints in distinct stages; all outputs uniquely owned."""
from pathlib import Path
import argparse,json,os,resource,socket,sys,time,traceback
from crforest.transparent import file_sha256 as sha,write_json
SOURCE=Path('/home/oliver/MainDiskData/label_revision_sensitivity_v1_20261028T151804Z/handoff_L1')
def main(a):
 start=time.perf_counter();src=Path(os.environ['DATA_DIR']);out=Path(os.environ['OUTPUT_DIR']);run=src.parent
 import instrument;instrument.install()
 try:
  if a.stage=='demo':
   from experiments.transparent_cif_v2 import demo
   sys.argv=['demo','--output',str(out/'demo')];demo.main()
  elif a.stage in ['train','predict']:
   from experiments.transparent_cif_v2 import run as entry
   args=[a.stage,'--source-input',str(SOURCE),'--output',str(out/a.stage)]
   if a.stage=='train':
    args+=['--features',str(SOURCE/'features.json'),'--config',str(src/'frozen_config.json'),'--seed',str(a.seed),'--tree-workers','1','--ebm-control']
    old=entry.read_metadata
    def train_only(source,role,*args,**kwargs):
     if role!='train':raise RuntimeError('Training attempted evaluation metadata access')
     return old(source,role,*args,**kwargs)
    entry.read_metadata=train_only
   else:
    frozen=json.loads((run/'audit/fits_frozen.json').read_text());fit=Path(frozen['seed_paths'][str(a.seed)])
    for name,h in frozen['file_hashes'][str(a.seed)].items():assert sha(fit/name)==h
    args+=['--trained',str(fit)]
   sys.argv=['run',*args];entry.main()
  elif a.stage=='bootstrap':
   from experiments.label_revision_sensitivity_v1.bootstrap_worker import run as boot
   boot(src,out,a.batch)
  elif a.stage=='audit':
   from experiments.transparent_cif_l1_v1.audit_native import main as audit
   audit(run,out,a.model,a.seed)
  else:raise ValueError(a.stage)
  write_json(out/('task_summary.json' if a.stage=='bootstrap' else 'summary.json'),dict(stage=a.stage,seed=a.seed,model=a.model,status='COMPLETE',host=socket.gethostname(),seconds=time.perf_counter()-start,peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,affinity_count=len(os.sched_getaffinity(0))))
 finally:instrument.save(out/'calls.json')
 write_json(out/'hashes.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage');p.add_argument('--seed',type=int,default=-1);p.add_argument('--model');p.add_argument('--batch',type=int);a=p.parse_args()
 try:main(a)
 except Exception:
  write_json(Path(os.environ['OUTPUT_DIR'])/'FAILED.json',dict(args=vars(a),traceback=traceback.format_exc()));raise
