"""Compile and verify saved 96-tree mean predictors; never fit a forest."""
import argparse,os,resource,socket,time,traceback
from pathlib import Path
import numpy as np
from crforest import CompetingRiskForest
from experiments.post_st_ablation_v1.verification import merge
from experiments.post_st_level_coverage_v1.worker import verify_export
from .common import *

def main(a):
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3));start=time.perf_counter()
    root=Path(os.environ['DATA_DIR']);out=Path(os.environ['OUTPUT_DIR']);out.mkdir(parents=True,exist_ok=True)
    rows=[r for r in load(root/'registry_formal.json') if r['year']==a.year and r['arm']==a.arm]
    assert sorted(r['seed'] for r in rows)==SEEDS
    models=[];pred=[];hashes={}
    for r in sorted(rows,key=lambda r:r['seed']):
        p=Path(r['path']);expected=load(p/'hashes.json')
        for n in ['forest_PRIVATE.json','predictions_PRIVATE.npz']:
            file=p/a.version/n;assert sha(file)==expected[a.version+'/'+n];hashes[str(file)]=sha(file)
        models.append(CompetingRiskForest.load(p/a.version/'forest_PRIVATE.json'))
        pred.append(arrays(p/a.version/'predictions_PRIVATE.npz')['cif'])
    f=merge(models);tr=arrays(root/f'folds/{a.year}/train_PRIVATE.npz');ev=arrays(root/f'folds/{a.year}/eval_PRIVATE.npz')
    error=float(abs(f.predict(ev['X'],TIMES)-np.mean(pred,axis=0)).max());assert error<1e-12
    check=verify_export(f,tr,ev,out/'representation')
    write(out/'summary.json',dict(status='COMPLETE',year=a.year,arm=a.arm,version=a.version,seed='ensemble',
        new_forests=0,checks=check,mean_prediction_error=error,source_hashes=hashes,
        seconds=time.perf_counter()-start,peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        host=socket.gethostname(),slurm_job=os.environ.get('SLURM_JOB_ID')))
    write(out/'hashes.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--year',type=int,required=True);p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--version',choices=['raw','shrunk'],required=True);a=p.parse_args()
    try:main(a)
    except Exception:
        out=Path(os.environ['OUTPUT_DIR']);out.mkdir(parents=True,exist_ok=True);write(out/'FAILED.json',dict(args=vars(a),traceback=traceback.format_exc()));raise
