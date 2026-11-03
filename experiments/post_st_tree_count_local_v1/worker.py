"""Read-only exact-prefix export. Fitting is actively disabled in this process."""
import argparse,fcntl,os,resource,socket,time,traceback,hashlib
from contextlib import contextmanager
from crforest import CompetingRiskForest
from experiments.post_st_ablation_v1.verification import merge
from experiments.post_st_level_coverage_v1.worker import verify_export
from experiments.post_st_tree_count_v1.nested import prefix
from .common import *

def forbidden_fit(*args,**kwargs):raise RuntimeError('This experiment forbids model fitting')

@contextmanager
def memory_slot(root):
    slots=8 if socket.gethostname()=='lemon'else 24
    key=hashlib.sha256(str(root).encode()).hexdigest()[:16]
    d=Path.home()/'.cache'/'xai-tree-local-slots'/key;d.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();held=None
    while held is None:
        for i in range(slots):
            f=(d/f'{i}.lock').open('a')
            try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);held=f;break
            except BlockingIOError:f.close()
        if held is None:
            if time.perf_counter()-start>3300:raise TimeoutError('memory slot wait')
            time.sleep(.1)
    try:yield
    finally:held.close()

def main(a):
    import crforest.core
    CompetingRiskForest.fit=forbidden_fit;crforest.core._grow_tree=forbidden_fit
    root=Path(os.environ['DATA_DIR']);out=Path(os.environ['OUTPUT_DIR']);out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
    with memory_slot(root):
        waited=time.perf_counter()-start
        tr=arrays(root/f'folds/{a.year}/train_PRIVATE.npz');ev=arrays(root/f'folds/{a.year}/eval_PRIVATE.npz')
        assert set(ev)=={'X','sample_ids'}and tr['X'].shape[1]==74
        manifest=load(root/'data_hashes.json')
        names=[f'folds/{a.year}/{v}_PRIVATE.npz'for v in ['train','eval']]
        seeds=SEEDS if a.seed==0 else [a.seed]
        for s in seeds:names.extend(f'parents/{a.year}/{s}/{n}'for n in ['raw_PRIVATE.json','shrunk_PRIVATE.json','tree_counts_PRIVATE.json'])
        for n in names:assert sha(root/n)==manifest[n]
        checks={}
        for version in ['raw','shrunk']:
            members=[]
            for seed in seeds:
                p=root/f'parents/{a.year}/{seed}'
                f=CompetingRiskForest.load(p/f'{version}_PRIVATE.json')
                members.append(prefix(f,a.count,load(p/'tree_counts_PRIVATE.json')))
            f=merge(members)if a.seed==0 else members[0]
            assert len(f.trees_)==a.count*len(members)
            error=float(abs(f.predict(ev['X'],TIMES)-np.mean([m.predict(ev['X'],TIMES)for m in members],axis=0)).max())
            assert error<=1e-12
            checks[version]=dict(**verify_export(f,tr,ev,out/version),ensemble_mean_error=error)
        write(out/'summary.json',dict(status='COMPLETE',year=a.year,seed=a.seed,count=a.count,
            host=socket.gethostname(),slurm_job=os.environ.get('SLURM_JOB_ID'),seconds=time.perf_counter()-start,
            memory_slot_wait_seconds=waited,peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            forest_fits=0,calibration_fits=0,new_trees=0,no_evaluation_labels=True,checks=checks))
        write(out/'hashes.json',{str(p.relative_to(out)):sha(p)for p in out.rglob('*')if p.is_file()})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--year',type=int,choices=YEARS,required=True)
    p.add_argument('--seed',type=int,choices=[0,*SEEDS],required=True);p.add_argument('--count',type=int,choices=COUNTS,required=True);a=p.parse_args()
    try:main(a)
    except Exception:
        out=Path(os.environ['OUTPUT_DIR']);out.mkdir(parents=True,exist_ok=True)
        write(out/'FAILED.json',dict(args=vars(a),traceback=traceback.format_exc()));raise

