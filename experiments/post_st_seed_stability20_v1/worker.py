import argparse,fcntl,os,resource,socket,time,traceback,hashlib
from contextlib import contextmanager
import pandas as pd
from crforest import CompetingRiskForest
from crforest.shrinkage import shrink_leaf_cifs
from experiments.post_st_level_coverage_v1.worker import verify_export
from experiments.post_st_cause_support_v1.verification import replay
from .common import *

@contextmanager
def memory_slot(root):
    slots=6 if socket.gethostname()=='lemon'else 24
    d=Path.home()/'.cache/xai-seed20-slots'/hashlib.sha256(str(root).encode()).hexdigest()[:16];d.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();held=None
    while held is None:
        for i in range(slots):
            f=(d/f'{i}.lock').open('a')
            try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);held=f;break
            except BlockingIOError:f.close()
        if held is None:
            if time.perf_counter()-start>3300:raise TimeoutError('memory slot timeout')
            time.sleep(.1)
    try:yield
    finally:held.close()

def main(a):
    root=Path(os.environ['DATA_DIR']);out=Path(os.environ['OUTPUT_DIR']);out.mkdir(parents=True,exist_ok=True)
    resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3));start=time.perf_counter()
    with memory_slot(root):
        waited=time.perf_counter()-start;tr=arrays(root/f'folds/{a.year}/train_PRIVATE.npz');ev=arrays(root/f'folds/{a.year}/eval_PRIVATE.npz')
        assert set(ev)=={'X','sample_ids'}and tr['X'].shape[1]==74
        cfg=configuration(a.seed,TREES);expected=pairing(tr['groups'],cfg,74);reused=a.seed in REUSED_SEEDS
        fit_seconds=shrink_seconds=0.
        if reused:
            p=root/f'reused/{a.year}/{a.seed}';f=CompetingRiskForest.load(p/'raw_PRIVATE.json');sh=CompetingRiskForest.load(p/'shrunk_PRIVATE.json')
            assert asdict(f.config)==asdict(cfg)and len(f.trees_)==TREES
            t=time.perf_counter();fresh=shrink_leaf_cifs(f,tr['X'],tr['time'],tr['event'],tr['groups'],10)
            assert json.dumps(fresh.trees_)==json.dumps(sh.trees_);del fresh
        else:
            t=time.perf_counter();f=CompetingRiskForest(cfg).fit(tr['X'],tr['time'],tr['event'],tr['groups'],n_jobs=1);fit_seconds=time.perf_counter()-t
            for actual,row in zip(f.inbag_,expected):assert array_hash(actual)==row['multiplicity_sha256']
            t=time.perf_counter();sh=shrink_leaf_cifs(f,tr['X'],tr['time'],tr['event'],tr['groups'],10);shrink_seconds=time.perf_counter()-t
        checks={v:verify_export(m,tr,ev,out/v)for v,m in [('raw',f),('shrunk',sh)]}
        t=time.perf_counter();nd,ft,pa,le,independent=replay(sh,tr);replay_seconds=time.perf_counter()-t
        for name,rows in [('node_search',nd),('feature_coverage',ft),('pair_coverage',pa),('leaf_support',le)]:pd.DataFrame(rows).to_csv(out/f'{name}_PRIVATE.csv',index=False)
        write(out/'pairing_PRIVATE.json',expected)
        summary=dict(status='COMPLETE',year=a.year,seed=a.seed,trees=TREES,reused=reused,forest_fits=int(not reused),new_trees=0 if reused else TREES,calibration_fits=0,
            fit_seconds=fit_seconds,shrink_seconds=shrink_seconds,replay_seconds=replay_seconds,checks=checks,independent=independent,
            actual_scored_candidates=sum(r['scored_candidates']for r in nd),guard_rejected_candidates=sum(r['guard_rejected']for r in nd),
            host=socket.gethostname(),slurm_job=os.environ.get('SLURM_JOB_ID'),seconds=time.perf_counter()-start,memory_slot_wait_seconds=waited,
            peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,no_evaluation_labels=True,training_sha256=sha(root/f'folds/{a.year}/train_PRIVATE.npz'))
        write(out/'summary.json',summary);write(out/'hashes.json',{str(p.relative_to(out)):sha(p)for p in out.rglob('*')if p.is_file()})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--year',type=int,choices=YEARS,required=True);p.add_argument('--seed',type=int,choices=SEEDS,required=True);a=p.parse_args()
    try:main(a)
    except Exception:
        out=Path(os.environ['OUTPUT_DIR']);out.mkdir(parents=True,exist_ok=True);write(out/'FAILED.json',dict(args=vars(a),traceback=traceback.format_exc()));raise
