import argparse,fcntl,hashlib,os,resource,socket,time,traceback
from contextlib import contextmanager
import pandas as pd
from crforest import CompetingRiskForest
from crforest.shrinkage import shrink_leaf_cifs
from experiments.post_st_level_coverage_v1.worker import verify_export
from experiments.post_st_cause_support_v1.verification import replay
from experiments.post_st_ablation_v1.verification import merge
from experiments.post_st_seed_ensemble_v1.common import average_predictions
from .common import *
@contextmanager
def memory_slot(root):
    # Preflight found about30GiB available on lemon; the prior6-slot cap
    # protected a different7GiB free-memory state. Slurm still owns CPU affinity.
    slots=16 if socket.gethostname()=='lemon'else 24
    folder=Path.home()/'.cache/xai-requested-seeds-slots'/hashlib.sha256(str(root).encode()).hexdigest()[:16]
    folder.mkdir(parents=True,exist_ok=True);start=time.perf_counter();held=None
    while held is None:
        for i in range(slots):
            f=(folder/f'{i}.lock').open('a')
            try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);held=f;break
            except BlockingIOError:f.close()
        if held is None:
            if time.perf_counter()-start>3300:raise TimeoutError('memory slot timeout')
            time.sleep(.1)
    try:yield
    finally:held.close()
def fit(a,root,out):
    assert a.seed in NEW_SEEDS
    tr=arrays(root/f'folds/{a.year}/train_PRIVATE.npz');ev=arrays(root/f'folds/{a.year}/eval_PRIVATE.npz')
    assert set(ev)=={'X','sample_ids'}and tr['X'].shape[1]==74
    cfg=configuration(a.seed,TREES);expected=pairing(tr['groups'],cfg,74)
    t=time.perf_counter();f=CompetingRiskForest(cfg).fit(tr['X'],tr['time'],tr['event'],tr['groups'],n_jobs=1);fit_seconds=time.perf_counter()-t
    for actual,row in zip(f.inbag_,expected):assert array_hash(actual)==row['multiplicity_sha256']
    t=time.perf_counter();sh=shrink_leaf_cifs(f,tr['X'],tr['time'],tr['event'],tr['groups'],10);shrink_seconds=time.perf_counter()-t
    checks={v:verify_export(m,tr,ev,out/v)for v,m in [('raw',f),('shrunk',sh)]}
    t=time.perf_counter();nd,ft,pa,le,independent=replay(sh,tr);replay_seconds=time.perf_counter()-t
    for name,rows in [('node_search',nd),('feature_coverage',ft),('pair_coverage',pa),('leaf_support',le)]:pd.DataFrame(rows).to_csv(out/f'{name}_PRIVATE.csv',index=False)
    write(out/'pairing_PRIVATE.json',expected)
    return dict(year=a.year,seed=a.seed,trees=32,forest_fits=1,new_trees=32,fit_seconds=fit_seconds,shrink_seconds=shrink_seconds,replay_seconds=replay_seconds,checks=checks,independent=independent,
        actual_scored_candidates=sum(z['scored_candidates']for z in nd),guard_rejected_candidates=sum(z['guard_rejected']for z in nd))
def ensemble(a,root,out):
    def forbidden(*args,**kwargs):raise RuntimeError('No fitting in ensemble export')
    CompetingRiskForest.fit=forbidden
    tr=arrays(root/f'folds/{a.year}/train_PRIVATE.npz');ev=arrays(root/f'folds/{a.year}/eval_PRIVATE.npz');assert set(ev)=={'X','sample_ids'}
    assert a.version in ['raw','shrunk']
    checks={};errors={}
    for v in [a.version]:
        members=[];pred=[]
        for seed in SEEDS:
            p=root/f'members/{a.year}/{seed}/{v}';m=CompetingRiskForest.load(p/'forest_PRIVATE.json')
            assert len(m.trees_)==32 and m.n_features_==74 and m.config.path_budget==2
            members.append(m);pred.append(arrays(p/'predictions_PRIVATE.npz'))
        f=merge(members);assert len(f.trees_)==512
        checks[v]=verify_export(f,tr,ev,out/v)
        mean=average_predictions(pred);actual=arrays(out/v/'predictions_PRIVATE.npz');np.testing.assert_array_equal(actual['sample_ids'],pred[0]['sample_ids'])
        err=float(abs(mean-actual['cif']).max());assert err<=1e-12
        X=np.vstack([tr['X'],ev['X']]);times=np.unique(np.r_[0.,f.grid_,TIMES]);full=0.
        for begin in range(0,len(X),32):
            xx=X[begin:begin+32];full=max(full,float(abs(f.predict(xx,times)-np.mean([m.predict(xx,times)for m in members],axis=0)).max()))
        assert full<=1e-12;errors[v]=dict(saved_mean_error=err,all_row_time_mean_error=full)
    return dict(year=a.year,seed='ensemble512',version=a.version,trees=512,forest_fits=0,new_trees=0,checks=checks,mean_checks=errors)
def main(a):
    root=Path(os.environ['DATA_DIR']);out=Path(os.environ['OUTPUT_DIR']);out.mkdir(parents=True,exist_ok=True)
    limit=(2 if a.mode=='fit'else 4)*1024**3;resource.setrlimit(resource.RLIMIT_AS,(limit,limit));start=time.perf_counter()
    with memory_slot(root):
        waited=time.perf_counter()-start;summary=(fit if a.mode=='fit'else ensemble)(a,root,out)
        summary.update(status='COMPLETE',mode=a.mode,calibration_fits=0,host=socket.gethostname(),slurm_job=os.environ.get('SLURM_JOB_ID'),seconds=time.perf_counter()-start,
            memory_slot_wait_seconds=waited,peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,no_evaluation_labels=True)
        write(out/'summary.json',summary);write(out/'hashes.json',{str(p.relative_to(out)):sha(p)for p in out.rglob('*')if p.is_file()})
    print(json.dumps(dict(status='COMPLETE',mode=a.mode,year=a.year,seed=summary['seed'],forest_fits=summary['forest_fits'])),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['fit','ensemble'],required=True);p.add_argument('--year',type=int,choices=YEARS,required=True);p.add_argument('--seed',type=int,choices=NEW_SEEDS);p.add_argument('--version',choices=['raw','shrunk']);a=p.parse_args()
    try:main(a)
    except Exception:
        out=Path(os.environ['OUTPUT_DIR']);out.mkdir(parents=True,exist_ok=True);write(out/'FAILED.json',dict(args=vars(a),traceback=traceback.format_exc()));raise
