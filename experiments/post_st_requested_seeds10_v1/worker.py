import argparse,os,resource,socket,time,traceback
from crforest import CompetingRiskForest
from experiments.post_st_level_coverage_v1.worker import verify_export
from experiments.post_st_ablation_v1.verification import merge
from experiments.post_st_seed_ensemble_v1.common import average_predictions
from .common import *
from experiments.post_st_requested_seeds14_v1.worker import memory_slot

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
        f=merge(members);assert len(f.trees_)==ENSEMBLE_TREES
        checks[v]=verify_export(f,tr,ev,out/v)
        mean=average_predictions(pred);actual=arrays(out/v/'predictions_PRIVATE.npz');np.testing.assert_array_equal(actual['sample_ids'],pred[0]['sample_ids'])
        err=float(abs(mean-actual['cif']).max());assert err<=1e-12
        X=np.vstack([tr['X'],ev['X']]);times=np.unique(np.r_[0.,f.grid_,TIMES]);full=0.
        for begin in range(0,len(X),32):
            xx=X[begin:begin+32];full=max(full,float(abs(f.predict(xx,times)-np.mean([m.predict(xx,times)for m in members],axis=0)).max()))
        assert full<=1e-12;errors[v]=dict(saved_mean_error=err,all_row_time_mean_error=full)
    return dict(year=a.year,seed=ENSEMBLE,version=a.version,trees=ENSEMBLE_TREES,forest_fits=0,new_trees=0,checks=checks,mean_checks=errors)
def main(a):
    root=Path(os.environ['DATA_DIR']);out=Path(os.environ['OUTPUT_DIR']);out.mkdir(parents=True,exist_ok=True)
    limit=4*1024**3;resource.setrlimit(resource.RLIMIT_AS,(limit,limit));start=time.perf_counter()
    with memory_slot(root):
        waited=time.perf_counter()-start;summary=ensemble(a,root,out)
        summary.update(status='COMPLETE',mode=a.mode,calibration_fits=0,host=socket.gethostname(),slurm_job=os.environ.get('SLURM_JOB_ID'),seconds=time.perf_counter()-start,
            memory_slot_wait_seconds=waited,peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,no_evaluation_labels=True)
        write(out/'summary.json',summary);write(out/'hashes.json',{str(p.relative_to(out)):sha(p)for p in out.rglob('*')if p.is_file()})
    print(json.dumps(dict(status='COMPLETE',mode=a.mode,year=a.year,seed=summary['seed'],forest_fits=summary['forest_fits'])),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['ensemble'],required=True);p.add_argument('--year',type=int,choices=YEARS,required=True);p.add_argument('--version',choices=['raw','shrunk']);a=p.parse_args()
    try:main(a)
    except Exception:
        out=Path(os.environ['OUTPUT_DIR']);out.mkdir(parents=True,exist_ok=True);write(out/'FAILED.json',dict(args=vars(a),traceback=traceback.format_exc()));raise
