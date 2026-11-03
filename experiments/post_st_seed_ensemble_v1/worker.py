"""Compile and verify fixed saved forests without any fit."""
import argparse,os,resource,socket,time,traceback
from crforest import CompetingRiskForest
from experiments.post_st_ablation_v1.verification import merge
from experiments.post_st_level_coverage_v1.worker import verify_export
from .common import *
def main(a):
    root=Path(os.environ['DATA_DIR']);out=Path(os.environ['OUTPUT_DIR']);out.mkdir(parents=True,exist_ok=True)
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3));start=time.perf_counter()
    def forbidden(*args,**kwargs):raise RuntimeError('Forest fitting forbidden in saved-prediction ensemble')
    CompetingRiskForest.fit=forbidden
    tr=arrays(root/f'folds/{a.year}/train_PRIVATE.npz');ev=arrays(root/f'folds/{a.year}/eval_PRIVATE.npz')
    assert set(ev)=={'X','sample_ids'}
    members=[];saved=[];hashes=[]
    for seed in ARMS[a.arm]:
        p=root/f'members/{a.year}/{seed}';m=CompetingRiskForest.load(p/'forest_PRIVATE.json')
        assert len(m.trees_)==32 and m.n_features_==74 and m.config.path_budget==2
        for tree in m.trees_:
            for node in tree:
                if 'cif'in node:assert node['leaf_shrinkage']['strength']==10
        members.append(m);saved.append(arrays(p/'predictions_PRIVATE.npz'));hashes.append(sha(p/'forest_PRIVATE.json'))
    f=merge(members);assert len(f.trees_)==32*len(ARMS[a.arm]);f.audit_=f.audit()
    check=verify_export(f,tr,ev,out/'representation')
    expected=average_predictions(saved);actual=arrays(out/'representation/predictions_PRIVATE.npz')
    np.testing.assert_array_equal(actual['sample_ids'],saved[0]['sample_ids'])
    error=float(abs(actual['cif']-expected).max());assert error<=1e-12
    X=np.vstack([tr['X'],ev['X']]);times=np.unique(np.r_[0.,f.grid_,TIMES]);fullerror=0.
    for begin in range(0,len(X),32):
        xx=X[begin:begin+32];fullerror=max(fullerror,float(abs(f.predict(xx,times)-np.mean([m.predict(xx,times)for m in members],axis=0)).max()))
    assert fullerror<=1e-12
    summary=dict(status='COMPLETE',year=a.year,arm=a.arm,seeds=ARMS[a.arm],trees=len(f.trees_),forest_fits=0,calibration_fits=0,new_trees=0,
        saved_mean_error=error,full_mean_error=fullerror,checks=check,source_forest_hashes=hashes,no_evaluation_labels=True,
        host=socket.gethostname(),slurm_job=os.environ.get('SLURM_JOB_ID'),seconds=time.perf_counter()-start,
        peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    write(out/'summary.json',summary);write(out/'hashes.json',{str(p.relative_to(out)):sha(p)for p in out.rglob('*')if p.is_file()})
    print(json.dumps(dict(status='COMPLETE',year=a.year,arm=a.arm,forest_fits=0,seconds=summary['seconds'])),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--year',type=int,choices=YEARS,required=True);p.add_argument('--arm',choices=list(ARMS)[1:],required=True);a=p.parse_args()
    try:main(a)
    except Exception:
        out=Path(os.environ['OUTPUT_DIR']);out.mkdir(parents=True,exist_ok=True);write(out/'FAILED.json',dict(args=vars(a),traceback=traceback.format_exc()));raise
