"""Fixed integer-strength derivations of saved raw forests; no fitting."""
import argparse,os,resource,socket,time,traceback
from crforest import CompetingRiskForest,ForestConfig
from crforest.shrinkage import shrink_leaf_cifs
from experiments.post_st_level_coverage_v1.worker import verify_export
from experiments.post_st_ablation_v1.verification import merge
from experiments.post_st_ablation_v1.common import pairing,array_hash
from .common import *
from experiments.post_st_shrinkage_strength_v1.verification import check_shrinkage

def derive(year,strength,inp,out):
    assert year in ALL_YEARS and strength in STRENGTHS
    tr=arrays(inp/f'folds/{year}/train_PRIVATE.npz');ev=arrays(inp/f'folds/{year}/eval_PRIVATE.npz')
    assert set(ev)=={'X','sample_ids'} and tr['X'].shape[1]==74
    members=[];predictions=[];records=[]
    for seed in SEEDS:
        raw=CompetingRiskForest.load(inp/f'raw/{year}/{seed}/forest_PRIVATE.json')
        clock=time.perf_counter();sh=shrink_leaf_cifs(raw,tr['X'],tr['time'],tr['event'],tr['groups'],strength)
        seconds=time.perf_counter()-clock
        replay=check_shrinkage(raw,sh,tr,strength)
        check=verify_export(sh,tr,ev,out/str(seed));assert check['max_path_variables']<=2
        reproduction={}
        if year in YEARS and strength==10:
            old=CompetingRiskForest.load(inp/f'reference/{year}/{seed}/forest_PRIVATE.json')
            assert old.trees_==sh.trees_
            saved=arrays(inp/f'reference/{year}/{seed}/predictions_PRIVATE.npz')
            np.testing.assert_array_equal(saved['sample_ids'],ev['sample_ids'])
            error=float(abs(saved['cif']-sh.predict(ev['X'],TIMES)).max());assert error==0
            reproduction=dict(trees_identical=True,prediction_error=error)
        records.append(dict(seed=seed,shrink_seconds=seconds,checks=check,**replay,reproduction=reproduction))
        members.append(sh);predictions.append(sh.predict(ev['X'],TIMES))
    f=merge(members);assert len(f.trees_)==96
    check=verify_export(f,tr,ev,out/'ensemble');mean=np.mean(predictions,axis=0)
    error=float(abs(mean-f.predict(ev['X'],TIMES)).max());assert error<1e-12
    np.savez_compressed(out/'mean_predictions_PRIVATE.npz',sample_ids=ev['sample_ids'],times=TIMES,cif=mean)
    return dict(mode='derive',year=year,strength=strength,forest_fits=0,derived_members=3,
        members=records,ensemble_check=check,ensemble_mean_error=error,
        source_train_sha256=sha(inp/f'folds/{year}/train_PRIVATE.npz'))

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['derive']);p.add_argument('year',type=int)
    p.add_argument('value',type=int);a=p.parse_args()
    resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3))
    inp=Path(os.environ['DATA_DIR']);out=Path(os.environ['OUTPUT_DIR']);out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    try:
        s=derive(a.year,a.value,inp,out)
        s.update(status='COMPLETE',host=socket.gethostname(),slurm_job=os.environ.get('SLURM_JOB_ID'),
            seconds=time.perf_counter()-start,peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            no_evaluation_labels_read=True)
        write(out/'summary.json',s);write(out/'hashes.json',{str(q.relative_to(out)):sha(q) for q in out.rglob('*') if q.is_file()})
        print(json.dumps({k:v for k,v in s.items() if k not in ['members','ensemble_check']}),flush=True)
    except Exception:
        write(out/'FAILED.json',dict(args=vars(a),traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
