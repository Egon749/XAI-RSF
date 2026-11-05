"""One independent year/count task: three raw fits, shrinkage and exact export."""
import argparse,os,resource,socket,time,traceback
from crforest import CompetingRiskForest,ForestConfig
from crforest.shrinkage import shrink_leaf_cifs
from experiments.post_st_level_coverage_v1.worker import verify_export
from experiments.post_st_ablation_v1.verification import merge
from experiments.post_st_ablation_v1.common import pairing,array_hash
from experiments.post_st_cause_support_v1.verification import replay
from experiments.post_st_shrinkage_strength_v1.verification import check_shrinkage
from .common import *

def execute(year,pool_count,inp,out):
    assert year in ALL_YEARS and pool_count in COUNTS
    tr=arrays(inp/f'folds/{year}/train_PRIVATE.npz');ev=arrays(inp/f'folds/{year}/eval_PRIVATE.npz')
    assert set(ev)=={'X','sample_ids'} and tr['X'].shape[1]==74
    (out/'raw').mkdir();members=[];predictions=[];records=[]
    for seed in SEEDS:
        cfg=ForestConfig(**load(inp/'configs.json')[str(pool_count)][str(seed)])
        expected=pairing(tr['groups'],cfg,74);base=pairing(tr['groups'],configuration(seed,REFERENCE),74)
        assert all(a['bootstrap_sha256']==b['bootstrap_sha256'] and a['multiplicity_sha256']==b['multiplicity_sha256'] for a,b in zip(expected,base))
        clock=time.perf_counter();raw=CompetingRiskForest(cfg).fit(tr['X'],tr['time'],tr['event'],tr['groups'],n_jobs=1)
        fit_seconds=time.perf_counter()-clock
        for a,actual in zip(expected,raw.inbag_):assert array_hash(actual)==a['multiplicity_sha256']
        clock=time.perf_counter();sh=shrink_leaf_cifs(raw,tr['X'],tr['time'],tr['event'],tr['groups'],10)
        shrink_seconds=time.perf_counter()-clock
        rawcheck=verify_export(raw,tr,ev,out/'raw'/str(seed));check=verify_export(sh,tr,ev,out/str(seed))
        assert check['max_path_variables']<=2 and rawcheck['max_path_variables']<=2
        write(out/str(seed)/'pairing_PRIVATE.json',expected)
        shrinkcheck=check_shrinkage(raw,sh,tr,10)
        clock=time.perf_counter();nodes,features,pairs,leaves,recheck=replay(sh,tr);replay_seconds=time.perf_counter()-clock
        for n,values in [('node_search',nodes),('feature_coverage',features),('pair_coverage',pairs),('leaf_support',leaves)]:
            pd.DataFrame(values).to_csv(out/str(seed)/f'{n}_PRIVATE.csv',index=False)
        reproduction={}
        if pool_count==REFERENCE:
            oldraw=CompetingRiskForest.load(inp/f'raw/{year}/{seed}/forest_PRIVATE.json');assert oldraw.trees_==raw.trees_
            assert sha(out/'raw'/str(seed)/'forest_PRIVATE.json')==sha(inp/f'raw/{year}/{seed}/forest_PRIVATE.json')
            reproduction=dict(raw_trees_identical=True,raw_file_identical=True)
            if year in YEARS:
                old=CompetingRiskForest.load(inp/f'reference/{year}/{seed}/forest_PRIVATE.json');assert old.trees_==sh.trees_
                z=arrays(inp/f'reference/{year}/{seed}/predictions_PRIVATE.npz');np.testing.assert_array_equal(z['sample_ids'],ev['sample_ids'])
                err=float(abs(z['cif']-sh.predict(ev['X'],TIMES)).max());assert err==0
                reproduction.update(shrunk_trees_identical=True,prediction_error=err)
        lp=pd.DataFrame(leaves)
        record=dict(seed=seed,forest_fits=1,fit_seconds=fit_seconds,shrink_seconds=shrink_seconds,replay_seconds=replay_seconds,
            checks=check,raw_checks=rawcheck,independent_checks=recheck,reproduction=reproduction,**shrinkcheck,
            bootstrap_paired=True,scored_candidates=sum(x['scored_candidates'] for x in nodes),
            guard_rejected_candidates=sum(x['guard_rejected'] for x in nodes),
            median_unique_leaf_companies=float(lp.unique_companies.median()),minimum_unique_leaf_companies=int(lp.unique_companies.min()))
        records.append(record);members.append(sh);predictions.append(sh.predict(ev['X'],TIMES))
        print(json.dumps(dict(year=year,pool_count=pool_count,seed=seed,fit_seconds=fit_seconds,status='VERIFIED')),flush=True)
    f=merge(members);assert len(f.trees_)==96
    check=verify_export(f,tr,ev,out/'ensemble');mean=np.mean(predictions,axis=0)
    error=float(abs(mean-f.predict(ev['X'],TIMES)).max());assert error<1e-12
    np.savez_compressed(out/'mean_predictions_PRIVATE.npz',sample_ids=ev['sample_ids'],times=TIMES,cif=mean)
    return dict(mode='fit',year=year,pool_count=pool_count,forest_fits=3,derived_members=3,members=records,
        ensemble_check=check,ensemble_mean_error=error,source_train_sha256=sha(inp/f'folds/{year}/train_PRIVATE.npz'))

def main():
    p=argparse.ArgumentParser();p.add_argument('year',type=int);p.add_argument('pool_count',type=int);a=p.parse_args()
    resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3))
    inp=Path(os.environ['DATA_DIR']);out=Path(os.environ['OUTPUT_DIR']);out.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    try:
        s=execute(a.year,a.pool_count,inp,out)
        s.update(status='COMPLETE',host=socket.gethostname(),slurm_job=os.environ.get('SLURM_JOB_ID'),
            seconds=time.perf_counter()-start,peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,no_evaluation_labels_read=True)
        write(out/'summary.json',s);write(out/'hashes.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})
        print(json.dumps({k:v for k,v in s.items() if k not in ['members','ensemble_check']}),flush=True)
    except Exception:
        write(out/'FAILED.json',dict(args=vars(a),traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
