"""One independent year/window task; three fits and one exact ensemble."""
import argparse, os, resource, socket, time, traceback
from dataclasses import asdict
from crforest import CompetingRiskForest
from crforest.shrinkage import shrink_leaf_cifs
from experiments.post_st_level_coverage_v1.worker import verify_export
from experiments.post_st_cause_support_v1.verification import replay
from experiments.post_st_ablation_v1.verification import merge
from experiments.post_st_ablation_v1.common import pairing, array_hash
from .common import *

def execute(year, start, inp, out):
    out.mkdir(parents=True,exist_ok=True)
    tr=arrays(inp/f'folds/{year}/{start}/train_PRIVATE.npz')
    ev=arrays(inp/f'folds/{year}/eval_PRIVATE.npz')
    assert set(ev)=={'X','sample_ids'} and tr['X'].shape[1]==74
    members=[];predictions=[];records=[]
    for seed in SEEDS:
        folder=out/str(seed);folder.mkdir()
        cfg=ForestConfig(**load(inp/'configs.json')[str(seed)])
        expected=pairing(tr['groups'],cfg,74)
        clock=time.perf_counter()
        f=CompetingRiskForest(cfg).fit(tr['X'],tr['time'],tr['event'],tr['groups'],n_jobs=1)
        fit_seconds=time.perf_counter()-clock
        for row,actual in zip(expected,f.inbag_):assert array_hash(actual)==row['multiplicity_sha256']
        write(folder/'pairing_PRIVATE.json',expected)
        clock=time.perf_counter()
        sh=shrink_leaf_cifs(f,tr['X'],tr['time'],tr['event'],tr['groups'],10)
        shrink_seconds=time.perf_counter()-clock
        checks={v:verify_export(m,tr,ev,folder/v) for v,m in [('raw',f),('shrunk',sh)]}
        for q in checks.values():assert q['max_path_variables']<=2
        reproduction={}
        if start==2012:
            old=CompetingRiskForest.load(inp/f'reference/{year}/{seed}/forest_PRIVATE.json')
            assert old.trees_==sh.trees_, 'Full-history forest differs from source'
            times=np.unique(np.r_[0.,old.grid_,sh.grid_,TIMES]);X=np.vstack([tr['X'],ev['X']]);err=0.
            for begin in range(0,len(X),32):
                xx=X[begin:begin+32];err=max(err,float(abs(old.predict(xx,times)-sh.predict(xx,times)).max()))
            assert err==0
            oldp=arrays(inp/f'reference/{year}/{seed}/predictions_PRIVATE.npz')
            np.testing.assert_array_equal(oldp['sample_ids'],ev['sample_ids'])
            pe=float(abs(oldp['cif']-sh.predict(ev['X'],TIMES)).max());assert pe==0
            reproduction=dict(trees_identical=True,all_row_time_error=err,saved_prediction_error=pe)
        clock=time.perf_counter();nodes,features,pairs,leaves,rechecks=replay(sh,tr)
        replay_seconds=time.perf_counter()-clock
        for name,values in [('node_search',nodes),('feature_coverage',features),('pair_coverage',pairs),('leaf_support',leaves)]:
            pd.DataFrame(values).to_csv(folder/f'{name}_PRIVATE.csv',index=False)
        lp=pd.DataFrame(leaves)
        row=dict(seed=seed,forest_fits=1,fit_seconds=fit_seconds,shrink_seconds=shrink_seconds,
            replay_seconds=replay_seconds,checks=checks,independent_checks=rechecks,reproduction=reproduction,
            scored_candidates=sum(x['scored_candidates'] for x in nodes),
            guard_rejected_candidates=sum(x['guard_rejected'] for x in nodes),
            leaf_count=len(leaves),zero_target_leaves=int((lp.target_companies==0).sum()),
            borrowed_target_leaves=int(lp.borrowed_positive_target.sum()),
            median_unique_leaf_companies=float(lp.unique_companies.median()),
            minimum_unique_leaf_companies=int(lp.unique_companies.min()),
            raw_model_sha256=sha(folder/'raw/forest_PRIVATE.json'),
            shrunk_model_sha256=sha(folder/'shrunk/forest_PRIVATE.json'))
        records.append(row);members.append(sh);predictions.append(sh.predict(ev['X'],TIMES))
        print(json.dumps(dict(year=year,start=start,seed=seed,fit_seconds=fit_seconds,status='VERIFIED')),flush=True)
    merged=merge(members);assert len(merged.trees_)==96
    ck=verify_export(merged,tr,ev,out/'ensemble')
    mean=np.mean(predictions,axis=0);err=float(abs(merged.predict(ev['X'],TIMES)-mean).max())
    X=np.vstack([tr['X'],ev['X']]);times=np.unique(np.r_[0.,merged.grid_,TIMES])
    for begin in range(0,len(X),32):
        xx=X[begin:begin+32];err=max(err,float(abs(merged.predict(xx,times)-np.mean([f.predict(xx,times) for f in members],axis=0)).max()))
    assert err<=1e-12
    # Metrics use literal arithmetic mean of the three company forecasts.
    np.savez_compressed(out/'mean_predictions_PRIVATE.npz',sample_ids=ev['sample_ids'],times=TIMES,cif=mean)
    return dict(status='COMPLETE',year=year,earliest_entry_year=start,forest_fits=3,
        training_companies=len(tr['X']),members=records,ensemble_check=ck,ensemble_mean_error=err,
        ensemble_forest_sha256=sha(out/'ensemble/forest_PRIVATE.json'),
        source_train_sha256=sha(inp/f'folds/{year}/{start}/train_PRIVATE.npz'),no_evaluation_labels_read=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('year',type=int,choices=YEARS);p.add_argument('start',type=int,choices=STARTS);a=p.parse_args()
    resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3))
    inp=Path(os.environ['DATA_DIR']);out=Path(os.environ['OUTPUT_DIR']);clock=time.perf_counter()
    try:
        summary=execute(a.year,a.start,inp,out)
        summary.update(host=socket.gethostname(),slurm_job=os.environ.get('SLURM_JOB_ID'),
            total_seconds=time.perf_counter()-clock,peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            affinity_count=len(os.sched_getaffinity(0)))
        write(out/'summary.json',summary)
        write(out/'hashes.json',{str(q.relative_to(out)):sha(q) for q in out.rglob('*') if q.is_file()})
    except Exception:
        out.mkdir(parents=True,exist_ok=True);write(out/'FAILED.json',dict(args=vars(a),traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
