"""Fixed pre-ST warning fit; evaluation input contains no outcomes."""
from pathlib import Path
import argparse,json,os,resource,socket,time,traceback,sys
import numpy as np
import pandas as pd
from crforest import CompetingRiskForest,ForestConfig
from crforest.shrinkage import shrink_leaf_cifs
from experiments.post_st_level_coverage_v1.worker import verify_export
from experiments.post_st_ablation_v1.verification import merge
from experiments.post_st_ablation_v1.common import pairing,array_hash
from experiments.post_st_cause_support_v1.verification import replay
from experiments.post_st_shrinkage_strength_v1.verification import check_shrinkage
from experiments.post_st_prognosis_v1.common import arrays,sha,write

TIMES=np.array([365.,366.,730.,731.])

def fit(year,seed,trees,inp,out):
    protocol=json.loads((inp/'frozen_protocol.json').read_text())
    tr=arrays(inp/f'folds/{year}/train_PRIVATE.npz');ev=arrays(inp/f'folds/{year}/eval_PRIVATE.npz')
    assert set(ev)=={'X','sample_ids'} and tr['X'].shape[1]==75
    cfg=ForestConfig(**{**protocol['configuration'],'seed':seed,'n_trees':trees})
    expected=pairing(tr['groups'],cfg,75)
    started=time.perf_counter();raw=CompetingRiskForest(cfg).fit(tr['X'],tr['time'],tr['event'],tr['groups'],n_jobs=1)
    fit_seconds=time.perf_counter()-started
    for a,actual in zip(expected,raw.inbag_):assert array_hash(actual)==a['multiplicity_sha256']
    # Save immediately, so a later audit/publication failure never triggers an
    # unrecorded second fit. Full validated exports remain separate.
    raw.save(out/'raw_fitted_PRIVATE.json')
    print(json.dumps(dict(stage='fit_saved',year=year,seed=seed,fit_seconds=fit_seconds)),flush=True)
    started=time.perf_counter();sh=shrink_leaf_cifs(raw,tr['X'],tr['time'],tr['event'],tr['groups'],10)
    shrink_seconds=time.perf_counter()-started;shrinkcheck=check_shrinkage(raw,sh,tr,10)
    rawcheck=verify_export(raw,tr,ev,out/'raw');check=verify_export(sh,tr,ev,out/'shrunk')
    started=time.perf_counter();nodes,features,pairs,leaves,recheck=replay(sh,tr)
    for n,v in [('node_search',nodes),('feature_coverage',features),('pair_coverage',pairs),('leaf_support',leaves)]:
        pd.DataFrame(v).to_csv(out/f'{n}_PRIVATE.csv',index=False)
    write(out/'pairing_PRIVATE.json',expected)
    return dict(year=year,seed=seed,trees=trees,forest_fits=1,fit_seconds=fit_seconds,shrink_seconds=shrink_seconds,
        replay_seconds=time.perf_counter()-started,checks=check,raw_checks=rawcheck,independent_checks=recheck,
        shrinkage_check=shrinkcheck,train_sha256=sha(inp/f'folds/{year}/train_PRIVATE.npz'),
        scored_candidates=sum(n['scored_candidates'] for n in nodes),guard_rejected=sum(n['guard_rejected'] for n in nodes))

def ensemble(year,inp,out):
    tr=arrays(inp/f'folds/{year}/train_PRIVATE.npz');ev=arrays(inp/f'folds/{year}/eval_PRIVATE.npz')
    records=json.loads((inp/'members.json').read_text())[str(year)];checks={}
    for version in ['raw','shrunk']:
        members=[CompetingRiskForest.load(Path(r)/version/'forest_PRIVATE.json') for r in records]
        f=merge(members);assert len(f.trees_)==96
        mean=np.mean([m.predict(ev['X'],TIMES) for m in members],axis=0)
        check=verify_export(f,tr,ev,out/version)
        err=float(abs(f.predict(ev['X'],TIMES)-mean).max());assert err<1e-12
        check['ensemble_mean_error']=err;checks[version]=check
    return dict(year=year,seed='ensemble',trees=96,forest_fits=0,checks=checks,source_members=records)

def main():
    p=argparse.ArgumentParser();p.add_argument('--year',type=int,required=True);p.add_argument('--seed',type=int,default=17)
    p.add_argument('--trees',type=int,default=32);p.add_argument('--ensemble',action='store_true');a=p.parse_args()
    inp=Path(os.environ['DATA_DIR']);out=Path(os.environ['OUTPUT_DIR']);out.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    try:
        s=ensemble(a.year,inp,out) if a.ensemble else fit(a.year,a.seed,a.trees,inp,out)
        s.update(status='COMPLETE',host=socket.gethostname(),slurm_job=os.environ.get('SLURM_JOB_ID'),seconds=time.perf_counter()-start,
            peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,no_evaluation_labels_read=True,
            environment=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,executable=sys.executable))
        write(out/'summary.json',s);write(out/'hashes.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})
        print(json.dumps({k:v for k,v in s.items() if 'check' not in k and k!='source_members'}),flush=True)
    except Exception:
        write(out/'FAILED.json',dict(args=vars(a),traceback=traceback.format_exc(),raw_forest_already_saved=(out/'raw_fitted_PRIVATE.json').exists()));raise

if __name__=='__main__':main()
