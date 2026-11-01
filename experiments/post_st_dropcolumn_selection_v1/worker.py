"""One isolated fit and complete raw/shrunk function verification per task."""
from pathlib import Path
import argparse,os,resource,socket,time,traceback
import numpy as np
from crforest import CompetingRiskForest
from crforest.shrinkage import shrink_leaf_cifs
from experiments.post_st_level_coverage_v1.worker import verify_export
from .common import *

def execute(a):
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3));start=time.perf_counter()
    root=Path(os.environ['DATA_DIR']);out=Path(os.environ['OUTPUT_DIR']);out.mkdir(parents=True,exist_ok=True)
    base=root/('outer' if a.stage in ['outer','bootstrap'] else 'folds')/str(a.year)
    original=arrays(base/'train_PRIVATE.npz');ev=arrays(base/'eval_PRIVATE.npz');assert set(ev)=={'X','sample_ids'}
    spec=load(root/'subsets.json')[a.subset];keep=spec['keep'];tr=reduced(original,keep);reference=reduced(original,keep);ev=reduced(ev,keep)
    resample_info={}
    if a.stage=='bootstrap':
        assert len(keep)==83 and not load(root/'support_gates.json')[str(a.year)]['passed']
        draws=[]
        for y in np.unique(original['year']):
            ids=np.flatnonzero(original['year']==y)
            rng=np.random.default_rng(np.random.SeedSequence([ANALYSIS_SEED+50,a.year,a.repeat,int(y)]))
            draws.extend(rng.choice(ids,len(ids),replace=True).tolist())
        ix=np.asarray(draws,int);tr={k:v[ix] for k,v in tr.items()}
        np.savez_compressed(out/'fit_rows_PRIVATE.npz',indices=ix)
        resample_info=dict(repeat=a.repeat,draw_hash=array_hash(ix),unique_training_companies=len(np.unique(tr['groups'])),
            unique_target_companies=len(np.unique(tr['groups'][tr['event']==1])),selection_status='FALLBACK_SUPPORT_GATE_CLOSED',selection_frequency_interpretable=False)
    cfg=configuration(a.seed,len(keep),spec['mode']);pp=pairing(tr['groups'],cfg,len(keep))
    t=time.perf_counter();raw=CompetingRiskForest(cfg).fit(tr['X'],tr['time'],tr['event'],tr['groups'],feature_sources=[(j,) for j in keep],n_jobs=1);fit_seconds=time.perf_counter()-t
    for b,p in zip(raw.inbag_,pp):assert array_hash(b)==p['multiplicity_sha256']
    t=time.perf_counter();shrunk=shrink_leaf_cifs(raw,tr['X'],tr['time'],tr['event'],tr['groups'],10);shrink_seconds=time.perf_counter()-t
    checks={k:verify_export(m,reference,ev,out/k) for k,m in [('raw',raw),('shrunk',shrunk)]}
    replay_error=None
    if a.stage in ['inner','pilot'] and len(keep)==83 and spec['mode']=='count':
        prior=CompetingRiskForest.load(base/f'reference/{a.seed}/forest_PRIVATE.json')
        replay_error=float(abs(prior.predict(ev['X'],TIMES)-shrunk.predict(ev['X'],TIMES)).max())
        assert replay_error<=1e-13
    np.savez_compressed(out/'train_rows_PRIVATE.npz',sample_ids=tr['sample_ids'],groups=tr['groups'])
    write(out/'pairing.json',pp);write(out/'subset.json',spec)
    summary=dict(status='COMPLETE',stage=a.stage,year=a.year,seed=a.seed,subset_id=a.subset,mode=spec['mode'],features=len(keep),
        feature_set_sha256=spec['subset_sha256'],tree_feature_count=cfg.tree_feature_count,new_forest_fits=1,derived_models=1,
        training_companies=len(np.unique(tr['groups'])),training_rows=len(tr['X']),training_termination=int((tr['event']==1).sum()),training_recovery=int((tr['event']==2).sum()),
        fit_seconds=fit_seconds,shrink_seconds=shrink_seconds,checks=checks,reference_reproduction_error=replay_error,
        seconds=time.perf_counter()-start,peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,host=socket.gethostname(),
        source_train_sha256=sha(base/'train_PRIVATE.npz'),no_evaluation_labels_read=True,**resample_info)
    write(out/'summary.json',summary);write(out/'hashes.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()});print(json.dumps({k:v for k,v in summary.items() if k!='checks'}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['pilot','inner','outer','bootstrap'],required=True);p.add_argument('--year',type=int,required=True);p.add_argument('--subset',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--repeat',type=int,default=-1);a=p.parse_args()
    try:execute(a)
    except Exception:
        out=Path(os.environ['OUTPUT_DIR']);out.mkdir(parents=True,exist_ok=True);write(out/'FAILED.json',dict(stage=a.stage,year=a.year,seed=a.seed,subset=a.subset,traceback=traceback.format_exc()));raise
