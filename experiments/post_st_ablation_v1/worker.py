"""One independent fit or saved-model verification; no hyperparameter selection."""
from pathlib import Path
import argparse, json, os, resource, socket, time, traceback
import numpy as np
from crforest import CompetingRiskForest
from crforest.shrinkage import shrink_leaf_cifs
from .common import *

def emit(f,tr,ev,out,arm,meta):
    out.mkdir();start=time.perf_counter()
    X=np.vstack([tr['X'],ev['X']]);n=len(tr['X'])
    pred=f.predict(X,TIMES);native=f.predict(X)
    assert np.isfinite(native).all() and native.min()>=-1e-12 and native.sum(-1).max()<=1+1e-12
    assert len(f.grid_)<2 or np.diff(native,axis=1).min()>=-1e-12
    predict_seconds=time.perf_counter()-start
    f.save(out/'forest_PRIVATE.json');loaded=CompetingRiskForest.load(out/'forest_PRIVATE.json')
    np.testing.assert_array_equal(loaded.predict(X),native)
    np.savez_compressed(out/'predictions_PRIVATE.npz',sample_ids=ev['sample_ids'],times=TIMES,cif=pred[n:],training_cif=pred[:n])
    write(out/'summary.json',dict(**meta,arm=arm,status='COMPLETE',**f.audit_,prediction_seconds=predict_seconds,reload_error=0.))
    write(out/'hashes.json',{p.name:sha(p) for p in out.iterdir() if p.is_file()})

def fit(a,root,out):
    src=root/'folds'/str(a.year);plan=json.loads((src/'plan.json').read_text())
    for n,h in plan['source_hashes'].items():assert sha(src/n)==h
    tr=arrays(src/'train_PRIVATE.npz');ev=arrays(src/'eval_PRIVATE.npz');assert set(ev)=={'X','sample_ids'}
    cfg=configuration(a.seed,a.arm);expected=json.loads((root/f'pairing_{a.year}.json').read_text())[str(a.seed)]
    pp=pairing(tr['groups'],cfg)
    for v,w in zip(pp,expected):
        assert v['bootstrap_sha256']==w['bootstrap_sha256'] and v['multiplicity_sha256']==w['multiplicity_sha256']
        if a.arm!='iid_features':assert v['features_sha256']==w['features_sha256']
    start=time.perf_counter();f=CompetingRiskForest(cfg).fit(tr['X'],tr['time'],tr['event'],tr['groups']);fit_seconds=time.perf_counter()-start
    inbag=np.stack(f.inbag_)
    for i,b in enumerate(inbag):assert array_hash(b)==pp[i]['multiplicity_sha256']
    np.savez_compressed(out/'bootstrap_PRIVATE.npz',multiplicity=inbag,**{f'indices_{i}':cluster_bootstrap(tr['groups'],cfg.seed,i) for i in range(cfg.n_trees)})
    write(out/'pairing.json',pp)
    meta=dict(year=a.year,seed=a.seed,design_arm=a.arm,code_commit=plan['code_commit'],forest_fit_seconds=fit_seconds,host=socket.gethostname())
    emit(f,tr,ev,out/'raw',a.arm+'_raw',meta)
    start=time.perf_counter();sh=shrink_leaf_cifs(f,tr['X'],tr['time'],tr['event'],tr['groups'],10)
    shrink_seconds=time.perf_counter()-start
    assert sh.audit_==f.audit_
    emit(sh,tr,ev,out/'shrunk',a.arm,dict(meta,shrinkage_seconds=shrink_seconds))
    return dict(**meta,new_forest_fits=1,leaf_updates=1,shrinkage_seconds=shrink_seconds,artifacts=['raw','shrunk'])

def main(a):
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
    start=time.perf_counter();cpu=time.process_time();root=Path(os.environ['DATA_DIR']);out=Path(os.environ['OUTPUT_DIR']);out.mkdir(exist_ok=True,parents=True)
    if a.mode=='fit':row=fit(a,root,out)
    else:
        from .verification import inspect
        row=inspect(a,root,out)
    row.update(status='COMPLETE',host=socket.gethostname(),affinity_count=len(os.sched_getaffinity(0)),seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    write(out/'summary.json',row)
    write(out/'hashes.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})
    print(json.dumps(row),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['fit','inspect'],required=True);p.add_argument('--year',type=int,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--arm',required=True);p.add_argument('--run',type=Path)
    a=p.parse_args()
    try:main(a)
    except Exception:
        out=Path(os.environ['OUTPUT_DIR']);out.mkdir(exist_ok=True,parents=True)
        write(out/'FAILED.json',dict(args=vars(a),traceback=traceback.format_exc()));raise
