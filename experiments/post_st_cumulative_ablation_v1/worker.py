from pathlib import Path
import argparse,json,os,resource,socket,time,traceback
from types import SimpleNamespace
import numpy as np
from crforest import CompetingRiskForest,cluster_bootstrap
from crforest.shrinkage import shrink_leaf_cifs
from experiments.post_st_ablation_v1.worker import emit
from experiments.post_st_ablation_v1.verification import inspect
from .common import *

def fit(a,root,out):
    assert a.arm in NEW
    src=root/'folds'/str(a.year);plan=json.loads((src/'plan.json').read_text())
    for name,h in plan['source_hashes'].items():assert sha(src/name)==h
    tr=arrays(src/'train_PRIVATE.npz');ev=arrays(src/'eval_PRIVATE.npz');assert set(ev)=={'X','sample_ids'}
    cfg=configuration(a.seed,a.arm);actual=pairing(tr['groups'],cfg)
    expected=json.loads((root/f'pairing_{a.year}.json').read_text())[str(a.seed)]
    for l,r in zip(actual,expected):
        assert l['bootstrap_sha256']==r['bootstrap_sha256'] and l['multiplicity_sha256']==r['multiplicity_sha256']
        if cfg.tree_feature_strategy=='balanced':assert l['features_sha256']==r['features_sha256']
    started=time.perf_counter();raw=CompetingRiskForest(cfg).fit(tr['X'],tr['time'],tr['event'],tr['groups']);fit_seconds=time.perf_counter()-started
    for i,bag in enumerate(raw.inbag_):assert array_hash(bag)==actual[i]['multiplicity_sha256']
    np.savez_compressed(out/'bootstrap_PRIVATE.npz',multiplicity=np.stack(raw.inbag_),**{f'indices_{i}':cluster_bootstrap(tr['groups'],a.seed,i) for i in range(32)})
    write(out/'pairing.json',actual)
    meta=dict(year=a.year,seed=a.seed,design_arm=a.arm,host=socket.gethostname(),code_commit=plan['code_commit'],forest_fit_seconds=fit_seconds)
    emit(raw,tr,ev,out/'raw','s4' if a.arm=='s3' else 's2_raw_auxiliary',meta)
    started=time.perf_counter();shrunk=shrink_leaf_cifs(raw,tr['X'],tr['time'],tr['event'],tr['groups'],10);shrink_seconds=time.perf_counter()-started
    assert raw.audit_==shrunk.audit_
    emit(shrunk,tr,ev,out/'shrunk',a.arm,dict(meta,shrinkage_seconds=shrink_seconds))
    return dict(**meta,new_forest_fits=1,new_shrunk_models=1,shrinkage_seconds=shrink_seconds)

def main(a):
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3));started=time.perf_counter();cpu=time.process_time()
    root=Path(os.environ['DATA_DIR']);out=Path(os.environ['OUTPUT_DIR']);out.mkdir(exist_ok=True,parents=True)
    if a.mode=='fit':row=fit(a,root,out)
    else:
        context=SimpleNamespace(**vars(a));context.run=a.run/'validation_context'
        row=inspect(context,root,out)
    row.update(status='COMPLETE',host=socket.gethostname(),affinity_count=len(os.sched_getaffinity(0)),seconds=time.perf_counter()-started,cpu_seconds=time.process_time()-cpu,peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    write(out/'summary.json',row);write(out/'hashes.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})
    print(json.dumps(row),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['fit','inspect'],required=True);p.add_argument('--year',type=int,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--arm',choices=STAGES,required=True);p.add_argument('--run',type=Path);a=p.parse_args()
    try:main(a)
    except Exception:
        out=Path(os.environ['OUTPUT_DIR']);out.mkdir(exist_ok=True,parents=True);write(out/'FAILED.json',dict(args=vars(a),traceback=traceback.format_exc()));raise
