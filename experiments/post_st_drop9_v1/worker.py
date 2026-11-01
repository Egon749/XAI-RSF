"""Exactly one raw forest fit, then its direct-parentAJ10 derivative."""
from pathlib import Path
import argparse,json,os,resource,socket,time,traceback
import numpy as np
import pandas as pd
from crforest import CompetingRiskForest
from crforest.shrinkage import shrink_leaf_cifs
from experiments.post_st_ablation_v1.verification import replay
from experiments.post_st_level_coverage_v1.worker import verify_export
from .common import *


def run(a):
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3));started=time.perf_counter()
    root=Path(os.environ['DATA_DIR']);out=Path(os.environ['OUTPUT_DIR']);out.mkdir(parents=True,exist_ok=True)
    tr=arrays(root/'train_PRIVATE.npz');ev=arrays(root/'eval_PRIVATE.npz');assert set(ev)=={'X','sample_ids'}
    original=arrays(root/'reference_train_PRIVATE.npz');protocol=json.loads((root/'protocol.json').read_text());keep=protocol['kept_original_indices']
    assert len(keep)==74 and a.seed in SEEDS
    for k in original:np.testing.assert_array_equal(tr[k],original[k][:,keep] if k=='X' else original[k])
    cfg=configuration(a.seed,'shallow_parent');paired=pairing(tr['groups'],cfg,p=74);reference=pairing(original['groups'],cfg,p=83)
    for x,y in zip(paired,reference):
        assert x['bootstrap_sha256']==y['bootstrap_sha256'] and x['multiplicity_sha256']==y['multiplicity_sha256']
    start=time.perf_counter()
    f=CompetingRiskForest(cfg).fit(tr['X'],tr['time'],tr['event'],tr['groups'],feature_sources=[(j,) for j in keep],n_jobs=1)
    fit_seconds=time.perf_counter()-start
    for i,b in enumerate(f.inbag_):assert array_hash(b)==paired[i]['multiplicity_sha256']
    start=time.perf_counter();sh=shrink_leaf_cifs(f,tr['X'],tr['time'],tr['event'],tr['groups'],10);shrink_seconds=time.perf_counter()-start
    checks={name:verify_export(model,tr,ev,out/name) for name,model in [('raw',f),('shrunk',sh)]}
    nodes,features,pairs,leaves,replay_checks=replay(sh,tr)
    for name,data in [('node_search',nodes),('feature_coverage',features),('pair_coverage',pairs),('leaf_support',leaves)]:
        pd.DataFrame(data).to_csv(out/f'{name}_PRIVATE.csv',index=False)
    write(out/'pairing.json',paired)
    summary=dict(status='COMPLETE',seed=a.seed,year=2024,arm='drop9',new_forest_fits=1,derived_shrunk_models=1,
        input_columns=74,training_companies=len(tr['X']),training_terminations=int((tr['event']==1).sum()),
        fit_seconds=fit_seconds,shrink_seconds=shrink_seconds,checks=checks,replay_checks=replay_checks,
        actual_scored_candidates=int(sum(r['scored_candidates'] for r in nodes)),no_test_labels_read=True,
        host=socket.gethostname(),seconds=time.perf_counter()-started,peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        source_train_sha256=sha(root/'train_PRIVATE.npz'),affinity_count=len(os.sched_getaffinity(0)))
    write(out/'summary.json',summary);write(out/'hashes.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})
    print(json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    try:run(a)
    except Exception:
        out=Path(os.environ['OUTPUT_DIR']);out.mkdir(parents=True,exist_ok=True);write(out/'FAILED.json',dict(seed=a.seed,traceback=traceback.format_exc()));raise
