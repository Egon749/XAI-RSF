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
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3));start=time.perf_counter()
    root=Path(os.environ['DATA_DIR'])/f'folds/{a.year}';out=Path(os.environ['OUTPUT_DIR']);out.mkdir(parents=True,exist_ok=True)
    tr=arrays(root/'train_PRIVATE.npz');ev=arrays(root/'eval_PRIVATE.npz');assert set(ev)=={'X','sample_ids'}
    plan=json_load(root/'plan.json');cfg=configuration(a.seed);pairs=pairing(tr['groups'],cfg,20);t=time.perf_counter()
    f=CompetingRiskForest(cfg).fit(tr['X'],tr['time'],tr['event'],tr['groups'],feature_sources=[(j,) for j in plan['original_indices']],n_jobs=1)
    fit_seconds=time.perf_counter()-t
    for i,b in enumerate(f.inbag_):assert array_hash(b)==pairs[i]['multiplicity_sha256']
    t=time.perf_counter();sh=shrink_leaf_cifs(f,tr['X'],tr['time'],tr['event'],tr['groups'],10);shrink_seconds=time.perf_counter()-t
    checks={k:verify_export(m,tr,ev,out/k) for k,m in [('raw',f),('shrunk',sh)]}
    nodes,features,ps,leaves,rc=replay(sh,tr)
    for k,rows in [('node_search',nodes),('feature_coverage',features),('pair_coverage',ps),('leaf_support',leaves)]:pd.DataFrame(rows).to_csv(out/f'{k}_PRIVATE.csv',index=False)
    write(out/'pairing.json',pairs)
    row=dict(status='COMPLETE',year=a.year,seed=a.seed,new_forest_fits=1,derived_models=1,fit_seconds=fit_seconds,shrink_seconds=shrink_seconds,
        checks=checks,replay_checks=rc,actual_scored_candidates=int(sum(n['scored_candidates'] for n in nodes)),no_test_labels_read=True,
        training_companies=len(tr['X']),input_columns=20,host=socket.gethostname(),seconds=time.perf_counter()-start,
        peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,affinity_count=len(os.sched_getaffinity(0)))
    write(out/'summary.json',row);write(out/'hashes.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()});print(json.dumps(row))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--year',type=int,choices=YEARS,required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    try:run(a)
    except Exception:
        out=Path(os.environ['OUTPUT_DIR']);out.mkdir(parents=True,exist_ok=True);write(out/'FAILED.json',dict(year=a.year,seed=a.seed,traceback=traceback.format_exc()));raise
