from pathlib import Path
import argparse,os,resource,socket,time,traceback
import numpy as np
import pandas as pd
from crforest import CompetingRiskForest
from crforest.shrinkage import shrink_leaf_cifs
from experiments.post_st_level_coverage_v1.worker import verify_export
from .common import *
from experiments.post_st_cause_support_v1.verification import replay

def main(a):
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3));start=time.perf_counter()
    root=Path(os.environ['DATA_DIR']);out=Path(os.environ['OUTPUT_DIR']);out.mkdir(parents=True,exist_ok=True)
    tr=arrays(root/f'folds/{a.year}/{a.arm}/train_PRIVATE.npz');ev=arrays(root/f'folds/{a.year}/{a.arm}/eval_PRIVATE.npz')
    assert set(ev)=={'X','sample_ids'} and tr['X'].shape[1]==74
    cfg=configuration(a.seed,a.arm,a.trees);expected=pairing(tr['groups'],cfg,74)
    t=time.perf_counter();f=CompetingRiskForest(cfg).fit(tr['X'],tr['time'],tr['event'],tr['groups'],n_jobs=1)
    fit_seconds=time.perf_counter()-t
    for actual,row in zip(f.inbag_,expected):assert array_hash(actual)==row['multiplicity_sha256']
    t=time.perf_counter();sh=shrink_leaf_cifs(f,tr['X'],tr['time'],tr['event'],tr['groups'],10);shrink_seconds=time.perf_counter()-t
    reference_equal=None;reference_error=None
    checks={label:verify_export(model,tr,ev,out/label) for label,model in [('raw',f),('shrunk',sh)]}
    t=time.perf_counter();nd,ft,pa,le,independent=replay(sh,tr);replay_seconds=time.perf_counter()-t
    for name,rows in [('node_search',nd),('feature_coverage',ft),('pair_coverage',pa),('leaf_support',le)]:
        pd.DataFrame(rows).to_csv(out/f'{name}_PRIVATE.csv',index=False)
    write(out/'pairing.json',expected)
    summary=dict(status='COMPLETE',year=a.year,seed=a.seed,arm=a.arm,trees=a.trees,new_forests=1,
        derived_models=1,training_cap_months=ARM_CAPS[a.arm],training_companies=len(tr['groups']),training_termination=int((tr['event']==1).sum()),
        fit_seconds=fit_seconds,shrink_seconds=shrink_seconds,replay_seconds=replay_seconds,checks=checks,
        independent=independent,reference_trees_identical=reference_equal,reference_prediction_error=reference_error,
        actual_scored_candidates=sum(r['scored_candidates'] for r in nd),
        guard_rejected_candidates=sum(r['guard_rejected'] for r in nd),
        host=socket.gethostname(),slurm_job=os.environ.get('SLURM_JOB_ID'),seconds=time.perf_counter()-start,
        peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        no_test_labels_read=True,source_train_sha256=sha(root/f'folds/{a.year}/{a.arm}/train_PRIVATE.npz'),
        affinity_count=len(os.sched_getaffinity(0)))
    write(out/'summary.json',summary);write(out/'hashes.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})
    print(json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--year',type=int,choices=YEARS,required=True);p.add_argument('--seed',type=int,choices=SEEDS,required=True)
    p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--trees',type=int,default=32);a=p.parse_args()
    try:main(a)
    except Exception:
        out=Path(os.environ['OUTPUT_DIR']);out.mkdir(exist_ok=True,parents=True)
        write(out/'FAILED.json',dict(args=vars(a),traceback=traceback.format_exc()));raise
