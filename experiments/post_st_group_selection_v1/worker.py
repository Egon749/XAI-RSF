"""One complete chronological eligibility/selection/final-fit resample."""
import argparse,os,resource,socket,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
from .selection import *
from experiments.post_st_dropcolumn_selection_v1.common import arrays,load,write,sha
from experiments.post_st_dropcolumn_selection_h24_v1.fitting import fit_bundle

def execute(index):
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3));start=time.perf_counter()
    root=Path(os.environ['DATA_DIR']);out=Path(os.environ['OUTPUT_DIR']);out.mkdir(parents=True,exist_ok=True)
    plan=load(root/'plan.json');item=plan['items'][index];year,repeat=item['year'],item['repeat']
    parent=Path(plan['parent']);source=root.parent
    data=arrays(parent/f'input/outer/{year}/train_PRIVATE.npz');meta=pd.read_parquet(source/f'{year}_metadata_PRIVATE.parquet')
    np.testing.assert_array_equal(data['sample_ids'],meta.sample_id.to_numpy(str))
    rng=np.random.default_rng(np.random.SeedSequence([2026110191,year,repeat]))
    ix=np.concatenate([rng.choice(np.flatnonzero(meta.origin.dt.year.to_numpy()==y),size=int((meta.origin.dt.year==y).sum()),replace=True) for y in sorted(meta.origin.dt.year.unique())])
    draw=data['groups'][ix];np.savez_compressed(out/'resample_PRIVATE.npz',indices=ix,groups=draw)
    groups,candidates=grouped_candidates(data,meta,year,draw)
    gate=time_gate(data,meta,year,draw)
    # Frozen gate precedes supervised search. Do not invent confirmation outcomes.
    if gate['passed']:raise RuntimeError('Unexpected supported branch: stop rather than silently omit a supervised stage')
    keep,status=choose(gate,[])
    fit_bundle(parent,keep,year,out/'final',draw,outer_final=True)
    summary=dict(status='COMPLETE',selection_status=status,gate=gate,year=year,repeat=repeat,
        groups=groups,candidates=[list(x) for x in candidates],keep=list(keep),
        supervised_search_executed=False,new_forests=3,host=socket.gethostname(),
        seconds=time.perf_counter()-start,peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        code_commit=plan['code_commit'],affinity=sorted(os.sched_getaffinity(0)),no_outer_labels_read=True)
    write(out/'summary.json',summary);write(out/'hashes.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})
    print({k:v for k,v in summary.items() if k not in ['groups','candidates','keep','gate']},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);a=p.parse_args()
    try:execute(a.index)
    except Exception:
        out=Path(os.environ['OUTPUT_DIR']);out.mkdir(parents=True,exist_ok=True)
        write(out/'FAILED.json',dict(index=a.index,traceback=traceback.format_exc()));raise
