"""Each task re-identifies groups, searches/refits, then fits its final forest."""
import argparse, os, resource, socket, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd
from .selection import *
from experiments.post_st_dropcolumn_selection_v1.common import arrays, load, write, sha
from experiments.post_st_dropcolumn_selection_h24_v1.fitting import fit_bundle, BootstrapStore, historical_support
from experiments.post_st_dropcolumn_selection_h24_v1.selection import Evaluator, gate24


def execute(index):
    resource.setrlimit(resource.RLIMIT_AS, (4*1024**3, 4*1024**3)); start=time.perf_counter()
    root=Path(os.environ['DATA_DIR']); out=Path(os.environ['OUTPUT_DIR']); out.mkdir(parents=True,exist_ok=True)
    plan=load(root/'plan.json'); item=plan['items'][index]; year,repeat=item['year'],item['repeat']
    parent=Path(plan['parent']); source=root.parent
    data=arrays(parent/f'input/outer/{year}/train_PRIVATE.npz'); meta=pd.read_parquet(source/f'{year}_metadata_PRIVATE.parquet')
    np.testing.assert_array_equal(data['sample_ids'],meta.sample_id.to_numpy(str))
    assert meta.group.is_unique and (meta.origin<pd.Timestamp(f'{year}-01-01')).all() and (meta.stop<pd.Timestamp(f'{year}-01-01')).all()
    if repeat < 0:
        ix=np.arange(len(meta)); draw=None
    else:
        rng=np.random.default_rng(np.random.SeedSequence([2026110191,year,repeat]))
        ix=np.concatenate([rng.choice(np.flatnonzero(meta.origin.dt.year.to_numpy()==y),size=int((meta.origin.dt.year==y).sum()),replace=True) for y in sorted(meta.origin.dt.year.unique())])
        draw=data['groups'][ix]
    np.savez_compressed(out/'resample_PRIVATE.npz',indices=ix,groups=data['groups'][ix])
    groups,candidates=grouped_candidates(data,meta,year,draw)
    support,counts=historical_support(parent,year,draw)
    gate=gate24(support)
    # Independent date-based audit must agree with the serialized endpoint evaluator.
    audit=discovery_gate(support_at(data,meta,year,draw))
    assert gate['passed']==audit['passed'] and gate['cohorts']==audit['cohorts'] and gate['target_events']==audit['targets'],(gate,audit)
    store=BootstrapStore(parent,out/'discovery',draw)
    evaluator=Evaluator(parent,year,store,support,counts,repeat)
    checker=InteractionCheck(parent,store,evaluator.cohorts[24],groups)
    keep,status,scans,checks=select(gate,candidates,evaluator,checker)
    write(out/'decision.json',dict(keep=list(keep),status=status,gate=gate,outer_labels_used=False))
    write(out/'candidate_scores.json',scans);write(out/'interaction_checks.json',checks)
    fit_bundle(parent,keep,year,out/'final',draw,outer_final=True)
    summary=dict(status='COMPLETE',selection_status=status,gate=gate,year=year,repeat=repeat,
        independent_confirmation=time_gate(data,meta,year,draw)['passed'],
        groups=groups,candidates=[list(x) for x in candidates],keep=list(keep),
        supervised_search_executed=bool(scans),scored_candidates=len(scans),
        discovery_bundles=store.bundles,discovery_cohorts=evaluator.cohorts,
        new_forests=3*(len(store.bundles)+1),host=socket.gethostname(),
        seconds=time.perf_counter()-start,peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        code_commit=plan['code_commit'],affinity=sorted(os.sched_getaffinity(0)),no_outer_labels_read=True)
    write(out/'summary.json',summary);write(out/'hashes.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})
    print({k:summary[k] for k in ['year','repeat','selection_status','scored_candidates','new_forests','seconds','host']},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);a=p.parse_args()
    try: execute(a.index)
    except Exception:
        out=Path(os.environ['OUTPUT_DIR']);out.mkdir(parents=True,exist_ok=True)
        write(out/'FAILED.json',dict(index=a.index,traceback=traceback.format_exc()));raise
