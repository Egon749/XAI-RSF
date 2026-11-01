"""Each task re-identifies groups, searches/refits, then fits its final forest."""
import argparse, os, resource, socket, time, traceback, shutil
from pathlib import Path
import numpy as np
import pandas as pd
from .selection import *
from experiments.post_st_dropcolumn_selection_v1.common import arrays, load, write, sha
from experiments.post_st_dropcolumn_selection_h24_v1.fitting import fit_bundle, BootstrapStore, historical_support, read_bundle
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
    recovered=[]
    if item.get('resume_source'):
        previous=Path(item['resume_source'])
        saved=arrays(previous/'resample_PRIVATE.npz')
        np.testing.assert_array_equal(saved['indices'],ix)
        np.testing.assert_array_equal(saved['groups'],data['groups'][ix])
        # Failed diagnostics are retained separately; complete fit bundles only
        # may be reused after every original output hash passes.
        archive=out/'prior_failed_attempt';archive.mkdir()
        for name in ['FAILED.json','decision.json','candidate_scores.json','interaction_checks.json']:
            if (previous/name).exists():shutil.copy2(previous/name,archive/name)
        for bundle in sorted((previous/'discovery').iterdir()):
            if not (bundle/'hashes.json').exists():raise ValueError('Incomplete fit requires explicit recovery audit')
            for rel,h in load(bundle/'hashes.json').items():assert sha(bundle/rel)==h
            shutil.copytree(bundle,out/'discovery'/bundle.name);recovered.append(bundle.name)
    groups,candidates=grouped_candidates(data,meta,year,draw)
    support,counts=historical_support(parent,year,draw)
    gate=gate24(support)
    # Independent date-based audit must agree with the serialized endpoint evaluator.
    audit=discovery_gate(support_at(data,meta,year,draw))
    assert gate['passed']==audit['passed'] and gate['cohorts']==audit['cohorts'] and gate['target_events']==audit['targets'],(gate,audit)
    store=BootstrapStore(parent,out/'discovery',draw)
    for name in recovered:
        folder=out/'discovery'/name;old=load(folder/'summary.json')
        key=(tuple(old['keep']),old['year'])
        store.cache[key]=read_bundle(folder);store.bundles.append(str(folder.relative_to(out)))
    evaluator=Evaluator(parent,year,store,support,counts,repeat)
    checker=InteractionCheck(parent,store,evaluator.cohorts[24],groups)
    keep,status,scans,checks=select(gate,candidates,evaluator,checker)
    write(out/'decision.json',dict(keep=list(keep),status=status,gate=gate,outer_labels_used=False))
    write(out/'candidate_scores.json',json_safe(scans));write(out/'interaction_checks.json',json_safe(checks))
    fit_bundle(parent,keep,year,out/'final',draw,outer_final=True)
    summary=dict(status='COMPLETE',selection_status=status,gate=gate,year=year,repeat=repeat,
        independent_confirmation=time_gate(data,meta,year,draw)['passed'],
        groups=groups,candidates=[list(x) for x in candidates],keep=list(keep),
        supervised_search_executed=bool(scans),scored_candidates=len(scans),
        discovery_bundles=store.bundles,discovery_cohorts=evaluator.cohorts,
        new_forests=3*(len(store.bundles)+1),recovered_forests=3*len(recovered),fits_this_attempt=3*(len(store.bundles)+1-len(recovered)),host=socket.gethostname(),
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
