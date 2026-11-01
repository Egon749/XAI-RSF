"""One independent, complete pre-cutoff selection pipeline per task."""
import argparse,os,resource,socket,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
from .selection import *
from .store import Store,previous_output
from .fitting import fit_bundle
from experiments.post_st_dropcolumn_selection_v1.common import write,sha
from experiments.post_st_dropcolumn_selection_h24_v1.fitting import historical_support
from experiments.post_st_dropcolumn_selection_h24_v1.selection import gate24


class OldCheck(InteractionCheck):
    def path(self,keep,cohort):return self.store.path(keep,cohort)


def execute(index):
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3));started=time.perf_counter()
    root=Path(os.environ['DATA_DIR']);out=Path(os.environ['OUTPUT_DIR']);out.mkdir(parents=True,exist_ok=True)
    plan=load(root/'plan.json');item=plan['items'][index];year,repeat=item['year'],item['repeat'];parent=Path(plan['parent'])
    data=arrays(parent/f'input/outer/{year}/train_PRIVATE.npz');meta=pd.read_parquet(root.parent/f'{year}_metadata_PRIVATE.parquet')
    np.testing.assert_array_equal(data['sample_ids'],meta.sample_id.to_numpy(str))
    assert meta.group.is_unique and (meta.origin<pd.Timestamp(f'{year}-01-01')).all() and (meta.stop<pd.Timestamp(f'{year}-01-01')).all()
    if repeat<0:ix=np.arange(len(meta));draw=None
    else:
        rng=np.random.default_rng(np.random.SeedSequence([2026110191,year,repeat]))
        ix=np.concatenate([rng.choice(np.flatnonzero(meta.origin.dt.year.to_numpy()==y),size=int((meta.origin.dt.year==y).sum()),replace=True) for y in sorted(meta.origin.dt.year.unique())]);draw=data['groups'][ix]
    prior=previous_output(year,repeat);old_draw=arrays(prior/'resample_PRIVATE.npz');np.testing.assert_array_equal(old_draw['indices'],ix)
    np.savez_compressed(out/'resample_PRIVATE.npz',indices=ix,groups=data['groups'][ix])
    groups,candidates=grouped_candidates(data,meta,year,draw);support,counts=historical_support(parent,year,draw);gate=gate24(support)
    stores={};scans=[];checks=[];old_checks=[];decisions={};periods=[];cohorts={}
    for mode in (['anchored','uncontrolled'] if repeat<0 else ['anchored']):
        store=Store(parent,out/mode,draw,groups,year,repeat,mode);stores[mode]=store
        ev=PeriodEvaluator(parent,year,store,support,counts);cohorts=ev.cohorts
        checker=GroupCheck(parent,store,ev.cohorts[24],groups,draw);old=OldCheck(parent,store,ev.cohorts[24],groups)
        local=[]
        if gate['passed']:
            for keep in sorted(set(map(tuple,candidates))-{FULL}):
                row=ev.compare(keep);gc=checker(keep);oc=old(keep)
                row.update(mode=mode,new_group_pass=all(r['passed'] for r in gc),old_pair_pass=bool(oc) and all(r['passed'] for r in oc),qualified_groups=sum(r['qualified'] for r in gc),failed_group_checks=sum(not r['passed'] for r in gc))
                periods.extend(dict(mode=mode,keep=list(keep),**p) for p in row.pop('periods'))
                checks.extend(dict(mode=mode,keep=list(keep),**r) for r in gc);old_checks.extend(dict(mode=mode,keep=list(keep),**r) for r in oc)
                local.append(row)
            for guard in ['new_group_pass','old_pair_pass']:
                keep,status=choose(local,guard);decisions[mode+'_'+guard]=dict(keep=list(keep),status=status)
        else:
            for guard in ['new_group_pass','old_pair_pass']:decisions[mode+'_'+guard]=dict(keep=list(FULL),status='INSUFFICIENT_DISCOVERY_SUPPORT')
        scans.extend(local)
    main=decisions['anchored_new_group_pass'];keep=tuple(main['keep'])
    write(out/'decision.json',json_safe(dict(primary=main,policies=decisions,gate=gate,outer_labels_used=False)))
    for name,value in [('candidate_scores',scans),('group_checks',checks),('old_interaction_checks',old_checks),('period_scores',periods)]:write(out/(name+'.json'),json_safe(value))
    # The final model is actually refitted for every point/resampled decision.
    fit_bundle(parent,keep,year,out/'final',draw,outer_final=True,groups_spec=groups,baseline_path=prior/'final')
    summary=dict(status='COMPLETE',year=year,repeat=repeat,keep=list(keep),selection_status=main['status'],gate=gate,
        groups=groups,candidates=[list(k) for k in candidates],policies=decisions,discovery_cohorts=cohorts,
        supervised_search_executed=gate['passed'],scored_candidates=sum(r['mode']=='anchored' for r in scans),
        bundles=[r for s in stores.values() for r in s.records],new_forests=3+sum(s.new_forests for s in stores.values()),
        host=socket.gethostname(),seconds=time.perf_counter()-started,peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        code_commit=plan['code_commit'],no_outer_labels_read=True,affinity=sorted(os.sched_getaffinity(0)))
    write(out/'summary.json',json_safe(summary));write(out/'hashes.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})
    print({k:summary[k] for k in ['year','repeat','keep','new_forests','seconds','host']},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);a=p.parse_args()
    try:execute(a.index)
    except Exception:
        out=Path(os.environ['OUTPUT_DIR']);out.mkdir(parents=True,exist_ok=True);write(out/'FAILED.json',dict(index=a.index,traceback=traceback.format_exc()));raise
