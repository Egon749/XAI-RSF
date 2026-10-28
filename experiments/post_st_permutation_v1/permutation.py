"""Fixed-design root maximum-score permutation audit; never fits a forest."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import resource
import socket
import time
import numpy as np

WEIGHTS=('combined','termination_only','recovery_only')
SCHEMES=('global_pair','within_entry_year_pair')
SEED=20261027
B=500


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path,data):
    with Path(path).open('x') as f:json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)


def permutation_indices(year,scheme,replicate):
    rng=np.random.default_rng(np.random.SeedSequence([SEED,SCHEMES.index(scheme),replicate,917]))
    if scheme=='global_pair':return rng.permutation(len(year))
    p=np.arange(len(year))
    for y in np.unique(year):
        ix=np.flatnonzero(year==y);p[ix]=rng.permutation(ix)
    return p


class ScoreCache:
    """Cache fixed outcome margins and evaluate all candidate masks in batches.

    A permutation jointly moves time and event. Reordering the fixed split
    masks into original outcome order is algebraically the same operation.
    Event-support gates are recomputed for EVERY permutation.
    """
    def __init__(self,masks,features,t,e,p,min_events=3):
        self.masks=np.asarray(masks,bool);self.features=np.asarray(features,int)
        self.p=p;self.min_events=min_events;self.n=len(t)
        self.order=np.argsort(t,kind='stable');st=np.asarray(t)[self.order]
        self.event=np.asarray(e)[self.order]
        self.starts=np.r_[0,np.flatnonzero(np.diff(st))+1]
        count=np.diff(np.r_[self.starts,len(st)])
        self.risk=np.cumsum(count[::-1])[::-1].astype(float)
        self.deaths=np.array([np.add.reduceat((self.event==k).astype(float),self.starts) for k in (1,2)])
        other=self.deaths.sum(0)[None,:]-self.deaths
        self.inclusive=self.risk[None,:]+np.cumsum(other,axis=1)-other
        self.correction=np.divide(self.inclusive-self.deaths,self.inclusive-1,
            out=np.zeros_like(self.inclusive),where=self.inclusive>1)

    def scores(self,permutation):
        p=np.asarray(permutation);assert np.array_equal(np.sort(p),np.arange(self.n))
        left=self.masks[:,np.argsort(p)[self.order]].astype(float)
        lc=np.add.reduceat(left,self.starts,axis=1);lr=np.cumsum(lc[:,::-1],axis=1)[:,::-1]
        ld=np.stack([np.add.reduceat(left*(self.event==k)[None,:],self.starts,axis=1) for k in (1,2)])
        other=ld.sum(0)[None,:,:]-ld
        li=lr[None,:,:]+np.cumsum(other,axis=2)-other
        prop=np.divide(li,self.inclusive[:,None,:],out=np.zeros_like(li),where=self.inclusive[:,None,:]>0)
        u=(ld-self.deaths[:,None,:]*prop).sum(2)
        v=(self.deaths[:,None,:]*prop*(1-prop)*self.correction[:,None,:]*(self.risk[None,None,:]>=2)).sum(2)
        numerator=np.stack([abs(u.sum(0)),abs(u[0]),abs(u[1])],axis=1)
        variance=np.stack([v.sum(0),v[0],v[1]],axis=1)
        scores=np.divide(numerator,np.sqrt(variance),out=np.zeros_like(numerator),where=variance>1e-18)
        left_events=ld.sum(axis=(0,2));total=self.deaths.sum()
        valid=(left_events>=self.min_events)&(total-left_events>=self.min_events)
        scores[~valid]=0.
        return scores,valid

    def maxima(self,permutation):
        scores,valid=self.scores(permutation)
        # No admissible split has the same score as choosing not to split: 0.
        per_feature=np.zeros((self.p,len(WEIGHTS)))
        np.maximum.at(per_feature,self.features,scores)
        return per_feature,per_feature.max(0),int(valid.sum())


def prepare(base,out):
    import pandas as pd
    from crforest.core import _split_parent,logrank_cr_score
    from crforest.partition import shared_quantile_cuts
    start=time.perf_counter();out.mkdir(exist_ok=True)
    inp=out/'input';inp.mkdir(exist_ok=False);reports=out/'preflight';reports.mkdir()
    old=json.loads(Path('experiments/post_st_prognosis_v1/execution_receipt.json').read_text())
    path=base/'input/train_PRIVATE.npz';assert sha(path)==old['model_input_hashes'][path.name]
    with np.load(path,allow_pickle=False) as z:a={k:z[k] for k in z.files}
    X,t,e,g=a['X'],a['time'],a['event'],a['groups']
    assert len(X)==309 and len(np.unique(g))==309 and a['year'].max()<=2022
    assert (e==1).sum()==14 and (e==2).sum()==184 and a['label_eligible'].all()
    names=pd.read_csv(base/'reports/feature_manifest.csv').name.to_numpy(str)
    frozen=json.loads((base/'audit/frozen_protocol.json').read_text());cfg=frozen['forest_common']
    cuts=shared_quantile_cuts(X,cfg['n_thresholds']);masks=[];features=[]
    for j,x in enumerate(X.T):
        missing=np.isnan(x);v=np.unique(x[~missing])
        use=cuts[j][(cuts[j]>=v[0])&(cuts[j]<v[-1])] if len(v) else []
        candidates=[(float(c),route) for c in sorted(use) for route in ([False,True] if missing.any() else [False])]
        if missing.any() and len(v):candidates.append((-np.inf,True))
        for cut,route in candidates:
            left=(x<=cut)|(missing&route)
            if min(left.sum(),(~left).sum())<cfg['min_leaf']:continue
            if any(len(np.unique(g[s]))<cfg['min_groups'] for s in (left,~left)):continue
            masks.append(left);features.append(j)
    masks=np.array(masks);features=np.array(features)
    cache=ScoreCache(masks,features,t,e,len(names),cfg['min_events'])
    observed,global_scores,valid_count=cache.maxima(np.arange(len(t)))
    prior=pd.read_csv('experiments/post_st_rule_features_v1/root_weight_audit/results/full_cohort_variable_ranking.csv')
    for row in prior.itertuples():
        np.testing.assert_allclose(observed[row.feature,WEIGHTS.index(row.weight)],row.best_score,rtol=0,atol=1e-12)
    # Validate all candidate scores against the original core on observed data
    # and both permutation designs; do not use this validation seed as a draw.
    checked=0
    for perm in [np.arange(len(t))]+[permutation_indices(a['year'],s,10000) for s in SCHEMES]:
        scores,valid=cache.scores(perm);parent=_split_parent(t[perm],e[perm],2)
        for i in np.flatnonzero(valid):
            for k,w in enumerate([[1,1],[1,0],[0,1]]):
                np.testing.assert_allclose(scores[i,k],logrank_cr_score(masks[i],e[perm],parent,w),rtol=0,atol=1e-12)
                checked+=1
    np.savez_compressed(inp/'permutation_input_PRIVATE.npz',masks=masks,features=features,time=t,event=e,
                        year=a['year'],feature_names=names,observed_feature_max=observed)
    protocol=dict(experiment='post_st_permutation_v1',status='FROZEN_BEFORE_NULL_RESULTS',schemes=list(SCHEMES),
        permutations_per_scheme=B,base_seed=SEED,weight_order=list(WEIGHTS),weight_vectors=[[1,1],[1,0],[0,1]],
        min_leaf=cfg['min_leaf'],min_groups=cfg['min_groups'],min_events=cfg['min_events'],
        time_status_permuted_as_pair=True,features_and_cutpoints_fixed=True,recompute_event_admissibility=True,
        observed_spatial_candidates=len(masks),observed_event_admissible_candidates=valid_count,
        reference_score_checks=checked,training_companies=len(t),training_terminations=int((e==1).sum()),
        training_recoveries=int((e==2).sum()),new_research_fits=0,test_arrays_read=False,entry_2025_outcomes_read=False,
        p_value_formula='(1 + count(null statistic >= observed statistic - 1e-12)) / (B + 1)',
        p_value_tie_tolerance=1e-12,quantile_method='higher',
        primary_family='maximum over all variables/candidates within each frozen cause-weight choice',
        secondary_families=['per-variable maximum over its candidates','joint maximum over variables and three weight choices'],
        global_null='fixed X independent of the joint observed (follow-up time, event type), assuming exchangeability',
        year_null='same independence conditional on entry year, assuming within-year exchangeability',
        limitations=['Administrative censoring and calendar composition can violate unrestricted exchangeability',
                    'Year blocks do not prove conditional exchangeability or remove all within-year censoring differences',
                    'No rejection is not proof of noise dominance; rejection is not temporal prediction validation'],
        source_sha256={str(path):sha(path),str(base/'audit/frozen_protocol.json'):sha(base/'audit/frozen_protocol.json')},
        input_sha256=sha(inp/'permutation_input_PRIVATE.npz'),script_sha256=sha(Path(__file__)))
    write(inp/'protocol.json',protocol)
    for s in SCHEMES:
        samples=[permutation_indices(a['year'],s,k) for k in range(5)]
        tick=time.perf_counter()
        for perm in samples:cache.maxima(perm)
        # Timing only: calibration outputs are not inspected or pooled into B.
        write(reports/f'timing_{s}.json',dict(repeat_safe=True,draws=5,seconds=time.perf_counter()-tick))
    print(json.dumps(dict(candidates=len(masks),observed_valid=valid_count,score_checks=checked,
                         preparation_seconds=time.perf_counter()-start,protocol_sha256=sha(inp/'protocol.json'))))


def worker(scheme,start,stop):
    resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
    tick=time.perf_counter();inp=Path(os.environ['DATA_DIR']);out=Path(os.environ['OUTPUT_DIR']);out.mkdir(exist_ok=True)
    plan=json.loads((inp/'protocol.json').read_text());path=inp/'permutation_input_PRIVATE.npz'
    assert sha(path)==plan['input_sha256'] and 0<=start<stop<=B
    assert sha(Path(__file__))==plan['script_sha256']
    with np.load(path,allow_pickle=False) as z:a={k:z[k] for k in z.files}
    cache=ScoreCache(a['masks'],a['features'],a['time'],a['event'],len(a['feature_names']),plan['min_events'])
    maxima=[];global_max=[];valid=[]
    for i in range(start,stop):
        p=permutation_indices(a['year'],scheme,i)
        if scheme=='within_entry_year_pair':assert np.array_equal(a['year'][p],a['year'])
        f,m,n=cache.maxima(p);maxima.append(f);global_max.append(m);valid.append(n)
    file=out/'null_scores_PRIVATE.npz'
    with file.open('xb') as f:np.savez_compressed(f,replicate=np.arange(start,stop),feature_max=np.array(maxima),
                            global_max=np.array(global_max),valid_candidates=np.array(valid))
    receipt=dict(status='COMPLETE',scheme=scheme,start=start,stop=stop,draws=stop-start,host=socket.gethostname(),
        seconds=time.perf_counter()-tick,peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        affinity_count=len(os.sched_getaffinity(0)),new_research_fits=0,protocol_sha256=sha(inp/'protocol.json'),
        output_sha256=sha(file),script_sha256=sha(Path(__file__)))
    write(out/'receipt.json',receipt);print(json.dumps(receipt))


if __name__=='__main__':
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='stage',required=True)
    a=s.add_parser('prepare');a.add_argument('--base',type=Path,required=True);a.add_argument('--out',type=Path,required=True)
    a=s.add_parser('worker');a.add_argument('--scheme',choices=SCHEMES,required=True);a.add_argument('--start',type=int,required=True);a.add_argument('--stop',type=int,required=True)
    a=p.parse_args()
    if a.stage=='prepare':prepare(a.base,a.out)
    else:worker(a.scheme,a.start,a.stop)
