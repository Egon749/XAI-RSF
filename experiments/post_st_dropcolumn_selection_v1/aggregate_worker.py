"""Compile and verify the complete three-seed predictor; never fit a forest."""
from pathlib import Path
import argparse,itertools,os,resource,socket,time,traceback
import numpy as np
import pandas as pd
from crforest import CompetingRiskForest,CIFExpansion
from experiments.post_st_ablation_v1.verification import merge
from experiments.post_st_level_coverage_v1.worker import verify_export
from .common import *

def execute(index):
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3));start=time.perf_counter()
    root=Path(os.environ['DATA_DIR']);out=Path(os.environ['OUTPUT_DIR']);out.mkdir(parents=True,exist_ok=True)
    plan=load(root/'plan.json');spec=plan['items'][index];run=Path(plan['run']);year=spec['year'];keep=spec['keep'];paths=[Path(p) for p in spec['paths']]
    base=run/'input'/('folds' if spec['kind']=='inner' else 'outer')/str(year)
    tr=reduced(arrays(base/'train_PRIVATE.npz'),keep);ev=reduced(arrays(base/'eval_PRIVATE.npz'),keep)
    members=[CompetingRiskForest.load(p/'forest_PRIVATE.json') for p in paths]
    bootstrap_gate=None
    if spec['kind']=='bootstrap':
        indices=arrays(paths[0].parent/'fit_rows_PRIVATE.npz')['indices']
        for p in paths[1:]:np.testing.assert_array_equal(indices,arrays(p.parent/'fit_rows_PRIVATE.npz')['indices'])
        sampled=set(tr['groups'][indices]);support=[]
        for v in range(2019,year):
            train=arrays(run/f'input/folds/{v}/train_PRIVATE.npz');lab=arrays(run/f'input/folds/{v}/labels_{year}_PRIVATE.npz')
            take=np.isin(lab['groups'],list(sampled));lab={k:a[take] for k,a in lab.items()}
            train_take=np.isin(train['groups'],list(sampled))
            for h in [12,24]:
                y,w,known,ok,_=evaluation_context(lab,h,1)
                support.append(dict(horizon=h,mature=v+h//12<year,supported=ok,train_termination=int(((train['event']==1)&train_take).sum()),termination=int((y&known).sum()),controls=int((~y&known).sum())))
        bootstrap_gate=support_gate(support)
        assert not bootstrap_gate['passed'],'Never call a support-gated fallback evidence of selected feature stability'
    for p,h in zip(paths,spec['forest_hashes']):assert sha(p/'forest_PRIVATE.json')==h
    f=merge(members);check=verify_export(f,tr,ev,out/'ensemble')
    actual=f.predict(ev['X'],TIMES);expected=np.mean([m.predict(ev['X'],TIMES) for m in members],axis=0)
    difference=float(abs(actual-expected).max());assert difference<=1e-13
    exs=[CIFExpansion.load(p/'functions_PRIVATE.npz') for p in paths]
    ex=CIFExpansion.load(out/'ensemble/functions_PRIVATE.npz')
    terms=arrays(out/'ensemble/components_PRIVATE.npz');keys=[tuple(map(int,k.split(','))) for k in terms['keys']]
    component_rows=[];seed_stability=[]
    for h in [12,24]:
        ix=np.searchsorted(TIMES,tr[f'h{h}'])
        values=terms['terms'][np.arange(len(tr['X']))[:,None],np.arange(len(keys))[None,:],ix[:,None],:]
        for k,key in enumerate(keys):
            for cause in [1,2]:
                a=values[:,k,cause-1]
                component_rows.append(dict(key=','.join(str(keep[j]) for j in key),order=len(key),horizon=h,cause=cause,mean_abs=float(abs(a).mean()),rms=float(np.sqrt(np.mean(a*a)))))
    for order in [1,2]:
        collections=[set((j,) for j in e.mains) if order==1 else set(e.pairs) for e in exs]
        for a,b in itertools.combinations(range(3),2):
            common=collections[a]&collections[b];union=collections[a]|collections[b]
            correlations=[];differences=[];amplitude_ratios=[]
            for key in common:
                u=exs[a].component(tr['X'],key,TIMES);v=exs[b].component(tr['X'],key,TIMES)
                for h in [12,24]:
                    ix=np.searchsorted(TIMES,tr[f'h{h}'])
                    for cause in [1,2]:
                        z=u[np.arange(len(u)),ix,cause-1];w=v[np.arange(len(v)),ix,cause-1]
                        corr=float(np.corrcoef(z,w)[0,1]) if z.std()>1e-14 and w.std()>1e-14 else np.nan
                        rms=float(np.sqrt(np.mean((z-w)**2)));ampu=float(np.sqrt(np.mean(z*z)));ampv=float(np.sqrt(np.mean(w*w)))
                        correlations.append(corr);differences.append(rms);amplitude_ratios.append(min(ampu,ampv)/max(ampu,ampv) if max(ampu,ampv)>1e-14 else np.nan)
            finite=np.asarray(correlations);finite=finite[np.isfinite(finite)]
            seed_stability.append(dict(order=order,seed_a=SEEDS[a],seed_b=SEEDS[b],jaccard=len(common)/len(union) if union else 1.,common_functions=len(common),union_functions=len(union),
                common_shape_correlation_mean=float(finite.mean()) if len(finite) else np.nan,common_rms_difference_mean=float(np.mean(differences)) if differences else np.nan,
                common_amplitude_ratio_mean=float(np.nanmean(amplitude_ratios)) if np.isfinite(amplitude_ratios).any() else np.nan))
    # Same deterministic historical rows for every subset; company identity stays private.
    local=tr['X'][:min(8,len(tr['X']))]
    values=np.stack([ex.component(local,k,TIMES) for k in keys],axis=1) if keys else np.empty((len(local),0,4,2))
    np.savez_compressed(out/'local_explanation_PRIVATE.npz',keys=np.array([','.join(str(keep[j]) for j in k) for k in keys]),terms=values,mu=ex._coeff(TIMES)[1],sample_ids=tr['sample_ids'][:len(local)])
    pd.DataFrame(component_rows).to_csv(out/'component_summary.csv',index=False);pd.DataFrame(seed_stability).to_csv(out/'seed_stability.csv',index=False)
    summary=dict(status='COMPLETE',kind=spec['kind'],year=year,subset_id=spec['subset_id'],repeat=spec.get('repeat',-1),new_forest_fits=0,derived_ensemble=1,
        features=len(keep),mode=spec['mode'],host=socket.gethostname(),check=check,ensemble_mean_error=difference,seconds=time.perf_counter()-start,
        peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,forest_hashes=spec['forest_hashes'],bootstrap_selection_gate=bootstrap_gate)
    write(out/'summary.json',summary);write(out/'hashes.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()});print(json.dumps({k:v for k,v in summary.items() if k not in ['check','forest_hashes']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);a=p.parse_args()
    try:execute(a.index)
    except Exception:
        out=Path(os.environ['OUTPUT_DIR']);out.mkdir(parents=True,exist_ok=True);write(out/'FAILED.json',dict(index=a.index,traceback=traceback.format_exc()));raise
