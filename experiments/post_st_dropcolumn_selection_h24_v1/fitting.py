"""Matched bundles; lossless storage and full native-grid numerical checks."""
from pathlib import Path
import gzip,itertools,json,shutil,time
import numpy as np
import pandas as pd
from crforest import CompetingRiskForest,CIFExpansion
from crforest.shrinkage import shrink_leaf_cifs
from experiments.post_st_ablation_v1.verification import merge
from experiments.post_st_level_coverage_v1.worker import verify_export
from experiments.post_st_dropcolumn_selection_v1.common import configuration,reduced,pairing,array_hash,sha,write
from .selection import *

def bootstrap_groups(parent,outer,repeat):
    registry=load(parent/'audit/registry_bootstrap.json')
    record=next(r for r in registry if r['year']==outer and r['repeat']==repeat and r['seed']==42)
    indices=arrays(Path(record['path'])/'fit_rows_PRIVATE.npz')['indices'];tr=arrays(parent/f'input/outer/{outer}/train_PRIVATE.npz')
    return tr['groups'][indices]

def historical_support(parent,outer,draw_groups=None):
    base=pd.read_csv(parent/'reports/historical_support.csv');base=base[base.outer_cutoff==outer].copy()
    counts={}
    for v in range(2019,outer):
        lab=arrays(parent/f'input/folds/{v}/labels_{outer}_PRIVATE.npz');tr=arrays(parent/f'input/folds/{v}/train_PRIVATE.npz')
        count=np.ones(len(lab['event']),int) if draw_groups is None else np.array([np.sum(draw_groups==g) for g in lab['groups']])
        counts[v]=count;active=count>0
        if draw_groups is None:continue
        ll={k:a[active] for k,a in lab.items()};train_take=np.isin(tr['groups'],np.unique(draw_groups))
        for h in [12,24]:
            y,w,known,ok,_=evaluation_context(ll,h,1);i=(base.cohort==v)&(base.horizon==h)
            base.loc[i,'train_termination']=int(((tr['event']==1)&train_take).sum())
            base.loc[i,'termination']=int((y&known).sum());base.loc[i,'controls']=int((~y&known).sum());base.loc[i,'supported']=ok
    return base.to_dict('records'),counts

def compact_verification(model,reference,evaluation,out):
    """Keep full models/functions/predictions; discard only regenerable scratch."""
    out.mkdir();scratch=out/'verification_work';check=verify_export(model,reference,evaluation,scratch)
    payload=(scratch/'forest_PRIVATE.json').read_bytes()
    with (out/'forest_PRIVATE.json.gz').open('xb') as f:f.write(gzip.compress(payload,compresslevel=6,mtime=0))
    assert gzip.decompress((out/'forest_PRIVATE.json.gz').read_bytes())==payload
    for name in ['functions_PRIVATE.npz','predictions_PRIVATE.npz','complexity_PRIVATE.json','verification.json']:shutil.copy2(scratch/name,out/name)
    # These files contain no unique fit: native model+full function+predictions
    # above preserve the actual results. A failed check never reaches cleanup.
    shutil.rmtree(scratch)
    return check

def fit_bundle(parent,keep,year,out,draw_groups=None,outer_final=False):
    started=time.perf_counter();keep=tuple(keep);out.mkdir(parents=True,exist_ok=False)
    root=parent/'input'/('outer' if outer_final else 'folds')/str(year)
    original=arrays(root/'train_PRIVATE.npz');evaluation=reduced(arrays(root/'eval_PRIVATE.npz'),keep);reference=reduced(original,keep)
    if draw_groups is None:indices=np.arange(len(original['X']))
    else:
        where={g:i for i,g in enumerate(original['groups'])};indices=np.array([where[g] for g in draw_groups if g in where],int)
    train={k:v[indices] for k,v in reference.items()};np.savez_compressed(out/'training_rows_PRIVATE.npz',indices=indices)
    members=[];predictions=[];checks=[];pairsets=[];fit_times=[];shrink_times=[]
    for seed in SEEDS:
        folder=out/str(seed);folder.mkdir();cfg=configuration(seed,len(keep),'count');expected=pairing(train['groups'],cfg,len(keep))
        t=time.perf_counter();raw=CompetingRiskForest(cfg).fit(train['X'],train['time'],train['event'],train['groups'],feature_sources=[(j,) for j in keep],n_jobs=1);fit_times.append(time.perf_counter()-t)
        for actual,want in zip(raw.inbag_,expected):assert array_hash(actual)==want['multiplicity_sha256']
        t=time.perf_counter();sh=shrink_leaf_cifs(raw,train['X'],train['time'],train['event'],train['groups'],10);shrink_times.append(time.perf_counter()-t)
        cc={}
        for label,model in [('raw',raw),('shrunk',sh)]:cc[label]=compact_verification(model,reference,evaluation,folder/label)
        ex=CIFExpansion.load(folder/'shrunk/functions_PRIVATE.npz');pairsets.append(set(ex.pairs));members.append(sh);predictions.append(sh.predict(evaluation['X'],TIMES));checks.append(cc)
        write(folder/'pairing.json',expected)
    forest=merge(members);ensemble_check=compact_verification(forest,reference,evaluation,out/'ensemble')
    pp=forest.predict(evaluation['X'],TIMES);difference=float(abs(pp-np.mean(predictions,axis=0)).max());assert difference<=1e-13
    overlaps=[len(a&b)/len(a|b) if a|b else 1. for a,b in itertools.combinations(pairsets,2)]
    summary=dict(year=year,keep=list(keep),subset_hash=subset_hash(keep),new_forests=3,training_rows=len(indices),training_companies=len(np.unique(train['groups'])),
        unique_training_termination=len(np.unique(train['groups'][train['event']==1])),fit_seconds=fit_times,shrink_seconds=shrink_times,seconds=time.perf_counter()-started,
        checks=checks,ensemble_check=ensemble_check,ensemble_mean_error=difference,cells=ensemble_check['spatial_table_cells'],pair_jaccard=float(np.mean(overlaps)))
    write(out/'summary.json',summary);write(out/'hashes.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})
    return dict(ensemble=pp,seeds=predictions,cells=summary['cells'],pair_jaccard=summary['pair_jaccard'])

def read_bundle(path):
    s=load(path/'summary.json')
    return dict(ensemble=arrays(path/'ensemble/predictions_PRIVATE.npz')['cif'],seeds=[arrays(path/f'{seed}/shrunk/predictions_PRIVATE.npz')['cif'] for seed in SEEDS],cells=s['cells'],pair_jaccard=s['pair_jaccard'])

class BootstrapStore:
    def __init__(self,parent,out,groups):self.parent,self.out,self.groups=parent,out,groups;self.cache={};self.bundles=[]
    def ensure(self,keeps,years):
        for keep in sorted(set(map(tuple,keeps))):
            for year in years:self.get(keep,year)
    def get(self,keep,year):
        key=(tuple(keep),year)
        if key not in self.cache:
            p=self.out/f'{year}_{subset_hash(keep)[:16]}'
            self.cache[key]=fit_bundle(self.parent,keep,year,p,self.groups)
            self.bundles.append(str(p.relative_to(self.out.parent)))
        return self.cache[key]
