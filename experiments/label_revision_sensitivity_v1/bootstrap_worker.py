"""Conditional paired company bootstrap of frozen predictions and alert lists.

Reuses byte-frozen legacy rank, endpoint and calendar-lead helpers. Companies
are sampled once per replicate, jointly across scopes, seeds and models. No fits,
threshold selection, re-ranking or censoring-nuisance re-estimation occurs.
"""
from pathlib import Path
import argparse,json,os,socket,sys,time
import numpy as np
import pandas as pd

def setup(source):
    source=Path(source);sys.path.insert(0,str(source/'legacy'))
    from bootstrap import rank_setup,weighted_rank
    from common import endpoint_weights
    from endpoint_guard import source_consistent_endpoint_weights
    from evaluate import top_flags,event_leads
    d=pd.read_parquet(source/'metadata.parquet');pred=dict(np.load(source/'predictions.npz'))
    groups,gi=np.unique(d.group.astype(str),return_inverse=True);lookup={g:i for i,g in enumerate(groups)}
    all_scopes=[(str(y),np.flatnonzero(d.year.to_numpy()==y)) for y in sorted(d.year.unique())]+[('pooled',np.arange(len(d)))]
    blocks=[]
    for scope,ix in all_scopes:
        dd=d.iloc[ix].reset_index(drop=True);y=(dd.event.to_numpy()==1).astype(float)
        ipcw,_=source_consistent_endpoint_weights(dd,endpoint_weights);rgi=gi[ix]
        bases={'record_equal':np.ones(len(ix)),'company_equal':1/dd.groupby('group').group.transform('size').to_numpy()}
        for label,p0 in sorted(pred.items()):
            model,seed=label.split('|');p=p0[ix];rank=rank_setup(y,p,ipcw>0)
            warn={}
            for f in [.2,.25]:
                flags=top_flags(dd,p,f);lf=event_leads(dd,flags);eg=np.array([lookup[str(g)] for g in lf.group])
                alertg=np.unique(rgi[flags]);hitg=np.unique(rgi[flags&(y==1)])
                unknown=flags&~(dd.label_eligible.to_numpy(bool)|(dd.event.to_numpy()==2))
                neg=flags&(((dd.event.to_numpy()==0)&dd.label_eligible.to_numpy(bool))|(dd.event.to_numpy()==2))
                warn[f]=(flags,lf,eg,alertg,hitg,unknown,neg)
            blocks.append((scope,model,seed,rgi,y,p,ipcw,bases,rank,warn))
    def calc(counts):
        result={}
        def ratio(a,b):return float(a/b) if b>0 else np.nan
        for scope,model,seed,rgi,y,p,ipcw,bases,rank,warn in blocks:
            rw=counts[rgi]
            for weighting,base in bases.items():
                w=rw*base*ipcw;auc,ap=weighted_rank(y,w,rank);prefix=f'{scope}|{model}|{seed}|{weighting}|'
                result[prefix+'auc']=auc;result[prefix+'ap']=ap
                result[prefix+'ipcw_brier']=ratio(np.sum(w*(y-p)**2),np.sum(rw*base))
            for f,(flags,lf,eg,ag,hg,unknown,neg) in warn.items():
                ew=counts[eg];det=np.sum(ew*lf.first_warning.notna());companies=np.sum(counts[ag]);prefix=f'{scope}|{model}|{seed}|Top{int(f*100)}|'
                result[prefix+'event_coverage']=ratio(det,ew.sum())
                result[prefix+'detected_events']=float(det)
                result[prefix+'independent_events']=float(ew.sum())
                result[prefix+'company_hit_fraction']=ratio(np.sum(counts[hg]),companies)
                result[prefix+'alert_companies']=float(companies)
                result[prefix+'alert_records']=float(np.sum(rw*flags))
                result[prefix+'unknown_alert_records']=float(np.sum(rw*unknown))
                result[prefix+'known_no_target_per_event']=ratio(np.sum(rw*neg),det)
                for month in [3,6]:
                    result[prefix+f'lead{month}m_detected']=float(np.sum(ew*lf[f'announcement_detected_{month}m']))
                    result[prefix+f'lead{month}m_opportunity']=float(np.sum(ew*lf[f'announcement_opportunity_{month}m']))
                    result[prefix+f'lead{month}m_coverage']=ratio(np.sum(ew*lf[f'announcement_detected_{month}m']),np.sum(ew*lf[f'announcement_opportunity_{month}m']))
        return result
    return groups,calc

def run(source,output,batch):
    import hashlib
    start=time.perf_counter();groups,calc=setup(source);point=calc(np.ones(len(groups)));keys=sorted(point);values=[];digests=[]
    for rep in range(batch*100,(batch+1)*100):
        rng=np.random.default_rng(np.random.SeedSequence(20261025,spawn_key=(rep,)))
        counts=np.bincount(rng.integers(0,len(groups),len(groups)),minlength=len(groups));res=calc(counts);values.append([res[k] for k in keys]);digests.append(hashlib.sha256(counts.tobytes()).hexdigest())
    with (output/'bootstrap.npz').open('xb') as f:np.savez_compressed(f,names=np.array(keys),point=np.array([point[k] for k in keys]),replicates=np.asarray(values),replicate_ids=np.arange(batch*100,(batch+1)*100),resampling_hashes=np.array(digests))
    with (output/'summary.json').open('x') as f:json.dump(dict(batch=batch,replicates=100,groups=len(groups),seconds=time.perf_counter()-start,host=socket.gethostname(),status='COMPLETE',scope='fixed-prediction conditional company bootstrap; no refits'),f,indent=2)
