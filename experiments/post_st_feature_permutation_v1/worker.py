"""Disjoint permutation items from immutable saved forests; fit is disabled."""
import argparse,json,os,resource,socket,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
from crforest import CompetingRiskForest,CIFExpansion
from experiments.post_st_ablation_v1.verification import merge
from experiments.post_st_prognosis_v1.common import horizon_predictions
from .common import *


def interval(x):
    z=np.asarray(x);z=z[np.isfinite(z)]
    return list(map(float,np.quantile(z,[.025,.975]))) if len(z) else [np.nan,np.nan]


def run(a):
    started=time.perf_counter();CompetingRiskForest.fit=prohibit_fit
    root=Path(os.environ['DATA_DIR']);out=Path(os.environ['OUTPUT_DIR']);out.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((root/'protocol.json').read_text());src=root/'folds'/str(a.cohort)
    tr,ev=arrays(src/'train_PRIVATE.npz'),arrays(src/'eval_PRIVATE.npz')
    ex=CIFExpansion.load(src/'functions_PRIVATE.npz');base=arrays(src/'baseline_PRIVATE.npz')['pred']
    members=[CompetingRiskForest.load(src/'models'/str(s)/'forest_PRIVATE.json') for s in SEEDS];forest=merge(members)
    draws=arrays(src/'draws_PRIVATE.npz');perm=draws['permutations'];counts=draws['bootstrap'];X=ev['X']
    items=json.loads((src/'items.json').read_text());features=json.loads((root/'features.json').read_text())
    contexts=[];base_rows=[]
    for outer in protocol['outer_cutoffs']:
        if a.cohort>=outer:continue
        labels=arrays(src/f'labels_{outer}_PRIVATE.npz')
        for h in [12,24]:
            hpred=horizon_predictions(base,TIMES,labels[f'h{h}'])
            for cause in [1,2]:
                y,w,known,supported,diag=evaluation_context(labels,h,cause)
                mature=a.cohort+h//12<outer
                tags=dict(outer_cutoff=outer,cohort=a.cohort,horizon=h,cause=cause,mature=mature,supported=supported,
                    train_events=int((tr['event']==cause).sum()),validation_events=int((y&known).sum()),controls=int((~y&known).sum()),unknown=int((~known).sum()),n=len(y))
                tags['informative']=bool(mature and supported and tags['train_events']>=1 and tags['validation_events']>=3 and tags['controls']>=5)
                metrics=score_batch(hpred[:,cause-1],y,w)
                base_rows.append(dict(**tags,**{k:float(v[0]) if supported else np.nan for k,v in metrics.items()},min_G=diag['min_G']))
                contexts.append((tags,labels,y,w,hpred[:,cause-1]))
    rows=[];boots=[];interaction_boots=[];components=[];errors=[];details=[]
    for item_index,item in enumerate(items):
        if item_index%protocol['shards_per_cohort']!=a.shard:continue
        moved=item['moved'];lookup=replacement_lookup(ex,X,moved)
        accelerated=base[None,:,:,:]+lookup[np.arange(len(X))[None,:],perm]
        xx=np.broadcast_to(X,(len(perm),*X.shape)).copy()
        for j in moved:xx[:,:,j]=X[perm,j]
        # Score the direct forest output for EVERY permutation. Algebraic
        # roundoff can otherwise split exact risk ties and alter AUC/AP.
        pp=forest.predict(xx.reshape(-1,83),TIMES).reshape(accelerated.shape)
        err=float(abs(pp-accelerated).max())
        assert err<1e-10
        assert pp.min()>=-1e-12 and pp.sum(-1).max()<=1+1e-12 and np.diff(pp,axis=2).min(initial=0)>=-1e-12
        errors.append(err)
        key=tuple(item['key']);kind=item['kind'];label=' × '.join(features[j]['label'] for j in key)
        descriptor=dict(item=item_index,kind=kind,key=','.join(map(str,key)),names=' | '.join(features[j]['name'] for j in key),label=label,
            moved=','.join(map(str,moved)),linked_group=len(moved)>len(key))
        component=ex.component(X,key,TIMES) if kind in ['single','pair'] else None
        training_component=ex.component(tr['X'],key,TIMES) if component is not None else None
        if component is not None:
            for h in [12,24]:
                # Validation h arrays are identical across outer label cutoffs.
                one=contexts[0][1];v=horizon_predictions(component,TIMES,one[f'h{h}'])
                t=horizon_predictions(training_component,TIMES,tr[f'h{h}'])
                for c in [1,2]:
                    components.append(dict(cohort=a.cohort,**descriptor,horizon=h,cause=c,train_rms=float(np.sqrt(np.mean(t[:,c-1]**2))),
                        validation_rms=float(np.sqrt(np.mean(v[:,c-1]**2))),train_mean_abs=float(abs(t[:,c-1]).mean()),validation_mean_abs=float(abs(v[:,c-1]).mean())))
        if kind=='pair':
            cuts=[ex.cuts(key,j) for j in key];bins=[]
            for j,cc in zip(key,cuts):
                b=np.searchsorted(cc,tr['X'][:,j],side='left');b[np.isnan(tr['X'][:,j])]=len(cc)+1;bins.append(b)
            shape=tuple(len(c)+2 for c in cuts);flat=np.ravel_multi_index(tuple(bins),shape);support=np.bincount(flat,minlength=int(np.prod(shape)))
            targets=np.bincount(flat,weights=tr['event']==1,minlength=len(support))
            details.append(dict(cohort=a.cohort,**descriptor,cells=len(support),zero_support_cells=int((support==0).sum()),
                cells_ge5_companies=int((support>=5).sum()),cells_ge2_target_events=int((targets>=2).sum()),
                train_mass_supported5=float(np.sum(support[support>=5])/len(tr['X']))))
        for context_index,(tags,labels,y,w,original) in enumerate(contexts):
            row=dict(**tags,**descriptor)
            h,c=tags['horizon'],tags['cause'];ti=np.searchsorted(TIMES,labels[f'h{h}'])
            p=pp[:,np.arange(len(X)),ti,c-1]
            boot=np.full((protocol['company_bootstrap'],2),np.nan);iboot=np.full(protocol['company_bootstrap'],np.nan)
            if tags['supported']:
                effects=metric_effect(original,p,y,w);boot=bootstrap_effect(original,p,y,w,counts)
                row.update({f'delta_{k}':v for k,v in effects.items()})
                for i,k in enumerate(['auc','brier']):
                    lo,hi=interval(boot[:,i]);row.update({f'{k}_lo':lo,f'{k}_hi':hi,f'{k}_bootstrap_undefined':int((~np.isfinite(boot[:,i])).sum())})
                # Pure native pair deletion changes a score, NOT a valid CIF.
                if kind=='pair':
                    term=component[np.arange(len(X)),ti,c-1];score=original-term
                    d=metric_effect(original,score[None,:],y,w)
                    iboot=bootstrap_effect(original,score[None,:],y,w,counts)[:,0]
                    row.update(component_delta_auc=d['auc'],component_delta_ap=d['ap'],component_score_outside01=int(((score<0)|(score>1)).sum()))
                    lo,hi=interval(iboot);row.update(component_auc_lo=lo,component_auc_hi=hi)
            else:
                row.update({f'delta_{k}':np.nan for k in ['auc','ap','brier']})
            row['bootstrap_row']=len(boots);rows.append(row);boots.append(boot);interaction_boots.append(iboot)
    pd.DataFrame(rows).to_csv(out/'effects.csv',index=False)
    pd.DataFrame(components).to_csv(out/'components.csv',index=False)
    pd.DataFrame(details).to_csv(out/'pair_support.csv',index=False)
    pd.DataFrame(base_rows).to_csv(out/'baseline.csv',index=False)
    np.savez_compressed(out/'bootstrap_PRIVATE.npz',effects=np.asarray(boots),component_auc=np.asarray(interaction_boots))
    summary=dict(status='COMPLETE',cohort=a.cohort,shard=a.shard,items=len(errors),metric_rows=len(rows),new_fits=0,
        permutations=protocol['repeats'],company_bootstraps=protocol['company_bootstrap'],maximum_exact_permutation_error=max(errors,default=0),
        host=socket.gethostname(),seconds=time.perf_counter()-started,peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    write(out/'summary.json',summary);write(out/'hashes.json',{p.name:sha(p) for p in out.iterdir() if p.is_file()})
    print(json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cohort',type=int,required=True);p.add_argument('--shard',type=int,required=True);a=p.parse_args()
    try:run(a)
    except Exception:
        out=Path(os.environ['OUTPUT_DIR']);out.mkdir(parents=True,exist_ok=True)
        write(out/'FAILED.json',dict(arguments=vars(a),traceback=traceback.format_exc()));raise
