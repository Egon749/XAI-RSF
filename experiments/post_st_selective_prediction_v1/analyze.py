import time
import pandas as pd
from scipy.stats import spearmanr
from .common import *
from .gate_selection import load_diagnostics,history
from .metrics import group_metrics,decomposition
from experiments.post_st_feature_permutation_v1.common import evaluation_context

def save(name,rows):pd.DataFrame(rows).to_csv(RUN/f'reports/{name}.csv',index=False)

def simple_bootstrap(y,p,mask,counts):
    mm=counts*mask;den=mm.sum(1);events=mm@y;total=counts@y
    divide=lambda a,b:np.divide(a,b,out=np.full(len(counts),np.nan),where=b>0)
    brier=divide(mm@((p-y)**2),den);risk=divide(mm@p,den);observed=divide(events,den)
    pos=divide(mm@(((p-y)**2)*y),events);neg=divide(mm@(((p-y)**2)*(~y)),den-events)
    return dict(brier=brier,coverage=den/counts.sum(1),event_retention=divide(events,total),risk_mean=risk,calibration_gap=risk-observed,balanced_mse=(pos+neg)/2)

def main():
    start=time.perf_counter();block_forest_fits();frozen=load(RUN/'audit/selection_frozen.json');assert frozen['status']=='PASS'
    assert frozen['masks_sha256']==sha(RUN/'private/masks_PRIVATE.npz');masks=arrays(RUN/'private/masks_PRIVATE.npz');hmasks=arrays(RUN/'private/historical_masks_PRIVATE.npz')
    d=load_diagnostics();lab={y:arrays(RUN/f'evaluation_input/{y}_test_PRIVATE.npz') for y in YEARS};rows=[];mix=[];historical=[];associations=[];contexts={};random_rows=[]
    for year in YEARS:
        np.testing.assert_array_equal(lab[year]['sample_ids'],d[year]['sample_ids'])
        for h in [12,24]:
            membership,parts=history(d,year,h);hl=arrays(RUN/f'private/history_labels_{year}_{h}_PRIVATE.npz');np.testing.assert_array_equal(membership['sample_ids'],hl['sample_ids'])
            for c in [1,2]:
                yy,w,known,supported,diag=evaluation_context(lab[year],h,c);p=d[year][f'{h}_{c}_risk'];ids=d[year]['sample_ids'];yr=np.full(len(ids),year)
                if supported:assert known.all() and np.all(w==1)
                reference=np.concatenate([matrix(v,h,c) for v in parts]);hy=hl['y'][:,c-1].astype(bool);hp=reference[:,5]
                old=arrays(SOURCE/f'input/calibration/{h}/{year}/calibration_PRIVATE.npz');np.testing.assert_allclose(hp,old['p'][:,c-1],atol=1e-14,rtol=0)
                for proxy in RAW:
                    j=RAW.index(proxy)
                    for src,x,y,pred in [('history',reference[:,j],hy,hp),('evaluation',matrix(d[year],h,c)[:,j],yy,p)]:
                        corr=float(spearmanr(x,(pred-y)**2).statistic) if (src=='history' or supported) and np.ptp(x)>0 and np.ptp((pred-y)**2)>0 else np.nan
                        associations.append(dict(year=year,months=h,cause=c,proxy=proxy,source=src,spearman_with_squared_error=corr,supported=(src=='history' or supported)))
                for arm,q in [('all',1.)]+[(a,q) for a in ARMS for q in COVERAGES]:
                    m=np.ones(len(hy),bool) if arm=='all' else hmasks[f'{year}|{h}|{c}|{arm}|{q}']
                    historical.append(dict(cutoff=year,months=h,cause=c,arm=arm,target_coverage=q,**group_metrics(hy,hp,m,membership['sample_ids'],membership['cohort'])))
                cm={'all|1.0':np.ones(len(p),bool)}
                for a in ARMS+['matched_'+a for a in ARMS]:
                    for q in COVERAGES:cm[f'{a}|{q}']=masks[f'{year}|{h}|{c}|{a}|{q}']
                contexts[(str(year),h,c)]=(yy,p,ids,yr,cm,supported)
    for h in [12,24]:
        ys=YEARS if h==12 else [2023,2024]
        for c in [1,2]:
            ctx=[contexts[(str(y),h,c)] for y in ys];assert all(x[5] for x in ctx)
            cm={k:np.concatenate([z[4][k] for z in ctx]) for k in ctx[0][4]}
            contexts[('pooled',h,c)]=(*[np.concatenate([z[i] for z in ctx]) for i in range(4)],cm,True)
    for (scope,h,c),(y,p,ids,yr,cm,supported) in contexts.items():
        full=group_metrics(y,p,cm['all|1.0'],ids,yr) if supported else None
        for key,mask in cm.items():
            arm,q=key.split('|');q=float(q)
            if supported:
                row=group_metrics(y,p,mask,ids,yr);mix.append(dict(scope=scope,months=h,cause=c,arm=arm,target_coverage=q,**decomposition(full,row)))
            else:
                row=dict(companies=len(y),accepted=int(mask.sum()),referred=int((~mask).sum()),coverage=float(mask.mean()),risk_mean=float(p[mask].mean()) if mask.any() else np.nan,
                    **{k:np.nan for k in ['all_events','accepted_events','referred_events','event_retention','brier','observed','calibration_gap','auc','ap','event_mse','nonevent_mse','balanced_mse','accepted_top20_checked','accepted_top20_captured','original_budget_checked','original_budget_captured','original_top20_referred']},reason='NO_FOLLOWUP_SUPPORT_AT_HORIZON')
            rows.append(dict(scope=scope,months=h,cause=c,arm=arm,target_coverage=q,supported=supported,**row))
        if not supported or scope=='pooled':continue
        rng=np.random.default_rng(np.random.SeedSequence([RANDOM_SEED,int(scope),h,c]));perm=np.array([rng.permutation(len(y)) for _ in range(1000)])
        np.savez_compressed(RUN/f'private/random_{scope}_{h}_{c}_PRIVATE.npz',permutations=perm)
        cache={};rank=np.lexsort((ids,-p));budget=int(np.ceil(.2*len(y)))
        for arm in ARMS:
            for q in COVERAGES:
                mask=cm[f'{arm}|{q}'];n=int(mask.sum())
                if n not in cache:
                    mm=np.zeros((1000,len(y)),bool);mm[np.arange(1000)[:,None],perm[:,:n]]=True
                    vals=simple_bootstrap(y,p,np.ones(len(y),bool),mm.astype(int)) if n else {k:np.full(1000,np.nan) for k in ['brier','coverage','event_retention','risk_mean','calibration_gap','balanced_mse']}
                    # simple_bootstrap normalizes by resampled total; here use
                    # all original events for retention, not random selectedevents.
                    vals['event_retention']=mm@y/y.sum() if y.any() else np.full(1000,np.nan)
                    vals['coverage']=np.full(1000,n/len(y))
                    picked=mm[:,rank]&(np.cumsum(mm[:,rank],axis=1)<=budget)
                    vals['original_budget_captured']=picked@y[rank];cache[n]=vals
                actual=group_metrics(y,p,mask,ids,yr)
                for metric,v in cache[n].items():
                    valid=np.isfinite(v);b=np.quantile(v[valid],[.025,.975]) if valid.any() else [np.nan,np.nan]
                    random_rows.append(dict(scope=scope,months=h,cause=c,arm=arm,target_coverage=q,matched_accepted=n,metric=metric,actual=actual[metric],random_mean=float(v[valid].mean()) if valid.any() else np.nan,
                        random_lower95=b[0],random_upper95=b[1],valid_repetitions=int(valid.sum()),random_fraction_leq_actual=float(np.mean(v[valid]<=actual[metric])) if valid.any() and np.isfinite(actual[metric]) else np.nan))
    save('METRICS',rows);save('BRIER_CASE_MIX',mix);save('HISTORICAL_GATE_EVALUATION',historical);save('PROXY_LOSS_ASSOCIATIONS',associations);save('MATCHED_RANDOM_REFERENCE',random_rows)
    # Paired conditional uncertainty. Masks/forests fixed. Same company draws
    # across arms, causes and horizons; pooled draws stratified by calendar year.
    rng=np.random.default_rng(BOOT_SEED);draw={y:rng.integers(len(lab[y]['X']),size=(1000,len(lab[y]['X']))) for y in YEARS}
    np.savez_compressed(RUN/'private/test_bootstrap_indices_PRIVATE.npz',**{str(y):v for y,v in draw.items()});ci=[]
    contrasts={'combined_minus_all':('combined|0.8','all|1.0'),'combined_minus_seed_fixed':('combined|0.8','seed_sd|0.8'),
        'combined_minus_seed_matched':('combined|0.8','matched_seed_sd|0.8'),'combined_minus_support_matched':('combined|0.8','matched_support|0.8'),
        'combined_minus_lowrisk_matched':('combined|0.8','matched_low_risk_control|0.8')}
    for (scope,h,c),(y,p,ids,yr,cm,supported) in contexts.items():
        if not supported:continue
        unique=np.unique(yr);index=np.concatenate([draw[int(v)]+np.flatnonzero(yr==v)[0] for v in unique],axis=1)
        counts=np.stack([np.bincount(i,minlength=len(y)) for i in index]);stats={a:simple_bootstrap(y,p,cm[a],counts) for pair in contrasts.values() for a in pair}
        point={a:group_metrics(y,p,cm[a],ids,yr) for a in stats}
        for name,(a,b) in contrasts.items():
            for metric in stats[a]:
                vals=stats[a][metric]-stats[b][metric];valid=np.isfinite(vals);bounds=np.quantile(vals[valid],[.025,.975]) if valid.any() else [np.nan,np.nan]
                ci.append(dict(scope=scope,months=h,cause=c,contrast=name,metric=metric,estimate=point[a][metric]-point[b][metric],lower95=bounds[0],upper95=bounds[1],replicates=1000,valid_replicates=int(valid.sum()),undefined_replicates=int((~valid).sum())))
    save('PAIRED_BOOTSTRAP_95',ci)
    df=pd.DataFrame(rows);original=pd.read_csv(REPO/'experiments/post_st_temporal_affine_v1/METRICS.csv',dtype={'scope':str})
    for r in df[df.arm=='all'].to_dict('records'):
        old=original[(original.scope==r['scope'])&(original.months==r['months'])&(original.cause==r['cause'])&(original.arm=='baseline')].iloc[0]
        np.testing.assert_allclose([r[k] for k in ['auc','ap','brier','risk_mean','observed']],[old[k] for k in ['auc','ap','brier','risk_mean','observed']],atol=1e-13,rtol=0,equal_nan=True)
    checks=dict(status='PASS',metric_rows=len(df),history_metric_rows=len(historical),paired_intervals=len(ci),random_repetitions=1000,
        original_metrics_reproduced=True,baseline_probabilities_unchanged=True,source_mask_sha256=sha(RUN/'private/masks_PRIVATE.npz'),
        supported_endpoints_complete_unit_IPCW=True,case_mix_max_residual=float(pd.DataFrame(mix).residual.abs().max()),new_forests=0,new_calibrators=0,seconds=time.perf_counter()-start)
    write(RUN/'reports/metric_validation.json',checks)
    print(df[(df.scope!='pooled')&(df.cause==1)&((df.arm=='all')|((df.target_coverage==.8)&df.arm.isin(ARMS)))][['scope','months','arm','accepted','referred','brier','auc','ap','accepted_events','referred_events','event_retention','original_budget_captured']].round(5).to_string(index=False))

if __name__=='__main__':main()
