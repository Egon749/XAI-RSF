"""Paired information-set comparisons. Inadmissible contrasts retain explicit NA rows."""
from itertools import product
from sklearn.metrics import roc_auc_score,average_precision_score
from experiments.post_st_requested_seeds_v1.analyze import fast_metrics
from experiments.post_st_dropcolumn_selection_v1.common import evaluation_context,metrics
from experiments.post_st_prognosis_v1.common import horizon_predictions
from experiments.post_st_ablation_v1.analyze import selected
from .common import *
NAMES=['auc','ap','negative_brier','mean_prediction','calibration_gap','capture','coverage','within_year_auc','oe']
DESIGNS={'audit_vs_finance':{'M1':1,'M0':-1},'market_vs_finance':{'M2':1,'M0':-1},'audit_given_market':{'M3':1,'M2':-1},'market_given_audit':{'M3':1,'M1':-1},'complementarity':{'M3':1,'M1':-1,'M2':-1,'M0':1},'market_numeric_vs_finance':{'M2_numeric':1,'M0':-1},'coverage_only_vs_finance':{'M0_coverage':1,'M0':-1},'coverage_given_market_numeric':{'M2':1,'M2_numeric':-1},'market_numeric_given_coverage':{'M2':1,'M0_coverage':-1},'audit_numeric_market_minus_audit':{'M3_numeric':1,'M1':-1}}
def score(yy,p,yrs,ids):
    z=fast_metrics(yy,p,yrs,ids)
    return np.array([z[0],z[1],-z[2],z[3],z[4],z[6],z[7],z[8],yy.mean()/z[3] if z[3]>0 else np.nan])
def summarize():
    r=run();g=read(D/'SOURCE_AND_TIME_AUDIT.json')
    arms=gate_arms(g);labels={y:arrays(r/f'eval/{y}.npz') for y in YEARS}
    draws=arrays(r/'eval/bootstrap_PRIVATE.npz');rng=np.random.default_rng(2026110201)
    for y in YEARS:np.testing.assert_array_equal(draws[str(y)],rng.integers(len(labels[y]['X']),size=(1000,len(labels[y]['X']))))
    paths={}
    for stage in ('fit','ensemble'):
        for z in read(r/stage/'published.json'):paths[z['year'],z['arm'],z['variant'],z['seed']]=Path(z['path'])/'shrunk'
    pred={};metricrows=[];cal=[];listrows=[];risks={}
    for y,a,v in product(YEARS,arms,['fixed25','proportional']):
        if v not in variants(a):continue
        sources=[]
        for s in SEEDS:
            z=arrays(paths[y,a,v,s]/'predictions_PRIVATE.npz')
            np.testing.assert_array_equal(z['times'],TIMES);np.testing.assert_array_equal(z['sample_ids'],labels[y]['sample_ids']);sources.append(z['cif'])
        ensemble=np.mean(sources,axis=0)
        np.testing.assert_allclose(ensemble,arrays(paths[y,a,v,0]/'predictions_PRIVATE.npz')['cif'],rtol=0,atol=1e-12)
        for s,cur in zip([*SEEDS,0],[*sources,ensemble]):
            for h,c in product([12,24],[1,2]):
                p=horizon_predictions(cur,TIMES,labels[y][f'h{h}'])[:,c-1];z=metrics(labels[y],p,h,c);yy,w,known,ok,_=evaluation_context(labels[y],h,c)
                key=dict(year=y,arm=a,variant=v,seed=s,months=h,cause=c)
                risks[y,a,v,s,h,c]=p
                metricrows.append(key|z|dict(within_year_auc=z['auc'],calibration_gap=z['risk_mean']-z['observed'],oe=z['observed']/p.mean() if p.mean()>0 else np.nan,status='EVALUATED' if ok else 'NA_FOLLOWUP'))
                if not ok:continue
                assert known.all() and (w==1).all()
                yrs=np.full(len(p),y);ids=labels[y]['sample_ids'];res=score(yy,p,yrs,ids)
                np.testing.assert_allclose(res[:3],[roc_auc_score(yy,p) if len(np.unique(yy))==2 else np.nan,average_precision_score(yy,p) if yy.any() else np.nan,-np.mean((p-yy)**2)],atol=1e-12,rtol=0,equal_nan=True)
                pred[y,a,v,s,h,c]=(p,yy,yrs,ids)
                if s!=0:continue
                chosen=selected(p,yrs,ids);assert int((yy&chosen).sum())==z['captured']
                listrows.extend(key|dict(row=i,sample_id=str(ids[i]),risk=float(p[i]),event=bool(yy[i]),top20=bool(chosen[i])) for i in range(len(p)))
                rank=np.lexsort((ids,-p));bins=np.array_split(rank,4)
                for j,ii in enumerate(bins):
                    obs=float(yy[ii].mean());avg=float(p[ii].mean())
                    cal.append(key|dict(risk_quartile_high_first=j+1,companies=len(ii),observed=obs,prediction=avg,gap=avg-obs,oe=obs/avg if avg>0 else np.nan))
    pd.DataFrame(listrows).to_parquet(r/'rankings_PRIVATE.parquet',index=False)
    pd.DataFrame(cal).to_csv(D/'CALIBRATION_DIAGNOSTICS.csv',index=False)
    np.savez_compressed(r/'risk_PRIVATE.npz',**{'|'.join(map(str,k)):a for k,a in risks.items()})
    points=[];intervals=[];overlap=[]
    for h,c,scope,v in product([12,24],[1,2],[*YEARS,'pooled'],['fixed25','proportional']):
        years=(YEARS if h==12 else YEARS[:2]) if scope=='pooled' else [scope]
        supported=not (h==24 and scope==2025)
        allowed=[a for a in arms if v in variants(a)]
        # An unchanged 74-feature M3 has identical pool under both variants, but inclusion requires disclosure proof.
        contrasts={k:m for k,m in DESIGNS.items() if v=='fixed25' or k in list(DESIGNS)[:5]}
        for s in [*SEEDS,0]:
            values={};boot={};alerts={}
            for a in allowed:
                if not supported:continue
                p,yy,yrs,ids=[np.concatenate(part) for part in zip(*(pred[y,a,v,s,h,c] for y in years))]
                z=score(yy,p,yrs,ids);values[a]=z
                if scope=='pooled':
                    metricrows.append(dict(year=scope,arm=a,variant=v,seed=s,months=h,cause=c,companies=len(p),target_events=int(yy.sum()),observed=yy.mean(),risk_mean=z[3],auc=z[0],ap=z[1],brier=-z[2],captured=z[5],coverage=z[6],within_year_auc=z[7],calibration_gap=z[4],oe=z[8],supported=True,status='EVALUATED',checked=sum(int(np.ceil(.2*(yrs==y).sum())) for y in years)))
                if s==0:
                    alerts[a]=selected(p,yrs,ids)
                    offsets=np.r_[0,np.cumsum([len(labels[y]['X']) for y in years])[:-1]]
                    ix=np.concatenate([draws[str(y)]+offsets[i] for i,y in enumerate(years)],axis=1)
                    boot[a]=np.stack([score(yy[draw],p[draw],yrs[draw],ids[draw]) for draw in ix])
            for effect,weights in contrasts.items():
                key=dict(year=scope,months=h,cause=c,variant=v,seed=s,effect=effect)
                ok=supported and set(weights)<=set(allowed)
                reason='NA_FOLLOWUP' if not supported else ('NA_AUDIT_CLOCK' if not ok else 'EVALUATED')
                est=sum(w*values[a] for a,w in weights.items()) if ok else np.full(len(NAMES),np.nan)
                for j,name in enumerate(NAMES):
                    points.append(key|dict(metric=name,delta=est[j],status=reason if not ok or np.isfinite(est[j]) else 'NA_ENDPOINT_NO_CASES_OR_CONTROLS'))
                if s!=0:continue
                if ok and len(weights)==2:
                    hi,lo=[next(a for a,w in weights.items() if w==i) for i in (1,-1)]
                    add=alerts[hi]&~alerts[lo];lose=alerts[lo]&~alerts[hi];common=alerts[hi]&alerts[lo]
                    overlap.append(key|dict(shared_companies=int(common.sum()),added=int(add.sum()),removed=int(lose.sum()),added_events=int(yy[add].sum()),removed_events=int(yy[lose].sum()),delta_captured=int(yy[add].sum()-yy[lose].sum())))
                b=sum(w*boot[a] for a,w in weights.items()) if ok else None
                for j,name in enumerate(NAMES):
                    good=b[:,j][np.isfinite(b[:,j])] if b is not None else np.array([])
                    low,high=np.quantile(good,[.025,.975]) if len(good) else [np.nan]*2
                    intervals.append(key|dict(metric=name,delta=est[j],lower95=low,upper95=high,valid=len(good),undefined=1000-len(good) if ok else np.nan,undefined_fraction=1-len(good)/1000 if ok else np.nan,repeats=1000 if ok else 0,status=reason))
    table=pd.DataFrame(metricrows)
    # Historical full-model evidence is reported in a separate file. No new audit contrast.
    unavailable=[]
    for a in ['M1','M3','M3_numeric']:
        for y,h,c,s,v in product([*YEARS,'pooled'],[12,24],[1,2],[*SEEDS,0],variants(a)):
            unavailable.append(dict(year=y,arm=a,variant=v,seed=s,months=h,cause=c,status='NA_AUDIT_CLOCK',reason='Actual disclosure of all audit fields not established on same complete cohort'))
    table=pd.concat([table,pd.DataFrame(unavailable)],ignore_index=True)
    table.to_csv(D/'PREDICTIONS_AND_METRICS.csv',index=False)
    df=pd.DataFrame(points);ensemble=df[df.seed==0];ensemble.to_csv(D/'INCREMENTAL_VALUE.csv',index=False);ensemble[ensemble.effect=='complementarity'].to_csv(D/'FACTORIAL_COMPLEMENTARITY.csv',index=False)
    df[df.seed!=0].to_csv(D/'SEED_SENSITIVITY.csv',index=False);pd.DataFrame(intervals).to_csv(D/'PAIRED_DIFFERENCES_95.csv',index=False)
    pd.DataFrame(overlap).to_csv(D/'TOP20_COMPARISONS.csv',index=False)
    annual=ensemble[ensemble.year!='pooled'];annual.groupby(['variant','months','cause','metric','effect']).delta.agg(defined='count',positive=lambda x:(x>1e-14).sum(),negative=lambda x:(x< -1e-14).sum(),tie=lambda x:(x.abs()<=1e-14).sum()).reset_index().to_csv(D/'YEAR_STABILITY.csv',index=False)
    old=pd.read_csv(ROOT/'experiments/post_st_requested_seeds10_v1/METRICS_ALL.csv');old[old.seed=='reference96'].query("version=='shrunk'").assign(status='HISTORICAL_ONLY_AUDIT_CLOCK_NOT_VERIFIED').to_csv(D/'HISTORICAL_M3_ONLY.csv',index=False)
if __name__=='__main__':summarize()
