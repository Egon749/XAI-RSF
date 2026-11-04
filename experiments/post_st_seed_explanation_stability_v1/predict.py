"""Evaluate each combination at each firm's calendar endpoint then pool if supported."""
import pandas as pd
from itertools import product
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score,average_precision_score
from experiments.post_st_requested_seeds_v1.analyze import fast_metrics,METRICS
from experiments.post_st_dropcolumn_selection_v1.common import evaluation_context,metrics
from experiments.post_st_prognosis_v1.common import horizon_predictions
from experiments.post_st_ablation_v1.analyze import selected
from .common import *
from .similarity import *

def analyze():
    r=run_path();root=r/'input'; reports=[];similar=[];private=[]
    labels={y:arrays(root/f'evaluation_input/{y}_test_PRIVATE.npz') for y in YEARS}
    historical=pd.read_csv(BASE/'METRICS_ALL.csv',dtype={'scope':str,'seed':str})
    cases={}
    for y,v in product(YEARS,VERSIONS):
        preds={}
        for arm in [*map(str,SEEDS),'ensemble320','reference96']:
            z=arrays(root/str(y)/arm/v/'predictions_PRIVATE.npz')
            np.testing.assert_array_equal(z['sample_ids'],labels[y]['sample_ids']);np.testing.assert_array_equal(z['times'],TIMES)
            preds[arm]=z['cif']
        for name,idx in zip(NAMES,TRIPLES):preds[name]=np.mean([preds[str(SEEDS[j])] for j in idx],axis=0)
        for h,c in product((12,24),(1,2)):
            years=np.full(len(labels[y]['X']),y);ids=labels[y]['sample_ids']
            yy,w,known,supported,_=evaluation_context(labels[y],h,c)
            if supported:assert known.all() and (w==1).all()
            allp={}
            for arm,pp in preds.items():
                p=horizon_predictions(pp,TIMES,labels[y][f'h{h}'])[:,c-1];allp[arm]=p
                z=metrics(labels[y],p,h,c);z.update(calibration_gap=z['risk_mean']-z['observed'],within_year_auc=z['auc'])
                if supported:
                    cases[y,arm,v,h,c]=(p,yy,years,ids)
                    if yy.any() and (~yy).any():np.testing.assert_allclose([z['auc'],z['ap']],[roc_auc_score(yy,p),average_precision_score(yy,p)],atol=1e-13,rtol=0)
                reports.append(dict(scope=str(y),model=arm,version=v,months=h,cause=c,**z))
            matrix=np.array([allp[a] for a in NAMES]);chosen=np.array([selected(p,years,ids) for p in matrix])
            context=dict(year=y,version=v,months=h,cause=c)
            for metric,values in [('rank_spearman',correlations(rankdata(matrix,axis=1))[LEFT,RIGHT]),('top20_jaccard',jaccards(chosen)[LEFT,RIGHT])]:
                similar+=similarity_rows(values,**context,metric=metric)
            for metric,values in [('company_risk_range',matrix.max(0)-matrix.min(0)),('company_risk_sd',matrix.std(0,ddof=1)),('company_top20_frequency',chosen.mean(0))]:
                similar.append(dict(**context,metric=metric,shared_members='firm',**distributions(values)))
            private.extend(dict(**context,row=i,score_min=matrix[:,i].min(),score_max=matrix[:,i].max(),top20_frequency=chosen[:,i].mean()) for i in range(len(ids)))
    for arm,v,h,c in product(preds,VERSIONS,(12,24),(1,2)):
        years=YEARS[:2] if h==24 else YEARS
        p,yy,yrs,ids=[np.concatenate(parts) for parts in zip(*(cases[y,arm,v,h,c] for y in years))]
        vals=fast_metrics(yy,p,yrs,ids);np.testing.assert_allclose(vals[:3],[roc_auc_score(yy,p),average_precision_score(yy,p),np.mean((p-yy)**2)],atol=1e-13,rtol=0)
        row=dict(zip(METRICS,vals));row['captured']=row.pop('top20_captured');row['coverage']=row.pop('top20_coverage')
        reports.append(dict(scope='pooled',model=arm,version=v,months=h,cause=c,companies=len(yy),target_events=int(yy.sum()),observed=yy.mean(),supported=True,checked=sum(int(np.ceil(.2*sum(yrs==y))) for y in years),unknown=0,**row))
    df=pd.DataFrame(reports)
    ix=['scope','model','version','months','cause'];fields=['auc','ap','brier','risk_mean','calibration_gap','within_year_auc','captured','coverage']
    hist=historical.rename(columns={'seed':'model'}).set_index(ix)
    np.testing.assert_allclose(df.set_index(ix).loc[hist.index,fields],hist[fields],atol=1e-12,rtol=0,equal_nan=True)
    out=[]
    for key,g in df.groupby(['scope','version','months','cause']):
        group=g.set_index('model')
        for metric in fields:
            x=group.loc[NAMES,metric].values
            ref=group.at['reference96',metric];big=group.at['ensemble320',metric]
            out.append(dict(zip(['scope','version','months','cause'],key))|dict(metric=metric,**distributions(x),reference96=ref,reference_percentile=midpercent(x,ref),ensemble320=big,ensemble320_percentile=midpercent(x,big)))
    for name,obj in [('ALL_METRICS',df),('METRIC_DISTRIBUTIONS',pd.DataFrame(out)),('PREDICTION_STABILITY',pd.DataFrame(similar))]:obj.to_csv(HERE/(name+'.csv'),index=False)
    pd.DataFrame(private).to_csv(r/'company_ranges_PRIVATE.csv',index=False)
if __name__=='__main__':analyze()
