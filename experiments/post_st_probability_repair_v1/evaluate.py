"""Fixed-prediction paired evaluation; no feedback into level parameters."""
import os
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from experiments.post_st_benchmark_calibration_v1.common import arrays,old,read,sha,write,endpoints
from experiments.post_st_benchmark_calibration_v1.bootstrap import assessment,FIELDS
from experiments.post_st_dropcolumn_selection_v1.common import evaluation_context

D=Path(__file__).resolve().parent
YEARS=[2023,2024,2025]


def main():
    root=Path(os.environ['REPAIR_RUN'])
    book={};validation=[];failed=[];diags=[]
    for path in root.glob('published/*/task-*/output/summary.json'):
        s=read(path);y=s['year'];assert y not in book
        for name,n,h in read(path.parent.parent/'complete.json')['manifest']:
            p=path.parent/name;assert p.stat().st_size==n and sha(p)==h
        book[y]={}
        for r in s['results']:
            name=r['arm'];book[y][name]=arrays(path.parent/name/'predictions_PRIVATE.npz')
            validation.append(dict(year=y,arm=name,**r['validation'],certificate=r['certificate'],seconds=r['seconds']))
            diags.append(dict(year=y,arm=name,**r['diagnostic']))
        failed.extend([dict(year=y,**f) for f in s['failures']])
    assert set(book)==set(YEARS)
    pd.DataFrame(validation).to_csv(D/'FUNCTION_VALIDATION.csv',index=False)
    pd.DataFrame(diags).to_csv(D/'MAPPING_DIAGNOSTICS.csv',index=False)
    write(D/'FAILURES.json',failed)
    labels={y:arrays(old('post_st_seed_explanation_stability_v1')/f'input/evaluation_input/{y}_test_PRIVATE.npz') for y in YEARS}
    drawpath=old('post_st_cause_support_v1')/'private/bootstrap_indices_PRIVATE.npz'
    draws=arrays(drawpath);rng=np.random.default_rng(2026110201)
    count={}
    for y,lab in labels.items():
        n=len(lab['groups'])
        np.testing.assert_array_equal(draws[str(y)],rng.integers(n,size=(1000,n)))
        count[y]=np.vstack([np.ones(n,int),*[np.bincount(z,minlength=n) for z in draws[str(y)]]])
        for z in book[y].values():np.testing.assert_array_equal(z['sample_ids'],lab['sample_ids'])
    rows=[];intervals=[];tiers=[];joint=[];identities=[]
    baseline=pd.read_csv(D.parent/'post_st_benchmark_calibration_v1/METRICS_ALL.csv',dtype={'scope':str})
    both={}
    for h in [12,24]:
        for cause in [1,2]:
            for scope in [*YEARS,'pooled']:
                ys=([scope] if scope!='pooled' else YEARS if h==12 else YEARS[:2])
                supported=(scope,h)!=(2025,24)
                truth=[];ids=[];years=[]
                for y in ys:
                    yy,w,known,sup,_=evaluation_context(labels[y],h,cause)
                    assert sup==supported
                    if sup:assert known.all() and np.all(w==1)
                    truth.extend(yy);ids.extend(labels[y]['sample_ids']);years.extend([y]*len(yy))
                truth=np.array(truth,int);ids=np.array(ids);years=np.array(years)
                counts=np.column_stack([count[y] for y in ys]);vals={}
                common=set.intersection(*[set(book[y]) for y in ys])
                for arm in ['reference']+sorted(common-{'reference'}):
                    p=np.concatenate([endpoints(book[y][arm]['cif'],labels[y][f'h{h}'])[:,cause-1] for y in ys])
                    base=dict(scope=str(scope),months=h,cause=cause,arm=arm,companies=len(p))
                    if not supported:
                        rows.append(base|dict(mean_prediction_only=p.mean(),reason='2025H24 followup insufficient'))
                        continue
                    a=assessment(counts,truth,p,years,ids);vals[arm]=a
                    rows.append(base|dict(zip(FIELDS,a[0]))|dict(events=int(truth.sum())))
                    both[scope,h,cause,arm]=a[:,FIELDS.index('ipcw_brier')]
                    if arm=='reference':
                        br=baseline[(baseline.scope==str(scope))&(baseline.months==h)&(baseline.cause==cause)&(baseline.model=='XAI/C039')].iloc[0]
                        np.testing.assert_allclose(a[0],[br[f] for f in FIELDS],atol=1e-12,rtol=0,equal_nan=True)
                    if scope!='pooled':
                        raw=endpoints(book[scope]['reference']['cif'],labels[scope][f'h{h}'])[:,cause-1]
                        rawpairs=np.sign(raw[:,None]-raw[None,:]);pairs=np.sign(p[:,None]-p[None,:])
                        assert np.array_equal(pairs,rawpairs)
                        check=['auc','ap','top20_captured','top20_coverage']
                        np.testing.assert_allclose(a[0,[FIELDS.index(k) for k in check]],vals['reference'][0,[FIELDS.index(k) for k in check]],rtol=0,atol=1e-12,equal_nan=True)
                        if len(set(truth))==2:np.testing.assert_allclose(a[0,:3:2],[roc_auc_score(truth,p),average_precision_score(truth,p)],atol=1e-12)
                        identities.append(base|dict(rank_and_ties_exact=True))
                        order=np.lexsort((ids,-raw));sel=np.zeros(len(p),bool);sel[order[:int(np.ceil(.2*len(p)))]]=True
                        for name,mask in [('top20',sel),('remaining80',~sel)]:
                            obs=truth[mask].mean();pred=p[mask].mean()
                            tiers.append(base|dict(tier=name,companies=int(mask.sum()),events=int(truth[mask].sum()),
                                                   observed=obs,predicted=pred,oe=obs/pred if pred>0 else np.nan))
                if not supported:continue
                for arm,a in vals.items():
                    if arm=='reference':continue
                    delta=a-vals['reference']
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore',RuntimeWarning)
                        bounds=np.nanquantile(delta[1:],[.025,.975],axis=0)
                    for j,field in enumerate(FIELDS):
                        intervals.append(dict(scope=str(scope),months=h,cause=cause,arm=arm,metric=field,
                                              difference=delta[0,j],lower95=bounds[0,j],upper95=bounds[1,j],
                                              undefined=int((~np.isfinite(delta[1:,j])).sum())))
    for (scope,h,cause,arm),a in both.items():
        if cause!=1:continue
        b=both[scope,h,2,arm];value=a+b
        diff=value-both[scope,h,1,'reference']-both[scope,h,2,'reference']
        q=np.quantile(diff[1:],[.025,.975])
        joint.append(dict(scope=str(scope),months=h,arm=arm,joint_brier=value[0],difference=diff[0],lower95=q[0],upper95=q[1]))
    for file,data in [('METRICS_ALL',rows),('PAIRED_DIFFERENCES',intervals),('RISK_GROUP_CALIBRATION',tiers),
                      ('JOINT_BRIER',joint),('RANK_IDENTITY',identities)]:
        pd.DataFrame(data).to_csv(D/(file+'.csv'),index=False)
    write(D/'EVALUATION_AUDIT.json',dict(reference_reproduced=True,draw_sha256=sha(drawpath),
          own_year_rank_and_ties_unchanged=True,
          intervals='1000 common company resamples, stratified test year, fixed predictions and mappings; conditional only',
          label_hashes={y:sha(old('post_st_seed_explanation_stability_v1')/f'input/evaluation_input/{y}_test_PRIVATE.npz') for y in YEARS}))


if __name__=='__main__':main()
