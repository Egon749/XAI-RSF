"""Historical-only choice precedes exposed outer evaluation; complete aggregates."""
import argparse,warnings
from itertools import combinations
from sklearn.metrics import roc_auc_score,average_precision_score
from experiments.post_st_benchmark_calibration_v1.bootstrap import FIELDS,assessment
from experiments.post_st_dropcolumn_selection_v1.common import metrics,evaluation_context
from experiments.post_st_reporting_drift_v1.metrics import paired_auc
from .common import *

def qtile(x):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        return np.nanquantile(x,[.025,.975],axis=0)

def registry():
    from .prepare import verify_output
    book={};cost=[];hashes=[]
    paths=sorted((run()/'published').glob('*/task-*/output'))
    assert len(paths)==156
    for p in paths:
        s=load(p/'summary.json');assert s['status']=='COMPLETE' and s['mode']=='fit' and s['forest_fits']==3
        verify_output(p)
        key=(s['pool_count'],s['year']);assert key not in book;book[key]=(p,s)
        cost.append(dict(**{k:v for k,v in s.items() if k not in ['members','ensemble_check']},
            total_fit_seconds=sum(m['fit_seconds'] for m in s['members'])))
        assert s['source_train_sha256']==sha(run()/f"input/folds/{s['year']}/train_PRIVATE.npz")
        for member in [*map(str,SEEDS),'ensemble',*[f'raw/{v}' for v in SEEDS]]:
            for name in ['forest_PRIVATE.json','functions_PRIVATE.npz','predictions_PRIVATE.npz']:
                pp=p/member/name;hashes.append(dict(pool_count=key[0],year=key[1],member=member,artifact=name,sha256=sha(pp),bytes=pp.stat().st_size))
    assert set(book)=={(s,y) for s in COUNTS for y in ALL_YEARS}
    pd.DataFrame(cost).to_csv(D/'COMPUTE_COST.csv',index=False)
    pd.DataFrame(hashes).to_csv(D/'ARTIFACT_HASHES.csv',index=False)
    return book

def prediction(book,pool_count,year,member='ensemble'):
    p,_=book[pool_count,year]
    return arrays(p/'mean_predictions_PRIVATE.npz' if member=='ensemble' else p/member/'predictions_PRIVATE.npz')

def historical_selection(book):
    rows=[];selected=[]
    for outer in YEARS:
        for h in [12,24]:
            labpaths=sorted((run()/f'selection_input/{outer}/{h}').glob('*_labels_PRIVATE.npz'))
            assert labpaths
            labels={int(p.name.split('_')[0]):arrays(p) for p in labpaths}
            for k in [1,2]:
                for scope in [*labels,'pooled']:
                    ys=list(labels) if scope=='pooled' else [scope]
                    ids=np.concatenate([labels[y]['sample_ids'] for y in ys])
                    years=np.concatenate([np.repeat(y,len(labels[y]['sample_ids'])) for y in ys])
                    yvalues=[]
                    for y in ys:
                        yy,w,known,supp,_=evaluation_context(labels[y],h,k)
                        assert supp and known.all() and (w==1).all();yvalues.extend(yy)
                    yvalues=np.array(yvalues,int)
                    for pool_count in COUNTS:
                        ps=[]
                        for y in ys:
                            z=prediction(book,pool_count,y)
                            np.testing.assert_array_equal(z['sample_ids'],labels[y]['sample_ids'])
                            ps.append(endpoint(z['cif'],labels[y][f'h{h}'])[:,k-1])
                        p=np.concatenate(ps)
                        point=dict(zip(FIELDS,assessment(np.ones((1,len(ids)),int),yvalues,p,years,ids)[0]))
                        rows.append(dict(outer_cutoff=outer,scope=scope,months=h,cause=k,pool_count=pool_count,
                            companies=len(ids),events=int(yvalues.sum()),**point))
        candidates=[r for r in rows if r['outer_cutoff']==outer and r['scope']=='pooled' and r['months']==12 and r['cause']==1]
        value,reason=select_count(candidates)
        chosen=next(r for r in candidates if r['pool_count']==value)
        selected.append(dict(outer_cutoff=outer,selected_count=value,reason=reason,
            historical_termination_events=chosen['events'],historical_companies=chosen['companies'],
            within_year_auc=chosen['within_year_auc'],ap=chosen['ap'],ipcw_brier=chosen['ipcw_brier']))
    pd.DataFrame(rows).to_csv(D/'HISTORICAL_VALIDATION_METRICS.csv',index=False)
    pd.DataFrame(selected).to_csv(D/'HISTORICAL_SELECTED_COUNTS.csv',index=False)
    write(run()/'selection_locked_PRIVATE.json',dict(selected=selected,outer_labels_read=False,
        protocol_sha256=sha(D/'FROZEN_PROTOCOL.json'),history_metrics_sha256=sha(D/'HISTORICAL_VALIDATION_METRICS.csv'),
        historical_label_hashes={str(p.relative_to(run())):sha(p) for p in (run()/'selection_input').rglob('*.npz')}))
    print(pd.DataFrame(selected).to_string(index=False))

def numerical(book):
    checks=[];replay=[];amplitude=[];stability=[];seed_variation=[];repro=[]
    for (pool_count,year),(p,s) in book.items():
        for m in s['members']:
            ident=dict(pool_count=pool_count,year=year,member=str(m['seed']))
            checks.append(dict(**ident,version='shrunk',**m['checks']))
            checks.append(dict(**ident,version='raw',**m['raw_checks']))
            replay.append(dict(**ident,**{k:v for k,v in m.items() if k not in ['seed','checks','raw_checks','reproduction','independent_checks']}))
            if m['reproduction']:repro.append(dict(**ident,**m['reproduction']))
        checks.append(dict(pool_count=pool_count,year=year,member='ensemble',version='shrunk',**s['ensemble_check'],ensemble_mean_error=s['ensemble_mean_error']))
        comps={str(seed):arrays(p/str(seed)/'components_PRIVATE.npz') for seed in SEEDS}
        comps['ensemble']=arrays(p/'ensemble/components_PRIVATE.npz')
        for member,z in comps.items():
            keys=list(z['keys'])
            for order in [1,2]:
                indices=[i for i,key in enumerate(keys) if key.count(',')+1==order]
                for hi,day in [(0,365),(2,730)]:
                    for k in [1,2]:
                        terms=z['terms'][:,indices,hi,k-1]
                        amplitude.append(dict(pool_count=pool_count,year=year,member=member,order=order,day=day,cause=k,
                            reference_mean_sum_absolute_contribution=float(abs(terms).sum(1).mean()),
                            reference_mean_abs_total_contribution=float(abs(terms.sum(1)).mean())))
        for a,b in combinations(map(str,SEEDS),2):
            ka=list(comps[a]['keys']);kb=list(comps[b]['keys'])
            for order in [1,2]:
                aa={x for x in ka if x.count(',')+1==order};bb={x for x in kb if x.count(',')+1==order};cc=sorted(aa&bb)
                for hi,day in [(0,365),(2,730)]:
                    for k in [1,2]:
                        cor=[];diff=[]
                        for key in cc:
                            x=comps[a]['terms'][:,ka.index(key),hi,k-1];z=comps[b]['terms'][:,kb.index(key),hi,k-1]
                            diff.append(float(abs(x-z).mean()))
                            if np.std(x)>1e-14 and np.std(z)>1e-14:cor.append(float(np.corrcoef(x,z)[0,1]))
                        stability.append(dict(pool_count=pool_count,year=year,seed_a=a,seed_b=b,order=order,day=day,cause=k,
                            jaccard=len(cc)/len(aa|bb) if aa|bb else 1.,common_functions=len(cc),
                            estimable_correlations=len(cor),mean_correlation=np.mean(cor) if cor else np.nan,
                            mean_abs_function_difference=np.mean(diff) if diff else np.nan))
        pp=np.stack([prediction(book,pool_count,year,str(s))['cif'] for s in SEEDS])
        for hi,day in [(0,365),(2,730)]:
            for k in [1,2]:
                sd=pp[:,:,hi,k-1].std(axis=0,ddof=1)
                seed_variation.append(dict(pool_count=pool_count,year=year,day=day,cause=k,mean_seed_sd=sd.mean(),max_seed_sd=sd.max()))
    for name,rows in [('NUMERICAL_COMPLEXITY',checks),('INDEPENDENT_SHRINKAGE',replay),('FUNCTION_AMPLITUDE',amplitude),
        ('FUNCTION_STABILITY',stability),('PREDICTION_STABILITY',seed_variation),('REFERENCE_REPRODUCTION',repro)]:
        pd.DataFrame(rows).to_csv(D/f'{name}.csv',index=False)
    assert len(repro)==18

def outer_evaluation(book):
    locked=load(run()/'selection_locked_PRIVATE.json')
    assert not locked['outer_labels_read'] and locked['protocol_sha256']==sha(D/'FROZEN_PROTOCOL.json')
    assert locked['history_metrics_sha256']==sha(D/'HISTORICAL_VALIDATION_METRICS.csv')
    selected={r['outer_cutoff']:r['selected_count'] for r in locked['selected']}
    labs={y:arrays(source()/f'input/evaluation_input/{y}_test_PRIVATE.npz') for y in YEARS}
    drawfile=Path(load(D.parent/'post_st_cause_support_v1/run_location.json')['private_run'])/'private/bootstrap_indices_PRIVATE.npz'
    draws=arrays(drawfile);rng=np.random.default_rng(2026110201);counts={}
    for y in YEARS:
        n=len(labs[y]['sample_ids']);assert len(set(labs[y]['groups']))==n
        np.testing.assert_array_equal(draws[str(y)],rng.integers(n,size=(1000,n)))
        counts[y]=np.vstack([np.ones(n,int),*[np.bincount(row,minlength=n) for row in draws[str(y)]]])
        tr=arrays(run()/f'input/folds/{y}/train_PRIVATE.npz');assert not set(tr['groups'])&set(labs[y]['groups'])
        for pool_count in COUNTS:
            for member in ['ensemble',*map(str,SEEDS)]:
                pp=prediction(book,pool_count,y,member)
                np.testing.assert_array_equal(pp['sample_ids'],labs[y]['sample_ids']);np.testing.assert_array_equal(pp['times'],TIMES)
    rows=[];intervals=[];audit=[]
    for h in [12,24]:
        for k in [1,2]:
            for scope in [*YEARS,'pooled']:
                ys=[scope] if scope!='pooled' else list(YEARS[:2] if h==24 else YEARS)
                ok=not(scope==2025 and h==24);labels=[];years=[];ids=[]
                for y in ys:
                    yy,w,known,supp,_=evaluation_context(labs[y],h,k);assert supp==ok
                    if ok:assert known.all() and (w==1).all()
                    labels.extend(yy);years.extend([y]*len(yy));ids.extend(labs[y]['sample_ids'])
                labels,years,ids=np.array(labels,int),np.array(years),np.array(ids);count=np.column_stack([counts[y] for y in ys])
                assert len(set(ids))==len(ids)
                baseline=None
                for pool_count in [REFERENCE,*[s for s in COUNTS if s!=REFERENCE],'historical_selected']:
                    for member in (['ensemble'] if pool_count=='historical_selected' else ['ensemble',*map(str,SEEDS)]):
                        prob=np.concatenate([endpoint(prediction(book,selected[y] if pool_count=='historical_selected' else pool_count,y,member)['cif'],labs[y][f'h{h}'])[:,k-1] for y in ys])
                        common=dict(pool_count=pool_count,scope=scope,months=h,cause=k,member=member,companies=len(ids),supported=ok,
                            events=int(labels.sum()) if ok else np.nan,checked=sum(int(np.ceil(.2*len(labs[y]['X']))) for y in ys))
                        if not ok:
                            rows.append(dict(**common,**dict.fromkeys(FIELDS,np.nan),mean_prediction_only=float(prob.mean()),reason='INCOMPLETE_2025_H24'));continue
                        vals=assessment(count if member=='ensemble' else count[:1],labels,prob,years,ids)
                        point=dict(zip(FIELDS,vals[0]));rows.append(dict(**common,**point,reason='NO_EVENTS_AUC_AP_UNDEFINED' if not labels.sum() else ''))
                        if scope!='pooled':
                            ref=metrics(labs[scope],prob,h,k)
                            np.testing.assert_allclose([point[f] for f in ['auc','ap','ipcw_brier','mean_predicted','observed','top20_captured']],
                                [ref[f] for f in ['auc','ap','brier','risk_mean','observed','captured']],rtol=0,atol=1e-12,equal_nan=True)
                        if member!='ensemble':continue
                        if len(set(labels))==2:
                            np.testing.assert_allclose([point['auc'],point['ap']],[roc_auc_score(labels,prob),average_precision_score(labels,prob)],rtol=0,atol=1e-12)
                        capture=sum(labels[sorted(np.flatnonzero(years==y),key=lambda i:(-prob[i],ids[i]))[:int(np.ceil(.2*sum(years==y)))]].sum() for y in ys)
                        assert capture==point['top20_captured']
                        num=0.;den=0
                        for y in ys:
                            ii=years==y;pairs=int(labels[ii].sum()*(1-labels[ii]).sum())
                            if pairs:num+=paired_auc(labels[ii],prob[ii])[0]*pairs;den+=pairs
                        if den:assert abs(num/den-point['within_year_auc'])<1e-12
                        audit.append(dict(**common,independent_metrics=True))
                        if pool_count==REFERENCE:baseline=vals.copy()
                        delta=vals-baseline;lo,hi=qtile(delta[1:])
                        for j,f in enumerate(FIELDS):
                            intervals.append(dict(**common,reference_count=REFERENCE,metric=f,difference=delta[0,j],lower95=lo[j],upper95=hi[j],
                                undefined=int((~np.isfinite(delta[1:,j])).sum()),repeats=1000))
    df=pd.DataFrame(rows);df.to_csv(D/'METRICS_ALL.csv',index=False)
    df[df.member=='ensemble'].to_csv(D/'METRICS_ENSEMBLE.csv',index=False)
    pd.DataFrame(intervals).to_csv(D/'PAIRED_DIFFERENCES_95.csv',index=False)
    pd.DataFrame(audit).to_csv(D/'INDEPENDENT_METRIC_CHECKS.csv',index=False)
    # Refitted pool25 estimates must also reproduce the previous public evaluator.
    old=pd.read_csv(D.parent/'post_st_training_start_v1/METRICS_ENSEMBLE.csv')
    old=old[old.earliest_entry_year==2012]
    for _,r in old.iterrows():
        rr=df[(df.pool_count==REFERENCE)&(df.scope.astype(str)==str(r.scope))&(df.months==r.months)&(df.cause==r.cause)&(df.member=='ensemble')]
        assert len(rr)==1
        np.testing.assert_allclose(rr[FIELDS].to_numpy(float)[0],r[FIELDS].to_numpy(float),atol=1e-12,rtol=0,equal_nan=True)
    write(D/'METRIC_VALIDATION.json',dict(drawfile_sha256=sha(drawfile),independent_checks=len(audit),
        selection_lock_sha256=sha(run()/'selection_locked_PRIVATE.json'),pool25_prior_metrics_reproduced=True,
        label_hashes={str(y):sha(source()/f'input/evaluation_input/{y}_test_PRIVATE.npz') for y in YEARS},supported_outer_cohorts_have_unit_IPCW=True))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['history','outer']);args=parser.parse_args()
    b=registry()
    if args.stage=='history':historical_selection(b)
    if args.stage=='outer':numerical(b);outer_evaluation(b)
    assert sha(ROOT/'CURRENT_POST_ST_MODEL.json')==load(D/'DATA_AUDIT.json')['current_pointer_sha256']

if __name__=='__main__':main()
