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
    reused=load(run()/'reused_outputs_PRIVATE.json')
    paths=[(Path(row['path']),True,row) for row in reused]
    paths += [(p,False,None) for p in (run()/'published').glob('*/task-*/output')]
    assert sum(not reused for _,reused,_ in paths)==90 and len(paths)==126
    for p,reused,record in paths:
        s=load(p/'summary.json');assert s['status']=='COMPLETE' and s['mode']=='derive' and s['forest_fits']==0
        if reused:assert sha(p/'summary.json')==record['summary_sha256'] and s['strength'] in REUSED
        else:assert s['strength'] in NEW
        verify_output(p)
        key=(s['strength'],s['year']);assert key not in book;book[key]=(p,s)
        cost.append(dict(**{k:v for k,v in s.items() if k not in ['members','ensemble_check']},
            reused=reused,current_run_compute_seconds=0 if reused else s['seconds']))
        assert s['source_train_sha256']==sha(run()/f"input/folds/{s['year']}/train_PRIVATE.npz")
        for member in [*map(str,SEEDS),'ensemble']:
            for name in ['forest_PRIVATE.json','functions_PRIVATE.npz','predictions_PRIVATE.npz']:
                pp=p/member/name;hashes.append(dict(strength=key[0],year=key[1],member=member,artifact=name,
                    reused=reused,sha256=sha(pp),bytes=pp.stat().st_size))
    assert set(book)=={(s,y) for s in STRENGTHS for y in ALL_YEARS}
    pd.DataFrame(cost).to_csv(D/'COMPUTE_COST.csv',index=False)
    pd.DataFrame(hashes).to_csv(D/'ARTIFACT_HASHES.csv',index=False)
    return book

def prediction(book,strength,year,member='ensemble'):
    p,_=book[strength,year]
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
                    for strength in STRENGTHS:
                        ps=[]
                        for y in ys:
                            z=prediction(book,strength,y)
                            np.testing.assert_array_equal(z['sample_ids'],labels[y]['sample_ids'])
                            ps.append(endpoint(z['cif'],labels[y][f'h{h}'])[:,k-1])
                        p=np.concatenate(ps)
                        point=dict(zip(FIELDS,assessment(np.ones((1,len(ids)),int),yvalues,p,years,ids)[0]))
                        rows.append(dict(outer_cutoff=outer,scope=scope,months=h,cause=k,strength=strength,
                            companies=len(ids),events=int(yvalues.sum()),**point))
        candidates=[r for r in rows if r['outer_cutoff']==outer and r['scope']=='pooled' and r['months']==12 and r['cause']==1]
        value,reason=select_strength(candidates)
        chosen=next(r for r in candidates if r['strength']==value)
        selected.append(dict(outer_cutoff=outer,selected_strength=value,reason=reason,
            historical_termination_events=chosen['events'],historical_companies=chosen['companies'],
            within_year_auc=chosen['within_year_auc'],ap=chosen['ap'],ipcw_brier=chosen['ipcw_brier']))
    pd.DataFrame(rows).to_csv(D/'HISTORICAL_VALIDATION_METRICS.csv',index=False)
    pd.DataFrame(selected).to_csv(D/'HISTORICAL_SELECTED_STRENGTHS.csv',index=False)
    write(run()/'selection_locked_PRIVATE.json',dict(selected=selected,outer_labels_read=False,
        protocol_sha256=sha(D/'FROZEN_PROTOCOL.json'),history_metrics_sha256=sha(D/'HISTORICAL_VALIDATION_METRICS.csv'),
        historical_label_hashes={str(p.relative_to(run())):sha(p) for p in (run()/'selection_input').rglob('*.npz')}))
    print(pd.DataFrame(selected).to_string(index=False))

def numerical(book):
    checks=[];replay=[];amplitude=[];stability=[];seed_variation=[];repro=[]
    for (strength,year),(p,s) in book.items():
        for m in s['members']:
            ident=dict(strength=strength,year=year,member=str(m['seed']))
            checks.append(dict(**ident,**m['checks']))
            replay.append(dict(**ident,**{k:v for k,v in m.items() if k not in ['seed','checks','reproduction']}))
            if m['reproduction']:repro.append(dict(**ident,**m['reproduction']))
        checks.append(dict(strength=strength,year=year,member='ensemble',**s['ensemble_check'],ensemble_mean_error=s['ensemble_mean_error']))
        comps={str(seed):arrays(p/str(seed)/'components_PRIVATE.npz') for seed in SEEDS}
        comps['ensemble']=arrays(p/'ensemble/components_PRIVATE.npz')
        for member,z in comps.items():
            keys=list(z['keys'])
            for order in [1,2]:
                indices=[i for i,key in enumerate(keys) if key.count(',')+1==order]
                for hi,day in [(0,365),(2,730)]:
                    for k in [1,2]:
                        terms=z['terms'][:,indices,hi,k-1]
                        amplitude.append(dict(strength=strength,year=year,member=member,order=order,day=day,cause=k,
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
                        stability.append(dict(strength=strength,year=year,seed_a=a,seed_b=b,order=order,day=day,cause=k,
                            jaccard=len(cc)/len(aa|bb) if aa|bb else 1.,common_functions=len(cc),
                            estimable_correlations=len(cor),mean_correlation=np.mean(cor) if cor else np.nan,
                            mean_abs_function_difference=np.mean(diff) if diff else np.nan))
        pp=np.stack([prediction(book,strength,year,str(s))['cif'] for s in SEEDS])
        for hi,day in [(0,365),(2,730)]:
            for k in [1,2]:
                sd=pp[:,:,hi,k-1].std(axis=0,ddof=1)
                seed_variation.append(dict(strength=strength,year=year,day=day,cause=k,mean_seed_sd=sd.mean(),max_seed_sd=sd.max()))
    for name,rows in [('NUMERICAL_COMPLEXITY',checks),('INDEPENDENT_SHRINKAGE',replay),('FUNCTION_AMPLITUDE',amplitude),
        ('FUNCTION_STABILITY',stability),('PREDICTION_STABILITY',seed_variation),('REFERENCE_REPRODUCTION',repro)]:
        pd.DataFrame(rows).to_csv(D/f'{name}.csv',index=False)
    assert len(repro)==9

def outer_evaluation(book):
    locked=load(run()/'selection_locked_PRIVATE.json')
    assert not locked['outer_labels_read'] and locked['protocol_sha256']==sha(D/'FROZEN_PROTOCOL.json')
    assert locked['history_metrics_sha256']==sha(D/'HISTORICAL_VALIDATION_METRICS.csv')
    selected={r['outer_cutoff']:r['selected_strength'] for r in locked['selected']}
    labs={y:arrays(source()/f'input/evaluation_input/{y}_test_PRIVATE.npz') for y in YEARS}
    drawfile=Path(load(D.parent/'post_st_cause_support_v1/run_location.json')['private_run'])/'private/bootstrap_indices_PRIVATE.npz'
    draws=arrays(drawfile);rng=np.random.default_rng(2026110201);counts={}
    for y in YEARS:
        n=len(labs[y]['sample_ids']);assert len(set(labs[y]['groups']))==n
        np.testing.assert_array_equal(draws[str(y)],rng.integers(n,size=(1000,n)))
        counts[y]=np.vstack([np.ones(n,int),*[np.bincount(row,minlength=n) for row in draws[str(y)]]])
        tr=arrays(run()/f'input/folds/{y}/train_PRIVATE.npz');assert not set(tr['groups'])&set(labs[y]['groups'])
        for strength in STRENGTHS:
            for member in ['ensemble',*map(str,SEEDS)]:
                pp=prediction(book,strength,y,member)
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
                for strength in [10,*[s for s in STRENGTHS if s!=10],'historical_selected']:
                    for member in (['ensemble'] if strength=='historical_selected' else ['ensemble',*map(str,SEEDS)]):
                        prob=np.concatenate([endpoint(prediction(book,selected[y] if strength=='historical_selected' else strength,y,member)['cif'],labs[y][f'h{h}'])[:,k-1] for y in ys])
                        common=dict(strength=strength,scope=scope,months=h,cause=k,member=member,companies=len(ids),supported=ok,
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
                        if strength==10:baseline=vals.copy()
                        delta=vals-baseline;lo,hi=qtile(delta[1:])
                        for j,f in enumerate(FIELDS):
                            intervals.append(dict(**common,reference_strength=10,metric=f,difference=delta[0,j],lower95=lo[j],upper95=hi[j],
                                undefined=int((~np.isfinite(delta[1:,j])).sum()),repeats=1000))
    df=pd.DataFrame(rows);df.to_csv(D/'METRICS_ALL.csv',index=False)
    df[df.member=='ensemble'].to_csv(D/'METRICS_ENSEMBLE.csv',index=False)
    pd.DataFrame(intervals).to_csv(D/'PAIRED_DIFFERENCES_95.csv',index=False)
    pd.DataFrame(audit).to_csv(D/'INDEPENDENT_METRIC_CHECKS.csv',index=False)
    # New lambda10 estimates must also reproduce the previous public evaluator.
    old=pd.read_csv(D.parent/'post_st_training_start_v1/METRICS_ENSEMBLE.csv')
    old=old[old.earliest_entry_year==2012]
    for _,r in old.iterrows():
        rr=df[(df.strength==10)&(df.scope.astype(str)==str(r.scope))&(df.months==r.months)&(df.cause==r.cause)&(df.member=='ensemble')]
        assert len(rr)==1
        np.testing.assert_allclose(rr[FIELDS].to_numpy(float)[0],r[FIELDS].to_numpy(float),atol=1e-12,rtol=0,equal_nan=True)
    coarse=pd.read_csv(D.parent/'post_st_shrinkage_strength_v1/METRICS_ENSEMBLE.csv')
    coarse=coarse[coarse.strength.isin(list(map(str,REUSED)))]
    coarse_checks=0
    for _,r in coarse.iterrows():
        rr=df[(df.strength==int(r.strength))&(df.scope.astype(str)==str(r.scope))&(df.months==r.months)&(df.cause==r.cause)&(df.member=='ensemble')]
        assert len(rr)==1
        np.testing.assert_allclose(rr[FIELDS].to_numpy(float)[0],r[FIELDS].to_numpy(float),atol=1e-12,rtol=0,equal_nan=True)
        coarse_checks+=1
    write(D/'METRIC_VALIDATION.json',dict(drawfile_sha256=sha(drawfile),independent_checks=len(audit),coarse_grid_metric_rows_reproduced=coarse_checks,
        selection_lock_sha256=sha(run()/'selection_locked_PRIVATE.json'),lambda10_prior_metrics_reproduced=True,
        label_hashes={str(y):sha(source()/f'input/evaluation_input/{y}_test_PRIVATE.npz') for y in YEARS},supported_outer_cohorts_have_unit_IPCW=True))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['history','outer','all'],default='all',nargs='?');args=parser.parse_args()
    b=registry()
    if args.stage in ['history','all']:historical_selection(b)
    if args.stage in ['outer','all']:numerical(b);outer_evaluation(b)
    assert sha(ROOT/'CURRENT_POST_ST_MODEL.json')==load(D/'DATA_AUDIT.json')['current_pointer_sha256']

if __name__=='__main__':main()
