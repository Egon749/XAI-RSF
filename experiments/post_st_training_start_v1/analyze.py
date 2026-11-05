"""Read verified saved forecasts once all windows finish; aggregate only."""
from itertools import combinations
import warnings
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
    paths=[run()/'pilot']+list((run()/'published').glob('*/task-*/output'))
    book={};public=[];hash_records=[]
    for p in paths:
        s=load(p/'summary.json');assert s['status']=='COMPLETE' and s['forest_fits']==3
        key=(s['earliest_entry_year'],s['year']);assert key not in book
        for name,digest in load(p/'hashes.json').items():assert sha(p/name)==digest
        if p.name!='pilot':
            receipt=load(p.parent/'complete.json')
            # Pool-managed publication is checked independently below as well.
            for item in receipt['manifest']:
                if len(item)==2:
                    assert item[1]=='directory' and (p/item[0]).is_dir()
                else:
                    name,size,digest=item
                    assert (p/name).stat().st_size==size and sha(p/name)==digest
        else:assert load(p/'recovery_execution.json')['forest_fits_additional']==0
        book[key]=(p,s)
        public.append({k:v for k,v in s.items() if k not in ['members','ensemble_check']})
        for seed in SEEDS:
            for v in ('raw','shrunk'):
                for name in ['forest_PRIVATE.json','functions_PRIVATE.npz','predictions_PRIVATE.npz']:
                    pp=p/str(seed)/v/name
                    hash_records.append(dict(earliest_entry_year=key[0],year=key[1],seed=seed,version=v,artifact=name,sha256=sha(pp),bytes=pp.stat().st_size))
        for name in ['forest_PRIVATE.json','functions_PRIVATE.npz','predictions_PRIVATE.npz']:
            pp=p/'ensemble'/name;hash_records.append(dict(earliest_entry_year=key[0],year=key[1],seed='ensemble',version='shrunk',artifact=name,sha256=sha(pp),bytes=pp.stat().st_size))
    assert set(book)=={(s,y) for s in STARTS for y in YEARS}
    pd.DataFrame(public).to_csv(D/'COMPUTE_COST.csv',index=False)
    pd.DataFrame(hash_records).to_csv(D/'ARTIFACT_HASHES.csv',index=False)
    return book

def complexity(book):
    checks=[];supports=[];costs=[];stability=[];reproduction=[];prediction_var=[]
    for (start,year),(p,s) in book.items():
        for m in s['members']:
            ident=dict(earliest_entry_year=start,year=year,seed=m['seed'])
            for v,c in m['checks'].items():checks.append(dict(**ident,version=v,**c))
            supports.append(dict(**ident,**{k:m[k] for k in ['leaf_count','zero_target_leaves','borrowed_target_leaves','median_unique_leaf_companies','minimum_unique_leaf_companies','scored_candidates','guard_rejected_candidates']},**m['independent_checks']))
            costs.append(dict(**ident,fit_seconds=m['fit_seconds'],shrink_seconds=m['shrink_seconds'],replay_seconds=m['replay_seconds']))
            if m['reproduction']:reproduction.append(dict(**ident,**m['reproduction']))
        checks.append(dict(earliest_entry_year=start,year=year,seed='ensemble',version='shrunk',**s['ensemble_check'],ensemble_mean_error=s['ensemble_mean_error']))
        for v in ('raw','shrunk'):
            comp={seed:arrays(p/f'{seed}/{v}/components_PRIVATE.npz') for seed in SEEDS}
            for a,b in combinations(SEEDS,2):
                ka=list(comp[a]['keys']);kb=list(comp[b]['keys'])
                for order in (1,2):
                    aa={x for x in ka if x.count(',')+1==order};bb={x for x in kb if x.count(',')+1==order}
                    cc=sorted(aa&bb);common=sorted(aa|bb)
                    for hi,day in [(0,365),(2,730)]:
                        for k in (1,2):
                            corr=[];diff=[]
                            for key in cc:
                                x=comp[a]['terms'][:,ka.index(key),hi,k-1];z=comp[b]['terms'][:,kb.index(key),hi,k-1]
                                diff.append(float(abs(x-z).mean()))
                                if np.std(x)>1e-14 and np.std(z)>1e-14:corr.append(float(np.corrcoef(x,z)[0,1]))
                            stability.append(dict(earliest_entry_year=start,year=year,version=v,seed_a=a,seed_b=b,order=order,day=day,cause=k,
                                jaccard=len(cc)/len(common) if common else 1.,common_functions=len(cc),union_functions=len(common),
                                estimable_correlations=len(corr),mean_correlation=np.mean(corr) if corr else np.nan,
                                mean_abs_function_difference=np.mean(diff) if diff else np.nan))
            pp=np.stack([arrays(p/f'{seed}/{v}/predictions_PRIVATE.npz')['cif'] for seed in SEEDS])
            for hi,day in [(0,365),(2,730)]:
                for k in (1,2):
                    sd=np.std(pp[:,:,hi,k-1],axis=0,ddof=1)
                    prediction_var.append(dict(earliest_entry_year=start,year=year,version=v,day=day,cause=k,mean_seed_sd=sd.mean(),max_seed_sd=sd.max()))
    pd.DataFrame(checks).to_csv(D/'NUMERICAL_COMPLEXITY.csv',index=False)
    pd.DataFrame(supports).to_csv(D/'LEAF_SUPPORT.csv',index=False)
    pd.DataFrame(costs).to_csv(D/'FIT_COST.csv',index=False)
    pd.DataFrame(stability).to_csv(D/'FUNCTION_STABILITY.csv',index=False)
    pd.DataFrame(reproduction).to_csv(D/'HISTORICAL_REPRODUCTION.csv',index=False)
    pd.DataFrame(prediction_var).to_csv(D/'PREDICTION_STABILITY.csv',index=False)

def evaluate(book):
    labs={y:arrays(source()/f'input/evaluation_input/{y}_test_PRIVATE.npz') for y in YEARS}
    drawfile=Path(load(D.parent/'post_st_cause_support_v1/run_location.json')['private_run'])/'private/bootstrap_indices_PRIVATE.npz'
    draws=arrays(drawfile);rng=np.random.default_rng(2026110201);counts={};books={};audit=[]
    for y in YEARS:
        n=len(labs[y]['sample_ids']);assert len(set(labs[y]['groups']))==n
        np.testing.assert_array_equal(draws[str(y)],rng.integers(n,size=(1000,n)))
        counts[y]=np.vstack([np.ones(n,int),*[np.bincount(row,minlength=n) for row in draws[str(y)]]])
        for start in STARTS:
            path,_=book[start,y]
            tr=arrays(run()/f'input/folds/{y}/{start}/train_PRIVATE.npz')
            assert not set(tr['groups']) & set(labs[y]['groups'])
            for member in ['ensemble',*map(str,SEEDS)]:
                for version in (['shrunk'] if member=='ensemble' else ['raw','shrunk']):
                    pp=arrays(path/'mean_predictions_PRIVATE.npz' if member=='ensemble' else path/f'{member}/{version}/predictions_PRIVATE.npz')
                    np.testing.assert_array_equal(pp['sample_ids'],labs[y]['sample_ids']);np.testing.assert_array_equal(pp['times'],TIMES)
                    books[start,y,member,version]=pp['cif']
    rows=[];differences=[]
    for h in (12,24):
        for k in (1,2):
            for scope in [*YEARS,'pooled']:
                ys=[scope] if scope!='pooled' else list(YEARS[:2] if h==24 else YEARS)
                ok=not (scope==2025 and h==24);labels=[];years=[];ids=[]
                for y in ys:
                    yy,w,known,supp,_=evaluation_context(labs[y],h,k);assert supp==ok
                    if ok:assert known.all() and (w==1).all()
                    labels.extend(yy);years.extend([y]*len(yy));ids.extend(labs[y]['sample_ids'])
                labels,years,ids=np.array(labels,int),np.array(years),np.array(ids)
                assert len(set(ids))==len(ids)
                count=np.column_stack([counts[y] for y in ys]);baseline=None
                for start in STARTS:
                    for member in ['ensemble',*map(str,SEEDS)]:
                        for version in (['shrunk'] if member=='ensemble' else ['raw','shrunk']):
                            prob=np.concatenate([endpoint(books[start,y,member,version],labs[y][f'h{h}'])[:,k-1] for y in ys])
                            common=dict(earliest_entry_year=start,scope=scope,months=h,cause=k,member=member,version=version,
                                companies=len(ids),supported=ok,events=int(labels.sum()) if ok else np.nan,checked=sum(int(np.ceil(.2*len(labs[y]['X']))) for y in ys))
                            if not ok:
                                rows.append(dict(**common,**dict.fromkeys(FIELDS,np.nan),mean_prediction_only=float(prob.mean()),reason='INCOMPLETE_2025_H24'))
                                continue
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
                                ii=years==y;npairs=int(labels[ii].sum()*(1-labels[ii]).sum())
                                if npairs:num+=paired_auc(labels[ii],prob[ii])[0]*npairs;den+=npairs
                            if den:assert abs(num/den-point['within_year_auc'])<1e-12
                            audit.append(dict(**common,independent_metrics=True))
                            if start==2012:baseline=vals.copy()
                            delta=vals-baseline;lo,hi=qtile(delta[1:])
                            for j,f in enumerate(FIELDS):
                                differences.append(dict(**common,reference_start=2012,metric=f,difference=delta[0,j],lower95=lo[j],upper95=hi[j],undefined=int((~np.isfinite(delta[1:,j])).sum()),repeats=1000))
    df=pd.DataFrame(rows);df.to_csv(D/'METRICS_ALL.csv',index=False)
    df[df.member=='ensemble'].to_csv(D/'METRICS_ENSEMBLE.csv',index=False)
    pd.DataFrame(differences).to_csv(D/'PAIRED_DIFFERENCES_95.csv',index=False)
    pd.DataFrame(audit).to_csv(D/'INDEPENDENT_METRIC_CHECKS.csv',index=False)
    write(D/'METRIC_VALIDATION.json',dict(drawfile_sha256=sha(drawfile),checks=len(audit),
        label_hashes={str(y):sha(source()/f'input/evaluation_input/{y}_test_PRIVATE.npz') for y in YEARS},
        supported_outer_cohorts_have_unit_IPCW=True,all_primary_mean_predictions_verified=True))

def main():
    b=registry();complexity(b);evaluate(b)
    assert sha(ROOT/'CURRENT_POST_ST_MODEL.json')==load(D/'DATA_AUDIT.json')['current_pointer_sha256']
    print(json.dumps(dict(complete_window_year_tasks=len(b),actual_forest_fits=sum(s['forest_fits'] for _,s in b.values()),core_and_reference_unchanged=True)))

if __name__=='__main__':main()
