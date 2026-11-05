"""All fixed paired perturbations, complete-case endpoint evaluator and stability."""
from itertools import combinations,product
import argparse,time,warnings
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score,average_precision_score
from experiments.post_st_benchmark_calibration_v1.bootstrap import assessment,FIELDS
from experiments.post_st_dropcolumn_selection_v1.common import evaluation_context
from .common import *

def register():
    inp=run()/'input'
    for key,value in load(run()/'snapshot.json')['input_sha256'].items():assert sha(inp/key)==value
    rows=[];models=[];num=[];cost=[];support=[];paths={};seen=set()
    for p in sorted((run()/'published').glob('*/task-*/output')):
        s=load(p/'summary.json');key=(s['year'],s['repeat'])
        assert s['status']=='COMPLETE' and key not in seen and s['forest_fits']==3
        seen.add(key);paths[key]=p
        for name,h in load(p/'hashes.json').items():assert sha(p/name)==h
        original=arrays(inp/f"folds/{s['year']}/train_PRIVATE.npz")
        draw=arrays(p/'draw_PRIVATE.npz')['indices']
        np.testing.assert_array_equal(draw,draw_indices(len(original['X']),*key))
        assert s['source_train_sha256']==sha(inp/f"folds/{s['year']}/train_PRIVATE.npz")
        cost.append({k:v for k,v in s.items() if k not in ['members','ensembles']})
        support.append({k:s[k] for k in ['year','repeat','outer_rows','unique_training_companies','distinct_termination','distinct_recovery','weighted_termination','weighted_recovery']})
        for m in s['members']:
            ident=dict(year=key[0],repeat=key[1],seed=m['seed'],strength=m['strength'])
            rows.append(dict(**ident,fit_seconds=m['fit_seconds'],shrink_seconds=m['shrink_seconds'],
               mean_unique_leaf_companies=m['mean_unique_leaf_companies'],
               singleton_target_leaves=m['singleton_target_leaves'],**m['independent']))
            num.append(dict(**ident,**m['checks']))
            folder=p/str(m['seed'])/str(m['strength'])
            models.append(dict(**ident,model_sha256=sha(folder/'forest_PRIVATE.json'),functions_sha256=sha(folder/'functions_PRIVATE.npz')))
        for m in s['ensembles']:
            num.append(dict(year=key[0],repeat=key[1],seed='ensemble',strength=m['strength'],**m['checks'],ensemble_mean_error=m['mean_error']))
            folder=p/f"ensemble_{m['strength']}"
            models.append(dict(year=key[0],repeat=key[1],seed='ensemble',strength=m['strength'],
                model_sha256=sha(folder/'forest_PRIVATE.json'),functions_sha256=sha(folder/'functions_PRIVATE.npz')))
    assert seen==set(product(YEARS,range(1,REPEATS+1))), (len(seen),150)
    for name,data in [('COMPUTE_COST',cost),('TRAINING_SUPPORT',support),('MEMBER_DIAGNOSTICS',rows),('NUMERICAL_VALIDATION',num),('MODEL_HASHES',models)]:
        pd.DataFrame(data).to_csv(D/f'{name}.csv',index=False)
    registry=[dict(year=y,repeat=r,path=str(p)) for (y,r),p in sorted(paths.items())]
    rp=run()/'registry_PRIVATE.json'
    if rp.exists():assert load(rp)==registry, 'existing registry differs; refusing overwrite'
    else:write(rp,registry)
    print('Verified 150 paired tasks, 450 fits, 1800 complete representations',flush=True)
    return paths

def score_vectors(y,p,years,ids):
    ans=dict(zip(FIELDS,assessment(np.ones((1,len(p)),int),y,p,years,ids)[0]))
    a=roc_auc_score(y,p) if 0<y.sum()<len(y) else np.nan
    ap=average_precision_score(y,p) if y.any() else np.nan
    captures=sum(y[ix[top_indices(p[ix],ids[ix])]].sum() for v in np.unique(years) for ix in [np.flatnonzero(years==v)])
    np.testing.assert_allclose([ans['auc'],ans['ap'],ans['ipcw_brier'],ans['top20_captured']],
        [a,ap,np.mean((p-y)**2),captures],atol=1e-12,rtol=0,equal_nan=True)
    return ans

def metrics_and_predictions(paths):
    src=Path(load(D.parent/'post_st_seed_explanation_stability_v1/run_location.json')['private_run'])
    labs={y:arrays(src/f'input/evaluation_input/{y}_test_PRIVATE.npz') for y in YEARS}
    pp={};records=[];validations=[]
    for y,r,s in product(YEARS,range(1,REPEATS+1),STRENGTHS):
        z=arrays(paths[y,r]/f'ensemble_{s}/mean_predictions_PRIVATE.npz')
        np.testing.assert_array_equal(z['sample_ids'],labs[y]['sample_ids'])
        np.testing.assert_array_equal(z['times'],TIMES)
        for h in [12,24]:pp[y,r,s,h]=endpoint(z['cif'],labs[y][f'h{h}'])
    for scope,h,k,r,s in product([*YEARS,'pooled'],[12,24],[1,2],range(1,REPEATS+1),STRENGTHS):
        ys=(list(YEARS) if h==12 else [2023,2024]) if scope=='pooled' else [scope]
        base=dict(scope=str(scope),months=h,cause=k,repeat=r,strength=s,
           companies=sum(len(labs[y]['X']) for y in ys),years_included=','.join(map(str,ys)))
        if scope==2025 and h==24:
            records.append(dict(**base,supported=False,reason='INSUFFICIENT_24_MONTH_FOLLOWUP',
                **{v:np.nan for v in FIELDS},events=np.nan));continue
        targets=[]
        for y in ys:
            yy,w,known,supp,_=evaluation_context(labs[y],h,k)
            assert supp and known.all() and np.all(w==1)
            targets.append(yy)
        yy=np.concatenate(targets);p=np.concatenate([pp[y,r,s,h][:,k-1] for y in ys])
        ids=np.concatenate([labs[y]['sample_ids'] for y in ys])
        years=np.concatenate([np.full(len(labs[y]['X']),y) for y in ys])
        m=score_vectors(yy,p,years,ids)
        records.append(dict(**base,supported=True,reason='',events=int(yy.sum()),**m))
    df=pd.DataFrame(records);df.to_csv(D/'METRICS_ALL.csv',index=False)
    summaries=[];delta=[]
    for key,g in df.groupby(['scope','months','cause','strength'],sort=False):
        for field in FIELDS:
            x=g[field].to_numpy(float);v=x[np.isfinite(x)]
            summaries.append(dict(zip(['scope','months','cause','strength'],key),metric=field,valid=len(v),
                undefined=len(x)-len(v),mean=np.mean(v) if len(v) else np.nan,median=np.median(v) if len(v) else np.nan,
                p05=np.quantile(v,.05) if len(v) else np.nan,p95=np.quantile(v,.95) if len(v) else np.nan,
                minimum=np.min(v) if len(v) else np.nan,maximum=np.max(v) if len(v) else np.nan))
    for (scope,h,k),g in df.groupby(['scope','months','cause'],sort=False):
        for a,b in [(10,0),(18,0),(18,10)]:
            left=g[g.strength==a].sort_values('repeat');right=g[g.strength==b].sort_values('repeat')
            np.testing.assert_array_equal(left['repeat'],right['repeat'])
            for field in FIELDS:
                x=left[field].to_numpy()-right[field].to_numpy();v=x[np.isfinite(x)]
                delta.append(dict(scope=scope,months=h,cause=k,contrast=f'{a}_minus_{b}',metric=field,
                  mean=np.mean(v) if len(v) else np.nan,median=np.median(v) if len(v) else np.nan,
                  p05=np.quantile(v,.05) if len(v) else np.nan,p95=np.quantile(v,.95) if len(v) else np.nan,
                  positive=int((v>1e-12).sum()),negative=int((v< -1e-12).sum()),ties=int((abs(v)<=1e-12).sum()),
                  valid=len(v),undefined=len(x)-len(v)))
    pd.DataFrame(summaries).to_csv(D/'METRIC_DISTRIBUTIONS.csv',index=False)
    pd.DataFrame(delta).to_csv(D/'PAIRED_PERFORMANCE_DIFFERENCES.csv',index=False)
    # Repeated full predictors, not constituent32-tree model agreement.
    stability=[];local=[]
    for y,s,h,k in product(YEARS,STRENGTHS,[12,24],[1,2]):
        p=np.stack([pp[y,r,s,h][:,k-1] for r in range(1,REPEATS+1)])
        stat=pair_stats(p);ids=labs[y]['sample_ids']
        masks=np.zeros(p.shape,bool)
        for r,v in enumerate(p):masks[r,top_indices(v,ids)]=True
        jac=[]
        for a,b in combinations(range(REPEATS),2):
            jac.append(np.logical_and(masks[a],masks[b]).sum()/np.logical_or(masks[a],masks[b]).sum())
        stat.update(top20_jaccard_mean=float(np.mean(jac)),top20_jaccard_median=float(np.median(jac)),
          companies_with_selection_frequency_between_20_and_80=int(((masks.mean(0)>.2)&(masks.mean(0)<.8)).sum()),
          normalized_prediction_sd=stat['mean_company_sd']/stat['between_company_sd'] if stat['between_company_sd']>1e-14 else np.nan)
        stability.append(dict(year=y,strength=s,months=h,cause=k,performance_supported=supported(y,h),**stat))
        np.savez_compressed(run()/f'probability_{y}_{s}_{h}_{k}_PRIVATE.npz',prediction=p,top20=masks)
    pd.DataFrame(stability).to_csv(D/'PREDICTION_STABILITY.csv',index=False)
    write(D/'METRIC_VALIDATION.json',dict(independent_sklearn_AP_AUC_Brier_Top20_checks=int(df.supported.sum()),
      all_supported_IPCW_one=True,label_sha256={str(y):sha(src/f'input/evaluation_input/{y}_test_PRIVATE.npz') for y in YEARS},
      no_metrics_for_2025H24=True,interval_scope='50 paired training perturbations; P5/P95 are descriptive, not confidence intervals'))
    print('Metrics and fixed-company stability complete',flush=True)

def function_summary(paths,year):
    """Common-reference term shapes; omitted terms are zero for magnitude variability."""
    from scipy.stats import rankdata
    results=[];localrows=[];complexity=[]
    rng=np.random.default_rng(2026110502+year)
    pairs=list(combinations(range(REPEATS),2));ai=np.array([a for a,b in pairs]);bi=np.array([b for a,b in pairs])
    for s in STRENGTHS:
        data=[arrays(paths[year,r]/f'ensemble_{s}/components_PRIVATE.npz') for r in range(1,REPEATS+1)]
        loc=[arrays(paths[year,r]/f'ensemble_{s}/local_components_PRIVATE.npz') for r in range(1,REPEATS+1)]
        keys=sorted(set().union(*[set(z['keys']) for z in data]))
        mapping={k:i for i,k in enumerate(keys)};orders=np.array([k.count(',')+1 for k in keys])
        for z in data:assert len(z['terms'])==len(data[0]['terms'])
        # Each h/c slice stays modest; only one is retained in memory.
        for hi,day in [(0,365),(2,730)]:
            for cause in [1,2]:
                a=np.zeros((REPEATS,len(data[0]['terms']),len(keys)))
                ll=np.zeros((REPEATS,len(loc[0]['terms']),len(keys)))
                for r,z in enumerate(data):
                    ix=[mapping[k] for k in z['keys']]
                    a[r][:,ix]=z['terms'][:,:,hi,cause-1]
                    ll[r][:,ix]=loc[r]['terms'][:,:,hi,cause-1]
                for order in [1,2]:
                    selected=np.flatnonzero(orders==order);v=a[:,:,selected]
                    amp=np.mean(abs(v),axis=1);presence=amp>1e-14
                    pair_corr=[];relative=[];directions=[]
                    for j in range(len(selected)):
                        x=v[:,:,j];center=x-x.mean(1,keepdims=True)
                        norms=np.linalg.norm(center,axis=1)
                        valid=(norms[ai]>1e-14)&(norms[bi]>1e-14)
                        gram=center@center.T
                        den=norms[ai]*norms[bi]
                        corr=np.divide(gram[ai,bi],den,out=np.full(len(pairs),np.nan),where=valid)
                        pair_corr.extend(corr[valid].tolist())
                        rms=np.sqrt(np.mean(x*x,axis=1))
                        mean_product=(x@x.T)/x.shape[1]
                        # Roundoff max is only for squared diagnostic distances, NEVER predictions.
                        distance=np.sqrt(np.maximum(0,rms[ai]**2+rms[bi]**2-2*mean_product[ai,bi]))
                        norm=(rms[ai]+rms[bi])/2;nonzero=norm>1e-14
                        relative.extend((distance[nonzero]/norm[nonzero]).tolist())
                        active=(abs(x)>=.001).astype(float)
                        positive=((x>=.001)).astype(float);negative=((x<=-.001)).astype(float)
                        count=(active@active.T)[ai,bi]
                        equal=(positive@positive.T+negative@negative.T)[ai,bi]
                        directions.extend((equal[count>0]/count[count>0]).tolist())
                    jac=[];topjac=[]
                    top=[]
                    for r in range(REPEATS):
                        take=np.argsort(-amp[r],kind='stable')[:10]
                        top.append(set(take[amp[r,take]>=.001]))
                    for ra,rb in pairs:
                        union=(presence[ra]|presence[rb]).sum()
                        jac.append((presence[ra]&presence[rb]).sum()/union if union else np.nan)
                        un=top[ra]|top[rb]
                        topjac.append(len(top[ra]&top[rb])/len(un) if un else np.nan)
                    def avg(x):return float(np.nanmean(x)) if len(x) and np.isfinite(x).any() else np.nan
                    # Magnitudes retain absent functions as zeros, rather than only common terms.
                    sd=np.std(v,axis=0,ddof=1);amplitude=float(amp.sum(1).mean())
                    variation=float(sd.sum(1).mean())
                    results.append(dict(year=year,strength=s,day=day,cause=cause,order=order,
                      reference_companies=v.shape[1],union_functions=len(selected),
                      mean_selected_functions=float(presence.sum(1).mean()),
                      functions_present_at_least_80_percent=int((presence.mean(0)>=.8).sum()),
                      functions_material_at_least_80_percent=int(((amp>=.001).mean(0)>=.8).sum()),
                      mean_absolute_contribution_sum=amplitude,mean_component_sd_sum=variation,
                      relative_component_sd=variation/amplitude if amplitude>1e-14 else np.nan,
                      mean_common_shape_correlation=avg(pair_corr),valid_shape_comparisons=len(pair_corr),
                      undefined_shape_comparisons=len(selected)*len(pairs)-len(pair_corr),
                      mean_normalized_function_difference=avg(relative),
                      material_direction_agreement=avg(directions),valid_direction_comparisons=len(directions),
                      mean_set_jaccard=avg(jac),mean_material_top10_jaccard=avg(topjac)))
                    lv=ll[:,:,selected];lsd=float(lv.std(0,ddof=1).sum(1).mean())
                    la=float(abs(lv).sum(2).mean())
                    localrows.append(dict(year=year,strength=s,day=day,cause=cause,order=order,
                      local_mean_component_sd_sum=lsd,local_mean_abs_contribution_sum=la,
                      local_relative_component_sd=lsd/la if la>1e-14 else np.nan))
                del a,ll
        for r in range(1,REPEATS+1):
            z=load(paths[year,r]/f'ensemble_{s}/verification.json')
            complexity.append(dict(year=year,repeat=r,strength=s,**{k:z[k] for k in ['main_functions','interaction_functions','spatial_table_cells','scalar_table_cells','function_bytes','compile_seconds','verification_seconds']}))
    out=run()/f'function_analysis_{year}';out.mkdir(exist_ok=False)
    for name,rr in [('FUNCTION_STABILITY',results),('LOCAL_EXPLANATION_STABILITY',localrows),('COMPLEXITY',complexity)]:
        pd.DataFrame(rr).to_csv(out/f'{name}.csv',index=False)
    print(f'Functions {year} complete',flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['metrics','functions','collect']);ap.add_argument('--year',type=int);a=ap.parse_args()
    if a.mode=='metrics':
        paths=register();metrics_and_predictions(paths)
    else:
        rr=load(run()/'registry_PRIVATE.json');paths={(r['year'],r['repeat']):Path(r['path']) for r in rr}
        if a.mode=='functions':function_summary(paths,a.year)
        else:
            for name in ['FUNCTION_STABILITY','LOCAL_EXPLANATION_STABILITY','COMPLEXITY']:
                pd.concat([pd.read_csv(run()/f'function_analysis_{y}/{name}.csv') for y in YEARS]).to_csv(D/f'{name}.csv',index=False)
            cc=pd.read_csv(D/'COMPLEXITY.csv')
            for k,g in cc.groupby(['year','repeat']):
                assert g.spatial_table_cells.nunique()==1
            print('Function tables collected')
if __name__=='__main__':main()
