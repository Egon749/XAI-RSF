"""Unified evaluation only after historical feature decisions are sealed."""
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from crforest import CIFExpansion
from experiments.post_st_dropcolumn_selection_v1.common import (
    load, write, arrays, TIMES, SEEDS, metrics, evaluation_context,
    horizon_predictions, bootstrap_differences)
from experiments.post_st_dropcolumn_selection_h24_v1.analyze import component_values
from experiments.post_st_dropcolumn_selection_h24_v1.stability_comparison import compare_variation
from .pipeline import outer_registry
from .search import FULL, LEGACY_REMOVED


def model_specs(run):
    protocol=load(run/'input/protocol.json');parent=Path(protocol['parent_private_run']);previous=Path(protocol['previous_h24_run'])
    registry=outer_registry(parent,previous,run)
    decisions=load(run/'audit/selection_decisions.json');prior=load(previous/'audit/selection_decisions.json')
    result={}
    for y in [2023,2024,2025]:
        candidates={'full83':FULL,'legacy74':tuple(j for j in FULL if j not in LEGACY_REMOVED),
                    'previous_auto':tuple(prior[str(y)]['keep']),
                    'selected_search':tuple(decisions[str(y)]['keep'])}
        for label,keep in candidates.items():
            result[(y,label)]=dict(keep=list(keep),**registry[(y,keep)])
    result[(2023,'historical75')]=load(run/'audit/historical75_model.json')
    return parent,result


def seed_folders(spec):
    p=Path(spec['path'])
    return [p/f'{seed}/shrunk' for seed in SEEDS] if spec['style']=='bundle' else list(map(Path,spec['paths']))


def concatenate(labels,years):
    return {k:np.concatenate([labels[y][k] for y in years]) for k in labels[years[0]]}


def prediction(run):
    parent,specs=model_specs(run)
    labels={y:arrays(parent/f'evaluation_input/{y}_test_PRIVATE.npz') for y in [2023,2024,2025]}
    rows=[];complexity=[];predictions={};calibration=[]
    for (year,label),spec in specs.items():
        p=Path(spec['path']);summary=load(p/'summary.json')
        check=summary['ensemble_check'] if spec['style']=='bundle' else summary['check']
        complexity.append(dict(year=year,model=label,features=len(spec['keep']),
            **{k:check[k] for k in ['main_functions','interaction_functions','spatial_table_cells','scalar_table_cells',
                                   'distinct_thresholds','function_bytes','max_path_variables',
                                   'low_support_cell_fraction','zero_support_cell_fraction']}))
        lab=labels[year]
        folders=[p/'ensemble']+seed_folders(spec)
        pp=[]
        for seed,folder in zip(['ensemble',*map(str,SEEDS)],folders):
            z=arrays(folder/'predictions_PRIVATE.npz');np.testing.assert_array_equal(z['sample_ids'],lab['sample_ids'])
            for h in [12,24]:
                risk=horizon_predictions(z['cif'],TIMES,lab[f'h{h}'])
                for cause in [1,2]:
                    rows.append(dict(year=str(year),model=label,features=len(spec['keep']),seed=seed,horizon=h,cause=cause,
                                     **metrics(lab,risk[:,cause-1],h,cause)))
            pp.append(z['cif'])
        np.testing.assert_allclose(pp[0],np.mean(pp[1:],axis=0),atol=1e-13,rtol=0)
        predictions[(year,label)]=pp[0]
        for h in [12,24]:
            risk=horizon_predictions(pp[0],TIMES,lab[f'h{h}'])
            for cause in [1,2]:
                y,w,known,ok,_=evaluation_context(lab,h,cause)
                order=np.lexsort((lab['sample_ids'],-risk[:,cause-1]))
                for bin_id,ix in enumerate(np.array_split(order,5),1):
                    calibration.append(dict(year=year,model=label,horizon=h,cause=cause,risk_quintile_high_to_low=bin_id,
                        companies=len(ix),supported=ok,predicted_mean=float(risk[ix,cause-1].mean()),
                        observed_IPCW=float(np.sum(w[ix]*y[ix])/len(ix)) if ok else np.nan,
                        observed_targets=int((known[ix]&y[ix]).sum()),unknown=int((~known[ix]).sum())))
    common=['full83','legacy74','previous_auto','selected_search']
    for label in common:
        for h in [12,24]:
            years=[y for y in labels if evaluation_context(labels[y],h,1)[3]]
            lab=concatenate(labels,years)
            risk=np.concatenate([horizon_predictions(predictions[(y,label)],TIMES,labels[y][f'h{h}']) for y in years])
            for cause in [1,2]:
                m=metrics(lab,risk[:,cause-1],h,cause)
                annual=[r for r in rows if r['year'] in list(map(str,years)) and r['model']==label and r['seed']=='ensemble' and r['horizon']==h and r['cause']==cause]
                m['captured']=sum(r['captured'] for r in annual);m['checked']=sum(r['checked'] for r in annual)
                m['coverage']=m['captured']/m['target_events'] if m['target_events'] else np.nan
                rows.append(dict(year='pooled',model=label,features=np.nan,seed='ensemble',horizon=h,cause=cause,
                                 cohorts=','.join(map(str,years)),**m))
    pd.DataFrame(rows).to_csv(run/'reports/MODEL_COMPARISON.csv',index=False)
    pd.DataFrame(complexity).to_csv(run/'reports/EXPLANATION_COMPLEXITY.csv',index=False)
    pd.DataFrame(calibration).to_csv(run/'reports/CALIBRATION_QUINTILES.csv',index=False)
    comparisons=[]
    for yeargroup in [[2023],[2024],[2025],[2023,2024,2025]]:
        for h in [12,24]:
            years=[y for y in yeargroup if evaluation_context(labels[y],h,1)[3]]
            if not years:continue
            lab=concatenate(labels,years);parts=[];offset=0;annual_draws={}
            for y in years:
                n=len(labels[y]['event'])
                draw=np.random.default_rng(np.random.SeedSequence([2026110174,y,h])).integers(n,size=(1000,n))
                annual_draws[y]=draw;parts.append(draw+offset);offset+=n
            draws=np.concatenate(parts,axis=1)
            for reference in ['full83','legacy74','previous_auto']:
                a=np.concatenate([horizon_predictions(predictions[(y,reference)],TIMES,labels[y][f'h{h}']) for y in years])
                b=np.concatenate([horizon_predictions(predictions[(y,'selected_search')],TIMES,labels[y][f'h{h}']) for y in years])
                for cause in [1,2]:
                    differences=bootstrap_differences(lab,a[:,cause-1],b[:,cause-1],h,cause,draws)
                    m1=metrics(lab,a[:,cause-1],h,cause);m2=metrics(lab,b[:,cause-1],h,cause)
                    ca=np.zeros(1000);cb=np.zeros(1000);ct=np.zeros(1000);pointa=pointb=pointt=0
                    for y in years:
                        ll=labels[y];yy,ww,known,ok,_=evaluation_context(ll,h,cause);dr=annual_draws[y]
                        ct+=(yy[dr]&known[dr]).sum(1);pointt+=int((yy&known).sum())
                        for name,total in [(reference,ca),('selected_search',cb)]:
                            risk=horizon_predictions(predictions[(y,name)],TIMES,ll[f'h{h}'])[:,cause-1]
                            # Same sampled companies and annual ceil20% quota for both models.
                            order=np.lexsort((ll['sample_ids'][dr],-risk[dr]),axis=1)[:,:int(np.ceil(.2*len(yy)))]
                            take=np.take_along_axis(dr,order,axis=1)
                            total+=(yy[take]&known[take]).sum(1)
                            point=metrics(ll,risk,h,cause)['captured']
                            if name==reference:pointa+=point
                            else:pointb+=point
                    differences['captured']=cb-ca
                    differences['coverage']=np.divide(cb-ca,ct,out=np.full(1000,np.nan),where=ct>0)
                    pointvalues={k:m2[k]-m1[k] for k in ['auc','ap','brier']}
                    pointvalues.update(captured=pointb-pointa,coverage=(pointb-pointa)/pointt if pointt else np.nan)
                    for metric,value in differences.items():
                        finite=value[np.isfinite(value)]
                        comparisons.append(dict(year=years[0] if len(yeargroup)==1 else 'pooled',cohorts=','.join(map(str,years)),
                            comparison='selected_search-minus-'+reference,horizon=h,cause=cause,metric=metric,
                            difference=pointvalues[metric],lower95=np.quantile(finite,.025) if len(finite) else np.nan,
                            upper95=np.quantile(finite,.975) if len(finite) else np.nan,undefined=1000-len(finite)))
    pd.DataFrame(comparisons).to_csv(run/'reports/PAIRED_DIFFERENCES.csv',index=False)
    features=load(parent/'input/features.json');selected=[]
    for (year,label),spec in specs.items():
        for j,f in enumerate(features):
            selected.append(dict(year=year,model=label,feature_id=j,feature_name=f['feature_name'],label=f['label'],selected=j in spec['keep']))
    pd.DataFrame(selected).to_csv(run/'reports/SELECTED_FEATURES.csv',index=False)
    print(pd.DataFrame(rows).query("seed == 'ensemble' and cause == 1")[['year','model','horizon','auc','ap','brier','captured','target_events']].to_string(index=False),flush=True)


def functions(run):
    parent,specs=model_specs(run);features=load(parent/'input/features.json')
    function_rows=[];seed_rows=[];local_rows=[]
    for (year,label),spec in specs.items():
        reference=arrays(parent/f'input/outer/{year}/train_PRIVATE.npz');n=len(reference['X']);zero=np.zeros((n,4,2))
        p=Path(spec['path']);anchor=component_values(p,spec['keep'],reference)
        seed_parts=[];seed_predictions=[]
        for folder in seed_folders(spec):
            # component_values expects an ensemble subdirectory; read directly here.
            ex=CIFExpansion.load(folder/'functions_PRIVATE.npz');x=reference['X'][:,spec['keep']]
            seed_parts.append({tuple(spec['keep'][j] for j in key):ex.component(x,key,TIMES)
                               for key in [(j,) for j in ex.mains]+ex.pairs})
            seed_predictions.append(ex.predict(x,TIMES))
        keys=set(anchor)|set().union(*(set(x) for x in seed_parts))
        for h in [12,24]:
            ix=np.searchsorted(TIMES,reference[f'h{h}']);rr=np.arange(n)
            for cause in [1,2]:
                pp=np.stack([x[rr,ix,cause-1] for x in seed_predictions])
                for order in [1,2]:
                    terms=[k for k in keys if len(k)==order];pairstats=[]
                    for i,j in itertools.combinations(range(3),2):
                        ai={k for k in terms if np.sqrt(np.mean(seed_parts[i].get(k,zero)[rr,ix,cause-1]**2))>1e-14}
                        aj={k for k in terms if np.sqrt(np.mean(seed_parts[j].get(k,zero)[rr,ix,cause-1]**2))>1e-14}
                        distances=[];correlations=[]
                        for key in sorted(set(terms)):
                            a=seed_parts[i].get(key,zero)[rr,ix,cause-1];b=seed_parts[j].get(key,zero)[rr,ix,cause-1]
                            distances.append(np.mean((a-b)**2))
                            if key in ai&aj and a.std()>1e-14 and b.std()>1e-14:correlations.append(np.corrcoef(a,b)[0,1])
                        pairstats.append(dict(jaccard=len(ai&aj)/len(ai|aj) if ai|aj else 1.,
                                              spread=float(np.sqrt(np.sum(distances))),
                                              correlation=float(np.mean(correlations)) if correlations else np.nan,
                                              correlations=len(correlations)))
                    seed_rows.append(dict(year=year,model=label,horizon=h,cause=cause,order=order,reference_companies=n,
                        seed_pair_jaccard=float(np.mean([s['jaccard'] for s in pairstats])),
                        seed_function_difference_l2=float(np.mean([s['spread'] for s in pairstats])),
                        mean_company_prediction_sd=float(np.std(pp,axis=0,ddof=1).mean()),
                        mean_common_function_correlation=float(np.nanmean([s['correlation'] for s in pairstats])) if any(np.isfinite(s['correlation']) for s in pairstats) else np.nan,
                        valid_common_correlations=sum(s['correlations'] for s in pairstats)))
                    for key in sorted(terms):
                        a=anchor.get(key,zero)[rr,ix,cause-1]
                        values=[np.mean(abs(s.get(key,zero)[rr,ix,cause-1])) for s in seed_parts]
                        function_rows.append(dict(year=year,model=label,horizon=h,cause=cause,order=order,key=','.join(map(str,key)),
                            feature_names=';'.join(features[j]['feature_name'] for j in key),
                            mean_absolute_ensemble_contribution=float(np.mean(abs(a))),seed_mean_absolute_mean=float(np.mean(values)),
                            seed_mean_absolute_min=float(min(values)),seed_mean_absolute_max=float(max(values)),
                            active_seeds=sum(v>1e-14 for v in values)))
        if label=='selected_search':
            base=component_values(Path(specs[(year,'full83')]['path']),FULL,reference);allkeys=set(base)|set(anchor)
            oldex=CIFExpansion.load(Path(specs[(year,'full83')]['path'])/'ensemble/functions_PRIVATE.npz')
            newex=CIFExpansion.load(p/'ensemble/functions_PRIVATE.npz')
            oldmu=oldex._coeff(TIMES)[1];newmu=newex._coeff(TIMES)[1]
            oldprediction=oldex.predict(reference['X'][:8],TIMES)
            newprediction=newex.predict(reference['X'][:8,spec['keep']],TIMES)
            oldsum=oldmu+sum((base.get(k,zero)[:8] for k in allkeys),start=np.zeros((8,4,2)))
            newsum=newmu+sum((anchor.get(k,zero)[:8] for k in allkeys),start=np.zeros((8,4,2)))
            error=float(max(abs(oldsum-oldprediction).max(),abs(newsum-newprediction).max()))
            assert error<=1e-10
            for h in [12,24]:
                ix=np.searchsorted(TIMES,reference[f'h{h}'][:8]);rr=np.arange(8)
                for cause in [1,2]:
                    changes=np.stack([(anchor.get(k,zero)-base.get(k,zero))[rr,ix,cause-1] for k in allkeys])
                    local_rows.append(dict(year=year,horizon=h,cause=cause,historical_companies=8,
                        mean_total_absolute_contribution_change=float(abs(changes).sum(0).mean()),
                        maximum_total_absolute_contribution_change=float(abs(changes).sum(0).max()),
                        mean_absolute_CIF_change=float(abs(newprediction-oldprediction)[rr,ix,cause-1].mean()),
                        maximum_reconstruction_error=error))
            np.savez_compressed(run/f'private/local_explanation_{year}_PRIVATE.npz',
                sample_ids=reference['sample_ids'][:8],keys=np.array([','.join(map(str,k)) for k in sorted(allkeys)]),
                original=np.stack([base.get(k,zero)[:8] for k in sorted(allkeys)]),
                selected=np.stack([anchor.get(k,zero)[:8] for k in sorted(allkeys)]),
                original_baseline=oldmu,selected_baseline=newmu,
                original_prediction=oldprediction,selected_prediction=newprediction)
    pd.DataFrame(seed_rows).to_csv(run/'reports/SEED_FUNCTION_STABILITY.csv',index=False)
    pd.DataFrame(function_rows).to_csv(run/'reports/FUNCTION_IMPORTANCE.csv',index=False)
    pd.DataFrame(local_rows).to_csv(run/'reports/LOCAL_EXPLANATION_CHANGES.csv',index=False)


def stability(run):
    parent,specs=model_specs(run);registry=load(run/'audit/registry_bootstrap.json')
    existing=load(parent/'audit/registry_aggregate_bootstrap.json');summary=[];paired=[];pred=[]
    for year in [2023,2024,2025]:
        reference=arrays(parent/f'input/outer/{year}/train_PRIVATE.npz')
        spec=specs[(year,'selected_search')];anchor=component_values(Path(spec['path']),spec['keep'],reference)
        base_spec=specs[(year,'full83')];base_anchor=component_values(Path(base_spec['path']),FULL,reference)
        parts=[];base_parts=[];mus=[];base_mus=[]
        for repeat in range(50):
            old=next(r for r in existing if r['year']==year and r['repeat']==repeat)
            base_path=Path(old['path']);base=component_values(base_path,FULL,reference)
            mu=CIFExpansion.load(base_path/'ensemble/functions_PRIVATE.npz')._coeff(TIMES)[1]
            base_parts.append(base);base_mus.append(mu)
            if tuple(spec['keep'])==FULL:parts.append(base);mus.append(mu)
            else:
                r=next(r for r in registry if r['year']==year and r['repeat']==repeat)
                path=Path(r['path']);assert r['keep']==spec['keep']
                parts.append(component_values(path,spec['keep'],reference))
                mus.append(CIFExpansion.load(path/'ensemble/functions_PRIVATE.npz')._coeff(TIMES)[1])
        aa=CIFExpansion.load(Path(base_spec['path'])/'ensemble/functions_PRIVATE.npz')._coeff(TIMES)[1]
        bb=CIFExpansion.load(Path(spec['path'])/'ensemble/functions_PRIVATE.npz')._coeff(TIMES)[1]
        a,b,c=compare_variation(year,reference,base_parts,parts,base_anchor,anchor,base_mus,mus,aa,bb)
        for r in a+c:
            if r['model']=='selected_h24':r['model']='selected_search'
            r['scope']='FIXED_SELECTED_FEATURE_SET; not repeated search'
        summary+=a;paired+=b;pred+=c
        print(f'Fixed-feature function bootstrap analyzed: {year}, 50 paired company draws',flush=True)
    pd.DataFrame(summary).to_csv(run/'reports/FIXED_SET_FUNCTION_STABILITY.csv',index=False)
    pd.DataFrame(paired).to_csv(run/'reports/PAIRED_FIXED_SET_STABILITY.csv',index=False)
    pd.DataFrame(pred).to_csv(run/'reports/FIXED_SET_PREDICTION_STABILITY.csv',index=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--stage',choices=['prediction','functions','stability'],required=True)
    args=parser.parse_args();globals()[args.stage](args.run)
