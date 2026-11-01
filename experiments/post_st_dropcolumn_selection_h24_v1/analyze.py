"""Selected predictor evaluation after historical decisions are sealed."""
from pathlib import Path
from collections import Counter
import argparse,json
import numpy as np
import pandas as pd
from crforest import CIFExpansion
from crforest.partition import tree_feature_subset
from experiments.post_st_dropcolumn_selection_v1.common import bootstrap_differences,REFERENCE_SHA
from .pipeline import ExtendedStore
from .fitting import *

def final_models(run,parent):
    decisions=load(run/'audit/selection_decisions.json');new={r['year']:Path(r['output'])/'bundle' for r in load(run/'audit/registry_final.json')}
    old=load(parent/'audit/registry_aggregate_outer.json');sets=load(parent/'input/subsets.json')
    full=next(sid for sid,s in sets.items() if len(s['keep'])==83 and s['mode']=='count')
    return {int(y):dict(path=new[int(y)] if len(d['keep'])<83 else Path(next(r['path'] for r in old if r['year']==int(y) and r['subset_id']==full)),
                       keep=d['keep'],new=len(d['keep'])<83) for y,d in decisions.items()}

def prediction_analysis(run,parent):
    finals=final_models(run,parent);models=[];complexities=[];predictions={};labels={};selected=[]
    features=load(parent/'input/features.json');decisions=load(run/'audit/selection_decisions.json')
    store=ExtendedStore(parent,run);historical=[];paths=[]
    for y,d in decisions.items():
        outer=int(y);support,_=historical_support(parent,outer);e=Evaluator(parent,outer,store,support)
        for name,keep in [('full83',list(range(83))),('selected_h24',d['keep'])]:
            if e.cohorts[24]:
                result=e.summary(keep)
                for h in [12,24]:
                    for cause in [1,2]:
                        row=dict(outer=outer,model=name,features=len(keep),horizon=h,cause=cause,cohorts=','.join(map(str,e.cohorts[h])),spatial_cells=result['cells'])
                        for metric in ['auc','ap','brier','captured']:row[metric]=result['metrics'][(h,cause,metric)]
                        historical.append(row)
        for step in d['path']:paths.append(dict(outer=outer,**step))
        for j,f in enumerate(features):selected.append(dict(outer=outer,feature_id=j,feature_name=f['feature_name'],feature_label=f['label'],selected=j in d['keep'],reason=d['status']))
    for year,spec in finals.items():
        p=spec['path'];z=arrays(p/'ensemble/predictions_PRIVATE.npz');lab=arrays(parent/f'evaluation_input/{year}_test_PRIVATE.npz');labels[year]=lab
        np.testing.assert_array_equal(z['sample_ids'],lab['sample_ids']);predictions[year]=z['cif']
        check=load(p/'summary.json')['ensemble_check' if spec['new'] else 'check']
        complexities.append(dict(year=year,model='selected_h24',features=len(spec['keep']),**{k:check[k] for k in ['main_functions','interaction_functions','spatial_table_cells','scalar_table_cells','distinct_thresholds','function_bytes','max_path_variables']}))
        for h in [12,24]:
            risk=horizon_predictions(z['cif'],TIMES,lab[f'h{h}'])
            for cause in [1,2]:models.append(dict(year=str(year),label='selected_h24',features=len(spec['keep']),horizon=h,cause=cause,seed='ensemble',**metrics(lab,risk[:,cause-1],h,cause)))
        if spec['new']:seedpaths=[p/f'{seed}/shrunk' for seed in SEEDS]
        else:
            r=next(r for r in load(parent/'audit/registry_aggregate_outer.json') if Path(r['path'])==p);seedpaths=list(map(Path,r['paths']))
        for seed,sp in zip(SEEDS,seedpaths):
            zz=arrays(sp/'predictions_PRIVATE.npz');np.testing.assert_array_equal(zz['sample_ids'],lab['sample_ids'])
            for h in [12,24]:
                risk=horizon_predictions(zz['cif'],TIMES,lab[f'h{h}'])
                for cause in [1,2]:models.append(dict(year=str(year),label='selected_h24',features=len(spec['keep']),horizon=h,cause=cause,seed=str(seed),**metrics(lab,risk[:,cause-1],h,cause)))
    for h in [12,24]:
        years=[y for y in finals if evaluation_context(labels[y],h,1)[3]];ll={k:np.concatenate([labels[y][k] for y in years]) for k in labels[years[0]]}
        pp=np.concatenate([horizon_predictions(predictions[y],TIMES,labels[y][f'h{h}']) for y in years])
        for cause in [1,2]:
            m=metrics(ll,pp[:,cause-1],h,cause);annual=[r for r in models if r['year'] in list(map(str,years)) and r['seed']=='ensemble' and r['horizon']==h and r['cause']==cause]
            m['captured']=sum(r['captured'] for r in annual);m['checked']=sum(r['checked'] for r in annual);m['coverage']=m['captured']/m['target_events'] if m['target_events'] else np.nan
            models.append(dict(year='pooled',label='selected_h24',features=np.nan,horizon=h,cause=cause,seed='ensemble',cohorts=','.join(map(str,years)),**m))
    old=pd.read_csv(parent/'reports/MODEL_COMPARISON.csv',dtype={'year':str,'seed':str});old=old[old.label.isin(['full83','legacy74'])&(old['mode']=='count')]
    frame=pd.concat([old,pd.DataFrame(models)],ignore_index=True);frame.to_csv(run/'reports/MODEL_COMPARISON.csv',index=False)
    oldcx=pd.read_csv(parent/'reports/EXPLANATION_COMPLEXITY.csv');oldcx=oldcx[oldcx.label.isin(['full83','legacy74'])&(oldcx['mode']=='count')].rename(columns={'label':'model'})
    pd.concat([oldcx,pd.DataFrame(complexities)],ignore_index=True).to_csv(run/'reports/EXPLANATION_COMPLEXITY.csv',index=False)
    pd.DataFrame(selected).to_csv(run/'reports/SELECTED_FEATURES.csv',index=False)
    pd.DataFrame(historical).to_csv(run/'reports/HISTORICAL_SELECTED_COMPARISON.csv',index=False)
    for row in paths:
        if 'evidence' in row:row['evidence']=json.dumps(row['evidence'])
        if 'removed' in row:row['removed']=json.dumps(row['removed'])
    pd.DataFrame(paths).to_csv(run/'reports/RECURSIVE_ELIMINATION_PATH.csv',index=False)
    scans=pd.read_csv(run/'reports/SELECTION_CONTRASTS.csv');guard_rows=[]
    for (year,phase),part in scans.groupby(['outer','phase']):
        counts=Counter(g for s in part.failed_guards for g in json.loads(s))
        for guard,n in counts.items():guard_rows.append(dict(year=year,phase=phase,guard=guard,rejections=n,candidates=len(part),metric_guard_passes=int(part.approved.sum())))
    pd.DataFrame(guard_rows).to_csv(run/'reports/GUARD_REJECTION_SUMMARY.csv',index=False)
    base={r['year']:arrays(Path(r['path'])/'ensemble/predictions_PRIVATE.npz')['cif'] for r in load(parent/'audit/registry_aggregate_outer.json') if len(r['keep'])==83}
    local_rows=[];removed_context=[]
    for year,spec in finals.items():
        reference=arrays(parent/f'input/outer/{year}/train_PRIVATE.npz');oldrecord=next(r for r in load(parent/'audit/registry_aggregate_outer.json') if r['year']==year and len(r['keep'])==83)
        oldfolder=Path(oldrecord['path']);n=min(8,len(reference['X']));query={'X':reference['X'][:n]}
        left=component_values(oldfolder,list(range(83)),query);right=component_values(spec['path'],spec['keep'],query)
        keys=sorted(set(left)|set(right));zero=np.zeros((n,4,2))
        a=np.stack([left.get(k,zero)[:n] for k in keys],axis=1);b=np.stack([right.get(k,zero)[:n] for k in keys],axis=1)
        ea=CIFExpansion.load(oldfolder/'ensemble/functions_PRIVATE.npz');eb=CIFExpansion.load(spec['path']/'ensemble/functions_PRIVATE.npz');ma=ea._coeff(TIMES)[1];mb=eb._coeff(TIMES)[1]
        # Describe the deleted fields in the saved full model, on the same
        # historical reference. This does not enter selection or refit a model.
        for j in sorted(set(range(83))-set(spec['keep'])):
            main=ea.component(reference['X'],(j,),TIMES) if j in ea.mains else np.zeros((len(reference['X']),4,2))
            pair_keys=[key for key in ea.pairs if j in key]
            pair_abs=sum((abs(ea.component(reference['X'],key,TIMES)) for key in pair_keys),start=np.zeros_like(main))
            for h in [12,24]:
                idx=np.searchsorted(TIMES,reference[f'h{h}']);rr=np.arange(len(idx))
                for cause in [1,2]:
                    removed_context.append(dict(year=year,feature_id=j,feature_name=features[j]['feature_name'],feature_label=features[j]['label'],horizon=h,cause=cause,
                        full83_main_present=j in ea.mains,full83_pair_count=len(pair_keys),mean_absolute_main=float(abs(main[rr,idx,cause-1]).mean()),
                        mean_sum_absolute_pairs=float(pair_abs[rr,idx,cause-1].mean()),reference_companies=len(rr)))
        pa=ea.predict(reference['X'][:n],TIMES);pb=eb.predict(reference['X'][:n,spec['keep']],TIMES)
        error=float(max(abs(pa-(ma+a.sum(1))).max(),abs(pb-(mb+b.sum(1))).max()));assert error<=1e-10
        np.savez_compressed(run/f'private/local_explanation_{year}_PRIVATE.npz',keys=np.array([','.join(map(str,k)) for k in keys]),sample_ids=reference['sample_ids'][:n],baseline_terms=a,selected_terms=b,baseline_mu=ma,selected_mu=mb,baseline_prediction=pa,selected_prediction=pb)
        for h in [12,24]:
            ix=np.searchsorted(TIMES,reference[f'h{h}'][:n]);rows=np.arange(n)
            for cause in [1,2]:local_rows.append(dict(year=year,horizon=h,cause=cause,historical_rows=n,mean_total_absolute_component_change=float(abs(b-a)[rows,:,ix,cause-1].sum(1).mean()),mean_absolute_CIF_change=float(abs(pb-pa)[rows,ix,cause-1].mean()),max_reconstruction_error=error))
    pd.DataFrame(local_rows).to_csv(run/'reports/LOCAL_EXPLANATION_CHANGES.csv',index=False)
    pd.DataFrame(removed_context).to_csv(run/'reports/REMOVED_FEATURE_CONTEXT.csv',index=False)
    schedule=[]
    cases=[(0,'single',j,[k for k in range(83) if k!=j]) for j in range(83)]
    cases += [(year,'selected_h24',-1,spec['keep']) for year,spec in finals.items()]
    for year,kind,j,keep in cases:
        for seed in SEEDS:
            overlaps=[];common=[];same=[]
            for tree in range(32):
                a=set(tree_feature_subset(83,25,seed,tree,'balanced'))
                b={keep[k] for k in tree_feature_subset(len(keep),25,seed,tree,'balanced')}
                overlaps.append(len(a&b)/len(a|b));common.append(len(a&b));same.append(a==b)
            schedule.append(dict(year=year,comparison=kind,removed_feature_id=j,seed=seed,input_features=len(keep),eligible_per_tree=25,
                eligible_jaccard_mean=float(np.mean(overlaps)),eligible_common_count_mean=float(np.mean(common)),exact_eligible_set_matches=sum(same),trees=32))
    pd.DataFrame(schedule).to_csv(run/'reports/FEATURE_SCHEDULE_DIAGNOSTIC.csv',index=False)
    contrasts=[]
    for yeargroup in [[2023],[2024],[2025],[2023,2024,2025]]:
        for h in [12,24]:
            years=[y for y in yeargroup if evaluation_context(labels[y],h,1)[3]]
            if not years:continue
            ll={k:np.concatenate([labels[y][k] for y in years]) for k in labels[years[0]]};pa=np.concatenate([horizon_predictions(base[y],TIMES,labels[y][f'h{h}']) for y in years]);pb=np.concatenate([horizon_predictions(predictions[y],TIMES,labels[y][f'h{h}']) for y in years])
            offset=0;parts=[]
            for y in years:
                n=len(labels[y]['event']);parts.append(np.random.default_rng(np.random.SeedSequence([SELECTION_SEED,y,h,99])).integers(n,size=(500,n))+offset);offset+=n
            draws=np.concatenate(parts,axis=1)
            for cause in [1,2]:
                dif=bootstrap_differences(ll,pa[:,cause-1],pb[:,cause-1],h,cause,draws);a=metrics(ll,pa[:,cause-1],h,cause);b=metrics(ll,pb[:,cause-1],h,cause)
                for metric,values in dif.items():
                    v=values[np.isfinite(values)];contrasts.append(dict(year=years[0] if len(yeargroup)==1 else 'pooled',cohorts=','.join(map(str,years)),horizon=h,cause=cause,metric=metric,difference=b[metric]-a[metric],lower95=np.quantile(v,.025) if len(v) else np.nan,upper95=np.quantile(v,.975) if len(v) else np.nan,undefined=500-len(v)))
    pd.DataFrame(contrasts).to_csv(run/'reports/PAIRED_DIFFERENCES.csv',index=False)
    print(frame[(frame.seed=='ensemble')&(frame.cause==1)][['year','label','horizon','auc','ap','brier','captured','target_events']].to_string(index=False),flush=True)

def component_values(folder,keep,reference):
    ex=CIFExpansion.load(folder/'ensemble/functions_PRIVATE.npz');x=reference['X'][:,keep];values={}
    for key in [(j,) for j in ex.mains]+ex.pairs:
        original=tuple(keep[j] for j in key);values[original]=ex.component(x,key,TIMES)
    return values

def stability_analysis(run,parent):
    registry=load(run/'audit/registry_bootstrap.json');features=load(parent/'input/features.json');finals=final_models(run,parent)
    selections=[];functions=[];gates=[];locals_=[]
    for year in [2023,2024,2025]:
        records=sorted([r for r in registry if r['outer']==year],key=lambda r:r['repeat']);assert [r['repeat'] for r in records]==list(range(50))
        reference=arrays(parent/f'input/outer/{year}/train_PRIVATE.npz');anchor=component_values(finals[year]['path'],finals[year]['keep'],reference);parts=[];statuses=[]
        for r in records:
            p=Path(r['output']);s=load(p/'summary.json');f=s['final'];folder=Path(f['path']) if f['reused'] else p/f['path']
            assert sha(folder/'ensemble/functions_PRIVATE.npz')==f['functions_sha256']
            scanned=bool(s['bundles']);statuses.append(s);parts.append(component_values(folder,s['keep'],reference))
            gates.append(dict(year=year,repeat=r['repeat'],support_passed=s['gate']['passed'],target_events=s['gate']['target_events'],selection_scan_executed=scanned,selected_features=len(s['keep']),new_forests=s['new_forests']))
        qualifying=[i for i,s in enumerate(statuses) if s['bundles']]
        for j,f in enumerate(features):
            selections.append(dict(outer=year,feature_id=j,feature_name=f['feature_name'],feature_label=f['label'],replicates=50,support_passes=sum(s['gate']['passed'] for s in statuses),actual_selection_scans=len(qualifying),
                unconditional_inclusion=float(np.mean([j in s['keep'] for s in statuses])),conditional_inclusion_given_scan=float(np.mean([j in statuses[i]['keep'] for i in qualifying])) if qualifying else np.nan,
                conditional_deletion_gain_support='conditional on frozen correlation groups and three fixed forest seeds; fallback inclusion is not indispensability'))
        keys=sorted(set(anchor).union(*(set(x) for x in parts)));zero=np.zeros((len(reference['X']),4,2))
        for key in keys:
            baseline=anchor.get(key,zero)
            for h in [12,24]:
                ix=np.searchsorted(TIMES,reference[f'h{h}'])
                for cause in [1,2]:
                    b=baseline[np.arange(len(ix)),ix,cause-1];amplitude=[];correlations=[];distance=[]
                    for part in parts:
                        v=part.get(key,zero)[np.arange(len(ix)),ix,cause-1];amplitude.append(float(np.sqrt(np.mean(v*v))));distance.append(float(np.sqrt(np.mean((v-b)**2))))
                        correlations.append(float(np.corrcoef(v,b)[0,1]) if v.std()>1e-14 and b.std()>1e-14 else np.nan)
                    valid=np.asarray(correlations);valid=valid[np.isfinite(valid)]
                    functions.append(dict(outer=year,key=','.join(map(str,key)),feature_names=';'.join(features[j]['feature_name'] for j in key),order=len(key),horizon=h,cause=cause,
                        reference_nonzero_frequency=float(np.mean(np.asarray(amplitude)>1e-14)),rms_mean=np.mean(amplitude),rms_sd=np.std(amplitude,ddof=1),rms_p025=np.quantile(amplitude,.025),rms_p975=np.quantile(amplitude,.975),
                        anchor_rms=float(np.sqrt(np.mean(b*b))),shape_correlation_mean=valid.mean() if len(valid) else np.nan,shape_correlation_valid=len(valid),rms_distance_from_anchor=np.mean(distance)))
        for i,part in enumerate(parts):
            total=sum((abs(part.get(k,zero)-anchor.get(k,zero)) for k in keys),start=zero.copy())
            locals_.append(dict(outer=year,repeat=i,historical_rows=min(8,len(total)),mean_absolute_component_change=float(total[:8].mean()),interpretation='same historical training rows, sum of absolute main/pair changes; no company identity released'))
    pd.DataFrame(selections).to_csv(run/'reports/FEATURE_SELECTION_STABILITY.csv',index=False);pd.DataFrame(gates).to_csv(run/'reports/BOOTSTRAP_SUPPORT_AND_SELECTION.csv',index=False)
    df=pd.DataFrame(functions);df[df.order==2].to_csv(run/'reports/INTERACTION_STABILITY.csv',index=False);df[df.order==1].to_csv(run/'reports/MAIN_FUNCTION_STABILITY.csv',index=False)
    pd.DataFrame(locals_).to_csv(run/'reports/LOCAL_EXPLANATION_STABILITY.csv',index=False)
    print(pd.DataFrame(gates).groupby('year').agg(support=('support_passed','sum'),scanned=('selection_scan_executed','sum'),min_features=('selected_features','min'),mean_features=('selected_features','mean'),new_forests=('new_forests','sum')).to_string(),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--stage',choices=['prediction','stability'],required=True);a=p.parse_args();parent=Path(load(a.run/'input/protocol.json')['parent_private_run'])
    (prediction_analysis if a.stage=='prediction' else stability_analysis)(a.run,parent)
