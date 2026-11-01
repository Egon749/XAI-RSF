"""Independently aggregate full pipelines; read outer outcomes only after sealing decisions."""
import datetime, itertools, json, shutil
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from crforest import CIFExpansion
from .run import RUN, PUBLIC, PARENT, SEARCH
from experiments.post_st_feature_combination_search_v1.analyze import model_specs, seed_folders
from experiments.post_st_dropcolumn_selection_v1.common import (
    arrays,load,write,sha,TIMES,metrics,evaluation_context,horizon_predictions,bootstrap_differences)


def save(name, rows):
    df=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
    df.to_csv(RUN/'reports'/name,index=False)
    shutil.copy2(RUN/'reports'/name,PUBLIC/name)


def stages():
    return ['pilot_recovered' if (RUN/'audit/registry_pilot_recovered.json').exists() else 'pilot','batch']


def collect():
    for p,h in load(RUN/'audit/source_hashes.json').items():assert sha(Path(p))==h
    records=sum([load(RUN/f'audit/registry_{s}.json') for s in stages()],[])
    assert {(r['year'],r['repeat']) for r in records}=={(y,i) for y in [2023,2024,2025] for i in [-1]+list(range(50))}
    assert len(records)==153
    tasks=[];checks=[];hashes=[];scans=[];interactions=[];groups=[];decisions={};costs=[]
    failed=load(RUN/'audit/failed_pilot_job2607.json')['completed'] if (RUN/'audit/failed_pilot_job2607.json').exists() else []
    for r in records:
        out=Path(r['output']);s=load(out/'summary.json');keep=s['keep']
        for rel,h in load(out/'hashes.json').items():
            assert sha(out/rel)==h,(out,rel)
            hashes.append(dict(year=r['year'],repeat=r['repeat'],relative=rel,sha256=h))
        scores=load(out/'candidate_scores.json');ic=load(out/'interaction_checks.json')
        assert s['scored_candidates']==len(scores)
        assert s['supervised_search_executed']==bool(scores)
        if s['gate']['passed']:
            assert len(scores)==len(s['candidates'])-1 and len(s['discovery_bundles'])>0
        else:assert not scores and keep==list(range(83)) and not s['discovery_bundles']
        assert s['new_forests']==3*(len(s['discovery_bundles'])+1)
        fit_summaries=[load(out/rel/'summary.json') for rel in s['discovery_bundles']]+[load(out/'final/summary.json')]
        initial_seconds=next((f['seconds'] for f in failed if (s['year'],s['repeat'])==((2024,-1) if f['id']==0 else (2025,0))),0.) if s.get('recovered_forests',0) else 0.
        costs.append(dict(year=s['year'],repeat=s['repeat'],host=s['host'],unique_raw_forests=s['new_forests'],
            recovered_forests=s.get('recovered_forests',0),fits_this_attempt=s.get('fits_this_attempt',s['new_forests']),
            successful_attempt_seconds=s['seconds'],failed_attempt_seconds=initial_seconds,
            fit_seconds=sum(sum(z['fit_seconds']) for z in fit_summaries),shrink_seconds=sum(sum(z['shrink_seconds']) for z in fit_summaries),
            bundle_wall_seconds=sum(z['seconds'] for z in fit_summaries),peak_rss_MiB=s['peak_rss_MiB']))
        if r['repeat']==-1:decisions[str(r['year'])]=dict(keep=keep,status=s['selection_status'],gate=s['gate'],output=r['output'])
        tasks.append(dict(year=s['year'],repeat=s['repeat'],host=s['host'],seconds=s['seconds'],peak_rss_MiB=s['peak_rss_MiB'],new_forests=s['new_forests'],
            selection_status=s['selection_status'],features=len(keep),candidate_count=len(s['candidates']),scored_candidates=len(scores),discovery_bundles=len(s['discovery_bundles']),
            discovery_targets=s['gate']['target_events'],discovery_pass=s['gate']['passed'],supervised_search_executed=s['supervised_search_executed']))
        for g in s['groups']:groups.append(dict(year=s['year'],repeat=s['repeat'],members=json.dumps(g['members']),one=json.dumps(g['one']),two=json.dumps(g['two'])))
        for row in scores:
            scans.append(dict(year=s['year'],repeat=s['repeat'],**{k:json.dumps(v) if isinstance(v,list) else v for k,v in row.items()}))
        for row in ic:interactions.append(dict(year=s['year'],repeat=s['repeat'],**{k:json.dumps(v) if isinstance(v,list) else v for k,v in row.items()}))
        for p in out.rglob('verification.json'):
            saved=arrays(p.parent/'predictions_PRIVATE.npz')['cif'];assert np.isfinite(saved).all()
            v=load(p);assert v['max_path_variables']<=2 and v['path_violations']==0
            assert max(v[k] for k in ['full_reconstruction','explicit_component_sum','function_reload','forest_reload'])<=1e-10
            assert v['probability_min']>=-1e-12 and v['probability_sum_max']<=1+1e-10 and v['minimum_increment']>=-1e-12
            checks.append(dict(year=s['year'],repeat=s['repeat'],part=str(p.relative_to(out)),**v))
    assert len(checks)==sum(r['new_forests']//3*7 for r in tasks)
    # Permit analysis-only replay after a reporting failure, never alter decisions.
    if (RUN/'audit/selection_decisions.json').exists():assert load(RUN/'audit/selection_decisions.json')==decisions
    else:write(RUN/'audit/selection_decisions.json',decisions)
    for name,rows in [('COMPUTE_COST.csv',costs),('PIPELINE_RUNS.csv',tasks),('GROUP_DISCOVERY_RESULTS.csv',scans),('INTERACTION_PROTECTION_DIAGNOSTICS.csv.gz',interactions),
        ('RESAMPLED_GROUPS.csv',groups),('NUMERICAL_VALIDATION.csv.gz',checks),('MODEL_HASHES.csv.gz',hashes)]:save(name,rows)
    interaction_frame=pd.DataFrame(interactions)
    if len(interaction_frame):
        interaction_frame['block_type']=np.where(interaction_frame.partner==-1,'within_group',np.where(interaction_frame.partner==-2,'no_affected_pair','group_external'))
        summary=interaction_frame.groupby(['year','repeat','keep','block_type'],dropna=False).agg(checks=('passed','size'),passed_checks=('passed','sum'),median_ratio=('ratio','median'),median_correlation=('correlation','median'),median_relative_l2=('relative_l2','median')).reset_index()
        summary['failed_checks']=summary.checks-summary.passed_checks
        save('INTERACTION_GUARD_BLOCK_SUMMARY.csv',summary)
    return records,tasks,checks,decisions


def stability(records,decisions):
    features=load(PARENT/'input/features.json');freq=[];selected=[];stability_rows=[];prediction_rows=[];group_rows=[];overlaps=[]
    for y in [2023,2024,2025]:
        rr=[r for r in records if r['year']==y and r['repeat']>=0]
        summaries=[load(Path(r['output'])/'summary.json') for r in rr]
        searching=[s for s in summaries if s['supervised_search_executed']]
        for f in features:
            j=f['feature_id']
            freq.append(dict(year=y,**f,process_replicates=50,supervised_search_replicates=len(searching),
                pipeline_retention_frequency=np.mean([j in s['keep'] for s in summaries]),
                conditional_retention_frequency=np.mean([j in s['keep'] for s in searching]) if searching else np.nan,
                deletion_frequency=np.mean([j not in s['keep'] for s in summaries]),
                status='ACTUAL_FINITE_SELECTION_PROCESS' if searching else 'NO_SUPPORTED_SEARCH'))
            selected.append(dict(year=y,**f,selected=j in decisions[str(y)]['keep'],reason=decisions[str(y)]['status']))
        group_keys=sorted({tuple(g['members']) for s in summaries for g in s['groups']})
        for key in group_keys:
            present=[s for s in summaries if any(tuple(g['members'])==key for g in s['groups'])]
            searched=[s for s in present if s['supervised_search_executed']]
            group_rows.append(dict(year=y,members=json.dumps(key),identification_frequency=len(present)/50,
                searchable_replicates=len(searched),reduced_replicates=sum(not set(key)<=set(s['keep']) for s in searched),
                conditional_reduction_frequency=np.mean([not set(key)<=set(s['keep']) for s in searched]) if searched else np.nan))
        data=arrays(PARENT/f'input/outer/{y}/train_PRIVATE.npz');X=data['X'];n=len(X)
        ix=np.stack([np.searchsorted(TIMES,data['h12']),np.searchsorted(TIMES,data['h24'])],axis=1);row_index=np.arange(n)[:,None]
        moments={};pred_sum=np.zeros((n,2,2));pred_sq=np.zeros_like(pred_sum);sets=[]
        for r,s in zip(rr,summaries):
            ex=CIFExpansion.load(Path(r['output'])/'final/ensemble/functions_PRIVATE.npz');keep=s['keep'];xx=X[:,keep]
            pred=ex.predict(xx,TIMES)[row_index,ix];pred_sum+=pred;pred_sq+=pred*pred
            keys=[(j,) for j in ex.mains]+ex.pairs;mapped=set()
            for key in keys:
                original=tuple(int(keep[j]) for j in key);mapped.add(original)
                v=ex.component(xx,key,TIMES)[row_index,ix]
                if original not in moments:moments[original]=[np.zeros_like(v),np.zeros_like(v),0]
                moments[original][0]+=v;moments[original][1]+=v*v;moments[original][2]+=1
            sets.append(mapped)
        for order in [1,2]:
            pairs=[]
            for a,b in itertools.combinations(sets,2):
                aa={k for k in a if len(k)==order};bb={k for k in b if len(k)==order}
                pairs.append(len(aa&bb)/len(aa|bb) if aa|bb else 1.)
            overlaps.append(dict(year=y,order=order,model_pairs=len(pairs),mean_jaccard=float(np.mean(pairs)),p05=float(np.quantile(pairs,.05)),p95=float(np.quantile(pairs,.95))))
        for key,(s,q,count) in moments.items():
            for t,h in enumerate([12,24]):
                for c in [1,2]:
                    mu=s[:,t,c-1]/50;var=np.maximum(q[:,t,c-1]/50-mu*mu,0)
                    stability_rows.append(dict(year=y,order=len(key),features=json.dumps(key),names=' × '.join(features[j]['label'] for j in key),horizon=h,cause=c,
                        structural_presence=count/50,mean_abs_mean_function=float(abs(mu).mean()),rms_function_sd=float(np.sqrt(var.mean())),
                        scope='COMPLETE_SELECTION_PIPELINE_ON_FIXED_TRAIN_REFERENCE'))
        for t,h in enumerate([12,24]):
            for c in [1,2]:
                mu=pred_sum[:,t,c-1]/50;var=np.maximum(pred_sq[:,t,c-1]/50-mu*mu,0)
                prediction_rows.append(dict(year=y,horizon=h,cause=c,mean_prediction=float(mu.mean()),rms_prediction_sd=float(np.sqrt(var.mean()))))
        print('Verified full-pipeline function stability',y,flush=True)
    for name,rows in [('FEATURE_SELECTION_STABILITY.csv',freq),('GROUP_SELECTION_STABILITY.csv',group_rows),('SELECTED_FEATURES.csv',selected),
        ('FUNCTION_STABILITY.csv',stability_rows),('INTERACTION_STABILITY.csv',[r for r in stability_rows if r['order']==2]),
        ('FUNCTION_SET_STABILITY.csv',overlaps),('PREDICTION_STABILITY.csv',prediction_rows)]:save(name,rows)


def evaluation(decisions):
    _,specs=model_specs(SEARCH);labels={};predictions={};rows=[];complexity=[];sources=[]
    for y in [2023,2024,2025]:
        lab=arrays(PARENT/f'evaluation_input/{y}_test_PRIVATE.npz');labels[y]=lab
        train=arrays(PARENT/f'input/outer/{y}/train_PRIVATE.npz');assert not set(train['groups'])&set(lab['groups'])
        for role in ['full83','legacy74','selected_pipeline']:
            if role=='selected_pipeline':
                folder=Path(decisions[str(y)]['output'])/'final';keep=decisions[str(y)]['keep'];seeds=[folder/f'{s}/shrunk' for s in [42,101,202]]
            else:
                spec=specs[(y,role)];folder=Path(spec['path']);keep=spec['keep'];seeds=seed_folders(spec)
            hashes=load(folder/'hashes.json')
            for rel in ['ensemble/functions_PRIVATE.npz','ensemble/predictions_PRIVATE.npz']:assert sha(folder/rel)==hashes[rel]
            pp=arrays(folder/'ensemble/predictions_PRIVATE.npz');np.testing.assert_array_equal(pp['sample_ids'],lab['sample_ids']);predictions[(y,role)]=pp['cif']
            np.testing.assert_allclose(pp['cif'],np.mean([arrays(z/'predictions_PRIVATE.npz')['cif'] for z in seeds],axis=0),atol=1e-13,rtol=0)
            for seed,p in [('ensemble',folder/'ensemble/predictions_PRIVATE.npz')]+list(zip(['42','101','202'],[z/'predictions_PRIVATE.npz' for z in seeds])):
                z=arrays(p);np.testing.assert_array_equal(z['sample_ids'],lab['sample_ids'])
                for h in [12,24]:
                    risk=horizon_predictions(z['cif'],TIMES,lab[f'h{h}'])
                    for c in [1,2]:rows.append(dict(year=y,model=role,seed=seed,horizon=h,cause=c,**metrics(lab,risk[:,c-1],h,c)))
            ex=CIFExpansion.load(folder/'ensemble/functions_PRIVATE.npz');cc=ex.complexity(train['X'][:,keep],support_min=10)
            complexity.append(dict(year=y,model=role,features=len(keep),**{k:v for k,v in cc.items() if k!='functions'}))
            sources.append(dict(year=y,model=role,path=str(folder),keep=json.dumps(keep),prediction_sha256=sha(folder/'ensemble/predictions_PRIVATE.npz'),function_sha256=sha(folder/'ensemble/functions_PRIVATE.npz')))
    for y in [2023,2024,2025]:
        if len(decisions[str(y)]['keep'])==83:
            np.testing.assert_allclose(predictions[(y,'selected_pipeline')],predictions[(y,'full83')],atol=1e-13,rtol=0)
    for h in [12,24]:
        years=[y for y in labels if evaluation_context(labels[y],h,1)[3]]
        lab={k:np.concatenate([labels[y][k] for y in years]) for k in labels[years[0]]}
        for role in ['full83','legacy74','selected_pipeline']:
            risk=np.concatenate([horizon_predictions(predictions[(y,role)],TIMES,labels[y][f'h{h}']) for y in years])
            for c in [1,2]:
                m=metrics(lab,risk[:,c-1],h,c)
                annual=[z for z in rows if z['year'] in years and z['model']==role and z['seed']=='ensemble' and z['horizon']==h and z['cause']==c]
                m.update(checked=sum(z['checked'] for z in annual),captured=sum(z['captured'] for z in annual));m['coverage']=m['captured']/m['target_events'] if m['target_events'] else np.nan
                rows.append(dict(year='pooled',cohorts=','.join(map(str,years)),model=role,seed='ensemble',horizon=h,cause=c,**m))
    save('MODEL_COMPARISON.csv',rows);save('EXPLANATION_COMPLEXITY.csv',complexity);save('MODEL_SOURCES.csv',sources)
    # Common year-stratified draws, fixed models/IPCW weights; all comparisons and signs retained.
    intervals=[];rng=np.random.default_rng(2026110192);draws={y:rng.integers(len(l['event']),size=(1000,len(l['event']))) for y,l in labels.items()}
    np.savez_compressed(RUN/'audit/paired_resamples_PRIVATE.npz',**{str(y):d for y,d in draws.items()})
    for h in [12,24]:
        supported=[y for y in labels if evaluation_context(labels[y],h,1)[3]]
        for scope,years in [(str(y),[y]) for y in supported]+[('pooled',supported)]:
            lab={k:np.concatenate([labels[y][k] for y in years]) for k in labels[years[0]]};offset=0;parts=[]
            for y in years:parts.append(draws[y]+offset);offset+=len(labels[y]['event'])
            dd=np.concatenate(parts,axis=1)
            for c in [1,2]:
                for a,b in [('full83','selected_pipeline'),('legacy74','selected_pipeline'),('full83','legacy74')]:
                    aa=np.concatenate([horizon_predictions(predictions[(y,a)],TIMES,labels[y][f'h{h}'])[:,c-1] for y in years]);bb=np.concatenate([horizon_predictions(predictions[(y,b)],TIMES,labels[y][f'h{h}'])[:,c-1] for y in years])
                    diffs=bootstrap_differences(lab,aa,bb,h,c,dd)
                    for key,values in diffs.items():
                        finite=values[np.isfinite(values)]
                        point=metrics(lab,bb,h,c)[key]-metrics(lab,aa,h,c)[key]
                        intervals.append(dict(year=scope,horizon=h,cause=c,comparison=f'{b}_minus_{a}',metric=key,difference=point,
                            lower95=float(np.quantile(finite,.025)) if len(finite) else np.nan,upper95=float(np.quantile(finite,.975)) if len(finite) else np.nan,undefined=1000-len(finite)))
    save('PAIRED_DIFFERENCES.csv',intervals)


def main():
    records,tasks,checks,decisions=collect();stability(records,decisions);evaluation(decisions)
    maximum=max(max(v[k] for k in ['full_reconstruction','explicit_component_sum','forest_reload','function_reload']) for v in checks)
    receipt=dict(status='COMPLETED_ACTUAL_EXPLORATORY_SELECTION',training_code_commit=load(RUN/'audit'/('code_snapshot_fixed.json' if (RUN/'audit/code_snapshot_fixed.json').exists() else 'code_snapshot.json'))['commit'],recovered_training_code_commit=load(RUN/'audit/code_snapshot.json')['commit'],analysis_source_sha256=sha(Path(__file__)),
        frozen_protocol_sha256=sha(RUN/'input/protocol.json'),point_decisions=decisions,pipeline_tasks=len(records),new_raw_forests=sum(r['new_forests'] for r in tasks),
        derived_shrunk_forests=sum(r['new_forests'] for r in tasks),merged_ensembles=len(checks)//7,verified_models=len(checks),maximum_reconstruction_error=maximum,
        point_searches=sum(r['supervised_search_executed'] for r in tasks if r['repeat']==-1),resampled_searches=sum(r['supervised_search_executed'] for r in tasks if r['repeat']>=0),
        candidate_comparisons=sum(r['scored_candidates'] for r in tasks),failed_fits=0,failed_task_attempts=2 if (RUN/'audit/failed_pilot_job2607.json').exists() else 0,recovered_discovery_forests=sum(load(Path(r['output'])/'summary.json').get('recovered_forests',0) for r in records),serialization_incident='Two initial pilots failed writing undefined interaction correlations; all completed fit bundles hash-verified and reused, unavailable values saved as null; no numeric or guard change.',
        jobs=[load(RUN/f'audit/gate_{s}.json') for s in stages()],hosts=pd.DataFrame(tasks).groupby('host').size().astype(int).to_dict(),
        original_model_pointer_unchanged=True,no_reference_replacement=True,point_model_reuse='18 original83/74seedmodels; all selected point and bootstrap pipelines newly fitted',
        limits=['No independent confirmation','Exploratory exposed2023-2025','2025H24 unsupported','PROVISIONAL original financial versions','Strict function guards preserve model evidence, not true interactions'],
        output_hashes={p.name:sha(p) for p in (RUN/'reports').iterdir() if p.is_file()})
    write(RUN/'audit/execution_receipt.json',receipt);write(PUBLIC/'execution_receipt.json',receipt)
    audit=PUBLIC/'audit';audit.mkdir(exist_ok=True)
    for name in ['source_hashes.json','point_support.json','code_snapshot.json','code_snapshot_fixed.json','recovery_plan.json','gate_pilot_recovered.json','gate_batch.json','selection_decisions.json','failed_pilot_job2607.json','pilot_numerical_gate.json','environment.json','point_reproduction.json']:
        if (RUN/'audit'/name).exists():shutil.copy2(RUN/'audit'/name,audit/name)
    for pattern in ['pytest*.log','failed_pilot*.log']:
        for p in (RUN/'audit').glob(pattern):shutil.copy2(p,audit/p.name)
    print(json.dumps({k:receipt[k] for k in ['pipeline_tasks','new_raw_forests','resampled_searches','candidate_comparisons','verified_models','maximum_reconstruction_error','hosts']}),flush=True)

if __name__=='__main__':main()
