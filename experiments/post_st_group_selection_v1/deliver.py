"""Verify every completed task and recompute fixed outer comparisons."""
import json,shutil,subprocess,datetime
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from crforest import CIFExpansion
from .run import RUN,REPO,PUBLIC,PARENT,SEARCH
from experiments.post_st_feature_combination_search_v1.analyze import model_specs,seed_folders
from experiments.post_st_dropcolumn_selection_v1.common import arrays,load,write,sha,TIMES,metrics,evaluation_context,horizon_predictions

def main():
    assert (RUN/'audit/selection_decisions.json').exists()
    for p,h in load(RUN/'audit/source_hashes.json').items():assert sha(Path(p))==h
    records=sum([load(RUN/f'audit/registry_{s}.json') for s in ['pilot','batch']],[])
    assert {(r['year'],r['repeat']) for r in records}=={(y,i) for y in [2023,2024,2025] for i in range(50)}
    tasks=[];checks=[];hashes=[];freq=[];stability=[];groups=[];keeps=[]
    parent,specs=model_specs(SEARCH)
    for r in records:
        out=Path(r['output']);s=load(out/'summary.json')
        for rel,h in load(out/'hashes.json').items():
            assert sha(out/rel)==h,(out,rel)
            hashes.append(dict(year=r['year'],repeat=r['repeat'],relative=rel,sha256=h))
        assert s['keep']==list(range(83)) and not s['supervised_search_executed'] and not s['gate']['passed']
        tasks.append(dict(year=s['year'],repeat=s['repeat'],host=s['host'],seconds=s['seconds'],peak_rss_MiB=s['peak_rss_MiB'],new_forests=s['new_forests'],
            selection_status=s['selection_status'],candidate_count=len(s['candidates']),groups=len(s['groups']),
            discovery_targets=s['gate']['discovery']['targets'],discovery_pass=s['gate']['discovery']['passed'],confirmation_pass=s['gate']['passed']))
        for g in s['groups']:groups.append(dict(year=s['year'],repeat=s['repeat'],members=json.dumps(g['members']),one=json.dumps(g['one']),two=json.dumps(g['two'])))
        for p in (out/'final').rglob('verification.json'):
            v=load(p);assert v['max_path_variables']<=2 and v['path_violations']==0
            assert max(v[k] for k in ['full_reconstruction','explicit_component_sum','function_reload','forest_reload'])<=1e-10
            assert v['probability_min']>=-1e-12 and v['probability_sum_max']<=1+1e-10 and v['minimum_increment']>=-1e-12
            checks.append(dict(year=s['year'],repeat=s['repeat'],part=str(p.relative_to(out)),**v))
    pd.DataFrame(tasks).to_csv(RUN/'reports/PIPELINE_STABILITY_RUNS.csv',index=False)
    pd.DataFrame(groups).to_csv(RUN/'reports/RESAMPLED_GROUPS.csv',index=False)
    pd.DataFrame(checks).to_csv(RUN/'reports/NUMERICAL_VALIDATION.csv.gz',index=False)
    pd.DataFrame(hashes).to_csv(RUN/'reports/MODEL_HASHES.csv.gz',index=False)
    features=load(PARENT/'input/features.json')
    for y in [2023,2024,2025]:
        for f in features:
            freq.append(dict(year=y,**f,process_replicates=50,supervised_search_replicates=0,selection_frequency=np.nan,
                fallback_inclusion_frequency=1.,status='NOT_ESTIMABLE_NO_HONEST_FORWARD_CONFIRMATION'))
    pd.DataFrame(freq).to_csv(RUN/'reports/FEATURE_SELECTION_STABILITY.csv',index=False)
    # Common training reference, same normalization, absent terms count as zero.
    for y in [2023,2024,2025]:
        data=arrays(PARENT/f'input/outer/{y}/train_PRIVATE.npz'); X=data['X']; n=len(X)
        moments={};prediction_sum=np.zeros((n,2,2));prediction_sq=np.zeros_like(prediction_sum)
        ix=np.stack([np.searchsorted(TIMES,data['h12']),np.searchsorted(TIMES,data['h24'])],axis=1)
        rr=np.arange(n)[:,None]
        for r in [z for z in records if z['year']==y]:
            ex=CIFExpansion.load(Path(r['output'])/'final/ensemble/functions_PRIVATE.npz')
            pred=ex.predict(X,TIMES)[rr,ix];prediction_sum+=pred;prediction_sq+=pred*pred
            for key in [(j,) for j in ex.mains]+ex.pairs:
                v=ex.component(X,key,TIMES)[rr,ix]
                if key not in moments:moments[key]=[np.zeros_like(v),np.zeros_like(v),0]
                moments[key][0]+=v;moments[key][1]+=v*v;moments[key][2]+=1
        for key,(s,q,count) in moments.items():
            for j,h in enumerate([12,24]):
                for c in [1,2]:
                    mu=s[:,j,c-1]/50;var=np.maximum(q[:,j,c-1]/50-mu*mu,0)
                    stability.append(dict(year=y,order=len(key),features=json.dumps(list(map(int,key))),names=' × '.join(features[k]['label'] for k in key),
                        horizon=h,cause=c,structural_presence=count/50,mean_abs_mean_function=float(abs(mu).mean()),rms_function_sd=float(np.sqrt(var.mean())),
                        scope='FALLBACK_FOREST_STABILITY_NOT_FEATURE_SELECTION_STABILITY'))
        for j,h in enumerate([12,24]):
            for c in [1,2]:
                mu=prediction_sum[:,j,c-1]/50;var=np.maximum(prediction_sq[:,j,c-1]/50-mu*mu,0)
                keeps.append(dict(year=y,horizon=h,cause=c,mean_prediction=float(mu.mean()),rms_prediction_sd=float(np.sqrt(var.mean()))))
        print('Function stability complete',y,flush=True)
    pd.DataFrame(stability).to_csv(RUN/'reports/FUNCTION_STABILITY.csv',index=False)
    pd.DataFrame(stability).query('order==2').to_csv(RUN/'reports/INTERACTION_STABILITY.csv',index=False)
    pd.DataFrame(keeps).to_csv(RUN/'reports/PREDICTION_STABILITY.csv',index=False)
    labels={};predictions={};rows=[];complexity=[];reused=[]
    for y in [2023,2024,2025]:
        lab=arrays(PARENT/f'evaluation_input/{y}_test_PRIVATE.npz');labels[y]=lab
        train=arrays(PARENT/f'input/outer/{y}/train_PRIVATE.npz');assert not set(train['groups'])&set(lab['groups'])
        for role,base in [('full83','full83'),('legacy74','legacy74'),('selected_pipeline','full83')]:
            spec=specs[(y,base)];folder=Path(spec['path']);hh=load(folder/'hashes.json')
            for rel in ['ensemble/functions_PRIVATE.npz','ensemble/predictions_PRIVATE.npz']:
                assert sha(folder/rel)==hh[rel]
            pp=arrays(folder/'ensemble/predictions_PRIVATE.npz');np.testing.assert_array_equal(pp['sample_ids'],lab['sample_ids'])
            predictions[(y,role)]=pp['cif']
            for seed,p in [('ensemble',folder/'ensemble/predictions_PRIVATE.npz')]+list(zip(['42','101','202'],[z/'predictions_PRIVATE.npz' for z in seed_folders(spec)])):
                z=arrays(p);np.testing.assert_array_equal(z['sample_ids'],lab['sample_ids'])
                for h in [12,24]:
                    risk=horizon_predictions(z['cif'],TIMES,lab[f'h{h}'])
                    for c in [1,2]:
                        m=metrics(lab,risk[:,c-1],h,c)
                        if m['supported']:
                            yy,w,known,_,_=evaluation_context(lab,h,c)
                            if yy[known].any() and (~yy[known]).any():assert abs(m['auc']-roc_auc_score(yy[known],risk[known,c-1],sample_weight=w[known]))<1e-12
                        rows.append(dict(year=y,model=role,seed=seed,horizon=h,cause=c,**m))
            ex=CIFExpansion.load(folder/'ensemble/functions_PRIVATE.npz');cc=ex.complexity(train['X'][:,spec['keep']],support_min=10)
            complexity.append(dict(year=y,model=role,features=len(spec['keep']),**{k:v for k,v in cc.items() if k!='functions'}))
            reused.append(dict(year=y,model=role,source_path=str(folder),prediction_sha256=sha(folder/'ensemble/predictions_PRIVATE.npz'),functions_sha256=sha(folder/'ensemble/functions_PRIVATE.npz')))
    for h in [12,24]:
        years=[y for y in labels if evaluation_context(labels[y],h,1)[3]]
        lab={k:np.concatenate([labels[y][k] for y in years]) for k in labels[years[0]]}
        for role in ['full83','legacy74','selected_pipeline']:
            risk=np.concatenate([horizon_predictions(predictions[(y,role)],TIMES,labels[y][f'h{h}']) for y in years])
            for c in [1,2]:
                m=metrics(lab,risk[:,c-1],h,c)
                annual=[z for z in rows if z['year'] in years and z['model']==role and z['seed']=='ensemble' and z['horizon']==h and z['cause']==c]
                m.update(checked=sum(z['checked'] for z in annual),captured=sum(z['captured'] for z in annual))
                m['coverage']=m['captured']/m['target_events'] if m['target_events'] else np.nan
                rows.append(dict(year='pooled',cohorts=','.join(map(str,years)),model=role,seed='ensemble',horizon=h,cause=c,**m))
    frame=pd.DataFrame(rows);frame.to_csv(RUN/'reports/MODEL_COMPARISON.csv',index=False)
    pd.DataFrame(complexity).to_csv(RUN/'reports/EXPLANATION_COMPLEXITY.csv',index=False)
    pd.DataFrame(reused).to_csv(RUN/'reports/REUSED_MODELS.csv',index=False)
    selected=[]
    for y in [2023,2024,2025]:
        for f in features:selected.append(dict(year=y,**f,selected=True,reason='FALLBACK83_NOT_IMPORTANCE_CLAIM'))
    pd.DataFrame(selected).to_csv(RUN/'reports/SELECTED_FEATURES.csv',index=False)
    diagnostics=pd.read_csv(RUN/'reports/INTERACTION_PROTECTION_DIAGNOSTICS.csv')
    sub=diagnostics[diagnostics.reason!='NEGLIGIBLE_BASE_COMPONENT']
    gp=sub.groupby(['year','cohort','keep']).agg(checks=('passed','size'),passes=('passed','sum')).reset_index()
    gp['all_function_checks_pass']=gp.checks==gp.passes;gp.to_csv(RUN/'reports/INTERACTION_GUARD_SUMMARY.csv',index=False)
    maximum=max(max(v[k] for k in ['full_reconstruction','explicit_component_sum','forest_reload','function_reload']) for v in checks)
    receipt=dict(status='COMPLETE_SUPPORTED_SCOPE_WITH_FALLBACK',training_code_commit=load(RUN/'audit/code_snapshot.json')['commit'],
        analysis_source_sha256=sha(Path(__file__)),frozen_protocol_sha256=sha(RUN/'input/protocol.json'),
        point_decisions=load(RUN/'audit/selection_decisions.json'),pipeline_tasks=len(records),new_raw_forests=sum(r['new_forests'] for r in records),
        derived_shrunk_forests=450,merged_ensembles=150,verified_models=len(checks),maximum_reconstruction_error=maximum,
        supervised_resampled_searches=0,selection_stability='NOT_ESTIMABLE_SUPPORT_GATES_CLOSED',
        point_forecasts='18 saved seed forests reused for83/74; selected duplicates83 by exact identity, no new point fits',
        failed_fits=0,jobs=[load(RUN/f'audit/gate_{s}.json') for s in ['pilot','batch']],
        hosts={k:int(v) for k,v in pd.DataFrame(tasks).groupby('host').size().items()},original_model_pointer_unchanged=True,
        unfinished_scientific_questions=['No supported independent confirmation of a reduced set','No executed supervised-search resampling; frequencies NA','Interaction protection predictive benefit unverified'],
        data_limit='PROVISIONAL original financial disclosure versions;2023-2025 exposed exploratory;2025H24 unsupported',
        output_hashes={p.name:sha(p) for p in (RUN/'reports').iterdir() if p.is_file()})
    write(RUN/'audit/execution_receipt.json',receipt)
    for p in (RUN/'reports').iterdir():shutil.copy2(p,PUBLIC/p.name)
    shutil.copy2(RUN/'audit/execution_receipt.json',PUBLIC/'execution_receipt.json')
    audit=PUBLIC/'audit';audit.mkdir(exist_ok=True)
    for name in ['source_hashes.json','point_gates.json','code_snapshot.json','gate_pilot.json','gate_batch.json']:
        shutil.copy2(RUN/'audit'/name,audit/name)
    for p in (RUN/'audit').glob('pytest*.log'):shutil.copy2(p,audit/p.name)
    print(json.dumps({k:receipt[k] for k in ['pipeline_tasks','new_raw_forests','verified_models','maximum_reconstruction_error','hosts']}),flush=True)

if __name__=='__main__':main()
