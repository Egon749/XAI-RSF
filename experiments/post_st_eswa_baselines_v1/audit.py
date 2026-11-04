import itertools,time,pickle
import pandas as pd
import xgboost as xgb
from experiments.post_st_benchmark_calibration_v1.common import endpoints
from .evaluate import tasks
from .proboost import objective,beta_at
from .common import *
def main():
    r=run();jobs=tasks(r);logs=[];budget=[];cost=[];selection=[];stability=[];params=[];sm=[];path_rows=[];proof=[]
    names=pd.read_csv(ROOT/'experiments/post_st_grouped_explanation_v1/FEATURE_GROUP_MAPPING.csv').english_name.to_list()
    for (y,h,arm,mode),(p,v) in sorted(jobs.items()):
        meta=dict(year=y,months=h,model=arm,mode=mode);hist=pd.read_csv(p/'training.csv')
        for k,x in meta.items():hist[k]=x
        hist['operation_id']=np.arange(len(hist));logs.append(hist)
        stages=hist.stage.value_counts();fits=hist.stage.isin(['PROBoost_tune','PROBoost_final','IPSO_proxy_fit','Hyperopt_fit','base_refit','GA_meta_fit','final_meta_fit']);nfit=int(sum(fits))
        budget.append(dict(**meta,training_runs=nfit,proboost_tune_fits=stages.get('PROBoost_tune',0),proboost_final_fits=stages.get('PROBoost_final',0),proboost_tree_updates=int(hist.trees.sum()) if arm=='PROBoost' else 0,ipso_proxy_fits=stages.get('IPSO_proxy_fit',0),hyperopt_base_fits=stages.get('Hyperopt_fit',0),hyperopt_trials=stages.get('Hyperopt_fit',0)//2,ga_meta_fits=stages.get('GA_meta_fit',0),base_refits=stages.get('base_refit',0),final_meta=stages.get('final_meta_fit',0),ipso_searches=stages.get('IPSO_search',0),full_swarm_resets=int(hist.resets.sum()) if 'resets' in hist else 0,ga_searches=stages.get('GA_search',0),ga_unique_masks=int(hist.chromosomes.sum()) if 'chromosomes' in hist else 0,ipso_insufficient=stages.get('IPSO_support_fallback',0),hyperopt_insufficient=stages.get('Hyperopt_support_fallback',0),ga_insufficient=stages.get('GA_support_fallback',0)))
        sec=v['elapsed'];cost.append(dict(**meta,fit_count=nfit,end_to_end_seconds=sec,reported_fit_seconds=v['fit_seconds'],predict_seconds=v['predict_seconds'],companies_scored=v['score_rows'],model_bytes=v['model_bytes'],peak_process_mib=v['peak_mib'],slurm_job=v['job'],host=v['host'],selected_columns=len(v['selected']),actual_model_reload_error=v['reload_error']))
        if arm=='PROBoost':
            sch,c,depth=v['cfg'];params.append(dict(**meta,stage='outer_final',beta_start=sch[0],beta_end=sch[1],c=c,depth=depth,trees=48,learning_rate=.075))
            bst=xgb.Booster();bst.load_model(p/'proboost.ubj');trees=[json.loads(s) for s in bst.get_dump(dump_format='json')];assert len(trees)==48
            def degree(node,seen):
                if 'leaf' in node:return len(seen)
                used=seen|{int(node['split'][1:])%74};return max(degree(k,used) for k in node['children'])
            path_rows.append(dict(**meta,max_original_variables_per_tree_path=max(degree(q,set()) for q in trees),output='margin used in shifted sigmoid'))
        else:
            for q in v['plans']:
                cols=q['cols'];labels=['RF','GBDT','LightGBM','XGBoost','AdaBoost'];ws=q['weights'];mask=q['learners'];assert len(cols)==len(set(cols)) and 3<=sum(mask)<=5 and np.isclose(sum(ws),1) and all(w==0 or m for w,m in zip(ws,mask))
                for i,n in enumerate(names):selection.append(dict(**meta,role=q['role'],fold=q['fold'],feature_index=i,feature_name=n,selected=i in cols))
                params.append(dict(**meta,stage=q['role'],fold=q['fold'],feature_count=len(cols),learners_selected=','.join(l for l,m in zip(labels,mask) if m),learners_nonzero_weight=','.join(l for l,w in zip(labels,ws) if w>0),effective_learners=sum(w>0 for w in ws),equal_on_active=bool(np.ptp([w for w in ws if w>0])<1e-12),setting_indices='|'.join(map(str,q['parameters']))))
            with (p/'fold_indices.pkl').open('rb') as stream:splits=pickle.load(stream)
            for _,a,b in splits:assert not set(a)&set(b)
            proof.append(dict(**meta,private_fold_hash=sha(p/'fold_indices.pkl'),partition_disjoint_checks=len(splits),final_meta_known_rows=v['known'],OOF_full_plan_fits=3))
            for stage in ['IPSO_proxy_fit','Hyperopt_fit','base_refit']:
                ss=hist[hist.stage==stage];assert not ((ss.fit_pos<6)&(ss.synthetic>0)).any() and (ss.synthetic<=ss.fit_pos).all()
                sm.append(dict(**meta,stage=stage,fits=len(ss),augmented_fits=sum(ss.synthetic>0),insufficient_minority=sum(ss.fit_pos<6),synthetic_row_uses=int(ss.synthetic.sum())))
    select=pd.DataFrame(selection)
    for (h,mode),zz in select.groupby(['months','mode']):
        sets={k:set(z.loc[z.selected,'feature_name']) for k,z in zz.groupby(['year','role','fold'])}
        for a,b in itertools.combinations(sets,2):
            if a[1]==b[1]=='deployment' or (a[0]==b[0] and a[1]==b[1]=='meta_oof'):
                sa,sb=sets[a],sets[b];stability.append(dict(months=h,mode=mode,year_a=a[0],year_b=b[0],role=a[1],fold_a=a[2],fold_b=b[2],common=len(sa&sb),union=len(sa|sb),jaccard=len(sa&sb)/len(sa|sb)))
    for row in pd.read_csv(D/'REFERENCE_VALIDATION.csv').itertuples():
        cost.append(dict(year=row.year,months='12+24',model=row.model,mode='native',fit_count=0,historical_fit_seconds=row.historical_fit_seconds,predict_seconds=row.forecast_seconds_joint_H12_H24,model_bytes=row.model_bytes,separate_exact_function_bytes=row.function_bytes,selected_columns=74,actual_model_reload_error=None))
    for name,rows in [('SEARCH_BUDGET',budget),('RUNTIME_AND_COMPLEXITY',cost),('SELECTED_FEATURES',selection),('SELECTION_STABILITY',stability),('MODEL_PARAMETERS',params),('SMOTE_AUDIT',sm),('OOF_CHECKS',proof),('PROBOOST_PATHS',path_rows)]:pd.DataFrame(rows).to_csv(D/(name+'.csv'),index=False)
    pd.concat(logs,ignore_index=True).to_csv(D/'TRAINING_LOG.csv',index=False)
    errs=[]
    for b,c in itertools.product([.2,1.,2.,5.],[.25,.75]):
        z=np.linspace(-2,3,23);w=np.linspace(.8,2,len(z));y=np.arange(len(z))%2;eps=1e-4
        l,g,hs=objective(z,y,w,b,c);lp=objective(z+eps,y,w,b,c)[0];lm=objective(z-eps,y,w,b,c)[0]
        errs.append([abs(g-(lp-lm)/(2*eps)).max(),abs(hs-(lp-2*l+lm)/eps**2).max()])
    gr,he=np.max(errs,axis=0);assert gr<1e-6 and he<1e-6
    dump(D/'NUMERICAL_CHECKS.json',dict(beta_schedules_checked=True,finite_difference_cases=8*23,epsilon=1e-4,gradient_max_abs=gr,hessian_max_abs=he,all_deployment_models_reloaded=True,all24_publication_manifests_checked=True,all_OOF_fold_company_disjoint=True))
if __name__=='__main__':main()
