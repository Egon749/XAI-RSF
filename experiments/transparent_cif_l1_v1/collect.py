"""Collect complete conditional contrasts and all model diagnostics, no fit."""
from pathlib import Path
import json,sys
import numpy as np,pandas as pd
from crforest.transparent import file_sha256 as sha,write_json
from experiments.transparent_cif_l1_v1.orchestrate import gate,MODELS

def main(run):
 out=run/'reports';chunks=[dict(np.load(p/'bootstrap.npz')) for p in gate(run,'bootstrap_recovery' if (run/'audit/bootstrap_recovery_plan.json').exists() else 'bootstrap',10)];ids=np.concatenate([x['replicate_ids'] for x in chunks]);assert np.array_equal(np.sort(ids),np.arange(1000));first=chunks[0]
 for x in chunks:assert np.array_equal(x['names'],first['names']) and np.allclose(x['point'],first['point'],rtol=0,atol=0,equal_nan=True)
 reps=np.concatenate([x['replicates'] for x in chunks])[np.argsort(ids)];point=dict(zip(first['names'],first['point']));index={k:i for i,k in enumerate(first['names'])};errors=[]
 for r in pd.read_csv(out/'metrics_by_year_seed.csv').to_dict('records'):
  for metric in ['auc','ap','ipcw_brier']:
   key=f"{r['scope']}|{r['model']}|{r['seed']}|{r['weighting']}|{metric}";errors.append(abs(point[key]-r[metric]))
 w=pd.read_csv(out/'warning_utility.csv');w=w[w.scope.isin(['2023','2024','pooled'])]
 for r in w.to_dict('records'):
  prefix=f"{r['scope']}|{r['model']}|{r['seed']}|Top{int(r['top_fraction']*100)}|"
  for field in ['event_coverage','company_hit_fraction','detected_events','independent_events']:errors.append(abs(point[prefix+field]-r[field]))
  for mo in [3,6]:
   for field in ['detected','opportunity','coverage']:errors.append(abs(point[prefix+f'lead{mo}m_{field}']-r[f'announcement_{field}_{mo}m']))
 assert max(errors)<1e-10
 contrasts=json.loads((run/'input/inference_protocol.json').read_text())['contrasts'];rows=[]
 for name,(lhs,rhs) in contrasts.items():
  for key in first['names']:
   scope,model,seed,weighting,metric=key.split('|')
   if model!=lhs:continue
   other='|'.join([scope,rhs,seed,weighting,metric]);a=reps[:,index[key]]-reps[:,index[other]];ok=np.isfinite(a);lo,hi=np.quantile(a[ok],[.025,.975]);rows.append(dict(contrast=name,scope=scope,seed=int(seed),weighting_or_budget=weighting,metric=metric,point_difference=point[key]-point[other],conditional_ci_lower=lo,conditional_ci_upper=hi,valid_replicates=int(ok.sum()),multiplicity='unadjusted exploratory'))
 pd.DataFrame(rows).to_csv(out/'paired_differences.csv',index=False);write_json(run/'audit/bootstrap_check.json',dict(replicates=1000,all_ids_unique=True,point_max_error=max(errors),min_valid=min(x['valid_replicates'] for x in rows),fits=0,all_models_scopes_seeds_common_draws=True))
 native=[]
 for p in gate(run,'native_audit',24):native.append(json.loads((p/'native_audit.json').read_text()))
 pd.DataFrame(native).to_csv(out/'native_complexity.csv',index=False)
 frozen=json.loads((run/'audit/fits_frozen.json').read_text());pairs=[];weights=[];ebm=[];banks=[];instrument=[]
 from crforest import CompetingRiskForest
 from crforest.transparent import TransparentCIF
 for seed,path in frozen['seed_paths'].items():
  fit=Path(path);pa=json.loads((fit/'pair_candidate_audit_PRIVATE.json').read_text());logs=pa['candidate_log_PRIVATE'];calls=json.loads((fit.parent/'calls.json').read_text());instrument.extend([dict(seed=int(seed),**x) for x in calls]);b=CompetingRiskForest.load(fit/'base_forest_PRIVATE.json');x=CompetingRiskForest.load(fit/'ordinary_extra_forest_PRIVATE.json');p=CompetingRiskForest.load(fit/'pair_forest_PRIVATE.json')
  assert b.trees_==x.trees_[:len(b.trees_)]==p.trees_[:len(b.trees_)] and len(x.trees_)==len(p.trees_)
  roles=np.load(fit/'roles_PRIVATE.npz');refroles=np.load(Path(next(iter(frozen['seed_paths'].values())))/'roles_PRIVATE.npz')
  for k in roles.files:np.testing.assert_array_equal(roles[k],refroles[k])
  pairs.append(dict(seed=int(seed),candidate_pairs=len(logs),evaluated_partitions=sum(v['evaluated_partitions'] for v in logs),supported_partitions=sum(v['supported_partitions'] for v in logs),unsupported_pairs=sum(v['supported_partitions']==0 for v in logs),selected_pairs=pa['added_trees'],constructed_candidate_trees=sum(v['score'] is not None and v['score']>0 for v in logs),max_allowed=16,score_min=min([v['score'] for v in logs if v['selected']],default=np.nan),score_max=max([v['score'] for v in logs if v['selected']],default=np.nan),base_trees_preserved=True,ordinary_tree_count_matched=True,pair_search_seconds=sum(v['seconds'] for v in calls if v['kind']=='pair_search'),ordinary_extension_seconds=sum(v['seconds'] for v in calls if v['kind']=='ordinary_extension')))
  for name,f in [('base',b),('ordinary_extra',x),('pair',p)]:
   r=json.loads((fit/f'{name}_weight_receipt.json').read_text());m=TransparentCIF.load(fit/(name+'_weighted'));e=TransparentCIF.load(fit/(name+'_equal'))
   assert len(m.weights)==len(e.weights)==len(f.trees_) and np.all(e.weights==1/len(f.trees_))
   weights.append(dict(seed=int(seed),bank=name,**{k:v for k,v in r.items() if not isinstance(v,(list,dict))},minimum_weight=float(m.weights.min()),constraint_residual=float(abs(m.weights.sum()-1))))
   banks.append(dict(seed=int(seed),bank=name,trees=len(f.trees_),candidate_evaluations=f.candidate_evaluations_,model_sha256=sha(fit/f'{name}_forest_PRIVATE.json')))
  for name in ['ebm_pair0','ebm_pair10']:
   r=json.loads((fit/name/'selection_receipt.json').read_text())
   for candidate in r['candidates']:ebm.append(dict(seed=int(seed),model=name,selected_rounds=r['selected_rounds'],**candidate))
 for name,data in [('candidate_pair_audit_summary',pairs),('weight_diagnostics',weights),('bank_diagnostics',banks),('ebm_selection',ebm),('real_execution_calls',instrument)]:pd.DataFrame(data).to_csv(out/(name+'.csv'),index=False)

 provenance=pd.read_csv(out/'model_provenance.csv');extra=[]
 for seed,path in frozen['seed_paths'].items():
  fit=Path(path)
  for item in json.loads((fit/'model_index.json').read_text()):
   row=dict(model=item['name'],seed=int(seed),training_code_sha=json.loads((run/'audit/code_freeze.json').read_text())['training_sha'])
   if item.get('kind')=='endpoint_ebm':row['model_sha256']=sha(fit/item['path']/'model_PRIVATE.pkl')
   else:row.update(model_sha256=sha(fit/item['path']/'manifest_PRIVATE.json'),bank_sha256=sha(fit/f"{item['bank']}_forest_PRIVATE.json"),functions_sha256=sha(fit/item['path']/'functions_PRIVATE.npz'))
   extra.append(row)
 provenance=provenance.merge(pd.DataFrame(extra),on=['model','seed'],how='left')
 m2=Path('/home/oliver/MainDiskData/label_revision_sensitivity_v1_20261028T151804Z/handoff_L1/M2_PRIVATE.npz')
 provenance=pd.concat([provenance,pd.DataFrame([dict(model='M2_allsector_reference',seed=s,reference=True,prediction_sha256=sha(m2),path=str(m2)) for s in [42,101,202]])],ignore_index=True)
 provenance.to_csv(out/'model_provenance.csv',index=False)
 demo=gate(run,'demo',1)[0]/'demo';s=json.loads((demo/'results.json').read_text());pd.DataFrame(s['records']).to_csv(out/'simulation_signal_recovery.csv',index=False)
 pd.DataFrame(s['records']).groupby(['bank','aggregation'])[['main_amplitude_ratio','pair_amplitude_ratio','uniform_grid_mse','effective_trees']].mean().to_csv(out/'simulation_means.csv')
 print(pd.DataFrame(rows).query("seed==-1 and scope=='pooled' and contrast=='PRIMARY_pair_weighted_effect' and weighting_or_budget in ['record_equal','Top20']").to_string(index=False))
if __name__=='__main__':main(Path(sys.argv[1]))
