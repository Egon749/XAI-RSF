"""Complete native function export and whole-cohort all-grid independent verification."""
from pathlib import Path
from dataclasses import replace
import copy,json,shutil,time
import numpy as np,pandas as pd
from crforest import CompetingRiskForest
from crforest.transparent import TransparentCIF,tree_predictions,write_json,file_sha256 as sha
SOURCE=Path('/home/oliver/MainDiskData/label_revision_sensitivity_v1_20261028T151804Z/handoff_L1')
def main(run,out,name,seed):
 frozen=json.loads((run/'audit/fits_frozen.json').read_text());seeds=[42,101,202] if seed==-1 else [seed];models=[];banks=[]
 for s in seeds:
  fit=Path(frozen['seed_paths'][str(s)]);m=TransparentCIF.load(fit/name);bank_name=name.rsplit('_',1)[0];f=CompetingRiskForest.load(fit/f'{bank_name}_forest_PRIVATE.json');models.append(m);banks.append(f)
 train=dict(np.load(SOURCE/'final_train_PRIVATE.npz'));ev=dict(np.load(SOURCE/'final_eval_PRIVATE.npz'));meta=pd.read_parquet(SOURCE/'final_train_metadata_PRIVATE.parquet')
 roles=np.load(Path(frozen['seed_paths'][str(seeds[0])])/'roles_PRIVATE.npz')['structure'];X=train['X'][roles];dd=meta.iloc[roles]
 if seed==-1:
  f=copy.copy(banks[0]);f.trees_=[t for b in banks for t in b.trees_];f.config=replace(f.config,n_trees=len(f.trees_));w=np.concatenate([m.weights/3 for m in models]);m=TransparentCIF.from_forest(f,X,w,models[0].feature_names,models[0].cause_names,{'status':'FIXED_MEAN_OF_THREE_SEEDS','learned_again':False})
 else:m=models[0];f=banks[0]
 m.save(out/'deployment');loaded=TransparentCIF.load(out/'deployment');grid=np.unique(np.r_[f.grid_,365.,366.]);assert (grid>0).all();assert np.max(abs(loaded.predict(ev['X'][:128],[0.])))==0
 tables=loaded.complete_tables(grid,X);arrays=[np.asarray(t['values']) for t in tables['functions']];groups=np.unique(train['groups'][roles],return_inverse=True)[1];events=train['event'][roles];eid=np.unique(dd.event_id.fillna('').astype(str),return_inverse=True)[1]
 low=zero=missingcells=0;company_zero=cause1_zero=cause2_zero=0
 for term in tables['functions']:
  key=tuple(term['indices']);shape=tuple(term['shape']);bins=[]
  for j in key:
   cut=loaded.expansion.cuts(key,j);b=np.searchsorted(cut,X[:,j],side='left');b[np.isnan(X[:,j])]=len(cut)+1;bins.append(b)
  flat=np.ravel_multi_index(tuple(bins),shape);n=int(np.prod(shape));support=np.asarray(term['reference_joint_record_support']);assert support.sum()==len(X)
  unique=np.unique(np.column_stack([flat,groups]),axis=0);cs=np.bincount(unique[:,0],minlength=n).reshape(shape);term['reference_joint_company_support']=cs.tolist();company_zero+=int((cs==0).sum())
  for cause in [1,2]:
   mask=events==cause;u=np.unique(np.column_stack([flat[mask],eid[mask]]),axis=0);cc=np.bincount(u[:,0],minlength=n).reshape(shape);term[f'cause{cause}_independent_event_support']=cc.tolist()
   if cause==1:cause1_zero+=int((cc==0).sum())
   else:cause2_zero+=int((cc==0).sum())
  low+=int((support<10).sum());zero+=int((support==0).sum());missingcells+=n-int(np.prod(np.array(shape)-1))
 write_json(out/'complete_tables_PRIVATE.json',tables);shutil.copy2(SOURCE/'features.json',out/'feature_contract_PRIVATE.json')
 tree_error=component_error=reload_error=0.;minimum=1.;maxsum=0.;mindiff=0.;scalars=0
 for start in range(0,len(ev['X']),64):
  xx=ev['X'][start:start+64];h=np.broadcast_to(grid,(len(xx),len(grid)));direct=tree_predictions(f,xx,h)@loaded.weights;pred=loaded.predict(xx,grid);reload_error=max(reload_error,float(abs(pred-m.predict(xx,grid)).max()))
  total=np.broadcast_to(np.asarray(tables['baseline']),pred.shape).copy()
  for term,values in zip(tables['functions'],arrays):
   ix=[]
   for j in term['indices']:
    cut=loaded.expansion.cuts(tuple(term['indices']),j);b=np.searchsorted(cut,xx[:,j],side='left');b[np.isnan(xx[:,j])]=len(cut)+1;ix.append(b)
   total+=values[tuple(ix)]
  tree_error=max(tree_error,float(abs(direct-pred).max()));component_error=max(component_error,float(abs(total-pred).max()));minimum=min(minimum,float(pred.min()));maxsum=max(maxsum,float(pred.sum(2).max()));mindiff=min(mindiff,float(np.diff(pred,axis=1).min(initial=0)));scalars+=pred.size
 assert max(tree_error,component_error,reload_error)<1e-10 and minimum>=-1e-10 and maxsum<=1+1e-10 and mindiff>=-1e-10
 h=ev['horizon'][:,None];end=loaded.predict_horizons(ev['X'],h);expected=np.mean([m.predict_horizons(ev['X'],h) for m in models],axis=0);assert abs(end-expected).max()<1e-10
 np.savez_compressed(out/'expected_PRIVATE.npz',sample_ids=ev['sample_ids'],cif=end)
 cases=loaded.explain(ev['X'][np.linspace(0,len(ev['X'])-1,6,dtype=int)],[90,180,365,366])
 def plain(x):
  if isinstance(x,np.ndarray):return x.tolist()
  if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
  if isinstance(x,list):return [plain(v) for v in x]
  return x
 write_json(out/'complete_cases_PRIVATE.json',plain(cases));cp=loaded.expansion.complexity(X);write_json(out/'complexity_PRIVATE.json',cp)
 write_json(out/'native_audit.json',dict(model=name,seed=seed,trees=len(loaded.weights),effective_trees=float(1/(loaded.weights@loaded.weights)),largest_weight=float(loaded.weights.max()),weight_sum_residual=float(abs(loaded.weights.sum()-1)),min_weight=float(loaded.weights.min()),main_functions=cp['main_functions'],interaction_functions=cp['interaction_functions'],retained_main_functions=len(loaded.expansion.mains),retained_interaction_functions=len(loaded.expansion.pairs),spatial_cells=cp['spatial_table_cells'],native_time_scalar_cells=cp['scalar_table_cells'],exported_boundary_scalar_cells=tables['spatial_cells']*len(grid)*2,factored_coefficients=cp['factored_leaf_coefficients'],low_support_cell_fraction=low/tables['spatial_cells'],zero_support_cell_fraction=zero/tables['spatial_cells'],company_zero_cells=company_zero,target_event_zero_cells=cause1_zero,competing_event_zero_cells=cause2_zero,missing_cells=missingcells,full_cohort_records=len(ev['X']),grid_times=len(grid),scalar_checks=scalars,weighted_tree_max_error=tree_error,explicit_all_function_max_error=component_error,reload_error=reload_error,minimum_cif=minimum,maximum_cause_sum=maxsum,minimum_time_increment=mindiff,function_file_bytes=(out/'deployment/functions_PRIVATE.npz').stat().st_size,complete_table_file_bytes=(out/'complete_tables_PRIVATE.json').stat().st_size,function_sha256=sha(out/'deployment/functions_PRIVATE.npz')))
