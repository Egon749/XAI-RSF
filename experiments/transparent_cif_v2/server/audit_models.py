"""Complete private function/support audit and public aggregate diagnostics."""
from pathlib import Path
from itertools import combinations
import argparse,json,pickle
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from crforest.transparent import TransparentCIF,file_sha256 as sha,write_json

def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,list):return [plain(v) for v in x]
    return x

def support_summary(tables,model,seed):
    row=dict(model=model,seed=seed,spatial_cells=tables['spatial_cells'])
    for kind in ['reference_joint_company_support','cause1_independent_event_support','cause2_independent_event_support']:
        values=np.concatenate([np.asarray(f[kind]).ravel() for f in tables['functions']])
        row[kind+'_zero_fraction']=float(np.mean(values==0));row[kind+'_lt3_fraction']=float(np.mean(values<3));row[kind+'_median']=float(np.median(values))
    return row

def main(a):
    run=a.run;dest=run/'model_audit';dest.mkdir();public=run/'reports';src=run/'input'
    with np.load(src/'final_train_PRIVATE.npz') as z:train={k:z[k] for k in z.files}
    with np.load(src/'final_eval_PRIVATE.npz') as z:ev={k:z[k] for k in z.files}
    meta=pd.read_parquet(src/'final_train_metadata_PRIVATE.parquet');evalmeta=pd.read_parquet(src/'final_eval_metadata_PRIVATE.parquet')
    with np.load(src/'references_PRIVATE.npz') as z:anchors=z['anchors']
    support_rows=[];complexities=[];weights=[];pairs=[];ebms=[];models={};modelhash=[];seedcost=[]
    for task,seed in enumerate([42,101,202]):
        out=run/'results/formal'/str(a.job)/f'task-{task}/output';fit=out/'fit';role=np.load(fit/'roles_PRIVATE.npz')['structure'];X=train['X'][role];dd=meta.iloc[role]
        groups=np.unique(train['groups'][role],return_inverse=True)[1];events=train['event'][role]
        eventids=np.unique(dd.event_id.fillna('').astype(str),return_inverse=True)[1]
        costs=json.loads((out/'summary.json').read_text());costs['training_seconds']=json.loads((fit/'execution_receipt.json').read_text())['seconds'];seedcost.append(costs)
        pa=json.loads((fit/'pair_candidate_audit_PRIVATE.json').read_text());cl=pa['candidate_log_PRIVATE']
        pairs.append(dict(seed=seed,original_trees=pa['original_trees'],added_trees=pa['added_trees'],candidate_pairs_present_before=sum(x['trees_with_pair_path']>0 for x in pa['pair_coverage_before']),candidate_pairs_present_after=sum(x['trees_with_pair_path']>0 for x in pa['pair_coverage_after']),actual_pairs_before=len(TransparentCIF.load(fit/'base_equal').expansion.pairs),actual_pairs_after=len(TransparentCIF.load(fit/'pair_equal').expansion.pairs),candidate_pairs=len(cl),evaluated_partitions=sum(c['evaluated_partitions'] for c in cl),supported_partitions=sum(c['supported_partitions'] for c in cl),selected_pairs=sum(c['selected'] for c in cl),no_supported_candidate_pairs=sum(c['supported_partitions']==0 for c in cl),selected_score_min=min(c['score'] for c in cl if c['selected']),selected_score_max=max(c['score'] for c in cl if c['selected'])))
        for bank in ['base','ordinary_extra','pair']:
            wr=json.loads((fit/f'{bank}_weight_receipt.json').read_text());weights.append(dict(seed=seed,bank=bank,**{k:v for k,v in wr.items() if not isinstance(v,(dict,list))}))
        for it in json.loads((fit/'model_index.json').read_text()):
            folder=fit/it['path'];modelhash.append(dict(model=it['name'],seed=seed,**{k:v for k,v in it.items() if k.endswith('sha256')}))
            if it.get('kind')=='endpoint_ebm':
                er=json.loads((folder/'selection_receipt.json').read_text())
                with (folder/'model_PRIVATE.pkl').open('rb') as f:em=pickle.load(f)
                for c in er['candidates']:ebms.append(dict(model=it['name'],seed=seed,selected_rounds=er['selected_rounds'],**c))
                cells=sum(v.size for v in em.term_scores_)
                complexities.append(dict(model=it['name'],seed=seed,native_scale='log_odds_then_logistic',active_main_functions=sum(len(t)==1 and np.any(v!=0) for t,v in zip(em.term_features_,em.term_scores_)),active_pair_functions=sum(len(t)==2 and np.any(v!=0) for t,v in zip(em.term_features_,em.term_scores_)),retained_main_functions=sum(len(t)==1 for t in em.term_features_),retained_pair_functions=sum(len(t)==2 for t in em.term_features_),spatial_cells=cells,native_time_scalar_cells=cells,missing_cells_explicit=True))
                continue
            modelhash[-1]['functions_sha256']=sha(folder/'functions_PRIVATE.npz');modelhash[-1]['original_bank_sha256']=sha(fit/f'{it["bank"]}_forest_PRIVATE.json')
            m=TransparentCIF.load(folder);models[it['name'],seed]=m;cp=json.loads((folder/'complexity_PRIVATE.json').read_text());tb=json.loads((folder/'complete_tables_PRIVATE.json').read_text());missingcells=0
            # Augment ALL cells with company and cause-specific independent event support.
            for term in tb['functions']:
                key=tuple(term['indices']);shape=tuple(term['shape']);bins=[]
                for j in key:
                    cc=m.expansion.cuts(key,j);b=np.searchsorted(cc,X[:,j],side='left');b[np.isnan(X[:,j])]=len(cc)+1;bins.append(b)
                flat=np.ravel_multi_index(tuple(bins),shape);n=int(np.prod(shape))
                unique=np.unique(np.column_stack([flat,groups]),axis=0);term['reference_joint_company_support']=np.bincount(unique[:,0],minlength=n).reshape(shape).tolist()
                for cause in [1,2]:
                    mask=events==cause;term[f'cause{cause}_record_support']=np.bincount(flat[mask],minlength=n).reshape(shape).tolist()
                    unique=np.unique(np.column_stack([flat[mask],eventids[mask]]),axis=0);term[f'cause{cause}_independent_event_support']=np.bincount(unique[:,0],minlength=n).reshape(shape).tolist()
                assert np.array_equal(np.bincount(flat,minlength=n).reshape(shape),np.asarray(term['reference_joint_record_support']))
                missingcells+=n-int(np.prod(np.array(shape)-1))
            support_rows.append(support_summary(tb,it['name'],seed))
            write_json(dest/f'{it["name"]}_{seed}_all_cells_PRIVATE.json',tb)
            # Six deterministic records, outcome-independent; full 3/6/12 calendar-month terms.
            examples=[]
            for i in np.linspace(0,len(ev['X'])-1,6,dtype=int):
                origin=pd.Timestamp(evalmeta.origin.iloc[i]);times=[(origin+pd.DateOffset(months=k)-origin).days for k in [3,6,12]]
                explanation=m.explain(ev['X'][i:i+1],times);assert explanation['max_error']<1e-10
                examples.append(dict(anonymous_id=f'case-{len(examples)+1}',sample_id=str(ev['sample_ids'][i]),origin=str(origin),times_days=times,explanation=plain(explanation)))
            write_json(dest/f'{it["name"]}_{seed}_complete_cases_PRIVATE.json',examples)
            complexities.append(dict(model=it['name'],seed=seed,native_scale='CIF_probability_identity',trees=it['trees'],effective_trees=it['effective_trees'],active_main_functions=cp['main_functions'],active_pair_functions=cp['interaction_functions'],retained_main_functions=len(m.expansion.mains),retained_pair_functions=len(m.expansion.pairs),spatial_cells=cp['spatial_table_cells'],native_time_scalar_cells=cp['scalar_table_cells'],factored_leaf_coefficients=cp['factored_leaf_coefficients'],native_grid_times=len(m.expansion.grid),low_support_cell_fraction=cp['low_support_cell_fraction'],zero_support_cell_fraction=cp['zero_support_cell_fraction'],missing_cells_explicit=True,missing_spatial_cells=missingcells,function_package_bytes=(folder/'functions_PRIVATE.npz').stat().st_size,native_error=it['native_error'],reload_error=it['reload_error']))
    stability=[]
    for name in sorted({k[0] for k in models}):
        for s1,s2 in combinations([42,101,202],2):
            aa,bb=models[name,s1],models[name,s2]
            for order in [1,2]:
                def keys(mm):return {(j,) for j in mm.expansion.mains} if order==1 else set(mm.expansion.pairs)
                ka,kb=keys(aa),keys(bb);union=sorted(ka|kb);va=np.stack([aa.expansion.component(anchors,k,[365])[:,0,0] for k in union]);vb=np.stack([bb.expansion.component(anchors,k,[365])[:,0,0] for k in union]);ia=abs(va).mean(1);ib=abs(vb).mean(1)
                activea={k for k,v in zip(union,ia) if v>1e-14};activeb={k for k,v in zip(union,ib) if v>1e-14};mask=(abs(va)>1e-12)&(abs(vb)>1e-12)
                stability.append(dict(model=name,seed_a=s1,seed_b=s2,order=order,reference='original frozen 32 evaluation anchors; same fixed structure product marginal centering',structural_jaccard=len(ka&kb)/len(ka|kb),anchor_active_jaccard=len(activea&activeb)/max(1,len(activea|activeb)),aligned_component_rmse=float(np.sqrt(np.mean((va-vb)**2))),importance_spearman=float(spearmanr(ia,ib).statistic),sign_agreement_on_common_nonzero=float(np.mean(np.sign(va[mask])==np.sign(vb[mask]))) if mask.any() else None,compared_nonzero_values=int(mask.sum())))
    for filename,records in [('cell_support_summary.csv',support_rows),('complexity.csv',complexities),('weight_diagnostics.csv',weights),('pair_search_diagnostics.csv',pairs),('ebm_selection.csv',ebms),('native_seed_stability.csv',stability),('task_costs.csv',seedcost),('model_hashes.csv',modelhash)]:pd.DataFrame(records).to_csv(public/filename,index=False)
    demo=json.loads((run/'results/formal'/str(a.job)/'task-3/output/demo/results.json').read_text());pd.DataFrame(demo['records']).to_csv(public/'synthetic_recovery.csv',index=False)
    write_json(dest/'audit_receipt.json',dict(status='PASS',native_models=18,complete_cell_support_files=18,complete_case_files=18,calendar_case_months=[3,6,12],complete_native_time_functions='in each lossless functions_PRIVATE.npz',fixed_anchor_rows=len(anchors),stability_scope='three seeds, fixed roles; not company bootstrap refitting',files={p.name:sha(p) for p in dest.iterdir() if p.is_file()}))
    print('Completed complete-cell support, calendar-case, complexity, convergence and seed-stability audit.')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--job',type=int,required=True);main(p.parse_args())
