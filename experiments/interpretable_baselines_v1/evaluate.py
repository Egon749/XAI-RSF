"""Common endpoint evaluation, native complexity and paired explanation audits."""
from pathlib import Path
import argparse,json,hashlib,pickle,sys
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score,average_precision_score
from crforest import CIFExpansion,HigherOrderCIFExpansion


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    with Path(p).open('x') as f:json.dump(d,f,indent=2,allow_nan=False)
def compare(a,b):
    ia=np.abs(a).mean(0);ib=np.abs(b).mean(0);ta=set(np.argsort(-ia,kind='stable')[:10]);tb=set(np.argsort(-ib,kind='stable')[:10])
    signmask=(abs(a)>1e-10)|(abs(b)>1e-10)
    an=a/np.maximum(abs(a).sum(1,keepdims=True),1e-15);bn=b/np.maximum(abs(b).sum(1,keepdims=True),1e-15)
    cor=[np.corrcoef(u,v)[0,1] for u,v in zip(a,b) if u.std()>1e-12 and v.std()>1e-12]
    rho=float(spearmanr(ia,ib).statistic) if ia.std()>0 and ib.std()>0 else None
    return dict(rmse=float(np.sqrt(np.mean((a-b)**2))),normalized_rmse=float(np.sqrt(np.mean((an-bn)**2))),
        importance_spearman=rho,top10_jaccard=len(ta&tb)/len(ta|tb),
        sign_agreement=float(np.mean(np.sign(a[signmask])==np.sign(b[signmask]))) if signmask.any() else None,
        median_anchor_correlation=float(np.median(cor)) if cor else None)


def main(run):
    inp=run/'input';out=run/'reports';plan=json.loads((inp/'plan.json').read_text())
    assert not (out/'execution_receipt.json').exists()
    for name,digest in json.loads((run/'audit/input_hashes.json').read_text()).items():assert sha(inp/name)==digest
    for name,digest in json.loads((run/'audit/coverage_freeze.json').read_text())['input_hashes'].items():assert sha(run/'coverage_input'/name)==digest
    for stage in ['pilot','formal','coverage']:assert json.loads((run/f'audit/{stage}_gate.json').read_text())['status']=='PASS'
    rows=[];paths={}
    for p in list((run/'results/formal').glob('*/task-*/output/summary.json'))+list((run/'results/coverage').glob('*/task-*/output/summary.json')):
        d=json.loads(p.read_text());assert not d['pilot'] and d['status']=='COMPLETE';rows.append(d)
        key=(d['kind'],d['model'],d['seed'],d['replicate'],d.get('scenario'),d.get('permutations'));assert key not in paths;paths[key]=p.parent
        for name,h in json.loads((p.parent/'hashes.json').read_text()).items():assert sha(p.parent/name)==h
    assert len(rows)==261
    diag=pd.DataFrame(rows);diag.to_csv(out/'task_diagnostics.csv',index=False)
    sim=diag[diag.kind=='simulation'].dropna(axis=1,how='all').copy();assert len(sim)==210
    # Undefined pair precision/recall are missing, not a fabricated zero score.
    sim['pair_precision']=sim.pair_tp/(sim.pair_tp+sim.pair_fp).replace(0,np.nan)
    sim['pair_recall']=sim.pair_tp/(sim.pair_tp+sim.pair_fn).replace(0,np.nan)
    sim.to_csv(out/'simulation_metrics.csv',index=False)
    sm=sim.groupby(['scenario','model']).agg({k:['mean','min','max'] for k in ['f1','precision','recall','pair_precision','pair_recall','probability_coefficient_rmse','native_coefficient_rmse','uniform_cube_probability_mse','observed_distribution_probability_mse','true_attribution_rmse']})
    sm.columns=['_'.join(c) for c in sm.columns];sm.reset_index().to_csv(out/'simulation_summary.csv',index=False)
    sd=diag[diag.kind=='shap'].dropna(axis=1,how='all');sd.to_csv(out/'shap_fidelity.csv',index=False)
    shapmc=[]
    for model in plan['shap_models']:
        for n in plan['shap_permutations']:
            for sa,sb in combinations(plan['shap_seeds'],2):
                a=np.load(paths['shap',model,sa,-1,None,n]/'explanations_PRIVATE.npz')['approx'];b=np.load(paths['shap',model,sb,-1,None,n]/'explanations_PRIVATE.npz')['approx']
                shapmc.append(dict(model=model,permutations=n,seed_a=sa,seed_b=sb,**compare(a,b)))
    pd.DataFrame(shapmc).to_csv(out/'shap_computation_stability.csv',index=False)
    instability=[];background=[];native=[]
    for model in plan['bootstrap_models']:
        native_data={}
        for r in range(8):
            folder=paths['bootstrap',model,42,r,None,None]
            if model!='ebm_ga2m10':
                z=np.load(folder/'background_PRIVATE.npz');background.append(dict(model=model,replicate=r,**compare(z['fixed'],z['resampled'])))
            if model!='unrestricted':
                z=np.load(folder/'native_PRIVATE.npz');native_data[r]={str(k):z['terms'][:,i] for i,k in enumerate(z['keys'])}
        for ra,rb in combinations(range(8),2):
            if model!='ebm_ga2m10':
                a=np.load(paths['bootstrap',model,42,ra,None,None]/'explanations_PRIVATE.npz')['exact'];b=np.load(paths['bootstrap',model,42,rb,None,None]/'explanations_PRIVATE.npz')['exact']
                instability.append(dict(model=model,replicate_a=ra,replicate_b=rb,reference='same fixed 16 joint background rows',scale='CIF Shapley',**compare(a,b)))
            if model!='unrestricted':
                aa,bb=native_data[ra],native_data[rb];a,b=set(aa),set(bb);corr=[]
                pa={k for k in a if ',' in k};pb={k for k in b if ',' in k}
                for k in a&b:
                    if aa[k].std()>1e-12 and bb[k].std()>1e-12:corr.append(float(np.corrcoef(aa[k],bb[k])[0,1]))
                native.append(dict(model=model,replicate_a=ra,replicate_b=rb,scale='log_odds' if model.startswith('ebm') else 'CIF',
                    functions_a=len(a),functions_b=len(b),jaccard=len(a&b)/len(a|b),pair_jaccard=len(pa&pb)/len(pa|pb) if pa|pb else None,pairs_a=len(pa),pairs_b=len(pb),common_nonconstant=len(corr),median_correlation=float(np.median(corr)) if corr else None))
    pd.DataFrame(instability).to_csv(out/'paired_training_stability.csv',index=False);pd.DataFrame(background).to_csv(out/'background_stability.csv',index=False);pd.DataFrame(native).to_csv(out/'native_function_stability.csv',index=False)
    # Matched comparison on each identical pair of training resamples (dependent
    # pair summaries, not 28 independent draws or a bootstrap confidence interval).
    a=pd.DataFrame(instability);paired=[]
    for (ra,rb),g in a.groupby(['replicate_a','replicate_b']):
        g=g.set_index('model');paired.append(dict(replicate_a=ra,replicate_b=rb,**{k:float(g.loc['q2',k]-g.loc['unrestricted',k]) for k in ['rmse','normalized_rmse','importance_spearman','top10_jaccard','sign_agreement']}))
    pd.DataFrame(paired).to_csv(out/'paired_stability_contrasts.csv',index=False)
    sys.path.insert(0,str(inp/'legacy'));from common import endpoint_weights
    from endpoint_guard import source_consistent_endpoint_weights
    from evaluate import top_flags,warning_metrics,event_leads,lead_summary
    data=pd.read_parquet(inp/'final_eval_metadata_PRIVATE.parquet').reset_index(drop=True);preds={}
    for model in plan['baseline_models']:
        for seed in plan['seeds']:
            z=np.load(paths['baseline',model,seed,-1,None,None]/'predictions_PRIVATE.npz');np.testing.assert_array_equal(z['sample_ids'],data.sample_id);preds[model,seed]=z['risk']
    for model in ['q2','q3','q4','unrestricted']:
        for seed in plan['seeds']:
            z=np.load(inp/f'reference_{model}_{seed}_PRIVATE.npz');np.testing.assert_array_equal(z['sample_ids'],data.sample_id);preds[model,seed]=z['cif12'][:,0]
    z=np.load(inp/'M2_PRIVATE.npz');np.testing.assert_array_equal(z['sample_ids'],data.sample_id)
    for j,seed in enumerate(z['seeds']):preds['M2_allsector',int(seed)]=z['probabilities'][j]
    models=sorted({x[0] for x in preds})
    for model in models:preds[model,-1]=np.mean([preds[model,s] for s in plan['seeds']],axis=0)
    metrics=[];warnings=[];calibration=[]
    for scope,ix in [('2023',np.flatnonzero(data.year==2023)),('2024',np.flatnonzero(data.year==2024)),('pooled',np.arange(len(data)))]:
        d=data.iloc[ix].reset_index(drop=True);w,wa=source_consistent_endpoint_weights(d,endpoint_weights);y=(d.event.to_numpy()==1).astype(int)
        for (model,seed),prediction in sorted(preds.items()):
            p=prediction[ix];assert np.isfinite(p).all() and p.min()>=0 and p.max()<=1
            agg='seed_ensemble' if seed==-1 else 'single_seed'
            for weighting in ['record_equal','company_equal']:
                base=np.ones(len(d)) if weighting=='record_equal' else 1/d.groupby('group').group.transform('size').to_numpy();ww=w*base;keep=ww>0
                metrics.append(dict(scope=scope,model=model,seed=seed,aggregation=agg,weighting=weighting,records=len(d),auc=float(roc_auc_score(y[keep],p[keep],sample_weight=ww[keep])),ap=float(average_precision_score(y[keep],p[keep],sample_weight=ww[keep])),ipcw_brier=float(np.sum(ww*(y-p)**2)/base.sum()),prediction_mean=float(np.average(p,weights=base)),observed_ipcw_incidence=float(np.sum(ww*y)/base.sum()),prediction_max=float(p.max()),**wa))
                for b,ii in enumerate(np.array_split(np.lexsort((d.sample_id.to_numpy(),p)),10)):
                    calibration.append(dict(scope=scope,model=model,seed=seed,aggregation=agg,weighting=weighting,bin=b+1,n=len(ii),prediction=float(np.average(p[ii],weights=base[ii])),observed=float(np.sum(ww[ii]*y[ii])/base[ii].sum()),known=int(keep[ii].sum())))
            for fraction in [.2,.25]:
                flags=top_flags(d,p,fraction);slices=[(scope,np.arange(len(d)))]
                if scope!='pooled':slices += [(str(q),np.asarray(ii)) for q,ii in d.groupby('quarter').indices.items()]
                for label,ii in slices:
                    dd=d.iloc[ii].reset_index(drop=True);ff=flags[ii];warnings.append(dict(scope=label,model=model,seed=seed,aggregation=agg,top_fraction=fraction,**warning_metrics(dd,ff),**lead_summary(event_leads(dd,ff))))
    m=pd.DataFrame(metrics);assert len(m)==192;m.to_csv(out/'metrics.csv',index=False);pd.DataFrame(warnings).to_csv(out/'warning_metrics.csv',index=False);pd.DataFrame(calibration).to_csv(out/'calibration.csv',index=False)
    means=m[m.aggregation=='single_seed'].groupby(['scope','model','weighting']).agg({k:['mean','min','max'] for k in ['auc','ap','ipcw_brier','prediction_mean','prediction_max']});means.columns=['_'.join(k) for k in means.columns];means=means.reset_index();means.to_csv(out/'metrics_summary.csv',index=False)
    # Native endpoint complexity: reduce forest coefficients to target365, retain
    # all structural table slots (including zero terms), report active counts too.
    parent=Path(plan['parent']);order=Path(plan['order_run']);tr=np.load(inp/'final_train_PRIVATE.npz')['X'];complexity=[]
    for source in [parent,order]:
        for p in (source/'results/assembled').glob('*/task-*/output/summary.json'):
            s=json.loads(p.read_text());model={'coverage':'q2'}.get(s['variant'],s['variant'])
            if model=='unrestricted':continue
            cls=CIFExpansion if model=='q2' else HigherOrderCIFExpansion;ex=cls.load(p.parent/'functions_PRIVATE.npz');c,mu=ex._coeff([365]);ex.coefficients=c[:,:,:1];ex.mu=mu[:,:1];ex.grid=np.array([365.])
            cc=ex.complexity(tr);active=[r for r in cc['functions'] if r['nonzero']]
            complexity.append(dict(model=model,seed=s['seed'],native_scale='CIF',link='identity',max_order=int(model[-1]),spatial_cells=cc['spatial_table_cells'],active_function_cells=sum(r['cells'] for r in active),active_functions=len(active),main_functions=sum(len(r['features'])==1 for r in active),pair_functions=sum(len(r['features'])==2 for r in active),reconstruction_error=s['reconstruction_error']))
    for model in plan['baseline_models']:
        for seed in plan['seeds']:
            folder=paths['baseline',model,seed,-1,None,None];s=json.loads((folder/'summary.json').read_text())
            with (folder/'model_PRIVATE.pkl').open('rb') as f:ebm=pickle.load(f)
            active=[v for v in ebm.term_scores_ if np.any(abs(v)>1e-14)]
            complexity.append(dict(model=model,seed=seed,native_scale='log_odds',link='logistic',max_order=1 if model=='ebm_gam' else 2,spatial_cells=s['spatial_cells'],active_function_cells=sum(v.size for v in active),active_functions=len(active),main_functions=s['main_functions'],pair_functions=s['pair_functions'],reconstruction_error=s['reconstruction_error']))
    pd.DataFrame(complexity).to_csv(out/'native_complexity.csv',index=False)
    model_hashes={};prediction_hashes={}
    for key,folder in paths.items():
        for filename in ['forest_PRIVATE.json','model_PRIVATE.pkl']:
            if (folder/filename).exists():model_hashes['/'.join(map(str,key))]=sha(folder/filename)
        if (folder/'predictions_PRIVATE.npz').exists():prediction_hashes['/'.join(map(str,key))]=sha(folder/'predictions_PRIVATE.npz')
    write(out/'execution_receipt.json',dict(status='COMPLETE_PROVISIONAL_POST_EXPOSURE',run=str(run),training_code=plan['code_commit'],coverage_training_code=json.loads((run/'audit/coverage_freeze.json').read_text())['training_code'],formal_tasks=261,formal_baseline_fits=9,training_resample_fits=24,simulation_fits=210,shap_tasks=18,pilot_tasks=3,failed_tasks=[],strict_blind=False,source_counts=plan['source_counts'],publication=[json.loads((run/f'audit/{s}_gate.json').read_text()) for s in ['pilot','formal','coverage']],model_hashes=model_hashes,prediction_hashes=prediction_hashes,plan_sha256=sha(inp/'plan.json'),evaluation_script_sha256=sha(Path(__file__)),core_model_changed=False,full_tests=55,independent_R_checks=2))
    print(means.query("weighting=='record_equal'").to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);main(p.parse_args().run)
