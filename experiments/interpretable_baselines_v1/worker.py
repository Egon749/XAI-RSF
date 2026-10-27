"""One independently owned baseline, resample, simulation or SHAP task."""
from pathlib import Path
import argparse,json,os,pickle,resource,socket,time,hashlib,traceback
import numpy as np
from scipy.special import logit
from crforest import CompetingRiskForest,ForestConfig,CIFExpansion,HigherOrderCIFExpansion
from experiments.interpretable_baselines_v1.methods import *


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    with Path(p).open('x') as f:json.dump(d,f,indent=2,allow_nan=False)
def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def save_pickle(m,p):
    with p.open('xb') as f:pickle.dump(m,f)
    with p.open('rb') as f:return pickle.load(f)


def run(a):
    resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,6*1024**3));start=time.perf_counter();cpu=time.process_time()
    src=Path(os.environ['DATA_DIR']);out=Path(os.environ['OUTPUT_DIR']);out.mkdir(parents=True,exist_ok=True)
    plan=json.loads((src/'plan.json').read_text());row=dict(kind=a.kind,model=a.model,seed=a.seed,replicate=a.replicate,pilot=a.pilot)
    if a.kind in ['baseline','bootstrap','shap']:
        tr=arrays(src/'final_train_PRIVATE.npz');ev=arrays(src/'final_eval_PRIVATE.npz');refs=arrays(src/'references_PRIVATE.npz')
        X=tr['X'];e=tr['event'];t=tr['time'];g=tr['groups'];w=np.load(src/'train_weights_PRIVATE.npy')
        if a.kind=='bootstrap':
            draw=arrays(src/f'bootstrap_{a.replicate}_PRIVATE.npz');ix=draw['rows'];X=X[ix];e=e[ix];t=t[ix];g=draw['groups'];w=draw['weights']
            row['resample_sha256']=sha(src/f'bootstrap_{a.replicate}_PRIVATE.npz')
        row.update(training_rows=len(X),known_weight_rows=int((w>0).sum()),training_target_rows=int((e==1).sum()))
        if a.kind=='shap':
            import shap
            f=CompetingRiskForest.load(src/f'reference_{a.model}_42_forest_PRIVATE.json');fun=ScalarForest(f)
            anchor=refs['anchors'][:12];bg=refs['background'];base,exact=fun.exact_shap(anchor,bg)
            masker=shap.maskers.Independent(bg,max_samples=len(bg));ex=shap.PermutationExplainer(fun,masker,seed=a.seed)
            st=time.perf_counter();res=ex(anchor,max_evals=(2*X.shape[1]+1)*a.permutations,silent=True);elapsed=time.perf_counter()-st
            reconstruction=res.base_values+res.values.sum(1);err=res.values-exact
            row.update(permutations=a.permutations,anchors=len(anchor),background=len(bg),shap_seconds=elapsed,
                additivity_max_error=float(abs(reconstruction-fun(anchor)).max()),attribution_mae=float(abs(err).mean()),
                attribution_rmse=float(np.sqrt(np.mean(err**2))),attribution_max_error=float(abs(err).max()),
                exact_reference_additivity=float(abs(base+exact.sum(1)-fun(anchor)).max()),masker='finite empirical joint row background',horizon_days=365)
            np.savez_compressed(out/'explanations_PRIVATE.npz',approx=res.values,exact=exact,base=res.base_values,prediction=fun(anchor))
        else:
            if a.model.startswith('ebm'):
                interactions={'ebm_gam':0,'ebm_ga2m10':10,'ebm_ga2m30':30}[a.model]
                f=fit_ebm(X,e,w,interactions,a.seed,rounds=50 if a.pilot else plan['ebm_rounds'])
                f2=save_pickle(f,out/'model_PRIVATE.pkl');p=f.predict_proba(ev['X'])[:,1]
                row.update(ebm_complexity(f,refs['anchors']));row['reload_error']=float(abs(p-f2.predict_proba(ev['X'])[:,1]).max())
                if a.kind=='bootstrap':
                    terms=f.eval_terms(refs['anchors']);np.savez_compressed(out/'native_PRIVATE.npz',terms=terms,keys=np.array([','.join(map(str,k)) for k in f.term_features_]),scale='log_odds')
            else:
                config=plan['forest_configs'][a.model].copy();config['seed']=a.seed
                if a.pilot:config['n_trees']=4
                f=CompetingRiskForest(ForestConfig(**config)).fit(X,t,e,g)
                f.save(out/'forest_PRIVATE.json');ff=CompetingRiskForest.load(out/'forest_PRIVATE.json')
                allp=f.predict(ev['X'],[365,366]);p=allp[np.arange(len(ev['X'])),ev['horizon']-365,0]
                row['reload_error']=float(abs(allp-ff.predict(ev['X'],[365,366])).max());row.update(f.audit_)
                fun=ScalarForest(f);np.testing.assert_allclose(fun(refs['anchors']),f.predict(refs['anchors'],[365])[:,0,0],atol=1e-14,rtol=0)
                base,phi=fun.exact_shap(refs['anchors'],refs['background'])
                np.savez_compressed(out/'explanations_PRIVATE.npz',exact=phi,base=base,prediction=fun(refs['anchors']))
                if a.kind=='bootstrap':
                    original=ScalarForest(CompetingRiskForest.load(src/f'reference_{a.model}_42_forest_PRIVATE.json'))
                    bg=group_reference(X,g,len(refs['background']),901)
                    b0,phi0=original.exact_shap(refs['anchors'],refs['background']);b1,phi1=original.exact_shap(refs['anchors'],bg)
                    np.savez_compressed(out/'background_PRIVATE.npz',fixed=phi0,resampled=phi1,base_fixed=b0,base_resampled=b1)
                if a.model=='q2':
                    ex=CIFExpansion.from_forest(f,tr['X']);ex.save(out/'functions_PRIVATE.npz');cc=ex.complexity(tr['X'])
                    row.update({k:v for k,v in cc.items() if k!='functions'})
                    keys=[(j,) for j in ex.mains]+ex.pairs;terms=np.column_stack([ex.component(refs['anchors'],k,[365])[:,0,0] for k in keys])
                    row['reconstruction_error']=float(abs(ex.mu[np.searchsorted(ex.grid,365,side='right')-1,0]+terms.sum(1)-fun(refs['anchors'])).max())
                    np.savez_compressed(out/'native_PRIVATE.npz',terms=terms,keys=np.array([','.join(map(str,k)) for k in keys]),scale='CIF')
            assert row['reload_error']<1e-12
            assert np.isfinite(p).all() and p.min()>=0 and p.max()<=1
            np.savez_compressed(out/'predictions_PRIVATE.npz',sample_ids=ev['sample_ids'],risk=p)
    elif a.kind=='simulation':
        s=simulation(plan['simulation_train_n'],a.seed,a.scenario);te=simulation(plan['simulation_eval_n'],a.seed+70000,a.scenario)
        X,e,t,g=s['X'],s['event'],s['time'],s['groups'];w=endpoint_weights(t,e,2.)
        grid=binary_grid();W=walsh(grid);true=truth(grid,a.scenario);truth_c=W.T@true/len(grid)
        if a.model.startswith('ebm'):
            f=fit_ebm(X,e,w,0 if a.model=='ebm_gam' else 10,a.seed,rounds=plan['simulation_ebm_rounds'])
            p=f.predict_proba(grid)[:,1];testp=f.predict_proba(te['X'])[:,1];native=f.decision_function(grid);native_truth=logit(true)
            save_pickle(f,out/'model_PRIVATE.pkl');row.update(ebm_complexity(f,grid))
        else:
            config=dict(n_trees=32,max_depth=6,path_budget={'q2':2,'q2_coverage':2,'q3':3,'unrestricted':None}[a.model],min_leaf=30,min_events=3,min_groups=15,mtry=8,n_thresholds=5,n_causes=2,seed=a.seed,cut_strategy='shared_quantile',tree_feature_count=3 if a.model=='q2_coverage' else None)
            f=CompetingRiskForest(ForestConfig(**config)).fit(X,t,e,g);f.save(out/'forest_PRIVATE.json');fun=ScalarForest(f,2.)
            p=fun(grid);testp=fun(te['X']);native=p;native_truth=true;row.update(f.audit_)
            if a.model!='unrestricted':
                ex=HigherOrderCIFExpansion.from_forest(f,grid);row['reconstruction_error']=float(abs(ex.predict(grid,[2])[:,0,0]-p).max())
        c=W.T@p/len(grid);native_c=W.T@native/len(grid);native_true_c=W.T@native_truth/len(grid)
        true_support=abs(truth_c[1:])>1e-10;selected=abs(c[1:])>=plan['simulation_effect_threshold']
        tp=int((true_support&selected).sum());fp=int((~true_support&selected).sum());fn=int((true_support&~selected).sum())
        pair=np.array([m.bit_count()==2 for m in range(1,256)]);ptp=int((true_support&selected&pair).sum());pfp=int((~true_support&selected&pair).sum());pfn=int((true_support&~selected&pair).sum())
        row.update(scenario=a.scenario,true_nonconstant_terms=int(true_support.sum()),selected_terms=int(selected.sum()),tp=tp,fp=fp,fn=fn,
            precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),f1=2*tp/max(2*tp+fp+fn,1),pair_tp=ptp,pair_fp=pfp,pair_fn=pfn,
            pair_precision=ptp/max(ptp+pfp,1),pair_recall=ptp/max(ptp+pfn,1),
            probability_coefficient_rmse=float(np.sqrt(np.mean((c[1:]-truth_c[1:])**2))),
            native_coefficient_rmse=float(np.sqrt(np.mean((native_c[1:]-native_true_c[1:])**2))),native_scale='log_odds' if a.model.startswith('ebm') else 'CIF',
            uniform_cube_probability_mse=float(np.mean((p-true)**2)),observed_distribution_probability_mse=float(np.mean((testp-truth(te['X'],a.scenario))**2)),
            high_order_energy=float(np.sum(c[[m.bit_count()>2 for m in range(256)]]**2)))
        # Exact uniform-product Shapley values from all Walsh coefficients; this
        # is common probability-scale truth recovery, not an invented SHAP pair.
        phi=np.column_stack([sum(c[m]*W[:,m]/m.bit_count() for m in range(1,256) if m>>j&1) for j in range(8)])
        truephi=np.column_stack([sum(truth_c[m]*W[:,m]/m.bit_count() for m in range(1,256) if m>>j&1) for j in range(8)])
        row['true_attribution_rmse']=float(np.sqrt(np.mean((phi-truephi)**2)))
        np.savez_compressed(out/'structure_PRIVATE.npz',coefficient=c,truth=truth_c,native=native_c,native_truth=native_true_c)
    else:raise ValueError(a.kind)
    row.update(status='COMPLETE',seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,host=socket.gethostname(),affinity_count=len(os.sched_getaffinity(0)))
    write(out/'summary.json',row);write(out/'hashes.json',{p.name:sha(p) for p in out.iterdir() if p.is_file()})
    print(json.dumps(row),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kind',required=True);p.add_argument('--model',required=True);p.add_argument('--seed',type=int,default=42);p.add_argument('--replicate',type=int,default=-1);p.add_argument('--scenario');p.add_argument('--permutations',type=int,default=8);p.add_argument('--pilot',action='store_true');a=p.parse_args()
    try:run(a)
    except Exception:
        out=Path(os.environ['OUTPUT_DIR']);out.mkdir(parents=True,exist_ok=True);write(out/'FAILED.json',dict(args=vars(a),traceback=traceback.format_exc()));raise
