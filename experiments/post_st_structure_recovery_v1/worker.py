"""One paired data repetition. Six fits; no result-dependent tuning/retries."""
from pathlib import Path
import argparse,gzip,json,os,resource,socket,tempfile,time,traceback
import numpy as np
import pandas as pd
from crforest import CompetingRiskForest,HigherOrderCIFExpansion
from crforest.shrinkage import shrink_leaf_cifs
from experiments.post_st_ablation_v1.verification import merge,replay
from .common import *
from .recovery import population_expansion,assess,tree_signal_audit

def error(a,b):return float(np.max(np.abs(a-b),initial=0.))

def save_model(f,path,X):
    # gzip is lossless packaging of the unchanged native JSON model format.
    with tempfile.TemporaryDirectory() as td:
        temp=Path(td)/'forest.json';f.save(temp)
        with gzip.open(path,'wb',compresslevel=3) as dest:dest.write(temp.read_bytes())
        with gzip.open(path,'rb') as src:temp.write_bytes(src.read())
        loaded=CompetingRiskForest.load(temp)
        err=error(f.predict(X),loaded.predict(X));assert err==0
    return err

def verify(f,X,train,ex,out,save=True):
    errors=dict(reload=0.,training_reference=0.,population_reference=0.,explicit_sum=0.,function_reload=0.,native_time=0.)
    # CIF validity follows for every x from valid leaves and positive averaging.
    lo,hi,inc=0.,0.,0.
    for tree in f.trees_:
        for node in tree:
            if 'cif' not in node:continue
            v=np.asarray(node['cif']);assert np.isfinite(v).all()
            lo=min(lo,float(v.min()));hi=max(hi,float(v.sum(1).max()))
            if len(v)>1:inc=min(inc,float(np.diff(v,axis=0).min()))
    assert lo>=-1e-12 and hi<=1+1e-12 and inc>=-1e-12
    if save:errors['reload']=save_model(f,out/'forest.json.gz',X[:32])
    ordinary=HigherOrderCIFExpansion.from_forest(f,train['X'])
    # Every native time/both causes on a deterministic training+test probe;
    # ALL train/test rows at both prespecified recovery horizons below.
    probe=X[np.linspace(0,len(X)-1,min(32,len(X)),dtype=int)]
    errors['native_time']=error(ordinary.predict(probe),f.predict(probe))
    pred=f.predict(X,TIMES)  # expansion internally batches; traverse each forest only once
    errors['training_reference']=error(ordinary.predict(X,TIMES),pred)
    errors['population_reference']=error(ex.predict(X),pred)
    p=probe[:4];total=np.broadcast_to(ex.mu,(len(p),*ex.mu.shape)).copy()
    for key in ex.keys:total+=ex.component(p,key)
    errors['explicit_sum']=error(total,f.predict(p,TIMES))
    ex.save(out/'population_functions.npz')
    loaded=HigherOrderCIFExpansion.load(out/'population_functions.npz')
    errors['function_reload']=error(loaded.predict(probe),ex.predict(probe))
    assert max(errors.values())<=1e-10,errors
    return dict(**errors,cif_minimum=lo,cif_sum_maximum=hi,increment_minimum=inc,
        native_probe_rows=len(probe),native_times=len(f.grid_),horizon_rows=len(X))

def complexity(f,ex,records):
    answer=dict(max_path_variables=f.audit()['max_path_variables'],candidate_evaluations=f.candidate_evaluations_,
        leaves=len(ex.features),native_times=len(f.grid_),factored_coefficients=ex.coefficients.size)
    cells=0
    for order in range(1,5):
        keys=[k for k in ex.keys if len(k)==order]
        count=sum(int(np.prod([len(ex.cuts(k,j))+2 for j in k])) for k in keys)
        cells+=count;answer[f'order{order}_structural_functions']=len(keys);answer[f'order{order}_cells']=count
        answer[f'order{order}_nonzero_target_24m']=sum(r['day']==730 and r['cause']==1 and r['order']==order and r['estimated_rms']>1e-12 for r in records)
    answer.update(spatial_cells_including_missing=cells,scalar_cells_two_horizons=cells*4,scalar_cells_native=cells*len(f.grid_)*2)
    return answer

def shrink_audit(raw,sh):
    error_max=0.;zero=borrowed=0;supports=[];alphas=[]
    assert raw.audit()==sh.audit()
    for ta,tb in zip(raw.trees_,sh.trees_):
        for a,b in zip(ta,tb):
            assert {k:v for k,v in a.items() if k!='cif'}=={k:v for k,v in b.items() if k not in ('cif','leaf_shrinkage')}
            if 'cif' not in a:continue
            info=b['leaf_shrinkage'];n=a['groups'];alpha=1. if info['parent_node'] is None else n/(n+10)
            assert info['unique_inbag_companies']==n and info['weight']==alpha
            np.testing.assert_array_equal(a['cif'],info['raw_leaf_cif'])
            expected=alpha*np.asarray(a['cif'])+(1-alpha)*np.asarray(info['raw_parent_cif'])
            error_max=max(error_max,error(expected,np.asarray(b['cif'])))
            supports.append(a['cause_company_support'][0]);alphas.append(alpha)
            if a['cause_company_support'][0]==0:
                zero+=1;borrowed+=int(np.asarray(b['cif'])[:,0].max()>0)
    assert error_max<=1e-14
    return dict(shrink_formula_error=error_max,zero_target_leaves=zero,borrowed_positive_target_leaves=borrowed,
        mean_leaf_target_companies=float(np.mean(supports)),mean_leaf_weight=float(np.mean(alphas)))

def run(a,out):
    tr,test=generate(a.scenario,a.rep);X=np.vstack([tr['X'],test]);kind=population_kind(a.scenario)
    seeds=[42] if a.pilot else SEEDS;truth_test=truth(test,a.scenario,a.rep)
    write(out/'data_identity.json',dict(scenario=a.scenario,rep=a.rep,train_sha256=data_hash(tr),test_sha256=data_hash({'X':test}),
        signal_coordinates=signals(a.rep).tolist(),train_companies=len(tr['X']),target_events=int((tr['event']==1).sum()),
        competing_events=int((tr['event']==2).sum()),censored=int((tr['event']==0).sum()),
        early_censored=int(((tr['event']==0)&(tr['time']<730)).sum()),configuration_differences=config_differences()))
    np.savez_compressed(out/'synthetic_data.npz',**tr,test_X=test)
    results=[];signal_rows=[];summary_rows=[];tree_rows=[];validations=[];model_rows=[];predictions=[]
    members={arm:{v:[] for v in ['raw','shrunk']} for arm in ARMS};pair_reference={}
    def inspect(f,arm,variant,seed,folder,cost):
        start=time.perf_counter();ex=population_expansion(f,tr['X'],kind)
        export_seconds=time.perf_counter()-start
        start=time.perf_counter();rr,ss,summary=assess(ex,a.scenario,a.rep);recovery_seconds=time.perf_counter()-start
        tags=dict(scenario=a.scenario,rep=a.rep,arm=arm,variant=variant,seed=str(seed))
        for target,rows in [(results,rr),(signal_rows,ss),(summary_rows,summary)]:target.extend(dict(**tags,**r) for r in rows)
        if seed!='ensemble':tree_rows.extend(dict(**tags,**r) for r in tree_signal_audit(ex,f,a.scenario,a.rep))
        start=time.perf_counter();v=verify(f,X,tr,ex,folder,save=seed!='ensemble');validation_seconds=time.perf_counter()-start
        validations.append(dict(**tags,**v))
        pred=f.predict(test,TIMES)
        for h,day in enumerate(TIMES):
            for c in [1,2]:
                pp=pred[:,h,c-1];tt=truth_test[:,h,c-1]
                predictions.append(dict(**tags,day=int(day),cause=c,joint_risk_mse=float(np.mean((pp-tt)**2)),
                    expected_brier=float(np.mean((pp-tt)**2+tt*(1-tt))),mean_prediction=float(pp.mean()),true_mean=float(tt.mean())))
        model_rows.append(dict(**tags,**cost,**complexity(f,ex,rr),function_compile_seconds=export_seconds,
            recovery_seconds=recovery_seconds,verification_seconds=validation_seconds,function_bytes=(folder/'population_functions.npz').stat().st_size,
            model_bytes=(folder/'forest.json.gz').stat().st_size if seed!='ensemble' else 0))
        return pred
    for seed in seeds:
        for arm in ARMS:
            cfg=configuration(seed,arm);pp=pairing(tr['groups'],cfg)
            if seed in pair_reference:assert pp==pair_reference[seed]
            else:pair_reference[seed]=pp
            start=time.perf_counter();raw=CompetingRiskForest(cfg).fit(tr['X'],tr['time'],tr['event'],tr['groups']);fit_seconds=time.perf_counter()-start
            for i,b in enumerate(raw.inbag_):assert array_hash(b)==pp[i]['multiplicity_sha256']
            start=time.perf_counter();sh=shrink_leaf_cifs(raw,tr['X'],tr['time'],tr['event'],tr['groups'],10);shrink_seconds=time.perf_counter()-start
            support=shrink_audit(raw,sh)
            if a.pilot:
                *_,checks=replay(sh,tr);write(out/f'replay_{arm}_{seed}.json',checks)
            for variant,f in [('raw',raw),('shrunk',sh)]:
                folder=out/f'{arm}_{seed}_{variant}';folder.mkdir()
                inspect(f,arm,variant,seed,folder,dict(fit_seconds=fit_seconds,shrink_seconds=shrink_seconds,**support))
                members[arm][variant].append(f)
    write(out/'pairing.json',pair_reference)
    if not a.pilot:
        for arm in ARMS:
            for variant,forests in members[arm].items():
                folder=out/f'{arm}_ensemble_{variant}';folder.mkdir();f=merge(forests)
                pred=inspect(f,arm,variant,'ensemble',folder,dict(fit_seconds=0.,shrink_seconds=0.))
                direct=np.mean([m.predict(test,TIMES) for m in forests],axis=0)
                assert error(pred,direct)<1e-13
    for name,rows in [('components',results),('tree_signal',tree_rows)]:pd.DataFrame(rows).to_csv(out/f'{name}.csv.gz',index=False,compression='gzip')
    for name,rows in [('signals',signal_rows),('recovery',summary_rows),('prediction',predictions),('models',model_rows),('validation',validations)]:pd.DataFrame(rows).to_csv(out/f'{name}.csv',index=False)
    return dict(scenario=a.scenario,rep=a.rep,pilot=a.pilot,new_forest_fits=2*len(seeds),derived_shrunk_models=2*len(seeds),
        seed_predictors=4*len(seeds),ensemble_predictors=0 if a.pilot else 4,paired_bootstrap_and_features=True,
        max_validation_error=max(max(r[k] for k in ['reload','training_reference','population_reference','explicit_sum','function_reload','native_time']) for r in validations))

def main(a):
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
    start=time.perf_counter();cpu=time.process_time();out=Path(os.environ['OUTPUT_DIR']);out.mkdir(exist_ok=True,parents=True)
    row=run(a,out)
    row.update(status='COMPLETE',host=socket.gethostname(),affinity_count=len(os.sched_getaffinity(0)),
        seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    write(out/'summary.json',row);write(out/'hashes.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})
    print(json.dumps(row),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scenario',choices=SCENARIOS,required=True);p.add_argument('--rep',type=int,required=True);p.add_argument('--pilot',action='store_true');a=p.parse_args()
    try:main(a)
    except Exception:
        out=Path(os.environ['OUTPUT_DIR']);out.mkdir(exist_ok=True,parents=True);write(out/'FAILED.json',dict(args=vars(a),traceback=traceback.format_exc()));raise
