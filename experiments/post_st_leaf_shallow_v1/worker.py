from pathlib import Path
import argparse,json,os,resource,socket,time,traceback
import numpy as np
from crforest import CompetingRiskForest,ForestConfig,CIFExpansion
from crforest.shrinkage import shrink_leaf_cifs
from experiments.post_st_prognosis_v1.common import arrays,sha,write


def emit(f,tr,ev,out,arm,meta):
    out.mkdir();started=time.perf_counter();X=np.vstack([tr['X'],ev['X']]);n=len(tr['X'])
    times=np.array([365.,366.,730.,731.]);prediction=f.predict(X,times);native=f.predict(X)
    assert np.isfinite(native).all() and native.min()>=-1e-12 and native.sum(-1).max()<=1+1e-10
    assert len(f.grid_)==1 or np.diff(native,axis=1).min()>=-1e-12
    f.save(out/'forest_PRIVATE.json');again=CompetingRiskForest.load(out/'forest_PRIVATE.json')
    reload_error=float(abs(again.predict(X,times)-prediction).max());assert reload_error==0
    ex=CIFExpansion.from_forest(f,tr['X']);ex.save(out/'functions_PRIVATE.npz')
    cc=ex.complexity(tr['X'],support_min=10);write(out/'complexity_PRIVATE.json',cc)
    keys=[(j,) for j in ex.mains]+ex.pairs
    terms=np.stack([ex.component(X,k,times) for k in keys],axis=1) if keys else np.zeros((len(X),0,len(times),2))
    mu=ex._coeff(times)[1]
    errors=dict(reconstruction_error=float(abs(ex.predict(X,times)-prediction).max()),
        function_reload_error=float(abs(CIFExpansion.load(out/'functions_PRIVATE.npz').predict(X,times)-prediction).max()),
        native_grid_reconstruction_error=float(abs(ex.predict(X[:16])-native[:16]).max()),
        complete_component_sum_error=float(abs(mu+terms.sum(1)-prediction).max()))
    assert max(errors.values())<1e-10
    np.savez_compressed(out/'components_PRIVATE.npz',keys=np.array([','.join(map(str,k)) for k in keys]),terms=terms,baseline=mu,times=times)
    np.savez_compressed(out/'predictions_PRIVATE.npz',sample_ids=ev['sample_ids'],times=times,cif=prediction[n:],training_cif=prediction[:n])
    np.savez_compressed(out/'native_curves_PRIVATE.npz',grid=f.grid_,cif=native[n:])
    leaves=[node for tree in f.trees_ for node in tree if 'cif' in node]
    weights=[node['leaf_shrinkage']['weight'] for node in leaves if 'leaf_shrinkage' in node]
    borrowed=sum(node['cause_company_support'][0]==0 and np.asarray(node['cif'])[:,0].max()>0 for node in leaves)
    row=dict(**meta,model='xai_q2',arm=arm,status='COMPLETE',**f.audit_,**{k:v for k,v in cc.items() if k!='functions'},**errors,
        reload_error=reload_error,training_companies=n,evaluation_companies=len(ev['X']),input_columns=X.shape[1],
        training_termination_events=int((tr['event']==1).sum()),training_recovery_events=int((tr['event']==2).sum()),
        target_unsupported_leaf_fraction=float(np.mean([x['cause_company_support'][0]==0 for x in leaves])),
        target_leaf_support_median=float(np.median([x['cause_company_support'][0] for x in leaves])),
        unsupported_leaves_with_borrowed_target_risk=borrowed,
        shrinkage_weight_min=float(min(weights)) if weights else 1.,shrinkage_weight_median=float(np.median(weights)) if weights else 1.,
        shrinkage_weight_max=float(max(weights)) if weights else 1.,
        probability_min=float(native.min()),max_cause_probability_sum=float(native.sum(-1).max()),
        model_export_seconds=time.perf_counter()-started)
    write(out/'summary.json',row);write(out/'hashes.json',{p.name:sha(p) for p in out.iterdir() if p.is_file()})
    return row


def run(a):
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3));started=time.perf_counter();cpu=time.process_time()
    root=Path(os.environ['DATA_DIR']);src=root/'folds'/str(a.year);out=Path(os.environ['OUTPUT_DIR']);out.mkdir(parents=True,exist_ok=True)
    plan=json.loads((src/'plan.json').read_text());protocol=json.loads((root/'protocol.json').read_text())
    for name,h in plan['source_hashes'].items():assert sha(src/name)==h
    tr=arrays(src/'train_PRIVATE.npz');ev=arrays(src/'eval_PRIVATE.npz');assert set(ev)=={'X','sample_ids'}
    fit=bool(a.pilot or a.depth==4);trees=4 if a.pilot else 32;fit_seconds=0.;reference_hash=None
    if fit:
        cfg={**plan['forest_common'],**plan['forest_variants']['xai_q2'], 'seed':a.seed,'max_depth':a.depth,'n_trees':trees}
        start=time.perf_counter();f=CompetingRiskForest(ForestConfig(**cfg)).fit(tr['X'],tr['time'],tr['event'],tr['groups']);fit_seconds=time.perf_counter()-start
    else:
        folder=root/'reference'/str(a.year)/str(a.seed);identity=json.loads((folder/'identity.json').read_text())
        assert sha(folder/'forest_PRIVATE.json')==identity['source_sha256'];reference_hash=identity['source_sha256']
        f=CompetingRiskForest.load(folder/'forest_PRIVATE.json')
        assert f.config.n_trees==32 and f.config.max_depth==6 and f.config.path_budget==2 and f.config.seed==a.seed
    meta=dict(test_year=a.year,seed=a.seed,pilot=a.pilot,host=socket.gethostname(),affinity_count=len(os.sched_getaffinity(0)),
        code_commit=plan['code_commit'],base_forest_grown_in_this_job=fit,base_fit_seconds=fit_seconds,reference_forest_sha256=reference_hash)
    artifacts=[]
    if fit:
        emit(f,tr,ev,out/'raw','baseline' if a.depth==6 else 'shallow',meta);artifacts.append('raw')
    start=time.perf_counter();sh=shrink_leaf_cifs(f,tr['X'],tr['time'],tr['event'],tr['groups'],protocol['shrinkage']['strength'])
    shrink_seconds=time.perf_counter()-start
    assert f.audit_==sh.audit_
    emit(sh,tr,ev,out/'shrunk','parent_shrink' if a.depth==6 else 'shallow_parent',dict(meta,leaf_update_seconds=shrink_seconds));artifacts.append('shrunk')
    row=dict(status='COMPLETE',test_year=a.year,seed=a.seed,depth=a.depth,pilot=a.pilot,host=socket.gethostname(),
        new_forest_fits=int(fit),leaf_updates=1,model_artifacts=len(artifacts),artifacts=artifacts,
        seconds=time.perf_counter()-started,cpu_seconds=time.process_time()-cpu,
        peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,code_commit=plan['code_commit'])
    write(out/'summary.json',row)
    write(out/'hashes.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})
    print(json.dumps(row),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--year',type=int,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--depth',type=int,choices=[4,6],required=True);p.add_argument('--pilot',action='store_true');a=p.parse_args()
    try:run(a)
    except Exception:
        out=Path(os.environ['OUTPUT_DIR']);out.mkdir(exist_ok=True,parents=True)
        write(out/'FAILED.json',dict(args=vars(a),traceback=traceback.format_exc()));raise
