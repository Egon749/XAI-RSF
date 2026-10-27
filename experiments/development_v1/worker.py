"""One fixed-budget real-data fit; all company details stay in private outputs."""
import argparse,hashlib,json,os,platform,resource,socket,time,traceback
from pathlib import Path
from dataclasses import replace
import numpy as np
from crforest import ForestConfig,CompetingRiskForest,CIFExpansion


def write(p,d):
    with Path(p).open('x') as f:json.dump(d,f,indent=2,ensure_ascii=False,allow_nan=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    with np.load(p,allow_pickle=False) as a:return {k:a[k] for k in a.files}
def audit(p):
    assert np.isfinite(p).all() and p.min()>=-1e-12 and p.sum(-1).max()<=1+1e-12
    assert np.diff(p,axis=1).min(initial=0)>=-1e-12

def run(a):
    start=time.perf_counter();resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,6*1024**3))
    src=Path(os.environ['DATA_DIR']);out=Path(os.environ['OUTPUT_DIR']);out.mkdir(parents=True,exist_ok=True)
    assert not (out/'summary.json').exists()
    plan=json.loads((src/'real_config.json').read_text());cfg=ForestConfig(**plan['real_forest'],seed=42)
    if a.variant=='depth2':cfg=replace(cfg,max_depth=2)
    elif a.variant=='q2':cfg=replace(cfg,path_budget=2)
    elif a.variant=='q1':cfg=replace(cfg,path_budget=1)
    elif a.variant!='unrestricted':raise ValueError(a.variant)
    if a.pilot:cfg=replace(cfg,n_trees=2)
    tr=read(src/f'{a.fold}_train_PRIVATE.npz');ev=read(src/f'{a.fold}_eval_PRIVATE.npz')
    print(json.dumps(dict(event='start',fold=a.fold,variant=a.variant,pilot=a.pilot,records=len(tr['X']))),flush=True)
    t=time.perf_counter();f=CompetingRiskForest(cfg).fit(tr['X'],tr['time'],tr['event'],tr['groups'],n_jobs=a.tree_workers);fit_seconds=time.perf_counter()-t
    f.save(out/'forest_PRIVATE.json');reload=CompetingRiskForest.load(out/'forest_PRIVATE.json')
    pred=f.predict(ev['X'],[365.,366.]);audit(pred)
    reload_error=float(abs(pred-reload.predict(ev['X'],[365.,366.])).max());assert reload_error==0
    risk=pred[np.arange(len(pred)),ev['horizon']-365]
    np.savez_compressed(out/'predictions_PRIVATE.npz',sample_ids=ev['sample_ids'],groups=ev['groups'],cif12=risk,horizon_days=ev['horizon'])
    row=dict(status='COMPLETED_PROVISIONAL',fold=a.fold,variant=a.variant,pilot=a.pilot,seed=42,training=plan['source_counts'][f'{a.fold}_train'],validation=plan['source_counts'][f'{a.fold}_eval'],input_columns=75,fit_seconds=fit_seconds,reload_error=reload_error,tree_workers=f.parallelism_['workers'],**f.audit_)
    grid=np.unique(np.r_[0.,f.grid_,365.,366.]);row['native_time_count']=len(f.grid_)
    # All validation companies at every native event time, not just the endpoint.
    max_error=0.;checks=0
    exp=CIFExpansion.from_forest(f,tr['X']) if a.variant!='unrestricted' else None
    if exp:
        exp.save(out/'functions_PRIVATE.npz');exp=CIFExpansion.load(out/'functions_PRIVATE.npz')
    for start_at in range(0,len(ev['X']),128):
        xx=ev['X'][start_at:start_at+128];pp=f.predict(xx,grid);audit(pp);checks+=pp.size
        if exp:
            ee=exp.predict(xx,grid);audit(ee);max_error=max(max_error,float(abs(pp-ee).max()))
    if exp:
        assert max_error<1e-10,max_error
        cc=exp.complexity(tr['X']);write(out/'complexity_PRIVATE.json',cc)
        row.update({k:v for k,v in cc.items() if k!='functions'})
        row['reconstruction_error']=max_error
        cases=[]
        for label,i in enumerate(np.unique([0,int(np.argmin(risk[:,0])),int(np.argmax(risk[:,0]))])):
            horizon=[float(ev['horizon'][i])];xx=ev['X'][i:i+1];base=exp._coeff(horizon)[1];total=base.copy();components=[]
            for key in [(j,) for j in exp.mains]+exp.pairs:
                val=exp.component(xx,key,horizon)[0];total+=val
                components.append(dict(features=list(map(int,key)),cif_contribution=val.tolist()))
            assert abs(total-risk[i]).max()<1e-10
            cases.append(dict(case=f'anonymous_{label+1}',baseline=base.tolist(),components=components,cif=total.tolist(),scale='CIF probability; not causal'))
        write(out/'cases_PRIVATE.json',cases)
    else:row['reconstruction_error']=None
    row['probability_scalar_checks']=checks
    leaves=[n for tree in f.trees_ for n in tree if 'cif' in n]
    row['leaf_zero_target_company_support']=sum(n['cause_company_support'][0]==0 for n in leaves)
    row['leaf_zero_competing_company_support']=sum(n['cause_company_support'][1]==0 for n in leaves)
    row.update(seconds=time.perf_counter()-start,peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,host=socket.gethostname(),python=platform.python_version(),numpy=np.__version__,affinity_count=len(os.sched_getaffinity(0)),model_sha256=sha(out/'forest_PRIVATE.json'),input_config_sha256=sha(src/'real_config.json'),penalty='NONE')
    write(out/'summary.json',row)
    write(out/'output_hashes.json',{p.name:sha(p) for p in out.iterdir() if p.is_file()})
    print(json.dumps(row),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',choices=['V19','V20'],required=True);p.add_argument('--variant',required=True);p.add_argument('--pilot',action='store_true');p.add_argument('--tree-workers',type=int,default=1);a=p.parse_args()
    try:run(a)
    except Exception:
        out=Path(os.environ['OUTPUT_DIR']);out.mkdir(exist_ok=True,parents=True)
        if not (out/'FAILED.json').exists():write(out/'FAILED.json',dict(fold=a.fold,variant=a.variant,traceback=traceback.format_exc()))
        raise
