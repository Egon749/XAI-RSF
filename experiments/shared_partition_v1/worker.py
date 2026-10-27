"""One fixed-budget real-data fit; all company details stay in private outputs."""
import argparse,hashlib,json,os,platform,resource,socket,time,traceback
from pathlib import Path
from dataclasses import replace
import numpy as np
from crforest import ForestConfig,CompetingRiskForest,CIFExpansion
from experiments.shared_partition_v1.engine import fit_shard,assemble


def write(p,d):
    with Path(p).open('x') as f:json.dump(d,f,indent=2,ensure_ascii=False,allow_nan=False)
def write_budget_log(p,rows):
    # Missing-only splits use -inf as an exact sentinel, not a learned threshold.
    # Encode that sentinel explicitly without changing live model diagnostics.
    import copy
    rows=copy.deepcopy(rows)
    for row in rows:
        rejected=row.get('best_rejected')
        if rejected is not None and rejected['threshold']==-np.inf:
            rejected['threshold']='-Infinity'
    write(p,rows)
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
    plan=json.loads((src/'plan.json').read_text())
    cfg=ForestConfig(**plan['configs'][f'{a.fold}/{a.seed}/{a.variant}'])
    if a.pilot:cfg=replace(cfg,n_trees=4)
    tr=read(src/f'{a.fold}_train_PRIVATE.npz');ev=read(src/f'{a.fold}_eval_PRIVATE.npz')
    print(json.dumps(dict(event='start',fold=a.fold,variant=a.variant,pilot=a.pilot,records=len(tr['X']))),flush=True)
    t=time.perf_counter();cpu_start=time.process_time()
    if a.mode=='shard':
        f=fit_shard(cfg,tr,a.tree_start,a.tree_count)
        f.save(out/'shard_PRIVATE.json')
        write(out/'shard_summary.json',dict(status='SHARD_COMPLETE',fold=a.fold,variant=a.variant,seed=a.seed,
              tree_start=a.tree_start,tree_count=a.tree_count,host=socket.gethostname(),
              seconds=time.perf_counter()-t,cpu_seconds=time.process_time()-cpu_start,
              model_sha256=sha(out/'shard_PRIVATE.json'),source_config_sha256=sha(src/'plan.json')))
        return
    elif a.mode=='assemble':
        index=json.loads(Path(a.index).read_text());parts=[];times=[];cpu_times=[]
        for item in index[f'{a.fold}/{a.seed}/{a.variant}']:
            folder=Path(item['path']);meta=json.loads((folder/'shard_summary.json').read_text())
            assert meta['fold']==a.fold and meta['variant']==a.variant and meta['seed']==a.seed
            assert sha(folder/'shard_PRIVATE.json')==meta['model_sha256']==item['sha256']
            assert meta['source_config_sha256']==sha(src/'plan.json')
            parts.append((meta['tree_start'],CompetingRiskForest.load(folder/'shard_PRIVATE.json')))
            times.append(meta['seconds']);cpu_times.append(meta['cpu_seconds'])
        f=assemble(parts);fit_seconds=sum(times);fit_cpu_seconds=sum(cpu_times)
    else:
        f=CompetingRiskForest(cfg).fit(tr['X'],tr['time'],tr['event'],tr['groups'])
        fit_seconds=time.perf_counter()-t;fit_cpu_seconds=time.process_time()-cpu_start
    f.save(out/'forest_PRIVATE.json');reload=CompetingRiskForest.load(out/'forest_PRIVATE.json')
    write_budget_log(out/'budget_log_PRIVATE.json',getattr(f,'budget_log_',[]))
    pred=f.predict(ev['X'],[365.,366.]);audit(pred)
    reload_error=float(abs(pred-reload.predict(ev['X'],[365.,366.])).max());assert reload_error==0
    risk=pred[np.arange(len(pred)),ev['horizon']-365]
    np.savez_compressed(out/'predictions_PRIVATE.npz',sample_ids=ev['sample_ids'],groups=ev['groups'],cif12=risk,horizon_days=ev['horizon'])
    row=dict(status='COMPLETED_PROVISIONAL',fold=a.fold,variant=a.variant,pilot=a.pilot,seed=a.seed,training=plan['source_counts'][f'{a.fold}_train'],validation=plan['source_counts'][f'{a.fold}_eval'],input_columns=75,fit_seconds=fit_seconds,reload_error=reload_error,execution_mode=a.mode,fit_cpu_seconds=fit_cpu_seconds,**f.audit_)
    grid=np.unique(np.r_[0.,f.grid_,365.,366.]);row['native_time_count']=len(f.grid_)
    # All validation companies at every native event time, not just the endpoint.
    max_error=0.;checks=0
    exp=CIFExpansion.from_forest(f,tr['X']) if cfg.path_budget is not None else None
    if exp:
        exp.save(out/'functions_PRIVATE.npz');exp=CIFExpansion.load(out/'functions_PRIVATE.npz')
    for start_at in range(0,len(ev['X']),128):
        xx=ev['X'][start_at:start_at+128];pp=f.predict(xx,grid);audit(pp);checks+=pp.size
        if exp:
            ee=exp.predict(xx,grid);audit(ee);max_error=max(max_error,float(abs(pp-ee).max()))
    if exp:
        assert max_error<1e-10,max_error
        cc=exp.complexity(tr['X']);write(out/'complexity_PRIVATE.json',cc)
        if cfg.global_cell_budget is not None:
            assert cc['spatial_table_cells']==f.partition_audit_['spatial_cells']<=cfg.global_cell_budget
            row.update(global_budget=cfg.global_cell_budget,budget_cells=f.partition_audit_['spatial_cells'],
                       budget_rejected_candidates=f.partition_audit_['rejected_candidates'],
                       zero_cost_splits=f.partition_audit_['zero_cost_splits'])
        if f.shared_cuts_ is not None:
            for tree in f.trees_:
                for node in tree:
                    if 'threshold' in node and np.isfinite(node['threshold']):
                        assert node['threshold'] in f.shared_cuts_[node['feature']]
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
    row.update(seconds=time.perf_counter()-start,peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,host=socket.gethostname(),python=platform.python_version(),numpy=np.__version__,affinity_count=len(os.sched_getaffinity(0)),model_sha256=sha(out/'forest_PRIVATE.json'),input_config_sha256=sha(src/'plan.json'),penalty='NONE')
    write(out/'summary.json',row)
    write(out/'output_hashes.json',{p.name:sha(p) for p in out.iterdir() if p.is_file()})
    print(json.dumps(row),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',choices=['V19','V20'],required=True);p.add_argument('--variant',required=True);p.add_argument('--pilot',action='store_true');p.add_argument('--seed',type=int,required=True);p.add_argument('--mode',choices=['fit','shard','assemble'],default='fit');p.add_argument('--tree-start',type=int);p.add_argument('--tree-count',type=int,default=4);p.add_argument('--index');a=p.parse_args()
    try:run(a)
    except Exception:
        out=Path(os.environ['OUTPUT_DIR']);out.mkdir(exist_ok=True,parents=True)
        if not (out/'FAILED.json').exists():write(out/'FAILED.json',dict(fold=a.fold,variant=a.variant,traceback=traceback.format_exc()))
        raise
