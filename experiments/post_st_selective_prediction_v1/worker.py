import argparse,os,resource,socket,time,traceback
from crforest import CompetingRiskForest,CIFExpansion
from experiments.post_st_ablation_v1.verification import merge
from .common import *
from .diagnostics import leaf_diagnostics,summarize_leaves

def main(year):
    start=time.perf_counter();block_forest_fits();resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
    root=Path(os.environ['DATA_DIR']);out=Path(os.environ['OUTPUT_DIR']);out.mkdir(exist_ok=True,parents=True)
    tr=arrays(root/f'folds/{year}/train_PRIVATE.npz');ev=arrays(root/f'folds/{year}/eval_PRIVATE.npz')
    assert len(np.unique(tr['groups']))==len(tr['groups']) and (tr['year']<year).all()
    assert set(ev)=={'X','sample_ids','h12','h24'}
    fs=[];pred=[];leaf=[]
    for seed in SEEDS:
        f=CompetingRiskForest.load(root/f'reference/{year}/{seed}/forest_PRIVATE.json');assert f.audit()['max_path_variables']<=2
        z=arrays(root/f'reference/{year}/{seed}/predictions_PRIVATE.npz');p=f.predict(ev['X'],TIMES)
        np.testing.assert_array_equal(z['sample_ids'],ev['sample_ids']);np.testing.assert_array_equal(p,z['cif'])
        fs.append(f);pred.append(p);leaf.append(leaf_diagnostics(f,tr,ev['X']))
    ensemble=merge(fs);ex=CIFExpansion.from_forest(ensemble,tr['X']);ex.save(out/'functions_PRIVATE.npz');again=CIFExpansion.load(out/'functions_PRIVATE.npz')
    X=np.vstack([tr['X'],ev['X']]);grid=np.unique(np.r_[0,ensemble.grid_,TIMES]);keys=[(j,) for j in ex.mains]+ex.pairs
    err=0.;reloaderr=0.;maxsum=0.;minimum=0.;inc=0.
    for first in range(0,len(X),32):
        xx=X[first:first+32];expected=ensemble.predict(xx,grid);total=np.broadcast_to(ex._coeff(grid)[1],expected.shape).copy()
        for key in keys:total+=ex.component(xx,key,grid)
        err=max(err,float(abs(total-expected).max()));reloaderr=max(reloaderr,float(abs(again.predict(xx,grid)-expected).max()))
        maxsum=max(maxsum,float(expected.sum(-1).max()));minimum=min(minimum,float(expected.min()));inc=min(inc,float(np.diff(expected,axis=1).min()))
    assert err<=1e-10 and reloaderr<=1e-10 and maxsum<=1+1e-10 and minimum>=-1e-12 and inc>=-1e-12
    sparse={h:np.zeros((len(ev['X']),2)) for h in [12,24]};den={h:np.zeros((len(ev['X']),2)) for h in [12,24]}
    zero={h:np.zeros((len(ev['X']),2)) for h in [12,24]};pair={h:np.zeros((len(ev['X']),2)) for h in [12,24]}
    min_support=np.full(len(ev['X']),len(tr['X']),int)
    for key in keys:
        count=cell_support(ex,key,tr['X'],ev['X'],tr['groups']);min_support=np.minimum(min_support,count)
        raw=ex.component(ev['X'],key,TIMES)
        for h in [12,24]:
            magnitude=abs(horizon_predictions(raw,TIMES,ev[f'h{h}']));den[h]+=magnitude
            sparse[h]+=magnitude*(count<10)[:,None];zero[h]+=magnitude*(count==0)[:,None]
            if len(key)==2:pair[h]+=magnitude
    result=dict(sample_ids=ev['sample_ids'],minimum_function_cell_support=min_support)
    for h in [12,24]:
        risk=np.array([horizon_predictions(p,TIMES,ev[f'h{h}']) for p in pred]);mu=risk.mean(0);sd=risk.std(0,ddof=1)
        np.testing.assert_allclose(mu,horizon_predictions(ensemble.predict(ev['X'],TIMES),TIMES,ev[f'h{h}']),atol=1e-14,rtol=0)
        for c in [1,2]:
            prefix=f'{h}_{c}_';result[prefix+'risk']=mu[:,c-1];result[prefix+'seed_sd']=sd[:,c-1]
            result.update({prefix+k:v for k,v in summarize_leaves(leaf,h,c).items()})
            for name,num in [('function_sparse_share',sparse[h]),('function_zero_share',zero[h]),('interaction_share',pair[h])]:
                result[prefix+name]=np.divide(num[:,c-1],den[h][:,c-1],out=np.zeros(len(mu)),where=den[h][:,c-1]>1e-14)
    np.savez_compressed(out/'diagnostics_PRIVATE.npz',**result)
    summary=dict(status='COMPLETE',year=year,new_forests=0,new_calibrators=0,trees_reused=96,train_companies=len(tr['X']),eval_companies=len(ev['X']),
        native_time_points=len(grid),full_component_error=err,function_reload_error=reloaderr,probability_min=minimum,probability_sum_max=maxsum,minimum_time_increment=inc,
        full_rows_checked=len(X),max_path_variables=2,main_functions=len(ex.mains),pair_functions=len(ex.pairs),no_evaluation_outcomes_read=True,
        host=socket.gethostname(),job=os.environ.get('SLURM_JOB_ID'),seconds=time.perf_counter()-start,peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    write(out/'summary.json',summary);write(out/'hashes.json',{p.name:sha(p) for p in out.iterdir() if p.is_file()});print(json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--year',type=int,required=True);a=p.parse_args()
    try:main(a.year)
    except Exception:
        out=Path(os.environ['OUTPUT_DIR']);out.mkdir(exist_ok=True,parents=True);write(out/'FAILED.json',dict(year=a.year,traceback=traceback.format_exc()));raise
