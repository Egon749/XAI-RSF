from pathlib import Path
import argparse
import json
import os
import resource
import socket
import subprocess
import time
import traceback
import numpy as np
import pandas as pd
from crforest import CompetingRiskForest, ForestConfig, CIFExpansion, cluster_bootstrap
from experiments.post_st_prognosis_v1.common import arrays,sha,write

RSCRIPT='/home/oliver/MainDiskData/financial_diagnosis/csf_joint_sensitivity_v1/environment/r/bin/Rscript'
RLIB='/home/oliver/MainDiskData/ernet_original_tau0_full_20261015/dynamic_competing_h24_20261015/rsf_er_competing_20261016/environment/rlib'


def reference_fit(tr,X,plan,seed,trees,out):
    med=np.array([np.median(v[np.isfinite(v)]) if np.isfinite(v).any() else 0. for v in tr['X'].T])
    np.save(out/'training_imputation_PRIVATE.npy',med)
    trainX=np.where(np.isnan(tr['X']),med,tr['X']);predictX=np.where(np.isnan(X),med,X)
    names=[f'x{j}' for j in range(X.shape[1])]
    df=pd.DataFrame(trainX,columns=names);df.insert(0,'status',tr['event']);df.insert(0,'time',tr['time'])
    df.to_csv(out/'train_PRIVATE.csv',index=False)
    pd.DataFrame(predictX,columns=names).to_csv(out/'predict_PRIVATE.csv',index=False)
    inbag=np.column_stack([np.bincount(cluster_bootstrap(tr['groups'],seed,i),minlength=len(df)) for i in range(trees)])
    np.savetxt(out/'inbag_PRIVATE.csv',inbag,fmt='%d',delimiter=',')
    cfg=plan['standard_rsf']
    rconfig=dict(train=str(out/'train_PRIVATE.csv'),predict=str(out/'predict_PRIVATE.csv'),
                 inbag=str(out/'inbag_PRIVATE.csv'),output=str(out),trees=trees,seed=seed,
                 nodesize=cfg['nodesize'],depth=cfg['nodedepth'],nsplit=cfg['nsplit'],mtry=cfg['mtry'])
    write(out/'r_config_PRIVATE.json',rconfig)
    env=os.environ.copy();env['EW_RLIB']=RLIB
    with (out/'R.log').open('x') as log:
        subprocess.run([RSCRIPT,'--vanilla',str(Path(__file__).with_name('reference.R')),str(out/'r_config_PRIVATE.json')],
                       env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    meta=json.loads((out/'r_meta.json').read_text())
    values=np.fromfile(out/'cif_PRIVATE.bin',dtype='<f8').reshape(meta['dims'],order='F')
    grid=np.atleast_1d(np.loadtxt(out/'times_PRIVATE.csv'))
    return values,grid,meta


def run(args):
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
    started=time.perf_counter();cpu=time.process_time()
    src=Path(os.environ['DATA_DIR']);out=Path(os.environ['OUTPUT_DIR']);out.mkdir(exist_ok=True,parents=True)
    plan=json.loads((src/'plan.json').read_text())
    for name,digest in plan['source_hashes'].items():assert sha(src/name)==digest
    tr=arrays(src/'train_PRIVATE.npz');ev=arrays(src/'eval_PRIVATE.npz')
    assert len(np.unique(tr['groups']))==len(tr['groups'])
    assert set(tr['event'])=={0,1,2}
    X=np.vstack([tr['X'],ev['X']]);n=len(tr['X']);times=np.array([365.,366.,730.,731.])
    trees=plan['pilot']['trees'] if args.pilot else plan['forest_common']['n_trees']
    row=dict(model=args.model,seed=args.seed,pilot=args.pilot,training_companies=n,evaluation_companies=len(ev['X']),
        input_columns=X.shape[1],trees=trees,training_termination_events=int((tr['event']==1).sum()),
        training_recovery_events=int((tr['event']==2).sum()),training_censored=int((tr['event']==0).sum()),
        training_h24_support=int((tr['time']>=tr['h24']).sum()),training_constant_or_empty_columns=int(sum(np.unique(v[np.isfinite(v)]).size<2 for v in tr['X'].T)),
        host=socket.gethostname(),affinity_count=len(os.sched_getaffinity(0)))
    fit_start=time.perf_counter()
    if args.model=='standard_rsf':
        native,grid,meta=reference_fit(tr,X,plan,args.seed,trees,out)
        ix=np.searchsorted(grid,times,side='right')-1
        prediction=native[:,np.maximum(ix,0)].copy();prediction[:,ix<0]=0
        row.update(implementation='randomForestSRC',reload_error=meta['reload_error'],leaves=meta['leaves'],
                   r_version=meta['version'],fit_predict_seconds=time.perf_counter()-fit_start,
                   distinct_thresholds=None,max_path_variables=None,main_functions=None,interaction_functions=None,
                   spatial_table_cells=None,reconstruction_error=None)
    else:
        config={**plan['forest_common'],**plan['forest_variants'][args.model], 'seed':args.seed,'n_trees':trees}
        f=CompetingRiskForest(ForestConfig(**config)).fit(tr['X'],tr['time'],tr['event'],tr['groups'])
        row['fit_seconds']=time.perf_counter()-fit_start
        f.save(out/'forest_PRIVATE.json');again=CompetingRiskForest.load(out/'forest_PRIVATE.json')
        prediction=f.predict(X,times);native=f.predict(X);grid=f.grid_
        row.update(implementation='crforest-reference',**f.audit_,reload_error=float(abs(prediction-again.predict(X,times)).max()))
        if args.model=='xai_q2':
            ex=CIFExpansion.from_forest(f,tr['X']);ex.save(out/'functions_PRIVATE.npz')
            ex2=CIFExpansion.load(out/'functions_PRIVATE.npz')
            cc=ex.complexity(tr['X'],support_min=10)
            write(out/'complexity_PRIVATE.json',cc)
            row.update({k:v for k,v in cc.items() if k!='functions'})
            row['reconstruction_error']=float(abs(ex.predict(X,times)-prediction).max())
            row['function_reload_error']=float(abs(ex2.predict(X,times)-prediction).max())
            row['native_grid_reconstruction_error']=float(abs(ex.predict(X[:min(16,n)])-native[:min(16,n)]).max())
            keys=[(j,) for j in ex.mains]+ex.pairs
            terms=np.stack([ex.component(X,k,times) for k in keys],axis=1) if keys else np.zeros((len(X),0,len(times),2))
            _,mu=ex._coeff(times)
            row['complete_component_sum_error']=float(abs(mu+terms.sum(1)-prediction).max())
            assert max(row[k] for k in ['reconstruction_error','function_reload_error','native_grid_reconstruction_error','complete_component_sum_error'])<1e-10
            np.savez_compressed(out/'components_PRIVATE.npz',keys=np.array([','.join(map(str,k)) for k in keys]),terms=terms,baseline=mu,times=times)
            # Sparse support and term concentration are descriptive, not confidence.
            importance=np.mean(abs(terms[:n]),axis=0)
            np.savez_compressed(out/'importance_PRIVATE.npz',importance=importance,keys=np.array([','.join(map(str,k)) for k in keys]))
            row['top10_term_absolute_share_h24_delist']=float(np.sort(importance[:,2,0])[-10:].sum()/importance[:,2,0].sum()) if importance[:,2,0].sum()>0 else 0.
    assert row['reload_error']<1e-12
    assert np.isfinite(native).all() and native.min()>=-1e-12 and native.max()<=1+1e-12
    assert np.max(native.sum(2))<=1+1e-10 and np.min(np.diff(native,axis=1))>=-1e-12
    assert np.isfinite(prediction).all()
    row.update(probability_min=float(native.min()),probability_max=float(native.max()),max_cause_probability_sum=float(native.sum(2).max()),
               native_time_points=len(grid),latest_native_event_day=float(grid.max()))
    np.savez_compressed(out/'predictions_PRIVATE.npz',sample_ids=ev['sample_ids'],times=times,cif=prediction[n:],training_cif=prediction[:n])
    np.savez_compressed(out/'native_curves_PRIVATE.npz',grid=grid,cif=native[n:])
    row.update(status='COMPLETE',seconds=time.perf_counter()-started,cpu_seconds=time.process_time()-cpu,
        peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        child_peak_rss_MiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss/1024,
        input_hashes=plan['source_hashes'],code_commit=plan['code_commit'])
    write(out/'summary.json',row);write(out/'hashes.json',{p.name:sha(p) for p in out.iterdir() if p.is_file()})
    print(json.dumps({k:v for k,v in row.items() if k!='input_hashes'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--model',choices=['xai_q2','unrestricted','standard_rsf'],required=True)
    parser.add_argument('--seed',type=int,required=True);parser.add_argument('--pilot',action='store_true');args=parser.parse_args()
    try:run(args)
    except Exception:
        out=Path(os.environ['OUTPUT_DIR']);out.mkdir(parents=True,exist_ok=True)
        write(out/'FAILED.json',dict(args=vars(args),traceback=traceback.format_exc()));raise
