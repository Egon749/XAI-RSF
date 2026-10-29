"""Explain one frozen year/seed pair; no training or access to test outcomes."""
import argparse,hashlib,json,os,platform,resource,subprocess,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import shap
from crforest import CompetingRiskForest,CIFExpansion
from .methods import (q2_adapter,rsf_adapter,load_rsf_export,route,predict,product_adapter,
                      default_tree_shap,coalition_interactions,product_components)


def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def errors(a,b):
    d=np.asarray(a)-np.asarray(b)
    return dict(max_abs=float(np.max(np.abs(d))),rmse=float(np.sqrt(np.mean(d*d))),mean_abs=float(np.mean(abs(d))))
def check(a,b):np.testing.assert_allclose(a,b,atol=1e-12,rtol=0)


def run(year,seed,out):
    start=time.perf_counter();resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
    out.mkdir(exist_ok=True);assert not any(out.iterdir())
    src=Path(os.environ['DATA_DIR'])/str(year)/str(seed);cfg=json.loads((src/'config.json').read_text())
    source_hashes={p.name:digest(p) for p in src.iterdir() if p.is_file()}
    z=np.load(src/'features_PRIVATE.npz',allow_pickle=False);n=cfg['n_train'];X=np.vstack([z['train_X'],z['test_X']]);p=X.shape[1]
    meds=np.load(src/'rsf_imputation_PRIVATE.npy');XR=np.where(np.isnan(X),meds,X)
    XR[n:].astype('<f8').tofile(out/'rsf_query_PRIVATE.bin')
    rconfig=dict(model=str(src/'rsf_forest_PRIVATE.rds'),query=str(out/'rsf_query_PRIVATE.bin'),
                 output=str(out/'r_export'),n_query=cfg['n_test'],p=p,horizon_days=cfg['horizon_days'])
    write(out/'r_config_PRIVATE.json',rconfig)
    env=dict(os.environ,EW_RLIB='/home/oliver/MainDiskData/ernet_original_tau0_full_20261015/dynamic_competing_h24_20261015/rsf_er_competing_20261016/environment/rlib')
    rscript='/home/oliver/MainDiskData/financial_diagnosis/csf_joint_sensitivity_v1/environment/r/bin/Rscript'
    with (out/'r_export.log').open('w') as log:
        subprocess.run([rscript,str(Path(__file__).with_name('export_rsf.R')),str(out/'r_config_PRIVATE.json')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    rex=load_rsf_export(out/'r_export');standard=rsf_adapter(rex)
    allrx=np.fromfile(out/'r_export/allx_PRIVATE.bin',dtype='<f8').reshape(-1,p)
    nativepred=np.fromfile(out/'r_export/original_prediction_PRIVATE.bin',dtype='<f8')
    memb=np.fromfile(out/'r_export/membership_PRIVATE.bin',dtype='<i4').reshape(-1,len(standard['trees']))
    check(allrx[:len(X)],XR)
    adaptererr=errors(predict(standard,allrx),nativepred);check(predict(standard,allrx),nativepred)
    for b,t in enumerate(standard['trees']):np.testing.assert_array_equal(t['leaf_ids'][route(t,allrx,True)],memb[:,b])
    qforest=CompetingRiskForest.load(src/'q2_forest_PRIVATE.json');qmodel=q2_adapter(qforest,cfg['horizon_days'])
    nativeq=qforest.predict(X,[cfg['horizon_days']])[:,0,0];check(predict(qmodel,X),nativeq)
    ex=CIFExpansion.load(src/'q2_functions_PRIVATE.npz');check(ex.predict(X,[cfg['horizon_days']])[:,0,0],nativeq)
    for label,expected in [('q2',nativeq[n:]),('rsf',nativepred[n:len(X)])]:
        zz=np.load(src/f'{label}_predictions_PRIVATE.npz',allow_pickle=False)
        np.testing.assert_array_equal(zz['sample_ids'],z['test_ids']);ix=np.flatnonzero(zz['times']==cfg['horizon_days']);assert len(ix)==1
        check(expected,zz['cif'][:,ix[0],0])
    savedPhi=np.zeros((len(X),p,p))
    for j in range(p):savedPhi[:,j,j]=ex.component(X,(j,),[cfg['horizon_days']])[:,0,0]
    for j,l in ex.pairs:
        v=ex.component(X,(j,l),[cfg['horizon_days']])[:,0,0]/2;savedPhi[:,j,l]=v;savedPhi[:,l,j]=v
    np.savez_compressed(out/'q2_native_PRIVATE.npz',Phi=savedPhi,baseline=ex._coeff([cfg['horizon_days']])[1][0,0],predictions=nativeq)
    checks=[];metrics=[];importance=[];dispersion=[];order_rows=[]
    for label,model,xx,original in [('standard_rsf',standard,XR,nativepred[:len(X)]),('q2',qmodel,X,nativeq)]:
        ref=xx[:n];prod=product_adapter(model,ref);comp=product_components(model,xx,ref)
        check(comp['order'].sum(1),original)
        keys=sorted(comp['terms'],key=lambda k:(len(k),k))
        np.savez_compressed(out/f'{label}_decomposition_PRIVATE.npz',keys=np.array([','.join(map(str,k)) for k in keys]),
                            terms=np.stack([comp['terms'][k] for k in keys],axis=1),order=comp['order'],
                            higher_phi=comp['higher_phi'],base=comp['base'],predictions=original)
        for background,m in [('default',model),('product',prod)]:
            calcstart=time.perf_counter();got=default_tree_shap(m,xx);base,phi,Phi=coalition_interactions(m,xx)
            ee=dict(official_vs_coalition_phi=errors(got['phi'],phi),official_vs_coalition_Phi=errors(got['Phi'],Phi),
                    prediction=errors(got['adapter_prediction'],original),additivity=errors(got['base']+got['phi'].sum(1),original),
                    interaction_rows=errors(got['Phi'].sum(2),got['phi']),symmetry=errors(got['Phi'],got['Phi'].transpose(0,2,1)))
            for e in ee.values():assert e['max_abs']<1e-12,e
            check(base,got['base'])
            if background=='product':
                check(got['base'],comp['base']);check(got['phi'],comp['shap_from_terms'])
                ee['exact_order_allocation']=errors(got['phi'],comp['shap_from_terms'])
            if label=='q2':
                ee['native_diagonal']=errors(got['Phi'][:,np.arange(p),np.arange(p)],savedPhi[:,np.arange(p),np.arange(p)])
                off=~np.eye(p,dtype=bool)
                ee['native_offdiagonal']=errors(got['Phi'][:,off],savedPhi[:,off])
                ee['native_baseline']=errors(got['base'],ex._coeff([cfg['horizon_days']])[1][0,0])
                if background=='product':
                    check(got['Phi'],savedPhi)
            np.savez_compressed(out/f'{label}_{background}_PRIVATE.npz',base=got['base'],phi=got['phi'],Phi=got['Phi'],predictions=original)
            checks.append(dict(model=label,background=background,errors=ee,seconds=time.perf_counter()-calcstart,resolved_perturbation=got['resolved_perturbation']))
            cuts={j:np.unique(np.concatenate([t['thresholds'][t['features']==j] for t in model['trees']])) for j in range(p)}
            for subset,sl in [('training',slice(0,n)),('evaluation',slice(n,None))]:
                vals=got['phi'][sl];diag=got['Phi'][sl,np.arange(p),np.arange(p)]
                for j in range(p):
                    importance.append(dict(model=label,background=background,subset=subset,feature=j,
                        mean_abs_phi=float(abs(vals[:,j]).mean()),mean_abs_diagonal=float(abs(diag[:,j]).mean())))
                    group=np.searchsorted(cuts[j],xx[sl,j],side='left');group[np.isnan(xx[sl,j])]=len(cuts[j])+1
                    residual=0.;n_repeat=0;max_range=0.
                    for g in np.unique(group):
                        v=vals[group==g,j]
                        if len(v)>=2:
                            residual+=float(((v-v.mean())**2).sum());n_repeat+=len(v);max_range=max(max_range,float(np.ptp(v)))
                    dispersion.append(dict(model=label,background=background,subset=subset,feature=j,
                        within_cell_rms=float(np.sqrt(residual/max(n_repeat,1))),n_repeated_cell_rows=n_repeat,max_within_cell_range=max_range))
                row=dict(model=label,background=background,subset=subset,n=len(vals),baseline=got['base'],
                         ordinary_shap_rms=float(np.sqrt(np.mean(vals**2))),max_additivity_error=ee['additivity']['max_abs'])
                if label=='q2':
                    delta=got['Phi'][sl]-savedPhi[sl]
                    row.update(main_max_abs=float(abs(delta[:,np.arange(p),np.arange(p)]).max()),main_rmse=float(np.sqrt(np.mean(delta[:,np.arange(p),np.arange(p)]**2))),
                               interaction_max_abs=float(abs(delta[:,~np.eye(p,dtype=bool)]).max()),interaction_rmse=float(np.sqrt(np.mean(delta[:,~np.eye(p,dtype=bool)]**2))),
                               native_baseline_difference=got['base']-comp['base'])
                metrics.append(row)
        for subset,sl in [('training',slice(0,n)),('evaluation',slice(n,None))]:
            for r in range(1,comp['max_order']+1):
                v=comp['order'][sl,r];rr=[k for k in keys if len(k)==r]
                order_rows.append(dict(model=label,subset=subset,order=r,structural_functions=len(rr),
                    nonzero_on_evaluated_rows=sum(np.any(abs(comp['terms'][k][sl])>1e-14) for k in rr),
                    combined_order_mean_abs=float(abs(v).mean()),combined_order_rms=float(np.sqrt(np.mean(v*v))),combined_order_max_abs=float(abs(v).max())))
            high=comp['order'][sl,3:].sum(1) if comp['max_order']>=3 else np.zeros(len(xx[sl]))
            order_rows.append(dict(model=label,subset=subset,order='>=3',structural_functions=sum(len(k)>=3 for k in keys),
                nonzero_on_evaluated_rows=sum(len(k)>=3 and np.any(abs(comp['terms'][k][sl])>1e-14) for k in keys),
                combined_order_mean_abs=float(abs(high).mean()),combined_order_rms=float(np.sqrt(np.mean(high*high))),combined_order_max_abs=float(abs(high).max()),
                allocated_higher_order_shap_rms=float(np.sqrt(np.mean(comp['higher_phi'][sl]**2)))))
    for filename,rows in [('alignment_metrics',metrics),('importance_PRIVATE',importance),('dispersion_PRIVATE',dispersion),('order_metrics',order_rows)]:
        for row in rows:row.update(year=year,seed=seed,horizon_days=cfg['horizon_days'])
        pd.DataFrame(rows).to_csv(out/f'{filename}.csv',index=False)
    write(out/'checks.json',checks)
    after={p.name:digest(p) for p in src.iterdir() if p.is_file()};assert after==source_hashes
    summary=dict(status='COMPLETE',year=year,seed=seed,new_model_fits=0,forests_reused=2,primary_arms=3,diagnostic_controls=1,
                 train_rows=n,test_rows=cfg['n_test'],boundary_probe_rows=rex['n_boundary'],horizon_days=cfg['horizon_days'],
                 original_hashes=source_hashes,rsf_adapter_error=adaptererr,rsf_aggregation_error=rex['aggregation_error'],
                 default_call='shap.TreeExplainer(model_dict)',default_resolved='tree_path_dependent',
                 product_method='same tree outputs; exact marginal-product node-region probabilities; official TreeSHAP plus independent coalition verification',
                 seconds=time.perf_counter()-start,peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                 host=platform.node(),python=platform.python_version(),shap=shap.__version__,numpy=np.__version__,
                 R=rex['R'],randomForestSRC=rex['randomForestSRC'],affinity_count=len(os.sched_getaffinity(0)))
    write(out/'summary.json',summary);write(out/'hashes.json',{str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file()})
    print(json.dumps({k:summary[k] for k in ['status','year','seed','new_model_fits','seconds','peak_rss_MiB','host']}))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--year',type=int,required=True);a.add_argument('--seed',type=int,required=True);args=a.parse_args()
    output=Path(os.environ['OUTPUT_DIR'])
    try:run(args.year,args.seed,output)
    except Exception:
        output.mkdir(exist_ok=True);write(output/'FAILED.json',dict(year=args.year,seed=args.seed,traceback=traceback.format_exc(),new_model_fits=0));raise
