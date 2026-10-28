"""Fit calibration on historical held-out data, freeze, then evaluate future data."""
from pathlib import Path
import argparse,json,datetime
import numpy as np
import pandas as pd
from crforest import CIFExpansion
from crforest.core import aalen_johansen
from experiments.post_st_prognosis_v1.common import arrays,sha,write,endpoint_weights,horizon_predictions
from experiments.post_st_train2024_h12_v1.evaluate import metric
from experiments.post_st_train2024_h12_v1.validate import direct_metrics
from .calibration import fit_affine,transform,shape_at


def forecasts(run):
    gate=json.loads((run/'audit/formal_gate.json').read_text());assert gate['status']=='PASS'
    manifest=[json.loads(s) for s in (run/'formal.jsonl').read_text().splitlines()]
    out={};details={};summaries=[]
    for i,task in enumerate(manifest):
        args=task['argv'];year=int(args[args.index('--year')+1]);model=args[args.index('--model')+1];seed=args[args.index('--seed')+1]
        folder=run/'results/formal'/str(gate['job'])/f'task-{i}'/'output'
        for name,h in json.loads((folder/'hashes.json').read_text()).items():assert sha(folder/name)==h
        data=arrays(folder/'predictions_PRIVATE.npz');key=(year,model,seed)
        assert key not in out;out[key]=data;details[key]=folder
        s=json.loads((folder/'summary.json').read_text());s.pop('input_hashes',None);s['test_year']=year;summaries.append(s)
    assert len(out)==18
    for year in [2023,2024]:
        for model in ['xai_q2','unrestricted','standard_rsf']:
            members=[out[(year,model,str(s))] for s in [42,101,202]]
            out[(year,model,'ensemble')]=dict(times=members[0]['times'],sample_ids=members[0]['sample_ids'],
                cif=np.mean([p['cif'] for p in members],axis=0))
    return out,details,summaries


def calibrate(run):
    predictions,_,_=forecasts(run);root=run/'calibration';root.mkdir(exist_ok=False)
    diagnostics=[];hashes={}
    for year in [2023,2024]:
        src=run/'input/folds'/str(year);cal=arrays(src/'calibration_PRIVATE.npz');tr=arrays(src/'train_PRIVATE.npz')
        grid=np.unique(np.r_[0.,tr['time'][tr['event']>0],365.,366.]);base=aalen_johansen(tr['time'],tr['event'],grid)
        folder=root/str(year);folder.mkdir()
        np.savez_compressed(folder/'shape_PRIVATE.npz',grid=grid,values=base)
        w,known,support=endpoint_weights(cal['analysis_time'],cal['event'],12.,cal['label_eligible'])
        y=np.c_[(cal['event']==1)&(cal['analysis_time']<=12),(cal['event']==2)&(cal['analysis_time']<=12)]
        for (yy,model,seed),p in predictions.items():
            if yy!=year:continue
            n=len(cal['X']);assert np.array_equal(cal['sample_ids'],p['sample_ids'][:n])
            risk=horizon_predictions(p['cif'][:n],p['times'],cal['h12'])
            np.testing.assert_array_equal(shape_at(cal['h12'],grid,base),np.ones_like(risk))
            result=fit_affine(risk,y,w)
            path=folder/f'{model}_{seed}_PRIVATE.json';write(path,result)
            hashes[str(path.relative_to(root))]=sha(path)
            diagnostics.append(dict(test_year=year,model=model,seed=seed,calibration_companies=n,
                known=int(known.sum()),unknown=int((~known).sum()),termination=int((known&y[:,0]).sum()),
                recovery=int((known&y[:,1]).sum()),ipcw_min_G=support['min_G'],
                **{k:v for k,v in result.items() if k not in ['c','b']}))
        hashes[str((folder/'shape_PRIVATE.npz').relative_to(root))]=sha(folder/'shape_PRIVATE.npz')
    pd.DataFrame(diagnostics).to_csv(run/'reports/calibration_solver.csv',index=False)
    write(root/'freeze.json',dict(status='COMPLETE',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        calibration_fits=24,test_labels_read=False,hashes=hashes,
        objective='joint IPCW Brier at 12 calendar months',parameters_chosen_using_test=False))
    print('24 calibration fits frozen before test evaluation; no test labels read.')


def check_curves(native,g,params):
    z=transform(native,g,params)
    assert np.isfinite(z).all() and z.min()>=-1e-12 and z.sum(-1).max()<=1+1e-10
    assert np.diff(z,axis=1).min()>=-1e-12 and np.max(abs(z[:,0]))<1e-12
    return dict(cif_min=float(z.min()),cif_max_sum=float(z.sum(-1).max()),minimum_time_increment=float(np.diff(z,axis=1).min()))


def evaluate(run):
    predictions,details,summaries=forecasts(run)
    frozen=json.loads((run/'calibration/freeze.json').read_text());assert frozen['status']=='COMPLETE'
    for name,h in frozen['hashes'].items():assert sha(run/'calibration'/name)==h
    out=run/'evaluation';out.mkdir(exist_ok=False);metrics=[];warnings=[];checks=[];complexity=[];bins=[];pair_deltas=[]
    for year in [2023,2024]:
        src=run/'input/folds'/str(year);tr=arrays(src/'train_PRIVATE.npz');cal=arrays(src/'calibration_PRIVATE.npz')
        ev=arrays(run/'evaluation_input'/str(year)/'test_PRIVATE.npz')
        meta=pd.read_parquet(run/'evaluation_input'/str(year)/'test_metadata_PRIVATE.parquet');ncal=len(cal['X'])
        shape=arrays(run/'calibration'/str(year)/'shape_PRIVATE.npz')
        year_predictions={}
        for (yy,model,seed),p in predictions.items():
            if yy!=year:continue
            assert np.array_equal(p['sample_ids'][ncal:],ev['sample_ids'])
            raw=horizon_predictions(p['cif'][ncal:],p['times'],ev['h12'])
            params=json.loads((run/'calibration'/str(year)/f'{model}_{seed}_PRIVATE.json').read_text())
            corrected=transform(raw,np.ones(2),params)
            order_equal=all(np.array_equal(np.lexsort((ev['sample_ids'],-raw[:,k])),
                np.lexsort((ev['sample_ids'],-corrected[:,k]))) for k in range(2))
            assert order_equal
            key=(year,model,seed)
            if seed!='ensemble':
                d=details[key];native=arrays(d/'native_curves_PRIVATE.npz')
                grid=np.unique(np.r_[0.,native['grid'],shape['grid'],365.,366.])
                ix=np.searchsorted(native['grid'],grid,side='right')-1
                values=native['cif'][ncal:][:,np.maximum(ix,0)].copy();values[:,ix<0]=0.
                g=shape_at(grid,shape['grid'],shape['values']);valid=check_curves(values,g,params)
                error=None
                if model=='xai_q2':
                    ex=CIFExpansion.load(d/'functions_PRIVATE.npz');X=np.vstack([tr['X'],cal['X'],ev['X']])
                    times=np.array([0.,90.,180.,365.,366.]);gg=shape_at(times,shape['grid'],shape['values'])
                    _,mu=ex._coeff(times);keys=[(j,) for j in ex.mains]+ex.pairs
                    contributions=sum((ex.component(X,k,times) for k in keys),np.zeros((len(X),len(times),2)))
                    reconstructed=transform(mu,gg,params)+contributions*np.array(params['b'])
                    error=float(abs(reconstructed-transform(ex.predict(X,times),gg,params)).max())
                    assert error<1e-10
                    original=np.mean(abs(np.stack([ex.component(tr['X'],k,[365.])[:,0] for k in keys])),axis=1)
                    scaled=original*np.asarray(params['b']);relative_error=float(abs(original/original.sum(0)-scaled/scaled.sum(0)).max())
                    assert relative_error<1e-12
                    cc=ex.complexity(tr['X'],support_min=10)
                    complexity.append(dict(test_year=year,model=model,seed=seed,main_functions=len(ex.mains),
                        pair_functions=len(ex.pairs),spatial_table_cells=cc['spatial_table_cells'],
                        structure_unchanged=True,reconstruction_error=error,relative_importance_error=relative_error,
                        cross_seed_selection_stability_unchanged=True))
                checks.append(dict(test_year=year,model=model,seed=seed,rank_unchanged=order_equal,
                    reconstruction_error=error,**valid))
            else:
                # Common affine coefficients applied after fixed seed averaging.
                if model=='xai_q2':
                    recon=[]
                    for ss in ['42','101','202']:
                        ex=CIFExpansion.load(details[(year,model,ss)]/'functions_PRIVATE.npz')
                        _,mu=ex._coeff([365.,366.]);keys=[(j,) for j in ex.mains]+ex.pairs
                        recon.append(mu+sum((ex.component(ev['X'],k,[365.,366.]) for k in keys),np.zeros((len(ev['X']),2,2))))
                    er=float(abs(transform(horizon_predictions(np.mean(recon,axis=0),[365.,366.],ev['h12']),np.ones(2),params)-corrected).max())
                    assert er<1e-10
                    checks.append(dict(test_year=year,model=model,seed=seed,rank_unchanged=True,reconstruction_error=er))
            year_predictions[(model,seed,'uncalibrated')]=raw
            year_predictions[(model,seed,'affine')]=corrected
        trainbase=aalen_johansen(tr['time'],tr['event'],np.array([365.,366.]))
        year_predictions[('training_AJ','none','baseline')]=trainbase[np.searchsorted([365,366],ev['h12'])]
        recent=aalen_johansen(cal['analysis_time'],cal['event'],np.array([12.]))
        year_predictions[('calibration_AJ','none','baseline')]=np.broadcast_to(recent[0],(len(ev['X']),2))
        for (model,seed,state),risk in year_predictions.items():
            for k,cause in enumerate(['termination','recovery']):
                result,w,y,known=metric(ev['analysis_time'],ev['event'],12.,risk[:,k],k+1,ev['label_eligible'])
                assert known.all();np.testing.assert_array_equal(w,np.ones(len(w)))
                np.testing.assert_allclose(direct_metrics(y,risk[:,k]),[result['auc'],result['ap'],result['brier']],rtol=0,atol=5e-14,equal_nan=True)
                base=dict(test_year=year,model=model,seed=seed,state=state,cause=cause)
                metrics.append({**base,**result})
                if seed in ['ensemble','none']:
                    for j,ids in enumerate(np.array_split(np.argsort(risk[:,k],kind='stable'),4)):
                        bins.append({**base,'bin':j+1,'n':len(ids),'predicted':float(risk[ids,k].mean()),'observed':float(y[ids].mean())})
                if k:continue
                for fraction in [.2,.25]:
                    ids=np.lexsort((ev['sample_ids'],-risk[:,k]))[:int(np.ceil(fraction*len(ev['X'])))];hit=y[ids]
                    warnings.append({**base,'top_fraction':fraction,'checked':len(ids),'captured':int(hit.sum()),
                        'target_events':int(y.sum()),'coverage':float(hit.sum()/y.sum()),'known_non_target':int((~hit).sum()),'unknown':0,
                        'lead3':int((hit & (ev['time'][ids]>=np.array([(o+pd.DateOffset(months=3)-o).days for o in meta.origin.iloc[ids]]))).sum()),
                        'lead6':int((hit & (ev['time'][ids]>=np.array([(o+pd.DateOffset(months=6)-o).days for o in meta.origin.iloc[ids]]))).sum())})
        for (model,seed,state),p in year_predictions.items():
            if state!='affine':continue
            old=year_predictions[(model,seed,'uncalibrated')]
            for k,cause in enumerate(['termination','recovery']):
                y=ev['event']==k+1
                pair_deltas.append(dict(test_year=year,model=model,seed=seed,cause=cause,
                    brier_change=float(np.mean((p[:,k]-y)**2-(old[:,k]-y)**2)),
                    risk_mean_change=float(np.mean(p[:,k]-old[:,k]))))
    df=pd.DataFrame(metrics);df.to_csv(out/'metrics.csv',index=False)
    numeric=['auc','ap','brier','risk_mean','ipcw_observed_incidence']
    ss=df[df.seed.isin(['42','101','202'])].groupby(['test_year','model','state','cause'])[numeric].agg(['mean','min','max'])
    ss.columns=['_'.join(x) for x in ss.columns];ss.reset_index().to_csv(out/'seed_summary.csv',index=False)
    pd.DataFrame(warnings).to_csv(out/'warning_metrics.csv',index=False)
    pd.DataFrame(checks).to_csv(out/'mathematical_checks.csv',index=False)
    pd.DataFrame(complexity).to_csv(out/'interpretability.csv',index=False)
    pd.DataFrame(bins).to_csv(out/'calibration_bins.csv',index=False)
    pd.DataFrame(pair_deltas).to_csv(out/'paired_changes.csv',index=False)
    pd.DataFrame(summaries).to_csv(out/'training_diagnostics.csv',index=False)
    for key,g in df[df.state.isin(['uncalibrated','affine'])].groupby(['test_year','model','seed','cause']):
        a=g.set_index('state');np.testing.assert_allclose(a.loc['uncalibrated',['auc','ap']].to_numpy(float),a.loc['affine',['auc','ap']].to_numpy(float),rtol=0,atol=1e-12,equal_nan=True)
    write(out/'validation.json',dict(status='PASS',formal_models=18,calibration_fits=24,independent_metric_rows=len(df),
        all_test_endpoints_known=True,rank_metrics_unchanged=True,test_outcomes_used_for_calibration=False,
        q2_seed_and_ensemble_reconstruction=True,global_simplex_constraint_certified=True,
        no_probability_clipping=True,reference_distribution_unchanged=True,
        calibration_freeze_sha256=sha(run/'calibration/freeze.json')))
    print(df[(df.seed.isin(['ensemble','none']))&(df.cause=='termination')][['test_year','model','state','auc','ap','brier','risk_mean','ipcw_observed_incidence']].to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--stage',choices=['calibrate','evaluate'],required=True)
    a=p.parse_args();calibrate(a.run) if a.stage=='calibrate' else evaluate(a.run)
