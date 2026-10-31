"""Independent wide-feature forest or historical-only ensemble calibration."""
from pathlib import Path
import argparse,json,os,resource,socket,time,traceback
import numpy as np
import pandas as pd
from crforest import CompetingRiskForest,CIFExpansion,aalen_johansen
from crforest.shrinkage import shrink_leaf_cifs
from experiments.post_st_ablation_v1.verification import merge,replay
from .common import *
from .affine import fit_affine,forest_envelope,hull,transform,transformed_expansion

def verify_export(f,tr,ev,out,cal=None):
    out.mkdir();started=time.perf_counter();X=np.vstack([tr['X'],ev['X']])
    base=CIFExpansion.from_forest(f,tr['X'])
    ex=base if cal is None else transformed_expansion(base,cal)
    compile_seconds=time.perf_counter()-started
    ex.save(out/'functions_PRIVATE.npz');again=CIFExpansion.load(out/'functions_PRIVATE.npz')
    f.save(out/'forest_PRIVATE.json');loaded=CompetingRiskForest.load(out/'forest_PRIVATE.json')
    if cal is not None:write(out/'calibrator_PRIVATE.json',cal)
    times=np.unique(np.r_[0.,f.grid_,ex.grid,TIMES]);keys=[(j,) for j in ex.mains]+ex.pairs
    errors=dict(full_reconstruction=0.,explicit_component_sum=0.,function_reload=0.,forest_reload=0.)
    minimum=0.;maximum=0.;increment=0.;start=time.perf_counter()
    for begin in range(0,len(X),32):
        xx=X[begin:begin+32];raw=f.predict(xx,times)
        pred=raw if cal is None else transform(raw,times,cal)
        errors['forest_reload']=max(errors['forest_reload'],float(abs(loaded.predict(xx,times)-raw).max()))
        errors['full_reconstruction']=max(errors['full_reconstruction'],float(abs(ex.predict(xx,times)-pred).max()))
        errors['function_reload']=max(errors['function_reload'],float(abs(again.predict(xx,times)-pred).max()))
        total=np.broadcast_to(ex._coeff(times)[1],pred.shape).copy()
        for key in keys:total+=ex.component(xx,key,times)
        errors['explicit_component_sum']=max(errors['explicit_component_sum'],float(abs(total-pred).max()))
        minimum=min(minimum,float(pred.min()));maximum=max(maximum,float(pred.sum(-1).max()))
        increment=min(increment,float(np.diff(pred,axis=1).min()))
    assert max(errors.values())<=1e-10 and minimum>=-1e-12 and maximum<=1+1e-10 and increment>=-1e-12
    p=f.predict(ev['X'],TIMES);p=p if cal is None else transform(p,TIMES,cal)
    np.savez_compressed(out/'predictions_PRIVATE.npz',times=TIMES,sample_ids=ev['sample_ids'],cif=p)
    components=np.stack([ex.component(tr['X'],k,TIMES) for k in keys],axis=1) if keys else np.empty((len(tr['X']),0,4,2))
    np.savez_compressed(out/'components_PRIVATE.npz',keys=np.array([','.join(map(str,k)) for k in keys]),terms=components,times=TIMES)
    complexity=base.complexity(tr['X'],support_min=10)
    write(out/'complexity_PRIVATE.json',complexity)
    summary=dict(**errors,**f.audit(),**{k:v for k,v in complexity.items() if k!='functions'},
        verified_rows=len(X),verified_times=len(times),native_times=len(f.grid_),probability_min=minimum,
        probability_sum_max=maximum,minimum_increment=increment,compile_seconds=compile_seconds,
        verification_seconds=time.perf_counter()-start,function_bytes=(out/'functions_PRIVATE.npz').stat().st_size,
        candidate_attempts=f.candidate_evaluations_)
    write(out/'verification.json',summary);return summary

def fit_wide(a,root,out):
    src=root/'folds'/str(a.year);tr=arrays(src/'train_PRIVATE.npz');ev=arrays(src/'eval_PRIVATE.npz')
    assert set(ev)=={'X','sample_ids'}
    cfg=configuration(a.seed);pp=pairing(tr['groups'],cfg);ref=pairing(tr['groups'],configuration(a.seed,'reference'))
    for x,y in zip(pp,ref):
        assert x['bootstrap_sha256']==y['bootstrap_sha256'] and x['multiplicity_sha256']==y['multiplicity_sha256']
    start=time.perf_counter();f=CompetingRiskForest(cfg).fit(tr['X'],tr['time'],tr['event'],tr['groups']);fit_seconds=time.perf_counter()-start
    for i,b in enumerate(f.inbag_):assert array_hash(b)==pp[i]['multiplicity_sha256']
    start=time.perf_counter();sh=shrink_leaf_cifs(f,tr['X'],tr['time'],tr['event'],tr['groups'],10);shrink_seconds=time.perf_counter()-start
    checks={}
    for name,m in [('raw',f),('shrunk',sh)]:checks[name]=verify_export(m,tr,ev,out/name)
    nodes,features,pairs,leaves,replay_checks=replay(sh,tr)
    for n,v in [('node_search',nodes),('feature_coverage',features),('pair_coverage',pairs),('leaf_support',leaves)]:pd.DataFrame(v).to_csv(out/f'{n}_PRIVATE.csv',index=False)
    write(out/'pairing.json',pp)
    return dict(mode='wide',year=a.year,seed=a.seed,new_forest_fits=1,derived_shrunk_models=1,calibration_fits=0,
        fit_seconds=fit_seconds,shrink_seconds=shrink_seconds,checks=checks,replay_checks=replay_checks,
        actual_scored_candidates=int(sum(r['scored_candidates'] for r in nodes)))

def calibrate(a,root,out):
    src=root/'folds'/str(a.year);tr=arrays(src/'train_PRIVATE.npz');ev=arrays(src/'eval_PRIVATE.npz')
    references=[CompetingRiskForest.load(root/'reference'/str(a.year)/str(s)/'forest_PRIVATE.json') for s in SEEDS]
    f=merge(references);vertices=forest_envelope(f)
    folder=root/'calibration'/str(a.year)/str(a.horizon);labels=arrays(folder/'labels_PRIVATE.npz')
    ys=json.loads((folder/'cohorts.json').read_text());ps=[];ids=[];source_hashes={}
    for v in ys:
        members=[]
        for s in SEEDS:
            p=root/'rolling'/str(v)/str(s)/'predictions_PRIVATE.npz';z=arrays(p);members.append(z);source_hashes[str(p)]=sha(p)
        for z in members[1:]:np.testing.assert_array_equal(z['sample_ids'],members[0]['sample_ids'])
        ps.append(np.mean([z['cif'] for z in members],axis=0));ids.extend(members[0]['sample_ids'].tolist())
    np.testing.assert_array_equal(np.array(ids),labels['sample_ids'])
    p=horizon_predictions(np.concatenate(ps),TIMES,labels[f'h{a.horizon}'])
    y=np.column_stack([(labels['event']==c)&(labels['analysis_time']<=a.horizon) for c in [1,2]]).astype(float)
    assert np.all((labels['analysis_time']>=a.horizon)|(labels['event']>0))
    # OOF p values are constrained too; their historical tree envelopes need not
    # be certified because this calibrator is deployed only on the current forest.
    vertices=hull(np.vstack([vertices,p]));start=time.perf_counter()
    cal=fit_affine(p,y,vertices,y.sum(0));solver_seconds=time.perf_counter()-start
    cap=365. if a.horizon==12 else 730.
    grid=np.unique(np.r_[0.,np.minimum(f.grid_,cap),cap])
    aj=aalen_johansen(tr['time'],tr['event'],grid).sum(1);assert aj[-1]>0
    cal.update(shape_grid=grid.tolist(),shape_values=(aj/aj[-1]).tolist(),year=a.year,horizon_months=a.horizon,
        calibration_companies=len(y),entry_years=ys,source_prediction_hashes=source_hashes,
        certificate='terminal Minkowski leaf envelope; all inputs/all times; no clipping',
        intermediate_time_calibration_claim=False)
    write(out/'calibrator_PRIVATE.json',cal);np.savez_compressed(out/'certificate_PRIVATE.npz',vertices=vertices,p=p,y=y)
    check=verify_export(f,tr,ev,out/'calibrated',cal)
    # A public summary omits actual parameters; all diagnostics remain available.
    return dict(mode='calibrate',year=a.year,horizon=a.horizon,new_forest_fits=0,calibration_fits=1,
        calibration_companies=len(y),cause_events=y.sum(0).astype(int).tolist(),entry_years=ys,
        solver_seconds=solver_seconds,gap=cal['gap'],primal_violation=cal['primal_violation'],
        objective_before=cal['objective_before'],objective_after=cal['objective_after'],
        fixed_sparse_slopes=cal['frozen_slope_for_sparse_cause'],certificate_vertices=len(vertices),
        identity_change_max=float(max(abs(np.asarray(cal['c'])).max(),abs(np.asarray(cal['b'])-1).max())),
        maximum_terminal_sum_bound=cal['maximum_terminal_sum_bound'],checks=check)

def main(a):
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3));start=time.perf_counter();cpu=time.process_time()
    root=Path(os.environ['DATA_DIR']);out=Path(os.environ['OUTPUT_DIR']);out.mkdir(parents=True,exist_ok=True)
    row=(fit_wide if a.mode=='wide' else calibrate)(a,root,out)
    row.update(status='COMPLETE',host=socket.gethostname(),seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,
        peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,no_test_labels_read=True)
    write(out/'summary.json',row);write(out/'hashes.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})
    print(json.dumps(row),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['wide','calibrate'],required=True);p.add_argument('--year',type=int,required=True);p.add_argument('--seed',type=int,default=0);p.add_argument('--horizon',type=int,choices=[12,24]);a=p.parse_args()
    try:main(a)
    except Exception:
        out=Path(os.environ['OUTPUT_DIR']);out.mkdir(parents=True,exist_ok=True);write(out/'FAILED.json',dict(args=vars(a),traceback=traceback.format_exc()));raise
