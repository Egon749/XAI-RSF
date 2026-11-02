from pathlib import Path
import argparse,os,resource,socket,time,traceback
import numpy as np
from crforest import CompetingRiskForest,CIFExpansion
from experiments.post_st_ablation_v1.verification import merge
from experiments.post_st_level_coverage_v1.affine import transform,transformed_expansion
from .common import *
from .calibration import fit

def verify_export(f,tr,ev,out,cal):
    out.mkdir(parents=True);start=time.perf_counter();base=CIFExpansion.from_forest(f,tr['X'])
    ex=base if cal is None else transformed_expansion(base,cal);compile_seconds=time.perf_counter()-start
    ex.save(out/'functions_PRIVATE.npz');reloaded=CIFExpansion.load(out/'functions_PRIVATE.npz')
    f.save(out/'fixed_forest_PRIVATE.json');again=CompetingRiskForest.load(out/'fixed_forest_PRIVATE.json')
    if cal is not None:write(out/'calibrator_PRIVATE.json',cal)
    X=np.vstack([tr['X'],ev['X']]);grid=np.unique(np.r_[0,f.grid_,ex.grid,TIMES]);keys=[(j,) for j in ex.mains]+ex.pairs
    errors=dict(reconstruction=0.,explicit_component_sum=0.,function_reload=0.,forest_reload=0.)
    minimum=0.;maxsum=0.;increment=0.;start=time.perf_counter()
    for first in range(0,len(X),32):
        xx=X[first:first+32];raw=f.predict(xx,grid);expected=raw if cal is None else transform(raw,grid,cal)
        errors['forest_reload']=max(errors['forest_reload'],float(abs(raw-again.predict(xx,grid)).max()))
        for key,obj in [('reconstruction',ex),('function_reload',reloaded)]:errors[key]=max(errors[key],float(abs(obj.predict(xx,grid)-expected).max()))
        total=np.broadcast_to(ex._coeff(grid)[1],expected.shape).copy()
        for key in keys:total+=ex.component(xx,key,grid)
        errors['explicit_component_sum']=max(errors['explicit_component_sum'],float(abs(total-expected).max()))
        minimum=min(minimum,float(expected.min()));maxsum=max(maxsum,float(expected.sum(-1).max()));increment=min(increment,float(np.diff(expected,axis=1).min()))
    assert max(errors.values())<=1e-10 and minimum>=-1e-12 and maxsum<=1+1e-10 and increment>=-1e-12
    for label,xx in [('eval',ev['X']),('train',tr['X'])]:
        raw=f.predict(xx,TIMES);p=raw if cal is None else transform(raw,TIMES,cal)
        np.savez_compressed(out/f'{label}_predictions_PRIVATE.npz',times=TIMES,cif=p,sample_ids=ev['sample_ids'] if label=='eval' else tr['sample_ids'])
    cc=ex.complexity(tr['X']);old=base.complexity(tr['X'])
    assert cc['main_functions']==old['main_functions'] and cc['interaction_functions']==old['interaction_functions'] and cc['spatial_table_cells']==old['spatial_table_cells']
    # Positive uniform cause slopes preserve every function; offset affects mu only.
    if cal is not None:
        for key in keys:
            np.testing.assert_allclose(ex.component(tr['X'],key,TIMES),base.component(tr['X'],key,TIMES)*np.asarray(cal['b']),rtol=0,atol=1e-12)
    audit=dict(**errors,**f.audit(),**{k:v for k,v in cc.items() if k!='functions'},
        rows=len(X),time_points=len(grid),original_native_times=len(f.grid_),expanded_native_times=len(ex.grid),
        probability_min=minimum,sum_max=maxsum,minimum_increment=increment,compile_seconds=compile_seconds,
        verification_seconds=time.perf_counter()-start,function_bytes=(out/'functions_PRIVATE.npz').stat().st_size)
    write(out/'verification.json',audit);return audit

def execute(a):
    block_forest_fits();resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3));start=time.perf_counter()
    root=Path(os.environ['DATA_DIR']);out=Path(os.environ['OUTPUT_DIR']);out.mkdir(parents=True,exist_ok=True)
    previous=None;records=[];checks=[];models={};new_calibrators=0
    draws=arrays(root/f'resamples/{a.repeat}.npz') if a.repeat>=0 else None
    for year in periods(a.months):
        d=arrays(root/f'calibration/{a.months}/{year}/calibration_PRIVATE.npz');index=np.arange(len(d['p']))
        if draws is not None:
            index=np.concatenate([np.flatnonzero(d['cohort']==v)[draws[str(v)]] for v in cohorts(year,a.months)])
        p,y=d['p'][index],d['y'][index];groups=d['groups'][index]
        events=np.array([len(np.unique(groups[y[:,k]>0])) for k in [0,1]])
        for arm in ['independent','temporal']:
            t=time.perf_counter();cal=fit(p,y,d['vertices'],events,previous if arm=='temporal' else None);seconds=time.perf_counter()-t;new_calibrators+=1
            cal.update(year=year,months=a.months,arm=arm,repeat=a.repeat,cohorts=cohorts(year,a.months),shape_grid=d['shape_grid'].tolist(),shape_values=d['shape_values'].tolist(),
                companies=len(index),unique_companies=len(np.unique(groups)),fit_seconds=seconds,source_calibration_sha256=sha(root/f'calibration/{a.months}/{year}/calibration_PRIVATE.npz'))
            if arm=='temporal':previous=cal
            write(out/f'{year}_{arm}_calibrator_PRIVATE.json',cal)
            records.append(cal)
            if a.repeat<0 and year in YEARS:
                if year not in models:
                    models[year]=merge([CompetingRiskForest.load(root/f'reference/{year}/{s}/forest_PRIVATE.json') for s in SEEDS])
                tr=arrays(root/f'folds/{year}/train_PRIVATE.npz');ev=arrays(root/f'folds/{year}/eval_PRIVATE.npz')
                result=verify_export(models[year],tr,ev,out/f'{year}_{arm}',cal);checks.append(dict(year=year,arm=arm,months=a.months,**result))
        if a.repeat<0 and year in YEARS:
            checks.append(dict(year=year,arm='baseline',months=a.months,**verify_export(models[year],tr,ev,out/f'{year}_baseline',None)))
    write(out/'calibrators_PRIVATE.json',records);write(out/'numerical_checks.json',checks)
    summary=dict(status='COMPLETE',months=a.months,repeat=a.repeat,new_forests=0,new_calibrators=new_calibrators,
        max_gap=max(abs(c['gap']) for c in records),max_primal_violation=max(c['primal_violation'] for c in records),
        optimizer_unsuccessful_but_certified=sum(not c['optimizer_success'] for c in records),seconds=time.perf_counter()-start,
        peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,host=socket.gethostname(),slurm_job=os.environ.get('SLURM_JOB_ID'),
        no_test_outcomes_read=True,chronological_chain=True,full_representation_checks=len(checks))
    write(out/'summary.json',summary);write(out/'hashes.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()});print(json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--months',type=int,choices=[12,24],required=True);p.add_argument('--repeat',type=int,default=-1);a=p.parse_args()
    try:execute(a)
    except Exception:
        out=Path(os.environ['OUTPUT_DIR']);out.mkdir(parents=True,exist_ok=True);write(out/'FAILED.json',dict(args=vars(a),traceback=traceback.format_exc()));raise
