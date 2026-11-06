"""Apply a fixed seven-arm experiment to an immutable C039 forest."""
import os
from pathlib import Path
import socket
import sys
import time
import traceback
import numpy as np
import pandas as pd
from crforest import CompetingRiskForest
from crforest.functions import CIFExpansion
from experiments.post_st_benchmark_calibration_v1.common import arrays, write, sha, DAYS
from experiments.post_st_level_coverage_v1.affine import forest_envelope, fit_affine
from experiments.post_st_calendar_level_v1.rates import spells, contributions, requested_slope, certified_slopes
from .module import age_model, mapping, transformed, certify

ARMS = ['reference','temporal_affine','calendar_2y','calendar_damped','age_scale','age_mix25','age_mix50']


def verify(f, base, tr, ev, cal, dest):
    dest.mkdir()
    ex = transformed(base, cal)
    ex.save(dest/'functions_PRIVATE.npz')
    reload = CIFExpansion.load(dest/'functions_PRIVATE.npz')
    times = np.unique(np.r_[0, f.grid_, ex.grid, DAYS])
    keys = [(j,) for j in ex.mains] + ex.pairs
    errors = dict(reconstruction=0., component_sum=0., reload=0.)
    lo, hi, inc = 0., 0., 0.
    x = np.vstack([tr['X'],ev['X']])
    for start in range(0,len(x),32):
        xx = x[start:start+32]
        expected = mapping(f.predict(xx,times),times,cal)
        predicted = ex.predict(xx,times)
        total = np.broadcast_to(ex._coeff(times)[1],expected.shape).copy()
        for key in keys:
            total += ex.component(xx,key,times)
        for name, value in [('reconstruction',predicted),('component_sum',total),('reload',reload.predict(xx,times))]:
            errors[name] = max(errors[name],float(abs(value-expected).max()))
        lo=min(lo,float(expected.min())); hi=max(hi,float(expected.sum(-1).max()))
        inc=min(inc,float(np.diff(expected,axis=1).min()))
    assert max(errors.values())<=1e-10 and lo>=-1e-12 and hi<=1+1e-10 and inc>=-1e-12
    p=mapping(f.predict(ev['X'],DAYS),DAYS,cal)
    np.savez_compressed(dest/'predictions_PRIVATE.npz',sample_ids=ev['sample_ids'],times=DAYS,cif=p)
    write(dest/'mapping_PRIVATE.json',cal)
    return dict(**errors,minimum=lo,maximum_sum=hi,minimum_increment=inc,rows=len(x),times=len(times),
                main_functions=len(ex.mains),pair_functions=len(ex.pairs),
                native_component_scale=cal['b'],bytes=(dest/'functions_PRIVATE.npz').stat().st_size)


def main(year):
    root=Path(os.environ['DATA_DIR'])/str(year)
    out=Path(os.environ['OUTPUT_DIR']);out.mkdir(exist_ok=True,parents=True)
    tr=arrays(root/'train_PRIVATE.npz');ev=arrays(root/'eval_PRIVATE.npz')
    assert set(ev)=={'X','sample_ids'}
    f=CompetingRiskForest.load(root/'forest_PRIVATE.json')
    base=CIFExpansion.load(root/'functions_PRIVATE.npz')
    original=arrays(root/'reference_predictions_PRIVATE.npz')
    np.testing.assert_array_equal(original['sample_ids'],ev['sample_ids'])
    err=float(abs(f.predict(ev['X'],DAYS)-original['cif']).max());assert err<=1e-12
    vertices=forest_envelope(f)
    meta=pd.read_parquet(root/'metadata_PRIVATE.parquet')
    cutoff=pd.Timestamp(year,1,1)
    age=age_model(meta,cutoff)
    write(out/'age_diagnostics.json',age['diagnostics'])
    # Own-cutoff calendar exposures, including censored companies.
    m=spells(meta,cutoff)
    hd,he=contributions(m,m.origin.min(),cutoff)
    wd,we=contributions(m,cutoff-pd.DateOffset(years=2),cutoff)
    request=[requested_slope(hd,he[:,k],wd,we[:,k]) for k in range(2)]
    write(out/'calendar_diagnostics.json',request)
    calset={'reference':dict(b=[1.,1.],shape_grid=[0.],offset_values=[[0.,0.]])}
    failures=[]
    diagnostics={}
    for name,slopes in [('calendar_2y',[r['requested'] for r in request]),
                        ('calendar_damped',np.sqrt([r['requested'] for r in request])),
                        ('age_scale',age['requested_slopes'])]:
        try:
            b,bound=certified_slopes(vertices,slopes)
            calset[name]=dict(b=b.tolist(),shape_grid=[0.],offset_values=[[0.,0.]])
            diagnostics[name]=dict(requested=list(map(float,slopes)),certified=b.tolist(),bound=bound)
        except Exception as exc:
            failures.append(dict(arm=name,stage='certificate',reason=str(exc)))
    for name,weight in [('age_mix25',.25),('age_mix50',.5)]:
        calset[name]=dict(b=[1-weight]*2,shape_grid=age['grid'],
                          offset_values=(weight*np.asarray(age['recent'])).tolist())
    try:
        hist=arrays(root/'calibration_PRIVATE.npz')
        fitted=fit_affine(hist['p'],hist['y'],vertices,hist['y'].sum(0),prior_companies=20.)
        calset['temporal_affine']=dict(b=fitted['b'],shape_grid=[0.,365.],offset_values=[[0.,0.],fitted['c']])
        diagnostics['temporal_affine']=fitted|dict(calibration_companies=len(hist['y']))
    except Exception as exc:
        failures.append(dict(arm='temporal_affine',stage='fit',reason=str(exc)))
    results=[]
    for arm in ARMS:
        if arm not in calset:
            continue
        start=time.perf_counter()
        try:
            cal=calset[arm]
            certificate=certify(vertices,cal)
            checks=verify(f,base,tr,ev,cal,out/arm)
            results.append(dict(arm=arm,status='ok',seconds=time.perf_counter()-start,
                                certificate=certificate,validation=checks,diagnostic=diagnostics.get(arm,{})))
        except Exception as exc:
            (out/f'{arm}_failed.txt').write_text(traceback.format_exc())
            failures.append(dict(arm=arm,stage='verification',reason=str(exc)))
    write(out/'summary.json',dict(year=year,results=results,failures=failures,
        original_reproduction_error=err,new_forest_fits=0,host=socket.gethostname(),
        job=os.environ.get('SLURM_JOB_ID'),source_forest_sha256=sha(root/'forest_PRIVATE.json')))
    write(out/'hashes.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})


if __name__=='__main__':
    main(int(sys.argv[1]))
