"""One independent annual full-function validation; never fit a model."""
import argparse, json, os, resource, socket, time, traceback
from pathlib import Path
import numpy as np
from crforest import CompetingRiskForest, CIFExpansion
from experiments.post_st_level_coverage_v1.affine import transformed_expansion, transform
from experiments.post_st_reporting_drift_v1.analyze import Reader, write_json, digest, block_fit


def main(year):
    CompetingRiskForest.fit = block_fit
    root, out = Path(os.environ['DATA_DIR']), Path(os.environ['OUTPUT_DIR'])
    reader = Reader(); start = time.perf_counter()
    f = CompetingRiskForest.load(reader.path(root/f'{year}_merged_PRIVATE.json'))
    tr = reader.arrays(root/f'folds/{year}/train_PRIVATE.npz')
    ev = reader.arrays(root/f'folds/{year}/eval_PRIVATE.npz')
    X = np.vstack([tr['X'], ev['X']])
    specs = reader.json(root/'specifications.json')[str(year)]
    base = CIFExpansion.from_forest(f, tr['X'])
    compile_seconds = time.perf_counter()-start
    times = np.unique(np.r_[0., f.grid_, 365., 366., 730., 731.])
    # Same full-row/native-time explicit component verification as verify_export;
    # share only the unchanged raw forecast and base compilation across maps.
    raw = f.predict(X, times)
    f.save(out/'reference_forest_PRIVATE.json')
    again_f = CompetingRiskForest.load(out/'reference_forest_PRIVATE.json')
    forest_reload = float(abs(again_f.predict(X, times)-raw).max())
    np.savez_compressed(out/'reference_NATIVE_PRIVATE.npz', times=times, cif=raw, sample_ids=np.r_[tr['sample_ids'],ev['sample_ids']])
    rows = []
    for arm, cal in specs.items():
        begin = time.perf_counter(); dest = out/arm; dest.mkdir()
        ex = transformed_expansion(base, cal)
        ex.save(dest/'functions_PRIVATE.npz'); loaded = CIFExpansion.load(dest/'functions_PRIVATE.npz')
        keys = [(j,) for j in ex.mains]+ex.pairs
        pred = transform(raw, times, cal)
        errors = dict(full_reconstruction=0., explicit_component_sum=0., function_reload=0., forest_reload=forest_reload)
        for i in range(0,len(X),32):
            xx=X[i:i+32]; p=pred[i:i+32]
            errors['full_reconstruction']=max(errors['full_reconstruction'],float(abs(ex.predict(xx,times)-p).max()))
            errors['function_reload']=max(errors['function_reload'],float(abs(loaded.predict(xx,times)-p).max()))
            total=np.broadcast_to(ex._coeff(times)[1],p.shape).copy()
            for key in keys: total += ex.component(xx,key,times)
            errors['explicit_component_sum']=max(errors['explicit_component_sum'],float(abs(total-p).max()))
        minimum=float(pred.min()); increment=float(np.diff(pred,axis=1).min()); sums=pred.sum(-1)
        violations=(sums>1+1e-10)|(pred< -1e-10).any(-1)
        assert max(errors.values())<=1e-10
        assert minimum>=-1e-10 and increment>=-1e-10
        if not cal['diagnostic']: assert not violations.any()
        np.savez_compressed(dest/'predictions_PRIVATE.npz',times=times,cif=pred[len(tr['X']):],sample_ids=ev['sample_ids'])
        write_json(dest/'linear_map.json',cal)
        rows.append(dict(year=year,arm=arm,**errors,verified_rows=len(X),training_rows=len(tr['X']),evaluation_rows=len(ev['X']),native_times=len(f.grid_),verified_times=len(times),minimum=minimum,minimum_increment=increment,maximum_sum=float(sums.max()),violating_train_companies=int(violations[:len(tr['X'])].any(1).sum()),violating_eval_companies=int(violations[len(tr['X']):].any(1).sum()),violating_sample_times=int(violations.sum()),legal_on_checked_samples=not bool(violations.any()),diagnostic=cal['diagnostic'],mains=len(ex.mains),pairs=len(ex.pairs),function_bytes=(dest/'functions_PRIVATE.npz').stat().st_size,seconds=time.perf_counter()-begin))
    reader.finish();write_json(out/'input_hashes_PRIVATE.json',reader.hashes)
    write_json(out/'summary.json',dict(status='COMPLETE',year=year,host=socket.gethostname(),forest_fits=0,calibrator_fits=0,compile_seconds=compile_seconds,total_seconds=time.perf_counter()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,checks=rows))
    write_json(out/'hashes.json',{str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file()})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--year',type=int,required=True);a=parser.parse_args()
    try: main(a.year)
    except Exception:
        (Path(os.environ['OUTPUT_DIR'])/'FAILURE.txt').write_text(traceback.format_exc());raise
