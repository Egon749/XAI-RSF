"""Stop-first provenance and denominator audit. No fitting or fallbacks."""
from pathlib import Path
from dataclasses import asdict
import json, os, subprocess, traceback
import numpy as np
import pandas as pd
from crforest import CompetingRiskForest, CIFExpansion
from experiments.post_st_reporting_drift_v1.analyze import Reader, write_json, digest, block_fit
from experiments.post_st_ablation_v1.verification import merge
from experiments.post_st_prognosis_v1.common import horizon_predictions
from experiments.post_st_feature_permutation_v1.common import evaluation_context

PUBLIC=Path(__file__).resolve().parent


def main():
    CompetingRiskForest.fit=block_fit
    loc=json.loads((PUBLIC/'run_location.json').read_text())
    run=Path(os.environ.get('RISK_INDEX_RUN',loc['private_run']));run.mkdir(exist_ok=True)
    out=run/'audit';out.mkdir();reader=Reader();rows=[]
    frozen=subprocess.check_output(['git','show','d8880611b40f22f641e25219839c93485d3aa2e7:experiments/post_st_risk_index_v1/frozen_protocol.json'])
    assert frozen==(PUBLIC/'frozen_protocol.json').read_bytes()
    source=Path(loc['source_run']);hist=Path(loc['historical_run'])
    reader.path(PUBLIC.parents[1]/'CURRENT_POST_ST_MODEL.json')
    registry=reader.json(source/'audit/registry_formal.json')
    original=reader.json(source/'audit/source_hashes.json')
    gates=[];configs={};reference_paths={}
    # Gate is based on original saved, uncalibrated reference forecasts.
    for year in [2023,2024,2025]:
        lab=reader.arrays(source/f'evaluation_input/{year}_test_PRIVATE.npz')
        assert digest(source/f'evaluation_input/{year}_test_PRIVATE.npz') in original.values()
        rs=sorted([r for r in registry if r['year']==year and r['arm']=='baseline'],key=lambda r:r['seed'])
        assert [r['seed'] for r in rs]==[42,101,202]
        ps=[]
        for r in rs:
            path=Path(r['path']);hashes=reader.json(path/'hashes.json')
            for name in ['shrunk/forest_PRIVATE.json','shrunk/predictions_PRIVATE.npz']:assert digest(reader.path(path/name))==hashes[name]
            z=reader.arrays(path/'shrunk/predictions_PRIVATE.npz');np.testing.assert_array_equal(z['sample_ids'],lab['sample_ids'])
            ps.append(z['cif'])
            f=CompetingRiskForest.load(path/'shrunk/forest_PRIVATE.json');configs[year,r['seed']]=asdict(f.config)
            reference_paths[year,r['seed']]=str(path/'shrunk/forest_PRIVATE.json')
        for h in [12,24]:
            if year==2025 and h==24:continue
            y,w,known,supported,_=evaluation_context(lab,h,1);assert supported and known.all() and (w==1).all()
            p=horizon_predictions(np.mean(ps,axis=0),z['times'],lab[f'h{h}'])
            actual=float(y.mean()/p[:,0].mean());target=({12:[5.8229,10.9851,1.6114],24:[5.1535,3.8376]})[h][year-2023]
            record=dict(year=year,months=h,actual=actual,expected=target,passed=round(actual,4)==target);gates.append(record)
            write_json(out/f'gate_{year}_{h}.json',record)
            assert record['passed'],'STOP: reference O/E mismatch'
    write_json(out/'REFERENCE_GATE.json',gates)
    historical_hashes=reader.json(hist/'audit/source_hashes.json')
    # Chronological validation: a required zero denominator halts immediately.
    for year in range(2019,2025):
        tr=reader.arrays(hist/f'input/folds/{year}/train_PRIVATE.npz')
        ev=reader.arrays(hist/f'input/folds/{year}/eval_PRIVATE.npz')
        assert tr['X'].shape[1]==83 and (tr['year']<year).all()
        assert digest(hist/f'input/folds/{year}/train_PRIVATE.npz') in historical_hashes.values()
        assert digest(hist/f'input/folds/{year}/eval_PRIVATE.npz') in historical_hashes.values()
        assert not set(tr['sample_ids']) & set(ev['sample_ids'])
        members=[];predictions=[]
        for seed in [42,101,202]:
            path=hist/f'input/reference/{year}/{seed}'
            for name in ['forest_PRIVATE.json','predictions_PRIVATE.npz']:assert digest(reader.path(path/name)) in historical_hashes.values()
            f=CompetingRiskForest.load(path/'forest_PRIVATE.json')
            assert asdict(f.config)==configs[2023,seed],f'STOP: historical config differs: {year}/{seed}'
            assert len(f.trees_)==32 and f.n_features_==83 and f.audit()['max_path_variables']<=2
            for tree in f.trees_:
                for n in tree:
                    if 'cif' in n:
                        assert n['leaf_shrinkage']['strength']==10
            pred=reader.arrays(path/'predictions_PRIVATE.npz');np.testing.assert_array_equal(pred['sample_ids'],ev['sample_ids'])
            np.testing.assert_array_equal(f.predict(ev['X'],pred['times']),pred['cif'])
            members.append(f);predictions.append(pred['cif'])
        forest=merge(members);ex=CIFExpansion.from_forest(forest,tr['X'])
        ex.save(out/f'{year}_functions_PRIVATE.npz')
        mu=ex._coeff([365.,730.])[1]
        np.savez_compressed(out/f'{year}_denominators_PRIVATE.npz',mu=mu)
        for hidx,h in enumerate([12,24]):
            for k in [1,2]:
                rows.append(dict(year=year,months=h,cause=k,training_companies=len(tr['X']),oof_companies=len(ev['X']),configuration_matches=True,saved_predictions_reproduced=True,denominator_positive=bool(mu[hidx,k-1]>0),denominator_finite=bool(np.isfinite(mu[hidx,k-1])),denominator_file_sha256=digest(out/f'{year}_denominators_PRIVATE.npz')))
        pd.DataFrame(rows).to_csv(out/'HISTORICAL_DENOMINATOR_AUDIT.csv',index=False)
        if not (np.isfinite(mu).all() and (mu>0).all()):
            reader.finish();write_json(out/'input_hashes_PRIVATE.json',reader.hashes)
            write_json(out/'STOP.json',dict(status='STOP_REQUIRED_HISTORY_INDEX_UNDEFINED',historical_year=year,invalid_cause_horizons=[dict(cause=r['cause'],months=r['months']) for r in rows if not(r['denominator_positive'] and r['denominator_finite'])],reason='Required own-model denominator is nonpositive/nonfinite; no cohort removal,epsilon,in-sample substitution or threshold estimation authorized',forest_fits=0,calibrator_fits=0,thresholds_estimated=0))
            print('STOP: required historical risk-index denominator invalid; see aggregate audit.');return
    reader.finish();write_json(out/'input_hashes_PRIVATE.json',reader.hashes)
    write_json(out/'PASS.json',dict(status='HISTORICAL_AUDIT_PASS'))


if __name__=='__main__':
    try:main()
    except Exception:
        loc=json.loads((PUBLIC/'run_location.json').read_text());run=Path(os.environ.get('RISK_INDEX_RUN',loc['private_run']))
        (run/'AUDIT_FAILURE_PRIVATE.txt').write_text(traceback.format_exc());raise
