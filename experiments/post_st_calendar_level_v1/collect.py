"""Validate publication receipts, source hashes, and saved-map alignment."""
from pathlib import Path
import json, os
import numpy as np
import pandas as pd
from experiments.post_st_reporting_drift_v1.analyze import digest, write_json
from experiments.post_st_prognosis_v1.common import horizon_predictions

PUBLIC=Path(__file__).resolve().parent


def collect(run):
    records=[json.loads(l) for l in (run/'launcher_validation.log').read_text().splitlines() if l.startswith('{')]
    launch=[r for r in records if 'result' in r][-1];assert launch['success']
    report=json.loads(Path(launch['result']).read_text());assert report['expected']==3 and not report['pending']
    assert len(report['completed'])==3
    loc=json.loads((PUBLIC/'run_location.json').read_text());source=Path(loc['source_run'])
    mapped=np.load(run/'evaluation/mapped_horizon_risks_PRIVATE.npz',allow_pickle=False)
    rows=[];registry=[];alignment=[]
    for task in report['completed']:
        assert task['returncode']==0 and task['compute_completed'] and task['publication_completed']
        dest=run/f'validation_results/{launch["job"]}/task-{task["id"]}/output'
        hashes=json.loads((dest/'hashes.json').read_text())
        for name,h in hashes.items():assert digest(dest/name)==h
        summary=json.loads((dest/'summary.json').read_text());assert summary['status']=='COMPLETE' and summary['forest_fits']==0
        year=summary['year'];rows.extend(summary['checks'])
        lab=np.load(source/f'evaluation_input/{year}_test_PRIVATE.npz',allow_pickle=False)
        for check in summary['checks']:
            arm=check['arm'];pred=np.load(dest/arm/'predictions_PRIVATE.npz',allow_pickle=False)
            np.testing.assert_array_equal(pred['sample_ids'],lab['sample_ids'])
            for h in [12,24]:
                actual=horizon_predictions(pred['cif'],pred['times'],lab[f'h{h}'])
                expected=mapped[f'{year}_{h}_{arm}'];error=float(abs(actual-expected).max())
                assert error<=1e-12
                alignment.append(dict(year=year,arm=arm,months=h,maximum_difference=error))
        registry.append(dict(year=year,job=launch['job'],task=task['id'],host=summary['host'],path=str(dest),seconds=summary['total_seconds'],compile_seconds=summary['compile_seconds'],peak_rss_kib=summary['peak_rss_kib']))
    assert sorted(r['year'] for r in registry)==[2023,2024,2025]
    # Verify source files as well as staged immutable NFS inputs after all work.
    input_hashes={}
    immutable=json.loads((source/'audit/immutable_input_hashes.json').read_text())
    originals=json.loads((source/'audit/source_hashes.json').read_text())
    pointer=json.loads((source/'audit/current_pointer.json').read_text())
    assert digest(PUBLIC.parents[1]/'CURRENT_POST_ST_MODEL.json')==pointer['sha256']
    for year in [2023,2024,2025]:
        for name in ['train_PRIVATE.npz','eval_PRIVATE.npz']:
            assert digest(source/f'input/folds/{year}/{name}')==immutable[f'folds/{year}/{name}']
        assert digest(source/f'audit/{year}_training_metadata_PRIVATE.parquet') in originals.values()
    for path in [run/'prepared/input_hashes_PRIVATE.json',run/'evaluation/input_hashes_PRIVATE.json']:
        for name,h in json.loads(path.read_text()).items():
            assert digest(name)==h;input_hashes[name]=h
    for name,h in json.loads((run/'validation_input_hashes.json').read_text()).items():assert digest(run/'validation_input'/name)==h
    out=run/'collected';out.mkdir()
    totals=json.loads((run/'prepared/PERSON_TIME_CHECKS_PRIVATE.json').read_text())
    slopes=pd.read_csv(run/'prepared/LEVEL_SLOPES_PRIVATE.csv')
    slope_checks=[]
    for row in slopes.itertuples():
        hist=next(r for r in totals if r['year']==row.year and r['window']=='hist')
        window=next(r for r in totals if r['year']==row.year and r['window']==row.window)
        independent=(window['events'][row.cause-1]/window['person_days'])/(hist['events'][row.cause-1]/hist['person_days'])
        np.testing.assert_allclose(independent,row.raw_ratio,atol=1e-14,rtol=0)
        slope_checks.append(dict(year=row.year,window=row.window,cause=row.cause,maximum_difference=abs(independent-row.raw_ratio)))
    pd.DataFrame(slope_checks).to_csv(out/'INDEPENDENT_SLOPE_CHECKS.csv',index=False)
    pd.DataFrame(rows).to_csv(out/'FUNCTION_VALIDATION_PRIVATE.csv',index=False)
    pd.DataFrame(alignment).to_csv(out/'PREDICTION_ALIGNMENT.csv',index=False)
    write_json(out/'registry_PRIVATE.json',registry)
    write_json(out/'verified_input_hashes_PRIVATE.json',input_hashes)
    write_json(out/'completion.json',dict(forest_fits=0,calibrator_fits=0,year_tasks=3,function_representations=18,job=launch['job'],nodes_allocated='both',actual_compute_hosts=sorted(set(r['host'] for r in registry)),pool_execution_seconds=launch['execution_seconds'],pool_elapsed_seconds=launch['elapsed_including_queue'],all_inputs_unchanged=True,all_results_published_and_hash_verified=True))


if __name__=='__main__':
    loc=json.loads((PUBLIC/'run_location.json').read_text());collect(Path(os.environ.get('CALENDAR_LEVEL_RUN',loc['private_run'])))
