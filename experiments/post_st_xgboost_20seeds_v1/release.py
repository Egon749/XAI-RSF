import sys,subprocess,platform,importlib.metadata,json
import numpy as np
import pandas as pd
import xgboost as xgb
from experiments.post_st_c061_xgboost_v1.pipeline import predict
from .common import *

def main():
    records=[(p.parent,read(p)) for p in run().glob('published/*/task-*/output/result.json')]
    expected={(y,h,s) for y in YEARS for h in (12,24) for s in SEEDS}
    found=set();nodes={};model_hashes={}
    assert len(records)==24 and sum(r['final_fits'] for _,r in records)==120 and sum(r['tuning_fits'] for _,r in records)==0
    for p,r in records:
        nodes[r['hostname']]=nodes.get(r['hostname'],0)+1
        for row in read(p.parent/'complete.json')['manifest']:
            if len(row)==2:assert (p/row[0]).is_dir();continue
            fn,size,hh=row;assert (p/fn).stat().st_size==size and sha(p/fn)==hh
        seeds=seed_batches()[r['batch']]
        for s in seeds:
            key=r['year'],r['months'],s
            assert key not in found;found.add(key)
            model_hashes['/'.join(map(str,key))]=sha(p/f'model_{s}.ubj')
        for role,name in [('train','training_predictions_PRIVATE.npz'),('eval','predictions_PRIVATE.npz')]:
            d=arrays(run()/f'input/{r["year"]}/{role}.npz');saved=arrays(p/name)
            np.testing.assert_array_equal(saved['sample_ids'],d['sample_ids'])
            for seed in seeds:
                f=xgb.Booster();f.load_model(p/f'model_{seed}.ubj')
                np.testing.assert_array_equal(predict(f,d['X']),saved[str(seed)])
    assert found==expected and list(SEEDS)==draw_seeds()
    base=read(HERE/'FROZEN_PROTOCOL.json')['base_commit']
    old_pointer=subprocess.check_output(['git','show',base+':CURRENT_POST_ST_MODEL.json'],cwd=ROOT)
    assert (ROOT/'CURRENT_POST_ST_MODEL.json').read_bytes()==old_pointer
    changed=subprocess.check_output(['git','diff','--name-only',base],cwd=ROOT,text=True).splitlines()
    assert all(p=='AGENTS.md' or p.startswith('experiments/'+HERE.name+'/') for p in changed)
    ids=set()
    for y in YEARS:
        for p in [source()/f'input/folds/{y}/train_PRIVATE.npz',source()/f'input/evaluation_input/{y}_test_PRIVATE.npz']:
            d=arrays(p);ids.update(map(str,d['sample_ids']));ids.update(map(str,d['groups']))
    checked=0
    for p in HERE.rglob('*'):
        if not p.is_file() or '__pycache__' in p.parts:continue
        assert p.suffix not in ('.npz','.npy','.pkl','.ubj','.parquet','.joblib') and '_PRIVATE' not in p.name
        if p.suffix=='.csv':
            cells=set(pd.read_csv(p,dtype=str).fillna('').to_numpy().ravel());assert not cells&ids,p.name;checked+=1
    m=pd.read_csv(HERE/'METRICS_ALL.csv');bad=m[(m.scope=='2025')&(m.months==24)]
    assert len(bad)==30 and bad[['auc','ap','ipcw_brier','observed','top20_captured']].isna().all().all()
    for (scope,h),g in m[m.model.isin([f'XGBoost/{s}' for s in SEEDS])].groupby(['scope','months']):assert len(g)==20
    logs=[json.loads(line) for line in (run()/'scheduler.log').read_text().splitlines() if line.startswith('{')]
    scheduler=logs[-1];assert scheduler['success'] and scheduler['by_node']==nodes
    write(HERE/'RELEASE_CHECKS.json',dict(formal_fits=120,tuning_fits=0,real_forest_fits=0,calibrator_fits=0,successful_tasks=24,failed_tasks=0,all_seed_tuples_exactly_once=True,reload_error=0.,current_pointer_unchanged=True,prior_files_unchanged=True,unsupported_performance_NA=True,public_csv_identifier_scan_files=checked,public_company_rows=False))
    if '--receipt' in sys.argv:
        rev=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        write(HERE/'execution_receipt.json',dict(base_commit=base,freeze_and_execution_commit=read(run()/'snapshot.json')['commit'],results_commit=rev,branch='research/xai-rsf-xgboost-20seeds-v1',private_run=str(run()),new_seeds=SEEDS,fit_accounting={'new_final_XGBoost':120,'new_tuning':0,'new_real_XAI_forests':0,'new_calibrators':0,'synthetic_tests':'separate from formal data fits'},slurm=scheduler,successful_tasks=24,failed_tasks=0,tests=(HERE/'pytest_final.log').read_text().splitlines()[-1],preflight_issues=['Initial test collection missing package __init__; fixed before freeze and preserved log.','Initial lemon probe omitted project dependency PYTHONPATH; verified with actual launcher environment before fitting.'],environment={'python':platform.python_version(),'packages':{p:importlib.metadata.version(p) for p in ['xgboost-cpu','numpy','pandas','scipy','scikit-learn','pytest']}},input_hashes=read(HERE/'INPUT_HASHES.json'),evaluation_hashes=read(HERE/'EVALUATION_HASHES.json'),model_hashes=model_hashes,limitations=['Fixed-data/parameter seed variation,not full-pipeline resampling.','Previously exposed cohorts and previously selected C061.','20-member mean changes both seeds and ensemble size versus3-member references.','2025H24 metricsNA.','Separate horizons need not form monotone CIFs;no clipping.'],output_hashes={str(p.relative_to(HERE)):sha(p) for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='execution_receipt.json'}))

if __name__=='__main__':main()
