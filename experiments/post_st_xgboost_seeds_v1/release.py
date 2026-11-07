import sys,subprocess,platform,importlib.metadata
import numpy as np
import pandas as pd
import xgboost as xgb
from experiments.post_st_c061_xgboost_v1.pipeline import predict
from .common import *

def main():
    records=[(p.parent,read(p)) for p in run().glob('published/*/task-*/output/result.json')]
    assert len(records)==6 and sum(r['final_fits'] for _,r in records)==18 and sum(r['tuning_fits'] for _,r in records)==0
    for p,r in records:
        for role,name in [('train','training_predictions_PRIVATE.npz'),('eval','predictions_PRIVATE.npz')]:
            d=arrays(run()/f'input/{r["year"]}/{role}.npz');saved=arrays(p/name)
            for seed in SEEDS:
                f=xgb.Booster();f.load_model(p/f'model_{seed}.ubj');np.testing.assert_array_equal(predict(f,d['X']),saved[str(seed)])
    old_pointer=subprocess.check_output(['git','show','5e86ec3a3ee3e978031e5fcd75740dc5bf08f21d:CURRENT_POST_ST_MODEL.json'],cwd=ROOT)
    assert (ROOT/'CURRENT_POST_ST_MODEL.json').read_bytes()==old_pointer
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
    assert len(bad)==9 and bad[['auc','ap','ipcw_brier','observed','top20_captured']].isna().all().all()
    write(HERE/'RELEASE_CHECKS.json',dict(formal_fits=18,tuning_fits=0,real_forest_fits=0,calibrator_fits=0,successful_tasks=6,failed_tasks=0,reload_error=0.,current_pointer_unchanged=True,unsupported_performance_NA=True,public_csv_identifier_scan_files=checked,public_company_rows=False))
    if '--receipt' in sys.argv:
        rev=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        write(HERE/'execution_receipt.json',dict(base_commit=read(HERE/'FROZEN_PROTOCOL.json')['base_commit'],freeze_and_execution_commit=read(run()/'snapshot.json')['commit'],results_commit=rev,branch='research/xai-rsf-xgboost-seeds-v1',private_run=str(run()),new_seeds=SEEDS,old_seeds=OLD_SEEDS,fit_accounting={'new_final_XGBoost':18,'new_tuning':0,'new_real_XAI_forests':0,'new_calibrators':0,'synthetic_tests':'separately executed, not part of formal data fits'},slurm={'job_id':2702,'requested_nodes':'both','actual_tasks_by_node':{'oliverdesktop':6},'elapsed_including_queue_seconds':3.9885384869994596,'successful_tasks':6,'failed_tasks':0},tests=(HERE/'pytest_final.log').read_text().splitlines()[-1],environment={'python':platform.python_version(),'packages':{p:importlib.metadata.version(p) for p in ['xgboost-cpu','numpy','pandas','scipy','scikit-learn','pytest']}},input_hashes=read(HERE/'INPUT_HASHES.json'),evaluation_hashes=read(HERE/'EVALUATION_HASHES.json'),limitations=['Fixed-parameter random-seed sensitivity, no complete tuning-pipeline resampling.','Exposed cohorts and previously selected C061; exploratory comparison.','2025H24 predictions only, metricsNA.','Independent horizon classifiers can violate time monotonicity; no clipping or correction.','Short tasks all ran on oliver despite requesting the two-node pool.'],output_hashes={str(p.relative_to(HERE)):sha(p) for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='execution_receipt.json'}))

if __name__=='__main__':main()
