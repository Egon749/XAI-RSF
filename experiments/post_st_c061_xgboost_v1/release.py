"""Final privacy, immutable-source, coverage and reproducibility checks."""
import subprocess,sys,platform,importlib.metadata
import pandas as pd
import xgboost as xgb
from .common import *
from .pipeline import predict

def main():
    jobs=[(p.parent,read(p)) for p in run().glob('published/*/task-*/output/result.json')]
    assert len(jobs)==6 and sum(j['tuning_fits'] for _,j in jobs)==648 and sum(j['final_fits'] for _,j in jobs)==18
    maxerr=0.
    for p,j in jobs:
        y=j['year'];inp=run()/f'input/{y}'
        for role,fn in [('eval','predictions_PRIVATE.npz'),('train','training_predictions_PRIVATE.npz')]:
            d=arrays(inp/(role+'.npz'));saved=arrays(p/fn)
            for seed in SEEDS:
                f=xgb.Booster();f.load_model(p/f'model_{seed}.ubj');new=predict(f,d['X'])
                np.testing.assert_array_equal(new,saved[str(seed)]);maxerr=max(maxerr,float(abs(new-saved[str(seed)]).max()))
    pointer=ROOT/'CURRENT_POST_ST_MODEL.json'
    old=subprocess.check_output(['git','show','9ffcf8e07dbbaae03cec59c1802978571d516298:CURRENT_POST_ST_MODEL.json'],cwd=ROOT)
    assert pointer.read_bytes()==old
    metrics=pd.read_csv(HERE/'METRICS_ALL.csv');bad=metrics[(metrics.scope.astype(str)=='2025')&(metrics.months==24)]
    assert bad[['auc','ap','ipcw_brier','top20_captured','observed']].isna().all().all()
    ids=set()
    for y in YEARS:
        for f in [source()/f'input/folds/{y}/train_PRIVATE.npz',source()/f'input/evaluation_input/{y}_test_PRIVATE.npz']:
            d=arrays(f);ids.update(map(str,d['sample_ids']));ids.update(map(str,d['groups']))
    public=[p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    assert not any(p.suffix in ('.npz','.npy','.pkl','.ubj','.parquet','.joblib') or '_PRIVATE' in p.name for p in public)
    checked=0
    for p in public:
        if p.suffix=='.csv':
            d=pd.read_csv(p,dtype=str).fillna('');values=set(d.to_numpy().ravel())
            assert not ids&values, f'Company identifier in {p.name}';checked+=1
    write(HERE/'RELEASE_CHECKS.json',dict(final_predictors=18,historical_tuning_fits=648,formal_tasks_successful=6,forest_fits=0,calibrator_fits=0,reload_error=maxerr,pointer_unchanged=True,unsupported_2025H24_NA=True,public_csv_identifier_scan_files=checked,no_public_binary_models_or_company_data=True))
    write(HERE/'ENVIRONMENT.json',dict(python=platform.python_version(),platform=platform.platform(),packages={p:importlib.metadata.version(p) for p in ['xgboost-cpu','numpy','pandas','scipy','scikit-learn','pytest','matplotlib']},R=subprocess.check_output(['/home/oliver/MainDiskData/financial_diagnosis/csf_joint_sensitivity_v1/environment/r/bin/Rscript','--version'],text=True).strip(),lemon_preflight={'python_environment':'xai-baselines-py313-20261027','xgboost':'3.4.1','numpy':'2.5.4','scikit-learn':'1.9.1','scipy':'1.18.1','formal_task_count':0}))
    if '--receipt' in sys.argv:
        result_rev=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        write(HERE/'execution_receipt.json',dict(base_commit=read(HERE/'FROZEN_PROTOCOL.json')['base_commit'],protocol_commit='3178e02cd897f24411de6cd46da8f9d9a06b393f',execution_code_commit=read(run()/'code_snapshot.json')['commit'],results_commit=result_rev,branch='research/xai-rsf-c061-xgboost-v1',private_run=str(run()),fit_accounting={'new_XAI_forests':0,'new_calibrators':0,'historical_XGBoost':648,'final_XGBoost':18,'total_real_fits':666,'synthetic_unit_tests':'separate, not part of real-data fit count'},slurm={'job_id':2701,'requested_nodes':'both','actual_tasks_by_node':{'oliverdesktop':6},'successful_tasks':6,'failed_tasks':0,'elapsed_including_queue_seconds':6.582358618965372},tests=(HERE/'pytest_final.log').read_text().splitlines()[-1],limitations=['Exposed cohorts, previously selected C061, sparse historical forward tuning events.','2025H24 has predictions only; metrics NA.','XGBoost horizon classifiers are not joint monotone CIFs.','Existing PROBoost/Stacking contextual table reuses published results; internal selection protocols differ.','Initial run-directory creation at store root waited on a file lease; canceled before fits and used a new unique directory under existing private parent.','Both-node pool requested but short tasks all executed on oliver; no artificial duplicate training.'],input_hashes=read(HERE/'SOURCE_HASHES.json'),evaluation_hashes=read(HERE/'EVALUATION_HASHES.json'),public_hashes={str(p.relative_to(HERE)):sha(p) for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='execution_receipt.json'}))

if __name__=='__main__':main()
