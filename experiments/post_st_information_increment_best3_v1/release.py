"""Validate the aggregate release and return the zero-overwrite provenance receipt."""
import platform,sys,importlib.metadata
from .common import *
def main():
    r=run();inputs=read(r/'sources_PRIVATE.json')
    assert all(sha(Path(p))==h for p,h in inputs.items())
    for y in YEARS:
        for role in ['train','eval']:
            path=r/f'input/folds/{y}/{role}_PRIVATE.npz';assert sha(path)==sha(source()/f'input/folds/{y}/{role}_PRIVATE.npz')
        assert sha(r/f'eval/{y}.npz')==sha(source()/f'evaluation_input/{y}_test_PRIVATE.npz')
    cfg=pd.read_csv(D/'MODEL_CONFIG_AUDIT.csv');assert set(cfg.seed)==set(SEEDS)
    df=pd.read_csv(D/'METRICS_ALL.csv')
    assert len(df)==4*4*4*2*2*2 and not df.duplicated(['year','arm','variant','seed','months','cause']).any()
    invalid=df.query("year=='2025' and months==24")
    assert invalid[['auc','ap','brier','captured','coverage','observed','oe']].isna().all().all()
    valid=df.query("year=='pooled' and months==12 and cause==1 and seed==0 and variant=='fixed25'")
    assert set(valid.arm)==set(SETS) and valid[['auc','ap','brier','within_year_auc']].notna().all().all()
    assert np.isclose(valid[valid.arm=='M3'].within_year_auc.item(),read(D/'SOURCE_TIME_AUDIT.json')['anchor']['preexisting_best_value'])
    assert len(pd.read_csv(D/'SEED_SENSITIVITY.csv'))==3*4*2*2*2*5*9
    assert len(pd.read_csv(D/'PAIRED_DIFFERENCES_95.csv'))==4*2*2*2*5*9
    assert len(pd.read_csv(D/'M3_ENSEMBLE_HASHES.csv'))==3
    for file in D.glob('*.csv'):
        assert not {'sample_id','company_id','stock_code','source_text','threshold','risk','origin_date'}&set(pd.read_csv(file,nrows=0).columns)
    assert not [file for file in D.rglob('*') if 'PRIVATE' in file.name or file.suffix in ('.npz','.parquet','.pkl')]
    assert len(list((D/'figures').glob('*.png')))==len(list((D/'figures').glob('*.pdf')))==6
    jobs={}
    for name in ['fit','ensemble']:
        receipt=[json.loads(l) for l in (r/name/'launcher.log').read_text().splitlines() if l.startswith('{') and '"result"' in l][-1]
        rows=read(r/name/'published.json');report=read(receipt['result'])
        assert receipt['success'] and report['expected']==len(rows)
        for row in rows:
            p=Path(row['path']);assert all(sha(p/n)==h for n,h in read(p/'hashes.json').items())
        jobs[name]=dict(slurm_job=receipt['job'],tasks=len(rows),by_node=receipt['by_node'],elapsed_s_with_queue=receipt['elapsed_including_queue'],report_sha256=sha(receipt['result']))
    assert '311 passed' in (D/'pytest_final.log').read_text() and ' skipped' not in (D/'pytest_final.log').read_text()
    files={str(p.relative_to(D)):sha(p) for p in sorted(D.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='execution_receipt.json'}
    dump(D/'execution_receipt.json',dict(status='ALL_FOUR_CELLS_COMPLETE_RETROSPECTIVE__AUDIT_REALTIME_UNCONFIRMED',
         source_commit='f01e07e8855787d8832d59d33128e6b66cc11f10',protocol_commit='65f66b3ce2acaa8327493affbcca87a71bcb7e4c',
         seeds=SEEDS,combo='C039',forest_fits=54,main_new_fits=27,sensitivity_new_fits=27,new_ensemble_exports=18,C039_member_forests_reused=9,C039_new_fits=0,
         calibration_fits=0,subgroup_fits=0,precision=read(D/'RECONSTRUCTION_AUDIT.json'),private_run=str(r),jobs=jobs,
         source_files_verified_before_after=len(inputs),private_source_digest=sha(r/'sources_PRIVATE.json'),
         tests='311 passed including independent R checks; one upstream EBM visualization warning',
         failures_and_limits=['No failed fit or Slurm publication; no retraining or seed replacement','Remote registry catalog unavailable (stale tunnel). Local frozen source and actual document/notice probes used; actual pre-origin audit release not verified for all records.','C039 selected on already exposed outcomes. Intervals conditional on forecasts and chosen members.','2025 H24 performance not supported.'],
         independent_primary_CI_error=read(D/'INDEPENDENT_CHECKS.json')['max_interval_error'],privacy='aggregate-only release; no firm predictions, IDs, raw cuts or models',environment=dict(python=sys.version,platform=platform.platform(),libraries={k:importlib.metadata.version(k) for k in ['numpy','scipy','pandas','scikit-learn','matplotlib']}),sha256=files))
if __name__=='__main__':main()
