"""Build a public receipt from completed jobs, immutable inputs and aggregate files."""
import platform,re,subprocess,sys
import importlib.metadata
import pandas as pd
from .common import *

def deliver():
    r=run_path()
    expected={'COMBINATIONS_120':120,'COMPLEXITY_TRADEOFF':792,'RECONSTRUCTION_DETAIL':720,'ALL_METRICS':4224}
    # 132 models x 2 versions x 2 causes x 2 horizons x (three years + pooled).
    for table,n in expected.items():assert len(pd.read_csv(HERE/(table+'.csv')))==n
    models=pd.read_csv(HERE/'ALL_METRICS.csv',dtype={'scope':str})
    u=models.query("scope=='2025' and months==24")
    assert len(u)==528 and u[['auc','ap','brier','captured','coverage','observed']].isna().all().all()
    forbidden={'sample_id','sample_ids','company','company_id','stock_code','origin_date','threshold','cutpoint','row','risk','prediction','private_path'}
    checked=[]
    for p in HERE.glob('*.csv'):
        cols=set(pd.read_csv(p,nrows=0).columns)
        assert not cols&forbidden,(p,cols&forbidden);checked.append(p.name)
    assert not [p for p in HERE.rglob('*') if 'PRIVATE' in p.name or p.suffix in ('.npz','.pkl','.parquet')]
    history=read(r/'hash_before.json')
    audit=read(HERE/'MODEL_HASH_AUDIT.json')
    assert audit['before']==history and audit['after_equals_before']
    for job in ['verify','explain']:
        assert (r/job/'published.json').is_file()
    jobs={}
    for name in ['verify','explain']:
        target=r/name
        receipt=[json.loads(l) for l in (target/'launcher.log').read_text().splitlines() if l.startswith('{') and '"result"' in l][-1]
        report=read(receipt['result'])
        jobs[name]=dict(job=receipt['job'],by_node=receipt['by_node'],tasks=report['expected'],published=len(read(target/'published.json')),pool_elapsed_seconds=receipt['elapsed_including_queue'],report_sha256=sha(receipt['result']))
        assert jobs[name]['tasks']==jobs[name]['published']
    tests=(HERE/'pytest_final.log').read_text()
    assert re.search(r'304 passed',tests) and ' skipped' not in tests
    files={str(p.relative_to(HERE)):sha(p) for p in sorted(HERE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='execution_receipt.json'}
    tag=read(HERE/'FROZEN_PROTOCOL.json')['source']
    dump(HERE/'execution_receipt.json',dict(source_commit=tag,protocol_commit='4f1b163637f8eb5027dbe7d1b54595accfd431b0',
        protocol_sha256=sha(HERE/'FROZEN_PROTOCOL.json'),private_run=str(r),status='COMPLETE_EXPLORATORY',
        environment=dict(controller_python=sys.version,controller_platform=platform.platform(),libraries={k:importlib.metadata.version(k) for k in ['numpy','pandas','scipy','scikit-learn','matplotlib']}),
        original_fit_artifacts_reused=30,forest_fits=0,calibration_fits=0,members=10,triples=120,annual_raw_shrunk_exports=720,annual_versions=6,
        combination_pairs=7140,disjoint_pairs=2100,closures=['2025 24m performance NA','2023 12m recovery AUC/AP NA'],
        reference_identity='74-feature drop9_support 96 trees; root 83-feature pointer preserved and excluded',
        public_privacy_checks=dict(csv_headers_reviewed=sorted(checked),no_private_model_or_company_rows=True,real_cases_private=True,public_case_group_min_companies=5),
        numerical=read(HERE/'RECONSTRUCTION_AUDIT.json'),independent_merged_max_error=read(HERE/'INDEPENDENT_AUDIT.json')['max_error'],jobs=jobs,
        tests='304 passed incl independent R checks; one upstream EBM visualization warning; three new targeted audit tests',
        source_hashes_identical_before_after=True,failures=[],omitted_fits='Design expressly forbids new forest or calibrator fits.',
        files_sha256=files))
if __name__=='__main__':deliver()
