"""Check the release surface against the private source and intended products."""
import platform,sys,importlib.metadata
from .common import *
def deliver():
    r=run();public=list(D.glob('*.csv'))
    forbidden={'company_id','sample_id','sample_ids','stock_code','row','risk','threshold','origin_date','prediction','case_id'}
    for p in public:assert not (set(pd.read_csv(p,nrows=0).columns)&forbidden),p
    assert not [p for p in D.rglob('*') if 'PRIVATE' in p.name or p.suffix in ('.pkl','.npz','.so','.npy','.parquet')]
    assert len(list((D/'figures').glob('*.pdf')))==len(list((D/'figures').glob('*.png')))==6
    assert len(list((r/'private_case_figures').glob('*.pdf')))==24
    expected={'FEATURE_GROUP_MAPPING':74,'RANDOM_MAPS':200,'EXACT_MODEL_CHECKS':396,'RANDOM_GROUP_BASELINE':201600,'GROUP_CANCELLATION_AUDIT':201*12*8,'GROUP_INTERACTIONS':12*3*28,'GROUP_MODEL_DISTRIBUTIONS':132*12*(8+8+28+4)}
    for name,n in expected.items():assert len(pd.read_csv(D/(name+'.csv')))==n
    jobs={}
    for st in ['verify','analysis']:
        receipt=[json.loads(l) for l in (r/st/'launcher.log').read_text().splitlines() if l.startswith('{') and '"result"' in l][-1]
        report=read(receipt['result']);assert len(read(r/st/'published.json'))==report['expected']
        jobs[st]=dict(job=receipt['job'],tasks=report['expected'],success=receipt['success'],by_node=receipt['by_node'],elapsed_seconds_with_queue=receipt['elapsed_including_queue'],report_sha256=sha(receipt['result']),worker_code_manifest_sha256=sha(r/(st+'_code_hashes.json')))
    source=read(r/'source_hashes.json')
    assert (r/'postcheck.log').exists() and (r/'postcheck.success').exists()
    assert '308 passed' in (D/'pytest_final.log').read_text() and ' skipped' not in (D/'pytest_final.log').read_text()
    files={str(p.relative_to(D)):sha(p) for p in sorted(D.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='execution_receipt.json'}
    dump(D/'execution_receipt.json',dict(status='COMPLETE_EXPLORATORY',source_commit=read(D/'FROZEN_PROTOCOL.json')['parent'],protocol_commit='270b03ee5db5dae19c1709bd4592482b0ffda671',
        hash_before_after_match=True,private_source_hash_manifest_sha256=sha(r/'source_hashes.json'),sources_individually_verified=len(source['hashes']),
        current_model_pointer_sha256=source['current_model_pointer'],main_bundle_spec_sha256=source['main_bundle_spec'],private_run=str(r),
        forest_fits=0,calibration_fits=0,new_trees=0,formal_performance_metrics_computed=0,annual_predictors=396,unique_triples=120,random_mappings=200,
        data_scope='actual train and held-out cohort rows; post-exposure; 2025 H24 function only; shrunk versions',
        numerical=read(D/'EXACT_RECONSTRUCTION_AUDIT.json'),independent_endpoints=read(D/'INDEPENDENT_ENDPOINT_AUDIT.json'),
        independent_random_metric_error=read(D/'CONTROL_AUDIT.json')['max_error'],jobs=jobs,
        tests='308 passed incl independent R; 1 upstream EBM visualization warning',formal_job_failures=[],public_privacy_checked=True,
        environment=dict(python=sys.version,platform=platform.platform(),libraries={k:importlib.metadata.version(k) for k in ['numpy','scipy','pandas','matplotlib']}),sha256=files))
if __name__=='__main__':deliver()
