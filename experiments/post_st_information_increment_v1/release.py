import platform,sys,importlib.metadata
from .common import *
def main():
    r=run();source_hashes=read(r/'sources_PRIVATE.json');assert all(sha(Path(p))==h for p,h in source_hashes.items())
    new=read(D/'EXACT_RECONSTRUCTION_AUDIT.json')
    checks=read(D/'INDEPENDENT_CHECKS.json');assert new['formal_new_fits']==54 and new['new_ensemble_exports']==18
    m=pd.read_csv(D/'PREDICTIONS_AND_METRICS.csv');assert m.query("year=='2025' and months==24").auc.isna().all()
    assert m[m.arm.isin(['M1','M3','M3_numeric'])].auc.isna().all()
    delta=pd.read_csv(D/'INCREMENTAL_VALUE.csv')
    assert delta[delta.effect.isin(['audit_vs_finance','audit_given_market','market_given_audit','complementarity'])].delta.isna().all()
    tables=list(D.glob('*.csv'))
    for file in tables:assert not {'sample_id','company_id','stock_code','source_text','threshold','risk','origin_date'}&set(pd.read_csv(file,nrows=0).columns)
    assert not [p for p in D.rglob('*') if 'PRIVATE' in p.name or p.suffix in ['.npz','.parquet','.pkl']]
    assert len(list((D/'figures').glob('*.png')))==len(list((D/'figures').glob('*.pdf')))==6
    jobs={}
    for name in ['fit','ensemble']:
        receipt=[json.loads(l) for l in (r/name/'launcher.log').read_text().splitlines() if l.startswith('{') and '"result"' in l][-1]
        rows=read(r/name/'published.json');report=read(receipt['result']);assert report['expected']==len(rows)
        for row in rows:
            p=Path(row['path']);assert all(sha(p/n)==h for n,h in read(p/'hashes.json').items())
        jobs[name]=dict(job=receipt['job'],tasks=len(rows),by_node=receipt['by_node'],elapsed_with_queue_s=receipt['elapsed_including_queue'],report_sha256=sha(receipt['result']))
    assert '311 passed' in (D/'pytest_final.log').read_text()
    files={str(p.relative_to(D)):sha(p) for p in sorted(D.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='execution_receipt.json'}
    dump(D/'execution_receipt.json',dict(status='PARTIAL_STRICT_AUDIT_SOURCE_GATE__MARKET_EXPERIMENT_COMPLETE',decision='INCONCLUSIVE',source_commit='f01e07e8855787d8832d59d33128e6b66cc11f10',
        protocol_commit='b2c0922a8b894c158900229d965a0f9b3293cb73',private_run=str(r),forest_fits=54,primary_new_fits=18,proportional_new_fits=18,coverage_supplement_new_fits=18,
        derived_ensembles=18,archived_M3_members_verified=9,qualifying_M3_comparisons=0,all_new_representations=126,calibration_fits=0,
        blocked=['M1 and M3 formal cells, including conditional effects and four-term interaction','M3 without market coverage','2025 H24 performance'],
        failures=['No failed computational fit','First supplemental terminal-reason pass stopped on joint price and capitalization grounds. Reviewed actual decision, preserved both grounds, restarted diagnostic only. Original traceback kept privately.','Remote research registry catalog unavailable: stale tunnel; verified local source copies, report archive, disclosure index, clock evidence and hashes.'],
        source_hash_files=len(source_hashes),before_after_identical=True,source_digest=sha(r/'sources_PRIVATE.json'),jobs=jobs,
        tests='311 passed, including independent R checks; one upstream EBM visualization warning',paired_primary_independently_reproduced=checks['primary_bootstrap_independent_concordance'],
        environment=dict(python=sys.version,platform=platform.platform(),libraries={k:importlib.metadata.version(k) for k in ['numpy','pandas','scipy','scikit-learn','matplotlib']}),
        privacy='Only aggregates, model hashes and code in release; no individual forecasts, rankings, issuer identities or fitted objects.',files_sha256=files))
if __name__=='__main__':main()
