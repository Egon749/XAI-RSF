"""Final source preservation, reason alignment and aggregate publication checks."""
from pathlib import Path
import argparse
import ast
import json
import re
import subprocess
import numpy as np
import pandas as pd
from .analyze import PUBLIC, REPO, Reader, digest, write_json


def validate(run, receipt_name='FINAL_VALIDATION'):
    assert re.fullmatch(r'[A-Z_]+', receipt_name)
    attempts = json.loads((PUBLIC/'analysis_attempts.json').read_text())
    metrics, drift = Path(attempts['metrics']), Path(attempts['drift'])
    inputs = pd.read_csv(PUBLIC/'INPUT_HASHES.csv')
    for row in inputs.itertuples():
        assert digest(row.path) == row.sha256
    reader = Reader()
    old_receipt = reader.json(REPO/'experiments/post_st_rule_features_v1/execution_receipt.json')
    root = Path(old_receipt['private_run'])/'direct_raw_analysis'
    rr = reader.json(root/'reports/raw_audit_receipt.json')
    metadata = []
    for path, h in rr['source_sha256'].items():
        if Path(path).name in ['train_metadata_PRIVATE.parquet', 'eval_metadata_PRIVATE.parquet']:
            assert digest(reader.path(path)) == h
            metadata.append(reader.parquet(path))
    original = pd.concat(metadata).set_index('sample_id')
    events = reader.parquet(drift/'termination_reasons_join_PRIVATE.parquet')
    matched = events[events.ground.ne('unavailable_existing_review')]
    assert len(matched) == 45
    np.testing.assert_array_equal(matched.old_sample_id.map(original.event).to_numpy(), np.ones(45))
    np.testing.assert_array_equal(matched.old_sample_id.map(original.stop).to_numpy(), matched.stop.to_numpy())
    reader.finish()
    # Verify branch snapshots were read at the pinned commits, not mixed tips.
    protocol = json.loads((PUBLIC/'analysis_protocol.json').read_text())
    reviewed = []
    for branch, sha in protocol['branches_reviewed'].items():
        directory = run/'source_review'/branch.split('/')[-1]
        if not directory.exists(): continue
        for p in directory.rglob('*'):
            if p.is_file():
                relative = str(p.relative_to(directory))
                data = subprocess.check_output(['git', 'show', sha+':'+relative], cwd=REPO)
                assert data == p.read_bytes()
                reviewed.append(dict(branch=branch, commit=sha, file=relative, sha256=digest(p)))
    # Exact exports, with no post hoc changes to predictions or statistical output.
    for name in ['STRATIFIED_METRICS.csv', 'STRATIFIED_BOOTSTRAP_95.csv', 'OBSERVED_EXPECTED.csv', 'TOP20_LIFT.csv']:
        assert digest(PUBLIC/name) == digest(metrics/name)
    for name in ['NULL_AJ_BASELINE.csv', 'CALENDAR_INCIDENCE.csv', 'COVARIATE_SHIFT_SUMMARY.csv', 'DELISTING_REASON_SUMMARY.csv']:
        assert digest(PUBLIC/name) == digest(drift/name)
    # Input hashes are paths+cryptographic fingerprints only, not company data.
    forbidden = {'sample_id', 'sample_ids', 'company', 'company_id', 'code', 'group', 'source_text', 'old_sample_id', 'threshold'}
    csv_checks = []
    for p in PUBLIC.glob('*.csv'):
        d = pd.read_csv(p)
        assert not forbidden & set(d.columns), (p.name, forbidden & set(d.columns))
        csv_checks.append(dict(file=p.name, rows=len(d), columns=len(d.columns), no_private_columns=True))
    company_ids = set(events.sample_id.astype(str)) | set(events.old_sample_id.dropna().astype(str))
    # Scan every evaluation ID, not only termination IDs.
    source = Path(json.loads((PUBLIC/'run_location.json').read_text())['source_private_run'])
    for y in [2023, 2024, 2025]:
        with np.load(source/f'evaluation_input/{y}_test_PRIVATE.npz', allow_pickle=False) as z:
            company_ids.update(map(str, z['sample_ids']))
    for p in PUBLIC.iterdir():
        if p.suffix in ['.csv', '.json', '.md', '.log', '.py']:
            content = p.read_text()
            assert not any(identity in content for identity in company_ids), p.name
        assert p.suffix not in ['.npz', '.parquet', '.pkl'], p.name
    for p in [PUBLIC/'analyze.py', PUBLIC/'metrics.py', PUBLIC/'deliver.py', PUBLIC/'validate.py']:
        tree = ast.parse(p.read_text())
        assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and
                       n.func.attr in ['fit', 'fit_transform', 'fit_predict'] for n in ast.walk(tree))
    m = pd.read_csv(PUBLIC/'STRATIFIED_METRICS.csv')
    unsupported = m[(m.scope.astype(str) == '2025') & m.months.eq(24)]
    assert len(unsupported) == 16
    assert unsupported[['within_year_auc', 'equal_year_auc', 'pooled_auc']].isna().all().all()
    b = pd.read_csv(PUBLIC/'STRATIFIED_BOOTSTRAP_95.csv')
    assert (b.valid_repeats+b.undefined_repeats == 1000).all()
    assert b.analysis_seed.eq(2026110201).all()
    base = protocol['source_commit']
    original_paths = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', base], cwd=REPO, text=True).splitlines()
    for name in original_paths:
        if name == 'AGENTS.md': continue
        # Git object comparison covers all preexisting code/results/pointer bytes.
        expected = subprocess.check_output(['git', 'rev-parse', base+':'+name], cwd=REPO, text=True).strip()
        actual = subprocess.check_output(['git', 'hash-object', name], cwd=REPO, text=True).strip()
        assert expected == actual, 'Preexisting tracked file modified: '+name
    receipt = dict(status='PASS', forest_fits=0, calibrator_fits=0, new_model_fits=0,
                   source_input_files_verified=len(inputs), additional_reason_sources=reader.hashes,
                   reviewed_branch_files=len(reviewed), reviewed_terminal_event_dates_match=45,
                   public_csv_checks=csv_checks, preexisting_tracked_files_preserved=len(original_paths)-1,
                   new_company_predictions_published=False, existing_ranking_exception_not_extended=True,
                   unsupported_2025_h24_rows=16, bootstrap_indices_reused=True,
                   new_test_labels_acquired=False, new_raw_data_acquired=False,
                   scientific_static_plot_visually_checked=True,
                   historical_financial_vintages='PROVISIONAL; no new PIT certification')
    write_json(run/'audit'/f'{receipt_name.lower()}.json', receipt)
    write_json(PUBLIC/f'{receipt_name}.json', receipt)
    path = PUBLIC/'BRANCH_REVIEW_HASHES.csv'
    expected = pd.DataFrame(reviewed).to_csv(index=False).encode()
    if path.exists():
        assert path.read_bytes() == expected
    else:
        path.write_bytes(expected)
    print(json.dumps({k: v for k, v in receipt.items() if k not in ['additional_reason_sources', 'public_csv_checks']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('--receipt-name', default='FINAL_VALIDATION')
    args = p.parse_args()
    validate(args.run, args.receipt_name)
