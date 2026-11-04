"""Verify aggregate release files, private inputs and the Slurm publication receipts."""
import csv
import json
import subprocess
from collections import Counter
from pathlib import Path

from .common import NAME, REPO, SOURCE_COMMIT, YEARS, arrays, load, previous_path, run_path, sha, write


def json_values(value):
    if isinstance(value, dict):
        for child in value.values():
            yield from json_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from json_values(child)
    else:
        yield str(value)


def check_public_files(public, identifiers):
    scanned = 0
    for path in public.iterdir():
        if not path.is_file():
            continue
        assert '_PRIVATE' not in path.name
        assert path.suffix in {'.csv', '.json', '.py', '.md', '.log', '.pdf', '.png'}, path.name
        if path.suffix == '.csv':
            with path.open() as stream:
                fields = {v for row in csv.DictReader(stream) for v in row.values()}
        elif path.suffix == '.json':
            fields = set(json_values(load(path)))
        else:
            continue
        assert fields.isdisjoint(identifiers), path.name
        scanned += 1
    return scanned


def main():
    private = run_path()
    public = REPO / 'experiments' / NAME
    identifiers = set()
    for year in YEARS:
        labels = Path('evaluation_input') / f'{year}_test_PRIVATE.npz'
        assert sha(private / labels) == sha(previous_path() / labels)
        for file in (private / labels, private / f'input/folds/{year}/train_PRIVATE.npz'):
            data = arrays(file)
            for key in ('sample_ids', 'groups'):
                if key in data:
                    identifiers.update(map(str, data[key].tolist()))
    bootstrap = Path('private/bootstrap_indices_PRIVATE.npz')
    assert sha(private / bootstrap) == sha(previous_path() / bootstrap)
    for folder, manifest in [('input', 'input_hashes'), ('ensemble_input', 'ensemble_input_hashes')]:
        for name, digest in load(private / f'audit/{manifest}.json').items():
            assert sha(private / folder / name) == digest
    for name, digest in load(private / 'audit/source_hashes.json').items():
        assert sha(Path(name)) == digest

    checked = 0
    for stage in ('pilot', 'formal', 'ensemble'):
        gate = load(private / f'audit/gate_{stage}.json')
        result = load(gate['result'])
        assert sha(Path(gate['result'])) == gate['report_sha256']
        assert not result['pending'] and len(result['completed']) == gate['tasks']
        assert all(t['returncode'] == 0 and t['compute_completed'] and t['publication_completed']
                   for t in result['completed'])
        for record in load(private / f'audit/registry_{stage}.json'):
            output = Path(record['path'])
            for name, digest in load(output / 'hashes.json').items():
                assert sha(output / name) == digest
                checked += 1

    protected = ['CURRENT_POST_ST_MODEL.json', '主模型', 'crforest']
    assert not subprocess.check_output(['git', 'diff', SOURCE_COMMIT, '--', *protected], cwd=REPO)
    fits = load(private / 'audit/registry_pilot.json') + load(private / 'audit/registry_formal.json')
    exports = load(private / 'audit/registry_ensemble.json')
    reused = load(private / 'input/reused_registry.json')
    assert (len(fits), len(exports), len(reused)) == (21, 6, 21)
    audit = dict(status='PASS', identifier_matches=0, structured_public_files_scanned=check_public_files(public, identifiers),
                 immutable_inputs_verified=True, protected_paths=protected, forest_fits=len(fits),
                 reused_forests=len(reused), ensemble_exports=len(exports), output_files_hash_verified=checked,
                 fit_nodes=dict(Counter(row['host'] for row in fits)))
    write(public / 'PUBLICATION_AUDIT.json', audit)
    receipt_path = public / 'execution_receipt.json'
    receipt = load(receipt_path)
    receipt['publication_audit'] = audit
    receipt['public_output_hashes'] = {p.name: sha(p) for p in public.iterdir() if p.is_file() and p != receipt_path}
    staged = private / 'audit/final_execution_receipt.json'
    write(staged, receipt)
    receipt_path.write_bytes(staged.read_bytes())
    assert all(sha(public / name) == digest for name, digest in receipt['public_output_hashes'].items())
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == '__main__':
    main()
