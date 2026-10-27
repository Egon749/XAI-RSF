"""Gate completed publication, numeric invariants and exact expected task IDs."""
import argparse,json,hashlib
from pathlib import Path

def main(run,stage,job):
    run=Path(run);pool=json.loads((run/f'{stage}.jsonl.results-{job}.json').read_text())
    expected={'pilot':5,'training':240,'assembly':30}[stage]
    done=pool['completed'];assert len(done)==expected and {x['id'] for x in done}==set(range(expected))
    assert all(x['returncode']==0 and x['compute_completed'] and x['publication_completed'] for x in done)
    results='assembled' if stage=='assembly' else stage
    entries=[]
    for x in done:
        out=run/'results'/results/str(job)/f"task-{x['id']}"/'output'
        if stage=='training':
            d=json.loads((out/'shard_summary.json').read_text())
            assert hashlib.sha256((out/'shard_PRIVATE.json').read_bytes()).hexdigest()==d['model_sha256']
        else:
            d=json.loads((out/'summary.json').read_text())
            assert d['reload_error']==0 and d['path_violations']==0
            if d['reconstruction_error'] is not None:assert d['reconstruction_error']<1e-10
            for f,h in json.loads((out/'output_hashes.json').read_text()).items():
                assert hashlib.sha256((out/f).read_bytes()).hexdigest()==h
        entries.append({k:d[k] for k in ['fold','seed','variant','seconds','host']})
    receipt=dict(status='PASS',stage=stage,job=job,tasks=expected,entries=entries,
        note='Numerical and resource gate only; no prediction metric used for configuration changes.')
    with (run/'audit'/f'{stage}_gate.json').open('x') as f:json.dump(receipt,f,indent=2)
    print(json.dumps(receipt))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--stage',required=True);p.add_argument('--job',required=True);a=p.parse_args();main(a.run,a.stage,a.job)
