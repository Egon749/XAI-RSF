from pathlib import Path
import argparse,json,subprocess,tarfile
from experiments.post_st_prognosis_v1.common import sha,write
from .prepare import prepare


def freeze(run,repo):
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True)
    commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    for p in (run/'input/folds').glob('*/*/plan.json'):
        data=json.loads(p.read_text());data['code_commit']=commit;p.write_text(json.dumps(data,indent=2)+'\n')
    with (run/'audit/code.tar').open('xb') as f:subprocess.run(['git','-C',str(repo),'archive',commit],stdout=f,check=True)
    with tarfile.open(run/'audit/code.tar') as f:f.extractall(run/'input/code',filter='data')
    protocol=json.loads((run/'audit/protocol.json').read_text())
    for stage in ['pilot','formal']:
        items=[(y,v,s) for y in protocol['test_years']
            for v in (['termination_only'] if stage=='pilot' else protocol['weights'])
            for s in ([42] if stage=='pilot' else protocol['seeds'])]
        tasks=[dict(kind='cpu',argv=['python3',str(run/'input/code/experiments/post_st_cause_weights_h24_v1/launch.py'),
            '--year',str(y),'--variant',v,'--seed',str(s),*(['--pilot'] if stage=='pilot' else [])],
            cwd=str(run/'input/code'),timeout=3600,
            storage=dict(source=str(run/'input'),mode='nfs',results=str(run/'results'/stage))) for y,v,s in items]
        (run/f'{stage}.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in tasks))
        (run/f'{stage}.py').write_text('# Run with hetero-run; independent jobs declared in sidecar.\n')
        write(run/f'{stage}.py.hetero.json',dict(independent=True,manifest=f'{stage}.jsonl'))
    write(run/'audit/immutable_input_hashes.json',{str(p.relative_to(run/'input')):sha(p) for p in (run/'input').rglob('*') if p.is_file()})
    print(commit)


def gate(run,stage,job):
    report=run/f'{stage}.jsonl.results-{job}.json';r=json.loads(report.read_text())
    n=len((run/f'{stage}.jsonl').read_text().splitlines())
    assert r['expected']==n and len(r['completed'])==n and not r['pending'] and r['srun_returncode']==0
    assert {x['id'] for x in r['completed']}==set(range(n))
    for x in r['completed']:
        assert x['returncode']==0 and x['compute_completed'] and x['publication_completed'],x
        out=run/'results'/stage/str(job)/f"task-{x['id']}"/'output'
        for name,h in json.loads((out/'hashes.json').read_text()).items():assert sha(out/name)==h
        s=json.loads((out/'summary.json').read_text());assert s['status']=='COMPLETE'
    result=dict(status='PASS',job=job,tasks=n,seconds=r['seconds'],report_sha256=sha(report),
        by_host={h:sum(x['host']==h for x in r['completed']) for h in {x['host'] for x in r['completed']}})
    write(run/f'audit/{stage}_gate.json',result);print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--repo',type=Path)
    p.add_argument('--stage',choices=['prepare','freeze','pilot','formal'],required=True);p.add_argument('--job');a=p.parse_args()
    if a.stage=='prepare':prepare(a.run)
    elif a.stage=='freeze':freeze(a.run,a.repo)
    else:gate(a.run,a.stage,a.job)
