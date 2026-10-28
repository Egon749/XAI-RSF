from pathlib import Path
import argparse
import json
import subprocess
import tarfile
from experiments.post_st_prognosis_v1.common import sha,write


def freeze(run,repo):
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True)
    code=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    plan=json.loads((run/'input/plan.json').read_text());plan['code_commit']=code
    (run/'input/plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    with (run/'audit/code.tar').open('xb') as f:
        subprocess.run(['git','-C',str(repo),'archive',code],stdout=f,check=True)
    with tarfile.open(run/'audit/code.tar') as t:t.extractall(run/'input/code',filter='data')
    def task(model,seed,pilot=False):
        return dict(kind='cpu',argv=['python3',str(run/'input/code/experiments/post_st_train2024_h12_v1/launch.py'),
                    '--model',model,'--seed',str(seed),*(['--pilot'] if pilot else [])],cwd=str(run/'input/code'),timeout=3600,
                    storage=dict(source=str(run/'input'),mode='nfs',results=str(run/'results'/('pilot' if pilot else 'formal'))))
    for stage,tasks in [('pilot',[task(m,42,True) for m in plan['pilot']['variants']]),
                        ('formal',[task(m,s) for m in plan['variants'] for s in plan['seeds']])]:
        with (run/f'{stage}.jsonl').open('x') as f:f.write(''.join(json.dumps(x)+'\n' for x in tasks))
        (run/f'{stage}.py').write_text('# Independent fits; invoke hetero-run on this sidecar-enabled entry.\n')
        write(run/f'{stage}.py.hetero.json',dict(independent=True,manifest=f'{stage}.jsonl'))
    write(run/'audit/immutable_input_hashes.json',{str(p.relative_to(run/'input')):sha(p) for p in (run/'input').rglob('*') if p.is_file()})
    print(code)


def gate(run,stage,job):
    path=run/f'{stage}.jsonl.results-{job}.json';report=json.loads(path.read_text());items=report['completed']
    n=2 if stage=='pilot' else 9
    assert len(items)==n and {x['id'] for x in items}==set(range(n))
    failed=[]
    for x in items:
        if x['returncode'] or not x['publication_completed'] or not x['compute_completed']:
            failed.append(x['id']);continue
        out=run/'results'/stage/str(job)/f"task-{x['id']}"/'output'
        for name,digest in json.loads((out/'hashes.json').read_text()).items():assert sha(out/name)==digest
        s=json.loads((out/'summary.json').read_text());assert s['status']=='COMPLETE' and s['reload_error']<1e-12
    result=dict(status='FAIL' if failed else 'PASS',stage=stage,job=int(job),tasks=n,failed_tasks=failed,
        seconds=report['seconds'],by_host={h:sum(x['host']==h for x in items) for h in sorted({x['host'] for x in items})},report_sha256=sha(path))
    write(run/f'audit/{stage}_gate.json',result);print(json.dumps(result));assert not failed,failed


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--repo',type=Path)
    p.add_argument('--stage',choices=['freeze','pilot','formal'],required=True);p.add_argument('--job');a=p.parse_args()
    freeze(a.run,a.repo) if a.stage=='freeze' else gate(a.run,a.stage,a.job)
