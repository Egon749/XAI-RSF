import argparse,io,subprocess,tarfile,shutil
from .common import *

def freeze(attempt=1):
    source=RUN/('input' if attempt==1 else f'input_attempt{attempt}')
    if attempt>1:
        shutil.copytree(RUN/'input',source,ignore=shutil.ignore_patterns('code'))
    suffix='' if attempt==1 else f'_attempt{attempt}'
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO).strip()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    files=[p for p in subprocess.check_output(['git','ls-files'],cwd=REPO,text=True).splitlines() if p.endswith('.py')]
    with tarfile.open(fileobj=io.BytesIO(subprocess.check_output(['git','archive',commit,'--',*files],cwd=REPO))) as t:t.extractall(source/'code',filter='data')
    write(RUN/f'audit/code_snapshot{suffix}.json',dict(commit=commit,protocol_sha256=sha(source/'protocol.json')))
    write(RUN/f'audit/input_hashes{suffix}.json',{str(p.relative_to(source)):sha(p) for p in source.rglob('*') if p.is_file()})

def execute(attempt=1):
    source=RUN/('input' if attempt==1 else f'input_attempt{attempt}')
    suffix='' if attempt==1 else f'_attempt{attempt}'
    tasks=[dict(kind='cpu',argv=['python3',str(source/f'code/experiments/{NAME}/launch.py'),'--year',str(y)],cwd=str(source/'code'),
        timeout=3600,storage=dict(source=str(source),mode='nfs',results=str(RUN/'results/diagnostics'))) for y in range(2019,2026)]
    with (RUN/f'tasks{suffix}.jsonl').open('x') as f:f.write(''.join(json.dumps(t)+'\n' for t in tasks))
    (RUN/f'diagnostics{suffix}.py').write_text('# seven independent saved-forest diagnostic tasks\n');write(RUN/f'diagnostics{suffix}.py.hetero.json',dict(independent=True,manifest=f'tasks{suffix}.jsonl'))
    receipt=None
    with (RUN/f'audit/launcher{suffix}.log').open('x') as log:
        proc=subprocess.Popen(['/home/oliver/.local/bin/hetero-run',str(RUN/f'diagnostics{suffix}.py')],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        for line in proc.stdout:
            print(line,end='',flush=True);log.write(line);log.flush()
            try:
                v=json.loads(line)
                if isinstance(v,dict) and 'job' in v and 'result' in v:receipt=v
            except json.JSONDecodeError:pass
        rc=proc.wait()
    if rc or not receipt or not receipt['success']:raise RuntimeError('Preserve failures; inspect compute/publication before any retry')
    report=load(receipt['result']);assert report['expected']==7 and len(report['completed'])==7 and not report['pending'];records=[]
    for t in report['completed']:
        assert t['returncode']==0 and t['compute_completed'] and t['publication_completed']
        p=RUN/f'results/diagnostics/{receipt["job"]}/task-{t["id"]}/output'
        for n,h in load(p/'hashes.json').items():assert sha(p/n)==h
        s=load(p/'summary.json');assert s['status']=='COMPLETE' and s['year']==2019+int(t['id']) and s['new_forests']==0
        records.append(dict(path=str(p),**s))
    write(RUN/'audit/registry.json',records);write(RUN/'audit/gate.json',dict(status='PASS',**receipt))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','execute']);p.add_argument('--attempt',type=int,default=1);a=p.parse_args();(freeze if a.stage=='freeze' else execute)(a.attempt)
