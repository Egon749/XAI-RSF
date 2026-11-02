import argparse,io,subprocess,tarfile
from .common import *

def freeze():
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO).strip()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    files=[p for p in subprocess.check_output(['git','ls-files'],cwd=REPO,text=True).splitlines() if p.endswith('.py')]
    with tarfile.open(fileobj=io.BytesIO(subprocess.check_output(['git','archive',commit,'--',*files],cwd=REPO))) as t:t.extractall(RUN/'input/code',filter='data')
    write(RUN/'audit/code_snapshot.json',dict(commit=commit,protocol_sha256=sha(RUN/'input/protocol.json')))
    write(RUN/'audit/immutable_input_hashes.json',{str(p.relative_to(RUN/'input')):sha(p) for p in (RUN/'input').rglob('*') if p.is_file()})

def execute(stage):
    if stage=='point':items=[dict(months=h,repeat=-1) for h in [12,24]]
    else:
        assert load(RUN/'audit/gate_point.json')['status']=='PASS'
        items=[dict(months=h,repeat=r) for r in range(50) for h in [12,24]]
    tasks=[dict(kind='cpu',argv=['python3',str(RUN/f'input/code/experiments/{NAME}/launch.py'),'--months',str(i['months']),'--repeat',str(i['repeat'])],
        cwd=str(RUN/'input/code'),timeout=3600,storage=dict(source=str(RUN/'input'),mode='nfs',results=str(RUN/f'results/{stage}'))) for i in items]
    with (RUN/f'{stage}.jsonl').open('x') as f:f.write(''.join(json.dumps(t)+'\n' for t in tasks))
    (RUN/f'{stage}.py').write_text('# Independent chronological calibration chains; zero forest fits.\n')
    write(RUN/f'{stage}.py.hetero.json',dict(independent=True,manifest=f'{stage}.jsonl'));receipt=None
    with (RUN/f'audit/launcher_{stage}.log').open('x') as log:
        proc=subprocess.Popen(['/home/oliver/.local/bin/hetero-run',str(RUN/f'{stage}.py')],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        for line in proc.stdout:
            log.write(line);log.flush();print(line,end='',flush=True)
            try:
                obj=json.loads(line)
                if isinstance(obj,dict) and 'job' in obj and 'result' in obj:receipt=obj
            except json.JSONDecodeError:pass
        rc=proc.wait()
    if rc or receipt is None or not receipt['success']:raise RuntimeError('Retain failures, do not auto-refit; inspect compute versus publication')
    rpt=load(Path(receipt['result']));assert rpt['expected']==len(items) and len(rpt['completed'])==len(items) and not rpt['pending']
    records=[]
    for task in rpt['completed']:
        assert task['returncode']==0 and task['compute_completed'] and task['publication_completed']
        i=int(task['id']);p=RUN/f'results/{stage}/{receipt["job"]}/task-{i}/output'
        for n,h in load(p/'hashes.json').items():assert sha(p/n)==h
        s=load(p/'summary.json');assert s['status']=='COMPLETE' and s['new_forests']==0
        records.append(dict(**items[i],path=str(p),host=s['host'],job=receipt['job'],task=i))
    write(RUN/f'audit/registry_{stage}.json',records)
    write(RUN/f'audit/gate_{stage}.json',dict(status='PASS',job=receipt['job'],tasks=len(items),seconds=receipt['execution_seconds'],report=receipt['result'],report_sha256=sha(Path(receipt['result']))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','point','bootstrap']);a=p.parse_args();freeze() if a.stage=='freeze' else execute(a.stage)
