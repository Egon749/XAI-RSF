import argparse,io,subprocess,tarfile
from .common import *

def freeze():
    r=run_path();assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO).strip()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    files=[p for p in subprocess.check_output(['git','-c','core.quotePath=false','ls-files'],cwd=REPO,text=True).splitlines()if p.endswith('.py')]
    write(r/'input/data_hashes.json',{str(p.relative_to(r/'input')):sha(p)for p in(r/'input').rglob('*')if p.is_file()})
    with tarfile.open(fileobj=io.BytesIO(subprocess.check_output(['git','archive',commit,'--',*files],cwd=REPO)))as t:t.extractall(r/'input/code',filter='data')
    write(r/'audit/code_snapshot.json',dict(commit=commit,protocol_sha256=sha(r/'input/protocol.json')))
    write(r/'audit/input_hashes.json',{str(p.relative_to(r/'input')):sha(p)for p in(r/'input').rglob('*')if p.is_file()})

def execute(stage):
    r=run_path();root=r/'input'
    items=[dict(year=y,seed=s,count=b)for b in NEW_COUNTS for y in YEARS for s in [0,*SEEDS]]
    assert len(items)==36
    tasks=[]
    for row in items:
        argv=['python3',str(root/f'code/experiments/{NAME}/launch.py')]
        for k,v in row.items():argv+=['--'+k,str(v)]
        tasks.append(dict(kind='cpu',argv=argv,cwd=str(root/'code'),timeout=3600,
            storage=dict(source=str(root),mode='nfs',results=str(r/f'results/{stage}'))))
    (r/f'{stage}.jsonl').write_text(''.join(json.dumps(t)+'\n'for t in tasks))
    (r/f'{stage}.py').write_text('# Independent saved-prefix exports; no fitting.\n')
    write(r/f'{stage}.py.hetero.json',dict(independent=True,manifest=f'{stage}.jsonl'))
    receipt=None
    with(r/f'audit/launcher_{stage}.log').open('x')as log:
        p=subprocess.Popen(['/home/oliver/.local/bin/hetero-run',str(r/f'{stage}.py')],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        for line in p.stdout:
            log.write(line);log.flush();print(line,end='',flush=True)
            try:
                z=json.loads(line)
                if isinstance(z,dict)and'job'in z and'result'in z:receipt=z
            except json.JSONDecodeError:pass
        rc=p.wait()
    if rc or not receipt or not receipt['success']:raise RuntimeError('Preserve failure; no automatic repeats')
    report=load(receipt['result']);assert report['expected']==len(items)and not report['pending']
    rows=[]
    for task in report['completed']:
        assert task['returncode']==0 and task['compute_completed']and task['publication_completed']
        i=int(task['id']);out=r/f'results/{stage}/{receipt["job"]}/task-{i}/output'
        for n,h in load(out/'hashes.json').items():assert sha(out/n)==h
        s=load(out/'summary.json');assert s['status']=='COMPLETE'and s['forest_fits']==0
        rows.append(dict(**items[i],path=str(out),job=receipt['job'],task=i,host=s['host']))
    assert len(rows)==len(items);write(r/f'audit/registry_{stage}.json',rows)
    write(r/f'audit/gate_{stage}.json',dict(status='PASS',tasks=len(rows),**receipt,report_sha256=sha(Path(receipt['result']))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','formal']);a=p.parse_args()
    freeze()if a.stage=='freeze'else execute(a.stage)
