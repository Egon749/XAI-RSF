"""Immutable snapshots and exact task accounting on the existing Slurm pool."""
import argparse,io,subprocess,tarfile
from .common import *

def freeze():
    r=run_path();assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO).strip()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    files=[p for p in subprocess.check_output(['git','-c','core.quotePath=false','ls-files'],cwd=REPO,text=True).splitlines()if p.endswith('.py')]
    with tarfile.open(fileobj=io.BytesIO(subprocess.check_output(['git','archive',commit,'--',*files],cwd=REPO)))as t:t.extractall(r/'input/code',filter='data')
    write(r/'audit/code_snapshot.json',dict(commit=commit,protocol_sha256=sha(r/'input/protocol.json')))
    write(r/'audit/input_hashes.json',{str(p.relative_to(r/'input')):sha(p)for p in(r/'input').rglob('*')if p.is_file()})

def execute(stage):
    r=run_path();root=r/'input';registry=None
    if stage=='pilot':items=[dict(mode='grow',year=2024,seed=42,start=32,stop=34)]
    elif stage=='grow':
        assert load(r/'audit/gate_pilot.json')['status']=='PASS'
        items=[dict(mode='grow',year=y,seed=s,start=i,stop=i+32)for y in YEARS for s in SEEDS for i in range(32,256,32)]
    elif stage=='assemble':
        assert load(r/'audit/gate_grow.json')['status']=='PASS';registry=r/'audit/registry_grow.json'
        items=[dict(mode='assemble',year=y,seed=s)for y in YEARS for s in SEEDS]
    else:
        assert load(r/'audit/gate_assemble.json')['status']=='PASS';registry=r/'audit/registry_assemble.json'
        # One largest ensemble first checks export memory before the full batch.
        items=[dict(mode='inspect',year=y,seed=s,count=b)for b in reversed(COUNTS)if b!=32 for y in YEARS for s in [0,*SEEDS]]
        if stage=='export_pilot':items=[items[0]]
        else:
            assert load(r/'audit/gate_export_pilot.json')['status']=='PASS';items=items[1:]
    tasks=[]
    for row in items:
        argv=['python3',str(root/f'code/experiments/{NAME}/launch.py')]
        for k,v in row.items():argv+=['--'+k,str(v)]
        if registry:argv+=['--registry',str(registry),'--registry-sha256',sha(registry)]
        tasks.append(dict(kind='cpu',argv=argv,cwd=str(root/'code'),timeout=3600,
            storage=dict(source=str(root),mode='nfs',results=str(r/f'results/{stage}'))))
    (r/f'{stage}.jsonl').write_text(''.join(json.dumps(t)+'\n'for t in tasks))
    (r/f'{stage}.py').write_text('# Independent original tree indices or saved-prefix exports.\n')
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
    if rc or not receipt or not receipt['success']:raise RuntimeError('Retain failure; no automatic repeat of fits')
    report=load(receipt['result']);assert report['expected']==len(items)and not report['pending']
    records=[]
    for task in report['completed']:
        assert task['returncode']==0 and task['compute_completed']and task['publication_completed']
        i=int(task['id']);out=r/f'results/{stage}/{receipt["job"]}/task-{i}/output'
        for name,h in load(out/'hashes.json').items():assert sha(out/name)==h
        result=load(out/'summary.json');assert result['status']=='COMPLETE'
        records.append(dict(**items[i],path=str(out),job=receipt['job'],task=i,host=result['host']))
    assert len(records)==len(items)
    write(r/f'audit/registry_{stage}.json',records)
    write(r/f'audit/gate_{stage}.json',dict(status='PASS',tasks=len(items),**receipt,report_sha256=sha(Path(receipt['result']))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','pilot','grow','assemble','export_pilot','inspect']);a=p.parse_args()
    freeze()if a.stage=='freeze'else execute(a.stage)
