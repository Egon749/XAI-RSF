"""Freeze an immutable source snapshot, then verify six saved ensembles in Slurm."""
import argparse,io,subprocess,tarfile
from .common import *
def freeze():
    r=run_path();assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO).strip()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    files=[p for p in subprocess.check_output(['git','-c','core.quotePath=false','ls-files'],cwd=REPO,text=True).splitlines()if p.endswith('.py')]
    with tarfile.open(fileobj=io.BytesIO(subprocess.check_output(['git','archive',commit,'--',*files],cwd=REPO)))as t:t.extractall(r/'input/code',filter='data')
    write(r/'audit/code_snapshot.json',dict(commit=commit,protocol_sha256=sha(r/'input/protocol.json')))
    write(r/'audit/input_hashes.json',{str(p.relative_to(r/'input')):sha(p)for p in(r/'input').rglob('*')if p.is_file()})
def execute():
    r=run_path();root=r/'input';items=tasks();manifest=[]
    for row in items:
        argv=['python3',str(root/f'code/experiments/{NAME}/launch.py'),'--year',str(row['year']),'--arm',row['arm']]
        manifest.append(dict(kind='cpu',argv=argv,cwd=str(root/'code'),timeout=3600,storage=dict(source=str(root),mode='nfs',results=str(r/'results'))))
    (r/'tasks.jsonl').write_text(''.join(json.dumps(x)+'\n'for x in manifest))
    (r/'exports.py').write_text('# Six independent saved-forest exports; no fitting.\n')
    write(r/'exports.py.hetero.json',dict(independent=True,manifest='tasks.jsonl'))
    receipt=None
    with(r/'audit/launcher.log').open('x')as log:
        p=subprocess.Popen(['/home/oliver/.local/bin/hetero-run',str(r/'exports.py')],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        for line in p.stdout:
            log.write(line);log.flush();print(line,end='',flush=True)
            try:
                z=json.loads(line)
                if isinstance(z,dict)and'job'in z and'result'in z:receipt=z
            except json.JSONDecodeError:pass
        rc=p.wait()
    if rc or not receipt or not receipt['success']:raise RuntimeError('Preserve failure; no automatic repeats')
    report=load(receipt['result']);assert report['expected']==6 and not report['pending'];rows=[]
    for task in report['completed']:
        assert task['returncode']==0 and task['compute_completed']and task['publication_completed']
        i=int(task['id']);out=r/f'results/{receipt["job"]}/task-{i}/output'
        for n,h in load(out/'hashes.json').items():assert sha(out/n)==h
        s=load(out/'summary.json');assert s['status']=='COMPLETE'and s['forest_fits']==0
        rows.append(dict(**items[i],path=str(out),job=receipt['job'],task=i,host=s['host']))
    assert len(rows)==6;write(r/'audit/registry.json',rows)
    write(r/'audit/gate.json',dict(status='PASS',tasks=6,**receipt,report_sha256=sha(Path(receipt['result']))))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','exports']);a=p.parse_args()
    freeze()if a.stage=='freeze'else execute()
