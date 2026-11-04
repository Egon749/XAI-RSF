"""Launch independent read-only tasks via the existing pool and verify publication."""
import shutil,subprocess,sys
from .common import *

def submit(stage,attempt):
    r=run_path();root=r/('input' if stage=='verify' else 'analysis_input'); snapshot=root/'code'/ 'experiments'/NAME
    snapshot.mkdir(exist_ok=True,parents=True)
    for path in HERE.glob('*.py'):
        dest=snapshot/path.name
        if not dest.exists():shutil.copy2(path,dest)
        else:assert sha(path)==sha(dest),path
    for d in ['crforest','experiments']:
        if d=='crforest':
            if not (root/'code'/d).exists():shutil.copytree(ROOT/d,root/'code'/d,ignore=shutil.ignore_patterns('__pycache__'))
        else:
            for p in (ROOT/d).rglob('*.py'):
                dest=root/'code'/p.relative_to(ROOT)
                if not dest.exists():
                    dest.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(p,dest)
    items=[]
    for y in YEARS:
        for v in VERSIONS:
            for extra in ([list(map(str,[i,min(i+10,120)])) for i in range(0,120,10)] if stage=='verify' else [[]]):
                items.append([str(y),v,*extra])
    target=r/(stage+attempt);target.mkdir()
    rows=[dict(kind='cpu',argv=['python3',str(snapshot/'launch.py'),stage,*args],cwd=str(root/'code'),timeout=7200,
               storage=dict(source=str(root),results=str(target/'results'),mode='nfs')) for args in items]
    manifest=target/'tasks.jsonl';manifest.write_text(''.join(json.dumps(z)+'\n' for z in rows))
    with (target/'launcher.log').open('x') as log:
        p=subprocess.run(['/home/oliver/.local/bin/hetero-run',str(manifest)],stdout=log,stderr=subprocess.STDOUT)
    p.check_returncode()
    receipt=[json.loads(l) for l in (target/'launcher.log').read_text().splitlines() if l.startswith('{') and '"result"' in l][-1]
    report=read(receipt['result']); assert receipt['success'] and report['expected']==len(items) and not report['pending']
    published=[]
    for item in report['completed']:
        assert item['returncode']==0 and item['compute_completed'] and item['publication_completed']
        idx=int(item['id']);dest=target/f'results/{receipt["job"]}/task-{idx}/output'
        assert read(dest/'done.json')['args']==items[idx]
        published.append(dict(args=items[idx],path=str(dest),host=read(dest/'done.json')['host']))
    assert len(published)==len(items)
    dump(target/'published.json',published)
if __name__=='__main__':submit(sys.argv[1],sys.argv[2] if len(sys.argv)>2 else '')
