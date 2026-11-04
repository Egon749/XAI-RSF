import shutil,subprocess,sys
from .common import *
def submit(stage):
    r=run(); data=r/('input' if stage=='fit' else 'ensemble_input')
    spec=read(r/'input/admissibility.json');arms=fit_arms(spec)
    if stage=='ensemble':
        data.mkdir()
        shutil.copytree(r/'input/folds',data/'folds')
        for name in ['admissibility.json','features.csv']:shutil.copy2(r/'input'/name,data/name)
        members={'|'.join(map(str,[z['year'],z['arm'],z['variant'],z['seed']])):z['path'] for z in read(r/'fit/published.json')}
        dump(data/'members.json',members)
    for p in [*list((ROOT/'experiments').rglob('*.py')),*list((ROOT/'crforest').glob('*.py'))]:
        dst=data/'code'/p.relative_to(ROOT);dst.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(p,dst)
    entries=[[stage,str(y),arm,v,str(seed)] for y in YEARS for arm in arms for v in variants(arm) for seed in (SEEDS if stage=='fit' else [0])]
    work=r/stage;work.mkdir()
    rows=[dict(kind='cpu',argv=['python3',str(data/'code/experiments'/D.name/'launch.py'),*a],cwd=str(data/'code'),timeout=3600,storage=dict(source=str(data),results=str(work/'results'),mode='nfs')) for a in entries]
    manifest=work/'tasks.jsonl';manifest.write_text(''.join(json.dumps(z)+'\n' for z in rows))
    with (work/'launcher.log').open('x') as log:subprocess.run(['/home/oliver/.local/bin/hetero-run',str(manifest)],stdout=log,stderr=subprocess.STDOUT,check=True)
    receipt=[json.loads(l) for l in (work/'launcher.log').read_text().splitlines() if l.startswith('{') and '"result"' in l][-1]
    report=read(receipt['result']);assert receipt['success'] and report['expected']==len(rows) and not report['pending']
    published=[]
    for item in report['completed']:
        idx=int(item['id']);p=work/f'results/{receipt["job"]}/task-{idx}/output'
        assert item['compute_completed'] and item['publication_completed']
        for name,h in read(p/'hashes.json').items():assert sha(p/name)==h
        s=read(p/'summary.json');assert [s['stage'],str(s['year']),s['arm'],s['variant'],str(s['seed'])]==entries[idx]
        published.append(dict(path=str(p),**{k:s[k] for k in ['year','arm','seed','variant','host','forest_fits']}))
    assert len(published)==len(rows);dump(work/'published.json',published)
if __name__=='__main__':submit(sys.argv[1])
