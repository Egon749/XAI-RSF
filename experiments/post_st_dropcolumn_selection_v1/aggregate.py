from pathlib import Path
import argparse,io,json,subprocess,tarfile
from .common import *

def prepare(run,repo,kind):
    registry=load(run/f'audit/registry_{kind}.json');sets=load(run/'input/subsets.json');groups={}
    for r in registry:groups.setdefault((r['year'],r['subset_id'],r['repeat']),[]).append(r)
    items=[]
    for (year,sid,rep),rr in sorted(groups.items()):
        rr=sorted(rr,key=lambda z:z['seed']);assert [r['seed'] for r in rr]==SEEDS
        paths=[str(Path(r['path'])/'shrunk') for r in rr]
        items.append(dict(kind=kind,year=year,subset_id=sid,repeat=rep,keep=sets[sid]['keep'],mode=sets[sid]['mode'],paths=paths,forest_hashes=[sha(Path(p)/'forest_PRIVATE.json') for p in paths]))
    if kind=='outer':
        arms=load(run/'input/arms.json')
        for year in YEARS:
            for arm,label in [('reference83','full83'),('drop9','legacy74')]:
                sid=next(a['subset_id'] for a in arms if a['label']==label and sets[a['subset_id']]['mode']=='count')
                paths=[str(p) for s,p in sorted(references(year,arm,repo).items())]
                items.append(dict(kind='outer',year=year,subset_id=sid,repeat=-1,keep=sets[sid]['keep'],mode='count',paths=paths,forest_hashes=[sha(Path(p)/'forest_PRIVATE.json') for p in paths],reused=True))
    dest=run/f'aggregate_input_{kind}';dest.mkdir()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    tracked=subprocess.check_output(['git','ls-files'],cwd=repo,text=True).splitlines();paths=[p for p in tracked if p.endswith('.py')]
    with tarfile.open(fileobj=io.BytesIO(subprocess.check_output(['git','archive',commit,'--',*paths],cwd=repo))) as f:f.extractall(dest/'code',filter='data')
    write(dest/'plan.json',dict(run=str(run),kind=kind,items=items,analysis_commit=commit))
    stage='aggregate_'+kind
    tasks=[dict(kind='cpu',argv=['python3',str(dest/'code/experiments'/NAME/'analysis_launch.py'),'--index',str(i)],cwd=str(dest/'code'),timeout=3600,
        storage=dict(source=str(dest),mode='nfs',results=str(run/'results'/stage))) for i in range(len(items))]
    with (run/f'{stage}.jsonl').open('x') as f:f.write(''.join(json.dumps(t)+'\n' for t in tasks))
    (run/f'{stage}.py').write_text('# Exact ensemble compilation only; no new forest fitting.\n');write(run/f'{stage}.py.hetero.json',dict(independent=True,manifest=f'{stage}.jsonl'))
    write(run/f'audit/manifest_{stage}.json',dict(tasks=len(tasks),manifest_sha256=sha(run/f'{stage}.jsonl'),plan_sha256=sha(dest/'plan.json'),analysis_commit=commit))
    print(json.dumps(dict(stage=stage,tasks=len(tasks),forest_fits=0)),flush=True)

def gate(run,kind,job,report):
    stage='aggregate_'+kind;r=load(report);items=load(run/f'aggregate_input_{kind}/plan.json')['items']
    assert r['expected']==len(items) and not r['pending'] and r['srun_returncode']==0
    states={int(x['id']):x for x in r['completed']};assert set(states)==set(range(len(items)))
    registry=[];failed=[]
    for i,spec in enumerate(items):
        state=states[i];out=run/f'results/{stage}/{job}/task-{i}/output'
        if not(state['returncode']==0 and state['compute_completed'] and state['publication_completed']):failed.append(i);continue
        summary=load(out/'summary.json');assert summary['status']=='COMPLETE' and summary['new_forest_fits']==0
        for n,h in load(out/'hashes.json').items():assert sha(out/n)==h
        registry.append({**spec,'path':str(out),'host':summary['host']})
    write(run/f'audit/registry_{stage}.json',registry)
    g=dict(status='PASS' if not failed else 'FAILED',stage=stage,job=job,tasks=len(items),failed=failed,seconds=r['seconds'],compute_and_publication_verified=not failed,report_sha256=sha(report),by_host={h:sum(x['host']==h for x in registry) for h in sorted({x['host'] for x in registry})})
    write(run/f'audit/gate_{stage}.json',g);print(json.dumps(g),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--kind',choices=['inner','outer','bootstrap'],required=True);p.add_argument('--action',choices=['prepare','gate'],required=True);p.add_argument('--job',type=int);p.add_argument('--report',type=Path);a=p.parse_args()
    if a.action=='prepare':prepare(a.run,Path(__file__).resolve().parents[2],a.kind)
    else:gate(a.run,a.kind,a.job,a.report)
