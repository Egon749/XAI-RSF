"""Immutable snapshots, finite manifests and publication/hash gates."""
from pathlib import Path
import argparse,datetime,io,json,subprocess,tarfile
from .common import *

def freeze(run,repo):
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo).strip()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    tracked=subprocess.check_output(['git','ls-files'],cwd=repo,text=True).splitlines()
    paths=[p for p in tracked if p.endswith('.py') or p in ['AGENTS.md','pyproject.toml']]
    with tarfile.open(fileobj=io.BytesIO(subprocess.check_output(['git','archive',commit,'--',*paths],cwd=repo))) as f:f.extractall(run/'input/code',filter='data')
    write(run/'audit/freeze.json',dict(commit=commit,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol_sha256=sha(run/'input/protocol.json')))
    write(run/'audit/immutable_input_hashes.json',{str(p.relative_to(run/'input')):sha(p) for p in (run/'input').rglob('*') if p.is_file()})
    manifest(run,'pilot')

def manifest(run,stage):
    sets=load(run/'input/subsets.json');arms=load(run/'input/arms.json');full=next(a['subset_id'] for a in arms if a['label']=='full83');jobs=[]
    if stage=='pilot':
        dropped=next(a['subset_id'] for a in arms if a['label']=='drop_00')
        ratio=next(a['subset_id'] for a in arms if a['label']=='legacy74' and sets[a['subset_id']]['mode']=='ratio')
        jobs=[dict(stage='pilot',year=2020,subset=full,seed=42),dict(stage='pilot',year=2023,subset=dropped,seed=42),dict(stage='pilot',year=2024,subset=ratio,seed=42)]
    elif stage=='inner':
        assert load(run/'audit/gate_pilot.json')['failed']==0
        jobs=[dict(stage='inner',year=v,subset=sid,seed=seed) for sid,s in sets.items() for v in FOLDS if any(v<y for y in s['outers']) for seed in SEEDS]
    elif stage=='bootstrap':
        gates=load(run/'input/support_gates.json');assert not any(g['passed'] for g in gates.values()),'Supported selection needs recursive bootstrap runner, never silently substitute fallback'
        jobs=[dict(stage='bootstrap',year=y,subset=full,seed=s,repeat=r) for y in YEARS for r in range(50) for s in SEEDS]
    elif stage=='outer':
        assert (run/'audit/selection_decisions.json').exists()
        chosen={a['subset_id'] for a in arms if a['kind']=='random9' or (a['label']=='legacy74' and sets[a['subset_id']]['mode']=='ratio')}
        jobs=[dict(stage='outer',year=y,subset=sid,seed=s) for y in YEARS for sid in sorted(chosen) for s in SEEDS]
    tasks=[]
    for spec in jobs:
        argv=['python3',str(run/'input/code/experiments'/NAME/'launch.py')]
        for key,value in spec.items():argv.extend(['--'+key,str(value)])
        tasks.append(dict(kind='cpu',argv=argv,cwd=str(run/'input/code'),timeout=3600,storage=dict(source=str(run/'input'),mode='nfs',results=str(run/'results'/stage))))
    with (run/f'{stage}.jsonl').open('x') as f:f.write(''.join(json.dumps(x)+'\n' for x in tasks))
    (run/f'{stage}.py').write_text('# Independent commands in the sidecar; no duplicate whole scripts.\n')
    write(run/f'{stage}.py.hetero.json',dict(independent=True,manifest=f'{stage}.jsonl'))
    write(run/f'audit/manifest_{stage}.json',dict(tasks=len(tasks),sha256=sha(run/f'{stage}.jsonl'),created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    print(json.dumps(dict(stage=stage,tasks=len(tasks))),flush=True)

def gate(run,stage,job,report):
    rr=load(report);lines=(run/f'{stage}.jsonl').read_text().splitlines();records=[];failures=[]
    # Never infer success from job status alone: each output manifest is checked.
    for i,line in enumerate(lines):
        out=run/f'results/{stage}/{job}/task-{i}/output'
        if not (out/'summary.json').exists():failures.append(dict(task=i,reason='MISSING_PUBLISHED_SUMMARY'));continue
        s=load(out/'summary.json');assert s['status']=='COMPLETE'
        for n,h in load(out/'hashes.json').items():assert sha(out/n)==h
        records.append(dict(task=i,path=str(out),year=s['year'],seed=s['seed'],subset_id=s['subset_id'],stage=s['stage'],host=s['host'],repeat=s.get('repeat',-1)))
    assert len(records)+len(failures)==len(lines)
    value=dict(status='PASS' if not failures else 'FAILED',job=job,stage=stage,tasks=len(lines),completed=len(records),failed=len(failures),failures=failures,report_sha256=sha(report),by_host={host:sum(r['host']==host for r in records) for host in sorted({r['host'] for r in records})})
    write(run/f'audit/gate_{stage}.json',value);write(run/f'audit/registry_{stage}.json',records);print(json.dumps(value),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--action',choices=['freeze','manifest','gate'],required=True);p.add_argument('--stage');p.add_argument('--job',type=int);p.add_argument('--report',type=Path);a=p.parse_args()
    if a.action=='freeze':freeze(a.run,Path(__file__).resolve().parents[2])
    elif a.action=='manifest':manifest(a.run,a.stage)
    else:gate(a.run,a.stage,a.job,a.report)
