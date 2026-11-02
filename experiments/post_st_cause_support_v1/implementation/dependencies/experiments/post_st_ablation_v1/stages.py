"""Verify pool receipts; schedule only read-only saved-model checks after fits."""
from pathlib import Path
import argparse,json,shutil
from .common import *

def gate(run,stage,job,report):
    tasks=[json.loads(x) for x in (run/f'{stage}.jsonl').read_text().splitlines()]
    incoming=json.loads(report.read_text())
    for i,task in enumerate(tasks):
        expected={k:task.get(k) for k in ('argv','cwd','env','kind','storage','min_gpu_mb')}
        assert json.loads(incoming['task_signatures'][str(i)])==expected
    path=run/f'{stage}.jsonl.results-{job}.json'
    if path.exists():assert sha(path)==sha(report)
    else:shutil.copy2(report,path)
    items=incoming['completed'];assert len(items)==len(tasks) and {x['id'] for x in items}==set(range(len(tasks))) and not incoming['pending']
    failures=[]
    for x in items:
        if x['returncode'] or not x['compute_completed'] or not x['publication_completed']:
            failures.append(dict(task=x['id'],host=x['host'],phase=x.get('phase'),returncode=x['returncode'],log=x['log']));continue
        out=run/'results'/stage/str(job)/f"task-{x['id']}"/'output'
        for n,h in json.loads((out/'hashes.json').read_text()).items():assert sha(out/n)==h
        assert json.loads((out/'summary.json').read_text())['status']=='COMPLETE'
    result=dict(status='FAIL' if failures else 'PASS',stage=stage,job=int(job),tasks=len(tasks),failed_tasks=failures,seconds=incoming['seconds'],
        by_host={h:sum(x['host']==h for x in items) for h in sorted({x['host'] for x in items})},report_sha256=sha(path),compute_and_publication_verified=not failures)
    write(run/f'audit/{stage}_gate.json',result);print(json.dumps(result));assert not failures,failures

def verification_manifest(run):
    assert json.loads((run/'audit/formal_gate.json').read_text())['status']=='PASS'
    # One job per saved single-seed or complete three-seed predictor. No new fit.
    tasks=[dict(kind='cpu',argv=['python3',str(run/'input/code/experiments/post_st_ablation_v1/launch.py'),
        '--mode','inspect','--year',str(y),'--seed',str(s),'--arm',a,'--run',str(run)],cwd=str(run/'input/code'),timeout=3600,
        storage=dict(source=str(run/'input'),mode='nfs',results=str(run/'results/verify')))
        for y in YEARS for a in ARMS for s in SEEDS+[0]]
    with (run/'verify.jsonl').open('x') as f:f.write(''.join(json.dumps(t)+'\n' for t in tasks))
    (run/'verify.py').write_text('# No fitting; independent saved-model verifications.\n')
    write(run/'verify.py.hetero.json',dict(independent=True,manifest='verify.jsonl'))
    print(len(tasks))

def validation_outputs(run):
    gate=json.loads((run/'audit/verify_gate.json').read_text());assert gate['status']=='PASS'
    for i,line in enumerate((run/'verify.jsonl').read_text().splitlines()):
        argv=json.loads(line)['argv'];get=lambda k:argv[argv.index(k)+1]
        y,s,a=int(get('--year')),int(get('--seed')),get('--arm')
        yield y,a,'ensemble' if s==0 else str(s),run/'results/verify'/str(gate['job'])/f'task-{i}/output'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--stage',choices=['formal','verify','verification_manifest'],required=True);p.add_argument('--job');p.add_argument('--report',type=Path);a=p.parse_args()
    verification_manifest(a.run) if a.stage=='verification_manifest' else gate(a.run,a.stage,a.job,a.report)
