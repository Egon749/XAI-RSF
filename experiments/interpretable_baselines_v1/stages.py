from pathlib import Path
import argparse,json,hashlib,subprocess,tarfile


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    with Path(p).open('x') as f:json.dump(d,f,indent=2,allow_nan=False)
def freeze(run,repo):
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True)
    code=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip();plan=json.loads((run/'input/plan.json').read_text());plan['code_commit']=code
    (run/'input/plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    with (run/'audit/code.tar').open('xb') as f:subprocess.run(['git','-C',str(repo),'archive',code],stdout=f,check=True)
    with tarfile.open(run/'audit/code.tar') as t:t.extractall(run/'input/code',filter='data')
    def task(kind,model,seed=42,extra=(),pilot=False):
        return dict(kind='cpu',argv=['python3',str(run/'input/code/experiments/interpretable_baselines_v1/launch.py'),'--kind',kind,'--model',model,'--seed',str(seed),*extra,*(['--pilot'] if pilot else [])],cwd=str(run/'input/code'),timeout=14400,storage=dict(source=str(run/'input'),mode='nfs',results=str(run/'results'/('pilot' if pilot else 'formal'))))
    pilots=[task('baseline','ebm_ga2m30',pilot=True),task('bootstrap','q2',extra=['--replicate','0'],pilot=True),task('shap','unrestricted',extra=['--permutations','32'],pilot=True)]
    tasks=[task('baseline',m,s) for m in plan['baseline_models'] for s in plan['seeds']]
    tasks += [task('bootstrap',m,42,['--replicate',str(r)]) for r in range(plan['bootstrap_replicates']) for m in plan['bootstrap_models']]
    tasks += [task('simulation',m,s,['--scenario',scenario]) for scenario in plan['simulation_scenarios'] for s in plan['simulation_seeds'] for m in plan['simulation_models']]
    tasks += [task('shap',m,s,['--permutations',str(n)]) for m in plan['shap_models'] for s in plan['shap_seeds'] for n in plan['shap_permutations']]
    assert len(tasks)==226
    for stage,items in [('pilot',pilots),('formal',tasks)]:
        with (run/f'{stage}.jsonl').open('x') as f:f.write(''.join(json.dumps(x)+'\n' for x in items))
        (run/f'{stage}.py').write_text('# Execute independent manifest with existing heteropool.\n');write(run/f'{stage}.py.hetero.json',dict(independent=True,manifest=f'{stage}.jsonl'))
    write(run/'audit/input_hashes.json',{str(p.relative_to(run/'input')):sha(p) for p in (run/'input').rglob('*') if p.is_file()})
    print(json.dumps(dict(code=code,pilot_tasks=len(pilots),formal_tasks=len(tasks))))
def gate(run,stage,job):
    p=run/f'{stage}.jsonl.results-{job}.json';report=json.loads(p.read_text());items=report['completed'];n=3 if stage=='pilot' else 226
    assert len(items)==n and {x['id'] for x in items}==set(range(n))
    failed=[x['id'] for x in items if x['returncode']!=0 or not x['publication_completed'] or not x['compute_completed']]
    assert not failed,failed
    for x in items:
        out=run/'results'/stage/str(job)/f"task-{x['id']}"/'output';s=json.loads((out/'summary.json').read_text());assert s['status']=='COMPLETE'
        for name,digest in json.loads((out/'hashes.json').read_text()).items():assert sha(out/name)==digest
        if 'reconstruction_error' in s:assert s['reconstruction_error']<1e-10
        if 'additivity_max_error' in s:assert s['additivity_max_error']<1e-10
    d=dict(stage=stage,job=int(job),tasks=n,seconds=report['seconds'],by_host={h:sum(x['host']==h for x in items) for h in sorted({x['host'] for x in items})},failed_tasks=[],status='PASS',report_sha256=sha(p))
    write(run/f'audit/{stage}_gate.json',d);print(json.dumps(d))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--repo',type=Path);p.add_argument('--stage',choices=['freeze','pilot','formal'],required=True);p.add_argument('--job');a=p.parse_args()
    freeze(a.run,a.repo) if a.stage=='freeze' else gate(a.run,a.stage,a.job)
