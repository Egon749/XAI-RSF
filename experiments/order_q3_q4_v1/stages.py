"""Immutable input/code freeze, publication gates and deterministic assembly."""
from pathlib import Path
import argparse,hashlib,json,subprocess,tarfile


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    with Path(p).open('x') as f:json.dump(d,f,indent=2,ensure_ascii=False,allow_nan=False)
def manifest(run,name,tasks):
    with (run/f'{name}.jsonl').open('x') as f:f.write(''.join(json.dumps(t)+'\n' for t in tasks))
    (run/f'{name}.py').write_text('# Independent task manifest; execute with existing heteropool.\n')
    write(run/f'{name}.py.hetero.json',dict(independent=True,manifest=f'{name}.jsonl'))
def task(run,seed,variant,mode,stage,extra=()):
    code=run/'input/code'
    return dict(kind='cpu',argv=['python3',str(code/'experiments/order_q3_q4_v1/launch.py'),'--seed',str(seed),'--variant',variant,'--mode',mode,*extra],cwd=str(code),timeout=7200,storage=dict(source=str(run/'input'),mode='nfs',results=str(run/'results'/stage)))
def freeze(run,repo):
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True)
    plan=json.loads((run/'input/plan.json').read_text())
    plan['code_commit']=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    # Plan finalized once before launching tasks; all later use is read-only.
    assert not (run/'audit/code.tar').exists()
    (run/'input/plan.json').write_text(json.dumps(plan,indent=2,ensure_ascii=False)+'\n')
    with (run/'audit/code.tar').open('xb') as f:subprocess.run(['git','-C',str(repo),'archive',plan['code_commit']],stdout=f,check=True)
    with tarfile.open(run/'audit/code.tar') as f:f.extractall(run/'input/code',filter='data')
    pilot=[task(run,42,v,'fit','pilot',['--pilot']) for v in plan['variants']]
    training=[task(run,s,v,'shard','training',['--tree-start',str(i)]) for s in plan['seeds'] for v in plan['variants'] for i in range(0,32,4)]
    manifest(run,'pilot',pilot);manifest(run,'training',training)
    write(run/'audit/immutable_input_hashes.json',{str(p.relative_to(run/'input')):sha(p) for p in (run/'input').rglob('*') if p.is_file()})
    print(json.dumps(dict(code=plan['code_commit'],pilot_tasks=2,training_tasks=len(training))))
def gate(run,stage,job):
    path=run/f'{stage}.jsonl.results-{job}.json';d=json.loads(path.read_text())
    expected={'pilot':2,'training':48,'assembly':6}[stage]
    done=d['completed'];assert len(done)==expected and {x['id'] for x in done}==set(range(expected))
    assert all(x['returncode']==0 and x['compute_completed'] and x['publication_completed'] for x in done)
    for item in done:
        output=run/'results'/('assembled' if stage=='assembly' else stage)/str(job)/f"task-{item['id']}"/'output'
        if stage=='training':
            s=json.loads((output/'shard_summary.json').read_text());assert sha(output/'shard_PRIVATE.json')==s['model_sha256']
        else:
            s=json.loads((output/'summary.json').read_text());assert s['status']=='COMPLETED_PROVISIONAL'
            assert s['reload_error']==0 and s['path_violations']==0
            assert s['reconstruction_error']<1e-10 and s['explicit_component_sum_error']<1e-10
            for name,digest in json.loads((output/'output_hashes.json').read_text()).items():assert sha(output/name)==digest
    result=dict(stage=stage,job=int(job),status='PASS',tasks=expected,seconds=d['seconds'],mode=d['mode'],by_host={h:sum(x['host']==h for x in done) for h in sorted({x['host'] for x in done})},result_sha256=sha(path),failed_tasks=[])
    write(run/f'audit/{stage}_gate.json',result)
    print(json.dumps(result),flush=True)
    if stage=='training':
        index={}
        for item in done:
            output=run/'results/training'/str(job)/f"task-{item['id']}"/'output'
            s=json.loads((output/'shard_summary.json').read_text())
            index.setdefault(f"final/{s['seed']}/{s['variant']}",[]).append(dict(path=str(output),tree_start=s['tree_start'],sha256=s['model_sha256']))
        assert len(index)==6
        for items in index.values():assert sorted(x['tree_start'] for x in items)==list(range(0,32,4))
        dest=run/'audit/shard_index.json';write(dest,index)
        manifest(run,'assembly',[task(run,int(k.split('/')[1]),k.split('/')[2],'assemble','assembled',['--index',str(dest)]) for k in sorted(index)])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--repo',type=Path);p.add_argument('--stage',choices=['freeze','pilot','training','assembly'],required=True);p.add_argument('--job');a=p.parse_args()
    freeze(a.run,a.repo) if a.stage=='freeze' else gate(a.run,a.stage,a.job)
