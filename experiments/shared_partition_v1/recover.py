"""Recover output-serialization failures only, with identical training code/data."""
from pathlib import Path
import argparse,copy,hashlib,json,subprocess,tarfile

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(run,repo,job):
    run,repo=Path(run),Path(repo)
    report=run/f'training.jsonl.results-{job}.json';pool=json.loads(report.read_text())
    assert len(pool['completed'])==156
    failed=sorted([x for x in pool['completed'] if x['returncode']!=0],key=lambda x:x['id'])
    assert failed,'no failures to recover'
    for x in failed:
        assert not x['compute_completed'] and x['compute_returncode']!=0
        assert 'Out of range float values are not JSON compliant: -inf' in x['tail']
    tasks=[json.loads(l) for l in (run/'training.jsonl').read_text().splitlines()]
    root=run/'recovery';root.mkdir()
    revision=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    with (root/'code.tar').open('xb') as f:subprocess.run(['git','-C',str(repo),'archive',revision],stdout=f,check=True)
    with tarfile.open(root/'code.tar') as t:t.extractall(root/'code',filter='data')
    identity={}
    for p in (run/'input/code/crforest').glob('*.py'):
        q=root/'code/crforest'/p.name;assert sha(p)==sha(q);identity['crforest/'+p.name]=sha(p)
    p=Path('experiments/shared_partition_v1/engine.py')
    assert sha(run/'input/code'/p)==sha(root/'code'/p)
    identity[str(p)]=sha(root/'code'/p)
    recovered=[];mapping=[]
    for x in failed:
        task=copy.deepcopy(tasks[x['id']]);argv=task['argv'];assert argv[argv.index('--mode')+1]=='fit'
        argv[1]=str(root/'code/experiments/shared_partition_v1/launch.py');task['cwd']=str(root/'code')
        task['storage']['results']=str(run/'results/recovery')
        mapping.append(dict(recovery_id=len(recovered),original_id=x['id'],fold=argv[argv.index('--fold')+1],seed=int(argv[argv.index('--seed')+1]),variant=argv[argv.index('--variant')+1]))
        recovered.append(task)
    with (run/'recovery.jsonl').open('x') as f:f.write(''.join(json.dumps(t)+'\n' for t in recovered))
    with (run/'audit/recovery_plan.json').open('x') as f:
        json.dump(dict(original_job=int(job),recovery_code=revision,reason='strict JSON could not serialize missing-only -inf diagnostic after fitting; fitted model not yet saved in original worker',training_modules_bitwise_identical=identity,config_and_inputs_unchanged=True,selection='all and only failed tasks; no metric-based selection',tasks=mapping),f,indent=2)
    print(json.dumps(dict(failed_tasks=len(failed),recovery_code=revision)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--repo',required=True);p.add_argument('--job',required=True)
    a=p.parse_args();main(a.run,a.repo,a.job)

