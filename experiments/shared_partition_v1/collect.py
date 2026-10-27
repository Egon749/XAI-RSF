"""Verify pool publications and construct assembly tasks without any new fitting."""
from pathlib import Path
import argparse,json,hashlib


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(run,job,recovery_job=None):
    run=Path(run);report=run/f'training.jsonl.results-{job}.json';d=json.loads(report.read_text())
    assert len(d['completed'])==156 and {x['id'] for x in d['completed']}==set(range(156))
    failed=[x for x in d['completed'] if x['returncode']!=0]
    success=[x for x in d['completed'] if x['returncode']==0]
    assert all(x['compute_completed'] and x['publication_completed'] for x in success)
    folders=[run/'results/training'/str(job)/f"task-{x['id']}"/'output' for x in success]
    if failed:
        assert recovery_job is not None
        plan=json.loads((run/'audit/recovery_plan.json').read_text())
        assert {x['id'] for x in failed}=={x['original_id'] for x in plan['tasks']}
        recovery=json.loads((run/f'recovery.jsonl.results-{recovery_job}.json').read_text())
        assert len(recovery['completed'])==len(failed)
        assert {x['id'] for x in recovery['completed']}==set(range(len(failed)))
        assert all(x['compute_completed'] and x['publication_completed'] and x['returncode']==0 for x in recovery['completed'])
        folders += [run/'results/recovery'/str(recovery_job)/f"task-{x['id']}"/'output' for x in recovery['completed']]
    index={};full=[]
    for p in folders:
        if (p/'shard_summary.json').exists():
            s=json.loads((p/'shard_summary.json').read_text());assert s['status']=='SHARD_COMPLETE'
            digest=sha(p/'shard_PRIVATE.json');assert digest==s['model_sha256']
            index.setdefault(f"{s['fold']}/{s['seed']}/{s['variant']}",[]).append(dict(path=str(p),tree_start=s['tree_start'],sha256=digest))
        else:
            s=json.loads((p/'summary.json').read_text());assert s['status']=='COMPLETED_PROVISIONAL' and not s['pilot'];full.append(s)
    assert len(index)==18 and len(full)==12
    for key,items in index.items():assert sorted(i['tree_start'] for i in items)==list(range(0,32,4))
    dest=run/'audit/shard_index.json'
    with dest.open('x') as f:json.dump(index,f,indent=2)
    tasks=[]
    for key in sorted(index):
        fold,seed,variant=key.split('/')
        tasks.append(dict(kind='cpu',argv=['python3',str(run/'input/code/experiments/shared_partition_v1/launch.py'),'--fold',fold,'--seed',seed,'--variant',variant,'--mode','assemble','--index',str(dest)],cwd=str(run/'input/code'),timeout=1800,storage=dict(source=str(run/'input'),mode='auto',results=str(run/'results/assembled'))))
    with (run/'assembly.jsonl').open('x') as f:f.write(''.join(json.dumps(t)+'\n' for t in tasks))
    with (run/'audit/training_publication_verification.json').open('x') as f:json.dump(dict(status='PASS_AFTER_SERIALIZATION_RECOVERY' if failed else 'PASS',job=job,tasks=156,failed_original_tasks=[x['id'] for x in failed],recovery_job=recovery_job,recovered_tasks=len(failed),full_budget_fits=12,shards=144,shard_index_sha256=sha(dest)),f,indent=2)
    print(json.dumps(dict(full_fits=12,complete_sharded_forests=18,assembly_tasks=len(tasks))))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--job',required=True);p.add_argument('--recovery-job');a=p.parse_args();main(a.run,a.job,a.recovery_job)
