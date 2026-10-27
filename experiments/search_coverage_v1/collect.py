"""Check every shard publication and build deterministic assembly commands."""
from pathlib import Path
import argparse,json,hashlib

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(run,job):
    run=Path(run);d=json.loads((run/f'training.jsonl.results-{job}.json').read_text())
    done=d['completed'];assert len(done)==240 and {x['id'] for x in done}==set(range(240))
    assert all(x['returncode']==0 and x['compute_completed'] and x['publication_completed'] for x in done)
    index={}
    for x in done:
        p=run/'results/training'/str(job)/f"task-{x['id']}"/'output'
        s=json.loads((p/'shard_summary.json').read_text());assert sha(p/'shard_PRIVATE.json')==s['model_sha256']
        index.setdefault(f"{s['fold']}/{s['seed']}/{s['variant']}",[]).append(dict(path=str(p),tree_start=s['tree_start'],sha256=s['model_sha256']))
    assert len(index)==30
    for items in index.values():assert sorted(i['tree_start'] for i in items)==list(range(0,32,4))
    dest=run/'audit/shard_index.json'
    with dest.open('x') as f:json.dump(index,f,indent=2)
    tasks=[]
    for key in sorted(index):
        fold,seed,v=key.split('/')
        tasks.append(dict(kind='cpu',argv=['python3',str(run/'input/code/experiments/search_coverage_v1/launch.py'),'--fold',fold,'--seed',seed,'--variant',v,'--mode','assemble','--index',str(dest)],cwd=str(run/'input/code'),timeout=7200,storage=dict(source=str(run/'input'),mode='auto',results=str(run/'results/assembled'))))
    with (run/'assembly.jsonl').open('x') as f:f.write(''.join(json.dumps(t)+'\n' for t in tasks))
    with (run/'assembly.py').open('x') as f:f.write('# Independent deterministic assembly, no training.\n')
    with (run/'assembly.py.hetero.json').open('x') as f:json.dump(dict(independent=True,manifest='assembly.jsonl'),f)
    print(json.dumps(dict(shards=len(done),assembly_tasks=len(tasks))))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--job',required=True);a=p.parse_args();main(a.run,a.job)
