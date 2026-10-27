"""Aggregate measured resources; distinguish sampled high-water from final peak."""
from pathlib import Path
from collections import defaultdict
import argparse,json
import pandas as pd

def main(run):
    r=Path(run);records=[]
    poolfiles=list(r.glob('training.jsonl.results-*.json'));assert len(poolfiles)==1
    pool=json.loads(poolfiles[0].read_text());job=poolfiles[0].stem.rsplit('-',1)[1]
    for task in pool['completed']:
        p=r/'results/training'/job/f"task-{task['id']}"/'output/shard_summary.json'
        d=json.loads(p.read_text())
        records.append(dict(task_id=task['id'],fold=d['fold'],seed=d['seed'],variant=d['variant'],tree_start=d['tree_start'],tree_count=d['tree_count'],host=d['host'],fit_seconds=d['seconds'],fit_cpu_seconds=d['cpu_seconds'],task_seconds=task['seconds'],compute_completed=task['compute_completed'],publication_completed=task['publication_completed']))
    pd.DataFrame(records).to_csv(r/'reports/shard_resources.csv',index=False)
    byhost=defaultdict(lambda:dict(samples=0,observed_max_active=0,observed_process_hwm_MiB=0,all_sampled_workers_single_cpu=True))
    for line in (r/'audit/resource_samples_verified.jsonl').read_text().splitlines():
        d=json.loads(line);h=byhost[d['host']];h['samples']+=1;h['observed_max_active']=max(h['observed_max_active'],len(d['processes']))
        for p in d['processes']:
            h['observed_process_hwm_MiB']=max(h['observed_process_hwm_MiB'],p['hwm_kib']/1024)
            h['all_sampled_workers_single_cpu'] &= len(p['affinity'])==1
    result=dict(hosts=dict(byhost),interpretation='External 5-second sampling of kernel process high-water marks. Lower bounds on final task peaks, not exact per-fit peaks. Completed assembly/validation peak RSS is separately recorded in training_diagnostics.csv. Initial unverified sample file excluded.')
    with (r/'reports/resource_observations.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();main(a.run)
