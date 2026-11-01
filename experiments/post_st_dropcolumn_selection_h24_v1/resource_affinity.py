"""Job-local tail balancing, with topology discovery and an append-only audit.

No new process, CPU ID constant, cgroup expansion, fit restart or pool change.
The controller acts only on this experiment's workers inside their allocation.
"""
from pathlib import Path
import argparse,datetime,json,os,socket

MODULE='experiments.post_st_dropcolumn_selection_h24_v1.worker'

def topology(cpu):
    root=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
    key=((root/'physical_package_id').read_text().strip(),(root/'core_id').read_text().strip())
    siblings=(root/'thread_siblings_list').read_text().strip()
    return key,(',' in siblings or '-' in siblings)

def inspect(job,run):
    rows=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            args=(proc/'cmdline').read_bytes().decode().split('\0')
            if MODULE not in args or f'job_{job}/' not in (proc/'cgroup').read_text():continue
            pid=int(proc.name);current=os.sched_getaffinity(pid)
            threads=list((proc/'task').iterdir())
            if len(current)!=1 or any(os.sched_getaffinity(int(t.name))!=current for t in threads):continue
            if any((t/'comm').read_text().strip() not in ('python','jemalloc_bg_thd') for t in threads):continue
            ppid=int(next(line.split()[1] for line in (proc/'status').read_text().splitlines() if line.startswith('PPid:')))
            assert f'job_{job}/' in Path(f'/proc/{ppid}/cgroup').read_text()
            allowed=os.sched_getaffinity(ppid);index=int(args[args.index('--index')+1]);cpu=next(iter(current))
            folders=list((Path.home()/'.cache/heteropool-work'/str(job)).glob(f'{index}-*/output/models'))
            completed=list(folders[0].glob('*/summary.json')) if len(folders)==1 else []
            cohorts={f.parent.name.split('_')[0] for f in completed}
            rows.append(dict(pid=pid,threads=[int(t.name) for t in threads],index=index,cpu=cpu,allowed=sorted(allowed),core=topology(cpu)[0],is_p=topology(cpu)[1],
                completed_bundles=len(completed),remaining_initial_estimate=max(0,107*len(cohorts)-len(completed))))
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
    return rows

def rebalance(job,run,apply=False,swap_smt=False):
    assert socket.gethostname() in ('oliverdesktop','lemon')
    assert (run/'bootstrap.jsonl').exists()
    assert f'job={job} ' in (run/'audit/launcher_bootstrap.log').read_text()
    rows=inspect(job,run);occupied={tuple(r['core']) for r in rows};representatives={}
    allowed=set.intersection(*(set(r['allowed']) for r in rows)) if rows else set()
    if swap_smt:
        assert socket.gethostname()=='lemon'
        groups={core:[r for r in rows if tuple(r['core'])==core] for core in occupied}
        shared=sorted([max(g,key=lambda r:r['remaining_initial_estimate']) for g in groups.values() if len(g)==2],key=lambda r:-r['remaining_initial_estimate'])
        exclusive=sorted([g[0] for g in groups.values() if len(g)==1],key=lambda r:r['remaining_initial_estimate'])
        swaps=[]
        for long,short in list(zip(shared,exclusive))[:2]:
            if short['remaining_initial_estimate']<=0 or long['remaining_initial_estimate']<1.35*short['remaining_initial_estimate']:continue
            assert {long['cpu'],short['cpu']}<=allowed
            if apply:
                for worker in [long,short]:
                    proc=Path(f'/proc/{worker["pid"]}')
                    assert MODULE in (proc/'cmdline').read_bytes().decode().split('\0') and f'job_{job}/' in (proc/'cgroup').read_text()
                    assert all(os.sched_getaffinity(tid)=={worker['cpu']} for tid in worker['threads'])
                for tid in short['threads']:os.sched_setaffinity(tid,{long['cpu']})
                for tid in long['threads']:os.sched_setaffinity(tid,{short['cpu']})
                assert all(os.sched_getaffinity(tid)=={long['cpu']} for tid in short['threads'])
                assert all(os.sched_getaffinity(tid)=={short['cpu']} for tid in long['threads'])
            swaps.append(dict(long_task=long['index'],short_task=short['index'],long_from=long['cpu'],long_to=short['cpu'],
                short_from=short['cpu'],short_to=long['cpu'],long_remaining_initial=long['remaining_initial_estimate'],short_remaining_initial=short['remaining_initial_estimate']))
        result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),host=socket.gethostname(),job=job,applied=apply,active_workers=len(rows),
            swaps=swaps,active_cpu_slots_unchanged=True,no_new_SMT_slots=True,
            note='One-off measured-workload tail balancing on lemon: longer remaining scans receive already occupied exclusive cores; no model or budget change, not a universal fastest claim.')
        if apply:
            with (run/'audit/affinity_adjustments.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
        print(json.dumps(result),flush=True);return
    for cpu in sorted(allowed):
        core,is_p=topology(cpu)
        if is_p:representatives.setdefault(core,cpu)
    free=[cpu for core,cpu in representatives.items() if core not in occupied]
    duplicated=[]
    for core in occupied:
        same=sorted([r for r in rows if tuple(r['core'])==core],key=lambda r:r['cpu'])
        if len(same)>1:duplicated.extend(same[1:])
    ecores=sorted([r for r in rows if not r['is_p']],key=lambda r:(-r['remaining_initial_estimate'],r['index']))
    moves=[]
    for worker,cpu in zip(duplicated+ecores,free):
        assert cpu in worker['allowed'] and topology(cpu)[0] not in occupied
        if apply:
            proc=Path(f'/proc/{worker["pid"]}')
            assert MODULE in (proc/'cmdline').read_bytes().decode().split('\0')
            assert f'job_{job}/' in (proc/'cgroup').read_text()
            assert os.sched_getaffinity(worker['pid'])=={worker['cpu']}
            for tid in worker['threads']:os.sched_setaffinity(tid,{cpu})
            assert all(os.sched_getaffinity(tid)=={cpu} for tid in worker['threads'])
        moves.append(dict(pid=worker['pid'],task=worker['index'],from_cpu=worker['cpu'],to_cpu=cpu,
            reason='remove_P_sibling_contention' if worker['is_p'] else 'move_remaining_work_from_E_to_free_P',
            completed_bundles=worker['completed_bundles'],remaining_initial_estimate=worker['remaining_initial_estimate']))
        occupied.add(topology(cpu)[0])
    result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),host=socket.gethostname(),job=job,applied=apply,active_workers=len(rows),
        discovered_manager_allowed_cpus=sorted(allowed),physical_only_destinations=True,moves=moves,
        note='Runtime manager allowed mask includes P siblings on this run; no Slurm setting changed. Original pool CPU labels are supplemented by this live-migration ledger.')
    if apply:
        with (run/'audit/affinity_adjustments.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',type=int,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--apply',action='store_true');p.add_argument('--swap-smt',action='store_true');a=p.parse_args()
    rebalance(a.job,a.run,a.apply,a.swap_smt)
