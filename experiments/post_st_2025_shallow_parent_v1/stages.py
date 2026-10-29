from pathlib import Path
import argparse,json,shutil,subprocess,tarfile
from experiments.post_st_prognosis_v1.common import sha,write


def freeze(run,repo):
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True)
    commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    p=run/'input/folds/2025/plan.json';plan=json.loads(p.read_text());plan['code_commit']=commit
    p.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    shutil.copy2(run/'audit/protocol.json',run/'input/protocol.json')
    with (run/'audit/code.tar').open('xb') as f:subprocess.run(['git','-C',str(repo),'archive',commit],stdout=f,check=True)
    with tarfile.open(run/'audit/code.tar') as t:t.extractall(run/'input/code',filter='data')
    protocol=json.loads((run/'audit/protocol.json').read_text())
    for stage in ['pilot','formal']:
        tasks=[dict(kind='cpu',argv=['python3',str(run/'input/code/experiments/post_st_2025_shallow_parent_v1/launch.py'),
            '--model',model,'--seed',str(seed),*(['--pilot'] if stage=='pilot' else [])],
            cwd=str(run/'input/code'),timeout=3600,
            storage=dict(source=str(run/'input'),mode='nfs',results=str(run/'results'/stage)))
            for model in protocol['models'] for seed in ([42] if stage=='pilot' else protocol['seeds'])]
        with (run/f'{stage}.jsonl').open('x') as f:f.write(''.join(json.dumps(t)+'\n' for t in tasks))
        with (run/f'{stage}.py').open('x') as f:f.write('# Run with hetero-run; independent tasks are in the sidecar.\n')
        write(run/f'{stage}.py.hetero.json',dict(independent=True,manifest=f'{stage}.jsonl'))
    write(run/'audit/immutable_input_hashes.json',{str(p.relative_to(run/'input')):sha(p) for p in (run/'input').rglob('*') if p.is_file()})
    print(commit,flush=True)


def gate(run,stage,job,report=None):
    path=run/f'{stage}.jsonl.results-{job}.json'
    tasks=[json.loads(x) for x in (run/f'{stage}.jsonl').read_text().splitlines()]
    if report is not None:
        # hetero-run writes its report beside the copied auto-run manifest.
        # Verify command identity before retaining a local copy; never rerun a fit.
        incoming=json.loads(report.read_text())
        for i,task in enumerate(tasks):
            expected={k:task.get(k) for k in ('argv','cwd','env','kind','storage','min_gpu_mb')}
            assert json.loads(incoming['task_signatures'][str(i)])==expected
        if path.exists():assert sha(path)==sha(report)
        else:shutil.copy2(report,path)
    r=json.loads(path.read_text());items=r['completed']
    assert len(items)==len(tasks) and {x['id'] for x in items}==set(range(len(tasks))) and not r['pending']
    failures=[]
    for x in items:
        if x['returncode'] or not x['compute_completed'] or not x['publication_completed']:
            failures.append(dict(task=x['id'],host=x['host'],phase=x.get('phase'),returncode=x['returncode'],log=x['log']));continue
        out=run/'results'/stage/str(job)/f"task-{x['id']}"/'output'
        for n,h in json.loads((out/'hashes.json').read_text()).items():assert sha(out/n)==h
        s=json.loads((out/'summary.json').read_text());assert s['status']=='COMPLETE'
        if (out/'shrunk').exists():
            child=json.loads((out/'shrunk/summary.json').read_text())
            assert child['reload_error']==0 and child['complete_component_sum_error']<1e-10
            assert child['max_path_variables']<=2
        else:assert s['reload_error']<1e-12
    result=dict(status='FAIL' if failures else 'PASS',stage=stage,job=int(job),tasks=len(tasks),
        failed_tasks=failures,seconds=r['seconds'],by_host={h:sum(x['host']==h for x in items) for h in sorted({x['host'] for x in items})},
        compute_and_publication_verified=not failures,report_sha256=sha(path))
    write(run/f'audit/{stage}_gate.json',result);print(json.dumps(result),flush=True)
    assert not failures,failures


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--repo',type=Path)
    p.add_argument('--stage',choices=['freeze','pilot','formal'],required=True);p.add_argument('--job');p.add_argument('--report',type=Path)
    a=p.parse_args();freeze(a.run,a.repo) if a.stage=='freeze' else gate(a.run,a.stage,a.job,a.report)
