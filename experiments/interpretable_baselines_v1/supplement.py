"""Add only the omitted current-method coverage simulation; retain original run."""
from pathlib import Path
import argparse,json,subprocess,tarfile,shutil,hashlib

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(run,repo,job=None):
    if job:
        p=run/f'coverage.jsonl.results-{job}.json';d=json.loads(p.read_text());items=d['completed']
        assert len(items)==35 and {x['id'] for x in items}==set(range(35))
        assert all(x['returncode']==0 and x['publication_completed'] and x['compute_completed'] for x in items)
        for x in items:
            out=run/'results/coverage'/str(job)/f"task-{x['id']}"/'output'
            s=json.loads((out/'summary.json').read_text());assert s['status']=='COMPLETE' and s['model']=='q2_coverage' and s['reconstruction_error']<1e-10
            for name,h in json.loads((out/'hashes.json').read_text()).items():assert sha(out/name)==h
        result=dict(stage='coverage',job=int(job),tasks=35,status='PASS',seconds=d['seconds'],by_host={h:sum(x['host']==h for x in items) for h in sorted({x['host'] for x in items})},failed_tasks=[],report_sha256=sha(p))
        with (run/'audit/coverage_gate.json').open('x') as f:json.dump(result,f,indent=2)
        print(json.dumps(result));return
    inp=run/'coverage_input';inp.mkdir();shutil.copy2(run/'input/plan.json',inp/'plan.json')
    code=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    with (run/'audit/coverage_code.tar').open('xb') as f:subprocess.run(['git','-C',str(repo),'archive',code],stdout=f,check=True)
    with tarfile.open(run/'audit/coverage_code.tar') as t:t.extractall(inp/'code',filter='data')
    assert sha(inp/'code/experiments/interpretable_baselines_v1/worker.py')==sha(repo/'experiments/interpretable_baselines_v1/worker.py')
    plan=json.loads((inp/'plan.json').read_text());tasks=[]
    for scenario in plan['simulation_scenarios']:
        for seed in plan['simulation_seeds']:
            tasks.append(dict(kind='cpu',argv=['python3',str(inp/'code/experiments/interpretable_baselines_v1/launch.py'),'--kind','simulation','--model','q2_coverage','--scenario',scenario,'--seed',str(seed)],cwd=str(inp/'code'),timeout=7200,storage=dict(source=str(inp),mode='nfs',results=str(run/'results/coverage'))))
    with (run/'coverage.jsonl').open('x') as f:f.write(''.join(json.dumps(t)+'\n' for t in tasks))
    (run/'coverage.py').write_text('# Independent coverage simulation audit.\n');(run/'coverage.py.hetero.json').write_text(json.dumps(dict(independent=True,manifest='coverage.jsonl')))
    with (run/'audit/coverage_freeze.json').open('x') as f:json.dump(dict(training_code=code,tasks=35,config_change='q2 path_budget2; tree_feature_count3/8 instead of all8',original_tasks_unchanged=True,post_initial_results_amendment=True,input_hashes={str(p.relative_to(inp)):sha(p) for p in inp.rglob('*') if p.is_file()}),f,indent=2)
    print(json.dumps(dict(code=code,tasks=35)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--repo',type=Path);p.add_argument('--job');a=p.parse_args();main(a.run,a.repo,a.job)
