"""Immutable analysis snapshot and publication-aware parallel aggregation."""
from pathlib import Path
import argparse,datetime,io,json,shutil,subprocess,tarfile
import pandas as pd
from .common import *
from experiments.post_st_structure_recovery_v1.common import SCENARIOS

def prepare(run):
    repo=Path(__file__).resolve().parents[2]
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo).strip()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    source=run/'analysis_input';code=source/'code';code.mkdir(parents=True,exist_ok=False)
    archive=subprocess.check_output(['git','archive',commit],cwd=repo)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tf:tf.extractall(code,filter='data')
    tasks=[]
    for scenario in list(SCENARIOS)+[None]:
        args=['python3',str(code/'experiments/post_st_level_coverage_v1/analysis_launch.py'),'--run',str(run)]
        if scenario:args+=['--scenario',scenario]
        tasks.append(dict(kind='cpu',argv=args,cwd=str(code),timeout=7200,
            storage=dict(source=str(source),mode='nfs',results=str(run/'results/analysis'))))
    with (run/'analysis.jsonl').open('x') as f:f.write(''.join(json.dumps(t)+'\n' for t in tasks))
    (run/'analysis.py').write_text('# Ten disjoint analysis tasks; no new fits.\n')
    write(run/'analysis.py.hetero.json',dict(independent=True,manifest='analysis.jsonl'))
    write(run/'audit/analysis_pool_freeze.json',dict(commit=commit,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),tasks=len(tasks),analysis_only=True,
        manifest_sha256=sha(run/'analysis.jsonl'),code_hashes={str(p.relative_to(code)):sha(p) for p in code.rglob('*.py')}))
    print(json.dumps(dict(tasks=len(tasks),commit=commit)))

def collect(run):
    folders=outputs(run,'analysis');assert len(folders)==10
    out=run/'reports';out.mkdir(exist_ok=True);private=run/'private';private.mkdir(exist_ok=True)
    for p in (folders[-1]/'reports').iterdir():shutil.copy2(p,out/p.name)
    shutil.copytree(folders[-1]/'private',private,dirs_exist_ok=False) if not private.exists() else shutil.copytree(folders[-1]/'private',private,dirs_exist_ok=True)
    for name in sorted({p.name for folder in folders[:9] for p in (folder/'reports').glob('*.csv*')}):
        frames=[]
        for folder in folders[:9]:
            path=folder/'reports'/name
            try:frames.append(pd.read_csv(path,dtype={'seed':str,'features':str}))
            except pd.errors.EmptyDataError:continue
        if frames:pd.concat(frames,ignore_index=True).to_csv(out/name,index=False,compression='infer')
    checks=[json.loads((folder/'reports/simulation_validation.json').read_text()) for folder in folders[:9]]
    assert all(c['status']=='PASS' for c in checks) and sum(c['datasets'] for c in checks)==450
    write(out/'simulation_validation.json',dict(status='PASS',datasets=450,new_fits=1350,reused_reference_fits=1350,data_regenerated_and_paired=True,
        maximum_reconstruction_error=max(c['maximum_reconstruction_error'] for c in checks),note=checks[0]['note']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);p.add_argument('--stage',choices=['prepare','collect'],required=True);a=p.parse_args()
    prepare(a.run) if a.stage=='prepare' else collect(a.run)
