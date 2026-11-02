"""Reproduce the finite experiment into a NEW private run, never old reports."""
from pathlib import Path
import argparse, io, json, os, shutil, subprocess, sys, tarfile, tempfile
from experiments.post_st_reporting_drift_v1.analyze import digest, write_json

PUBLIC=Path(__file__).resolve().parent
REPO=PUBLIC.parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path);a=p.parse_args()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO).strip(), 'Require committed code snapshot'
    loc=json.loads((PUBLIC/'run_location.json').read_text());source=Path(loc['source_run'])
    if a.run_dir:
        run=a.run_dir.resolve();run.mkdir(parents=True,exist_ok=False)
    else:run=Path(tempfile.mkdtemp(prefix='calendar_level_repeat_',dir=Path(loc['private_run']).parent))
    os.environ['CALENDAR_LEVEL_RUN']=str(run);os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
    def call(module,log):
        with (run/log).open('x') as f:subprocess.run([sys.executable,'-m',f'experiments.post_st_calendar_level_v1.{module}'],cwd=REPO,stdout=f,stderr=subprocess.STDOUT,check=True)
    call('prepare','prepare.log')
    dest=run/'validation_input';dest.mkdir();shutil.copytree(source/'input/folds',dest/'folds')
    for path in (run/'prepared').glob('*merged_PRIVATE.json'):shutil.copy2(path,dest/path.name)
    shutil.copy2(run/'prepared/specifications.json',dest/'specifications.json')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    files=[f for f in subprocess.check_output(['git','ls-files'],cwd=REPO,text=True).splitlines() if f.endswith('.py')]
    with tarfile.open(fileobj=io.BytesIO(subprocess.check_output(['git','archive',commit,'--',*files],cwd=REPO))) as archive:archive.extractall(dest/'code',filter='data')
    write_json(run/'validation_input_hashes.json',{str(f.relative_to(dest)):digest(f) for f in dest.rglob('*') if f.is_file()})
    write_json(run/'code_snapshot.json',dict(commit=commit))
    tasks=[dict(kind='cpu',argv=['python3',str(dest/'code/experiments/post_st_calendar_level_v1/launch.py'),'--year',str(y)],cwd=str(dest/'code'),timeout=3600,storage=dict(source=str(dest),mode='nfs',results=str(run/'validation_results'))) for y in [2023,2024,2025]]
    (run/'validation.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in tasks));(run/'validation.py').write_text('# Independent full-curve checks; no fits.\n')
    write_json(run/'validation.py.hetero.json',dict(independent=True,manifest='validation.jsonl'))
    with (run/'launcher_validation.log').open('x') as log:subprocess.run(['/home/oliver/.local/bin/hetero-run',str(run/'validation.py')],stdout=log,stderr=subprocess.STDOUT,check=True)
    call('analyze','evaluation.log');call('collect','collection.log');call('report','report.log')
    print(f'Complete: {run}; public-safe aggregates in deliverables/; no existing report overwritten')


if __name__=='__main__':main()
