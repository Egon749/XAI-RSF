"""Dependency-aware A-then-B orchestration; never fits a forest."""
from pathlib import Path
import argparse,io,json,os,shutil,subprocess,sys,tarfile,tempfile
import numpy as np
import pandas as pd
from experiments.post_st_reporting_drift_v1.analyze import write_json,digest
from experiments.post_st_reporting_drift_v1.metrics import stratified_metrics
from experiments.post_st_feature_permutation_v1.common import evaluation_context

PUBLIC=Path(__file__).resolve().parent;REPO=PUBLIC.parents[1]


def execute(run,part):
    root=run/('input' if part=='A' else 'inputB')
    tasks=[dict(kind='cpu',argv=['python3',str(root/'code/experiments/post_st_risk_index_v2/launch.py'),'--part',part,'--year',str(y)],cwd=str(root/'code'),timeout=3600,storage=dict(source=str(root),mode='nfs',results=str(run/f'results_{part}'))) for y in [2023,2024,2025]]
    (run/f'{part}.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in tasks));(run/f'{part}.py').write_text('# Three independent annual checks,zero fits.\n')
    write_json(run/f'{part}.py.hetero.json',dict(independent=True,manifest=f'{part}.jsonl'))
    with (run/f'launcher_{part}.log').open('x') as log:
        proc=subprocess.Popen(['/home/oliver/.local/bin/hetero-run',str(run/f'{part}.py')],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        receipt=None
        for line in proc.stdout:
            log.write(line);log.flush();print(line,end='',flush=True)
            if line.startswith('{'):
                obj=json.loads(line)
                if 'result' in obj:receipt=obj
        if proc.wait()!=0 or receipt is None or not receipt['success']:raise RuntimeError('STOP: task failed; preserve logs')
    result=json.loads(Path(receipt['result']).read_text());assert result['expected']==3 and not result['pending'] and len(result['completed'])==3
    records=[]
    for task in result['completed']:
        assert task['returncode']==0 and task['compute_completed'] and task['publication_completed']
        out=run/f'results_{part}/{receipt["job"]}/task-{task["id"]}/output'
        for name,h in json.loads((out/'hashes.json').read_text()).items():assert digest(out/name)==h
        summary=json.loads((out/'summary.json').read_text());assert summary['status']=='COMPLETE' and summary['forest_fits']==0
        records.append(dict(year=summary['year'],part=part,path=str(out),job=receipt['job'],task=task['id'],host=summary['host'],seconds=summary['seconds'],peak_rss_kib=summary['peak_rss_kib']))
    assert sorted(r['year'] for r in records)==[2023,2024,2025]
    write_json(run/f'registry_{part}.json',records)
    return records


def finish_a(run,records):
    within=[]
    for h in [12,24]:
        for c in [1,2]:
            ys=[2023,2024,2025] if h==12 else [2023,2024]
            labels=[];cifs=[];scores=[];year=[];ids=[]
            for y in ys:
                r=next(r for r in records if r['year']==y)
                with np.load(run/f'input/{y}/eval_PRIVATE.npz') as z:lab=dict(z)
                with np.load(Path(r['path'])/'endpoints_PRIVATE.npz') as z:
                    cifs.append(z[f'cif_{h}'][:,c-1]);scores.append(z[f'index_{h}'][:,c-1])
                labels.append(evaluation_context(lab,h,c)[0]);year.append(np.full(len(lab['X']),y));ids.append(lab['sample_ids'])
            yy,yr,ii=map(np.concatenate,(labels,year,ids));a=stratified_metrics(yy,np.concatenate(cifs),yr,ii);b=stratified_metrics(yy,np.concatenate(scores),yr,ii)
            for name in ['within_year_auc','equal_year_auc']:np.testing.assert_allclose(a[name],b[name],rtol=0,atol=0,equal_nan=True)
            within.append(dict(months=h,cause=c,reference_within_year_auc=a['within_year_auc'],index_within_year_auc=b['within_year_auc'],exact_equal=True))
    pd.DataFrame(within).to_csv(run/'WITHIN_YEAR_IDENTITY.csv',index=False)
    write_json(run/'A_GATE.json',dict(status='PASS',years=[2023,2024,2025],within_year_identity=True,reference_gate='PASS',forest_fits=0))


def stage(run):
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    files=[n for n in subprocess.check_output(['git','ls-files'],cwd=REPO,text=True).splitlines() if n.endswith('.py')]
    with tarfile.open(fileobj=io.BytesIO(subprocess.check_output(['git','archive',commit,'--',*files],cwd=REPO))) as archive:archive.extractall(run/'input/code',filter='data')
    write_json(run/'code_snapshot.json',dict(commit=commit))
    write_json(run/'immutable_A_input_hashes.json',{str(p.relative_to(run/'input')):digest(p) for p in (run/'input').rglob('*') if p.is_file()})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepared-run',type=Path);a=parser.parse_args()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO).strip(),'Commit code before execution'
    loc=json.loads((PUBLIC/'run_location.json').read_text())
    run=a.prepared_run if a.prepared_run else Path(tempfile.mkdtemp(prefix='risk_index_v2_repeat_',dir=Path(loc['private_run']).parent))
    os.environ['RISK_INDEX_RUN']=str(run)
    if not a.prepared_run:
        with (run/'prepare.log').open('x') as log:subprocess.run([sys.executable,'-m','experiments.post_st_risk_index_v2.prepare'],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
    stage(run);records=execute(run,'A');finish_a(run,records)
    # Input A remains immutable; B receives a fresh copy only after global A PASS.
    shutil.copytree(run/'input',run/'inputB')
    for y in [2023,2024,2025]:(run/f'inputB/{y}/eval_PRIVATE.npz').unlink()  # B cannot read evaluation data.
    for r in records:
        dest=run/f'inputB/A/{r["year"]}';dest.mkdir(parents=True)
        shutil.copy2(Path(r['path'])/'denominators_PRIVATE.npz',dest/'denominators_PRIVATE.npz')
    shutil.copy2(run/'A_GATE.json',run/'inputB/A_GATE.json')
    write_json(run/'immutable_B_input_hashes.json',{str(p.relative_to(run/'inputB')):digest(p) for p in (run/'inputB').rglob('*') if p.is_file()})
    execute(run,'B');print(f'Both phases complete: {run}')


if __name__=='__main__':main()
