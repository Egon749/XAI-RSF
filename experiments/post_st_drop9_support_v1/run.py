"""Prepare immutable inputs, freeze code, and monitor paired two-node jobs."""
from dataclasses import asdict
from pathlib import Path
import argparse,datetime,io,json,shutil,subprocess,tarfile
import numpy as np
import pandas as pd
from crforest import CompetingRiskForest
from experiments.post_st_paper_features_v1.common import references,history
from experiments.post_st_drop9_transport_v1.common import horizon_support
from .common import *

def prepare():
    from .prepare import prepare as action
    action()

def freeze():
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO).strip()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    tracked=subprocess.check_output(['git','ls-files'],cwd=REPO,text=True).splitlines();files=[p for p in tracked if p.endswith('.py')]
    with tarfile.open(fileobj=io.BytesIO(subprocess.check_output(['git','archive',commit,'--',*files],cwd=REPO))) as t:t.extractall(RUN/'input/code',filter='data')
    write(RUN/'audit/code_snapshot.json',dict(commit=commit,protocol_sha256=sha(RUN/'input/protocol.json')))
    write(RUN/'audit/immutable_input_hashes.json',{str(p.relative_to(RUN/'input')):sha(p) for p in (RUN/'input').rglob('*') if p.is_file()})

def freeze_analysis():
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO).strip()
    dest=RUN/'analysis_input';dest.mkdir(exist_ok=False)
    shutil.copytree(RUN/'input/folds',dest/'folds');shutil.copy2(RUN/'audit/registry_formal.json',dest/'registry_formal.json')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    files=[p for p in subprocess.check_output(['git','ls-files'],cwd=REPO,text=True).splitlines() if p.endswith('.py')]
    with tarfile.open(fileobj=io.BytesIO(subprocess.check_output(['git','archive',commit,'--',*files],cwd=REPO))) as t:t.extractall(dest/'code',filter='data')
    write(RUN/'audit/analysis_code_snapshot.json',dict(commit=commit))
    write(RUN/'audit/immutable_analysis_hashes.json',{str(p.relative_to(dest)):sha(p) for p in dest.rglob('*') if p.is_file()})

def execute(stage):
    items=([dict(year=2024,seed=42,arm='drop9_support',trees=4)] if stage=='pilot'
           else [dict(year=y,seed=s,arm='drop9_support',trees=32) for y in YEARS for s in SEEDS])
    if stage=='formal':assert load(RUN/'audit/gate_pilot.json')['status']=='PASS'
    if stage=='ensemble':
        assert load(RUN/'audit/gate_formal.json')['status']=='PASS'
        items=[dict(year=y,arm=a,version=v) for y in YEARS for a in ARMS for v in ['raw','shrunk']]
    input_root=RUN/('analysis_input' if stage=='ensemble' else 'input')
    tasks=[]
    for r in items:
        args=['--ensemble','--version',r['version']] if stage=='ensemble' else ['--seed',str(r['seed']),'--trees',str(r['trees'])]
        tasks.append(dict(kind='cpu',argv=['python3',str(input_root/f'code/experiments/{NAME}/launch.py'),
            '--year',str(r['year']),'--arm',r['arm'],*args],cwd=str(input_root/'code'),timeout=3600,
            storage=dict(source=str(input_root),mode='nfs',results=str(RUN/f'results/{stage}'))))
    (RUN/f'{stage}.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in tasks))
    (RUN/f'{stage}.py').write_text('# Independent paired annual seed forests.\n')
    write(RUN/f'{stage}.py.hetero.json',dict(independent=True,manifest=f'{stage}.jsonl'))
    assert not (RUN/f'audit/launcher_{stage}.log').exists()
    receipt=None
    with (RUN/f'audit/launcher_{stage}.log').open('x') as log:
        p=subprocess.Popen(['/home/oliver/.local/bin/hetero-run',str(RUN/f'{stage}.py')],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        for line in p.stdout:
            log.write(line);log.flush();print(line,end='',flush=True)
            try:
                obj=json.loads(line)
                if isinstance(obj,dict) and 'job' in obj and 'result' in obj:receipt=obj
            except json.JSONDecodeError:pass
        status=p.wait()
    if status or receipt is None or not receipt['success']:raise RuntimeError('Retain failed compute/publication evidence; do not automatically refit')
    report=load(receipt['result']);assert report['expected']==len(items) and not report['pending']
    records=[]
    for task in report['completed']:
        assert task['returncode']==0 and task['compute_completed'] and task['publication_completed']
        index=int(task['id']);folder=RUN/f'results/{stage}/{receipt["job"]}/task-{index}/output'
        for name,h in load(folder/'hashes.json').items():assert sha(folder/name)==h
        result=load(folder/'summary.json');assert result['status']=='COMPLETE'
        records.append(dict(**items[index],path=str(folder),host=result['host'],job=receipt['job'],task=index))
    assert len(records)==len(items)
    write(RUN/f'audit/registry_{stage}.json',records)
    write(RUN/f'audit/gate_{stage}.json',dict(status='PASS',job=receipt['job'],tasks=len(items),seconds=receipt['execution_seconds'],report=receipt['result'],report_sha256=sha(Path(receipt['result']))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','freeze','pilot','formal','freeze_analysis','ensemble']);a=p.parse_args()
    {'prepare':prepare,'freeze':freeze,'pilot':lambda:execute('pilot'),'formal':lambda:execute('formal'),
        'freeze_analysis':freeze_analysis,'ensemble':lambda:execute('ensemble')}[a.stage]()
