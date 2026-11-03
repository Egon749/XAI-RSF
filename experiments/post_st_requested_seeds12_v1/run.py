import argparse,io,subprocess,tarfile,shutil
from .common import *
def freeze():
    r=run_path();assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO).strip()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    files=[p for p in subprocess.check_output(['git','-c','core.quotePath=false','ls-files'],cwd=REPO,text=True).splitlines()if p.endswith('.py')]
    with tarfile.open(fileobj=io.BytesIO(subprocess.check_output(['git','archive',commit,'--',*files],cwd=REPO)))as t:t.extractall(r/'input/code',filter='data')
    write(r/'audit/code_snapshot.json',dict(commit=commit,protocol_sha256=sha(r/'input/protocol.json')))
    write(r/'audit/input_hashes.json',{str(p.relative_to(r/'input')):sha(p)for p in(r/'input').rglob('*')if p.is_file()})
def stage_ensemble():
    r=run_path();assert load(r/'audit/gate_formal.json')['status']=='PASS';root=r/'ensemble_input';root.mkdir()
    records=load(r/'input/reused_registry.json')+load(r/'audit/registry_pilot.json')+load(r/'audit/registry_formal.json')
    assert len(records)==36 and len({(x['year'],x['seed'])for x in records})==36
    shutil.copytree(r/'input/code',root/'code');shutil.copytree(r/'input/folds',root/'folds')
    for rec in records:
        p=Path(rec['path'])
        for n,h in load(p/'hashes.json').items():assert sha(p/n)==h
        for v in ['raw','shrunk']:
            dest=root/f'members/{rec["year"]}/{rec["seed"]}/{v}';dest.mkdir(parents=True)
            for n in ['forest_PRIVATE.json','predictions_PRIVATE.npz']:
                shutil.copy2(p/v/n,dest/n);assert sha(dest/n)==sha(p/v/n)
    write(r/'audit/ensemble_input_hashes.json',{str(p.relative_to(root)):sha(p)for p in root.rglob('*')if p.is_file()})
def execute(stage):
    r=run_path();root=r/('ensemble_input'if stage=='ensemble'else'input');items=task_rows(stage)
    if stage=='formal':
        assert load(r/'audit/gate_pilot.json')['status']=='PASS'
        assert load(Path(load(r/'audit/registry_pilot.json')[0]['path'])/'summary.json')['peak_rss_MiB']<=900
    if stage=='ensemble':assert load(r/'audit/gate_formal.json')['status']=='PASS'
    mode='ensemble'if stage=='ensemble'else'fit';manifest=[]
    for row in items:
        argv=['python3',str(root/f'code/experiments/{NAME}/launch.py'),'--mode',mode]
        for k,v in row.items():argv+=['--'+k,str(v)]
        manifest.append(dict(kind='cpu',argv=argv,cwd=str(root/'code'),timeout=3600,storage=dict(source=str(root),mode='nfs',results=str(r/f'results/{stage}'))))
    (r/f'{stage}.jsonl').write_text(''.join(json.dumps(z)+'\n'for z in manifest));(r/f'{stage}.py').write_text('# Independent fixed seed tasks.\n')
    write(r/f'{stage}.py.hetero.json',dict(independent=True,manifest=f'{stage}.jsonl'));receipt=None
    with(r/f'audit/launcher_{stage}.log').open('x')as log:
        p=subprocess.Popen(['/home/oliver/.local/bin/hetero-run',str(r/f'{stage}.py')],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        for line in p.stdout:
            log.write(line);log.flush();print(line,end='',flush=True)
            try:
                z=json.loads(line)
                if isinstance(z,dict)and'job'in z and'result'in z:receipt=z
            except json.JSONDecodeError:pass
        rc=p.wait()
    if rc or not receipt or not receipt['success']:raise RuntimeError('Preserve failure; no automatic retraining')
    report=load(receipt['result']);assert report['expected']==len(items)and not report['pending'];records=[]
    for task in report['completed']:
        assert task['returncode']==0 and task['compute_completed']and task['publication_completed']
        i=int(task['id']);out=r/f'results/{stage}/{receipt["job"]}/task-{i}/output'
        for n,h in load(out/'hashes.json').items():assert sha(out/n)==h
        s=load(out/'summary.json');assert s['status']=='COMPLETE'and s['forest_fits']==int(mode=='fit')
        records.append(dict(**items[i],path=str(out),job=receipt['job'],task=i,host=s['host']))
    assert len(records)==len(items);write(r/f'audit/registry_{stage}.json',records)
    write(r/f'audit/gate_{stage}.json',dict(status='PASS',tasks=len(records),**receipt,report_sha256=sha(Path(receipt['result']))))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','pilot','formal','stage_ensemble','ensemble']);a=p.parse_args()
    if a.stage=='freeze':freeze()
    elif a.stage=='stage_ensemble':stage_ensemble()
    else:execute(a.stage)
