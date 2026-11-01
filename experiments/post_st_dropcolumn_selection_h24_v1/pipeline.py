"""Finite dependent stages; only independent fit bundles use the node pool."""
from pathlib import Path
import argparse,io,json,os,subprocess,tarfile,time
import numpy as np
import pandas as pd
from .initial import ExistingStore,NeedFits
from .fitting import *

NAME='post_st_dropcolumn_selection_h24_v1'

def snapshot(run,repo,suffix=''):
    if subprocess.check_output(['git','status','--porcelain'],cwd=repo).strip():raise ValueError('Commit code before snapshot')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    tracked=subprocess.check_output(['git','ls-files'],cwd=repo,text=True).splitlines();paths=[p for p in tracked if p.endswith('.py')]
    code=run/'input'/('code'+suffix)
    with tarfile.open(fileobj=io.BytesIO(subprocess.check_output(['git','archive',commit,'--',*paths],cwd=repo))) as f:f.extractall(code,filter='data')
    protocol=load(repo/f'experiments/{NAME}/frozen_protocol.json')
    if suffix:assert protocol==load(run/'input/protocol.json')
    else:write(run/'input/protocol.json',protocol)
    write(run/f'audit/code_snapshot{suffix}.json',dict(commit=commit,code_dir=code.name,protocol_sha256=sha(run/'input/protocol.json'),code_hashes={str(p.relative_to(code)):sha(p) for p in code.rglob('*.py')}))

def pool(run,stage,items,code_record='code_snapshot.json'):
    root=run/'input'/stage;root.mkdir();meta=load(run/'audit'/code_record);parent=load(run/'input/protocol.json')['parent_private_run'];code=run/'input'/meta.get('code_dir','code')
    write(root/'plan.json',dict(parent=parent,code_commit=meta['commit'],items=items))
    # Stage-specific plan and shared immutable code snapshot; both are NFS paths.
    tasks=[dict(kind='cpu',argv=['python3',str(code/f'experiments/{NAME}/launch.py'),'--index',str(i)],cwd=str(code),timeout=21600,
           storage=dict(source=str(root),mode='nfs',results=str(run/'results'/stage))) for i in range(len(items))]
    with (run/f'{stage}.jsonl').open('x') as f:f.write(''.join(json.dumps(x)+'\n' for x in tasks))
    (run/f'{stage}.py').write_text('# Independent bundle/selection commands are in the sidecar.\n');write(run/f'{stage}.py.hetero.json',dict(independent=True,manifest=f'{stage}.jsonl'))
    receipt=None
    with (run/f'audit/launcher_{stage}.log').open('x') as log:
        p=subprocess.Popen(['/home/oliver/.local/bin/hetero-run',str(run/f'{stage}.py')],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        for line in p.stdout:
            log.write(line);log.flush();print(line,end='',flush=True)
            try:
                value=json.loads(line)
                if isinstance(value,dict) and 'job' in value and 'result' in value:receipt=value
            except json.JSONDecodeError:pass
        status=p.wait()
    if status or receipt is None or not receipt['success']:raise RuntimeError(f'{stage}: retain failed compute/publication evidence; do not automatically retrain')
    report=load(Path(receipt['result']));assert report['expected']==len(items) and not report['pending']
    records=[]
    for state in report['completed']:
        assert state['returncode']==0 and state['compute_completed'] and state['publication_completed']
        i=int(state['id']);out=run/f'results/{stage}/{receipt["job"]}/task-{i}/output'
        for n,h in load(out/'hashes.json').items():assert sha(out/n)==h
        s=load(out/'summary.json');records.append(dict(**items[i],output=str(out),new_forests=s['new_forests'],host=s['host']))
    assert len(records)==len(items);write(run/f'audit/registry_{stage}.json',records)
    write(run/f'audit/gate_{stage}.json',dict(status='PASS',stage=stage,job=receipt['job'],tasks=len(items),new_forests=sum(r['new_forests'] for r in records),seconds=receipt['execution_seconds'],report=receipt['result'],report_sha256=sha(Path(receipt['result']))))
    return records

class ExtendedStore(ExistingStore):
    def __init__(self,parent,run):
        super().__init__(parent);self.extra={}
        for p in sorted((run/'audit').glob('registry_*.json')):
            for r in load(p):
                if r['kind']=='bundle':self.extra[(tuple(r['keep']),r['year'])]=Path(r['output'])/'bundle'
    def ensure(self,keeps,years):
        missing=[dict(keep=list(k),year=y) for k in sorted(set(map(tuple,keeps))) for y in years if (y,self.sid.get(k)) not in self.registry and (k,y) not in self.extra]
        if missing:raise NeedFits(missing)
    def get(self,keep,year):
        key=(tuple(keep),year)
        if key in self.extra:
            if key not in self.cache:self.cache[key]=read_bundle(self.extra[key])
            return self.cache[key]
        return super().get(keep,year)

def advance(run):
    protocol=load(run/'input/protocol.json');parent=Path(protocol['parent_private_run']);store=ExtendedStore(parent,run)
    outcomes={};requests={};allrows=[]
    for outer in [2023,2024,2025]:
        support,_=historical_support(parent,outer);g=gate24(support);ev=Evaluator(parent,outer,store,support)
        groups=[r['members'] for r in load(parent/'input/group_specs.json') if r['outer']==outer]
        try:
            keep,path,scans=select(ev,initial_units(parent,outer),g,groups)
            outcomes[str(outer)]=dict(gate=g,keep=list(keep),path=path,status='SELECTED' if len(keep)<83 else 'RETAINED83')
            for r in scans:allrows.append(dict(outer=outer,**r))
        except NeedFits as missing:
            for r in missing.items:requests[(r['year'],tuple(r['keep']))]=dict(kind='bundle',**r)
            outcomes[str(outer)]=dict(gate=g,status='NEEDS_MATCHED_FITS',requested_bundles=len(missing.items))
    index=len(list((run/'audit').glob('advance_*.json')));write(run/f'audit/advance_{index:02d}.json',outcomes)
    if not requests:
        write(run/'audit/selection_decisions.json',outcomes)
        rows=pd.DataFrame(allrows)
        for k in ['keep','failed_guards']:rows[k]=rows[k].apply(json.dumps)
        rows.to_csv(run/'reports/SELECTION_CONTRASTS.csv',index=False)
        print(json.dumps(dict(selection_complete=True,features={y:len(x['keep']) for y,x in outcomes.items()})),flush=True)
        return True,[]
    return False,list(requests.values())

def main(run,action):
    repo=Path(__file__).resolve().parents[2]
    if action=='freeze':snapshot(run,repo);return
    if action=='pilot':
        x=load(run/'audit/initial_selection.json');items=[dict(kind='bundle',**r) for r in x['2024']['requests'][:2]];pool(run,'pilot',items);return
    if action=='main':
        for step in range(26):
            complete,items=advance(run)
            if complete:return
            pool(run,f'main_{step:02d}',items)
        raise RuntimeError('Frozen maximum recursion/repair stage bound exceeded')
    if action=='bootstrap':
        assert (run/'audit/selection_decisions.json').exists()
        items=[dict(kind='bootstrap',outer=y,repeat=r) for y in [2023,2024,2025] for r in range(50)]
        pool(run,'bootstrap',items)
    if action=='final':
        decisions=load(run/'audit/selection_decisions.json');snapshot(run,repo,'_final')
        items=[dict(kind='bundle',year=int(y),keep=d['keep'],outer_final=True) for y,d in decisions.items() if len(d['keep'])<83]
        if items:pool(run,'final',items,'code_snapshot_final.json')
        else:write(run/'audit/registry_final.json',[])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--action',choices=['freeze','pilot','main','bootstrap','final'],required=True);a=p.parse_args();main(a.run,a.action)
