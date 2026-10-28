"""Extend the immutable eight-replicate experiment without repeating its fits."""
from pathlib import Path
import argparse,json,hashlib,shutil,sys,subprocess,tarfile
import numpy as np
import pandas as pd
from experiments.interpretable_baselines_v1.methods import resample_groups


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,data):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(data,f,indent=2,allow_nan=False)
def prepare(run,parent):
    inp=run/'input';inp.mkdir();sources={}
    parent_hashes=json.loads((parent/'audit/input_hashes.json').read_text())
    for name,h in parent_hashes.items():assert sha(parent/'input'/name)==h
    repo=Path(__file__).resolve().parents[2];old=parent/'input/code'
    for rel in ['experiments/interpretable_baselines_v1/methods.py',*map(str,[p.relative_to(repo) for p in (repo/'crforest').rglob('*.py')])]:
        assert sha(repo/rel)==sha(old/rel),(rel,'implementation changed')
    rel='experiments/interpretable_baselines_v1/worker.py'
    assert (repo/rel).read_text().split("    elif a.kind=='simulation':")[0]==(old/rel).read_text().split("    elif a.kind=='simulation':")[0]
    write(run/'audit/reuse_compatibility.json',dict(forest_and_methods_byte_identical=True,real_worker_prefix_identical=True,worker_only_change='simulation q2_coverage branch; not used in this extension'))
    def copy(src,dst):
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);assert sha(src)==sha(dst);sources[str(src)]=sha(src)
    for name in ['plan.json','final_train_PRIVATE.npz','final_eval_PRIVATE.npz','final_train_metadata_PRIVATE.parquet','train_weights_PRIVATE.npy','references_PRIVATE.npz','reference_q2_42_forest_PRIVATE.json','reference_unrestricted_42_forest_PRIVATE.json']:
        copy(parent/'input'/name,inp/name)
    for p in (parent/'input/legacy').glob('*.py'):copy(p,inp/'legacy'/p.name)
    for r in range(8):copy(parent/f'input/bootstrap_{r}_PRIVATE.npz',inp/f'bootstrap_{r}_PRIVATE.npz')
    sys.path.insert(0,str(inp/'legacy'));from common import endpoint_weights
    from endpoint_guard import source_consistent_endpoint_weights
    tr=dict(np.load(inp/'final_train_PRIVATE.npz'));meta=pd.read_parquet(inp/'final_train_metadata_PRIVATE.parquet')
    np.testing.assert_array_equal(tr['sample_ids'],meta.sample_id)
    assert (meta.outcome_available.dropna()<pd.Timestamp('2022-01-01')).all()
    for r in range(50):
        rows,groups=resample_groups(tr['groups'],8000+r)
        if r<8:
            z=np.load(inp/f'bootstrap_{r}_PRIVATE.npz');np.testing.assert_array_equal(rows,z['rows']);np.testing.assert_array_equal(groups,z['groups'])
        else:
            w,_=source_consistent_endpoint_weights(meta.iloc[rows].reset_index(drop=True),endpoint_weights)
            np.savez_compressed(inp/f'bootstrap_{r}_PRIVATE.npz',rows=rows,groups=groups,weights=w)
    reused=[]
    for p in sorted((parent/'results/formal/2530').glob('task-*/output/summary.json')):
        s=json.loads(p.read_text())
        if s['kind']!='bootstrap' or s['model'] not in ['q2','unrestricted']:continue
        assert s['status']=='COMPLETE' and s['seed']==42 and not s['pilot']
        for name,h in json.loads((p.parent/'hashes.json').read_text()).items():assert sha(p.parent/name)==h
        assert s['resample_sha256']==sha(inp/f"bootstrap_{s['replicate']}_PRIVATE.npz")
        reused.append(dict(model=s['model'],replicate=s['replicate'],output=str(p.parent),hashes=json.loads((p.parent/'hashes.json').read_text())))
    assert {(x['model'],x['replicate']) for x in reused}=={(m,r) for m in ['q2','unrestricted'] for r in range(8)}
    write(run/'audit/reused_outputs.json',reused);write(run/'audit/input_provenance.json',sources)
    plan=json.loads((inp/'plan.json').read_text());plan.update(bootstrap_replicates=50,bootstrap_models=['q2','unrestricted'],bootstrap_fits=100,new_fits=84,reused_fits=16,extension_parent=str(parent),primary_metric='normalized_rmse',interval='conditional_MC_delete_one_replicate_jackknife_t')
    (inp/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(run=str(run),new_fits=84,reused_fits=16,train_rows=len(meta),company_groups=meta.group.nunique())))
def freeze(run,repo):
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True)
    code=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    with (run/'audit/code.tar').open('xb') as f:subprocess.run(['git','-C',str(repo),'archive',code],stdout=f,check=True)
    with tarfile.open(run/'audit/code.tar') as f:f.extractall(run/'input/code',filter='data')
    tasks=[]
    for m in ['unrestricted','q2']:
        for r in range(8,50):
            tasks.append(dict(kind='cpu',argv=['python3',str(run/'input/code/experiments/interpretable_baselines_v1/launch.py'),'--kind','bootstrap','--model',m,'--seed','42','--replicate',str(r)],cwd=str(run/'input/code'),timeout=14400,storage=dict(source=str(run/'input'),mode='nfs',results=str(run/'results/formal'))))
    with (run/'formal.jsonl').open('x') as f:f.write(''.join(json.dumps(x)+'\n' for x in tasks))
    (run/'formal.py').write_text('# Independent manifest entrypoint for hetero-run.\n');write(run/'formal.py.hetero.json',dict(independent=True,manifest='formal.jsonl'))
    write(run/'audit/freeze.json',dict(code_commit=code,code_archive_sha256=sha(run/'audit/code.tar'),manifest_sha256=sha(run/'formal.jsonl'),input_hashes={str(p.relative_to(run/'input')):sha(p) for p in (run/'input').rglob('*') if p.is_file()}))
    print(json.dumps(dict(code_commit=code,tasks=len(tasks))))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--parent',type=Path);p.add_argument('--repo',type=Path);p.add_argument('--freeze',action='store_true');a=p.parse_args()
    freeze(a.run,a.repo) if a.freeze else prepare(a.run,a.parent)
