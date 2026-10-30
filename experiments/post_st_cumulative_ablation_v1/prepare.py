from pathlib import Path
from dataclasses import asdict
import argparse,datetime,json,shutil,subprocess,tarfile
import numpy as np
import pandas as pd
from crforest import CompetingRiskForest
from experiments.post_st_ablation_v1.common import registry as prior_registry
from .common import *

def prepare(run):
    for n in ['audit','input','evaluation_input','reports','private','results']:(run/n).mkdir(exist_ok=False)
    repo=Path(__file__).resolve().parents[2];sources={}
    def source(p):p=Path(p);sources[str(p)]=sha(p);return p
    previous=json.loads(source(repo/'experiments/post_st_ablation_v1/execution_receipt.json').read_text())
    assert previous['status']=='COMPLETE_EXPLORATORY' and previous['new_formal_forest_fits']==18
    for n,h in json.loads(source(PREVIOUS/'audit/immutable_input_hashes.json').read_text()).items():assert sha(PREVIOUS/'input'/n)==h
    p=Path(__file__).with_name('frozen_protocol.json');shutil.copy2(p,run/'audit/protocol.json')
    ref=source(repo/'CURRENT_POST_ST_MODEL.json');inputrows=[]
    for y in YEARS:
        src=PREVIOUS/'input/folds'/str(y);dest=run/'input/folds'/str(y);dest.mkdir(parents=True)
        plan=json.loads(source(src/'plan.json').read_text())
        for n in ['train_PRIVATE.npz','eval_PRIVATE.npz']:
            f=source(src/n);assert sha(f)==plan['source_hashes'][n];shutil.copy2(f,dest/n)
            inputrows.append(dict(year=y,file=n,sha256=sha(f)))
        plan.pop('code_commit',None);write(dest/'plan.json',plan)
        tr=arrays(dest/'train_PRIVATE.npz');ev=arrays(dest/'eval_PRIVATE.npz');assert set(ev)=={'X','sample_ids'} and tr['X'].shape[1]==83
        for n in [f'{y}_test_PRIVATE.npz',f'{y}_fit_metadata_PRIVATE.parquet',f'{y}_test_metadata_PRIVATE.parquet']:
            f=source(PREVIOUS/'evaluation_input'/n);shutil.copy2(f,run/'evaluation_input'/n)
        labels=arrays(run/'evaluation_input'/f'{y}_test_PRIVATE.npz')
        for k in ev:np.testing.assert_array_equal(ev[k],labels[k])
        assert not set(tr['groups'])&set(labels['groups'])
        write(run/'input'/f'pairing_{y}.json',{str(s):pairing(tr['groups'],configuration(s,'s0')) for s in SEEDS})
    selected=[]
    for r in prior_registry(PREVIOUS):
        if r['arm'] not in ['shallow_parent','q_unrestricted']:continue
        row=dict(r);row['source_arm']=r['arm'];row['arm']={'shallow_parent':'s0','q_unrestricted':'s1'}[r['arm']];row['reuse']=True
        folder=Path(r['path'])
        for n,h in json.loads(source(folder/'hashes.json').read_text()).items():assert sha(source(folder/n))==h
        f=CompetingRiskForest.load(folder/'forest_PRIVATE.json');assert asdict(f.config)==asdict(configuration(r['seed'],row['arm']))
        assert all(n['leaf_shrinkage']['strength']==10 for t in f.trees_ for n in t if 'cif' in n)
        row.update(forest_sha256=sha(folder/'forest_PRIVATE.json'),predictions_sha256=sha(folder/'predictions_PRIVATE.npz'));selected.append(row)
    assert len(selected)==12;write(run/'audit/reused_artifacts.json',selected)
    ix=source(PREVIOUS/'private/bootstrap_indices_PRIVATE.npz');shutil.copy2(ix,run/'private/bootstrap_indices_PRIVATE.npz')
    counts=source(PREVIOUS/'reports/cohort_counts.csv');shutil.copy2(counts,run/'reports/cohort_counts.csv')
    diffs=[]
    for seed in SEEDS:
        for i in range(1,5):
            a,b=STAGES[i-1],STAGES[i];left=asdict(configuration(seed,a));right=asdict(configuration(seed,b))
            changed={k for k in left if left[k]!=right[k]}
            assert changed==[{'path_budget'},{'cut_strategy'},{'tree_feature_strategy'},set()][i-1]
            for k in left:diffs.append(dict(seed=seed,from_stage=a,to_stage=b,stage='fit',parameter=k,before=json.dumps(left[k]),after=json.dumps(right[k]),changed=k in changed))
            diffs.append(dict(seed=seed,from_stage=a,to_stage=b,stage='leaf_estimator',parameter='strength',before=str(strength(a)),after=str(strength(b)),changed=strength(a)!=strength(b)))
    pd.DataFrame(diffs).to_csv(run/'reports/parameter_differences.csv',index=False)
    pd.DataFrame(selected).to_csv(run/'reports/artifact_reuse.csv',index=False);pd.DataFrame(inputrows).to_csv(run/'reports/input_hashes.csv',index=False)
    write(run/'audit/source_hashes.json',sources)
    write(run/'audit/baseline_audit.json',dict(status='PASS',reference_sha256=sha(ref),reused_models=12,new_fits_required=12,baseline_reproduction_fits=0,all_inputs_byte_identical=True,source_commit=SOURCE_COMMIT,bootstrap_indices_sha256=sha(ix),prior_data_audit_sha256=sha(PREVIOUS/'audit/baseline_audit.json')))
    print('PASS:12 reusable chain models;12 new fits required;inputs unchanged')

def manifest(run,mode):
    if mode=='formal':entries=[(y,s,a) for y in YEARS for a in NEW for s in SEEDS]
    else:
        rows=registry(run);assert len(rows)==30
        context=run/'validation_context/audit';context.mkdir(parents=True)
        # Read-only manifest adapter to the existing, unchanged verification code.
        write(context/'reused_artifacts.json',rows)
        entries=[(y,s,a) for y in YEARS for a in STAGES for s in SEEDS+[0]]
    tasks=[dict(kind='cpu',argv=['python3',str(run/'input/code/experiments/post_st_cumulative_ablation_v1/launch.py'),'--mode','fit' if mode=='formal' else 'inspect','--year',str(y),'--seed',str(s),'--arm',a,*(['--run',str(run)] if mode!='formal' else [])],cwd=str(run/'input/code'),timeout=3600,storage=dict(source=str(run/'input'),mode='nfs',results=str(run/'results'/mode))) for y,s,a in entries]
    with (run/f'{mode}.jsonl').open('x') as f:f.write(''.join(json.dumps(t)+'\n' for t in tasks))
    with (run/f'{mode}.py').open('x') as f:f.write('# Independent tasks in the heteropool sidecar.\n')
    write(run/f'{mode}.py.hetero.json',dict(independent=True,manifest=f'{mode}.jsonl'));print(len(tasks))

def freeze(run):
    repo=Path(__file__).resolve().parents[2];assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True)
    commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    for p in (run/'input/folds').glob('*/plan.json'):
        d=json.loads(p.read_text());d['code_commit']=commit;p.write_text(json.dumps(d,indent=2)+'\n')
    shutil.copy2(run/'audit/protocol.json',run/'input/protocol.json')
    with (run/'audit/code.tar').open('xb') as f:subprocess.run(['git','-C',str(repo),'archive',commit],stdout=f,check=True)
    with tarfile.open(run/'audit/code.tar') as t:t.extractall(run/'input/code',filter='data')
    manifest(run,'formal')
    write(run/'audit/immutable_input_hashes.json',{str(p.relative_to(run/'input')):sha(p) for p in (run/'input').rglob('*') if p.is_file()})
    write(run/'audit/pre_fit_freeze.json',dict(code_commit=commit,protocol_sha256=sha(run/'audit/protocol.json'),manifest_sha256=sha(run/'formal.jsonl'),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),new_results_seen=False))
    print(commit)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--stage',choices=['prepare','freeze','verify_manifest'],required=True);a=p.parse_args()
    {'prepare':prepare,'freeze':freeze,'verify_manifest':lambda r:manifest(r,'verify')}[a.stage](a.run)
