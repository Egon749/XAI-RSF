"""Freeze a finite experiment; copy only V19/V20 and hash all private inputs."""
import argparse, hashlib, json, shutil, subprocess, sys, tarfile
from pathlib import Path
from dataclasses import asdict
import numpy as np
import pandas as pd
from crforest import ForestConfig
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'crforest_pathbudget_v1'))
from simulation import SCENARIOS, SEEDS, sample


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    with Path(p).open('x') as f: json.dump(d,f,indent=2,ensure_ascii=False,allow_nan=False)

def main(run,repo,source):
    run,repo,source=map(Path,(run,repo,source));run.mkdir(exist_ok=False)
    inp=run/'input';inp.mkdir();(run/'audit').mkdir();(run/'private').mkdir();(run/'reports').mkdir()
    hashes={};counts={}
    for fold in ['V19','V20']:
        for role in ['train','eval']:
            src=source/f'{fold}_{role}';dst=inp/src.name;shutil.copytree(src,dst)
            for p in dst.iterdir():
                if p.is_file():
                    assert sha(p)==sha(src/p.name);hashes[str(p.relative_to(inp))]=sha(p)
            meta=json.loads((dst/'manifest.json').read_text());qm=json.loads((dst/'quality_manifest.json').read_text())
            assert sha(dst/'bundle.npz')==meta['bundle_sha256']
            assert sha(dst/'quality_PROVISIONAL.npz')==qm['quality_bundle_sha256'] and qm['status']=='PROVISIONAL'
            with np.load(dst/'bundle.npz') as a: a={k:a[k] for k in a.files}
            with np.load(dst/'quality_PROVISIONAL.npz') as q:q={k:q[k] for k in q.files}
            d=pd.read_parquet(dst/'metadata_PRIVATE.parquet')
            assert np.array_equal(a['sample_ids'],d.sample_id) and np.array_equal(a['sample_ids'],q['sample_ids'])
            assert np.array_equal(a['groups'],d.group.astype(str))
            assert np.array_equal(a['origins'],d.origin.to_numpy().astype('datetime64[D]'))
            for clock in [a['available_at'],q['available_at']]:
                assert not ((~np.isnat(clock)) & (clock>=a['origins'][:,None])).any()
            t=(d.stop_date-d.origin).dt.days.to_numpy();h=(d.horizon-d.origin).dt.days.to_numpy()
            assert set(h)<= {365,366} and (t>=0).all() and (t<=h).all()
            if role=='train':
                cutoff=np.datetime64(meta['fit_cutoff'])
                assert (t>0).all() and (a['observed_end']<cutoff).all() and (a['outcome_available_at']<cutoff).all()
                assert np.array_equal(a['event_type'],d.event)
                assert np.array_equal(a['observed_end'],d.stop_date.to_numpy().astype('datetime64[D]'))
            X=np.column_stack([a['X'],q['X']]);assert X.shape[1]==75 and not np.isinf(X).any()
            np.savez_compressed(inp/f'{fold}_{role}_PRIVATE.npz',X=X,time=t,event=d.event.to_numpy(),groups=a['groups'],sample_ids=a['sample_ids'],horizon=h)
            counts[f'{fold}_{role}']=dict(records=len(d),groups=int(d.group.nunique()),target_records=int((d.event==1).sum()),target_events=int(d.loc[d.event==1,'event_id'].nunique()),competing_records=int((d.event==2).sum()),competing_events=int(d.loc[d.event==2,'event_id'].nunique()),origin_unresolved=int((d.origin_state!='N_observed').sum()),zero_followup=int((t==0).sum()))
        shutil.copy2(source/f'{fold}_M2_PRIVATE.npz',inp/f'{fold}_M2_PRIVATE.npz')
    # Identical evaluation helpers from the authoritative historical protocol.
    legacy=Path('/home/oliver/LLMworkspace/early-warning-12m-model-comparison-2012plus/experiments/early_warning_12m')
    (inp/'legacy').mkdir()
    for name in ['common.py','evaluate.py','artifacts.py']:shutil.copy2(legacy/name,inp/'legacy'/name)
    shutil.copy2('/home/oliver/LLMworkspace/newRSF2/experiments/real_development_v1/endpoint_guard.py',inp/'legacy/endpoint_guard.py')
    for name in ['DATA_ALIGNMENT_AUDIT_ZH.md','execution_receipt.json']:
        p=Path('/home/oliver/LLMworkspace/newRSF2/experiments/real_development_v1')/name
        if p.exists():shutil.copy2(p,run/'audit'/name)
    real=asdict(ForestConfig(n_trees=32,max_depth=6,min_leaf=100,min_events=3,min_groups=20,mtry=75,n_thresholds=5));real.pop('seed')
    plan=dict(status='FROZEN_PROVISIONAL',folds=['V19','V20'],real_seeds=[42],real_variants=['unrestricted','depth2','q2','q1'],formal_real_fits=8,pilot_fits=2,real_forest=real,
              synthetic_tasks=18,synthetic_fits=90,synthetic_seeds=SEEDS,synthetic_scenarios=SCENARIOS,tuning_trials=0,penalty='none; no forest-specific identifiability claim',
              feature_semantics='75 input coordinates: 72 financial/market plus 3 quality; native missingness; derived ratios treated as declared inputs, not primitive raw-item order',
              reference='each fold training empirical marginals; product reference; no validation reference fitting',
              time='days since quarterly origin; predict separately at 365 or 366 days to match each 12-calendar-month horizon',
              resampling='company-cluster bootstrap; fixed seed and tree draws shared across variants',
              candidate_search='all 75 features; up to 5 observed cutpoints per node-feature; native missing routing; no two-step lookahead',
              complexity='path-order and depth/leaf budgets only; no global pair/threshold budget: readable complexity is measured, not guaranteed',
              evaluation='frozen M2; identical guarded legacy IPCW; record/company equal; pooled AP recomputed; quarterly Top20/25 and 3/6-month lead',
              real_limitations=['historical PIT vintages not certified','quality defaults with unknown clocks retained and flagged','origin unresolved rows retained as historical','terminal/suspension censoring semantics provisional','temporal validation is not company-disjoint','one seed per real model: no training-uncertainty claim'],
              resources=dict(cpu_threads_per_task=1,address_limit_GiB=6,timeout_seconds=7200,storage='NFS'),
              code_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),source_counts=counts)
    write(inp/'real_config.json',plan);write(repo/'experiments/development_v1/frozen_protocol.json',plan)
    simcfg=asdict(ForestConfig());simcfg.pop('seed')
    write(inp/'config.json',dict(forest=simcfg,scenarios=SCENARIOS,seeds=SEEDS,status='FROZEN_SYNTHETIC_MECHANISM_CHECK'))
    for i,s in enumerate(SCENARIOS):
        np.savez_compressed(inp/f'{s}_test.npz',**sample(1800,900000+i*1000,s));ref=[]
        for seed in SEEDS:
            tr=sample(1200,seed+i*10000,s);np.savez_compressed(inp/f'{s}_{seed}_train.npz',**tr);ref.append(tr['X'])
        np.save(inp/f'{s}_reference.npy',np.vstack(ref))
    archive=run/'audit/code_snapshot.tar'
    with archive.open('xb') as f:subprocess.run(['git','-C',str(repo),'archive',plan['code_commit']],stdout=f,check=True)
    with tarfile.open(archive) as tf:tf.extractall(inp/'code',filter='data')
    def task(kind,args,stage):
        script='experiments/development_v1/launch.py'
        return dict(kind='cpu',argv=['python3',str(inp/'code'/script),'--kind',kind,*args],cwd=str(inp/'code'),timeout=7200,storage=dict(source=str(inp),mode='auto',results=str(run/'results'/stage)))
    manifests=dict(pilot=[task('real',['--fold','V19','--variant',v,'--pilot'],'pilot') for v in ['unrestricted','q2']],
                   formal=[task('real',['--fold',f,'--variant',v],'formal') for f in plan['folds'] for v in plan['real_variants']],
                   simulation=[task('simulation',['--scenario',s,'--seed',str(seed)],'simulation') for s in SCENARIOS for seed in SEEDS])
    for name,tasks in manifests.items():
        (run/f'{name}.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in tasks))
        (run/f'{name}.py').write_text('# Use hetero-run with the sidecar manifest.\n')
        write(run/f'{name}.py.hetero.json',dict(independent=True,manifest=f'{name}.jsonl'))
    hashes={str(p.relative_to(inp)):sha(p) for p in inp.rglob('*') if p.is_file()}
    write(run/'audit/input_hashes.json',hashes)
    print(json.dumps(dict(run=str(run),counts=counts,config_sha256=sha(inp/'real_config.json'))))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--repo',required=True);p.add_argument('--source',required=True);a=p.parse_args();main(a.run,a.repo,a.source)
