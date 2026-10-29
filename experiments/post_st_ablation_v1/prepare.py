"""Audit originals, stage byte-identical inputs and freeze an immutable run."""
from dataclasses import asdict
from pathlib import Path
import argparse, datetime, json, shutil, subprocess, tarfile
import numpy as np
import pandas as pd
from experiments.post_st_leaf_shallow_v1.analyze import artifacts
from experiments.post_st_next_open_info_v1.prepare import next_open
from experiments.post_st_prognosis_v1.prepare import MARKET
from experiments.post_st_prognosis_v1.common import calendar_elapsed
from crforest import CompetingRiskForest
from .common import *

def prepare(run, repo):
    for name in ['audit','input','evaluation_input','reports','private','results']:
        (run/name).mkdir(exist_ok=False)
    protocol=Path(__file__).with_name('frozen_protocol.json')
    shutil.copy2(protocol,run/'audit/protocol.json')
    sources={};reuse=[];configs=[];counts=[];inputrows=[]
    def source(p):
        p=Path(p);sources[str(p)]=sha(p);return p
    current=json.loads(source(repo/'CURRENT_POST_ST_MODEL.json').read_text())
    assert current['arm']=='shallow_parent' and current['private_run']==str(HISTORY)
    differences=subprocess.check_output(['git','-C',str(repo),'diff','--stat',SOURCE_COMMIT,'3931d0c69141e294ec2fb5e744bb912c80fe2c1a'],text=True)
    (run/'audit/source_to_starting_commit.diffstat').write_text(differences)
    assert not subprocess.check_output(['git','-C',str(repo),'diff',SOURCE_COMMIT,'3931d0c69141e294ec2fb5e744bb912c80fe2c1a','--','crforest','CURRENT_POST_ST_MODEL.json','experiments/post_st_leaf_shallow_v1'],text=True)
    cal=np.load(source(MARKET/'market_calendar.npy'),allow_pickle=False)
    for y in YEARS:
        dest=run/'input/folds'/str(y);dest.mkdir(parents=True)
        src=HISTORY/'input/folds'/str(y);plan=json.loads(source(src/'plan.json').read_text())
        for name in ['train_PRIVATE.npz','eval_PRIVATE.npz']:
            p=source(src/name);assert sha(p)==plan['source_hashes'][name]
            shutil.copy2(p,dest/name)
            inputrows.append(dict(year=y,file=name,sha256=sha(p),source=str(p)))
        write(dest/'plan.json',plan)
        tr=arrays(dest/'train_PRIVATE.npz');fit_ev=arrays(dest/'eval_PRIVATE.npz')
        assert set(fit_ev)=={'X','sample_ids'}
        ep=source(HISTORY/'evaluation_input'/f'{y}_test_PRIVATE.npz');shutil.copy2(ep,run/'evaluation_input'/ep.name)
        ev=arrays(ep)
        for k in fit_ev:np.testing.assert_array_equal(fit_ev[k],ev[k])
        assert not set(tr['groups'])&set(ev['groups'])
        for role,d in [('fit',tr),('test',ev)]:
            mp=source(HISTORY/'evaluation_input'/f'{y}_{role}_metadata_PRIVATE.parquet');shutil.copy2(mp,run/'evaluation_input'/mp.name)
            m=pd.read_parquet(mp)
            np.testing.assert_array_equal(m.sample_id,d['sample_ids']);np.testing.assert_array_equal(m.group,d['groups'])
            np.testing.assert_array_equal(m.event,d['event']);np.testing.assert_array_equal((m.stop-m.origin).dt.days,d['time'])
            np.testing.assert_array_equal(pd.to_datetime([next_open(v,cal) for v in m.entry_announcement]),m.origin)
            np.testing.assert_allclose([calendar_elapsed(o,s) for o,s in zip(m.origin,m.stop)],d['analysis_time'],atol=0,rtol=0)
            assert m.group.is_unique and d['X'].shape==(len(m),83) and m.label_eligible.all()
            if role=='fit':assert (m.stop<pd.Timestamp(f'{y}-01-01')).all()
            for h in [12,24]:
                hit=(d['event']>0)&(d['analysis_time']<=h);known=hit|(d['analysis_time']>=h)
                counts.append(dict(year=y,role=role,months=h,companies=len(m),termination=int(((d['event']==1)&hit).sum()),recovery=int(((d['event']==2)&hit).sum()),unknown=int((~known).sum())))
        write(run/'input'/f'pairing_{y}.json',{str(s):pairing(tr['groups'],configuration(s,'shallow_parent')) for s in SEEDS})
    for y,a,s,p in artifacts(HISTORY):
        for n,h in json.loads(source(p/'hashes.json').read_text()).items():assert sha(source(p/n))==h
        row=dict(year=y,arm=a,seed=int(s),path=str(p),reuse=True)
        row['forest_sha256']=sha(p/('forest_PRIVATE.rds' if a=='standard_rsf' else 'forest_PRIVATE.json'))
        row['predictions_sha256']=sha(p/'predictions_PRIVATE.npz');reuse.append(row)
        if a=='standard_rsf':continue
        f=CompetingRiskForest.load(p/'forest_PRIVATE.json');actual=asdict(f.config);expected=asdict(configuration(int(s),a))
        assert actual==expected,(a,actual,expected)
        shrink=[n['leaf_shrinkage']['strength'] for t in f.trees_ for n in t if 'leaf_shrinkage' in n]
        assert (len(shrink)>0)==(a in ['parent_shrink','shallow_parent'])
        if shrink:assert set(shrink)=={10.}
        configs.append(dict(year=y,arm=a,seed=int(s),configuration=canonical(actual),shrinkage_strength=10 if shrink else 0))
        if a=='shallow_parent':
            match=[z for z in current['saved_models'] if z['test_year']==y and z['seed']==int(s)][0]
            assert row['forest_sha256']==match['forest_sha256'] and row['predictions_sha256']==match['predictions_sha256']
    assert len(reuse)==30
    write(run/'audit/reused_artifacts.json',reuse);write(run/'audit/baseline_configurations.json',configs)
    write(run/'audit/source_hashes.json',sources)
    pd.DataFrame(counts).to_csv(run/'reports/cohort_counts.csv',index=False)
    pd.DataFrame(inputrows).to_csv(run/'reports/input_hashes.csv',index=False)
    pd.DataFrame(reuse).to_csv(run/'reports/artifact_reuse.csv',index=False)
    diffs=[]
    for s in SEEDS:
        base=asdict(configuration(s,'shallow_parent'))
        for a in NEW:
            cfg=asdict(configuration(s,a));changed=[k for k in base if cfg[k]!=base[k]]
            assert changed==[{'q_unrestricted':'path_budget','local_cuts':'cut_strategy','iid_features':'tree_feature_strategy'}[a]]
            for k in base:diffs.append(dict(seed=s,arm=a,stage='forest_fit',parameter=k,reference=json.dumps(base[k]),value=json.dumps(cfg[k]),changed=k in changed))
            for k,v in [('parent_strength',10),('time_weight',1),('primary_months',24),('secondary_months',12)]:
                diffs.append(dict(seed=s,arm=a,stage='derived_estimator' if k in ['parent_strength','time_weight'] else 'evaluation',parameter=k,reference=str(v),value=str(v),changed=False))
    pd.DataFrame(diffs).to_csv(run/'reports/parameter_differences.csv',index=False)
    write(run/'audit/baseline_audit.json',dict(status='PASS',historical_models=24,external_references=6,
        missing_artifacts=0,required_reproduction_fits=0,source_core_unchanged_at_start=True,
        current_reference_sha256=sha(repo/'CURRENT_POST_ST_MODEL.json'),original_data_PIT_certified=False,
        discovered_differences=['historical plan common depth6 overridden by worker to4; saved models verified depth4','new explicit tree_feature_strategy defaults balanced; missing legacy field means balanced'],
        protocol_sha256=sha(protocol),created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    print(pd.DataFrame(counts).to_string(index=False))

def freeze(run,repo):
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True)
    commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    for p in (run/'input/folds').glob('*/plan.json'):
        d=json.loads(p.read_text());d['code_commit']=commit;p.write_text(json.dumps(d,indent=2)+'\n')
    shutil.copy2(run/'audit/protocol.json',run/'input/protocol.json')
    with (run/'audit/code.tar').open('xb') as f:subprocess.run(['git','-C',str(repo),'archive',commit],stdout=f,check=True)
    with tarfile.open(run/'audit/code.tar') as f:f.extractall(run/'input/code',filter='data')
    tasks=[dict(kind='cpu',argv=['python3',str(run/'input/code/experiments/post_st_ablation_v1/launch.py'),'--mode','fit','--year',str(y),'--seed',str(s),'--arm',a],cwd=str(run/'input/code'),timeout=3600,
        storage=dict(source=str(run/'input'),mode='nfs',results=str(run/'results/formal')))
        for y in YEARS for a in NEW for s in SEEDS]
    with (run/'formal.jsonl').open('x') as f:f.write(''.join(json.dumps(t)+'\n' for t in tasks))
    (run/'formal.py').write_text('# Independent frozen task manifest; run using hetero-run.\n')
    write(run/'formal.py.hetero.json',dict(independent=True,manifest='formal.jsonl'))
    write(run/'audit/immutable_input_hashes.json',{str(p.relative_to(run/'input')):sha(p) for p in (run/'input').rglob('*') if p.is_file()})
    write(run/'audit/pre_fit_freeze.json',dict(code_commit=commit,protocol_sha256=sha(run/'audit/protocol.json'),manifest_sha256=sha(run/'formal.jsonl'),new_real_predictions_seen=False,utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    print(commit)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[2]);p.add_argument('--stage',choices=['prepare','freeze'],required=True);a=p.parse_args()
    (prepare if a.stage=='prepare' else freeze)(a.run,a.repo)
