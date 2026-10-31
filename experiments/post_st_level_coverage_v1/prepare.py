"""Audit and freeze sources before any new real or formal synthetic prediction."""
from dataclasses import asdict
from pathlib import Path
import argparse,datetime,json,shutil,subprocess,tarfile
import numpy as np
import pandas as pd
from .common import *
from experiments.post_st_structure_recovery_v1.common import SCENARIOS
from experiments.post_st_leaf_shallow_v1.analyze import artifacts

def definitions(repo):
    protocol=dict(name=NAME,source_commit='0a9e0dc3824ec239eaf372bcb05a84a6a4c23a29',
        reference_sha256=REFERENCE_SHA,years=YEARS,seeds=SEEDS,
        purpose='Separate probability-level calibration from q2 feature coverage; no combined arm or reference promotion.',
        reference=asdict(configuration(42,'reference')),wide83=asdict(configuration(42)),
        only_forest_change={'tree_feature_count':[25,83]},mtry_note='mtry stays75; actual pre-q-limit candidates rise from25 to75. This intentionally increases search coverage/cost, not an equal-computation comparison.',
        current_forests='Six frozen shallow_parent models; same old-C83 inputs, next-open origin, first termination versus recovery, announcement dates,24-calendar-month train cap; inherited PROVISIONAL PIT limitations.',
        calibration=dict(years_by_cutoff={str(y):{str(h):calibration_years(y,h) for h in [12,24]} for y in YEARS},
            unit='One three-seed ensemble calibrator per year and horizon;4 fits. Never calibrate and wide83 together.',
            temporal_rule='Latest TWO calendar-entry cohorts whose full horizon ends strictly before Jan1 of outer year; select by origin+horizon, NEVER by whether the event is known. All rows kept; require complete endpoints for this declared design, abort otherwise.',
            oof='Reuse unit-weight baseline historical rolling forests fitted strictly before each validation entry year; predict held-out companies. No in-sample calibration. Includes historical cohorts from withdrawn temporal study only as immutable unit-weight artifacts, not adoption of its selected policy.',
            mapping='Fnew_k(t,x)=c_k*g_H(t)+b_k*F_k(t,x); c>=0; time-constant .5<=b<=4; fewer than10 target-cause events fixes that cause slope to1. No gate manipulation or extra cohorts.',
            shape='g_H=outer-training pooled-event AJ normalized at day365/730, held1 after that day. Origin0. Assumed offset shape, not claimed calibrated at intermediate times.',
            objective='Joint two-cause squared probability loss at the given calendar horizon plus20/n times mean squared change from identity; no hyperparameter selection.',
            constraints='Terminal convex hull of average independent tree-leaf choices (Minkowski sum) gives an outer bound for all x. c1+c2+b dot vertex<=1, including all observed OOF probability vectors. Global all-time certificate follows from nonnegative monotone offsets and positive slopes.',
            numeric='Convex4-variable QP; SLSQP plus active KKT polish and linear-oracle gap<=1e-8, primal<=1e-10. No clipping. Failed certificate retained, no fake successful calibration.',
            labels='Use cutoff-specific frozen training metadata; stop strictly before outer Jan1, original labels and sources unchanged.'),
        simulation=dict(scenarios=SCENARIOS,repetitions=list(range(50)),base_seed=20261030,
            primary='Reuse exactly450 preceding data repetitions and q2 controls; fit onlywide83 (1350 forests), raw and parent-shrunk derivatives. Same pairing; newly exposed follow-up simulation, not a new independent confirmation.',
            true_recovery='Unchanged analytic population-marginal-product reference, amplitude projection and fullL2/RMS, structure detectionRMS>=.005 with .0025/.01 sensitivity; both causes365/730days, ensemble primary.',
            pairing='Exact same generated input/outcome hashes and company bootstrap multiplicities. Eligibility intentionally differs. mtry/cutpoint RNG rule unchanged; structures need not match.',
            validation='All train/test rows at both horizons;32 fixed probes all native times; explicit component sums on4 probes. Native leaf CIF certificate for all inputs. Same validation scope as preceding simulation.'),
        enterprise=dict(new_fits=6,derived_shrunk=6,baseline_reused=6,
            main='24m termination, ensemblefirst;12m and recovery secondary. Save all seeds/raw. Annual and pooled AUC,AP,IPCW Brier,mean risk/incidence,Top20/25,calibration bins.',
            uncertainty='1000 paired company resamples stratified by test year; seed20261031; fixed predictions, common indices for every contrast. Conditional descriptive intervals, not complete training or selection uncertainty.',
            validation='ALL train/test rows at ALL native times and12/24 endpoints, both causes; full main+pair sum, save/reload; max error1e-10. Same training product reference for model comparisons.'),
        budget=dict(pilot_tasks=2,pilot_fits=2,formal_simulation_tasks=450,formal_simulation_fits=1350,enterprise_tasks=6,enterprise_fits=6,calibration_tasks=4,calibration_fits=4,
            cpu_threads_per_task=1,address_space_GiB=4,timeout_per_task_seconds=3600,pool='oliver+lemon;Slurm managed P/E/SMT affinity;NFS;no GPU or cross-node single fit'),
        boundaries=['No2025 outcomes or evaluation','No new forest calibration-window refits','No nonlinear mapping or probability clipping','No new thresholds/seeds/weights/q/depth searches','Preserve withdrawn experiments and current reference','All actual failed tasks/negative results retained','Raw enterprise data,parameters,thresholds,predictions remain private'])
    folder=repo/'experiments'/NAME
    write(folder/'frozen_protocol.json',protocol)
    rows=[]
    for s in SEEDS:
        a,b=asdict(configuration(s,'reference')),asdict(configuration(s))
        assert [k for k in a if a[k]!=b[k]]==['tree_feature_count']
        rows.extend(dict(seed=s,parameter=k,reference=json.dumps(a[k]),wide83=json.dumps(b[k]),changed=a[k]!=b[k]) for k in a)
    pd.DataFrame(rows).to_csv(folder/'parameter_differences.csv',index=False)

def prepare(run,repo):
    for name in ['input','audit','reports','private','evaluation_input','results']:(run/name).mkdir(exist_ok=False)
    sources={};reuse=[];counts=[]
    def copy(p,q):
        sources[str(p)]=sha(p);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
    # Forest fitting sees evaluation X only; labels are kept out of pool input.
    for y in YEARS:
        src=HISTORY/'input/folds'/str(y);dst=run/'input/folds'/str(y)
        for n in ['train_PRIVATE.npz','eval_PRIVATE.npz']:copy(src/n,dst/n)
        meta=pd.read_parquet(HISTORY/'evaluation_input'/f'{y}_fit_metadata_PRIVATE.parquet')
        sources[str(HISTORY/'evaluation_input'/f'{y}_fit_metadata_PRIVATE.parquet')]=sha(HISTORY/'evaluation_input'/f'{y}_fit_metadata_PRIVATE.parquet')
        tr=arrays(dst/'train_PRIVATE.npz');np.testing.assert_array_equal(meta.sample_id,tr['sample_ids'])
        assert (meta.stop<pd.Timestamp(f'{y}-01-01')).all()
        assert tr['X'].shape[1]==83 and set(arrays(dst/'eval_PRIVATE.npz'))=={'X','sample_ids'}
        for kind in ['test_PRIVATE.npz','test_metadata_PRIVATE.parquet']:copy(HISTORY/'evaluation_input'/f'{y}_{kind}',run/'evaluation_input'/f'{y}_{kind}')
        for h in [12,24]:
            chunks=[];ys=calibration_years(y,h)
            for v in ys:
                src_label=ROLLING/'private'/f'validation_{y}_{v}_labels_PRIVATE.npz';labels=arrays(src_label)
                sources[str(src_label)]=sha(src_label)
                selected=meta.origin.dt.year==v
                np.testing.assert_array_equal(meta.sample_id[selected],labels['sample_ids'])
                assert all(o+pd.DateOffset(months=h)<pd.Timestamp(f'{y}-01-01') for o in meta.origin[selected])
                assert np.all((labels['analysis_time']>=h)|(labels['event']>0))
                chunks.append(labels)
            cal={k:np.concatenate([d[k] for d in chunks]) for k in chunks[0]}
            folder=run/'input/calibration'/str(y)/str(h);folder.mkdir(parents=True)
            np.savez_compressed(folder/'labels_PRIVATE.npz',**cal)
            write(folder/'cohorts.json',ys)
            e=cal['event'];t=cal['analysis_time']
            counts.append(dict(outer_year=y,horizon_months=h,entry_years=','.join(map(str,ys)),companies=len(e),termination=int(((e==1)&(t<=h)).sum()),recovery=int(((e==2)&(t<=h)).sum()),unknown=0,cutoff_exclusive=f'{y}-01-01'))
    for y,arm,s,path in artifacts(HISTORY):
        if arm!='shallow_parent':continue
        current=json.loads((repo/'CURRENT_POST_ST_MODEL.json').read_text())
        entry=next(z for z in current['saved_models'] if z['test_year']==y and z['seed']==int(s))
        assert sha(path/'forest_PRIVATE.json')==entry['forest_sha256']
        assert sha(path/'predictions_PRIVATE.npz')==entry['predictions_sha256']
        for n in ['forest_PRIVATE.json','predictions_PRIVATE.npz']:
            expected=json.loads((path/'hashes.json').read_text())[n];assert sha(path/n)==expected
            copy(path/n,run/'input/reference'/str(y)/s/n)
        reuse.append(dict(kind='current_reference',year=y,seed=s,source=str(path),forest_sha256=sha(path/'forest_PRIVATE.json')))
    assert len(reuse)==6
    gate=json.loads((ROLLING/'audit/validation_gate.json').read_text());assert gate['status']=='PASS'
    for i,line in enumerate((ROLLING/'validation.jsonl').read_text().splitlines()):
        a=json.loads(line)['argv'];v=int(a[a.index('--validation-year')+1]);s=int(a[a.index('--seed')+1])
        src=ROLLING/'results/validation'/str(gate['job'])/f'task-{i}/output/baseline'
        for n in ['forest_PRIVATE.json','predictions_PRIVATE.npz']:
            assert sha(src/n)==json.loads((src/'hashes.json').read_text())[n]
            copy(src/n,run/'input/rolling'/str(v)/str(s)/n)
        reuse.append(dict(kind='rolling_oof',year=v,seed=s,source=str(src),forest_sha256=sha(src/'forest_PRIVATE.json')))
    old_gate=json.loads((SIMULATION/'audit/formal_gate.json').read_text());assert old_gate['status']=='PASS'
    (run/'input/sim_reference').mkdir()
    for i,line in enumerate((SIMULATION/'formal.jsonl').read_text().splitlines()):
        a=json.loads(line)['argv'];scenario=a[a.index('--scenario')+1];rep=int(a[a.index('--rep')+1])
        old=SIMULATION/'results/formal'/str(old_gate['job'])/f'task-{i}/output'
        names=['data_identity.json','pairing.json']
        for n in names:
            assert sha(old/n)==json.loads((old/'hashes.json').read_text())[n];sources[str(old/n)]=sha(old/n)
        write(run/'input/sim_reference'/f'{scenario}_{rep}.json',dict(identity=json.loads((old/'data_identity.json').read_text()),pairing=json.loads((old/'pairing.json').read_text()),old_path=str(old),hashes_sha256=sha(old/'hashes.json')))
    write(run/'audit/source_hashes.json',sources);write(run/'audit/reuse.json',reuse)
    pd.DataFrame(counts).to_csv(run/'reports/calibration_support.csv',index=False)
    print(pd.DataFrame(counts).to_string(index=False))

def freeze(run,repo):
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True)
    assert sha(repo/'CURRENT_POST_ST_MODEL.json')==REFERENCE_SHA
    commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    source=repo/'experiments'/NAME/'frozen_protocol.json'
    shutil.copy2(source,run/'input/protocol.json');shutil.copy2(source,run/'audit/protocol.json')
    with (run/'audit/code.tar').open('xb') as f:subprocess.run(['git','-C',str(repo),'archive',commit],stdout=f,check=True)
    with tarfile.open(run/'audit/code.tar') as f:f.extractall(run/'input/code',filter='data')
    stages={'pilot':[['sim_worker','--scenario',s,'--rep',str(r),'--pilot'] for s,r in [('pair_small',10001),('continuous_pair_large',10002)]],
        'formal':[['sim_worker','--scenario',s,'--rep',str(r)] for r in range(50) for s in SCENARIOS]+[['worker','--mode','wide','--year',str(y),'--seed',str(s)] for y in YEARS for s in SEEDS]+[['worker','--mode','calibrate','--year',str(y),'--horizon',str(h)] for y in YEARS for h in [12,24]]}
    for stage,args in stages.items():
        tasks=[dict(kind='cpu',argv=['python3',str(run/f'input/code/experiments/{NAME}/launch.py'),*a],cwd=str(run/'input/code'),timeout=3600,
            storage=dict(source=str(run/'input'),mode='nfs',results=str(run/'results'/stage))) for a in args]
        with (run/f'{stage}.jsonl').open('x') as f:f.write(''.join(json.dumps(t)+'\n' for t in tasks))
        (run/f'{stage}.py').write_text('# Independent frozen tasks; use hetero-run.\n')
        write(run/f'{stage}.py.hetero.json',dict(independent=True,manifest=f'{stage}.jsonl'))
    write(run/'input/code_identity.json',dict(commit=commit))
    write(run/'audit/immutable_input_hashes.json',{str(p.relative_to(run/'input')):sha(p) for p in (run/'input').rglob('*') if p.is_file()})
    write(run/'audit/pre_fit_freeze.json',dict(code_commit=commit,protocol_sha256=sha(source),reference_sha256=REFERENCE_SHA,
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),new_results_seen=False,manifests={s:sha(run/f'{s}.jsonl') for s in stages}))
    print(commit)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['definitions','prepare','freeze'],required=True);p.add_argument('--run',type=Path);p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[2]);a=p.parse_args()
    definitions(a.repo) if a.stage=='definitions' else (prepare if a.stage=='prepare' else freeze)(a.run,a.repo)
