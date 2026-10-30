"""Freeze exact code, finite matrix and independent simulation seeds before fits."""
from dataclasses import asdict
from pathlib import Path
import argparse,datetime,json,shutil,subprocess,tarfile
import pandas as pd
from .common import *
from experiments.post_st_ablation_v1.stages import gate

def protocol():
    return dict(name='post_st_structure_recovery_v1',source_commit='65776adaac9fa0ad65625d5ef5dc90415dd8adc3',
        current_reference_sha256='aa12ff6d41a515625566cf94f20faa4435d8f9c8e71fdeef8d8faba77fc57c65',
        purpose='Known-CIF structure AND amplitude recovery of the current complete predictor, not enterprise validation or model promotion.',
        arms={a:asdict(configuration(42,a)) for a in ARMS},only_fit_difference=config_differences(),
        post_fit=dict(shrinkage='raw direct parent AJ',strength=10,weight='unique inbag companies/(unique inbag companies+10)',raw_secondary=True),
        scenarios=SCENARIOS,independent_data_repetitions=REPETITIONS,repetitions=list(range(REPETITIONS)),
        seed_streams=dict(base=20261030,training_X=0,training_outcome=1,test_X=2,signal_coordinate_assignment=99),
        forest_seeds=SEEDS,primary_predictor='mean of three 32-tree seed predictions (96 trees)',
        input_count=P,signal_location='7 distinct coordinates uniformly sampled independently of outcomes per data repetition; same mapping across scenarios',
        test_X_count=5000,horizons_days=TIMES.tolist(),synthetic_origin='2021-01-01 (365/730 days equal12/24 calendar months)',
        truth=dict(target='additive:.18+.04*z0+.03*z1; pair adds.05*z0*z1; pure_pair:.18+.04*z2+.05*z0*z1; rare:.06+.01*z0+.01*z1+.02*z0*z1; triple additionally+.04*z4*z5*z6',
            competing='.40+.04*z3',time_shape='H(t)=(1-exp(-min(max(t,0),730)/450))/(1-exp(-730/450))',
            event_times='conditional cause sampled from endpoint probabilities; common truncated exponential and ceil to positive integer days',
            censoring='independent ceil exponential with mean730/censor, administratively capped730; event wins integer-day ties',
            binary='independent +/-1 with equal probability',continuous='independent U[-1,1]',
            correlation='second true-pair coordinate copies first with probability.7, otherwise independent; margins unchanged'),
        primary_metrics=['24m cause1 true-pair signed projection coefficient / true coefficient','exact function L2 error and RMS','structure precision/recall/F1 at RMS>=.005','joint-test true-CIF MSE','exact product-reference true-CIF MSE'],
        secondary_metrics=['12m and cause2','threshold sensitivity .0025 and.01','raw vs parent-shrunk from same fitted partition','true triple recovery and spurious higher-order energy','eligible/coefficient-nonzero trees and per-tree amplitude','seed variability','logical cells including missing; factored bytes; times; memory'],
        recovery_reference='Known population marginal product, analytic interval masses/moments; only recentering changes, never predictions. Correlated scenario joint prediction evaluated separately.',
        amplitude='Coefficient projection onto known centered monomial; for binary this is complete component amplitude. Continuous projection is supplemented by exact L2 error/RMS, not asserted to characterize every step-function shape.',
        uncertainty='50 independent data repetitions; mean, SD, MCSE, t-based95% mean intervals and paired q2-minus-unrestricted intervals. Seeds are not independent datasets. No result-dependent significance or tuning.',
        validation=dict(probability='all native-time leaves plus positive averaging; no clipping',reconstruction_tolerance=1e-10,
            reconstruction_scope='all training+5000 test rows at365/730days;32 deterministic training/test probe rows at all native times; explicit component sum on4 probes; full orders<=4',
            reload='native JSON gzip round trip and function NPZ; exact comparison',pairing='same X/outcomes/company bootstrap multiplicities/eligible features per seed/tree',
            pilots='2 independent repetitions10001/10002, one seed42 each:pair_small and continuous_pair_large; complete node replay including raw parent AJ and candidate scores'),
        budget=dict(formal_tasks=9*REPETITIONS,formal_forest_fits=9*REPETITIONS*6,formal_derived_shrunk_models=9*REPETITIONS*6,
            pilot_tasks=2,pilot_fits=4,threads_per_task=1,address_space_GiB=4,timeout_seconds_per_task=3600,pool='oliver+lemon Slurm; dynamic P/E-aware affinity; NFS'),
        boundaries=['No original enterprise inputs/outcomes read','No new parameter search; current reference unchanged','No causal identification from known statistical DGP','No missing-data or all possible economic regimes validation','Unrestricted comparator is same depth4 implementation, not external standard RSF','Failures retained; publication-only recovery cannot refit'])

def definitions(repo):
    dest=repo/'experiments/post_st_structure_recovery_v1'
    write(dest/'frozen_protocol.json',protocol())
    rows=[]
    for s in SEEDS:
        a,b=[asdict(configuration(s,k)) for k in ARMS]
        rows.extend(dict(seed=s,stage='fit',parameter=k,q2=json.dumps(a[k]),unrestricted=json.dumps(b[k]),changed=a[k]!=b[k]) for k in a)
    pd.DataFrame(rows).to_csv(dest/'parameter_differences.csv',index=False)

def freeze(run,repo):
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True)
    assert sha(repo/'CURRENT_POST_ST_MODEL.json')==protocol()['current_reference_sha256']
    assert not subprocess.check_output(['git','-C',str(repo),'diff',protocol()['source_commit'],'--','crforest','CURRENT_POST_ST_MODEL.json'],text=True)
    commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    for name in ['input','audit','reports','results']:(run/name).mkdir(exist_ok=False)
    source=repo/'experiments/post_st_structure_recovery_v1/frozen_protocol.json'
    for p in [run/'input/protocol.json',run/'audit/protocol.json']:shutil.copy2(source,p)
    with (run/'audit/code.tar').open('xb') as f:subprocess.run(['git','-C',str(repo),'archive',commit],stdout=f,check=True)
    with tarfile.open(run/'audit/code.tar') as f:f.extractall(run/'input/code',filter='data')
    for stage,items in [('pilot',[('pair_small',10001),('continuous_pair_large',10002)]),('formal',[(s,r) for r in range(REPETITIONS) for s in SCENARIOS])]:
        tasks=[dict(kind='cpu',argv=['python3',str(run/'input/code/experiments/post_st_structure_recovery_v1/launch.py'),'--scenario',s,'--rep',str(r)]+(['--pilot'] if stage=='pilot' else []),
            cwd=str(run/'input/code'),timeout=3600,storage=dict(source=str(run/'input'),mode='nfs',results=str(run/'results'/stage))) for s,r in items]
        with (run/f'{stage}.jsonl').open('x') as f:f.write(''.join(json.dumps(t)+'\n' for t in tasks))
        (run/f'{stage}.py').write_text('# Frozen independent tasks; invoke hetero-run.\n')
        write(run/f'{stage}.py.hetero.json',dict(independent=True,manifest=f'{stage}.jsonl'))
    write(run/'audit/immutable_input_hashes.json',{str(p.relative_to(run/'input')):sha(p) for p in (run/'input').rglob('*') if p.is_file()})
    write(run/'audit/pre_fit_freeze.json',dict(code_commit=commit,protocol_sha256=sha(source),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        new_simulation_results_seen=False,manifest_sha256={s:sha(run/f'{s}.jsonl') for s in ['pilot','formal']}))
    print(commit)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['definitions','freeze','pilot_gate','formal_gate'],required=True)
    p.add_argument('--run',type=Path);p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[2]);p.add_argument('--job');p.add_argument('--report',type=Path);a=p.parse_args()
    if a.stage=='definitions':definitions(a.repo)
    elif a.stage=='freeze':freeze(a.run,a.repo)
    else:gate(a.run,a.stage.removesuffix('_gate'),a.job,a.report)
