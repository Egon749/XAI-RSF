import datetime,shutil,tempfile,subprocess
import pandas as pd
from crforest import CompetingRiskForest
from experiments.post_st_tree_count_v1.nested import prefix
from .common import *

def main():
    old=previous_path();pub=REPO/'experiments'/NAME
    r=Path(tempfile.mkdtemp(prefix='seed20_32_20261103_',dir=SOURCE/'private'))
    write(pub/'run_location.json',dict(private_run=str(r),source_run=str(old)))
    for n in ['input','audit','evaluation_input','private','reports','results']:(r/n).mkdir()
    sources={}
    def record(p):sources[str(p)]=sha(p);return p
    manifest=load(old/'audit/input_hashes.json')
    for name,h in manifest.items():
        if not(name.startswith('folds/')or name in ['features.json','old_ensemble_registry.json','old_registry.json']):continue
        p=record(old/'input'/name);assert sha(p)==h
        dest=r/'input'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    for y in YEARS:shutil.copy2(record(old/f'evaluation_input/{y}_test_PRIVATE.npz'),r/f'evaluation_input/{y}_test_PRIVATE.npz')
    shutil.copy2(record(old/'private/bootstrap_indices_PRIVATE.npz'),r/'private/bootstrap_indices_PRIVATE.npz')
    for key,source in load(r/'input/old_registry.json').items():
        y,s=map(int,key.split('|'));src=Path(source);dest=r/f'input/reused/{y}/{s}';dest.mkdir(parents=True)
        hs=load(record(src/'hashes.json'))
        for v in ['raw','shrunk']:
            p=record(src/f'{v}/forest_PRIVATE.json');assert sha(p)==hs[f'{v}/forest_PRIVATE.json']
            f=CompetingRiskForest.load(p);assert len(f.trees_)==TREES
            shutil.copy2(p,dest/f'{v}_PRIVATE.json')
        record(src/'pairing.json')
    for rec in load(r/'input/old_ensemble_registry.json'):
        p=Path(rec['path']);hs=load(record(p/'hashes.json'))
        for n,h in hs.items():assert sha(record(p/n))==h
    # Protected objects and source metrics are retained byte-for-byte.
    record(REPO/'experiments/post_st_tree_count_70_120_v1/METRICS_ENSEMBLE.csv')
    shutil.copy2(REPO/'experiments/post_st_tree_count_70_120_v1/COHORT_AUDIT.csv',pub/'COHORT_AUDIT.csv')
    differences=[]
    for s in SEEDS:
        for k,v in asdict(configuration(s,TREES)).items():
            ref=asdict(configuration(42,32))[k];assert k=='seed'or v==ref
            differences.append(dict(seed=s,parameter=k,reference=json.dumps(ref),actual=json.dumps(v),changed=v!=ref))
    pd.DataFrame(differences).to_csv(pub/'PARAMETER_DIFFERENCES.csv',index=False)
    protocol=dict(name=NAME,status='FROZEN_BEFORE_NEW_SEED_RESULTS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        years=YEARS,seeds=SEEDS,trees_per_independent_seed=TREES,
        seed_provenance='Fixed explicit list includes prior42/101/202;17 others chosen before new seed results, not performance-selected.',
        interpretation='User-confirmed20 independent32-tree predictors; preserve current3x32 equal-average96-tree reference. A single32-tree seed is a component,not an equal-budget96-tree replacement.',
        configuration=asdict(configuration(42,TREES)),features=74,training_followup_calendar_months=24,
        input='Exact prior74 conservativeC snapshots; own pre-Jan1 information, next-trading-day preopen origin, firsttermination1/recovery2; no input/label revisions.',
        computation=dict(models=60,reused_forests=9,new_forest_fits=51,new_trees=51*TREES,pilot_fits=1,formal_fits=50,pilot_reused_in_grid=True,calibration_fits=0),
        pilot=dict(year=PILOT[0],seed=PILOT[1],trees=TREES,gate='No outcome metrics. Full numeric checks and peakRSS<=900MiB; reuse pilot as its formal grid result.'),
        outcomes='Report every seed,2023/24/25 annual and pooled12months,bothcauses;24months2023/24,2025H24NA. Shrunk primary,raw retained.',
        aggregation='No20-seed combined predictor; across-seed mean,SD,median,min,max and quantiles describe algorithmic variation at fixed cohorts. Report per-seed pooled metrics recomputed,not annual averages.',
        uncertainty='Reuse1000 year-stratified company draws seed2026110201; seed-specific difference vs fixed3x32 reference,conditional on predictions. No best-seed selection; no claim20 independent validation cohorts.',
        functions='Every raw/shrunk forest on all train/eval rows,all native times,bothcauses; full/independent component sum/reload<=1e-10,legalCIF,no clipping; between-seed function and prediction stability at same training reference.',
        resources='Existing2nodeSlurm,NFS;singlethread jobs,24oliver/6lemon memoryslots,2GiB processcap,pilotRSSgate900MiB; no hardcodedCPUaffinity. No automatic failed-fit retries.',
        privacy='Company data,models,cutpoints,predictions/functions private;onlyaggregate metrics and hashes public.',
        decision='No automatic seed selection,parameter search,reference replacement,branch merge or20seedensemble adoption. Exposed2023–2025 exploratory;historicalPIT and74selection limitations inherited.')
    write(pub/'frozen_protocol.json',protocol);write(r/'input/protocol.json',protocol)
    write(r/'audit/source_hashes.json',sources);write(r/'audit/current_pointer.json',dict(sha256=sha(REPO/'CURRENT_POST_ST_MODEL.json')))
    print(r,flush=True)
if __name__=='__main__':main()
