import datetime,shutil,tempfile,subprocess
import pandas as pd
from .common import *

def main():
    old=previous_path();pub=REPO/'experiments'/NAME
    r=Path(tempfile.mkdtemp(prefix='requested_seeds8_20261103_',dir=SOURCE/'private'))
    write(pub/'run_location.json',dict(private_run=str(r),source_run=str(old)))
    for n in ['input','audit','evaluation_input','private','reports','results']:(r/n).mkdir()
    sources={}
    def record(p):sources[str(p)]=sha(p);return p
    for name,h in load(old/'audit/input_hashes.json').items():
        if not(name.startswith('folds/')or name in ['features.json','old_ensemble_registry.json','old_registry.json']):continue
        p=record(old/'input'/name);assert sha(p)==h;dest=r/'input'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    for y in YEARS:shutil.copy2(record(old/f'evaluation_input/{y}_test_PRIVATE.npz'),r/f'evaluation_input/{y}_test_PRIVATE.npz')
    shutil.copy2(record(old/'private/bootstrap_indices_PRIVATE.npz'),r/'private/bootstrap_indices_PRIVATE.npz')
    prior=load(record(old/'audit/registry_pilot.json'))+load(record(old/'audit/registry_formal.json'))
    prior=[z for z in prior if z['seed']in HISTORICAL_SEEDS];assert len(prior)==21
    for rec in prior+load(r/'input/old_ensemble_registry.json'):
        p=Path(rec['path']);hs=load(record(p/'hashes.json'))
        for n,h in hs.items():assert sha(record(p/n))==h
    write(r/'input/historical_registry.json',prior)
    record(REPO/'experiments/post_st_seed_stability20_v1/METRICS_ALL.csv')
    shutil.copy2(REPO/'experiments/post_st_seed_stability20_v1/COHORT_AUDIT.csv',pub/'COHORT_AUDIT.csv')
    differences=[]
    for s in SEEDS:
        for k,v in asdict(configuration(s,TREES)).items():
            ref=asdict(configuration(42,32))[k];assert k=='seed'or v==ref
            differences.append(dict(seed=s,parameter=k,reference=json.dumps(ref),actual=json.dumps(v),changed=v!=ref))
    pd.DataFrame(differences).to_csv(pub/'PARAMETER_DIFFERENCES.csv',index=False)
    protocol=dict(name=NAME,status='FROZEN_BEFORE_RERUN_AND_SEED300_RESULTS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        years=YEARS,seeds=SEEDS,trees_per_seed=32,configuration=asdict(configuration(42,32)),features=74,training_followup_calendar_months=24,
        authorization='User requested seven seeds then separately42/101/202;101/202 already included. Deduplicated to8seeds; all actually refit.',
        provenance='User-selected subset after exposed20seed12monthAUC rankings plus new300. Exploratory selected-seed rerun,not independent validation or unbiased sampling of seeds.',
        input='Exact own pre-Jan1 snapshots; next-trading-day preopen origin,firsttermination1/recovery2,conservative74 input;no data/labelrevision.',
        computation=dict(models=24,new_forest_fits=24,new_trees=768,reused_forests=0,historical_models_for_verification_only=21,pilot_fits=1,formal_fits=23,pilot_included_in_grid=True,calibration_fits=0),
        pilot=dict(year=2024,seed=300,trees=32,gate='No performance selection; full numeric checks,peakRSS<=900MiB; pilot is this formal grid cell.'),
        evaluation='Each seed separately12/24months,bothcauses,raw/shrunk,annual and pooled;2025H24 performanceNA; original96 reference separate. Pooled metrics recomputed,Top20per-yearceil(.2n).',
        uncertainty='1000original year-stratified company resamples seed2026110201,fixed predictions;conditional seedminusreference96 intervals,not valid postselection confirmation.',
        verification='Replay all splits/leafAJ,parentAJ; train+eval allrows,allnative times,2causes,exact components/reload<=1e-10,CIFvalid. Seven knownseeds must reproduce tree structures/pairing/predictions andmetrics; new300 has no previous counterpart.',
        resources='Existing2nodeSlurm,NFS,singlethreadCPUtasks,24oliver/6lemon memoryslots,2GiB cap,pilotRSSgate900MiB;noCPUaffinityoverride.',
        privacy='Company data,models,thresholds,predictions/functions private;aggregate results/code/hashes public.',
        decision='Preserve original96(42/101/202each32),currentpointer/mainbundle/core and oldresults; no8seedensemble,seedselection,modelreplacement,merge or furthersearch.')
    write(pub/'frozen_protocol.json',protocol);write(r/'input/protocol.json',protocol)
    write(r/'audit/source_hashes.json',sources);write(r/'audit/current_pointer.json',dict(sha256=sha(REPO/'CURRENT_POST_ST_MODEL.json')))
    print(r,flush=True)
if __name__=='__main__':main()
