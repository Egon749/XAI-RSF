import datetime,shutil,tempfile
import pandas as pd
from .common import *
def main():
    old=previous_path();pub=REPO/'experiments'/NAME;assert not(pub/'run_location.json').exists()
    r=Path(tempfile.mkdtemp(prefix='requested_seeds17_20261103_',dir=SOURCE/'private'))
    for n in ['input','audit','evaluation_input','private','reports','results']:(r/n).mkdir()
    sources={}
    def record(p):sources[str(p)]=sha(p);return p
    for name,h in load(old/'audit/input_hashes.json').items():
        if not(name.startswith('folds/')or name in ['features.json','old_ensemble_registry.json','old_registry.json']):continue
        p=record(old/'input'/name);assert sha(p)==h;dest=r/'input'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    for y in YEARS:shutil.copy2(record(old/f'evaluation_input/{y}_test_PRIVATE.npz'),r/f'evaluation_input/{y}_test_PRIVATE.npz')
    shutil.copy2(record(old/'private/bootstrap_indices_PRIVATE.npz'),r/'private/bootstrap_indices_PRIVATE.npz')
    prior=load(record(old/'input/reused_registry.json'))+load(record(old/'audit/registry_pilot.json'))+load(record(old/'audit/registry_formal.json'))
    stability = stability_path()
    for year in YEARS:
        for filename in ("train_PRIVATE.npz", "eval_PRIVATE.npz"):
            relative = Path("input/folds") / str(year) / filename
            assert sha(record(stability / relative)) == sha(r / relative)
        label_file = Path("evaluation_input") / f"{year}_test_PRIVATE.npz"
        assert sha(record(stability / label_file)) == sha(r / label_file)
    historical = load(record(stability / "audit/registry_formal.json"))
    prior.extend(row for row in historical if row["seed"] == 2026)
    record(REPO / "experiments/post_st_seed_stability20_v1/METRICS_ALL.csv")
    prior=[z for z in prior if z['seed']in REUSED_SEEDS];assert len(prior)==N_REUSED
    assert {(z['year'], z['seed']) for z in prior} == {(year, seed) for year in YEARS for seed in REUSED_SEEDS}
    from crforest import CompetingRiskForest
    for z in prior:
        m=CompetingRiskForest.load(Path(z['path'])/'raw/forest_PRIVATE.json')
        assert asdict(m.config)==asdict(configuration(z['seed'],32)) and m.n_features_==74
    for rec in prior+load(r/'input/old_ensemble_registry.json'):
        p=Path(rec['path']);hs=load(record(p/'hashes.json'))
        for n,h in hs.items():assert sha(record(p/n))==h
    write(r/'input/reused_registry.json',prior)
    record(REPO/'experiments/post_st_requested_seeds14_v2/METRICS_ALL.csv')
    shutil.copy2(REPO/'experiments/post_st_requested_seeds14_v2/COHORT_AUDIT.csv',pub/'COHORT_AUDIT.csv')
    differences=[]
    for s in SEEDS:
        for k,v in asdict(configuration(s,TREES)).items():
            ref=asdict(configuration(42,32))[k];assert k=='seed'or v==ref
            differences.append(dict(seed=s,parameter=k,reference=json.dumps(ref),actual=json.dumps(v),changed=v!=ref))
    pd.DataFrame(differences).to_csv(pub/'PARAMETER_DIFFERENCES.csv',index=False)
    protocol=dict(name=NAME,status='FROZEN_BEFORE_NEW_SEED_RESULTS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_commit=SOURCE_COMMIT,
        years=YEARS,seeds=SEEDS,new_seeds=NEW_SEEDS,reused_seeds=REUSED_SEEDS,trees_per_seed=32,configuration=asdict(configuration(42,32)),features=74,training_followup_calendar_months=24,
        authorization='Seventeen distinct user-specified seeds, 32 trees each. Report individuals and a fixed equal-weight 544-tree ensemble.',
        provenance='Selected-seed exploratory expansion after exposedannualAUC rankings;not unbiasedseed distribution or independentvalidation.',
        input='Exact own preJan1snapshot,nexttradingdaypreopen origin,firsttermination1/recovery2,conservative74;no data orlabelrevision.',
        computation=dict(individual_models=N_FITS+N_REUSED,new_forest_fits=N_FITS,new_trees=N_FITS*TREES,reused_forests=N_REUSED,pilot_fits=1,formal_fits=N_FITS-1,pilot_included=True,ensemble_exports=6,versions=['raw','shrunk'],calibration_fits=0),
        pilot=dict(year=PILOT[0],seed=PILOT[1],trees=32,gate='Fullnumerics andpeakRSS<=900MiB,no performance gate;counts toward21fits.'),
        evaluation='Everyseed andequal17seedensemble,reference96separate;bothcauses/raw/shrunk,annualandpooled12/24months;2025H24NA. Equalcompanypredictionmeans before metrics;pooledrecomputed;Top20ceil(.2n)withinyear.',
        uncertainty='Original1000pairedyearstratifiedcompanydraws seed2026110201,allseed/ensembleminusreference96,fixedpredictions;conditional exploratory intervals,no postselectionadjustment.',
        verification='Allsplitcandidate/leafAJ/directparent replay fornewmodels;alltrain/evalrows/allnative times/bothcauses exactcomponents/reload<=1e-10,CIFvalid,q<=2;reusedhashes andmetricsmatch;ensembleequalsmembermeans.',
        resources='2nodeSlurm,NFS,singlethreadCPU;preflightfreeRAMabout53GiBoliver/30GiBlemon;memoryslots24oliver/16lemon,2GiBfitaddresscap,4GiBensembleaddresscap;priorfitRSS<300MiB;SlurmownsaffinityincludingP/EandSMT. Independent fitandensemble stages andimmutable inputmanifests;rawandshrunkensembleexportsareseparateindependenttasks,predictionsemanticsunchanged.',
        privacy='Companydata/IDs/models/cuts/predictions/functionsprivate;code/config/hash/aggregatespublic.',
        decision='Preserve original96,currentpointer/mainbundle/coreandoldresults;no automaticreplacement,seed/weightoptimization,peryearselection orbranchmerge.')
    write(pub/'frozen_protocol.json',protocol);write(r/'input/protocol.json',protocol)
    write(r/'audit/source_hashes.json',sources);write(r/'audit/current_pointer.json',dict(sha256=sha(REPO/'CURRENT_POST_ST_MODEL.json')))
    write(pub/'run_location.json',dict(private_run=str(r),source_run=str(old),seed2026_source=str(stability)))
    print(r,flush=True)
if __name__=='__main__':main()
