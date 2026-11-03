"""Stage exact saved prefixes and freeze protocol before evaluating local grid."""
import datetime,tempfile,shutil,subprocess
import pandas as pd
from crforest import CompetingRiskForest
from experiments.post_st_tree_count_v1.nested import prefix
from .common import *

def main():
    PREVIOUS=previous_path()
    pub=REPO/'experiments'/NAME
    run=Path(tempfile.mkdtemp(prefix='tree_local_20261103_',dir=SOURCE/'private'))
    write(pub/'run_location.json',dict(private_run=str(run),source_run=str(PREVIOUS)))
    for n in ['input','audit','evaluation_input','private','reports','results']:(run/n).mkdir()
    sources={}
    def record(p):sources[str(p)]=sha(p);return p
    old_hashes=load(PREVIOUS/'audit/input_hashes.json')
    for name in ['features.json','old_registry.json','old_ensemble_registry.json',*[f'folds/{y}/{r}_PRIVATE.npz'for y in YEARS for r in ['train','eval']]]:
        p=record(PREVIOUS/'input'/name);assert sha(p)==old_hashes[name]
        dst=run/'input'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
    for y in YEARS:
        # Opaque byte copying only; no evaluation outcomes are opened before freezing.
        shutil.copy2(record(PREVIOUS/f'evaluation_input/{y}_test_PRIVATE.npz'),run/f'evaluation_input/{y}_test_PRIVATE.npz')
    shutil.copy2(record(PREVIOUS/'private/bootstrap_indices_PRIVATE.npz'),run/'private/bootstrap_indices_PRIVATE.npz')
    prior=load(record(PREVIOUS/'reports/data_validation.json'));assert prior['status']=='PASS'
    registry=load(record(PREVIOUS/'audit/registry_assemble.json'));assert len(registry)==9
    rows=[]
    for rec in registry:
        y,s=rec['year'],rec['seed'];src=Path(rec['path']);out=run/f'input/parents/{y}/{s}';out.mkdir(parents=True)
        hashes=load(record(src/'hashes.json'));counts=load(record(src/'tree_counts_PRIVATE.json'))
        assert sha(src/'tree_counts_PRIVATE.json')==hashes['tree_counts_PRIVATE.json']
        for v in ['raw','shrunk']:
            p=record(src/f'{v}_PRIVATE.json');assert sha(p)==hashes[p.name]
            full=CompetingRiskForest.load(p);f=prefix(full,40,counts);f.save(out/f'{v}_PRIVATE.json')
            assert json.dumps(f.trees_)==json.dumps(full.trees_[:40])
            assert len(f.trees_)==40 and f.config.path_budget==2
        write(out/'tree_counts_PRIVATE.json',counts[:40])
        rows.append(dict(year=y,seed=s,source_trees=256,staged_prefix=40,exact_raw_and_shrunk=True,new_fits=0,
            source_raw_sha256=hashes['raw_PRIVATE.json'],source_shrunk_sha256=hashes['shrunk_PRIVATE.json']))
    pd.DataFrame(rows).to_csv(pub/'SOURCE_MODEL_AUDIT.csv',index=False)
    shutil.copy2(REPO/'experiments/post_st_tree_count_v1/COHORT_AUDIT.csv',pub/'COHORT_AUDIT.csv')
    parameters=[]
    for b in COUNTS:
        for k,v in asdict(configuration(42,b)).items():
            ref=asdict(configuration(42,32))[k];assert k=='n_trees' or v==ref
            parameters.append(dict(trees_per_seed=b,combined_trees=3*b,parameter=k,reference=json.dumps(ref),actual=json.dumps(v),changed=v!=ref))
    pd.DataFrame(parameters).to_csv(pub/'PARAMETER_DIFFERENCES.csv',index=False)
    protocol=dict(name=NAME,status='FROZEN_BEFORE_LOCAL_GRID_RESULTS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        years=YEARS,seeds=SEEDS,trees_per_seed=COUNTS,combined_trees=[3*b for b in COUNTS],
        range_interpretation='80–120 TOTAL; equal trees and equal weights across three seeds permit81,84,...,120;80 is not an admissible equal-size total',
        config=asdict(configuration(42,32)),training_followup_months=24,features=74,
        forest_fits=0,calibration_fits=0,new_trees=0,source_forests=9,source_trees_per_forest=256,
        prefix='Exact original indices0..b-1 of each saved raw/shrunk forest; no tree selection or further shrinkage; equal-seed mean equals equal-tree mean',
        selection='Exploratory12month pooled within-year termination AUC descending; ties within1e-12 then annualTop20 total descending, pooledAP descending, Brier ascending, total trees ascending; never auto-adopt or switch by testyear',
        primary='12months termination, year-specific/within-year/pooled; three-seed mean first',
        secondary='Both causes,raw,all seeds,24months2023/2024;2025H24 performanceNA',
        uncertainty=dict(repeats=1000,seed=ANALYSIS_SEED,draws='Original cause-support company bootstrap, year-stratified; each local count minus32; fixed predictions only'),
        verification='Alltrain+eval/all native times/two causes; reconstruction,reload<=1e-10,CIF nonnegative/monotone/sum<=1; exact32metrics reproduction; independent sklearn and pair-count AUC',
        tasks=dict(total_new_exports=156,pilot_exports=1,remaining_exports=155,reused32_exports=12,seed_models=126,ensemble_models=42,raw_and_shrunk_representations=336),
        resources='Existing two-nodeSlurm/NFS; stage only first40 saved trees; single-thread exports;oliver24/lemon8 concurrent slots,2GiB process limit; largest120-tree pilot gate peakRSS<=600MiB before batch',
        privacy='Companies,X,model parameters,predictions,functions private; aggregates and hashes only',
        limitations='Follow-up designed after exposed broad-tree results.2023–2025 reused for local optimization, not independent validation. Original PIT/version/74-selection limitations inherited. Preserve96,mainbundle,currentpointer; no additional grid,seed/window/depth/feature search')
    write(pub/'frozen_protocol.json',protocol);write(run/'input/protocol.json',protocol)
    write(run/'audit/source_hashes.json',sources)
    write(run/'audit/current_pointer.json',dict(sha256=sha(REPO/'CURRENT_POST_ST_MODEL.json')))
    print(run,flush=True)

if __name__=='__main__':main()
