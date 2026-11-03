"""Freeze the finite tree-count experiment without reading new test outcomes."""
import datetime,shutil,tempfile,subprocess
import pandas as pd
from crforest import CompetingRiskForest
from crforest.partition import training_cut_dictionary
from .common import *

def main():
    pub=REPO/'experiments'/NAME
    run=Path(tempfile.mkdtemp(prefix='tree_count_20261103_',dir=SOURCE/'private'))
    write(pub/'run_location.json',dict(private_run=str(run),source_run=str(SOURCE)))
    for n in ['input','audit','evaluation_input','private','reports','results']:(run/n).mkdir()
    frozen=load(SOURCE/'audit/immutable_input_hashes.json');sources={};cohorts=[]
    def src(path):sources[str(path)]=sha(path);return path
    for name in ['features.json',*[f'folds/{y}/{role}_PRIVATE.npz'for y in YEARS for role in ['train','eval']]]:
        file=src(SOURCE/'input'/name);assert sha(file)==frozen[name]
        dest=run/'input'/name;dest.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(file,dest)
    for y in YEARS:
        tr=arrays(run/f'input/folds/{y}/train_PRIVATE.npz');ev=arrays(run/f'input/folds/{y}/eval_PRIVATE.npz')
        meta=pd.read_parquet(src(SOURCE.parent/f'cause_support_20261102_run1/audit/{y}_training_metadata_PRIVATE.parquet'))
        np.testing.assert_array_equal(meta.sample_id,tr['sample_ids']);np.testing.assert_array_equal(meta.group,tr['groups'])
        cut=pd.Timestamp(f'{y}-01-01');assert (meta.origin<cut).all()and(meta.stop<cut).all()
        assert tr['X'].shape[1]==74 and set(ev)=={'X','sample_ids'}
        assert not set(tr['sample_ids'])&set(ev['sample_ids'])
        label=src(SOURCE/f'evaluation_input/{y}_test_PRIVATE.npz');shutil.copy2(label,run/f'evaluation_input/{y}_test_PRIVATE.npz')
        cohorts.append(dict(year=y,training_companies=len(tr['X']),training_termination=int((tr['event']==1).sum()),
            training_recovery=int((tr['event']==2).sum()),evaluation_companies=len(ev['X']),features=74,training_followup_months=24,pre_cutoff=True))
    old={}
    for r in load(src(SOURCE/'audit/registry_formal.json')):
        p=Path(r['path']);old[f'{r["year"]}|{r["seed"]}']=str(p)
        for name,h in load(src(p/'hashes.json')).items():assert sha(src(p/name))==h
        f=CompetingRiskForest.load(p/'raw/forest_PRIVATE.json')
        assert asdict(f.config)==asdict(configuration(r['seed'],32))
        tr=arrays(run/f'input/folds/{r["year"]}/train_PRIVATE.npz')
        assert load(p/'pairing.json')==pairing(tr['groups'],f.config,74)
        for a,b in zip(f.shared_cuts_,training_cut_dictionary(tr['X'],f.config)):np.testing.assert_array_equal(a,b)
    assert len(old)==9;write(run/'input/old_registry.json',old)
    ensemble=load(src(SOURCE/'audit/registry_ensemble.json'))
    for r in ensemble:
        p=Path(r['path'])
        for n,h in load(src(p/'hashes.json')).items():assert sha(src(p/n))==h
    write(run/'input/old_ensemble_registry.json',ensemble)
    shutil.copy2(src(SOURCE/'private/bootstrap_indices_PRIVATE.npz'),run/'private/bootstrap_indices_PRIVATE.npz')
    rows=[]
    for b in COUNTS:
        for k,v in asdict(configuration(42,b)).items():rows.append(dict(trees_per_seed=b,combined_trees=3*b,parameter=k,
            reference=json.dumps(asdict(configuration(42,32))[k]),actual=json.dumps(v),changed=v!=asdict(configuration(42,32))[k]))
    pd.DataFrame(rows).to_csv(pub/'PARAMETER_DIFFERENCES.csv',index=False)
    pd.DataFrame(cohorts).to_csv(pub/'COHORT_AUDIT.csv',index=False)
    protocol=dict(name=NAME,status='FROZEN_BEFORE_NEW_RESULTS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        years=YEARS,seeds=SEEDS,trees_per_seed=COUNTS,combined_trees=[3*b for b in COUNTS],config=asdict(configuration(42,32)),
        input='Identical frozen74 conservativeC; own pre-Jan1 snapshots, next-trading-day preopen origin; firsttermination1/recovery2,24calendar-month training followup; no new data preparation',
        only_change='n_trees; all old trees retained as exact prefixes, independent global tree IDs/bootstrap/feature pools/node RNG unchanged',
        computation=dict(reused_forests=9,reused_trees=288,new_trees=2016,new_grow_batches=63,trees_per_batch=32,
            assembled_256_tree_forests=9,pilot_new_trees=2,pilot='2024 seed42 IDs32–33, numerical/runtime only; no performance selection'),
        shrinkage='Each leaf mixes its own raw AJ and raw direct parent AJ; unique inbag companies/(companies+10); all causes/times sameweight; derive full forest once, prefixes unchanged',
        primary='12calendar-month termination, annual and pooled/within-year; three-seed mean before metrics',
        secondary='Both causes,raw versions,24months2023/24;2025H24NA; all seeds and functions, complexity/support/cost',
        evaluation='Existing same evaluator; Top20 ceil(.2n) separately peryear then aggregate; pooled metrics recomputed; unknown not negative',
        uncertainty=dict(repeats=1000,analysis_seed=ANALYSIS_SEED,unit='company',strata='year',draws='reuse cause-support',contrasts='each count minus32',conditional='fixed predictions only'),
        best_rule='No single automatic adoption or cross-year switching. Report each metric maximum and Pareto tradeoffs; main view12month within-yearAUC/AP/Top20/Brier; no new confirmed optimal-tree claim from exposed tests',
        convergence='Report successive prefix changes and vs256: predictionRMS/max,Spearman,Top20overlap; seedSD; no metric-driven extension beyond256',
        verification='Synthetic independent fullfit versus chunk assembly/prefix equality; exact old32raw/shrunk/predictions; all training+evalX/all native times/two causes exact functions<=1e-10, reload,CIF legal; hashes, counts and publication',
        resources='Existing two-nodeSlurm,NFS,disjoint CPUtree batches; grow3GiB process limit; assembly/export concurrency oliver12/lemon2 at3GiB each to fit available memory; no affinity override; both-host timings recorded',
        privacy='company IDs/data/predictions/functions/thresholds private; onlyaggregate public; preserve mainbundle/currentpointer',
        limitations='2023–2025 previously exposed;74-feature selection provenance and original disclosure-vintage limits inherited; exploratory, no model replacement')
    write(pub/'frozen_protocol.json',protocol);write(run/'input/protocol.json',protocol)
    write(run/'audit/source_hashes.json',sources);write(run/'audit/current_pointer.json',dict(sha256=sha(REPO/'CURRENT_POST_ST_MODEL.json')))
    print(run);print(pd.DataFrame(cohorts).to_string(index=False))

if __name__=='__main__':main()
