"""Verify existing83 and74 snapshots, freeze the one missing factorial cell."""
from dataclasses import asdict,replace
import datetime,shutil
import numpy as np
import pandas as pd
from crforest import CompetingRiskForest
from crforest.partition import training_cut_dictionary
from experiments.post_st_paper_features_v1.common import references
from experiments.post_st_drop9_v1.common import kept_columns,reduced,DROP_NAMES
from .common import *

def prepare():
    public=REPO/'experiments'/NAME
    assert not (public/'frozen_protocol.json').exists()
    for n in ['input','audit','reports','evaluation_input','private']:(RUN/n).mkdir(parents=True,exist_ok=True)
    sources={};cohorts=[];reuse=[];diffs=[];pairchecks=[]
    def register(p):sources[str(p)]=sha(p)
    frozen=load(SOURCE/'audit/immutable_input_hashes.json')
    features=load(SOURCE/'input/features.json');register(SOURCE/'input/features.json')
    keep=kept_columns([r['feature_name'] for r in features]);assert len(keep)==74
    write(RUN/'input/features.json',[features[j] for j in keep]);write(RUN/'audit/keep_PRIVATE.json',keep)
    for year in YEARS:
        name='post_st_drop9_v1' if year==2024 else 'post_st_drop9_transport_v1'
        oldrun=Path(load(REPO/'experiments'/name/'execution_receipt.json')['private_run'])
        oldinput=oldrun/'input' if year==2024 else oldrun/f'input/folds/{year}'
        parts={}
        for role in ['train','eval']:
            source=SOURCE/f'input/folds/{year}/{role}_PRIVATE.npz'
            assert sha(source)==frozen[str(source.relative_to(SOURCE/'input'))];register(source)
            data=reduced(arrays(source),keep);oldpath=oldinput/f'{role}_PRIVATE.npz';register(oldpath);old=arrays(oldpath)
            # Historical eval may contain labels; new workers receive X/IDs only.
            for k in data:np.testing.assert_array_equal(data[k],old[k])
            folder=RUN/f'input/folds/{year}';folder.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(folder/f'{role}_PRIVATE.npz',**data);parts[role]=data
        tr,ev=parts['train'],parts['eval'];assert set(ev)=={'X','sample_ids'}
        meta_path=SOURCE/f'audit/{year}_training_metadata_PRIVATE.parquet';register(meta_path)
        meta=pd.read_parquet(meta_path);np.testing.assert_array_equal(meta.sample_id,tr['sample_ids'])
        cut=pd.Timestamp(f'{year}-01-01');assert (meta.origin<cut).all() and (meta.stop<cut).all()
        assert (meta.origin>pd.to_datetime(meta.entry_announcement)).all()
        assert not set(tr['sample_ids'])&set(ev['sample_ids'])
        label=SOURCE/f'evaluation_input/{year}_test_PRIVATE.npz';register(label)
        shutil.copy2(label,RUN/f'evaluation_input/{year}_test_PRIVATE.npz') # copy, not used by fits
        cohorts.append(dict(year=year,training_companies=len(tr['X']),termination_events=int((tr['event']==1).sum()),recovery_events=int((tr['event']==2).sum()),evaluation_companies=len(ev['X']),features=74,pre_cutoff=True,old74_arrays_identical=True,train_sha256=sha(folder/'train_PRIVATE.npz'),eval_sha256=sha(folder/'eval_PRIVATE.npz')))
        for seed,path in references(year,'drop9',REPO).items():
            old=CompetingRiskForest.load(path/'forest_PRIVATE.json');cfg=configuration(seed)
            changed={k for k,v in asdict(cfg).items() if v!=asdict(old.config)[k]};assert changed=={'split_min_target_groups'}
            assert pairing(tr['groups'],cfg,74)==pairing(tr['groups'],old.config,74)
            for new_cuts,old_cuts in zip(training_cut_dictionary(tr['X'],cfg),training_cut_dictionary(tr['X'],old.config)):
                np.testing.assert_array_equal(new_cuts,old_cuts)
            for k,v in asdict(cfg).items():diffs.append(dict(year=year,seed=seed,parameter=k,old74=json.dumps(asdict(old.config)[k]),new74_support=json.dumps(v),changed=k in changed))
            pairchecks.append(dict(year=year,seed=seed,trees=32,bootstrap_identical=True,feature_pools_identical=True,retained_cut_dictionary_identical=True))
            for version in ['raw','shrunk']:
                pp=path.parent/version
                for f in pp.rglob('*'):
                    if f.is_file():register(f)
                reuse.append(dict(year=year,seed=seed,arm='drop9',version=version,path=str(pp),forest_sha256=sha(pp/'forest_PRIVATE.json')))
    for r in load(SOURCE/'audit/registry_formal.json'):
        if r['arm'] not in ['baseline','support']:continue
        p=Path(r['path'])
        for n,h in load(p/'hashes.json').items():assert sha(p/n)==h;register(p/n)
        for v in ['raw','shrunk']:reuse.append(dict(year=r['year'],seed=r['seed'],arm='reference83' if r['arm']=='baseline' else 'support83',version=v,path=str(p/v),forest_sha256=sha(p/v/'forest_PRIVATE.json')))
    bootstrap=SOURCE/'private/bootstrap_indices_PRIVATE.npz';register(bootstrap);shutil.copy2(bootstrap,RUN/'private/bootstrap_indices_PRIVATE.npz')
    protocol=dict(name=NAME,status='FROZEN_BEFORE_NEW_FITS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_commit='fcf98aac4c68ac4ac1cadac4227262e0937e94a3',
        years=YEARS,seeds=SEEDS,new_arm='drop9_support',config=asdict(configuration(42)),drop_names=DROP_NAMES,
        design='Missing cell of83/74 inputs x support off/on. Only split_min_target_groups changes0 to2 relative to historical74. Same signed sum weights1,1; own direct raw-parentAJ10, unique inbag firms. Child cause1 support must be0 or>=2, never fake events.',
        preserved='Original83,74,83support, current pointer and every earlier result. No calibration, cutoff changes, feature selection, outcome-based tuning, or automatic adoption.',
        fits=dict(formal=9,pilot=1,pilot_trees=4,pilot_year=2024,pilot_seed=42,pilot_purpose='numerical/runtime only, never select using outer performance'),
        derived='9 own-parent shrunk models plus pilot derivative;raw retained. Compile6 96-tree ensembles, no extra fit.',
        data='Exact conservativeC with fixed9 deletion, pre-Jan1 snapshots, nexttradingdaypreopen origin, firstfinaltermination1 versusfirstrecovery2, cap24calendar months. Historical field clock/vintage limitations inherited.2023 selection list pre2024 provenance retained; all2023-25 exploratory.',
        primary='shrunk drop9_support minus shrunk drop9, annual supported24month termination2023/24 and12month2025; report all12/24 causes and years plus pooled and within-year AUC. Mean3seed predictions before metrics.',
        comparisons=['new74support-old74','new74support-old83support','new74support-old83','(74support-74)-(83support-83)'],
        secondary='raw counterparts; seed variation, exact main/pair/cell counts, search/leaf support, compute costs, feature/interaction stability. No independent training uncertainty claim.',
        metrics='Annual+pooled AUC/AP/IPCW Brier/mean/observed/Top20. Annual ceil(.2n) quotas then sum. Pooled recalculated, within-year AUC weighted by event-control pairs.2025H24 performanceNA, predictions retained. Unknown never negatives.',
        uncertainty=dict(repeats=1000,seed=ANALYSIS_SEED,unit='company',stratify='evaluation year',indices='reuse cause-support original',condition='fixed predictions;95percentile paired differences, undefined counts; exploratory multiple contrasts'),
        verification='Full existing pytest including independentR; all train+evalrows/all native times/bothcauses: nonnegative,nondecreasing,sum<=1,q<=2,full explicit component reconstruction andsave/reload<=1e-10. Independent replay scores/guard/parentAJ. Paired74bootstrap+features+cuts.',
        resources='Independent singlethreadCPU fits,4GiB/process,3600seconds, existing two-nodeSlurm/NFS; no duplicated whole scripts; verify outputs and publication.',
        privacy='All company records, predictions, parameters and cuts private. Public aggregated metrics/diagnostics/hashes only; no rankings authorization extension.')
    write(public/'frozen_protocol.json',protocol);write(RUN/'input/protocol.json',protocol)
    write(RUN/'audit/source_hashes.json',sources);write(RUN/'audit/reused_artifacts.json',reuse)
    write(RUN/'audit/current_pointer.json',dict(path=str(REPO/'CURRENT_POST_ST_MODEL.json'),sha256=sha(REPO/'CURRENT_POST_ST_MODEL.json')))
    for name,rows in [('COHORT_AUDIT',cohorts),('PARAMETER_DIFFERENCES',diffs),('PAIRING_AUDIT',pairchecks)]:pd.DataFrame(rows).to_csv(public/f'{name}.csv',index=False)
    print(json.dumps(cohorts))
