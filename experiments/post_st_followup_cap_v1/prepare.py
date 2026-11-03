"""Rebuild training outcomes only; keep the frozen company/X/test alignment."""
from dataclasses import asdict
import datetime,shutil,subprocess
import pandas as pd
from crforest.partition import training_cut_dictionary
from experiments.post_st_prognosis_v1.prepare import OLD,AUDIT
from experiments.post_st_next_open_info_v1.prepare import label_record,packed
from .labels import rebuild
from .common import *


def prepare():
    run=run_path();public=REPO/'experiments'/NAME
    for n in ['input','audit','reports','evaluation_input','private','results']:(run/n).mkdir(exist_ok=False)
    sources={}
    def source(path):sources[str(path)]=sha(path);return path
    oldhash=load(source(SOURCE/'audit/immutable_input_hashes.json'))
    fp=source(SOURCE/'input/features.json');assert sha(fp)==oldhash['features.json']
    shutil.copy2(fp,run/'input/features.json')
    # Raw source contains other companies/dates but is filtered per historical
    # cutoff and fixed training group before any endpoint is reconstructed.
    events=pd.read_csv(source(OLD/'phase1_data_freeze_v1/03_event_history.csv'),dtype=str,keep_default_na=False)
    mapping=load(source(AUDIT/'assembly/group_mapping.json'))
    events['group']=events.Symbol.map(lambda x:mapping.get(x,x))
    events['announcement']=pd.to_datetime(events.adjudicated_announcement_date.where(
        events.adjudicated_announcement_date.ne(''),events.Annoudt),errors='raise')
    reviews=pd.read_csv(source(OLD/'phase1_data_freeze_v1/03_case_review.csv'),dtype=str,
        keep_default_na=False).set_index('event_id').to_dict('index')
    scope=load(source(OLD/'method_freeze_input_build_v2/evidence/source_scope_confirmation.json'))
    assert pd.Timestamp(scope['primary_snapshot_execution_bound'])>=pd.Timestamp('2025-01-01')
    cohorts=[];differences=[];alignment=[];reuses=[]
    for year in YEARS:
        cut=pd.Timestamp(f'{year}-01-01')
        meta=pd.read_parquet(source(METADATA_SOURCE/f'audit/{year}_training_metadata_PRIVATE.parquet'))
        tr=arrays(source(SOURCE/f'input/folds/{year}/train_PRIVATE.npz'))
        ev=arrays(source(SOURCE/f'input/folds/{year}/eval_PRIVATE.npz'))
        for role in ['train','eval']:
            rel=f'folds/{year}/{role}_PRIVATE.npz'
            assert sha(SOURCE/'input'/rel)==oldhash[rel]
        np.testing.assert_array_equal(meta.sample_id,tr['sample_ids'])
        np.testing.assert_array_equal(meta.group,tr['groups'])
        assert meta.group.is_unique and (meta.origin<cut).all()
        assert not set(tr['sample_ids'])&set(ev['sample_ids']) and set(ev)=={'X','sample_ids'}
        visible=events[events.group.isin(meta.group)&events.announcement.lt(cut)].copy()
        assert visible.announcement.lt(cut).all()
        grouped={k:g for k,g in visible.groupby('group')}
        visible.to_parquet(run/f'audit/{year}_past_events_PRIVATE.parquet',index=False)
        for arm,months in ARM_CAPS.items():
            rows=[]
            for row in meta.to_dict('records'):
                eg=grouped.get(row['group'],visible.iloc[:0])
                value=rebuild(row,eg,reviews,cut,months)
                if months==24:
                    legacy=label_record(row,eg,reviews,cut-pd.Timedelta(days=1))
                    assert value==legacy
                rows.append({**row,**value})
            mm=pd.DataFrame(rows)
            dd=packed(tr['X'],mm)
            for key in ['X','groups','sample_ids','year','h12','h24']:
                np.testing.assert_array_equal(dd[key],tr[key])
            if arm=='cap24':
                for key in dd:np.testing.assert_array_equal(dd[key],tr[key])
            assert (mm.stop<cut).all() and mm.time.gt(0).all()
            folder=run/f'input/folds/{year}/{arm}';folder.mkdir(parents=True)
            np.savez_compressed(folder/'train_PRIVATE.npz',**dd)
            np.savez_compressed(folder/'eval_PRIVATE.npz',**ev)
            mm.to_parquet(run/f'private/{year}_{arm}_training_metadata_PRIVATE.parquet',index=False)
            if months is not None:assert all(s<=o+pd.DateOffset(months=months)for o,s in zip(mm.origin,mm.stop))
            nterm=int(mm.event.eq(1).sum());nrec=int(mm.event.eq(2).sum())
            cohorts.append(dict(year=year,arm=arm,cap_months=months,training_companies=len(mm),
                termination_events=nterm,recovery_events=nrec,censored=int(mm.event.eq(0).sum()),
                termination_within12=int(((mm.event==1)&(dd['analysis_time']<=12)).sum()),
                recovery_within12=int(((mm.event==2)&(dd['analysis_time']<=12)).sum()),
                termination_after24=int(((mm.event==1)&(dd['analysis_time']>24)).sum()),
                recovery_after24=int(((mm.event==2)&(dd['analysis_time']>24)).sum()),
                median_followup_months=float(np.median(dd['analysis_time'])),
                maximum_followup_months=float(dd['analysis_time'].max()),
                before_cutoff=True,train_sha256=sha(folder/'train_PRIVATE.npz'),
                same_training_X=True,same_training_companies=True,same_evaluation_X=True))
            for seed in SEEDS:
                cfg=configuration(seed,arm)
                assert pairing(dd['groups'],cfg,74)==pairing(tr['groups'],cfg,74)
                for aa,bb in zip(training_cut_dictionary(dd['X'],cfg),training_cut_dictionary(tr['X'],cfg)):
                    np.testing.assert_array_equal(aa,bb)
                alignment.append(dict(year=year,arm=arm,seed=seed,bootstrap_identical=True,
                    feature_pools_identical=True,cut_dictionary_identical=True))
            for k,v in asdict(configuration(42,arm)).items():
                differences.append(dict(year=year,arm=arm,parameter=k,reference=json.dumps(v),actual=json.dumps(v),changed=False))
            differences.append(dict(year=year,arm=arm,parameter='training_followup_cap_months',reference='24',actual=json.dumps(months),changed=months!=24))
        # Existing test labels are copied unchanged and isolated from worker inputs.
        label=source(SOURCE/f'evaluation_input/{year}_test_PRIVATE.npz')
        shutil.copy2(label,run/f'evaluation_input/{year}_test_PRIVATE.npz')
    for rec in load(source(SOURCE/'audit/registry_formal.json')):
        p=Path(rec['path'])
        for n,h in load(source(p/'hashes.json')).items():assert sha(source(p/n))==h
        reuses.append({**rec,'arm':'cap24','reused':True})
    write(run/'audit/reused_formal.json',reuses)
    reused_ensemble=[]
    for rec in load(source(SOURCE/'audit/registry_ensemble.json')):
        p=Path(rec['path'])
        for n,h in load(source(p/'hashes.json')).items():assert sha(source(p/n))==h
        reused_ensemble.append({**rec,'arm':'cap24','reused':True})
    write(run/'audit/reused_ensemble.json',reused_ensemble)
    bootstrap=source(SOURCE/'private/bootstrap_indices_PRIVATE.npz')
    shutil.copy2(bootstrap,run/'private/bootstrap_indices_PRIVATE.npz')
    protocol=dict(name=NAME,status='FROZEN_BEFORE_NEW_FITS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        model='74-input drop9_support/shrunk, NOT the unchanged83 CURRENT pointer',years=YEARS,seeds=SEEDS,
        arms=ARM_CAPS,config=asdict(configuration(42,'cap24')),parent_shrinkage_strength=10,
        training='Exact same company cohorts and entry-time conservativeC X; first recovery/termination announcement chains; min(month cap if any, Dec31 before forecast year). Uncapped never means future or test outcomes.',
        primary='12-calendar-month first termination; report recovery too. Three-seed average before metrics; annual, pooled and within-year AUC.',
        secondary='Raw counterparts;24-month predictions/performance ONLY for cap24/cap36/uncapped2023/24. cap12 H24 NA because outside training window;2025H24 NA because evaluation follow-up insufficient.',
        formal_new_fits=27,reused_fits=9,pilots=dict(count=3,trees=4,year=2025,seed=42,arms=NEW_ARMS),
        endpoints='same evaluation cohorts/labels/horizons and same metrics as source; unknown never negative',
        metrics=['AUC','within_year_AUC','AP','IPCW_Brier','mean_risk','observed_incidence','Top20_capture','calibration_gap'],
        paired_uncertainty=dict(repeats=1000,seed=ANALYSIS_SEED,unit='company',strata='test_year',
            indices='reuse source',contrasts='each non24 arm minus cap24',condition='fixed model predictions only'),
        diagnostics='all-row/all-native-time/two-cause CIF legal and exact reconstruction<=1e-10; unique-company parentAJ replay; functions/pairs/cells, seed variability and timing',
        scope='exploratory previously exposed2023-25; fixed nine-deletion provenance retained; no selector, calibration, further search or automatic replacement',
        limitations='Inherited source PIT/version and retrospective event-chain review evidence; no new independent original-announcement audit',
        resources='singlethread independent tasks,4GiB limit,3600s timeout; two-node existing Slurm with managed affinity/NFS',
        failures='retain all; unresolved extended event chains block affected arm instead of silently deleting companies',
        privacy='company inputs/predictions/model parameters/functions private; aggregate reports only')
    write(public/'frozen_protocol.json',protocol);write(run/'input/protocol.json',protocol)
    write(run/'audit/source_hashes.json',sources)
    write(run/'audit/current_pointer.json',dict(path=str(REPO/'CURRENT_POST_ST_MODEL.json'),sha256=sha(REPO/'CURRENT_POST_ST_MODEL.json')))
    for name,data in [('COHORT_AUDIT',cohorts),('PARAMETER_DIFFERENCES',differences),('PAIRING_AUDIT',alignment)]:
        pd.DataFrame(data).to_csv(public/f'{name}.csv',index=False)
    print(pd.DataFrame(cohorts)[['year','arm','training_companies','termination_events','recovery_events','termination_after24','maximum_followup_months']].to_string(index=False))

if __name__=='__main__':prepare()
