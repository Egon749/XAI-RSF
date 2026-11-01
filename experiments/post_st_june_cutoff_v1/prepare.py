from pathlib import Path
from dataclasses import asdict
import argparse,datetime,io,json,subprocess,tarfile
import numpy as np
import pandas as pd
from crforest.partition import shared_quantile_cuts,tree_feature_subset
from experiments.post_st_prognosis_v1.prepare import OLD,AUDIT
from experiments.post_st_leaf_shallow_v1.explain_company import LABELS
from .common import *


def prepare(run,repo):
    for n in ['input','evaluation_input','audit','reports','private','results']:(run/n).mkdir(exist_ok=False)
    sources={}
    def source(p):sources[str(p)]=sha(p);return p
    original=json.loads(source(BASE/'audit/immutable_input_hashes.json').read_text());data=[]
    for name in ['train','eval']:
        p=source(BASE/f'input/folds/2025/{name}_PRIVATE.npz')
        assert sha(p)==original[f'folds/2025/{name}_PRIVATE.npz'];data.append(arrays(p))
    names=json.loads(source(BASE/'input/folds/2025/plan.json').read_text())['feature_names']
    meta=pd.read_parquet(source(BASE/'private/shifted_cohort_PRIVATE.parquet')).reset_index(drop=True)
    ids=np.concatenate([d['sample_ids'] for d in data]);X=np.vstack([d['X'] for d in data])
    ix=pd.Index(ids).get_indexer(meta.sample_id);assert len(meta)==443 and meta.group.is_unique and (ix>=0).all()
    X=X[ix];assert X.shape==(443,83) and not np.isinf(X).any()
    clock=arrays(source(BASE/'private/financial_availability_PRIVATE.npz'))
    np.testing.assert_array_equal(clock['sample_ids'],meta.sample_id)
    assert (np.isfinite(X[:,:48])<=~np.isnat(clock['dates'])).all()
    assert not ((~np.isnat(clock['dates']))&(clock['dates']>=meta.origin.to_numpy(dtype='datetime64[D]')[:,None])).any()
    info=pd.read_parquet(source(BASE/'private/information_alignment_PRIVATE.parquet'))
    np.testing.assert_array_equal(info.sample_id,meta.sample_id)
    assert (info.financial_available.isna()|(info.financial_available<meta.origin)).all()
    assert (info.market_available.isna()|(info.market_available<meta.origin)).all()
    assert (meta.entry_announcement<meta.origin).all()
    event=pd.read_csv(source(OLD/'phase1_data_freeze_v1/03_event_history.csv'),dtype=str,keep_default_na=False)
    mp=json.loads(source(AUDIT/'assembly/group_mapping.json').read_text());event['group']=event.Symbol.map(lambda x:mp.get(x,x))
    event['announcement']=pd.to_datetime(event.adjudicated_announcement_date.where(event.adjudicated_announcement_date.ne(''),event.Annoudt),errors='raise')
    groups={k:g for k,g in event.groupby('group')}
    reviews=pd.read_csv(source(OLD/'phase1_data_freeze_v1/03_case_review.csv'),dtype=str,keep_default_na=False).set_index('event_id').to_dict('index')
    decisions=pd.DataFrame(json.loads(source(AUDIT/'audit_v4_observed_first/event_decisions.json').read_text()))
    accepted=decisions[decisions.observed_first_accepted]
    assert pd.to_datetime(accepted.corrected_event_date).max()<pd.Timestamp('2026-01-01')
    scope=json.loads(source(OLD/'method_freeze_input_build_v2/evidence/source_scope_confirmation.json').read_text())
    assert scope['primary_snapshot_execution_bound']=='2026-09-03' and OBSERVED_END<=pd.Timestamp(scope['primary_snapshot_execution_bound'])
    current=json.loads(source(repo/'CURRENT_POST_ST_MODEL.json').read_text());assert sha(repo/'CURRENT_POST_ST_MODEL.json')==REFERENCE_SHA
    from experiments.post_st_leaf_shallow_v1.analyze import artifacts
    from experiments.post_st_2025_shallow_parent_v1.analyze import outputs
    for y,arm,seed,p in artifacts(Path(current['private_run'])):
        if arm!='shallow_parent':continue
        r=next(z for z in current['saved_models'] if z['test_year']==y and z['seed']==int(seed))
        for n,k in [('forest_PRIVATE.json','forest_sha256'),('predictions_PRIVATE.npz','predictions_sha256'),('functions_PRIVATE.npz','functions_sha256')]:assert sha(source(p/n))==r[k]
    oldhash=pd.read_csv(source(repo/'experiments/post_st_2025_shallow_parent_v1/results/model_hashes.csv'))
    for model,seed,p in outputs(BASE):
        if model!='xai_q2':continue
        r=oldhash[(oldhash.model==model)&(oldhash.seed==int(seed))].iloc[0]
        for n,k in [('forest_PRIVATE.json','forest_sha256'),('predictions_PRIVATE.npz','prediction_sha256'),('functions_PRIVATE.npz','functions_sha256')]:assert sha(source(p/n))==r[k]
    keep=kept_columns(names);audit=[];diff=[];pairrows=[];fmap=[];coverage=[];exclusions=[]
    selection=repo/'experiments/post_st_drop9_v1/frozen_protocol.json';assert json.loads(source(selection).read_text())['removed_names']==DROP_NAMES
    for year in YEARS:
        cutoff=pd.Timestamp(f'{year}-06-30');trainmask,testmask=masks(meta.origin,cutoff)
        tm=relabel(meta[trainmask],groups,reviews,cutoff);good=tm.label_eligible&tm.time.gt(0)
        for r in tm.loc[~good].to_dict('records'):exclusions.append(dict(cutoff=str(cutoff.date()),role='train',**r))
        tm=tm.loc[good];em=relabel(meta[testmask],groups,reviews,OBSERVED_END,endpoint_months=12)
        assert em.label_eligible.all() and em.time.gt(0).all()
        tr=packed(X[tm.index],tm);ev=packed(X[em.index],em)
        assert not set(tr['groups'])&set(ev['groups'])
        assert (tm.stop<=cutoff).all() and (tm.origin<=cutoff).all()
        assert (em.origin>cutoff).all() and (em.origin<=cutoff+pd.DateOffset(months=12)).all()
        complete=cutoff+pd.DateOffset(months=12)<=ENTRY_COVERAGE_END;status=evaluation_status(len(em),complete)
        fold=run/f'input/folds/{year}';fold.mkdir(parents=True)
        for arm,indices in [('reference83',list(range(83))),('drop9',keep)]:
            d=fold/arm;d.mkdir();np.savez_compressed(d/'train_PRIVATE.npz',**reduced(tr,indices))
            np.savez_compressed(d/'eval_PRIVATE.npz',X=ev['X'][:,indices],sample_ids=ev['sample_ids'])
            write(d/'plan.json',dict(year=year,cutoff=str(cutoff.date()),arm=arm,feature_names=[names[j] for j in indices],
                original_indices=indices,configuration=asdict(configuration(42,'shallow_parent')),evaluation_status=status))
        tm.to_parquet(run/f'evaluation_input/{year}_train_metadata_PRIVATE.parquet',index=False)
        em.to_parquet(run/f'evaluation_input/{year}_test_metadata_PRIVATE.parquet',index=False)
        np.savez_compressed(run/f'evaluation_input/{year}_test_PRIVATE.npz',**ev)
        for j,n in enumerate(names):fmap.append(dict(year=year,original_index=j,reduced_index=keep.index(j) if j in keep else -1,name=n,label=LABELS[n],removed=n in DROP_NAMES))
        before,after=shared_quantile_cuts(tr['X'],5),shared_quantile_cuts(tr['X'][:,keep],5)
        for j,k in enumerate(keep):np.testing.assert_array_equal(before[k],after[j])
        for seed in SEEDS:
            cfg=configuration(seed,'shallow_parent');left,right=pairing(tr['groups'],cfg,83),pairing(tr['groups'],cfg,74)
            for i,(l,r) in enumerate(zip(left,right)):
                assert l['bootstrap_sha256']==r['bootstrap_sha256'] and l['multiplicity_sha256']==r['multiplicity_sha256']
                pairrows.append(dict(year=year,seed=seed,tree=i,bootstrap_sha256=l['bootstrap_sha256'],multiplicity_sha256=l['multiplicity_sha256'],reference_features_sha256=l['features_sha256'],drop9_features_sha256=r['features_sha256']))
            for j,n in enumerate(names):coverage.append(dict(year=year,seed=seed,name=n,removed=n in DROP_NAMES,
                reference_eligible=sum(j in tree_feature_subset(83,25,seed,i) for i in range(32)),
                drop9_eligible=sum(j in [keep[int(k)] for k in tree_feature_subset(74,25,seed,i)] for i in range(32))))
        audit.append(dict(cutoff_year=year,training_cutoff=str(cutoff.date()),test_entry_from=str((cutoff+pd.Timedelta(days=1)).date()),
            test_entry_through=str((cutoff+pd.DateOffset(months=12)).date()),training_companies=len(tm),
            training_target_events=int((tr['event']==1).sum()),training_recovery_events=int((tr['event']==2).sum()),
            training_target_events_within12=int(((tr['event']==1)&(tr['analysis_time']<=12)).sum()),
            evaluation_companies_available=len(em),entry_cohort_complete=bool(complete),evaluation_status=status,
            maximum_training_stop=str(tm.stop.max().date()),train_test_groups_disjoint=True))
        for k,v in asdict(configuration(42,'shallow_parent')).items():diff.append(dict(year=year,parameter=k,reference=json.dumps(v),drop9=json.dumps(v),changed=False))
        diff.extend([dict(year=year,parameter=k,reference=a,drop9=b,changed=a!=b) for k,a,b in [('input_columns','83','74'),('parent_AJ_strength','10','10'),('training_horizon_cap','24','24'),('evaluation_months','12','12')]])
    # One company originates exactly on2023-06-30 and has zero observable
    # follow-up at that cutoff; it cannot enter a positive-time survival fit.
    assert [r['training_companies'] for r in audit]==[343,390,442]
    assert len(exclusions)==1 and exclusions[0]['reason']=='no_positive_verified_followup'
    assert [r['evaluation_companies_available'] for r in audit]==[46,52,1]
    pd.DataFrame(exclusions).to_parquet(run/'private/exclusions_PRIVATE.parquet',index=False)
    for n,d in [('cohort_audit',audit),('configuration_diff',diff),('feature_map',fmap),('tree_pairing',pairrows),('feature_eligibility',coverage)]:pd.DataFrame(d).to_csv(run/f'reports/{n}.csv',index=False)
    protocol=dict(name=NAME,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        cutoffs=[f'{y}-06-30' for y in YEARS],cutoff_semantics='User-selected inclusive training information date; not a common landmark or test entry end',
        training='All eligible historical entries on/before cutoff; entry-time predictors unchanged; outcomes rederived through cutoff;24-calendar-month cap retained',
        testing='New entries strictly after cutoff and on/before cutoff+12calendar months; each evaluated12calendar months after its own next-trading-day entry origin',
        observation_cutoff=str(OBSERVED_END.date()),entry_coverage_end=str(ENTRY_COVERAGE_END.date()),
        data_blocker_2025='Audited first-financial-ST entries end2025; no audited2026 cohort or matched input set. One frozen test company; suppress public performance and single-company risk/label summaries.',
        original_source_execution_bound='2026-09-03',frozen_features='conservative C83, paired fixed drop9 subset74',removed_names=DROP_NAMES,
        selection_source_sha256=sha(selection),configuration=asdict(configuration(42,'shallow_parent')),seeds=SEEDS,arms=ARMS,
        shrinkage_strength=10,case_time_weight=1,forest_fits=18,derived_shrunk_models=18,reused_forest_fits=0,pilot_fits=0,
        fitting_budget='3 cutoffs x2 inputs x3 seeds;32trees each; no additional selection or calibration',
        prediction_aggregation='Average three seed company CIFs before12-month scoring; save all seeds and both causes',
        metric_policy='AUC,AP,IPCW Brier,mean risk,observed incidence,Top20; incomplete2026 cohort suppresses ALL public test metrics for2025 including single-company risk/label summaries',
        top20='ceil(.20*n) within each post-cutoff entry window; descending risk then sample_id; unknowns remain unknown',
        uncertainty=dict(repeats=1000,seed=2026110130,unit='company',paired=True,scope='complete supported windows only; fixed forecasts; undefined metrics counted'),
        verification='All train/eval rows, native times and both causes; q2 full sum error<=1e-10; reload,CIF constraints,raw direct-parentAJ10 and company multiplicity replay',
        pairing='Identical company bootstrap, same25 eligible count;83->74 changes balanced candidate identities and blocks4->3',
        resources='18 independent single-thread CPU tasks through two-node Slurm/NFS,4GiB process caps',current_reference_preserved=True,status='FROZEN_BEFORE_NEW_FIT')
    public=repo/'experiments'/NAME/'frozen_protocol.json'
    if public.exists():
        old=json.loads(public.read_text());assert {k:v for k,v in old.items() if k!='source_commit'}=={k:v for k,v in protocol.items() if k!='source_commit'}
    else:write(public,protocol)
    write(run/'input/protocol.json',protocol);write(run/'audit/source_hashes.json',sources)
    write(run/'audit/data_alignment.json',dict(status='PASS_WITH_2025_EVALUATION_BLOCKER',folds=audit,input_vectors_reused_exactly=443,
        field_availability_verified=True,training_labels_rebuilt_at_each_cutoff=True,source_model_files_preserved=27,exclusions=len(exclusions)))
    write(run/'audit/source_search.json',dict(approved_registry_tool='unavailable: tunnel client disconnected; local authoritative files inspected',
        audited_accepted_entry_counts_by_year=pd.to_datetime(accepted.corrected_event_date).dt.year.value_counts().sort_index().to_dict(),
        raw_2026_event_records=int(event.announcement.dt.year.eq(2026).sum()),raw_records_do_not_certify_first_ST_eligibility=True,
        missing_upstream=['2026 observed-first financial/audit ST adjudication and nonfinancial eligibility','matched83-column original entry-time inputs/provenance for approved2026 entries','sufficient12-month follow-up for full2025-07 to2026-06 entry window']))
    print(json.dumps(audit),flush=True)


def freeze(run,repo):
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo).strip()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    with tarfile.open(fileobj=io.BytesIO(subprocess.check_output(['git','archive',commit],cwd=repo))) as f:f.extractall(run/'input/code',filter='data')
    tasks=[dict(kind='cpu',argv=['python3',str(run/'input/code/experiments'/NAME/'launch.py'),'--year',str(y),'--arm',a,'--seed',str(s)],
        cwd=str(run/'input/code'),timeout=3600,storage=dict(source=str(run/'input'),mode='nfs',results=str(run/'results/formal')))
        for y in YEARS for a in ARMS for s in SEEDS]
    with (run/'formal.jsonl').open('x') as f:f.write(''.join(json.dumps(t)+'\n' for t in tasks))
    (run/'formal.py').write_text('# Eighteen independent matched cutoff fits.\n');write(run/'formal.py.hetero.json',dict(independent=True,manifest='formal.jsonl'))
    write(run/'audit/freeze.json',dict(commit=commit,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),tasks=18,protocol_sha256=sha(run/'input/protocol.json'),manifest_sha256=sha(run/'formal.jsonl')))
    write(run/'audit/immutable_input_hashes.json',{str(p.relative_to(run/'input')):sha(p) for p in (run/'input').rglob('*') if p.is_file()});print(commit)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--stage',choices=['prepare','freeze'],required=True);a=p.parse_args()
    (prepare if a.stage=='prepare' else freeze)(a.run,Path(__file__).resolve().parents[2])
