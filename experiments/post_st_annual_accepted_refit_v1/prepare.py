"""Freeze already prepared accepted annual values and verify exact matched labels."""
from pathlib import Path
import argparse,datetime,hashlib,json,shutil,subprocess
import numpy as np
import pandas as pd
from experiments.post_st_prognosis_v1.common import arrays,sha,write,calendar_elapsed
from experiments.post_st_next_open_info_v1.analyze import outputs


def prepare(run):
    repo=Path(__file__).resolve().parents[2]
    protocol_path=Path(__file__).with_name('frozen_protocol.json');protocol=json.loads(protocol_path.read_text())
    write(run/'audit/protocol.json',protocol)
    write(run/'audit/pre_fit_freeze.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        protocol_sha256=sha(protocol_path),existing_results_exposed=True,new_fit_or_prediction_seen=False))
    prepared=Path(protocol['prepared_run']);old=Path(protocol['reference_run']);sources={}
    def source(p):
        p=Path(p);sources[str(p)]=sha(p);return p
    receipt=json.loads(source(repo/'experiments/post_st_annual_source_acceptance_v1/execution_receipt.json').read_text())
    for name,h in receipt['source_hashes'].items():assert sha(source(name))==h
    for name,h in receipt['private_input_sha256'].items():assert sha(source(prepared/name))==h
    assert sha(source(repo/'experiments/post_st_annual_source_acceptance_v1/source_policy.json'))==receipt['policy_sha256']
    shutil.copy2(repo/'experiments/post_st_annual_source_acceptance_v1/source_policy.json',run/'audit/source_policy.json')
    prior_freeze=json.loads(source(old/'audit/immutable_input_hashes.json').read_text())
    for name,h in prior_freeze.items():assert sha(old/'input'/name)==h
    summaries=[];groups=set();target_groups=set()
    for year in protocol['test_years']:
        v='announcement_fields';origin=prepared/'input'/str(year)/v;folder=run/'input/folds'/str(year)/v;folder.mkdir(parents=True)
        newplan=json.loads((origin/'prepared_configuration_NOT_RUN.json').read_text())
        oldplan=json.loads(source(old/'input/folds'/str(year)/v/'plan.json').read_text())
        for k in ['forest_common','forest_variants','standard_rsf','feature_names','pilot']:
            assert newplan[k]==oldplan[k],k
        tr=arrays(origin/'train_UNTRAINED_PRIVATE.npz');ev=arrays(origin/'eval_UNTRAINED_PRIVATE.npz')
        before_train=arrays(old/'input/folds'/str(year)/v/'train_PRIVATE.npz')
        before_test=arrays(source(old/'evaluation_input'/f'{year}_{v}_test_PRIVATE.npz'))
        for role,data,prior in [('fit',tr,before_train),('test',ev,before_test)]:
            meta=pd.read_parquet(source(prepared/'private'/f'{year}_{role}_metadata_PRIVATE.parquet'))
            np.testing.assert_array_equal(data['sample_ids'],meta.sample_id)
            for name in data:
                if name!='X':np.testing.assert_array_equal(data[name],prior[name])
            assert data['X'].shape==(len(meta),83) and meta.group.is_unique
            np.testing.assert_array_equal(data['X'][:,48:72],prior['X'][:,48:72])
            np.testing.assert_array_equal(data['X'][:,74:],prior['X'][:,74:])
            np.testing.assert_array_equal(data['analysis_time'],[calendar_elapsed(o,s) for o,s in zip(meta.origin,meta.stop)])
            assert meta.label_eligible.all() and (meta.financial_available<meta.origin).all()
            if role=='fit':
                assert (meta.origin<pd.Timestamp(f'{year}-01-01')).all() and (meta.stop<pd.Timestamp(f'{year}-01-01')).all()
            clocks=arrays(source(prepared/'private'/f'{year}_{role}_availability_PRIVATE.npz'))
            np.testing.assert_array_equal(clocks['sample_ids'],data['sample_ids'])
            dt=clocks['financial_available'];assert (np.isfinite(data['X'][:,:48])<=~np.isnat(dt)).all()
            assert not ((~np.isnat(dt))&(dt>=meta.origin.to_numpy('datetime64[D]')[:,None])).any()
            same=(data['X']==prior['X'])|(np.isnan(data['X'])&np.isnan(prior['X']))
            changed=int((~same).any(axis=1).sum())
            assert changed==({2023:38,2024:41}[year] if role=='fit' else {2023:3,2024:1}[year])
            meta.to_parquet(run/'evaluation_input'/f'{year}_{role}_metadata_PRIVATE.parquet',index=False)
            summaries.append(dict(test_year=year,role=role,companies=len(meta),changed_input_companies=changed,
                changed_input_cells=int((~same).sum()),termination=int((data['event']==1).sum()),
                recovery=int((data['event']==2).sum()),unknown=int(((data['event']==0)&(data['analysis_time']<24)).sum()),
                labels_changed=0,columns=83))
        assert not set(tr['groups'])&set(ev['groups'])
        groups.update(tr['groups']);target_groups.update(tr['groups'][tr['event']==1])
        shutil.copy2(origin/'train_UNTRAINED_PRIVATE.npz',folder/'train_PRIVATE.npz')
        np.savez_compressed(folder/'eval_PRIVATE.npz',X=ev['X'],sample_ids=ev['sample_ids'])
        shutil.copy2(origin/'eval_UNTRAINED_PRIVATE.npz',run/'evaluation_input'/f'{year}_{v}_test_PRIVATE.npz')
        plan={k:oldplan[k] for k in ['forest_common','forest_variants','standard_rsf','feature_names','pilot']}
        plan.update(test_year=year,variant=v,calibration_n=0,source_status=protocol['source_status'],
            source_policy_sha256=receipt['policy_sha256'],new_training_authorized=True,
            source_hashes={n:sha(folder/n) for n in ['train_PRIVATE.npz','eval_PRIVATE.npz']})
        write(folder/'plan.json',plan)
    assert len(groups)==345 and len(target_groups)==26
    references=[]
    for year,variant,model,seed,folder in outputs(old):
        if variant!='announcement_fields':continue
        for name,h in json.loads(source(folder/'hashes.json').read_text()).items():assert sha(source(folder/name))==h
        summary=json.loads((folder/'summary.json').read_text());assert summary['status']=='COMPLETE'
        references.append(dict(year=year,model=model,seed=seed,path=str(folder)))
    assert len(references)==12
    write(run/'audit/reference_models.json',references)
    delivered=json.loads(source(repo/'experiments/post_st_next_open_info_v1/execution_receipt.json').read_text())
    for name in ['metrics.csv','warning_metrics.csv','seed_stability.csv','ensemble_complexity.csv']:
        assert sha(source(old/'evaluation'/name))==delivered['result_sha256']['results/'+name]
    # The exact forest and worker implementation must match the old C fit.
    fit_commit=json.loads((old/'input/folds/2023/announcement_fields/plan.json').read_text())['code_commit']
    implementation=list((repo/'crforest').glob('*.py'))+[repo/'experiments/post_st_prognosis_v1'/x for x in ['worker.py','reference.R']]
    impl={}
    for p in implementation:
        rel=str(p.relative_to(repo));before=subprocess.check_output(['git','-C',str(repo),'show',fit_commit+':'+rel])
        assert hashlib.sha256(before).hexdigest()==sha(p),rel;impl[rel]=sha(p)
    for name in ['data_changes.csv','field_change_counts.csv','cohort_counts_UNCHANGED.csv']:
        shutil.copy2(source(prepared/'reports'/name),run/'reports'/name)
    pd.DataFrame(summaries).to_csv(run/'reports/matched_data_audit.csv',index=False)
    write(run/'audit/data_alignment.json',dict(status='PASS',source_status=protocol['source_status'],
        independent_original_PDF_verification=False,researcher_confirmation_is_controlling_source_policy=True,
        training_company_union=345,training_termination_union=26,test_companies=81,test_terminations=32,
        all_labels_times_groups_order_identical=True,all_nonfinancial_inputs_identical=True,
        model_configuration_and_feature_order_identical=True,model_implementation_identical_to_old_fit=fit_commit,
        implementation_sha256=impl,unchanged_source_hashes=sources,data=summaries,new_fits_at_prepare=0))
    for p,h in sources.items():assert sha(Path(p))==h
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);prepare(p.parse_args().run)
