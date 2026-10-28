"""Three-way temporal split from frozen as-of features and audited endpoints."""
from pathlib import Path
import argparse,json,shutil,datetime
import numpy as np
import pandas as pd
from experiments.post_st_prognosis_v1.common import arrays,sha,write,classify_first,calendar_elapsed

SOURCE=Path('/home/oliver/MainDiskData/xai_rsf_post_st_train2024_h12_20261027_j0yr95uc')
ORIGINAL=Path('/home/oliver/MainDiskData/xai_rsf_post_st_prognosis_v1_20261027_u6l00ebk')


def prepare(run):
    protocol=json.loads(Path(__file__).with_name('frozen_protocol.json').read_text())
    write(run/'audit/protocol.json',protocol)
    write(run/'audit/pre_fit_freeze.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        protocol_sha256=sha(Path(__file__).with_name('frozen_protocol.json')),
        prior_outcomes_exposed=True,selection_using_new_model_performance=False))
    sources=[SOURCE/'input/train_PRIVATE.npz',SOURCE/'input/train_metadata_PRIVATE.parquet',
        SOURCE/'input/plan.json',SOURCE/'private/endpoint_ledger_PRIVATE.parquet',
        ORIGINAL/'input/eval_metadata_PRIVATE.parquet',SOURCE/'audit/data_alignment.json']
    hashes={str(p):sha(p) for p in sources}
    oldplan=json.loads((SOURCE/'input/plan.json').read_text());data=arrays(SOURCE/'input/train_PRIVATE.npz')
    m=pd.read_parquet(SOURCE/'input/train_metadata_PRIVATE.parquet')
    assert len(m)==390 and np.array_equal(m.sample_id.to_numpy(str),data['sample_ids']) and m.group.is_unique
    full_audit=pd.read_parquet(ORIGINAL/'input/eval_metadata_PRIVATE.parquet')
    assert full_audit.label_eligible.all() and m.label_eligible.all()
    assert set(full_audit.sample_id)==set(m.loc[m.entry_year.isin([2023,2024]),'sample_id'])
    assert (m.financial_available<m.origin).all() and (m.market_available<m.origin).all()
    e=pd.read_parquet(SOURCE/'private/endpoint_ledger_PRIVATE.parquet')
    e=e[e.group.isin(m.group)];endpoints={k:[(r.announcement,1 if r.type.endswith('X') else 2) for r in g.itertuples()]
        for k,g in e.groupby('group')}
    # Verify that restoring full test follow-up retains the previously audited first A/X.
    for row in full_audit.itertuples():
        z=classify_first(row.origin,endpoints.get(row.group,[]),row.effective_followup_cap,months=24)
        assert z['event']==row.event and z['stop']==row.stop and not z['conflict']
    counts=[]
    for year in protocol['test_years']:
        folder=run/'input/folds'/str(year);folder.mkdir(parents=True)
        testing=run/'evaluation_input'/str(year);testing.mkdir(parents=True)
        parts={}
        for role,lo,hi,cutoff in [('fit',2012,year-3,f'{year-2}-01-01'),
                ('calibration',year-2,year-1,f'{year}-01-01'),('test',year,year,'2026-09-04')]:
            select=m.entry_year.between(lo,hi).to_numpy();dd=m.loc[select].reset_index(drop=True).copy()
            cap=pd.Timestamp(cutoff)-pd.Timedelta(days=1)
            labels=[classify_first(r.origin,endpoints.get(r.group,[]),cap,months=12) for r in dd.itertuples()]
            for k in ['time','event','stop','reason','conflict']:dd[k]=[v[k] for v in labels]
            assert not dd.conflict.any() and dd.time.gt(0).all() and (dd.time<=dd.h12).all() and (dd.stop< pd.Timestamp(cutoff)).all()
            a=dict(X=data['X'][select],time=dd.time.to_numpy(float),event=dd.event.to_numpy(int),
                groups=dd.group.to_numpy(str),sample_ids=dd.sample_id.to_numpy(str),year=dd.entry_year.to_numpy(int),
                h12=dd.h12.to_numpy(int),label_eligible=np.ones(len(dd),bool),
                analysis_time=np.array([calendar_elapsed(o,s) for o,s in zip(dd.origin,dd.stop)]))
            assert np.array_equal(a['time']>=a['h12'],a['analysis_time']>=12)
            dest=testing if role=='test' else folder
            np.savez_compressed(dest/f'{role}_PRIVATE.npz',**a)
            dd.to_parquet(dest/f'{role}_metadata_PRIVATE.parquet',index=False)
            parts[role]=a
            counts.append(dict(test_year=year,role=role,entry_year_min=lo,entry_year_max=hi,
                information_cutoff_exclusive=cutoff,companies=len(dd),termination=int((dd.event==1).sum()),
                recovery=int((dd.event==2).sum()),complete_no_event=int(dd.reason.eq('complete_horizon').sum()),
                unknown=int(dd.reason.eq('administrative_censoring').sum())))
        for a,b in [('fit','calibration'),('fit','test'),('calibration','test')]:
            assert not set(parts[a]['groups'])&set(parts[b]['groups'])
        # Existing worker's prediction input contains no test outcomes at all.
        shutil.copy2(folder/'fit_PRIVATE.npz',folder/'train_PRIVATE.npz')
        np.savez_compressed(folder/'eval_PRIVATE.npz',X=np.vstack([parts[k]['X'] for k in ['calibration','test']]),
            sample_ids=np.concatenate([parts[k]['sample_ids'] for k in ['calibration','test']]))
        plan={k:oldplan[k] for k in ['forest_common','forest_variants','standard_rsf','feature_names']}
        plan.update(variants=protocol['models'],seeds=protocol['seeds'],pilot={'trees':4},test_year=year,
                    calibration_n=len(parts['calibration']['X']),test_n=len(parts['test']['X']))
        plan['source_hashes']={p.name:sha(p) for p in folder.iterdir() if p.is_file()}
        write(folder/'plan.json',plan)
    pd.DataFrame(counts).to_csv(run/'reports/cohort_counts.csv',index=False)
    shutil.copy2(SOURCE/'reports/feature_manifest.csv',run/'reports/feature_manifest_inherited.csv')
    assert all(sha(p)==v for p,v in hashes.items())
    write(run/'audit/data_alignment.json',dict(status='PROVISIONAL',source_sha256=hashes,counts=counts,
        feature_rows_reused_exactly=True,original_inputs=75,test_outcomes_absent_from_worker_input=True,
        announcement_origin_unchanged=True,original_financial_vintages_PIT_certified=False,
        rule_features_added=False,source_coverage_end='2026-09-03',prior_test_audit_records=81,
        group_overlap=0,test2025_used=False,all_endpoints_recomputed=True))
    print(pd.DataFrame(counts).to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);prepare(p.parse_args().run)
