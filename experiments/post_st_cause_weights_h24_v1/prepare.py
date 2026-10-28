"""Reuse audited full-history h24 folds and independently replay their clocks."""
from pathlib import Path
import argparse,datetime,json
import numpy as np
import pandas as pd
from experiments.post_st_prognosis_v1.common import arrays,sha,write,calendar_elapsed,classify_first
from experiments.post_st_train2023_v1.prepare import relabel


def calendar_arrays(data,meta):
    result=dict(data)
    np.testing.assert_array_equal(result['sample_ids'],meta.sample_id.to_numpy(str))
    np.testing.assert_array_equal(result['groups'],meta.group.to_numpy(str))
    assert meta.group.is_unique and meta.label_eligible.all()
    assert (meta.source_coverage_end>=meta.stop).all()
    result['analysis_time']=np.array([calendar_elapsed(o,s) for o,s in zip(meta.origin,meta.stop)])
    for horizon in [12,24]:
        np.testing.assert_array_equal(result[f'h{horizon}'],[(o+pd.DateOffset(months=horizon)-o).days for o in meta.origin])
        np.testing.assert_array_equal(result['time']>=result[f'h{horizon}'],result['analysis_time']>=horizon)
    assert (result['time']<=result['h24']).all()
    return result


def prepare(run):
    repo=Path(__file__).resolve().parents[2]
    protocol=json.loads(Path(__file__).with_name('frozen_protocol.json').read_text())
    write(run/'audit/protocol.json',protocol)
    write(run/'audit/pre_fit_freeze.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        protocol_sha256=sha(Path(__file__).with_name('frozen_protocol.json')),prior_outcomes_exposed=True,
        no_new_test_performance_seen=True))
    source0=Path(protocol['reference_runs']['2023'])
    receipt24=json.loads((repo/'experiments/post_st_train2023_v1/execution_receipt.json').read_text())
    endpoints=source0/'private/endpoint_ledger_PRIVATE.parquet';gaps=source0/'private/chain_diagnostics_PRIVATE.parquet'
    sources={}
    for path in [endpoints,gaps]:
        assert sha(path)==receipt24['source_sha256'][str(path)];sources[str(path)]=sha(path)
    counts=[];features=None
    for year in protocol['test_years']:
        source=Path(protocol['reference_runs'][str(year)])
        project='post_st_prognosis_v1' if year==2023 else 'post_st_train2023_v1'
        receipt=json.loads((repo/'experiments'/project/'execution_receipt.json').read_text())
        for name,digest in receipt['model_input_hashes'].items():
            path=source/'input'/name;assert sha(path)==digest;sources[str(path)]=digest
        planpath=source/'input/plan.json';sources[str(planpath)]=sha(planpath)
        oldplan=json.loads(planpath.read_text())
        if features is None:features=oldplan['feature_names']
        else:assert features==oldplan['feature_names']
        tr0=arrays(source/'input/train_PRIVATE.npz');tm=pd.read_parquet(source/'input/train_metadata_PRIVATE.parquet')
        cutoff=pd.Timestamp(f'{year}-01-01');assert tm.entry_year.max()==year-1 and (tm.stop<cutoff).all()
        filters=[('announcement','<',cutoff),('group','in',tm.group.tolist())]
        end=pd.read_parquet(endpoints,columns=['group','announcement','type'],filters=filters)
        gap=pd.read_parquet(gaps,columns=['group','announcement','resolved_for_first_A_X'],filters=filters)
        replay=relabel(tm,end,gap,cutoff)
        for name in ['time','event','stop','label_eligible']:
            np.testing.assert_array_equal(replay[name],tm[name])
        assert (tm.financial_available<tm.origin).all() and (tm.market_available<tm.origin).all()
        tr=calendar_arrays(tr0,tm)
        ev0=arrays(source/'input/eval_PRIVATE.npz');em0=pd.read_parquet(source/'input/eval_metadata_PRIVATE.parquet')
        selected=ev0['year']==year;em=em0.loc[selected].reset_index(drop=True)
        ev=calendar_arrays({k:v[selected] for k,v in ev0.items()},em)
        assert not set(tr['groups'])&set(ev['groups'])
        assert len(tr['X'])=={2023:309,2024:345}[year] and len(ev['X'])=={2023:36,2024:45}[year]
        # Test endpoint chain eligibility and source caps were already audited; independently replay first A/X.
        ee=pd.read_parquet(endpoints,columns=['group','announcement','type'],filters=[('group','in',em.group.tolist())])
        for row in em.itertuples():
            eei=ee[ee.group==row.group]
            pairs=[(r.announcement,1 if r.type.endswith('X') else 2) for r in eei.itertuples()]
            z=classify_first(row.origin,pairs,row.effective_followup_cap,months=24)
            assert not z['conflict'] and z['event']==row.event and z['stop']==row.stop
        for role,data,meta in [('fit',tr,tm),('test',ev,em)]:
            counts.append(dict(test_year=year,role=role,companies=len(meta),entry_year_min=int(meta.entry_year.min()),
                entry_year_max=int(meta.entry_year.max()),information_cutoff_exclusive=f'{year}-01-01' if role=='fit' else '2026-09-04',
                termination=int((data['event']==1).sum()),recovery=int((data['event']==2).sum()),
                complete_no_event=int(((data['event']==0)&(data['analysis_time']>=24)).sum()),
                unknown=int(((data['event']==0)&(data['analysis_time']<24)).sum())))
        target=run/'evaluation_input'/str(year);target.mkdir()
        np.savez_compressed(target/'test_PRIVATE.npz',**ev);em.to_parquet(target/'test_metadata_PRIVATE.parquet',index=False)
        tm.to_parquet(target/'train_metadata_PRIVATE.parquet',index=False)
        for variant,weights in protocol['weights'].items():
            folder=run/'input/folds'/str(year)/variant;folder.mkdir(parents=True)
            np.savez_compressed(folder/'train_PRIVATE.npz',**tr)
            np.savez_compressed(folder/'eval_PRIVATE.npz',X=ev['X'],sample_ids=ev['sample_ids'])
            plan=json.loads(json.dumps({k:oldplan[k] for k in ['forest_common','forest_variants','standard_rsf','feature_names']}))
            plan['forest_variants']['xai_q2']['cause_weights']=weights
            plan.update(pilot={'trees':4},variant=variant,test_year=year,calibration_n=0,test_n=len(ev['X']),
                source_hashes={name:sha(folder/name) for name in ['train_PRIVATE.npz','eval_PRIVATE.npz']})
            write(folder/'plan.json',plan)
    pd.DataFrame(counts).to_csv(run/'reports/cohort_counts.csv',index=False)
    write(run/'audit/data_alignment.json',dict(status='PROVISIONAL',unchanged_source_hashes=sources,
        source_arrays_identical=True,original_financial_vintages_PIT_certified=False,rolling_records_added=False,
        calibration_refitted=False,test2025_used=False,source_coverage_end='2026-09-03',counts=counts,
        columns=75,feature_order_sha256=__import__('hashlib').sha256(json.dumps(features).encode()).hexdigest(),
        prior_chain_eligibility_replayed=True,labels_recomputed_to_independently_verify_existing_folds=True))
    print(pd.DataFrame(counts).to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();prepare(a.run)
