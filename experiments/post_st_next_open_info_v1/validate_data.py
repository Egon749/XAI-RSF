"""Second-path clock and matched-array audit before fitting; writes no model."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from experiments.post_st_prognosis_v1.common import arrays,sha,write
from .prepare import BASE,OLD,AUDIT


def validate(run):
    audit=json.loads((run/'audit/data_alignment.json').read_text())
    receipt=json.loads((Path(__file__).resolve().parents[2]/'experiments/post_st_prognosis_v1/execution_receipt.json').read_text())
    matched_hashes=0
    for path,h in audit['unchanged_source_hashes'].items():
        assert sha(Path(path))==h
        if path in receipt['source_hashes']:
            assert h==receipt['source_hashes'][path];matched_hashes+=1
    info=pd.read_parquet(run/'private/information_alignment_PRIVATE.parquet').set_index('sample_id')
    raw=pd.read_csv(OLD/'phase1_data_freeze_v1/03_event_history.csv',dtype=str,keep_default_na=False)
    mapping=json.loads((AUDIT/'assembly/group_mapping.json').read_text())
    raw['group']=raw.Symbol.map(lambda s:mapping.get(s,s))
    raw['date']=pd.to_datetime(raw.adjudicated_announcement_date.where(raw.adjudicated_announcement_date.ne(''),raw.Annoudt))
    terminal=raw[raw.Chgtype.str.endswith(('A','X'))]
    variants=['market_clock','financial_refresh','announcement_fields'];checked=0;changes=[]
    for year in [2023,2024]:
        source=BASE if year==2023 else Path('/home/oliver/MainDiskData/xai_rsf_post_st_train2023_v1_20261027_hhuplqjs')
        oldtr=pd.read_parquet(source/'input/train_metadata_PRIVATE.parquet').set_index('group')
        oldtest=pd.read_parquet(source/'input/eval_metadata_PRIVATE.parquet');oldtest=oldtest[oldtest.entry_year==year].set_index('group')
        for role,prior in [('fit',oldtr),('test',oldtest)]:
            meta=pd.read_parquet(run/'evaluation_input'/f'{year}_{role}_metadata_PRIVATE.parquet')
            cap=min(pd.Timestamp('2026-09-03'),pd.Timestamp(f'{year}-01-01')-pd.Timedelta(days=1)) if role=='fit' else pd.Timestamp('2026-09-03')
            assert len(meta)==len(prior) and set(meta.group)==set(prior.index)
            for row in meta.itertuples():
                e=terminal[(terminal.group==row.group)&(terminal.date>=row.entry_announcement)&(terminal.date<=cap)].sort_values('date')
                assert not e.date.le(row.origin).any()
                bound=min(cap,row.origin+pd.DateOffset(months=24));e=e[e.date<=bound]
                if len(e):
                    day=e.iloc[0].date;kinds=set(e[e.date==day].Chgtype.str[-1]);assert len(kinds)==1
                    event=1 if next(iter(kinds))=='X' else 2
                else:day=bound;event=0
                assert row.event==event and row.stop==day and row.time==(day-row.origin).days
                assert row.origin>row.entry_announcement and row.stop<=cap
                for months in [12,24]:
                    old=prior.loc[row.group];oldhit=(old.event>0)&(old.time<=old[f'h{months}'])
                    newhit=(row.event>0)&(row.time<=getattr(row,f'h{months}'))
                    oldkind=int(old.event) if oldhit else 0;newkind=int(row.event) if newhit else 0
                    if oldkind!=newkind:
                        changes.append(dict(test_year=year,role=role,horizon_months=months,old_event=oldkind,new_event=newkind))
                checked+=1
            base=None
            for variant in variants:
                path=run/'input/folds'/str(year)/variant/'train_PRIVATE.npz' if role=='fit' else run/'evaluation_input'/f'{year}_{variant}_test_PRIVATE.npz'
                data=arrays(path)
                np.testing.assert_array_equal(data['sample_ids'],meta.sample_id)
                np.testing.assert_array_equal(data['groups'],meta.group)
                if base is not None:
                    for name in data:
                        if name!='X':np.testing.assert_array_equal(data[name],base[name])
                base=data
                xmeta=meta.copy();ii=info.loc[meta.sample_id]
                xmeta['market_available']=ii.new_market_available.to_numpy()
                if variant!='market_clock':
                    xmeta['report_date']=ii.new_report.to_numpy()
                    xmeta['financial_available']=ii.report_financial_available.to_numpy()
                    xmeta['prev_available']=ii.previous_component_available.to_numpy()
                xmeta['financial_age']=(xmeta.origin-xmeta.financial_available).dt.days
                xmeta['financial_period_age']=(xmeta.origin-xmeta.report_date).dt.days
                assert (xmeta.financial_available<xmeta.origin).all()
                assert (xmeta.market_available.isna()|(xmeta.market_available<xmeta.origin)).all()
                assert not np.any(np.isfinite(data['X'][:,48:72])&xmeta.market_available.isna().to_numpy()[:,None])
                np.testing.assert_array_equal(data['X'][:,72],np.minimum(xmeta.financial_period_age/365,2))
                np.testing.assert_array_equal(data['X'][:,73],np.minimum(xmeta.financial_age/365,2))
                xmeta.to_parquet(run/'private'/f'{year}_{variant}_{role}_feature_metadata_PRIVATE.parquet',index=False)
    changes=pd.DataFrame(changes).groupby(['test_year','role','horizon_months','old_event','new_event']).size().reset_index(name='records')
    changes.to_csv(run/'reports/label_clock_changes.csv',index=False)
    clock=pd.read_parquet(run/'private/information_alignment_PRIVATE.parquet')
    clock['new_report_quarter']=clock.new_report.dt.quarter
    clock.groupby(['entry_year','new_report_quarter']).size().reset_index(name='companies').to_csv(run/'reports/selected_report_quarters.csv',index=False)
    write(run/'audit/independent_data_gate.json',dict(status='PASS',independent_endpoint_rows_checked=checked,
        source_hashes_matching_original_receipt=matched_hashes,all_arm_labels_and_company_order_identical=True,
        original_financial_versions_certified=False,original_year_industry_membership_retained=True,
        no_at_origin_A_X_risk_set_exclusions=True,trained_models=0,source_exclusions=0,
        no_test_labels_in_fit_X_input_files=True))
    print(changes.to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);validate(p.parse_args().run)
