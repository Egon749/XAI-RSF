import numpy as np
import pandas as pd
from experiments.post_st_annual_source_acceptance_v1.prepare import confirmed_annual_frame
from experiments.post_st_next_open_info_v1.prepare import financial_record


def example():
    return pd.DataFrame(dict(Stkcd=['example']*3,Accper=['2022-12-31','2023-12-31','2024-03-31'],
        prev_period=['2021-12-31','2022-12-31','2023-03-31'],
        ann=pd.to_datetime(['2023-04-20','2024-04-30','2024-04-30']),
        available=pd.to_datetime(['2025-01-01','2024-08-01','2024-09-01']),
        prev_available=pd.to_datetime([None,'2025-01-01',None]),
        correction_0=pd.to_datetime(['2025-01-01','2024-08-01','2024-09-01']),
        ratio=[1.,2.,3.],revenue_growth=[np.nan,1.,np.nan]))


def test_confirmation_accepts_annual_and_rejoins_prior_component_without_editing_values():
    source=example();f=confirmed_annual_frame(source)
    assert source.loc[1,'available']==pd.Timestamp('2024-08-01')
    assert f.loc[1,'available']==pd.Timestamp('2024-04-30')
    assert f.loc[1,'prev_available']==pd.Timestamp('2023-04-20')
    np.testing.assert_array_equal(f.correction_0,source.correction_0)
    np.testing.assert_array_equal(f.ratio,source.ratio)
    x,dates,meta=financial_record(f,['ratio','revenue_growth'],'2024-05-06')
    np.testing.assert_array_equal(x,[2.,1.]);assert meta['report_date']==pd.Timestamp('2023-12-31')
    assert (dates==np.datetime64('2024-04-30')).all()


def test_source_confirmation_does_not_accept_future_or_missing_original_announcement():
    f=example();f.loc[1,'ann']=pd.Timestamp('2024-05-07')
    x,_,m=financial_record(confirmed_annual_frame(f),['ratio','revenue_growth'],'2024-05-06')
    assert m['report_date']==pd.Timestamp('2022-12-31') and x[0]==1
    f.loc[1,'ann']=pd.NaT
    x,_,m=financial_record(confirmed_annual_frame(f),['ratio','revenue_growth'],'2024-05-06')
    assert m['report_date']==pd.Timestamp('2022-12-31')


def test_nonannual_report_keeps_conservative_clock_and_newer_eligible_quarter_stays_preferred():
    f=example();f.loc[2,'available']=pd.Timestamp('2024-04-30')
    revised=confirmed_annual_frame(f)
    assert revised.loc[2,'available']==f.loc[2,'available']
    x,_,meta=financial_record(revised,['ratio','revenue_growth'],'2024-05-06')
    assert meta['report_date']==pd.Timestamp('2024-03-31') and x[0]==3


def test_confirmation_does_not_override_unavailable_prior_component():
    f=example();f.loc[0,'ann']=pd.Timestamp('2024-05-07')
    x,dates,_=financial_record(confirmed_annual_frame(f),['ratio','revenue_growth'],'2024-05-06')
    assert x[0]==2 and np.isnan(x[1]) and np.isnat(dates[1])
