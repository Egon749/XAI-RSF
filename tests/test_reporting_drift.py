import numpy as np
from sklearn.metrics import roc_auc_score
from experiments.post_st_reporting_drift_v1.metrics import (
    stratified_metrics, paired_auc, paired_draws, alert_metrics,
)


def test_zero_event_year_contributes_no_pairs():
    z = stratified_metrics(np.array([0, 0, 0, 1], bool), [9, 8, 0, 1],
                           [2023, 2023, 2024, 2024], ['a', 'b', 'c', 'd'])
    assert z['within_year_auc'] == z['equal_year_auc'] == 1
    assert z['within_pairs'] == 1 and z['zero_pair_years'] == 1
    assert z['pooled_auc'] == 1/3


def test_ties_half_and_single_year_equals_ordinary_auc():
    y = np.array([1, 0, 1, 0], bool)
    p = np.array([.3, .3, .5, .1])
    z = stratified_metrics(y, p, [2023]*4, np.arange(4))
    assert z['within_year_auc'] == roc_auc_score(y, p) == paired_auc(y, p)[0]
    assert z['within_year_auc'] == .875
    assert z['within_minus_pooled'] == 0


def test_all_single_class_is_undefined_not_zero():
    z = stratified_metrics(np.zeros(4, bool), np.arange(4), [1, 1, 2, 2], np.arange(4))
    assert np.isnan(z['within_year_auc']) and np.isnan(z['equal_year_auc'])
    assert z['within_pairs'] == 0 and z['zero_pair_years'] == 2


def test_pair_weighting_not_simple_average():
    y = np.array([1, 0, 1, 0, 0, 0], bool)
    p = [1, 0, 0, 1, 2, 3]
    z = stratified_metrics(y, p, [1, 1, 2, 2, 2, 2], np.arange(6))
    assert z['within_year_auc'] == .25 and z['equal_year_auc'] == .5


def test_draws_match_legacy_stream_and_alerts_annual_quota():
    sizes = {2023: 6, 2024: 7, 2025: 8}
    got = paired_draws(sizes)
    rng = np.random.default_rng(2026110201)
    for year, n in sizes.items():
        np.testing.assert_array_equal(got[year], rng.integers(n, size=(1000, n)))
    y = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1], bool)
    z = alert_metrics(y, np.arange(10), [1]*5+[2]*5, np.arange(10))
    assert z['checked'] == 2 and z['captured'] == 1 and z['lift'] == 2.5


def test_calendar_risk_stops_at_first_event_and_handles_jan1():
    import pandas as pd
    from experiments.post_st_reporting_drift_v1.analyze import calendar_table
    f = pd.DataFrame(dict(company=['a', 'b'], cohort=[2023, 2024],
                          origin=pd.to_datetime(['2023-12-30', '2024-01-01']),
                          stop=pd.to_datetime(['2024-01-01', '2024-01-03']), event=[1, 2]))
    t = calendar_table(f)
    assert t.person_days.sum() == 4
    assert t.loc[t.calendar_year.eq(2024), 'termination'].sum() == 1
    assert t.termination.sum() == t.recovery.sum() == 1
    assert t[t.company.eq('a')].calendar_year.max() == 2024


def test_suppression_keeps_zero_distinct_from_small_unknown():
    import pandas as pd
    from experiments.post_st_reporting_drift_v1.analyze import suppress_calendar
    f = pd.DataFrame(dict(termination_events=[0, 2, 5], recovery_events=[1, 0, 8],
                          termination_rate_per100_person_months=[0., 2., 5.],
                          recovery_rate_per100_person_months=[1., 0., 8.]))
    r = suppress_calendar(f)
    assert r.termination_events.iloc[0] == 0
    assert np.isnan(r.termination_events.iloc[1])
    assert np.isnan(r.termination_rate_per100_person_months.iloc[1])
    assert r.termination_events.iloc[2] == 5
