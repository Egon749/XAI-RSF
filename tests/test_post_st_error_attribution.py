import numpy as np
import pytest
from experiments.post_st_error_attribution_v1.analyze import brier_parts, case_control_auc, period, rule_evidence


def test_calendar_bands_do_not_assign_applicable_rule():
    assert period('2024-04-11') == 'before_20240412'
    assert period('2024-04-12') == '20240412_to_20240429'
    assert period('2024-04-30') == '20240430_to_20241029'
    assert period('2024-10-30') == 'from_20241030'
    # A new procedural version cited beside the old explicit trigger must not
    # turn three hundred million into five hundred million.
    label,_,cap=rule_evidence('market_cap_threshold','股票收盘市值均低于3亿元，股票上市规则(2023年8月修订)，股票上市规则(2024年修订)')
    assert label=='explicit_3e8_market_cap_trigger' and cap=='3'
    assert rule_evidence('price_below_one_yuan','股票上市规则(2025年修订)')[0]=='preexisting_one_yuan_trigger'
    with pytest.raises(ValueError):
        rule_evidence('market_cap_threshold','未明确门槛')


def test_other_termination_causes_are_not_controls():
    event=np.array([1,1,2,0]);p=np.array([.3,.9,.2,.3])
    assert case_control_auc(event,p,[True,False,False,False])==.75
    with pytest.raises(ValueError):
        case_control_auc(event,p,[False,False,True,False])
    assert np.isnan(case_control_auc([1],[.5],[True]))


def test_brier_algebra_is_not_new_calibrated_prediction():
    y=np.array([0,1,1,0]);p=np.array([.1,.2,.3,.4]);before=p.copy()
    parts=brier_parts(y,p)
    assert parts['squared_mean_bias']==pytest.approx(.0625)
    assert parts['brier']==pytest.approx(parts['squared_mean_bias']+parts['centered_residual'])
    assert parts['brier']==pytest.approx(parts['event_case_loss']+parts['other_outcome_loss'])
    np.testing.assert_array_equal(p,before)
