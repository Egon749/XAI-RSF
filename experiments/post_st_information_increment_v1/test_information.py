import pytest
import numpy as np
from .common import features,pool_size,gate_arms
from .metrics import DESIGNS,score

def test_partition_and_matching_pools():
    f=features()
    assert f['info'].value_counts().to_dict()=={'F':45,'M':21,'A':7,'Q':1}
    assert (f.M1 & f.M2).equals(f.M0) and (f.M1|f.M2).equals(f.M3)
    assert f.loc[f['info']=='Q','english_name'].item()=='market_missing_coverage'
    assert np.array_equal(f.M2_numeric|f.M0_coverage,f.M2)
    assert [pool_size(n,'proportional') for n in (45,52,67,74)]==[15,18,23,25]

def test_source_gate_excludes_only_permitted_arms_in_current_state():
    gate=dict(financial_clock_pass=True,market_clock_pass=True,matched_rows_pass=True,audit_release_pass=False)
    assert set(gate_arms(gate))=={'M0','M2','M2_numeric','M0_coverage'}
    assert not {'M1','M3','M3_numeric'}&set(gate_arms(gate))
    with pytest.raises(ValueError):gate_arms(gate|dict(matched_rows_pass=False))

def test_negative_loss_utility_and_factorial_weights():
    y=np.array([0,0,1,1],bool);year=np.array([1,1,2,2]);ids=np.arange(4)
    assert score(y,np.array([.1,.2,.8,.9]),year,ids)[2]>score(y,np.full(4,.5),year,ids)[2]
    assert DESIGNS['complementarity']=={'M3':1,'M1':-1,'M2':-1,'M0':1}
