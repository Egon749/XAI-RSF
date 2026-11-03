from dataclasses import asdict,replace
import numpy as np
from experiments.post_st_drop9_support_v1.common import configuration,pairing
from experiments.post_st_drop9_v1.common import kept_columns,reduced,DROP_NAMES
from experiments.post_st_ablation_v1.common import configuration as old_configuration

def test_support_changes_only_one_training_parameter():
    for seed in [42,101,202]:
        a=old_configuration(seed,'shallow_parent');b=configuration(seed)
        assert {k for k in asdict(a) if asdict(a)[k]!=asdict(b)[k]}=={'split_min_target_groups'}
        assert b.split_min_target_groups==2 and b.split_target_cause==1
        assert pairing(np.repeat(np.arange(20),2),a,74)==pairing(np.repeat(np.arange(20),2),b,74)

def test_fixed_deletion_preserves_order_labels_and_missing_values():
    names=[f'other_{j}' for j in range(74)]+DROP_NAMES
    x=np.arange(166.,dtype=float).reshape(2,83);x[1,2]=np.nan
    data={'X':x,'event':np.array([1,2])};got=reduced(data,kept_columns(names))
    np.testing.assert_array_equal(got['X'],x[:,:74]);np.testing.assert_array_equal(got['event'],data['event'])
    assert not np.shares_memory(got['X'],x)
