import numpy as np
import pytest
from sklearn.metrics import roc_auc_score
from experiments.post_st_seed_ensemble_v1.common import average_predictions,ARMS,tasks

def test_average_predictions_is_not_average_auc():
    records=[dict(sample_ids=np.array(['a','b']),times=np.array([365.]),cif=np.array(p)[:,None,None])for p in [[.3,.1],[.2,.25],[.4,.1]]]
    p=average_predictions(records)[:,0,0]
    np.testing.assert_allclose(p,[.3,.15])
    assert roc_auc_score([1,0],p)==1
    assert np.mean([roc_auc_score([1,0],z['cif'][:,0,0])for z in records])==pytest.approx(2/3)
    bad={**records[1],'sample_ids':np.array(['b','a'])}
    with pytest.raises(AssertionError):average_predictions([records[0],bad])

def test_fixed_memberships_no_duplicate_seed():
    assert ARMS['reference96']==[42,101,202]
    assert ARMS['requested7_224']==[1009,503,17,202,101,419,300]
    assert ARMS['requested8_256']==ARMS['requested7_224']+[42]
    assert all(len(s)==len(set(s))for s in ARMS.values())
    assert len(tasks())==6
