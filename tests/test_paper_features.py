import numpy as np
from experiments.post_st_paper_features_v1.common import PAPER_NAMES,PAPER_IDS,SELECTED,transform,configuration,pairing,reference_configuration


def test_paper_subset_preserves_missingness_and_labels_without_placeholder_columns():
    names=[f'other_{j}' for j in range(83)]
    for j,k in enumerate(PAPER_IDS):names[j]=SELECTED[k][0]
    x=np.arange(3*83,dtype=float).reshape(3,83);j=PAPER_IDS.index(53)
    x[:,j]=[np.log(1000),np.nan,np.log(2500)]
    d=dict(X=x,time=np.array([1.,2.,3.]),event=np.array([0,1,2]),groups=np.array(['a','b','c']))
    z=transform(d,names)
    assert len(PAPER_NAMES)==54 and len(PAPER_IDS)==20 and z['X'].shape==(3,20)
    np.testing.assert_allclose(z['X'][:,j],[1000,np.nan,2500]);np.testing.assert_array_equal(np.isnan(z['X']),np.isnan(x[:,:20]))
    for k in ['time','event','groups']:np.testing.assert_array_equal(z[k],d[k])
    np.testing.assert_array_equal(d['X'],x)


def test_paper_candidate_cap_preserves_company_bootstrap_randomness():
    groups=np.repeat(np.arange(17).astype(str),2)
    a=pairing(groups,reference_configuration(42,'shallow_parent'),83);b=pairing(groups,configuration(42),20)
    assert configuration(42).tree_feature_count==20 and configuration(42).mtry==75
    for x,y in zip(a,b):
        assert x['bootstrap_sha256']==y['bootstrap_sha256'] and x['multiplicity_sha256']==y['multiplicity_sha256']
