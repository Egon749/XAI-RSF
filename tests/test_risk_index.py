import numpy as np
import pytest
from experiments.post_st_risk_index_v1.index import risk_index,inverse_index,point_components,historical_thresholds,tiers


def test_fictional_hand_decomposition_baseline_not100_and_no_clipping():
    # Entirely fictional; no saved company/model inputs. CIF=.042+.016-.008+.010=.060.
    baseline,terms=point_components(.042,np.array([.016,-.008,.010]),.04)
    np.testing.assert_allclose(baseline,105,atol=1e-12)
    np.testing.assert_allclose(terms,[40,-20,25],atol=1e-12)
    score=risk_index(.060,.04)
    assert score==150 and baseline!=100
    assert abs(baseline+terms.sum()-score)<1e-10
    assert abs(inverse_index(score,.04)-.060)<1e-10


@pytest.mark.parametrize('factor',[1e-5,.5,2.,1234.])
def test_scale_invariance(factor):
    p=np.array([0.,.01,.03,.03,.08]);mu=.04
    np.testing.assert_allclose(risk_index(p*factor,mu*factor),risk_index(p,mu),atol=1e-10,rtol=0)
    np.testing.assert_array_equal(np.argsort(p,kind='stable'),np.argsort(risk_index(p,mu),kind='stable'))
    np.testing.assert_array_equal(np.diff(np.sort(p))==0,np.diff(np.sort(risk_index(p,mu)))==0)


def test_threshold_direction_and_ties():
    h=np.arange(1.,101.);q=historical_thresholds(h);t=tiers(h,q)
    assert [(t==s).sum() for s in ['low','medium','high']]==[50,30,20]
    assert tiers([q[0],q[1]],q).tolist()==['medium','high']
    assert (tiers(np.ones(30),historical_thresholds(np.ones(30)))=='high').all()


def test_support_and_invalid_denominators():
    assert historical_thresholds(np.arange(29.)) is None
    assert tiers([50,150],None).tolist()==['UNSUPPORTED','UNSUPPORTED']
    for mu in [0.,-1.,np.nan,np.inf]:
        with pytest.raises(ValueError,match='denominator'):risk_index(0.,mu)
        with pytest.raises(ValueError,match='denominator'):inverse_index(0.,mu)
    with pytest.raises(ValueError):historical_thresholds([np.nan]*30)


def test_score_array_save_reload(tmp_path):
    s=risk_index(np.array([.01,.03,.12]),.04)
    p=tmp_path/'fictional.npz';np.savez_compressed(p,score=s)
    with np.load(p) as z:np.testing.assert_array_equal(z['score'],s)
