import numpy as np
import pandas as pd
import pytest
from crforest import CIFExpansion
from experiments.post_st_level_coverage_v1.affine import forest_envelope
from tests.test_crforest import hand_forest
from .module import age_exposures, age_model, piecewise_cif, mapping, transformed, certify


def sample():
    return pd.DataFrame(dict(group=['a','b','c','d'], origin=pd.to_datetime(['2020-01-01','2021-01-01','2021-06-01','2023-01-02']),
                             stop=pd.to_datetime(['2021-01-01','2022-02-01','2022-06-01','2023-06-01']), event=[1,2,0,1]))


def test_calendar_age_boundaries_and_censoring():
    hd,he,rd,re=age_exposures(sample(),'2023-01-01')
    # Leap-year anniversary belongs in early bin, event exactly one year is early.
    np.testing.assert_array_equal(hd,[[366,0],[365,31],[365,0]])
    np.testing.assert_array_equal(he.sum(0),[[1,0],[0,1]])
    np.testing.assert_array_equal(re.sum(0),[[1,0],[0,1]])
    np.testing.assert_array_equal(rd,[[0,0],[365,31],[365,0]])


def test_cutoff_isolation_and_empty_event_bins():
    m=sample();a=age_model(m,'2023-01-01')
    m.loc[3,'event']=2;m.loc[3,'stop']=pd.Timestamp('2029-12-01')
    assert age_model(m,'2023-01-01')==a
    m=sample();m['event']=0;b=age_model(m,'2023-01-01')
    assert not np.asarray(b['recent']).any()
    assert b['requested_slopes']==[1.,1.]


def test_piecewise_hazards_and_prior():
    rates=np.array([[.001,.002],[.002,.001]])
    val=piecewise_cif(rates,np.array([0.,365.,730.,800.]))
    one=-np.expm1(-.003*365)
    np.testing.assert_allclose(val[1],one*np.array([1/3,2/3]))
    np.testing.assert_allclose(val[2],val[1]+(1-one)*one*np.array([2/3,1/3]))
    np.testing.assert_array_equal(val[2],val[3])
    b=age_model(sample(),'2023-01-01')
    for d in b['diagnostics']:
        r=np.asarray(d['historical_rates']);prior=d['prior_days']
        np.testing.assert_allclose(prior*r.sum(),5.)
        np.testing.assert_allclose(d['recent_rates'],(np.asarray(d['recent_events'])+prior*r)/(d['recent_days']+prior))


def test_exact_q2_mapping_and_global_certificate(tmp_path):
    f=hand_forest('two')
    ref=np.array([[-2,-1,-1],[0,1,1],[2,0,-1],[np.nan,1,np.nan]],float)
    ex=CIFExpansion.from_forest(f,ref)
    b=age_model(sample(),'2023-01-01')
    cal=dict(b=[.5,.5],shape_grid=b['grid'],offset_values=(.5*np.asarray(b['recent'])).tolist())
    certify(forest_envelope(f),cal)
    times=np.unique(np.r_[f.grid_, b['grid'],365,366,730,731])
    p=mapping(f.predict(ref,times),times,cal)
    out=transformed(ex,cal)
    np.testing.assert_allclose(out.predict(ref,times),p,atol=1e-14)
    total=np.broadcast_to(out._coeff(times)[1],p.shape).copy()
    for k in [(j,) for j in out.mains]+out.pairs: total+=out.component(ref,k,times)
    np.testing.assert_allclose(total,p,atol=1e-14)
    out.save(tmp_path/'f.npz')
    np.testing.assert_allclose(CIFExpansion.load(tmp_path/'f.npz').predict(ref,times),p,atol=1e-14)
    assert p.min()>=-1e-14 and p.sum(-1).max()<=1+1e-14 and np.diff(p,axis=1).min()>=-1e-14
    zero=np.zeros((1,4,2));v=mapping(zero,[365,366,730,731],cal)
    np.testing.assert_array_equal(v[:,0],v[:,1]);np.testing.assert_array_equal(v[:,2],v[:,3])
    bad=dict(cal,b=[5.,5.])
    with pytest.raises(ValueError):certify(forest_envelope(f),bad)
