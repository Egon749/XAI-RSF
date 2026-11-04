import numpy as np
from .common import SEEDS,features,pool_size
from .metrics import DESIGNS,score
from .attribution import block

def test_c039_members_and_matched_information_sets():
    f=features()
    assert SEEDS==[17,503,22] and [f[a].sum() for a in ['M0','M1','M2','M3']]==[45,52,67,74]
    assert (f.M1&f.M2).equals(f.M0) and (f.M1|f.M2).equals(f.M3)
    assert [pool_size(p,'proportional') for p in [45,52,67,74]]==[15,18,23,25]

def test_same_four_utilities_obey_factorial_identity():
    values=dict(M0=.4,M1=.55,M2=.5,M3=.7)
    gains={k:sum(values[a]*w for a,w in weights.items()) for k,weights in DESIGNS.items()}
    np.testing.assert_allclose(gains['complementarity'],gains['audit_given_market']-gains['audit_vs_finance'])
    np.testing.assert_allclose(gains['complementarity'],gains['market_given_audit']-gains['market_vs_finance'])
    y=np.array([0,1,0,1],bool);ids=np.arange(4);years=np.array([1,1,2,2])
    assert score(y,np.array([.1,.9,.1,.9]),years,ids)[2]>score(y,np.full(4,.5),years,ids)[2]

def test_market_blocks_are_disjoint_and_include_coverage():
    f=features();f['block']=[block(n,g) for n,g in zip(f.english_name,f['info'])]
    assert set(f[f['info']=='M'].block)=={'volatility_tail','price_cap_return','liquidity_trading'}
    assert f[f.block=='Q'].english_name.tolist()==['market_missing_coverage']
