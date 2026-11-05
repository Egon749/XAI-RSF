from dataclasses import asdict,replace
import numpy as np
import pytest
from crforest import CompetingRiskForest,CIFExpansion
from crforest.partition import tree_feature_subset,training_cut_dictionary
from crforest.shrinkage import shrink_leaf_cifs
from experiments.post_st_feature_pool_grid_v1.common import COUNTS,configuration,select_count
from experiments.post_st_ablation_v1.common import pairing,array_hash
from experiments.post_st_shrinkage_strength_v1.verification import check_shrinkage

def test_only_declared_configuration_and_pairing_change():
    groups=np.repeat(np.arange(20),2);base=asdict(configuration(17,25));ref=pairing(groups,configuration(17,25),74)
    assert COUNTS==tuple(range(10,36))
    for count in COUNTS:
        cfg=configuration(17,count);actual=asdict(cfg)
        assert [k for k in actual if actual[k]!=base[k]]==(['tree_feature_count'] if count!=25 else [])
        paired=pairing(groups,cfg,74)
        for ti,(a,b) in enumerate(zip(paired,ref)):
            assert a['bootstrap_sha256']==b['bootstrap_sha256'] and a['multiplicity_sha256']==b['multiplicity_sha256']
            pool=tree_feature_subset(74,count,17,ti,'balanced')
            assert len(pool)==len(set(pool))==count and pool.min()>=0 and pool.max()<74
            assert array_hash(pool)==a['features_sha256']
    with pytest.raises(ValueError):configuration(17,36)

def test_candidate_pool_changes_keep_exact_q2_missing_and_shrinkage(tmp_path):
    rng=np.random.default_rng(110525);X=rng.normal(size=(64,74));X[::5,3]=np.nan
    t=rng.integers(1,25,64).astype(float);event=rng.integers(0,3,64);groups=np.arange(64)
    tr=dict(X=X,time=t,event=event,groups=groups);previous_cuts=None;previous_boot=None
    for count in [10,25,35]:
        cfg=replace(configuration(17,count),n_trees=2,min_leaf=4,min_events=1,min_groups=3)
        raw=CompetingRiskForest(cfg).fit(X,t,event,groups)
        cuts=training_cut_dictionary(X,cfg)
        if previous_cuts is not None:
            for a,b in zip(cuts,previous_cuts):np.testing.assert_array_equal(a,b)
            for a,b in zip(raw.inbag_,previous_boot):np.testing.assert_array_equal(a,b)
        previous_cuts=cuts;previous_boot=raw.inbag_
        sh=shrink_leaf_cifs(raw,X,t,event,groups,10);check_shrinkage(raw,sh,tr,10)
        assert sh.audit()['max_path_variables']<=2
        ex=CIFExpansion.from_forest(sh,X);times=np.r_[0.,raw.grid_,100.];p=sh.predict(X,times)
        np.testing.assert_allclose(ex.predict(X,times),p,atol=1e-10,rtol=0)
        assert p.min()>=0 and p.sum(-1).max()<=1+1e-12 and np.diff(p,axis=1).min()>=-1e-12
        sh.save(tmp_path/f'{count}.json');again=CompetingRiskForest.load(tmp_path/f'{count}.json')
        np.testing.assert_array_equal(again.predict(X,times),p)

def test_historical_selector_ties_and_support_fallback():
    rows=[dict(pool_count=c,within_year_auc=.7,ap=.3,ipcw_brier=.1) for c in COUNTS]
    assert select_count(rows)[0]==25
    rows[0]['within_year_auc']=.8;assert select_count(rows)[0]==10
    for r in rows:r['within_year_auc']=np.nan
    assert select_count(rows)[0]==25
