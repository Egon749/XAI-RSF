import json
import os
import subprocess
import numpy as np
import pytest
from crforest import CompetingRiskForest, ForestConfig, CIFExpansion, aalen_johansen
from crforest.core import _split_parent, logrank_cr_score
from experiments.post_st_rule_features_v1.audit_root_weights import independent_components


@pytest.mark.parametrize('weights', [[], [1], [1, 2, 3], [0, 0], [-1, 1], [np.nan, 1], [1, np.inf], [[1, 0]]])
def test_invalid_weights_rejected(weights):
    with pytest.raises(ValueError, match='cause_weights'):
        ForestConfig(cause_weights=weights)
    with pytest.raises(ValueError, match='cause_weights'):
        logrank_cr_score(np.array([True, False]), np.array([1, 2]),
                         _split_parent(np.array([1., 2.]), np.array([1, 2]), 2), weights)


@pytest.mark.parametrize('weights', [(1, 1), (2, 1), (1, 0), (0, 1)])
def test_weighted_score_independent_risk_sets(weights):
    t=np.array([1,1,2,2,3,4,5,6.]);e=np.array([1,0,2,1,2,0,1,2])
    left=np.array([1,0,1,1,0,1,0,0],bool)
    u,v=independent_components(t,e,left);w=np.array(weights)
    expected=abs(w@u)/np.sqrt((w*w)@v)
    assert logrank_cr_score(left,e,_split_parent(t,e,2),w)==pytest.approx(expected,abs=1e-14)
    assert logrank_cr_score(left,e,_split_parent(t,e,2),w*3)==pytest.approx(expected,abs=1e-14)


def test_fit_uses_weights_preserves_aj_and_exact_explanation(tmp_path):
    x=np.arange(80.).reshape(-1,1);t=np.ones(80);e=np.where(x[:,0]<40,1,2)
    settings=dict(n_trees=3,max_depth=1,path_budget=2,min_leaf=5,min_groups=3,min_events=1,
                  mtry=1,n_thresholds=79,seed=42)
    base=CompetingRiskForest(ForestConfig(**settings)).fit(x,t,e)
    equal=CompetingRiskForest(ForestConfig(**settings,cause_weights=[1,1])).fit(x,t,e)
    target=CompetingRiskForest(ForestConfig(**settings,cause_weights=[1,0])).fit(x,t,e)
    assert base.trees_==equal.trees_
    assert base.trees_!=target.trees_
    assert base.audit_['leaves']==3 and target.audit_['leaves']==6
    for tree,counts in zip(target.trees_,target.inbag_):
        root=tree[0];rows=np.repeat(np.arange(len(x)),counts)
        left=x[rows,0]<=root['threshold']
        for side,name in [(left,'left'),(~left,'right')]:
            expected=aalen_johansen(t[rows[side]],e[rows[side]],target.grid_)
            np.testing.assert_array_equal(tree[root[name]]['cif'],expected)
    p=target.predict(x,[0,1,2]);ex=CIFExpansion.from_forest(target,x)
    np.testing.assert_allclose(ex.predict(x,[0,1,2]),p,atol=1e-14,rtol=0)
    assert p.min()>=0 and p.sum(-1).max()<=1+1e-14 and np.diff(p,axis=1).min()>=0
    target.save(tmp_path/'target.json')
    loaded=CompetingRiskForest.load(tmp_path/'target.json')
    assert loaded.config.cause_weights==(1.,0.)
    np.testing.assert_array_equal(loaded.predict(x,[0,1,2]),p)
    base.save(tmp_path/'base.json')
    old=json.loads((tmp_path/'base.json').read_text());old['config'].pop('cause_weights')
    (tmp_path/'old.json').write_text(json.dumps(old))
    np.testing.assert_array_equal(CompetingRiskForest.load(tmp_path/'old.json').predict(x),base.predict(x))


@pytest.mark.parametrize('weights', [(2,1),(1,0)])
def test_weighted_split_against_independent_rfsrc(weights,tmp_path):
    r=os.environ.get('CRFOREST_RSCRIPT')
    if not r:pytest.skip('set CRFOREST_RSCRIPT for independent R check')
    rng=np.random.default_rng(114);x=np.arange(80.);t=rng.integers(1,20,len(x)).astype(float)
    e=np.where(x<33,rng.choice([0,1,2],len(x),p=[.2,.7,.1]),rng.choice([0,1,2],len(x),p=[.5,.1,.4]))
    scores=[logrank_cr_score(x<=cut,e,_split_parent(t,e,2),weights) for cut in x[1:-2]]
    cut=x[1:-2][np.argmax(scores)]
    np.savetxt(tmp_path/'data.csv',np.c_[t,e,x],delimiter=',',header='time,event,x',comments='')
    script=tmp_path/'check.R'
    script.write_text('''args <- commandArgs(TRUE)
.libPaths(c(Sys.getenv("EW_RLIB"),.libPaths()))
suppressPackageStartupMessages(library(randomForestSRC));library(survival);options(rf.cores=1)
d <- read.csv(args[1])
f <- rfsrc(Surv(time,event)~x,data=d,ntree=1,bootstrap="by.user",samp=matrix(1,nrow(d),1),
 nodesize=2,nodedepth=1,mtry=1,nsplit=0,splitrule="logrankCR",cause=as.numeric(args[3:4]),ntime=0,perf.type="none",seed=-31)
p <- predict(f,newdata=d,perf.type="none")
write.table(cbind(p$time.interest,p$cif[1,,],p$cif[nrow(d),,]),args[2],row.names=FALSE,col.names=FALSE,sep=",")
''')
    subprocess.run([r,'--vanilla',str(script),str(tmp_path/'data.csv'),str(tmp_path/'out.csv'),*map(str,weights)],check=True)
    z=np.loadtxt(tmp_path/'out.csv',delimiter=',')
    np.testing.assert_allclose(aalen_johansen(t[x<=cut],e[x<=cut],z[:,0]),z[:,1:3],atol=2e-14)
    np.testing.assert_allclose(aalen_johansen(t[x>cut],e[x>cut],z[:,0]),z[:,3:5],atol=2e-14)
