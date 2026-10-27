"""Stage A: manually checkable trees, competing AJ, independent R checks."""
import json
import os
import subprocess
from itertools import product
import numpy as np
import pytest
from crforest import ForestConfig, CompetingRiskForest, CIFExpansion, aalen_johansen, cluster_bootstrap
from crforest.core import _split_parent, logrank_cr_score


def hand_forest(kind):
    f = CompetingRiskForest(ForestConfig(n_trees=1, path_budget=2))
    f.grid_, f.n_features_, f.candidate_evaluations_ = np.array([1., 2., 3.]), 3, 0
    nodes = []
    def leaf(used, depth, time, event):
        nodes.append(dict(ancestors=sorted(used), depth=depth, cif=aalen_johansen(time, event, f.grid_).tolist()))
        return len(nodes) - 1
    def split(j, cut, used, depth, missing=False):
        nodes.append(dict(feature=j, threshold=cut, missing_left=missing, ancestors=sorted(used), depth=depth))
        return len(nodes) - 1
    root = split(0, 0., set(), 0, True)
    if kind == 'one':
        nodes[root]['left'] = leaf({0}, 1, [1, 2, 3], [1, 2, 0])
        nodes[root]['right'] = leaf({0}, 1, [1, 2, 3], [0, 2, 0])
    else:
        a = split(1, 0., {0}, 1)
        nodes[root]['left'] = a
        b = split(0, -1., {0, 1}, 2, True)
        nodes[a]['left'] = b  # repeated x0, depth three, still two variables
        nodes[b]['left'] = leaf({0, 1}, 3, [1, 1, 2, 3, 3], [1, 0, 2, 1, 0])
        nodes[b]['right'] = leaf({0, 1}, 3, [1, 2, 3], [2, 2, 0])
        nodes[a]['right'] = leaf({0, 1}, 2, [1, 2, 3], [0, 0, 0])
        c = split(2, 0., {0}, 1, True)  # different branch uses pair (0,2)
        nodes[root]['right'] = c
        nodes[c]['left'] = leaf({0, 2}, 2, [1, 2, 3], [1, 2, 0])
        nodes[c]['right'] = leaf({0, 2}, 2, [1, 2, 3], [1, 1, 1])
    f.trees_, f.audit_ = [nodes], None
    f.audit_ = f.audit()
    return f


def test_aj_hand_ties_censor_and_no_target():
    a = aalen_johansen([1, 1, 2, 3, 3], [1, 0, 2, 1, 0], [0, 1, 2, 3, 10])
    np.testing.assert_allclose(a, [[0, 0], [.2, 0], [.2, 4/15], [7/15, 4/15], [7/15, 4/15]], atol=1e-15)
    b = aalen_johansen([1, 2, 2, 3], [2, 0, 2, 0], [1, 2, 3])
    assert not b[:, 0].any()
    assert (np.diff(a, axis=0) >= 0).all() and (a.sum(1) <= 1).all()


@pytest.mark.parametrize('kind', ['one', 'two'])
def test_hand_tree_export_all_times_and_product_centering(kind, tmp_path):
    f = hand_forest(kind)
    ref = np.array(list(product([-2., -1., 0., 1., np.nan], [-1., 0., 1.], [-1., 1., np.nan])))
    ex = CIFExpansion.from_forest(f, ref)
    grid = np.unique(np.r_[f.grid_, [0, .5, 1.5, 4]])
    p = f.predict(ref, grid)
    np.testing.assert_allclose(p, ex.predict(ref, grid), atol=1e-14)
    total = np.broadcast_to(ex._coeff(grid)[1], p.shape).copy()
    for key in [(j,) for j in ex.mains] + ex.pairs:
        c = ex.component(ref, key, grid)
        total += c
        assert np.max(abs(c.mean(0))) < 1e-14
        if len(key) == 2:
            for j in key:
                for value in np.unique(ref[:, j]):
                    mask = np.isnan(ref[:, j]) if np.isnan(value) else ref[:, j] == value
                    assert np.max(abs(c[mask].mean(0))) < 1e-14
    np.testing.assert_allclose(total, p, atol=1e-14)
    ex.save(tmp_path/'functions.npz')
    np.testing.assert_allclose(CIFExpansion.load(tmp_path/'functions.npz').predict(ref, grid), p, atol=1e-14)
    if kind == 'two':
        assert f.audit_['max_depth'] == 3 and f.audit_['max_path_variables'] == 2
        assert ex.pairs == [(0, 1), (0, 2)]


def test_logrank_cr_independent_loop():
    t = np.array([1, 1, 2, 2, 3, 4, 5, 6.])
    e = np.array([1, 0, 2, 1, 2, 0, 1, 2])
    left = np.array([1, 0, 1, 1, 0, 1, 0, 0], bool)
    u, v = 0., 0.
    for k in (1, 2):
        for s in np.unique(t[e > 0]):
            risk = (t >= s) | ((t < s) & (e > 0) & (e != k))
            y, yl = risk.sum(), (risk & left).sum()
            d, dl = ((t == s) & (e == k)).sum(), ((t == s) & (e == k) & left).sum()
            u += dl - d * yl / y
            if (t >= s).sum() >= 2:
                v += d * yl / y * (1 - yl / y) * (y - d) / (y - 1)
    assert logrank_cr_score(left, e, _split_parent(t, e, 2)) == pytest.approx(abs(u)/np.sqrt(v), abs=1e-14)


def test_group_resampling_is_whole_company():
    groups = np.repeat(np.arange(60), [1, 2, 3] * 20)
    for tree in range(10):
        ix = cluster_bootstrap(groups, 42, tree)
        counts = np.bincount(ix, minlength=len(groups))
        assert all(len(np.unique(counts[groups == j])) == 1 for j in np.unique(groups))
        assert sum(counts[np.flatnonzero(groups == j)[0]] for j in np.unique(groups)) == 60


@pytest.mark.parametrize('budget', [1, 2, None])
def test_fitted_path_audit_missing_probability_and_reload(budget, tmp_path):
    rng = np.random.default_rng(31)
    x = rng.normal(size=(210, 5)); x[::7, 0] = np.nan
    t = rng.integers(1, 12, size=len(x)).astype(float)
    e = rng.choice([0, 1, 2], size=len(x))
    f = CompetingRiskForest(ForestConfig(n_trees=5, max_depth=5, path_budget=budget,
                                        min_leaf=8, min_events=1, min_groups=3, n_thresholds=5)).fit(x, t, e, np.arange(len(x))//2)
    a = f.audit()
    assert budget is None or a['max_path_variables'] <= budget
    grid = np.r_[0, f.grid_, 13]
    p = f.predict(x, grid)
    assert p.min() >= 0 and np.diff(p, axis=1).min() >= -1e-14 and p.sum(2).max() <= 1 + 1e-14
    f.save(tmp_path/'model.json')
    np.testing.assert_array_equal(CompetingRiskForest.load(tmp_path/'model.json').predict(x, grid), p)
    if budget:
        ex = CIFExpansion.from_forest(f, x)
        np.testing.assert_allclose(ex.predict(x, grid), p, atol=1e-13)


def test_composite_and_duplicate_feature_sources_rejected():
    x = np.zeros((20, 2)); t = np.arange(1, 21); e = np.ones(20)
    for sources in [[(0, 1), (2,)], [(0,), (0,)]]:
        with pytest.raises(ValueError, match='original'):
            CompetingRiskForest().fit(x, t, e, feature_sources=sources)


def test_export_rejects_higher_order():
    f = hand_forest('two'); f.trees_[0][2]['feature'] = 2
    # Rebuild stored ancestry so audit examines the actual higher-order path.
    def update(index, used):
        n = f.trees_[0][index]; n['ancestors'] = sorted(used)
        if 'cif' not in n:
            for c in ['left', 'right']: update(n[c], used | {n['feature']})
    update(0, set()); f.config = ForestConfig(path_budget=None)
    with pytest.raises(ValueError): CIFExpansion.from_forest(f, np.zeros((5, 3)))


def test_r_survival_independent_aj(tmp_path):
    r = os.environ.get('CRFOREST_RSCRIPT')
    if not r: pytest.skip('set CRFOREST_RSCRIPT for independent R survival check')
    rng = np.random.default_rng(930)
    t = np.r_[1, 1, 2, 3, 3, rng.integers(1, 30, 100)]
    e = np.r_[1, 0, 2, 1, 0, rng.choice([0, 1, 2], 100)]
    np.savetxt(tmp_path/'data.csv', np.c_[t, e], delimiter=',', header='time,event', comments='')
    script = tmp_path/'aj.R'
    script.write_text('''args <- commandArgs(TRUE)
.libPaths(c(Sys.getenv("EW_RLIB"),.libPaths()))
library(survival)
d <- read.csv(args[1]); d$status <- factor(d$event,levels=c(0,1,2),labels=c("censor","one","two"))
fit <- survfit(Surv(time,status)~1,data=d)
write.table(cbind(fit$time,fit$pstate[,2:3]),args[2],row.names=FALSE,col.names=FALSE,sep=",")
''')
    subprocess.run([r, '--vanilla', str(script), str(tmp_path/'data.csv'), str(tmp_path/'out.csv')], check=True)
    result = np.loadtxt(tmp_path/'out.csv', delimiter=',')
    np.testing.assert_allclose(aalen_johansen(t, e, result[:, 0]), result[:, 1:], atol=2e-14)


def test_rfsrc_exhaustive_single_split_and_leaf_aj(tmp_path):
    r = os.environ.get('CRFOREST_RSCRIPT')
    if not r: pytest.skip('set CRFOREST_RSCRIPT for R reference check')
    rng = np.random.default_rng(114)
    x = np.arange(80.)
    t = rng.integers(1, 20, len(x)).astype(float)
    e = np.where(x < 33, rng.choice([0, 1, 2], len(x), p=[.2, .7, .1]),
                 rng.choice([0, 1, 2], len(x), p=[.5, .1, .4]))
    parent = _split_parent(t, e, 2)
    scores = [logrank_cr_score(x <= cut, e, parent) for cut in x[1:-2]]
    cut = x[1:-2][np.argmax(scores)]
    np.savetxt(tmp_path/'data.csv', np.c_[t, e, x], delimiter=',', header='time,event,x', comments='')
    script = tmp_path/'one.R'
    script.write_text('''args <- commandArgs(TRUE)
.libPaths(c(Sys.getenv("EW_RLIB"),.libPaths()))
suppressPackageStartupMessages(library(randomForestSRC));library(survival);options(rf.cores=1)
d <- read.csv(args[1])
f <- rfsrc(Surv(time,event)~x,data=d,ntree=1,bootstrap="by.user",samp=matrix(1,nrow(d),1),
 nodesize=2,nodedepth=1,mtry=1,nsplit=0,splitrule="logrankCR",cause=c(1,1),ntime=0,perf.type="none",seed=-31)
p <- predict(f,newdata=d,perf.type="none")
write.table(cbind(p$time.interest,t(p$cif[1,,]),t(p$cif[nrow(d),,])),args[2],row.names=FALSE,col.names=FALSE,sep=",")
'''.replace('t(p$cif[1,,]),t(p$cif[nrow(d),,])', 'p$cif[1,,],p$cif[nrow(d),,]'))
    subprocess.run([r, '--vanilla', str(script), str(tmp_path/'data.csv'), str(tmp_path/'out.csv')], check=True)
    z = np.loadtxt(tmp_path/'out.csv', delimiter=',')
    np.testing.assert_allclose(aalen_johansen(t[x<=cut], e[x<=cut], z[:,0]), z[:,1:3], atol=2e-14)
    np.testing.assert_allclose(aalen_johansen(t[x>cut], e[x>cut], z[:,0]), z[:,3:5], atol=2e-14)


def test_single_cause_is_kaplan_meier_complement():
    t = np.array([1., 1., 2., 3., 5.]); e = np.array([1, 0, 1, 0, 1])
    # KM survival: 4/5, (4/5)*(2/3), unchanged, zero.
    np.testing.assert_allclose(aalen_johansen(t,e,[0,1,2,3,5],1)[:,0],
                               [0,1/5,7/15,7/15,1],atol=1e-15)


@pytest.mark.parametrize('event', [[0,1.5],[0,-1],[0,3],[0,float('nan')]])
def test_aj_rejects_invalid_events(event):
    with pytest.raises(ValueError): aalen_johansen([1,2],event,[1,2])


def test_fit_rejects_zero_followup_and_bad_budgets():
    with pytest.raises(ValueError): CompetingRiskForest().fit(np.zeros((2,1)),[0,1],[0,1])
    with pytest.raises(ValueError): CompetingRiskForest(ForestConfig(mtry=0)).fit(np.zeros((2,1)),[1,2],[0,1])


def test_exhaustive_boundary_missing_export():
    f=hand_forest('two')
    vals=[-2.,np.nextafter(-1.,-np.inf),-1.,np.nextafter(-1.,np.inf),np.nextafter(0.,-np.inf),0.,np.nextafter(0.,np.inf),1.,np.nan]
    x=np.array(list(product(vals,repeat=3)))
    ex=CIFExpansion.from_forest(f,x)
    np.testing.assert_allclose(ex.predict(x,[0,1,2,3,5]),f.predict(x,[0,1,2,3,5]),rtol=0,atol=1e-14)


@pytest.mark.parametrize('budget', [None,1,2])
def test_parallel_trees_bitwise_equal_serial(budget,tmp_path):
    from crforest.core import physical_worker_cpus
    if len(physical_worker_cpus())<2: pytest.skip('requires two allocated physical cores')
    rng=np.random.default_rng(9301)
    x=rng.normal(size=(180,5));x[::9,2]=np.nan
    t=rng.integers(1,20,len(x)).astype(float);e=rng.choice([0,1,2],len(x));groups=np.arange(len(x))//3
    cfg=ForestConfig(n_trees=6,max_depth=4,path_budget=budget,min_leaf=8,min_groups=3,min_events=1,n_thresholds=4)
    serial=CompetingRiskForest(cfg).fit(x,t,e,groups,n_jobs=1)
    parallel=CompetingRiskForest(cfg).fit(x,t,e,groups,n_jobs=2)
    assert serial.trees_==parallel.trees_
    assert serial.audit_==parallel.audit_
    for a,b in zip(serial.inbag_,parallel.inbag_):np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(serial.predict(x),parallel.predict(x))
    serial.save(tmp_path/'serial.json');parallel.save(tmp_path/'parallel.json')
    assert (tmp_path/'serial.json').read_bytes()==(tmp_path/'parallel.json').read_bytes()


def test_parallelism_respects_scheduler_affinity(monkeypatch):
    import crforest.core as core
    monkeypatch.setattr(core,'physical_worker_cpus',lambda:[0])
    with pytest.raises(ValueError,match='physical cores'):
        CompetingRiskForest().fit(np.zeros((10,1)),np.ones(10),np.ones(10),n_jobs=2)
