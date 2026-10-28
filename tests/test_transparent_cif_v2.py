"""Synthetic/analytical tests only: no enterprise performance claims."""
import copy
import json
from dataclasses import replace
from itertools import product
import numpy as np
import pytest
from crforest.core import ForestConfig, CompetingRiskForest, aalen_johansen, logrank_cr_score, _split_parent
from crforest.functions import CIFExpansion
from crforest.transparent import (TransparentCIF, WeightConfig, solve_simplex_brier,
    learn_tree_weights, tree_predictions, validate_weights, group_roles, disjoint_groups, state_targets)
from crforest.pair_candidates import PairConfig, augment_pair_bank, extend_ordinary_bank, pair_coverage


def sample(n=800, seed=41):
    rng = np.random.default_rng(seed)
    X = rng.choice([-1., 1.], size=(n, 5))
    prob = .2 + .03 * X[:, 0] + .07 * X[:, 1] * X[:, 2]
    u = rng.random(n)
    e = np.where(u < prob, 1, np.where(u < prob + .08, 2, 0))
    t = np.where(e > 0, rng.uniform(.1, .9, n), 1.)
    return X, t, e, np.arange(n)


def config():
    return ForestConfig(n_trees=8, path_budget=2, min_leaf=10, min_groups=5,
                        min_events=1, cut_strategy='shared_quantile', n_thresholds=3, mtry=5)


@pytest.fixture(scope='module')
def fitted():
    X, t, e, g = sample()
    a, b = group_roles(g, .25)
    f = CompetingRiskForest(config()).fit(X[a], t[a], e[a], g[a])
    return X, t, e, g, a, b, f


@pytest.mark.parametrize('weights', [[-.1, 1.1], [.1, .2], [np.nan, 1.], [[.5, .5]], [1.]])
def test_illegal_weights_rejected(weights):
    with pytest.raises(ValueError): validate_weights(weights, 2)


def test_group_split_order_invariance_and_whole_groups():
    g = np.repeat(np.arange(80).astype(str), 4)
    a, b = group_roles(g, .2)
    c, d = group_roles(g[::-1], .2)
    assert set(g[a]) == set(g[::-1][c])
    assert set(g[b]) == set(g[::-1][d])
    assert len(b) == 64 and not set(g[a]) & set(g[b])
    assert disjoint_groups(g[a], g[b])['group_overlap'] == 0
    with pytest.raises(ValueError): disjoint_groups(g[a], g)


@pytest.mark.parametrize('fraction', [0, 1, -.2])
def test_group_split_invalid(fraction):
    with pytest.raises(ValueError): group_roles(['a', 'b'], fraction)


def test_endpoint_time_and_competing_causes():
    # Events beyond the first horizon belong to the event-free state at time .5.
    t = [1., .75, .25, .25, .5]; e = [0, 1, 2, 0, 1]
    h = np.tile([.5, 1.], (5, 1)); eligible = np.ones_like(h, bool)
    eligible[0, 1] = False
    Y, W, audit = state_targets(t, e, h, eligible, [1, 1], [0, 0])
    assert np.array_equal(Y.argmax(2), [[0, 0], [0, 1], [2, 2], [0, 0], [1, 1]])
    assert np.array_equal(W, [[1, 0], [1, 1], [1, 1], [0, 0], [1, 1]])
    assert audit['unknown_by_horizon'] == [1, 2]


def test_ipcw_left_limits_independent_hand_check():
    Y, W, _ = state_targets([1., 2.], [1, 0], np.ones((2, 1))*2,
                            np.ones((2, 1), bool), [1., 2., 3., 4.], [0, 1, 0, 0])
    # G(1-)=1, G(2-)=3/4, including event/censor ties correctly.
    np.testing.assert_allclose(W[:, 0], [1., 4/3])


def test_unknown_full_horizon_never_negative():
    _, W, _ = state_targets([1.], [0], np.array([[1.]]), np.array([[False]]), [1.], [0])
    assert W[0, 0] == 0
    with pytest.raises(ValueError, match='eligibility'):
        state_targets([1.], [0], np.array([[1.]]), np.array([[1]]), [1.], [0])


def test_positivity_not_silently_floored():
    with pytest.raises(ValueError, match='positivity'):
        state_targets([3.], [0], np.array([[2.]]), np.array([[True]]), [1.], [0])


def test_equal_weights_match_original_all_causes_times_missing(fitted):
    X, _, _, _, a, _, f = fitted
    x = X[:30].copy(); x[::5, 0] = np.nan
    times = [.01, .5, 1., 1.2]
    model = TransparentCIF.from_forest(f, X[a])
    np.testing.assert_allclose(model.predict(x, times), f.predict(x, times), atol=1e-13, rtol=0)
    P = tree_predictions(f, x, np.tile(times, (len(x), 1)))
    np.testing.assert_allclose(P.mean(3), f.predict(x, times), atol=1e-14, rtol=0)


def test_weighted_expansion_and_complete_explanations(fitted):
    X, _, _, _, a, _, f = fitted
    w = np.arange(1, len(f.trees_)+1, dtype=float); w /= w.sum()
    m = TransparentCIF.from_forest(f, X[a], w)
    h = np.tile([.4, 1.], (50, 1))
    truth = tree_predictions(f, X[:50], h) @ w
    np.testing.assert_allclose(m.predict(X[:50], [.4, 1.]), truth, atol=1e-13, rtol=0)
    explanation = m.explain(X[:50], [.4, 1.])
    assert explanation['complete'] and explanation['max_error'] < 1e-12
    assert len(explanation['components']) == len(m.expansion.mains) + len(m.expansion.pairs)
    assert not m.audit['hidden_predictor']
    assert all(item['order'] <= 2 for item in explanation['components'])


def test_exported_model_works_without_forest_and_is_immutable(fitted, tmp_path):
    X, _, _, _, a, _, f = fitted
    m = TransparentCIF.from_forest(f, X[a])
    expected = m.predict(X[:30], [.5, 1.]); m.save(tmp_path/'m')
    m2 = TransparentCIF.load(tmp_path/'m')
    assert not hasattr(m2, 'forest') and not hasattr(m2, 'trees_')
    np.testing.assert_array_equal(m2.predict(X[:30], [.5, 1.]), expected)
    with pytest.raises(FileExistsError): m.save(tmp_path/'m')
    manifest = tmp_path/'m/manifest_PRIVATE.json'
    manifest.write_text(manifest.read_text().replace('cause_1', 'wrong_label'))
    with pytest.raises(ValueError, match='hash'): TransparentCIF.load(tmp_path/'m')


def test_table_values_are_complete_native_functions(fitted):
    X, _, _, _, a, _, f = fitted
    m = TransparentCIF.from_forest(f, X[a])
    tables = m.complete_tables([1.], X[a])
    assert tables['complete']
    assert len(tables['functions']) == len(m.expansion.mains) + len(m.expansion.pairs)
    for table in tables['functions']:
        assert np.asarray(table['values']).shape == (*table['shape'], 1, 2)
        assert np.sum(table['reference_joint_record_support']) == len(a)
        assert all(axis[-1] == {'missing': True} for axis in table['axes'])
    json.dumps(tables, allow_nan=False)


def test_cif_simplex_monotonic_for_arbitrary_simplex_weights(fitted):
    X, _, _, _, a, _, f = fitted
    rng = np.random.default_rng(11)
    for _ in range(5):
        w = rng.dirichlet(np.ones(len(f.trees_)))
        model = TransparentCIF.from_forest(f, X[a], w)
        p = model.predict(X[:60], np.linspace(.01, 1.2, 24))
        assert p.min() >= -1e-12 and p.sum(2).max() <= 1 + 1e-12
        assert np.diff(p, axis=1).min() >= -1e-12


def test_calendar_horizon_rows_are_not_collapsed(fitted):
    X, _, _, _, a, _, f = fitted
    m = TransparentCIF.from_forest(f, X[a])
    h = np.array([[.4, .9], [.5, 1.]])
    p = m.predict_horizons(X[:2], h)
    for i in range(2):
        np.testing.assert_array_equal(p[i], m.predict(X[i:i+1], h[i])[0])


def test_qp_against_independent_dense_grid():
    P = np.array([[[[.1, .8]]], [[[.7, .3]]], [[[.2, .5]]]])
    Y = np.eye(2)[np.array([[0], [1], [0]])]
    c = WeightConfig(ridge_to_uniform=.01)
    w, r = solve_simplex_brier(P, Y, np.ones((3, 1)), c)
    grid = np.linspace(0, 1, 10001)
    def loss(z):
        prob = (P[...,0]*z+P[...,1]*(1-z))[...,0]
        return np.mean((prob-Y[...,1])**2+(1-prob-Y[...,0])**2) + .01*((z-.5)**2+(.5-z)**2)
    exact = np.array([loss(z) for z in grid])
    assert abs(w[0]-grid[exact.argmin()]) <= .00011
    assert r['frank_wolfe_gap'] <= 1e-8
    assert r['objective'] <= r['uniform_objective'] + 1e-12


def test_weight_learning_does_not_use_forest_inbag_labels(fitted):
    X, t, e, g, a, b, f = fitted
    h = np.tile([.5, 1.], (len(b), 1))
    w, r = learn_tree_weights(f, X[b], t[b], e[b], h, np.ones_like(h, bool), g[a], g[b], t[a], e[a])
    assert r['isolation']['group_overlap'] == 0 and r['status']=='CONVERGED'
    assert w.shape == (len(f.trees_),) and w.min() >= 0
    with pytest.raises(ValueError, match='disjoint'):
        learn_tree_weights(f, X[b], t[b], e[b], h, np.ones_like(h, bool), g, g[b], t, e)


def test_pair_bank_original_unchanged_and_matched_ordinary_control(fitted):
    X, t, e, g, a, b, f = fitted
    before = json.dumps(f.trees_, sort_keys=True)
    augmented, audit = augment_pair_bank(f, X[a], t[a], e[a], g[a], PairConfig(max_pairs=10,max_trees=3,max_cuts=2))
    assert json.dumps(f.trees_, sort_keys=True) == before
    assert audit['added_trees'] > 0 and len(augmented.trees_) == len(f.trees_) + audit['added_trees']
    ordinary = extend_ordinary_bank(f, X[a], t[a], e[a], g[a], audit['added_trees'])
    assert len(ordinary.trees_) == len(augmented.trees_)
    assert augmented.audit()['max_path_variables'] == 2
    for tree in augmented.trees_[len(f.trees_):]:
        assert tree[0]['joint_pair'] == tree[-1]['ancestors']
        assert all(n['groups']>=f.config.min_groups and n['events']>=f.config.min_events for n in tree if 'cif' in n)
    m = TransparentCIF.from_forest(augmented, X[a])
    np.testing.assert_allclose(m.predict(X[b],[1.]), augmented.predict(X[b],[1.]),atol=1e-13)


def test_joint_pair_can_find_zero_marginal_xor():
    # Exactly balanced cells and deterministic outcomes: both single split scores=0.
    X=np.repeat(np.array(list(product([-1.,1.],repeat=2))),100,axis=0)
    e=np.zeros(len(X),int)
    for cell in range(4):
        n=70 if X[cell*100,0]*X[cell*100,1]>0 else 10
        e[cell*100:cell*100+n]=1
    t=np.where(e>0,.5,1.);g=np.arange(len(X));parent=_split_parent(t,e,1)
    assert logrank_cr_score(X[:,0]<0,e,parent)==0
    assert logrank_cr_score(X[:,1]<0,e,parent)==0
    # Baseline constant tree isolates the candidate's actual joint-search logic.
    f=CompetingRiskForest(ForestConfig(n_trees=1,path_budget=2,max_depth=2,
        n_causes=1,min_leaf=10,min_groups=5,min_events=1,cut_strategy='shared_quantile',mtry=2)).fit(X,t,e,g)
    aug,audit=augment_pair_bank(f,X,t,e,g,PairConfig(max_pairs=1,max_trees=1,max_cuts=1))
    assert audit['added_trees']==1
    tree=aug.trees_[-1]
    vals=[np.asarray(n['cif'])[-1,0] for n in tree if 'cif' in n]
    assert (vals[0]+vals[3]-vals[1]-vals[2])/4>.2


@pytest.mark.parametrize('kwargs',[{'max_pairs':0},{'max_trees':65},{'max_cuts':0},{'min_score':-1}])
def test_invalid_pair_configuration(kwargs):
    with pytest.raises(ValueError):PairConfig(**kwargs)


@pytest.mark.parametrize('kwargs',[{'ridge_to_uniform':-1},{'max_iterations':0},{'gap_tolerance':0}])
def test_invalid_weight_configuration(kwargs):
    with pytest.raises(ValueError):WeightConfig(**kwargs)


def test_disabled_new_methods_leave_original_fit_deterministic(fitted):
    X,t,e,g,a,b,f=fitted
    again=CompetingRiskForest(config()).fit(X[a],t[a],e[a],g[a])
    assert f.trees_==again.trees_
    np.testing.assert_array_equal(f.predict(X[b],[1.]),again.predict(X[b],[1.]))


def test_pair_bank_rejects_active_global_budget(fitted):
    X,t,e,g,a,b,f=fitted;ff=copy.copy(f);ff.config=replace(f.config,global_cell_budget=1000)
    with pytest.raises(ValueError,match='ledger'):
        augment_pair_bank(ff,X[a],t[a],e[a],g[a])


def test_constant_forest_compiles_and_explains():
    X=np.zeros((10,2));t=np.arange(1,11);e=np.ones(10,int)
    f=CompetingRiskForest(ForestConfig(n_trees=2,path_budget=2,max_depth=0,n_causes=1)).fit(X,t,e)
    model=TransparentCIF.from_forest(f,X)
    assert model.explain(X,[5.])['components']==[]
    assert model.explain(X,[5.])['max_error']==0
