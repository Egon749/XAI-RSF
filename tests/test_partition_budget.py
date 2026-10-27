from dataclasses import replace
import numpy as np
import pytest
from crforest import CompetingRiskForest, ForestConfig, CIFExpansion
from crforest.partition import PartitionLedger, split_region, shared_quantile_cuts
from crforest.core import physical_worker_cpus


def data():
    rng=np.random.default_rng(804)
    X=rng.normal(size=(240,5));X[::7,0]=np.nan;X[::11,3]=np.nan
    return X,rng.integers(1,20,240).astype(float),rng.choice([0,1,2],240),np.arange(240)//2


def config(**kw):
    return replace(ForestConfig(n_trees=8,max_depth=5,path_budget=2,min_leaf=8,min_events=1,min_groups=3,n_thresholds=5),**kw)


def test_quantiles_are_training_observed_values_and_handle_missing():
    X=np.array([[0.,np.nan,1],[0,np.nan,1],[2,np.nan,1],[4,np.nan,1],[8,np.nan,1]])
    cuts=shared_quantile_cuts(X,5)
    assert set(cuts[0])<={0,2,4} and len(cuts[1])==len(cuts[2])==0
    np.testing.assert_array_equal(cuts[0],shared_quantile_cuts(X[::-1],5)[0])


def test_ledger_includes_ancestor_bounds_and_missing_states():
    led=PartitionLedger(100)
    children=split_region({},0,0.,True);led.commit(led.project(children))
    assert led.cells==3
    deeper=split_region(children[0],1,2.,False);led.commit(led.project(deeper))
    assert led.cells==3+3+9
    before=led.cells;led.commit(led.project(deeper))
    assert led.cells==before and led.reused_splits==1
    assert not PartitionLedger(2).accepts(PartitionLedger(2).project(children))


@pytest.mark.parametrize('strategy',['local_random','shared_quantile'])
@pytest.mark.parametrize('budget',[0,30,120])
def test_whole_forest_budget_equals_exporter(strategy,budget,tmp_path):
    X,t,e,g=data();f=CompetingRiskForest(config(cut_strategy=strategy,global_cell_budget=budget)).fit(X,t,e,g)
    ex=CIFExpansion.from_forest(f,X);cc=ex.complexity(X)
    assert cc['spatial_table_cells']==f.partition_audit_['spatial_cells']<=budget
    assert f.audit()['path_violations']==0
    np.testing.assert_allclose(f.predict(X),ex.predict(X),rtol=0,atol=1e-13)
    if strategy=='shared_quantile':
        for tree in f.trees_:
            for node in tree:
                if 'threshold' in node and np.isfinite(node['threshold']):assert node['threshold'] in f.shared_cuts_[node['feature']]
    f.save(tmp_path/'f.json');loaded=CompetingRiskForest.load(tmp_path/'f.json')
    assert loaded.partition_audit_==f.partition_audit_
    np.testing.assert_array_equal(loaded.predict(X),f.predict(X))
    assert all(r.get('cells_after',r['cells_before'])<=budget for r in f.budget_log_)
    if budget==0:assert all(len(tree)==1 for tree in f.trees_)


@pytest.mark.parametrize('strategy',['local_random','shared_quantile'])
def test_inactive_budget_preserves_all_trees(strategy):
    X,t,e,g=data();cfg=config(cut_strategy=strategy)
    a=CompetingRiskForest(cfg).fit(X,t,e,g)
    b=CompetingRiskForest(replace(cfg,global_cell_budget=1000000)).fit(X,t,e,g)
    assert a.trees_==b.trees_ and a.audit_==b.audit_
    np.testing.assert_array_equal(a.predict(X),b.predict(X))


def test_shared_cuts_parallel_equal_and_budget_parallel_rejected():
    if len(physical_worker_cpus())<2:pytest.skip('two allocated physical cores required')
    X,t,e,g=data();cfg=config(cut_strategy='shared_quantile')
    a=CompetingRiskForest(cfg).fit(X,t,e,g,n_jobs=1);b=CompetingRiskForest(cfg).fit(X,t,e,g,n_jobs=2)
    assert a.trees_==b.trees_
    np.testing.assert_array_equal(a.predict(X),b.predict(X))
    with pytest.raises(ValueError,match='sequential'):
        CompetingRiskForest(replace(cfg,global_cell_budget=100)).fit(X,t,e,g,n_jobs=2)


def test_invalid_budget_and_higher_order_rejected():
    X,t,e,g=data()
    for kw in [dict(global_cell_budget=-1),dict(global_cell_budget=4.5),dict(global_cell_budget=True),dict(global_cell_budget=100,path_budget=None),dict(cut_strategy='unknown')]:
        with pytest.raises(ValueError):CompetingRiskForest(config(**kw)).fit(X,t,e,g)


@pytest.mark.parametrize('strategy',['local_random','shared_quantile'])
def test_tree_shards_identical_to_full_fit(strategy,tmp_path):
    from experiments.shared_partition_v1.engine import fit_shard,assemble
    X,t,e,g=data();cfg=config(cut_strategy=strategy)
    d=dict(X=X,time=t,event=e,groups=g)
    full=CompetingRiskForest(cfg).fit(X,t,e,g)
    parts=[(i,fit_shard(cfg,d,i,2)) for i in range(0,cfg.n_trees,2)]
    joined=assemble(parts[::-1]);assert full.trees_==joined.trees_ and full.audit_==joined.audit_
    full.save(tmp_path/'a.json');joined.save(tmp_path/'b.json')
    assert (tmp_path/'a.json').read_bytes()==(tmp_path/'b.json').read_bytes()
    with pytest.raises(ValueError):assemble(parts[:-1])
    with pytest.raises(ValueError):assemble(parts+[parts[0]])
