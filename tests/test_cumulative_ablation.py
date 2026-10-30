from dataclasses import asdict,replace
import json
from types import SimpleNamespace
import numpy as np
from crforest import CompetingRiskForest
from crforest.shrinkage import shrink_leaf_cifs
from experiments.post_st_ablation_v1.worker import emit
from experiments.post_st_ablation_v1.verification import inspect
from experiments.post_st_cumulative_ablation_v1.common import configuration,strength,pairing,STAGES

def test_each_cumulative_step_only_changes_the_declared_design():
    cfgs=[asdict(configuration(42,a)) for a in STAGES]
    for i,expected in enumerate([{'path_budget'},{'cut_strategy'},{'tree_feature_strategy'},set()]):
        assert {k for k in cfgs[i] if cfgs[i][k]!=cfgs[i+1][k]}==expected
    assert [strength(a) for a in STAGES]==[10,10,10,10,0]
    groups=np.arange(80)//2;base=pairing(groups,configuration(42,'s0'))
    for a in STAGES:
        p=pairing(groups,configuration(42,a))
        assert [v['bootstrap_sha256'] for v in p]==[v['bootstrap_sha256'] for v in base]
        if a in ['s0','s1','s2']:assert p==base
    assert pairing(groups,configuration(42,'s3'))==pairing(groups,configuration(42,'s4'))

def test_local_iid_unrestricted_export_adapter_checks_all_rows(tmp_path):
    rng=np.random.default_rng(611);X=rng.normal(size=(100,6));X[::9,1]=np.nan
    tr=dict(X=X,time=rng.integers(1,35,100).astype(float),event=rng.integers(0,3,100),groups=np.arange(100))
    ev=dict(X=X[:5],sample_ids=np.array([f'anonymous{i}' for i in range(5)]))
    cfg=replace(configuration(42,'s3'),n_trees=3,mtry=6,tree_feature_count=5,min_leaf=5,min_groups=3,min_events=1)
    raw=CompetingRiskForest(cfg).fit(tr['X'],tr['time'],tr['event'],tr['groups'])
    shrunk=shrink_leaf_cifs(raw,tr['X'],tr['time'],tr['event'],tr['groups'],10)
    folder=tmp_path/'model';emit(shrunk,tr,ev,folder,'s3',{})
    context=tmp_path/'context/audit';context.mkdir(parents=True)
    (context/'reused_artifacts.json').write_text(json.dumps([dict(year=2023,seed=42,arm='s3',path=str(folder))]))
    root=tmp_path/'input';fold=root/'folds/2023';fold.mkdir(parents=True)
    np.savez_compressed(fold/'train_PRIVATE.npz',**tr);np.savez_compressed(fold/'eval_PRIVATE.npz',**ev)
    out=tmp_path/'verified';out.mkdir()
    result=inspect(SimpleNamespace(run=tmp_path/'context',year=2023,seed=42,arm='s3'),root,out)
    assert result['verified_records']==105 and result['component_sum_error']<1e-10
    assert result['max_leaf_estimator_error']<1e-12
