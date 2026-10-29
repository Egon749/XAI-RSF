"""Independently replay leaf AJ and retain the complete 96-tree function table."""
from dataclasses import asdict
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from crforest import CompetingRiskForest,ForestConfig,CIFExpansion,cluster_bootstrap
from experiments.post_st_prognosis_v1.common import arrays,sha,write
from experiments.post_st_leaf_shallow_v1.audit import direct_aj
from experiments.post_st_2025_shallow_parent_v1.analyze import outputs


def audit(run):
    tr=arrays(run/'input/folds/2025/train_PRIVATE.npz');ev=arrays(run/'evaluation_input/2025_test_PRIVATE.npz')
    x,t,e,g=[tr[k] for k in ['X','time','event','groups']]
    members=[];checks=[];standard=0
    for model,seed,folder in outputs(run):
        if model=='standard_rsf':
            expected=np.column_stack([np.bincount(cluster_bootstrap(g,int(seed),i),minlength=len(g)) for i in range(32)])
            np.testing.assert_array_equal(np.loadtxt(folder/'inbag_PRIVATE.csv',delimiter=','),expected)
            med=np.array([np.median(v[np.isfinite(v)]) if np.isfinite(v).any() else 0. for v in x.T])
            np.testing.assert_array_equal(med,np.load(folder/'training_imputation_PRIVATE.npy'));standard+=1;continue
        f=CompetingRiskForest.load(folder/'forest_PRIVATE.json');rawf=CompetingRiskForest.load(folder.parent/'raw/forest_PRIVATE.json')
        assert f.config.n_trees==32 and f.config.max_depth==4 and f.config.path_budget==2
        assert tuple(f.config.cause_weights)==(1.,1.) and asdict(f.config)==asdict(rawf.config)
        stripped=lambda tree:[{k:v for k,v in node.items() if k not in ['cif','leaf_shrinkage']} for node in tree]
        assert [stripped(z) for z in f.trees_]==[stripped(z) for z in rawf.trees_]
        nodes=0;leaves=0;max_error=0.;unsupported=0;borrowed=0;low_support=0
        for i,tree in enumerate(f.trees_):
            sample=cluster_bootstrap(g,int(seed),i)
            def visit(index,ids,parent_index=None,parent_cif=None):
                nonlocal nodes,leaves,max_error,unsupported,borrowed,low_support
                node=tree[index];nodes+=1
                assert node['n']==len(ids) and node['groups']==len(np.unique(g[ids]))
                for k in [1,2]:assert node['cause_company_support'][k-1]==len(np.unique(g[ids[e[ids]==k]]))
                raw=direct_aj(t[ids],e[ids],f.grid_)
                if 'cif' in node:
                    leaves+=1;info=node['leaf_shrinkage'];assert info['strength']==10 and info['parent_node']==parent_index
                    n=len(np.unique(g[ids]));w=1. if parent_cif is None else n/(n+10.)
                    assert info['weight']==w
                    prior=raw if parent_cif is None else parent_cif
                    np.testing.assert_allclose(info['raw_leaf_cif'],raw,atol=1e-14,rtol=0)
                    np.testing.assert_allclose(info['raw_parent_cif'],prior,atol=1e-14,rtol=0)
                    actual=np.asarray(node['cif']);err=float(abs(actual-(w*raw+(1-w)*prior)).max())
                    max_error=max(max_error,err);assert err<1e-14
                    support=node['cause_company_support'][0];unsupported+=support==0;low_support+=support<=1
                    borrowed+=support==0 and actual[:,0].max()>0
                    return
                z=x[ids,node['feature']];left=(z<=node['threshold'])|(np.isnan(z)&node['missing_left'])
                visit(node['left'],ids[left],index,raw);visit(node['right'],ids[~left],index,raw)
            visit(0,sample)
        checks.append(dict(seed=seed,nodes=nodes,leaves=leaves,zero_target_leaves=unsupported,at_most_one_target_leaves=low_support,
            zero_target_leaves_with_borrowed_risk=borrowed,max_independent_leaf_error=max_error,raw_and_shrunk_geometry_identical=True))
        members.append(f)
    assert len(members)==standard==3 and all(np.array_equal(m.grid_,members[0].grid_) for m in members)
    # Algebraic concatenation of saved trees, not another fitted or selected model.
    forest=CompetingRiskForest(ForestConfig(**{**asdict(members[0].config),'n_trees':96}))
    forest.grid_=members[0].grid_;forest.n_features_=83;forest.trees_=[t for m in members for t in m.trees_]
    forest.candidate_evaluations_=sum(m.candidate_evaluations_ for m in members);forest.audit_=forest.audit()
    X=np.vstack([x,ev['X']]);times=np.unique(np.r_[0.,forest.grid_,365.,366.,730.,731.])
    predicted=forest.predict(X,times);mean=np.mean([m.predict(X,times) for m in members],axis=0)
    np.testing.assert_allclose(predicted,mean,atol=1e-14,rtol=0)
    assert predicted.min()>=-1e-12 and predicted.sum(-1).max()<=1+1e-10 and np.diff(predicted,axis=1).min()>=-1e-12
    expansion=CIFExpansion.from_forest(forest,x);cc=expansion.complexity(x,support_min=10)
    error=float(abs(expansion.predict(X,times)-mean).max());assert error<1e-10
    folder=run/'ensemble_private';folder.mkdir(exist_ok=False)
    expansion.save(folder/'complete_functions_PRIVATE.npz')
    reload_error=float(abs(CIFExpansion.load(folder/'complete_functions_PRIVATE.npz').predict(X,times)-mean).max());assert reload_error<1e-10
    forest.save(folder/'algebraic_96_tree_ensemble_PRIVATE.json');write(folder/'complexity_PRIVATE.json',cc)
    result=dict(status='PASS',saved_models_replayed=3,standard_RSF_bootstraps_checked=standard,
        nodes_checked=sum(z['nodes'] for z in checks),leaf_estimators_checked=sum(z['leaves'] for z in checks),
        max_independent_leaf_error=max(z['max_independent_leaf_error'] for z in checks),
        all_443_records_all_native_times=True,native_and_horizon_time_points=len(times),
        all_time_ensemble_reconstruction_error=error,all_time_function_reload_error=reload_error,
        new_fits=0,full_functions_sha256=sha(folder/'complete_functions_PRIVATE.npz'))
    pd.DataFrame(checks).to_csv(run/'evaluation/node_audit.csv',index=False)
    pd.DataFrame([dict(trees=96,**{k:v for k,v in cc.items() if k!='functions'},reconstruction_error=error,reload_error=reload_error)]).to_csv(run/'evaluation/ensemble_complexity.csv',index=False)
    write(run/'evaluation/independent_audit.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);audit(p.parse_args().run)
