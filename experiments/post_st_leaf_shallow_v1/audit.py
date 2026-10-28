"""Independent AJ replay, paired geometry checks and full 96-tree explanations."""
from dataclasses import asdict
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from crforest import CompetingRiskForest,ForestConfig,CIFExpansion,cluster_bootstrap
from experiments.post_st_prognosis_v1.common import arrays,sha,write
from experiments.post_st_rule_features_v1.audit_root_weights import independent_components
from .analyze import artifacts


def direct_aj(t,e,grid):
    """Explicit event/risk-set recursion, separate from production AJ routine."""
    survival=1.;cum=np.zeros(2);result=[]
    for point in grid:
        risk=int((t>=point).sum());deaths=np.array([((t==point)&(e==k)).sum() for k in [1,2]])
        if risk:
            cum=cum+survival*deaths/risk;survival*=1-deaths.sum()/risk
        result.append(cum.copy())
    return np.array(result)


def audit(run):
    by={};models={};checks=[];total_nodes=0;total_leaves=0;standard=0
    for year,arm,seed,folder in artifacts(run):
        tr=arrays(run/'input/folds'/str(year)/'train_PRIVATE.npz');x,t,e,g=[tr[k] for k in ['X','time','event','groups']]
        if arm=='standard_rsf':
            expected=np.column_stack([np.bincount(cluster_bootstrap(g,int(seed),i),minlength=len(g)) for i in range(32)])
            np.testing.assert_array_equal(np.loadtxt(folder/'inbag_PRIVATE.csv',delimiter=','),expected)
            med=np.array([np.median(v[np.isfinite(v)]) if np.isfinite(v).any() else 0. for v in x.T])
            np.testing.assert_array_equal(med,np.load(folder/'training_imputation_PRIVATE.npy'));standard+=1;continue
        f=CompetingRiskForest.load(folder/'forest_PRIVATE.json');by.setdefault((year,arm),[]).append(f);models[(year,arm,seed)]=f
        assert f.config.path_budget==2 and f.config.n_trees==32 and tuple(f.config.cause_weights)==(1.,1.)
        maxscore=0.;maxleaf=0.;stumps=0;no_target=0;nodes=0;leaves=0;zero=0;one=0;borrowed=0;weights=[]
        for tree_id,tree in enumerate(f.trees_):
            sampled=cluster_bootstrap(g,int(seed),tree_id);no_target+=not np.any(e[sampled]==1);stumps+='cif' in tree[0]
            def visit(index,ids,parent_index=None,parent_aj=None):
                nonlocal nodes,leaves,maxscore,maxleaf,zero,one,borrowed
                node=tree[index];nodes+=1
                assert node['n']==len(ids) and node['groups']==len(np.unique(g[ids]))
                for k in [1,2]:assert node['cause_company_support'][k-1]==len(np.unique(g[ids[e[ids]==k]]))
                raw=direct_aj(t[ids],e[ids],f.grid_)
                if 'cif' in node:
                    leaves+=1;support=node['cause_company_support'][0];zero+=support==0;one+=support<=1
                    expected=raw;weight=1.
                    if 'leaf_shrinkage' in node:
                        meta=node['leaf_shrinkage'];assert arm in ['parent_shrink','shallow_parent']
                        assert meta['strength']==10 and meta['parent_node']==parent_index
                        weight=1. if parent_aj is None else len(np.unique(g[ids]))/(len(np.unique(g[ids]))+10.)
                        assert meta['weight']==weight
                        np.testing.assert_allclose(meta['raw_leaf_cif'],raw,atol=1e-14,rtol=0)
                        prior=raw if parent_aj is None else parent_aj
                        np.testing.assert_allclose(meta['raw_parent_cif'],prior,atol=1e-14,rtol=0)
                        expected=weight*raw+(1-weight)*prior
                    else:assert arm in ['baseline','shallow']
                    weights.append(weight);actual=np.asarray(node['cif'])
                    err=float(abs(expected-actual).max());maxleaf=max(maxleaf,err);assert err<1e-14
                    borrowed+=support==0 and actual[:,0].max()>0
                    return
                z=x[ids,node['feature']];left=(z<=node['threshold'])|(np.isnan(z)&node['missing_left'])
                u,v=independent_components(t[ids],e[ids],left);w=np.asarray(f.config.cause_weights);den=(w*w)@v
                expected=abs(w@u)/np.sqrt(den) if den>1e-18 else 0.
                err=abs(float(expected)-node['score']);maxscore=max(maxscore,err);assert err<1e-10
                visit(node['left'],ids[left],index,raw);visit(node['right'],ids[~left],index,raw)
            visit(0,sampled)
        total_nodes+=nodes;total_leaves+=leaves
        checks.append(dict(test_year=year,arm=arm,seed=seed,nodes=nodes,leaves=leaves,max_score_error=maxscore,max_leaf_estimator_error=maxleaf,
            stump_trees=stumps,trees_without_target_in_bootstrap=no_target,zero_target_leaves=zero,at_most_one_target_leaves=one,
            zero_target_leaf_fraction=zero/leaves,at_most_one_target_leaf_fraction=one/leaves,
            zero_target_leaves_with_borrowed_risk=borrowed,weight_min=min(weights),weight_median=float(np.median(weights)),weight_max=max(weights)))
    matched=[]
    for year in [2023,2024]:
        for seed in ['42','101','202']:
            baseline=models[(year,'baseline',seed)];shallow=models[(year,'shallow',seed)];prefix_nodes=0
            for rawarm,shrarm in [('baseline','parent_shrink'),('shallow','shallow_parent')]:
                a=models[(year,rawarm,seed)];b=models[(year,shrarm,seed)]
                assert asdict(a.config)==asdict(b.config)
                stripped=lambda tree:[{k:v for k,v in n.items() if k not in ['cif','leaf_shrinkage']} for n in tree]
                assert [stripped(t) for t in a.trees_]==[stripped(t) for t in b.trees_]
            ca,cb=asdict(baseline.config),asdict(shallow.config);ca.pop('max_depth');cb.pop('max_depth');assert ca==cb
            for ta,tb in zip(baseline.trees_,shallow.trees_):
                def compare(ia,ib):
                    nonlocal prefix_nodes
                    a,b=ta[ia],tb[ib];prefix_nodes+=1
                    for k in ['ancestors','depth','n','groups','events','cause_event_rows','cause_company_support']:assert a[k]==b[k]
                    if 'cif' in a:
                        assert 'cif' in b;np.testing.assert_array_equal(a['cif'],b['cif']);return
                    if b['depth']==4:assert 'cif' in b;return
                    for k in ['feature','threshold','missing_left','score']:assert a[k]==b[k]
                    compare(a['left'],b['left']);compare(a['right'],b['right'])
                compare(0,0)
            matched.append(dict(test_year=year,seed=seed,shrunk_geometry_identical=True,depth4_prefix_identical=True,prefix_nodes_checked=prefix_nodes))
    out=run/'ensemble_private';out.mkdir(exist_ok=False);rows=[]
    for (year,arm),members in by.items():
        tr=arrays(run/'input/folds'/str(year)/'train_PRIVATE.npz');ev=arrays(run/'evaluation_input'/f'{year}_test_PRIVATE.npz')
        assert len(members)==3 and all(np.array_equal(f.grid_,members[0].grid_) for f in members)
        f=CompetingRiskForest(ForestConfig(**{**asdict(members[0].config),'n_trees':96}))
        f.grid_=members[0].grid_;f.n_features_=members[0].n_features_;f.trees_=[tree for m in members for tree in m.trees_]
        f.candidate_evaluations_=sum(m.candidate_evaluations_ for m in members);f.audit_=f.audit()
        times=np.unique(np.r_[0.,f.grid_,365.,366.,730.,731.]);x=np.vstack([tr['X'],ev['X']])
        pred=f.predict(x,times);average=np.mean([m.predict(x,times) for m in members],axis=0)
        np.testing.assert_allclose(pred,average,atol=1e-14,rtol=0)
        assert pred.min()>=-1e-12 and pred.sum(-1).max()<=1+1e-10 and np.diff(pred,axis=1).min()>=-1e-12
        ex=CIFExpansion.from_forest(f,tr['X']);cc=ex.complexity(tr['X'],support_min=10)
        error=float(abs(ex.predict(x,times)-average).max());assert error<1e-10
        path=out/f'{year}_{arm}_PRIVATE.npz';ex.save(path)
        reload_error=float(abs(CIFExpansion.load(path).predict(x,times)-average).max());assert reload_error<1e-10
        write(out/f'{year}_{arm}_complexity_PRIVATE.json',cc)
        rows.append(dict(test_year=year,arm=arm,trees=96,reconstruction_error=error,reload_error=reload_error,
            **{k:v for k,v in cc.items() if k!='functions'},functions_sha256=sha(path)))
    pd.DataFrame(checks).to_csv(run/'evaluation/node_audit.csv',index=False)
    pd.DataFrame(matched).to_csv(run/'evaluation/matched_geometry.csv',index=False)
    pd.DataFrame(rows).to_csv(run/'evaluation/ensemble_complexity.csv',index=False)
    write(run/'evaluation/independent_audit.json',dict(status='PASS',q2_models=len(checks),standard_RSF_bootstraps_checked=standard,
        nodes_checked=total_nodes,leaf_estimators_checked=total_leaves,selected_scores_checked=total_nodes-total_leaves,
        ensembles=len(rows),new_fits=0,matched_seed_pairs=len(matched),max_score_error=max(x['max_score_error'] for x in checks),
        max_leaf_estimator_error=max(x['max_leaf_estimator_error'] for x in checks),
        max_ensemble_all_native_time_reconstruction_error=max(x['reconstruction_error'] for x in rows)))
    print(pd.DataFrame(rows)[['test_year','arm','main_functions','interaction_functions','spatial_table_cells','reconstruction_error']].to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);audit(p.parse_args().run)
