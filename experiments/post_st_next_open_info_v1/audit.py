"""Independently replay node statistics and compile complete 96-tree functions."""
from dataclasses import asdict
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from crforest import CompetingRiskForest,ForestConfig,CIFExpansion,cluster_bootstrap,aalen_johansen
from experiments.post_st_prognosis_v1.common import arrays,sha,write
from experiments.post_st_rule_features_v1.audit_root_weights import independent_components
from .analyze import outputs


def audit(run):
    by={};checks=[];nodes=0;leaves=0;standard_checked=0
    for year,variant,model,seed,folder in outputs(run):
        tr=arrays(run/'input/folds'/str(year)/variant/'train_PRIVATE.npz');x,t,e,g=[tr[k] for k in ['X','time','event','groups']]
        if model=='standard_rsf':
            expected=np.column_stack([np.bincount(cluster_bootstrap(g,int(seed),i),minlength=len(g)) for i in range(32)])
            np.testing.assert_array_equal(np.loadtxt(folder/'inbag_PRIVATE.csv',delimiter=','),expected)
            med=np.array([np.median(v[np.isfinite(v)]) if np.isfinite(v).any() else 0. for v in x.T])
            np.testing.assert_array_equal(med,np.load(folder/'training_imputation_PRIVATE.npy'))
            standard_checked+=1;continue
        f=CompetingRiskForest.load(folder/'forest_PRIVATE.json');by.setdefault((year,variant),[]).append(f)
        maxscore=0.;maxleaf=0.;stumps=0;no_target=0
        for tree_id,tree in enumerate(f.trees_):
            rows=cluster_bootstrap(g,int(seed),tree_id);no_target+=not np.any(e[rows]==1);stumps+='cif' in tree[0]
            def visit(index,ids):
                nonlocal nodes,leaves,maxscore,maxleaf
                n=tree[index];nodes+=1
                assert n['n']==len(ids) and n['groups']==len(np.unique(g[ids]))
                for k in [1,2]:assert n['cause_company_support'][k-1]==len(np.unique(g[ids[e[ids]==k]]))
                if 'cif' in n:
                    leaves+=1;err=float(abs(aalen_johansen(t[ids],e[ids],f.grid_)-np.array(n['cif'])).max());maxleaf=max(maxleaf,err)
                    assert err<1e-14;return
                z=x[ids,n['feature']];left=(z<=n['threshold'])|(np.isnan(z)&n['missing_left'])
                u,v=independent_components(t[ids],e[ids],left);w=np.asarray(f.config.cause_weights)
                den=(w*w)@v;expected=abs(w@u)/np.sqrt(den) if den>1e-18 else 0.
                err=abs(float(expected)-n['score']);maxscore=max(maxscore,err);assert err<1e-10
                visit(n['left'],ids[left]);visit(n['right'],ids[~left])
            visit(0,rows)
        checks.append(dict(test_year=year,variant=variant,seed=seed,max_score_error=maxscore,max_leaf_aj_error=maxleaf,
                           stump_trees=stumps,trees_without_target_in_bootstrap=no_target))
    out=run/'ensemble_private';out.mkdir(exist_ok=False);rows=[]
    for (year,variant),members in by.items():
        tr=arrays(run/'input/folds'/str(year)/variant/'train_PRIVATE.npz')
        ev=arrays(run/'evaluation_input'/f'{year}_{variant}_test_PRIVATE.npz')
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
        path=out/f'{year}_{variant}_PRIVATE.npz';ex.save(path)
        reload_error=float(abs(CIFExpansion.load(path).predict(x,times)-average).max());assert reload_error<1e-10
        write(out/f'{year}_{variant}_complexity_PRIVATE.json',cc)
        rows.append(dict(test_year=year,variant=variant,trees=96,reconstruction_error=error,
            reload_error=reload_error,**{k:v for k,v in cc.items() if k!='functions'},functions_sha256=sha(path)))
    pd.DataFrame(checks).to_csv(run/'evaluation/node_audit.csv',index=False)
    pd.DataFrame(rows).to_csv(run/'evaluation/ensemble_complexity.csv',index=False)
    write(run/'evaluation/independent_audit.json',dict(status='PASS',q2_models=len(checks),standard_RSF_bootstraps_checked=standard_checked,
        nodes_checked=nodes,leaf_aj_checked=leaves,selected_scores_checked=nodes-leaves,ensembles=len(rows),new_fits=0,
        max_score_error=max(x['max_score_error'] for x in checks),max_leaf_aj_error=max(x['max_leaf_aj_error'] for x in checks),
        max_ensemble_all_native_time_reconstruction_error=max(x['reconstruction_error'] for x in rows)))
    print(pd.DataFrame(rows)[['test_year','variant','main_functions','interaction_functions','spatial_table_cells','reconstruction_error']].to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);audit(p.parse_args().run)
