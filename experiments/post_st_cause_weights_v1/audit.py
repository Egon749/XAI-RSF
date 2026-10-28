"""Post-fit audit of saved node statistics and complete three-seed ensembles; no fit."""
from dataclasses import asdict
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from crforest import CompetingRiskForest,ForestConfig,CIFExpansion,cluster_bootstrap,aalen_johansen
from experiments.post_st_prognosis_v1.common import arrays,sha,write
from experiments.post_st_rule_features_v1.audit_root_weights import independent_components


def audit(run):
    gate=json.loads((run/'audit/formal_gate.json').read_text());assert gate['status']=='PASS'
    tasks=[json.loads(x) for x in (run/'formal.jsonl').read_text().splitlines()]
    by={};checks=[];node_count=0;leaf_count=0
    for i,task in enumerate(tasks):
        argv=task['argv'];year=int(argv[argv.index('--year')+1]);variant=argv[argv.index('--variant')+1];seed=int(argv[argv.index('--seed')+1])
        folder=run/'results/formal'/str(gate['job'])/f'task-{i}'/'output'
        f=CompetingRiskForest.load(folder/'forest_PRIVATE.json');by.setdefault((year,variant),[]).append(f)
        tr=arrays(run/'input/folds'/str(year)/variant/'train_PRIVATE.npz');x,t,e,g=[tr[k] for k in ['X','time','event','groups']]
        max_score_error=0.;max_leaf_error=0.;stumps=0;no_target_trees=0
        for tree_id,tree in enumerate(f.trees_):
            rows=cluster_bootstrap(g,seed,tree_id);no_target_trees+=not np.any(e[rows]==1);stumps+='cif' in tree[0]
            def visit(index,ids):
                nonlocal node_count,leaf_count,max_score_error,max_leaf_error
                n=tree[index];node_count+=1
                assert n['n']==len(ids) and n['groups']==len(np.unique(g[ids]))
                for k in [1,2]:assert n['cause_company_support'][k-1]==len(np.unique(g[ids[e[ids]==k]]))
                if 'cif' in n:
                    leaf_count+=1;expected=aalen_johansen(t[ids],e[ids],f.grid_)
                    err=float(abs(expected-np.array(n['cif'])).max());max_leaf_error=max(max_leaf_error,err)
                    assert err<1e-14
                    return
                z=x[ids,n['feature']];left=(z<=n['threshold'])|(np.isnan(z)&n['missing_left'])
                u,v=independent_components(t[ids],e[ids],left);w=np.asarray(f.config.cause_weights)
                den=(w*w)@v;expected=abs(w@u)/np.sqrt(den) if den>1e-18 else 0.
                err=abs(float(expected)-n['score']);max_score_error=max(max_score_error,err)
                assert err<1e-10
                visit(n['left'],ids[left]);visit(n['right'],ids[~left])
            visit(0,rows)
        checks.append(dict(test_year=year,variant=variant,seed=seed,max_score_error=max_score_error,
            max_leaf_aj_error=max_leaf_error,stump_trees=stumps,trees_without_target_in_bootstrap=no_target_trees))
    out=run/'ensemble_private';out.mkdir(exist_ok=False)
    rows=[]
    for (year,variant),members in by.items():
        tr=arrays(run/'input/folds'/str(year)/variant/'train_PRIVATE.npz')
        ev=arrays(run/'evaluation_input'/str(year)/'test_PRIVATE.npz')
        assert len(members)==3 and all(np.array_equal(f.grid_,members[0].grid_) for f in members)
        f=CompetingRiskForest(ForestConfig(**{**asdict(members[0].config),'n_trees':96}))
        f.grid_=members[0].grid_;f.n_features_=members[0].n_features_;f.trees_=[tree for m in members for tree in m.trees_]
        f.candidate_evaluations_=sum(m.candidate_evaluations_ for m in members);f.audit_=f.audit()
        times=np.unique(np.r_[0.,f.grid_,365.,366.]);x=np.vstack([tr['X'],ev['X']])
        prediction=f.predict(x,times);average=np.mean([m.predict(x,times) for m in members],axis=0)
        np.testing.assert_allclose(prediction,average,atol=1e-14,rtol=0)
        assert prediction.min()>=-1e-12 and prediction.sum(-1).max()<=1+1e-10 and np.diff(prediction,axis=1).min()>=-1e-12
        ex=CIFExpansion.from_forest(f,tr['X']);cc=ex.complexity(tr['X'],support_min=10)
        error=float(abs(ex.predict(x,times)-average).max());assert error<1e-10
        path=out/f'{year}_{variant}_PRIVATE.npz';ex.save(path)
        write(out/f'{year}_{variant}_complexity_PRIVATE.json',cc)
        rows.append(dict(test_year=year,variant=variant,trees=96,reconstruction_error=error,
            **{k:v for k,v in cc.items() if k!='functions'},functions_sha256=sha(path)))
    pd.DataFrame(checks).to_csv(run/'evaluation/node_audit.csv',index=False)
    pd.DataFrame(rows).to_csv(run/'evaluation/ensemble_complexity.csv',index=False)
    write(run/'evaluation/independent_audit.json',dict(status='PASS',formal_models=18,nodes_checked=node_count,
        leaf_aj_checked=leaf_count,selected_scores_checked=node_count-leaf_count,
        all_company_bootstrap_and_node_support_reproduced=True,ensembles=6,new_fits=0,
        max_score_error=max(x['max_score_error'] for x in checks),max_leaf_aj_error=max(x['max_leaf_aj_error'] for x in checks),
        max_ensemble_all_native_time_reconstruction_error=max(x['reconstruction_error'] for x in rows)))
    print(pd.DataFrame(rows)[['test_year','variant','main_functions','interaction_functions','spatial_table_cells','reconstruction_error']].to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();audit(a.run)
