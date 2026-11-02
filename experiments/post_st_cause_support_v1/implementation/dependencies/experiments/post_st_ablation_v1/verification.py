"""Saved-node audit and complete function verification, without fitting trees."""
from dataclasses import asdict
from itertools import combinations
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
from crforest import CompetingRiskForest,ForestConfig,CIFExpansion,HigherOrderCIFExpansion
from crforest.core import _split_parent,logrank_cr_score
from crforest.partition import training_cut_dictionary
from experiments.post_st_leaf_shallow_v1.audit import direct_aj
from .common import *

def replay(f,tr):
    """Replay RNG/candidates at observed nodes. Never grow a replacement forest."""
    x,t,e,g=[tr[k] for k in ['X','time','event','groups']];cfg=f.config
    cuts=training_cut_dictionary(x,cfg);node_rows=[];feature_rows=[];pair_rows=[];leaf_rows=[]
    selected=np.zeros(x.shape[1],int);split_nodes=np.zeros_like(selected);split_trees=np.zeros_like(selected)
    pair_eligible=np.zeros((x.shape[1],x.shape[1]),int);pair_paths=np.zeros_like(pair_eligible)
    total_attempts=0;max_score_error=0.;max_leaf_error=0.;max_parent_error=0.
    for ti,tree in enumerate(f.trees_):
        eligible=tree_feature_subset(x.shape[1],cfg.tree_feature_count,cfg.seed,ti,cfg.tree_feature_strategy)
        selected[eligible]+=1;used_in_tree=set();paths_in_tree=set()
        for j,k in combinations(eligible,2):pair_eligible[j,k]+=1
        def visit(index,rows,used,depth,node_key,parent_curve=None):
            nonlocal total_attempts,max_score_error,max_leaf_error,max_parent_error
            n=tree[index];unique=len(np.unique(g[rows]));support=[int(len(np.unique(g[rows[e[rows]==k]]))) for k in [1,2]]
            assert n['groups']==unique and n['n']==len(rows) and n['cause_company_support']==support
            raw=direct_aj(t[rows],e[rows],f.grid_)
            if 'cif' in n:
                if 'leaf_shrinkage' in n:
                    sh=n['leaf_shrinkage'];alpha=1. if parent_curve is None else unique/(unique+10.)
                    prior=raw if parent_curve is None else parent_curve
                    assert sh['strength']==10 and sh['unique_inbag_companies']==unique and abs(sh['weight']-alpha)<1e-15
                    max_parent_error=max(max_parent_error,float(abs(np.asarray(sh['raw_parent_cif'])-prior).max()))
                    np.testing.assert_allclose(sh['raw_leaf_cif'],raw,atol=1e-13,rtol=0)
                    estimate=alpha*raw+(1-alpha)*prior
                else:alpha=1.;estimate=raw
                max_leaf_error=max(max_leaf_error,float(abs(np.asarray(n['cif'])-estimate).max()))
                leaf_rows.append(dict(tree=ti,node=index,unique_companies=unique,bootstrap_rows=len(rows),target_companies=support[0],recovery_companies=support[1],weight=alpha,borrowed_positive_target=bool(support[0]==0 and np.asarray(n['cif'])[:,0].max()>0)))
                paths_in_tree.update(combinations(sorted(used),2))
            stopping=(depth>=cfg.max_depth or len(rows)<2*cfg.min_leaf or n['events']<2*cfg.min_events or unique<2*cfg.min_groups)
            if not stopping:
                active=eligible if cfg.path_budget is None or len(used)<cfg.path_budget else np.array(sorted(used))
                rng=np.random.default_rng(np.random.SeedSequence([cfg.seed,ti,node_key,1]))
                features=rng.choice(active,min(cfg.mtry or int(np.ceil(np.sqrt(x.shape[1]))),len(active)),replace=False)
                parent=_split_parent(t[rows],e[rows],cfg.n_causes);best=0.;winner=None
                nr=dict(tree=ti,node=index,depth=depth,features_considered=len(features),finite_cuts=0,missing_only_cuts=0,routed_attempts=0,scored_candidates=0,features_with_fewer_than_five_cuts=0)
                for j in features:
                    z=x[rows,j];missing=np.isnan(z);vals=np.unique(z[~missing])
                    if cuts is None:
                        cc=vals[:-1]
                        if len(cc)>cfg.n_thresholds:cc=rng.choice(cc,cfg.n_thresholds,replace=False)
                    else:cc=cuts[j][(cuts[j]>=vals[0])&(cuts[j]<vals[-1])] if len(vals) else np.array([])
                    nr['finite_cuts']+=len(cc);nr['features_with_fewer_than_five_cuts']+=len(cc)<cfg.n_thresholds
                    candidates=[(float(c),route) for c in sorted(cc) for route in ([False,True] if missing.any() else [False])]
                    if missing.any() and len(vals):candidates.append((-np.inf,True));nr['missing_only_cuts']+=1
                    for c,ml in candidates:
                        nr['routed_attempts']+=1;left=(z<=c)|(missing&ml)
                        if min(left.sum(),(~left).sum())<cfg.min_leaf:continue
                        if any((e[rows[side]]>0).sum()<cfg.min_events for side in (left,~left)):continue
                        if any(len(np.unique(g[rows[side]]))<cfg.min_groups for side in (left,~left)):continue
                        nr['scored_candidates']+=1;score=logrank_cr_score(left,e[rows],parent,cfg.cause_weights)
                        if score>best+1e-14:best=score;winner=(int(j),c,bool(ml))
                if 'cif' in n:assert winner is None
                else:
                    assert winner==(n['feature'],n['threshold'],n['missing_left'])
                    max_score_error=max(max_score_error,abs(best-n['score']))
                total_attempts+=nr['routed_attempts'];node_rows.append(nr)
            if 'cif' not in n:
                j=n['feature'];split_nodes[j]+=1;used_in_tree.add(j)
                z=x[rows,j];left=(z<=n['threshold'])|(np.isnan(z)&n['missing_left'])
                visit(n['left'],rows[left],used|{j},depth+1,node_key*2,raw)
                visit(n['right'],rows[~left],used|{j},depth+1,node_key*2+1,raw)
        visit(0,cluster_bootstrap(g,cfg.seed,ti),set(),0,1)
        for j in used_in_tree:split_trees[j]+=1
        for j,k in paths_in_tree:pair_paths[j,k]+=1
    assert total_attempts==f.candidate_evaluations_,(total_attempts,f.candidate_evaluations_)
    assert max(max_leaf_error,max_parent_error)<1e-10 and max_score_error<1e-12
    for j in range(x.shape[1]):feature_rows.append(dict(feature=j,eligible_trees=int(selected[j]),split_trees=int(split_trees[j]),split_nodes=int(split_nodes[j])))
    for j,k in combinations(range(x.shape[1]),2):pair_rows.append(dict(feature_j=j,feature_k=k,eligible_trees=int(pair_eligible[j,k]),co_path_trees=int(pair_paths[j,k])))
    return node_rows,feature_rows,pair_rows,leaf_rows,dict(max_score_error=max_score_error,max_leaf_estimator_error=max_leaf_error,max_raw_parent_error=max_parent_error)

def merge(members):
    first=members[0];f=CompetingRiskForest(ForestConfig(**{**asdict(first.config),'n_trees':sum(len(m.trees_) for m in members)}))
    assert all(np.array_equal(m.grid_,first.grid_) for m in members)
    f.grid_=first.grid_;f.n_features_=first.n_features_;f.trees_=[t for m in members for t in m.trees_]
    f.candidate_evaluations_=sum(m.candidate_evaluations_ for m in members);f.audit_=f.audit();return f

def inspect(a,root,out):
    records=[r for r in registry(a.run) if r['year']==a.year and r['arm']==a.arm and (a.seed==0 or r['seed']==a.seed)]
    assert len(records)==(3 if a.seed==0 else 1)
    records=sorted(records,key=lambda r:r['seed']);members=[];sources=[]
    for r in records:
        p=Path(r['path']);members.append(CompetingRiskForest.load(p/'forest_PRIVATE.json'));sources.append(sha(p/'forest_PRIVATE.json'))
    f=members[0] if a.seed else merge(members)
    tr=arrays(root/'folds'/str(a.year)/'train_PRIVATE.npz');ev=arrays(root/'folds'/str(a.year)/'eval_PRIVATE.npz')
    X=np.vstack([tr['X'],ev['X']]);times=np.unique(np.r_[0.,f.grid_,TIMES]);start=time.perf_counter()
    ex=HigherOrderCIFExpansion.from_forest(f,tr['X']);compile_seconds=time.perf_counter()-start
    start=time.perf_counter();ex.save(out/'functions_PRIVATE.npz');export_seconds=time.perf_counter()-start
    again=HigherOrderCIFExpansion.load(out/'functions_PRIVATE.npz')
    start=time.perf_counter();cc=ex.complexity(tr['X'],support_min=10);complexity_seconds=time.perf_counter()-start
    write(out/'complexity_PRIVATE.json',cc)
    start=time.perf_counter();f.save(out/'forest_PRIVATE.json');loaded=CompetingRiskForest.load(out/'forest_PRIVATE.json')
    errors={k:0. for k in ['reconstruction_error','component_sum_error','function_reload_error','forest_reload_error','ensemble_average_error','q2_exporter_error']}
    minimum=0.;maxsum=0.;mindiff=0.;co=CIFExpansion.from_forest(f,tr['X']) if f.audit_['max_path_variables']<=2 else None
    # Every company, every native time and both causes. No row/time subsampling.
    for first in range(0,len(X),64):
        xx=X[first:first+64];p=f.predict(xx,times);z=ex.predict(xx,times)
        errors['reconstruction_error']=max(errors['reconstruction_error'],float(abs(z-p).max()))
        errors['function_reload_error']=max(errors['function_reload_error'],float(abs(again.predict(xx,times)-p).max()))
        errors['forest_reload_error']=max(errors['forest_reload_error'],float(abs(loaded.predict(xx,times)-p).max()))
        errors['ensemble_average_error']=max(errors['ensemble_average_error'],float(abs(np.mean([m.predict(xx,times) for m in members],axis=0)-p).max()))
        total=np.broadcast_to(ex._coeff(times)[1],p.shape).copy()
        for key in ex.keys:total+=ex.component(xx,key,times)
        errors['component_sum_error']=max(errors['component_sum_error'],float(abs(total-p).max()))
        if co is not None:errors['q2_exporter_error']=max(errors['q2_exporter_error'],float(abs(co.predict(xx,times)-p).max()))
        assert np.isfinite(p).all();minimum=min(minimum,float(p.min()));maxsum=max(maxsum,float(p.sum(-1).max()));mindiff=min(mindiff,float(np.diff(p,axis=1).min()))
    validation_seconds=time.perf_counter()-start
    assert max(errors.values())<=1e-10 and minimum>=-1e-12 and maxsum<=1+1e-12 and mindiff>=-1e-12
    expected=np.mean([arrays(Path(r['path'])/'predictions_PRIVATE.npz')['cif'] for r in records],axis=0)
    np.testing.assert_allclose(f.predict(ev['X'],TIMES),expected,atol=1e-14,rtol=0)
    # Training-only joint rows for function comparisons; product reference unchanged.
    start=time.perf_counter();terms=np.stack([ex.component(tr['X'],k,[365.,730.]) for k in ex.keys],axis=1) if ex.keys else np.zeros((len(tr['X']),0,2,2))
    np.savez_compressed(out/'components_PRIVATE.npz',keys=np.array([','.join(map(str,k)) for k in ex.keys]),terms=terms,times=[365.,730.])
    order=ex.predict_by_order(X,[365.,730.]);order_rows=[]
    for h in range(2):
        for c in range(2):
            for r in range(ex.max_order+1):
                v=order[:,r,h,c];order_rows.append(dict(display_day=[365,730][h],cause=c+1,order=r,mean_abs=float(abs(v).mean()),max_abs=float(abs(v).max()),rms=float(np.sqrt(np.mean(v*v)))))
    pd.DataFrame(order_rows).to_csv(out/'order_amplitudes.csv',index=False)
    q2_residual=float(abs(order[:,3:].sum(1)).max()) if ex.max_order>2 else 0.
    component_seconds=time.perf_counter()-start
    node_audit={};replay_seconds=0.
    if a.seed:
        start=time.perf_counter();nodes,features,pairs,leaves,node_audit=replay(f,tr);replay_seconds=time.perf_counter()-start
        for name,rows in [('node_search',nodes),('feature_coverage',features),('pair_coverage',pairs),('leaf_support',leaves)]:pd.DataFrame(rows).to_csv(out/f'{name}_PRIVATE.csv',index=False)
        nd=pd.DataFrame(nodes);ld=pd.DataFrame(leaves)
        node_audit.update(**{k:int(nd[k].sum()) for k in ['finite_cuts','missing_only_cuts','routed_attempts','scored_candidates','features_considered','features_with_fewer_than_five_cuts']},
            zero_target_leaves=int((ld.target_companies==0).sum()),one_or_zero_target_leaves=int((ld.target_companies<=1).sum()),
            borrowed_positive_target_leaves=int(ld.borrowed_positive_target.sum()),unique_companies_min=int(ld.unique_companies.min()),unique_companies_median=float(ld.unique_companies.median()),
            target_companies_median=float(ld.target_companies.median()),recovery_companies_median=float(ld.recovery_companies.median()))
        write(out/'pairing.json',pairing(tr['groups'],f.config))
    return dict(year=a.year,arm=a.arm,seed='ensemble' if a.seed==0 else str(a.seed),new_forest_fits=0,
        source_model_sha256=sources,**f.audit_,**{k:v for k,v in cc.items() if k!='functions'},**errors,**node_audit,
        probability_min=minimum,probability_sum_max=maxsum,probability_min_increment=mindiff,
        q2_truncation_max_abs_diagnostic=q2_residual,function_file_bytes=(out/'functions_PRIVATE.npz').stat().st_size,
        function_sha256=sha(out/'functions_PRIVATE.npz'),function_compile_seconds=compile_seconds,function_export_seconds=export_seconds,
        complexity_seconds=complexity_seconds,full_verification_seconds=validation_seconds,component_seconds=component_seconds,search_replay_seconds=replay_seconds,
        verified_records=len(X),verified_times=len(times),native_time_count=len(f.grid_))
