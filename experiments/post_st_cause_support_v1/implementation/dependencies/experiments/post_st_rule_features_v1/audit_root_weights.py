"""Training-only score replay; never grows, refits or replaces a tree.

Uses the frozen q2 shared cuts and original bootstrap samples. All three
weight choices score the SAME admissible candidates. Thresholds stay private.
"""
from pathlib import Path
import argparse
import json
import time
import numpy as np
import pandas as pd
from crforest.core import _split_parent, logrank_cr_score, cluster_bootstrap
from crforest.partition import tree_feature_subset, shared_quantile_cuts
from experiments.post_st_prognosis_v1.common import arrays, sha, write

WEIGHTS={'combined':np.array([1.,1.]),'termination_only':np.array([1.,0.]),
         'recovery_only':np.array([0.,1.])}


def components(left,event,parent):
    bins,risk,deaths,inclusive=parent
    lc=np.bincount(bins,weights=left,minlength=len(risk));lr=np.cumsum(lc[::-1])[::-1]
    ld=np.array([np.bincount(bins,weights=left&(event==k+1),minlength=len(risk)) for k in range(len(deaths))])
    lo=ld.sum(0)[None,:]-ld
    li=lr[None,:]+np.cumsum(lo,axis=1)-lo
    prop=np.divide(li,inclusive,out=np.zeros_like(li),where=inclusive>0)
    u=(ld-deaths*prop).sum(1)
    correction=np.divide(inclusive-deaths,inclusive-1,out=np.zeros_like(inclusive),where=inclusive>1)
    v=(deaths*prop*(1-prop)*correction*(risk[None,:]>=2)).sum(1)
    return u,v


def independent_components(t,e,left):
    u=np.zeros(2);v=np.zeros(2)
    for k in (1,2):
        for s in np.unique(t[e>0]):
            at_risk=(t>=s)|((t<s)&(e>0)&(e!=k))
            y=int(at_risk.sum());yl=int((at_risk&left).sum())
            d=int(((t==s)&(e==k)).sum());dl=int(((t==s)&(e==k)&left).sum())
            u[k-1]+=dl-d*yl/y
            if (t>=s).sum()>=2 and y>1:v[k-1]+=d*yl/y*(1-yl/y)*(y-d)/(y-1)
    return u,v


def scored_candidates(X,t,e,groups,cuts,feature_order,cfg):
    parent=_split_parent(t,e,2);records=[];checks=0
    for j in feature_order:
        x=X[:,j];missing=np.isnan(x);values=np.unique(x[~missing])
        valid_cuts=cuts[j][(cuts[j]>=values[0])&(cuts[j]<values[-1])] if len(values) else []
        proposed=[(float(c),route) for c in sorted(valid_cuts) for route in ([False,True] if missing.any() else [False])]
        if missing.any() and len(values):proposed.append((-np.inf,True))
        for cut,route in proposed:
            left=(x<=cut)|(missing&route)
            nl,nr=int(left.sum()),int((~left).sum())
            if min(nl,nr)<cfg['min_leaf']:continue
            if any((e[side]>0).sum()<cfg['min_events'] for side in (left,~left)):continue
            if any(len(np.unique(groups[side]))<cfg['min_groups'] for side in (left,~left)):continue
            u,v=components(left,e,parent)
            row=dict(feature=int(j),threshold=cut,missing_left=route,
                U_termination=float(u[0]),U_recovery=float(u[1]),
                V_termination=float(v[0]),V_recovery=float(v[1]),
                opposite_sign=bool(u[0]*u[1]<0),
                cancellation_fraction=float(1-abs(u.sum())/abs(u).sum()) if abs(u).sum()>0 else np.nan,
                termination_variance_share=float(v[0]/v.sum()) if v.sum()>0 else np.nan,
                left_rows=nl,right_rows=nr)
            for name,w in WEIGHTS.items():
                den=(w*w)@v;score=float(abs(w@u)/np.sqrt(den)) if den>1e-18 else 0.
                actual=logrank_cr_score(left,e,parent,w)
                np.testing.assert_allclose(score,actual,rtol=0,atol=2e-14)
                row[name]=score;checks+=1
            # Separate numerator cancellation from the added recovery variance.
            row['signed_numerator_factor']=float(abs(u.sum())/abs(u[0])) if u[0]!=0 else np.nan
            row['variance_factor']=float(np.sqrt(v[0]/v.sum())) if v.sum()>0 else np.nan
            records.append(row)
    return pd.DataFrame(records),checks


def pick(frame,weight):
    best=0.;winner=None
    for ix,row in frame.iterrows():
        if row[weight]>best+1e-14:best=float(row[weight]);winner=ix
    return winner


def feature_scores(frame,weight,names):
    rows=[]
    for j,g in frame.groupby('feature',sort=True):
        ix=pick(g,weight)
        if ix is None:ix=g.index[0]
        row=g.loc[ix].to_dict();row.update(feature_name=names[j],weight=weight,
            best_score=float(row[weight]),candidate_index=int(ix),admissible_candidates=len(g))
        rows.append(row)
    result=pd.DataFrame(rows)
    result['rank']=[int(1+(result.best_score>x+1e-12).sum()) for x in result.best_score]
    result['rankable_features']=len(result)
    return result.sort_values(['rank','feature'],kind='stable')


def validate_selected(frame,indices,X,t,e):
    for ix in sorted(set(indices)):
        r=frame.loc[ix];x=X[:,int(r.feature)]
        left=(x<=r.threshold)|(np.isnan(x)&bool(r.missing_left))
        u,v=independent_components(t,e,left)
        np.testing.assert_allclose(u,[r.U_termination,r.U_recovery],rtol=0,atol=2e-12)
        np.testing.assert_allclose(v,[r.V_termination,r.V_recovery],rtol=0,atol=2e-12)
    return len(set(indices))


def run(base,out):
    started=time.perf_counter();out.mkdir(exist_ok=True)
    private=out/'private';private.mkdir(exist_ok=False);reports=out/'reports';reports.mkdir(exist_ok=False)
    old=json.loads(Path('experiments/post_st_prognosis_v1/execution_receipt.json').read_text())
    train_path=base/'input/train_PRIVATE.npz';assert sha(train_path)==old['model_input_hashes'][train_path.name]
    tr=arrays(train_path);assert len(tr['X'])==309 and (tr['event']==1).sum()==14 and (tr['event']==2).sum()==184
    assert tr['year'].max()<=2022 and np.all(tr['label_eligible'])
    names=pd.read_csv(base/'reports/feature_manifest.csv').name.tolist();price=names.index('log_price_last')
    model_paths=[p for p in sorted((base/'results/formal/2534').glob('task-*/output/forest_PRIVATE.json'))
                 if json.loads(p.read_text())['config']['path_budget']==2]
    assert len(model_paths)==3
    f=json.loads(model_paths[0].read_text());cfg=f['config'];cuts=tuple(np.asarray(c) for c in f['shared_cuts'])
    for a,b in zip(cuts,shared_quantile_cuts(tr['X'],cfg['n_thresholds'])):np.testing.assert_array_equal(a,b)
    source_hashes={str(train_path):sha(train_path),str(base/'reports/feature_manifest.csv'):sha(base/'reports/feature_manifest.csv')}
    full,checks=scored_candidates(tr['X'],tr['time'],tr['event'],tr['groups'],cuts,np.arange(len(names)),cfg)
    full.to_parquet(private/'all_full_cohort_candidates_PRIVATE.parquet',index=False)
    ranks=pd.concat([feature_scores(full,w,names) for w in WEIGHTS],ignore_index=True)
    ranks.to_parquet(private/'all_feature_best_candidates_PRIVATE.parquet',index=False)
    public_columns=[c for c in ranks if c not in ['threshold','missing_left','candidate_index']]
    ranks[public_columns].to_csv(reports/'full_cohort_variable_ranking.csv',index=False)
    price_rows=ranks[ranks.feature.eq(price)].copy();price_rows[public_columns].to_csv(reports/'price_best_by_weight.csv',index=False)
    independent=validate_selected(full,list(price_rows.candidate_index)+list(ranks[ranks['rank'].eq(1)].candidate_index),tr['X'],tr['time'],tr['event'])
    pc=full[full.feature.eq(price)].drop(columns=['threshold','missing_left']).copy()
    pc.insert(0,'private_candidate_index',full.index[full.feature.eq(price)])
    pc.to_csv(reports/'price_same_candidate_scores.csv',index=False)
    root_rows=[];root_verified=0;root_details=[]
    for path in model_paths:
        f=json.loads(path.read_text());cfg=f['config'];seed=cfg['seed']
        expect=next(x for x in old['model_hashes'] if x['model']=='xai_q2' and x['seed']==seed)
        assert sha(path)==expect['model_sha256'];source_hashes[str(path)]=sha(path)
        for tid,nodes in enumerate(f['trees']):
            rows=cluster_bootstrap(tr['groups'],seed,tid)
            X=tr['X'][rows];t=tr['time'][rows];e=tr['event'][rows];groups=tr['groups'][rows]
            eligible=tree_feature_subset(len(names),cfg['tree_feature_count'],seed,tid)
            rng=np.random.default_rng(np.random.SeedSequence([seed,tid,1,1]))
            order=rng.choice(eligible,min(cfg['mtry'],len(eligible)),replace=False)
            all_order=np.r_[order,np.setdiff1d(np.arange(len(names)),order)]
            a,ncheck=scored_candidates(X,t,e,groups,cuts,all_order,cfg);checks+=ncheck
            offered=a[a.feature.isin(order)]
            current=offered.loc[pick(offered,'combined')];root=nodes[0]
            assert int(current.feature)==root['feature'] and current.threshold==root['threshold'] and bool(current.missing_left)==root['missing_left']
            np.testing.assert_allclose(current.combined,root['score'],rtol=0,atol=2e-12);root_verified+=1
            wanted=[int(current.name)]
            for scope,frame in [('actual_eligible',offered),('all_75',a)]:
                for weight in WEIGHTS:
                    ranked=feature_scores(frame,weight,names);winner=frame.loc[pick(frame,weight)]
                    pr=ranked[ranked.feature.eq(price)]
                    row=dict(seed=seed,tree=tid,scope=scope,weight=weight,price_eligible=bool(price in eligible),
                        winner_feature=names[int(winner.feature)],winner_is_price=int(winner.feature)==price,
                        winner_opposite_sign=bool(winner.opposite_sign),winner_cancellation=float(winner.cancellation_fraction),
                        price_rank=float(pr['rank'].iloc[0]) if len(pr) else np.nan,
                        price_score=float(pr.best_score.iloc[0]) if len(pr) else np.nan,
                        price_opposite_sign=bool(pr.opposite_sign.iloc[0]) if len(pr) else None,
                        price_cancellation=float(pr.cancellation_fraction.iloc[0]) if len(pr) else np.nan,
                        rankable_features=len(ranked))
                    root_rows.append(row)
                    if scope=='actual_eligible' and len(pr):wanted.append(int(pr.candidate_index.iloc[0]))
            independent+=validate_selected(a,wanted,X,t,e)
            a['seed']=seed;a['tree']=tid;a['actual_eligible']=a.feature.isin(eligible);root_details.append(a)
    roots=pd.DataFrame(root_rows);roots.to_parquet(private/'root_weight_comparison_PRIVATE.parquet',index=False)
    pd.concat(root_details,ignore_index=True).to_parquet(private/'root_candidates_PRIVATE.parquet',index=False)
    summary=[]
    for (scope,weight),g in roots.groupby(['scope','weight']):
        p=g[g.price_rank.notna()]
        summary.append(dict(scope=scope,weight=weight,roots=len(g),roots_with_price_candidate=len(p),
            price_wins=int(g.winner_is_price.sum()),price_rank_median=float(p.price_rank.median()),
            price_rank_q25=float(p.price_rank.quantile(.25)),price_rank_q75=float(p.price_rank.quantile(.75)),
            price_opposite_sign_count=int(p.price_opposite_sign.sum()),price_cancellation_median=float(p.price_cancellation.median()),
            winner_opposite_sign_count=int(g.winner_opposite_sign.sum()),winner_cancellation_median=float(g.winner_cancellation.median())))
    pd.DataFrame(summary).to_csv(reports/'bootstrap_root_summary.csv',index=False)
    roots.groupby(['scope','weight','winner_feature']).size().rename('roots').reset_index().to_csv(reports/'root_winner_frequencies.csv',index=False)
    for filename in ['core.py','partition.py']:assert sha(Path('crforest')/filename)==sha(base/'input/code/crforest'/filename)
    receipt=dict(status='COMPLETE_TRAINING_ONLY_ROOT_SCORE_DIAGNOSTIC',new_research_fits=0,new_predictions=0,
        train_companies=309,termination_events=14,recovery_events=184,censored_records=111,
        candidate_strategy='Original shared empirical quantiles and native missing routing; same constraints for every weight',
        full_cohort_candidates=len(full),rankable_full_cohort_features=int(ranks.rankable_features.iloc[0]),
        bootstrap_roots_verified=root_verified,original_score_comparisons=checks,independent_loop_candidates=independent,
        test_arrays_read=False,entry_2025_outcomes_read=False,holdout_2025_physically_sealed=False,
        old_models_changed=False,core_changed=False,weights_implemented_in_training=False,
        elapsed_seconds=time.perf_counter()-started,source_sha256=source_hashes,
        code_sha256={str(p):sha(p) for p in [Path(__file__),Path('crforest/core.py'),Path('crforest/partition.py')]},
        artifact_sha256={str(p.relative_to(out)):sha(p) for directory in [private,reports]
                         for p in sorted(directory.rglob('*')) if p.is_file()})
    write(reports/'calculation_receipt.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items() if not k.endswith('sha256')},ensure_ascii=False))
    print(price_rows[['weight','best_score','rank','rankable_features','U_termination','U_recovery','V_termination','V_recovery','cancellation_fraction']].to_string(index=False))
    print(pd.DataFrame(summary).to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.base,a.out)
