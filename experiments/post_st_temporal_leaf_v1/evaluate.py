from pathlib import Path
from dataclasses import asdict
import argparse,itertools,json
import numpy as np
import pandas as pd
from crforest import CompetingRiskForest,ForestConfig,CIFExpansion,aalen_johansen,cluster_bootstrap
from experiments.post_st_prognosis_v1.common import arrays,sha,write,horizon_predictions
from experiments.post_st_train2024_h12_v1.evaluate import metric
from experiments.post_st_train2024_h12_v1.validate import direct_metrics
from .stages import artifacts


def evaluate(root):
    selection=json.loads((root/'audit/selection_freeze.json').read_text());assert not selection['test_outcomes_used']
    protocol=json.loads((root/'input/protocol.json').read_text());old=Path(protocol['reference_run'])
    for name,h in json.loads((root/'audit/immutable_input_hashes.json').read_text()).items():assert sha(root/'input'/name)==h
    for name,h in json.loads((root/'audit/step0_freeze.json').read_text())['preserved_public_sha256'].items():assert sha(Path(name))==h
    out=root/'evaluation';out.mkdir();ensemble_dir=root/'ensemble_private';ensemble_dir.mkdir()
    data={};metadata={};sources={}
    for y in protocol['outer_years']:
        for name in [f'{y}_test_PRIVATE.npz',f'{y}_test_metadata_PRIVATE.parquet']:
            p=old/'evaluation_input'/name;sources[str(p)]=sha(p)
        data[y]=arrays(old/'evaluation_input'/f'{y}_test_PRIVATE.npz')
        metadata[y]=pd.read_parquet(old/'evaluation_input'/f'{y}_test_metadata_PRIVATE.parquet')
        np.testing.assert_array_equal(data[y]['sample_ids'],metadata[y].sample_id)
    data['pooled']={k:np.concatenate([data[y][k] for y in protocol['outer_years']]) for k in data[2023]}
    metadata['pooled']=pd.concat([metadata[y] for y in protocol['outer_years']],ignore_index=True)
    pred={};forests={};folders={};diagnostics=[];hashes=[];nodes=[];geometry=[]
    for year,_,seed,folder in artifacts(root,'formal'):
        tr=arrays(root/'input/formal'/str(year)/'train_PRIVATE.npz')
        for arm in protocol['arms']:
            p=folder/arm;key=(year,arm,seed);folders[key]=p
            for name,h in json.loads((p/'hashes.json').read_text()).items():assert sha(p/name)==h
            f=CompetingRiskForest.load(p/'forest_PRIVATE.json');forests[key]=f;pred[key]=arrays(p/'predictions_PRIVATE.npz')
            np.testing.assert_array_equal(pred[key]['sample_ids'],data[year]['sample_ids'])
            d=json.loads((p/'summary.json').read_text());diagnostics.append(d)
            hashes.append(dict(year=year,arm=arm,seed=seed,forest_sha256=sha(p/'forest_PRIVATE.json'),predictions_sha256=sha(p/'predictions_PRIVATE.npz'),functions_sha256=sha(p/'functions_PRIVATE.npz')))
            weights=np.where(tr['year']>=2021,{'baseline':1,'time_w2':2,'time_w4':4,'leaf20':1}[arm],1)
            errors=[];leaf_support=[];effective=[]
            def independent_curve(ids):
                expanded=np.repeat(ids,weights[ids]);return aalen_johansen(tr['time'][expanded],tr['event'][expanded],f.grid_,2)
            for ti,tree in enumerate(f.trees_):
                def visit(index,ids,parent=None):
                    node=tree[index];curve=independent_curve(ids)
                    if 'cif' in node:
                        n=len(np.unique(tr['groups'][ids]));a=1. if parent is None else n/(n+10)
                        expected=curve if parent is None else a*curve+(1-a)*parent
                        errors.append(float(abs(expected-np.asarray(node['cif'])).max()))
                        target=len(np.unique(tr['groups'][ids[tr['event'][ids]==1]]));leaf_support.append((n,target))
                        _,inv=np.unique(tr['groups'][ids],return_inverse=True);ww=np.bincount(inv,weights=weights[ids]);effective.append(float(ww.sum()**2/(ww@ww)))
                        return
                    z=tr['X'][ids,node['feature']];left=(z<=node['threshold'])|(np.isnan(z)&node['missing_left'])
                    visit(node['left'],ids[left],curve);visit(node['right'],ids[~left],curve)
                visit(0,cluster_bootstrap(tr['groups'],int(seed),ti))
            assert max(errors)<1e-12
            nodes.append(dict(year=year,arm=arm,seed=seed,leaves=len(errors),max_independent_leaf_error=max(errors),
                zero_target_leaves=sum(t==0 for n,t in leaf_support),at_most_one_target_leaves=sum(t<=1 for n,t in leaf_support),
                unique_company_support_min=min(n for n,t in leaf_support),unique_company_support_median=float(np.median([n for n,t in leaf_support])),
                effective_company_support_median=float(np.median(effective))))
        strip=lambda f:[[{k:v for k,v in n.items() if k not in ['cif','leaf_shrinkage']} for n in tree] for tree in f.trees_]
        for arm in ['time_w2','time_w4']:
            assert strip(forests[(year,arm,seed)])==strip(forests[(year,'baseline',seed)])
            geometry.append(dict(year=year,arm=arm,seed=seed,partitions_identical=True))
    assert len(pred)==24
    complexity=[];stability=[]
    for y,arm in itertools.product(protocol['outer_years'],protocol['arms']):
        members=[forests[(y,arm,str(s))] for s in protocol['seeds']];ps=[pred[(y,arm,str(s))] for s in protocol['seeds']]
        pred[(y,arm,'ensemble')]=dict(sample_ids=ps[0]['sample_ids'],times=ps[0]['times'],cif=np.mean([p['cif'] for p in ps],axis=0))
        tr=arrays(root/'input/formal'/str(y)/'train_PRIVATE.npz');X=np.vstack([tr['X'],data[y]['X']])
        f=CompetingRiskForest(ForestConfig(**{**asdict(members[0].config),'n_trees':96}))
        f.grid_=members[0].grid_;f.n_features_=members[0].n_features_;f.trees_=[t for m in members for t in m.trees_]
        f.candidate_evaluations_=sum(m.candidate_evaluations_ for m in members);f.audit_=f.audit()
        times=np.unique(np.r_[0,f.grid_,365,366,730,731]);prediction=f.predict(X,times)
        average=np.mean([m.predict(X,times) for m in members],axis=0);np.testing.assert_allclose(prediction,average,atol=1e-14,rtol=0)
        assert prediction.min()>=-1e-12 and prediction.sum(-1).max()<=1+1e-12 and np.diff(prediction,axis=1).min()>=-1e-12
        ex=CIFExpansion.from_forest(f,tr['X']);cc=ex.complexity(tr['X'],support_min=10)
        error=float(abs(ex.predict(X,times)-average).max());assert error<1e-10
        file=ensemble_dir/f'{y}_{arm}_PRIVATE.npz';ex.save(file)
        reload_error=float(abs(CIFExpansion.load(file).predict(X,times)-average).max());assert reload_error<1e-10
        write(ensemble_dir/f'{y}_{arm}_complexity_PRIVATE.json',cc)
        complexity.append(dict(year=y,arm=arm,trees=96,reconstruction_error=error,reload_error=reload_error,functions_sha256=sha(file),**{k:v for k,v in cc.items() if k!='functions'}))
        for a,b in itertools.combinations(range(3),2):
            ca=json.loads((folders[(y,arm,str(protocol['seeds'][a]))]/'complexity_PRIVATE.json').read_text());cb=json.loads((folders[(y,arm,str(protocol['seeds'][b]))]/'complexity_PRIVATE.json').read_text())
            pair=lambda c:{tuple(x['features']) for x in c['functions'] if x['order']==2 and x['nonzero']}
            sa,sb=pair(ca),pair(cb);stability.append(dict(year=y,arm=arm,seed_a=protocol['seeds'][a],seed_b=protocol['seeds'][b],pair_jaccard=len(sa&sb)/max(1,len(sa|sb))))
    risks={}
    for (y,arm,seed),p in pred.items():
        for h in [12,24]:risks[(str(y),arm,seed,h)]=horizon_predictions(p['cif'],p['times'],data[y][f'h{h}'])
    for arm,seed,h in {(a,s,h) for y,a,s,h in risks}:
        risks[('pooled',arm,seed,h)]=np.concatenate([risks[(str(y),arm,seed,h)] for y in protocol['outer_years']])
    metrics=[];warnings=[];bins=[]
    for (scope,arm,seed,h),risk in risks.items():
        year='pooled' if scope=='pooled' else int(scope);ev=data[year];meta=metadata[year]
        for k,cause in [(1,'termination'),(2,'recovery')]:
            common=dict(scope=scope,arm=arm,seed=seed,horizon_months=h,cause=cause,weighting='one_company_one_record')
            result,w,y,known=metric(ev['analysis_time'],ev['event'],np.full(len(risk),h),risk[:,k-1],k,ev['label_eligible'])
            assert known.all();np.testing.assert_array_equal(w,np.ones(len(w)))
            np.testing.assert_allclose([result[z] for z in ['auc','ap','brier']],direct_metrics(y,risk[:,k-1]),atol=1e-12,rtol=0,equal_nan=True)
            metrics.append({**common,**result,'observed_minus_predicted_pp':100*(result['ipcw_observed_incidence']-result['risk_mean'])})
            if seed=='ensemble':
                for b,ids in enumerate(np.array_split(np.lexsort((ev['sample_ids'],risk[:,k-1])),4)):
                    bins.append(dict(**common,bin=b+1,n=len(ids),predicted=float(risk[ids,k-1].mean()),observed=float(y[ids].mean()),target_events=int(y[ids].sum())))
            if k!=1:continue
            for fraction in [.20,.25]:
                selected=np.zeros(len(risk),bool)
                for v in np.unique(ev['year']):
                    ix=np.flatnonzero(ev['year']==v);order=np.lexsort((ev['sample_ids'][ix],-risk[ix,0]));selected[ix[order[:int(np.ceil(len(ix)*fraction))]]]=True
                warnings.append(dict(**common,top_fraction=fraction,checked=int(selected.sum()),captured=int((selected&y).sum()),target_events=int(y.sum()),
                    coverage=float((selected&y).sum()/y.sum()) if y.any() else np.nan,known_non_target=int((selected&~y).sum()),unknown=0,
                    lead3=int((selected&y&(ev['analysis_time']>=3)).sum()),lead6=int((selected&y&(ev['analysis_time']>=6)).sum()),repeat_alerts=0))
    df=pd.DataFrame(metrics);df.to_csv(out/'metrics.csv',index=False)
    original=pd.read_csv('experiments/post_st_leaf_shallow_v1/results/metrics.csv');checks=0
    for row in df[df.arm=='baseline'].itertuples():
        match=original[(original.scope.astype(str)==row.scope)&(original.arm=='shallow_parent')&(original.seed.astype(str)==row.seed)&(original.cause==row.cause)&(original.horizon_months==row.horizon_months)].iloc[0]
        np.testing.assert_allclose([getattr(row,k) for k in ['auc','ap','brier','risk_mean']],[match[k] for k in ['auc','ap','brier','risk_mean']],atol=1e-12,rtol=0,equal_nan=True);checks+=1
    for name,values in [('warning_metrics',warnings),('calibration_bins',bins),('training_diagnostics',diagnostics),('model_hashes',hashes),('node_audit',nodes),('matched_geometry',geometry),('ensemble_complexity',complexity),('seed_stability',stability)]:
        pd.DataFrame(values).to_csv(out/f'{name}.csv',index=False)
    write(out/'independent_audit.json',dict(status='PASS',models=24,ensembles=8,metric_rows=len(df),baseline_metric_rows_reproduced=checks,
        maximum_independent_leaf_error=max(n['max_independent_leaf_error'] for n in nodes),maximum_all_time_reconstruction_error=max(c['reconstruction_error'] for c in complexity),
        exact_weighted_integer_replication_audit=True,no_probability_clipping=True,no_calibration_fitted=True,new_2025_outcomes_read=False,
        test_source_sha256=sources,existing_reference_results_preserved=True))
    print(df[(df.seed=='ensemble')&(df.cause=='termination')&(df.horizon_months==24)][['scope','arm','auc','ap','brier','risk_mean','observed_minus_predicted_pp']].to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);evaluate(p.parse_args().run)
