"""Verify C039 without using any other seed combination as a comparator."""
from dataclasses import asdict
from crforest import CompetingRiskForest,CIFExpansion
from crforest.shrinkage import shrink_leaf_cifs
from experiments.post_st_ablation_v1.verification import merge
from experiments.post_st_seed_explanation_stability_v1.common import run_path as triple_run
from experiments.post_st_dropcolumn_selection_v1.common import metrics
from experiments.post_st_prognosis_v1.common import horizon_predictions
from .common import *

def verify_anchor(r,source,record):
    exp=ROOT/'experiments/post_st_seed_explanation_stability_v1'
    selection=pd.read_csv(record(exp/'COMBINATIONS_120.csv')).set_index('combo').loc['C039']
    assert selection.tolist()==SEEDS
    allmetrics=pd.read_csv(record(exp/'ALL_METRICS.csv'),dtype={'model':str,'scope':str})
    known=allmetrics.query("scope=='pooled' and version=='shrunk' and cause==1 and months==12")
    triples=known[known.model.str.fullmatch(r'C\d{3}')]
    z=triples.loc[triples.model=='C039','within_year_auc'].item();assert z==triples.within_year_auc.max() and len(triples)==120
    prior=triple_run()
    check=pd.read_csv(record(exp/'RECONSTRUCTION_DETAIL.csv')).query("version=='shrunk' and model=='C039'").set_index('year')
    hashes=pd.read_csv(record(ROOT/'experiments/post_st_requested_seeds10_v1/MODEL_HASHES.csv'),dtype={'seed':str}).set_index(['year','seed','version'])
    manifests=[i for i in read(record(source/'input/reused_registry.json')) if i['seed'] in SEEDS]
    assert len(manifests)==9
    dirs={(i['year'],i['seed']):Path(i['path']) for i in manifests}
    exports={}
    for item in read(record(prior/'verify/published.json')):
        y,v,lo,hi=item['args']
        if v=='shrunk' and int(lo)<=38<int(hi):exports[int(y)]=Path(item['path'])/'C039_PRIVATE.npz'
    model_records=[];cache={};means=[];errors=[]
    for y in YEARS:
        fold=r/f'input/folds/{y}'
        tr,ev=(arrays(fold/f'{role}_PRIVATE.npz') for role in ['train','eval']);lab=arrays(r/f'eval/{y}.npz')
        X=np.vstack([tr['X'],ev['X']]);members=[];preds=[]
        for seed in SEEDS:
            path=dirs[y,seed];orig=Path(str(path).split('/results/')[0])
            for role in ['train','eval']:assert sha(record(orig/f'input/folds/{y}/{role}_PRIVATE.npz'))==sha(fold/f'{role}_PRIVATE.npz')
            assert sha(record(orig/'input/features.json'))==sha(source/'input/features.json')
            assert sha(record(orig/f'evaluation_input/{y}_test_PRIVATE.npz'))==sha(r/f'eval/{y}.npz')
            files=read(record(path/'hashes.json'))
            for f,h in files.items():record(path/f,h)
            raw=CompetingRiskForest.load(path/'raw/forest_PRIVATE.json')
            shr=CompetingRiskForest.load(path/'shrunk/forest_PRIVATE.json')
            assert all(asdict(m.config)==asdict(cfg(seed,74,'fixed25')) and len(m.trees_)==32 and m.n_features_==74 and m.audit()['max_path_variables']<=2 for m in [raw,shr])
            replay=shrink_leaf_cifs(raw,tr['X'],tr['time'],tr['event'],tr['groups'],strength=10)
            assert replay.trees_==shr.trees_
            for v in ['raw','shrunk']:
                row=hashes.loc[(y,str(seed),v)]
                for filename,col in [('forest_PRIVATE.json','forest_sha256'),('functions_PRIVATE.npz','function_sha256'),('predictions_PRIVATE.npz','prediction_sha256')]:
                    assert files[v+'/'+filename]==row[col]
                model_records.append(dict(year=y,seed=seed,arm='M3',version=v,**row.to_dict()))
            target=arrays(path/'shrunk/predictions_PRIVATE.npz')
            np.testing.assert_array_equal(target['sample_ids'],lab['sample_ids'])
            np.testing.assert_array_equal(target['times'],TIMES);np.testing.assert_array_equal(shr.predict(ev['X'],TIMES),target['cif'])
            cache[f'{y}|{seed}']=str(path/'shrunk')
            members.append(shr);preds.append(target['cif'])
        full=merge(members);path=record(exports[y],check.at[y,'function_sha256']);ex=CIFExpansion.load(path)
        times=np.unique(np.r_[0.,full.grid_,TIMES])
        err=0.
        for offset in range(0,len(X),64):
            xx=X[offset:offset+64]
            orig=full.predict(xx,times);err=max(err,float(abs(ex.predict(xx,times)-orig).max()),float(abs(np.mean([a.predict(xx,times) for a in members],axis=0)-orig).max()))
        assert err<=1e-10
        avg=np.mean(preds,axis=0);d=r/f'C039/{y}';d.mkdir(parents=True)
        np.savez_compressed(d/'predictions_PRIVATE.npz',sample_ids=lab['sample_ids'],times=TIMES,cif=avg)
        cache[f'{y}|0']=str(d);means.append(dict(year=y,source_function_sha256=sha(path),company_mean_sha256=sha(d/'predictions_PRIVATE.npz'),all_time_max_error=err))
        for model,pp in zip([*map(str,SEEDS),'C039'],[*preds,avg]):
            for h in [12,24]:
                for c in [1,2]:
                    risk=horizon_predictions(pp,TIMES,lab[f'h{h}'])[:,c-1];v=metrics(lab,risk,h,c)
                    row=allmetrics[(allmetrics.scope==str(y))&(allmetrics.model==model)&(allmetrics.version=='shrunk')&(allmetrics.months==h)&(allmetrics.cause==c)].iloc[0]
                    for name in ['risk_mean','auc','ap','brier','captured','coverage']:
                        np.testing.assert_allclose(v[name],row[name],atol=1e-13,rtol=0,equal_nan=True)
                        if np.isfinite(v[name]):errors.append(abs(v[name]-row[name]))
    dump(r/'anchor_paths_PRIVATE.json',cache);pd.DataFrame(model_records).to_csv(D/'M3_MEMBER_HASHES.csv',index=False)
    return dict(combo='C039',seeds=SEEDS,member_forests=9,ensemble_trees_per_year=96,new_M3_fits=0,preexisting_best_value=z,tied_triples_at_max=int((triples.within_year_auc==z).sum()),
        annual_metric_max_error=float(max(errors)),source_snapshots_match=True,raw_parent_replay_equal=True,member_saved_predictions_bitwise_equal=True,annual=means)
