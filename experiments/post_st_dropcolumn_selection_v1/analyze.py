"""Unified historical/outer evaluation, no new fits and no external selection."""
from pathlib import Path
import argparse,itertools,json
import numpy as np
import pandas as pd
from .common import *
from .selector import eliminate

def predictor(run,record):
    p=Path(record['path']);z=arrays(p/'ensemble/predictions_PRIVATE.npz')
    return z,load(p/'summary.json'),pd.read_csv(p/'seed_stability.csv')

def scalar(x):return float(x) if np.isfinite(x) else None

def history_analysis(run):
    registry=load(run/'audit/registry_aggregate_inner.json');assert load(run/'audit/gate_aggregate_inner.json')['status']=='PASS'
    sets=load(run/'input/subsets.json');arms=load(run/'input/arms.json');features=load(run/'input/features.json');support=pd.read_csv(run/'reports/historical_support.csv')
    full=next(a['subset_id'] for a in arms if a['label']=='full83');records={(r['year'],r['subset_id']):r for r in registry}
    predictions={};complexity=[];complexity_map={};stability=[];seedpred={}
    for r in registry:
        z,s,st=predictor(run,r);predictions[(r['year'],r['subset_id'])]=z
        d={k:s['check'][k] for k in ['main_functions','interaction_functions','spatial_table_cells','scalar_table_cells','distinct_thresholds','max_path_variables','function_bytes','compile_seconds','low_support_cell_fraction','zero_support_cell_fraction']}
        row=dict(scope='inner',year=r['year'],subset_id=r['subset_id'],features=len(r['keep']),mode=r['mode'],**d)
        complexity.append(row);complexity_map[(r['year'],r['subset_id'])]=row
        for x in st.to_dict('records'):stability.append(dict(scope='inner',year=r['year'],subset_id=r['subset_id'],**x))
    raw=load(run/'audit/registry_inner.json')
    for r in raw:seedpred[(r['year'],r['subset_id'],r['seed'])]=arrays(Path(r['path'])/'shrunk/predictions_PRIVATE.npz')
    metric_rows=[];drop_rows=[];pooled_rows=[]
    for outer in YEARS:
        subset_ids=[sid for sid,s in sets.items() if outer in s['outers']]
        # Include full baseline for every comparison, same yearly risk sets.
        for v in range(2019,outer):
            lab=arrays(run/f'input/folds/{v}/labels_{outer}_PRIVATE.npz');n=len(lab['event'])
            draws=np.random.default_rng(np.random.SeedSequence([ANALYSIS_SEED,outer,v,11])).integers(n,size=(500,n))
            for h in [12,24]:
                sp=support[(support.outer_cutoff==outer)&(support.cohort==v)&(support.horizon==h)].iloc[0]
                risks={sid:horizon_predictions(predictions[(v,sid)]['cif'],TIMES,lab[f'h{h}']) for sid in subset_ids}
                for cause in [1,2]:
                    values={}
                    for sid in subset_ids:
                        m=metrics(lab,risks[sid][:,cause-1],h,cause);values[sid]=m
                        metric_rows.append(dict(scope='inner',outer_cutoff=outer,year=v,subset_id=sid,features=len(sets[sid]['keep']),mode=sets[sid]['mode'],seed='ensemble',horizon=h,cause=cause,mature=bool(sp.mature),**m))
                        for seed in SEEDS:
                            z=seedpred[(v,sid,seed)];risk=horizon_predictions(z['cif'],TIMES,lab[f'h{h}'])[:,cause-1]
                            metric_rows.append(dict(scope='inner',outer_cutoff=outer,year=v,subset_id=sid,features=len(sets[sid]['keep']),mode=sets[sid]['mode'],seed=str(seed),horizon=h,cause=cause,mature=bool(sp.mature),**metrics(lab,risk,h,cause)))
                    for arm in [a for a in arms if a['kind']=='single']:
                        sid=arm['subset_id'];feature=features[arm['feature']];b,d=values[full],values[sid]
                        boot=bootstrap_differences(lab,risks[full][:,cause-1],risks[sid][:,cause-1],h,cause,draws)
                        intervals={}
                        for k in ['auc','ap','brier']:
                            vv=boot.get(k,np.array([]));vv=vv[np.isfinite(vv)]
                            intervals[k]=dict(lower95=scalar(np.quantile(vv,.025)) if len(vv) else None,upper95=scalar(np.quantile(vv,.975)) if len(vv) else None,undefined=500-len(vv))
                        seed_deltas=[]
                        for seed in SEEDS:
                            xx=[]
                            for key in [full,sid]:
                                pp=seedpred[(v,key,seed)];p=horizon_predictions(pp['cif'],TIMES,lab[f'h{h}'])[:,cause-1];xx.append(metrics(lab,p,h,cause)['ap'])
                            seed_deltas.append(xx[1]-xx[0])
                        cb,cd=complexity_map[(v,full)],complexity_map[(v,sid)]
                        row=dict(feature_id=feature['feature_id'],feature_name=feature['feature_name'],feature_label=feature['label'],feature_group=feature['feature_group'],
                            historical_train_cutoff=f'{v}-01-01',outcome_information_cutoff=f'{outer}-01-01',outer_cutoff=outer,validation_year=v,horizon=h,cause=cause,
                            number_of_training_companies=int(sp.train_companies),known_termination_events=int(sp.train_termination),known_recovery_events=int(sp.train_recovery),
                            validation_target_events=d['target_events'],validation_unknown=d['unknown'],mature=bool(sp.mature),supported=bool(sp.supported),
                            main_effect_change=cd['main_functions']-cb['main_functions'],interaction_count_change=cd['interaction_functions']-cb['interaction_functions'],spatial_cell_change=cd['spatial_table_cells']-cb['spatial_table_cells'],
                            seed_variability=float(np.std(np.asarray(seed_deltas)[np.isfinite(seed_deltas)],ddof=1)) if np.isfinite(seed_deltas).sum()>1 else np.nan,
                            seed_improvement_fraction=float(np.mean(np.asarray(seed_deltas)[np.isfinite(seed_deltas)]>1e-12)) if np.isfinite(seed_deltas).any() else np.nan,
                            uncertainty_estimate=json.dumps(intervals),selection_status='INSUFFICIENT_SELECTION_SUPPORT',unestimable_reason=d['reason'])
                        for k in ['auc','ap','brier','captured']:
                            name='capture20' if k=='captured' else k
                            row.update({f'baseline_{name}':b[k],f'dropped_{name}':d[k],f'delta_{name}':d[k]-b[k]})
                        row.update(ap_delta_lower95=intervals['ap']['lower95'],ap_delta_upper95=intervals['ap']['upper95'])
                        drop_rows.append(row)
        # Recompute pooled metrics on the concatenated predictions, not mean AP.
        for h in [12,24]:
            vv=support[(support.outer_cutoff==outer)&(support.horizon==h)&support.mature&support.supported].cohort.tolist()
            labs=[arrays(run/f'input/folds/{v}/labels_{outer}_PRIVATE.npz') for v in vv]
            if not labs:continue
            lab={k:np.concatenate([d[k] for d in labs]) for k in labs[0]}
            for sid in subset_ids:
                p=np.concatenate([horizon_predictions(predictions[(v,sid)]['cif'],TIMES,d[f'h{h}']) for v,d in zip(vv,labs)])
                for cause in [1,2]:
                    m=metrics(lab,p[:,cause-1],h,cause)
                    annual=[next(r for r in metric_rows if r['outer_cutoff']==outer and r['year']==v and r['subset_id']==sid and r['horizon']==h and r['cause']==cause and r['seed']=='ensemble') for v in vv]
                    m['captured']=sum(r['captured'] for r in annual);m['checked']=sum(r['checked'] for r in annual);m['coverage']=m['captured']/m['target_events'] if m['target_events'] else np.nan
                    pooled_rows.append(dict(scope='inner_mature_pooled',outer_cutoff=outer,year='pooled',subset_id=sid,features=len(sets[sid]['keep']),mode=sets[sid]['mode'],seed='ensemble',horizon=h,cause=cause,mature=True,
                        annual_mean_ap=float(np.nanmean([r['ap'] for r in annual])) if np.isfinite([r['ap'] for r in annual]).any() else np.nan,cohorts=','.join(map(str,vv)),**m))
    df=pd.DataFrame(metric_rows+pooled_rows);dr=pd.DataFrame(drop_rows)
    good=dr[dr.mature&dr.supported&dr.delta_ap.notna()]
    consistency=good.groupby(['outer_cutoff','feature_id','horizon','cause']).delta_ap.agg(lambda x:float((x>1e-12).mean())).rename('year_consistency').reset_index()
    dr=dr.merge(consistency,on=['outer_cutoff','feature_id','horizon','cause'],how='left')
    dr.to_csv(run/'reports/DROP_COLUMN_ALL_83.csv',index=False);df.to_csv(run/'reports/INTERNAL_MODEL_COMPARISON.csv',index=False)
    pd.DataFrame(complexity).to_csv(run/'reports/INTERNAL_EXPLANATION_COMPLEXITY.csv',index=False);pd.DataFrame(stability).to_csv(run/'reports/SEED_FUNCTION_STABILITY.csv',index=False)
    groups=[]
    for spec in load(run/'input/group_specs.json'):
        outer=spec['outer'];gid=spec['group_id']
        aliases=[dict(label=gid+'_A_all',subset_id=full,policy='A_all'),dict(label=gid+'_E_auto',subset_id=full,policy='E_auto_SUPPORT_FALLBACK')]
        aliases += [a for a in arms if a['kind']=='group' and a['group_id']==gid and a['outer']==outer]
        for a in aliases:
            rows=df[(df.outer_cutoff==outer)&(df.subset_id==a['subset_id'])]
            for r in rows.to_dict('records'):
                base=df[(df.outer_cutoff==outer)&(df.subset_id==full)&(df.year==r['year'])&(df.seed==r['seed'])&(df.horizon==r['horizon'])&(df.cause==r['cause'])].iloc[0]
                groups.append(dict(**r,group_id=gid,group_names=';'.join(features[j]['feature_name'] for j in spec['members']),policy=a['policy'],
                    delta_auc=r['auc']-base.auc,delta_ap=r['ap']-base.ap,delta_brier=r['brier']-base.brier,delta_capture20=r['captured']-base.captured,selection_status='INSUFFICIENT_SELECTION_SUPPORT'))
    pd.DataFrame(groups).to_csv(run/'reports/GROUP_DROP_COLUMN_RESULTS.csv',index=False)
    # Gate checked on frozen pre-cutoff data, before any outer scoring.
    paths=[];selected=[];decisions={}
    for outer in YEARS:
        gate=load(run/'input/support_gates.json')[str(outer)]
        assert not gate['passed'],'Supported case needs the declared recursive fitting callback; never use initial ranking as a substitute'
        keep,path=eliminate(range(83),[(j,) for j in range(83)],gate,lambda *a:(_ for _ in ()).throw(AssertionError('gate failed to stop')))
        decisions[str(outer)]=dict(features=list(keep),subset_id=full,status='FALLBACK83_INSUFFICIENT_SUPPORT',support=gate)
        paths.extend(dict(outer_cutoff=outer,**r) for r in path)
        for method in ['single_drop_selection','group_selection','recursive_selection','stability_selection']:
            for j in range(83):selected.append(dict(outer_cutoff=outer,method=method,feature_id=j,feature_name=features[j]['feature_name'],included=j in keep,selection_status='FALLBACK_NOT_ESTABLISHED_OPTIMUM'))
    pd.DataFrame(paths).to_csv(run/'reports/RECURSIVE_ELIMINATION_PATH.csv',index=False);pd.DataFrame(selected).to_csv(run/'reports/SELECTED_FEATURES.csv',index=False)
    write(run/'audit/selection_decisions.json',decisions)
    print(json.dumps(dict(drop_rows=len(dr),all_features=int(dr.feature_id.nunique()),group_rows=len(groups),selection='FALLBACK83',new_outer_outcomes_read=False)),flush=True)

def outer_analysis(run):
    registry=load(run/'audit/registry_aggregate_outer.json');sets=load(run/'input/subsets.json');arms=load(run/'input/arms.json')
    labels={y:arrays(run/f'evaluation_input/{y}_test_PRIVATE.npz') for y in YEARS};rows=[];cx=[];allpred={};bins=[]
    full=next(a['subset_id'] for a in arms if a['label']=='full83')
    for r in registry:
        z,s,st=predictor(run,r);year=r['year'];lab=labels[year];np.testing.assert_array_equal(z['sample_ids'],lab['sample_ids']);allpred[(year,r['subset_id'])]=z
        label=next(a['label'] for a in arms if a['subset_id']==r['subset_id']);check=s['check']
        cx.append(dict(scope='outer',year=year,label=label,subset_id=r['subset_id'],features=len(r['keep']),mode=r['mode'],**{k:check[k] for k in ['main_functions','interaction_functions','spatial_table_cells','scalar_table_cells','distinct_thresholds','function_bytes','compile_seconds','max_path_variables']}))
        for h in [12,24]:
            risks=horizon_predictions(z['cif'],TIMES,lab[f'h{h}'])
            for cause in [1,2]:
                m=metrics(lab,risks[:,cause-1],h,cause)
                rows.append(dict(scope='outer',year=year,label=label,subset_id=r['subset_id'],features=len(r['keep']),mode=r['mode'],horizon=h,cause=cause,seed='ensemble',**m))
                if m['supported']:
                    yy,w,known,_,_=evaluation_context(lab,h,cause)
                    for b,ix in enumerate(np.array_split(np.argsort(risks[:,cause-1],kind='stable'),4)):
                        bins.append(dict(year=year,label=label,subset_id=r['subset_id'],horizon=h,cause=cause,quartile=b+1,n=len(ix),predicted=float(risks[ix,cause-1].mean()),observed=float(np.sum(w[ix]*yy[ix])/len(ix)),unknown=int((~known[ix]).sum())))
        for seed,p in zip(SEEDS,r['paths']):
            zseed=arrays(Path(p)/'predictions_PRIVATE.npz')
            for h in [12,24]:
                risks=horizon_predictions(zseed['cif'],TIMES,lab[f'h{h}'])
                for cause in [1,2]:rows.append(dict(scope='outer',year=year,label=label,subset_id=r['subset_id'],features=len(r['keep']),mode=r['mode'],horizon=h,cause=cause,seed=str(seed),**metrics(lab,risks[:,cause-1],h,cause)))
    sidlist=sorted({r['subset_id'] for r in registry})
    for h in [12,24]:
        yy=[y for y in YEARS if evaluation_context(labels[y],h,1)[3]]
        lab={k:np.concatenate([labels[y][k] for y in yy]) for k in labels[yy[0]]}
        for sid in sidlist:
            risk=np.concatenate([horizon_predictions(allpred[(y,sid)]['cif'],TIMES,labels[y][f'h{h}']) for y in yy])
            label=next(a['label'] for a in arms if a['subset_id']==sid)
            for cause in [1,2]:
                m=metrics(lab,risk[:,cause-1],h,cause);annual=[r for r in rows if r['year'] in yy and r['subset_id']==sid and r['horizon']==h and r['cause']==cause and r['seed']=='ensemble']
                m['captured']=sum(r['captured'] for r in annual);m['checked']=sum(r['checked'] for r in annual);m['coverage']=m['captured']/m['target_events'] if m['target_events'] else np.nan
                rows.append(dict(scope='outer_supported_pooled',year='pooled',label=label,subset_id=sid,features=len(sets[sid]['keep']),mode=sets[sid]['mode'],horizon=h,cause=cause,seed='ensemble',cohorts=','.join(map(str,yy)),**m))
    df=pd.DataFrame(rows)
    # Explicit aliases disclose that the automatic procedure retained83; no new fit is invented.
    auto=df[df.subset_id==full].copy();auto['label']='automatic_fallback83';auto['selection_status']='INSUFFICIENT_HISTORICAL_SUPPORT'
    df=pd.concat([df,auto],ignore_index=True)
    df.to_csv(run/'reports/MODEL_COMPARISON.csv',index=False);pd.DataFrame(cx).to_csv(run/'reports/EXPLANATION_COMPLEXITY.csv',index=False);pd.DataFrame(bins).to_csv(run/'reports/CALIBRATION_BINS.csv',index=False)
    print(df[(df.seed=='ensemble')&(df.cause==1)&df.label.isin(['full83','legacy74','automatic_fallback83'])][['year','label','mode','horizon','auc','ap','brier','captured','target_events','supported']].to_string(index=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--stage',choices=['inner','outer'],required=True);a=p.parse_args()
    (history_analysis if a.stage=='inner' else outer_analysis)(a.run)
