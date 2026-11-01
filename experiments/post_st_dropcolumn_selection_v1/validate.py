"""Final evidence checks including redundant effective configurations and provenance."""
from pathlib import Path
import argparse,json
from collections import defaultdict
import numpy as np
import pandas as pd
from experiments.post_st_train2024_h12_v1.validate import direct_metrics
from .common import *

def validate(run,repo):
    for p,h in load(run/'audit/immutable_input_hashes.json').items():assert sha(run/'input'/p)==h
    for p,h in load(run/'audit/source_hashes.json').items():assert sha(Path(p))==h
    assert sha(repo/'CURRENT_POST_ST_MODEL.json')==REFERENCE_SHA
    fits=[];checks=[];hashes=[]
    for stage in ['pilot','inner','bootstrap','outer']:
        gate=load(run/f'audit/gate_{stage}.json');assert gate['status']=='PASS'
        for r in load(run/f'audit/registry_{stage}.json'):
            p=Path(r['path']);s=load(p/'summary.json')
            fits.append({k:v for k,v in s.items() if k not in ['checks']})
            for version in ['raw','shrunk']:
                c=s['checks'][version];assert c['max_path_variables']<=2 and c['path_violations']==0
                for k in ['full_reconstruction','explicit_component_sum','function_reload','forest_reload']:assert c[k]<=1e-10
                assert c['probability_min']>=-1e-12 and c['probability_sum_max']<=1+1e-10 and c['minimum_increment']>=-1e-12
                checks.append(dict(stage=stage,year=s['year'],seed=s['seed'],repeat=s.get('repeat',-1),subset_id=s['subset_id'],version=version,**c))
                hashes.append(dict(stage=stage,year=s['year'],seed=s['seed'],repeat=s.get('repeat',-1),subset_id=s['subset_id'],version=version,
                    forest_sha256=sha(p/version/'forest_PRIVATE.json'),functions_sha256=sha(p/version/'functions_PRIVATE.npz'),predictions_sha256=sha(p/version/'predictions_PRIVATE.npz')))
    for kind in ['inner','outer','bootstrap']:
        assert load(run/f'audit/gate_aggregate_{kind}.json')['status']=='PASS'
        for r in load(run/f'audit/registry_aggregate_{kind}.json'):
            s=load(Path(r['path'])/'summary.json');c=s['check']
            assert s['ensemble_mean_error']<=1e-13
            checks.append(dict(stage='aggregate_'+kind,year=r['year'],seed='ensemble',repeat=r.get('repeat',-1),subset_id=r['subset_id'],version='shrunk',**c))
    f=pd.DataFrame(fits);assert len(f)==3090 and (f.new_forest_fits==1).all()
    dr=pd.read_csv(run/'reports/DROP_COLUMN_ALL_83.csv');assert len(dr)==4980 and set(dr.feature_id)==set(range(83))
    assert dr[~dr.supported][['delta_auc','delta_ap','delta_brier','delta_capture20']].isna().all().all()
    stable=pd.read_csv(run/'reports/FEATURE_SELECTION_STABILITY.csv');assert len(stable)==249 and stable.selection_frequency.isna().all()
    assert (stable.actual_refit_replicates==50).all()
    assert not pd.read_csv(run/'reports/BOOTSTRAP_SUPPORT_GATES.csv').gate_passed.any()
    # Verify planned count/ratio modes which round to the SAME actual budget.
    sets=load(run/'input/subsets.json');equiv=defaultdict(list)
    for sid,s in sets.items():equiv[(tuple(s['keep']),configuration(42,len(s['keep']),s['mode']).tree_feature_count)].append(sid)
    rr={(r['year'],r['subset_id'],r['seed']):r for r in load(run/'audit/registry_inner.json')};duplicate_rows=[]
    for key,sids in equiv.items():
        if len(sids)<2:continue
        assert len(sids)==2
        for year in FOLDS:
            for seed in SEEDS:
                a,b=[Path(rr[(year,sid,seed)]['path']) for sid in sids]
                pa=arrays(a/'shrunk/predictions_PRIVATE.npz')['cif'];pb=arrays(b/'shrunk/predictions_PRIVATE.npz')['cif'];error=float(abs(pa-pb).max());assert error<=1e-13
                duplicate_rows.append(dict(year=year,seed=seed,subset_a=sids[0],subset_b=sids[1],features=len(key[0]),effective_tree_features=key[1],prediction_max_difference=error,
                    status='REDUNDANT_EFFECTIVE_CONFIGURATION_RETAINED_AND_COUNTED'))
    pd.DataFrame(duplicate_rows).to_csv(run/'reports/CONFIGURATION_EQUIVALENCE_AUDIT.csv',index=False)
    write(run/'audit/scheduling_deviation.json',dict(status='DISCLOSED_NO_MATHEMATICAL_CHANGE',declared_mode_subsets=len(sets),distinct_effective_configurations=len(equiv),
        equivalent_mode_pairs=9,extra_inner_fits=len(duplicate_rows),reason='Manifest deduplicated by subset and mode, not rounded effective tree-feature count. Nine count/ratio pairs therefore repeated identical fits.',
        treatment='Keep all162 extra fits and hashes. Verify equivalent predictions. Never pool duplicates into larger ensembles or independent evidence. No selective deletion or refit.',
        frozen_protocol_text_exception='The intended deduplication of equal effective configurations was incomplete; all actual costs counted.'))
    # Independent day-count endpoint and metric reconstruction on supported outer predictions.
    outer=pd.read_csv(run/'reports/MODEL_COMPARISON.csv',dtype={'year':str,'seed':str});verified=0
    for r in load(run/'audit/registry_aggregate_outer.json'):
        year=r['year'];lab=arrays(run/f'evaluation_input/{year}_test_PRIVATE.npz');z=arrays(Path(r['path'])/'ensemble/predictions_PRIVATE.npz')
        for h in [12,24]:
            ytime=lab['time']<=lab[f'h{h}'];np.testing.assert_array_equal(ytime,lab['analysis_time']<=h)
            if not evaluation_context(lab,h,1)[3]:continue
            for cause in [1,2]:
                pred=horizon_predictions(z['cif'],TIMES,lab[f'h{h}'])[:,cause-1];y=(lab['event']==cause)&ytime
                yy,w,known,_,_=evaluation_context(lab,h,cause);np.testing.assert_array_equal(y,yy)
                row=outer[(outer.year==str(year))&(outer.subset_id==r['subset_id'])&(outer.horizon==h)&(outer.cause==cause)&(outer.seed=='ensemble')&(outer.label!='automatic_fallback83')].iloc[0]
                if known.all():np.testing.assert_allclose(direct_metrics(y,pred),row[['auc','ap','brier']].to_numpy(float),atol=1e-12,rtol=0,equal_nan=True)
                else:
                    from sklearn.metrics import roc_auc_score,average_precision_score
                    active=w>0
                    want=[roc_auc_score(y[active],pred[active],sample_weight=w[active]),average_precision_score(y[active],pred[active],sample_weight=w[active]),np.sum(w*(pred-y)**2)/len(y)]
                    np.testing.assert_allclose(want,row[['auc','ap','brier']].to_numpy(float),atol=1e-12,rtol=0,equal_nan=True)
                verified+=1
    for label,name in [('full83','reference83'),('legacy74','drop9')]:
        history_df=pd.concat([pd.read_csv(repo/f'experiments/{e}/results/metrics.csv',dtype={'seed':str}) for e in ['post_st_drop9_v1','post_st_drop9_transport_v1']],ignore_index=True)
        for row in outer[(outer.label==label)&(outer['mode']=='count')&(outer.seed=='ensemble')&outer.year.isin(['2023','2024','2025'])&outer.supported].itertuples():
            old=history_df[(history_df.year==int(row.year))&(history_df.arm==name)&(history_df.seed=='ensemble')&(history_df.horizon_months==row.horizon)&(history_df.cause==row.cause)].iloc[0]
            np.testing.assert_allclose([row.auc,row.ap,row.brier],old[['auc','ap','brier']].to_numpy(float),atol=1e-12,rtol=0,equal_nan=True)
    f.to_csv(run/'reports/EXECUTION_TASKS.csv',index=False);pd.DataFrame(hashes).to_csv(run/'reports/MODEL_HASHES.csv',index=False);pd.DataFrame(checks).to_csv(run/'reports/NUMERICAL_VALIDATION.csv',index=False)
    error=max(c[k] for c in checks for k in ['full_reconstruction','explicit_component_sum','function_reload','forest_reload'])
    value=dict(status='PASS_WITH_SUPPORT_AND_SCHEDULING_LIMITATIONS',actual_new_forests=len(f),pilot_forests=3,inner_forests=2538,bootstrap_forests=450,outer_control_forests=99,
        extra_equivalent_inner_fits=len(duplicate_rows),unique_effective_inner_configs=len(equiv),reused_outer_reference_forests=18,
        ensemble_compilations=1035,all83_features_checked=True,drop_column_rows=len(dr),max_exact_reconstruction_error=error,
        independent_outer_metric_rows=verified,original_sources_unchanged=True,current_reference_unchanged=True,
        automatic_elimination_approved=False,selection_stability_estimated=False,fallback_bootstrap_replicates_per_year=50,failed_fits=0)
    write(run/'audit/final_validation.json',value);print(json.dumps(value),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();validate(a.run,Path(__file__).resolve().parents[2])
