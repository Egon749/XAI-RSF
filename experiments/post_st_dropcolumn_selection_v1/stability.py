"""Actual company-bootstrap model stability, distinguished from blocked selection."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from .common import *

def local_distance(a,b):
    np.testing.assert_array_equal(a['sample_ids'],b['sample_ids'])
    left={k:x for k,x in zip(a['keys'],np.moveaxis(a['terms'],1,0))};right={k:x for k,x in zip(b['keys'],np.moveaxis(b['terms'],1,0))}
    zero=np.zeros((len(a['sample_ids']),4,2));l1=np.zeros_like(zero)
    for key in set(left)|set(right):l1+=abs(left.get(key,zero)-right.get(key,zero))
    pa=a['mu']+a['terms'].sum(1);pb=b['mu']+b['terms'].sum(1)
    return dict(mean_total_absolute_component_change=float(l1.mean()),mean_absolute_baseline_change=float(abs(a['mu']-b['mu']).mean()),mean_absolute_CIF_change=float(abs(pa-pb).mean()),same_historical_rows=len(a['sample_ids']))

def analyze(run):
    boot=load(run/'audit/registry_aggregate_bootstrap.json');outer=load(run/'audit/registry_aggregate_outer.json');inner=load(run/'audit/registry_aggregate_inner.json')
    features=load(run/'input/features.json');sets=load(run/'input/subsets.json');arms=load(run/'input/arms.json');full=next(a['subset_id'] for a in arms if a['label']=='full83')
    refs={(r['year'],r['kind']):r for r in outer+inner if r['subset_id']==full}
    frequency=[];interactions=[];mainrows=[];local=[];gates=[]
    for kind,records in [('inner',inner),('outer',outer)]:
        for r in records:
            ref=arrays(Path(refs[(r['year'],kind)]['path'])/'local_explanation_PRIVATE.npz');p=arrays(Path(r['path'])/'local_explanation_PRIVATE.npz')
            local.append(dict(scope=kind,year=r['year'],subset_id=r['subset_id'],features=len(r['keep']),**local_distance(ref,p)))
    for year in YEARS:
        reps=sorted([r for r in boot if r['year']==year],key=lambda r:r['repeat']);assert [r['repeat'] for r in reps]==list(range(50))
        base=Path(refs[(year,'outer')]['path']);basecomp=arrays(base/'ensemble/components_PRIVATE.npz');basekeys=list(basecomp['keys'])
        reference=arrays(run/f'input/outer/{year}/train_PRIVATE.npz');parts=[];summaries=[]
        for r in reps:
            p=Path(r['path']);s=load(p/'summary.json');assert s['bootstrap_selection_gate']['passed'] is False
            gates.append(dict(year=year,repeat=r['repeat'],gate_passed=False,unique_mature12_targets=s['bootstrap_selection_gate']['horizons']['12']['mature_target_events'],selected_columns=83,interpretation='fallback, not selection success'))
            parts.append(arrays(p/'ensemble/components_PRIVATE.npz'));summaries.append(pd.read_csv(p/'component_summary.csv'))
            local.append(dict(scope='bootstrap',year=year,subset_id=full,repeat=r['repeat'],features=83,**local_distance(arrays(base/'local_explanation_PRIVATE.npz'),arrays(p/'local_explanation_PRIVATE.npz'))))
        for j,f in enumerate(features):
            frequency.append(dict(outer_cutoff=year,feature_id=j,feature_name=f['feature_name'],feature_label=f['label'],actual_refit_replicates=50,selection_frequency=np.nan,
                fallback_inclusion_frequency=1.,main_structural_presence_frequency=float(np.mean([str(j) in p['keys'] for p in parts])),status='SELECTION_NOT_ESTIMABLE_SUPPORT_GATE_CLOSED',
                explanation='50 actual bootstrap refits of83-column fallback; inclusion is a deterministic safeguard, not statistical feature-selection evidence'))
        keys=sorted(set().union(*(set(p['keys']) for p in parts)))
        for key in keys:
            order=len(key.split(','));indices=list(map(int,key.split(',')))
            for h in [12,24]:
                ix=np.searchsorted(TIMES,reference[f'h{h}']);bindex=basekeys.index(key) if key in basekeys else None
                for cause in [1,2]:
                    amplitudes=[];present=[];correlations=[];differences=[]
                    bv=basecomp['terms'][np.arange(len(ix)),bindex,ix,cause-1] if bindex is not None else np.zeros(len(ix))
                    for p in parts:
                        kk=list(p['keys']);i=kk.index(key) if key in kk else None
                        v=p['terms'][np.arange(len(ix)),i,ix,cause-1] if i is not None else np.zeros(len(ix))
                        amp=float(np.sqrt(np.mean(v*v)));amplitudes.append(amp);present.append(amp>1e-14)
                        correlations.append(float(np.corrcoef(v,bv)[0,1]) if v.std()>1e-14 and bv.std()>1e-14 else np.nan)
                        differences.append(float(np.sqrt(np.mean((v-bv)**2))))
                    finite=np.asarray(correlations);finite=finite[np.isfinite(finite)]
                    row=dict(outer_cutoff=year,key=key,feature_names=';'.join(features[j]['feature_name'] for j in indices),order=order,horizon=h,cause=cause,
                        bootstrap_models=50,nonzero_function_frequency=float(np.mean(present)),rms_mean=float(np.mean(amplitudes)),rms_sd=float(np.std(amplitudes,ddof=1)),rms_p025=float(np.quantile(amplitudes,.025)),rms_p975=float(np.quantile(amplitudes,.975)),
                        shape_correlation_to_original_mean=float(finite.mean()) if len(finite) else np.nan,shape_correlation_valid=len(finite),rms_difference_from_original_mean=float(np.mean(differences)),
                        original_rms=float(np.sqrt(np.mean(bv*bv))),reference='same original outer TRAINING empirical marginal-product distribution; original83 predictor anchor',
                        status='FALLBACK_MODEL_STABILITY_NOT_SELECTION_STABILITY')
                    (interactions if order==2 else mainrows).append(row)
    nonzero={(r['outer_cutoff'],int(r['key'])):r['nonzero_function_frequency'] for r in mainrows if r['horizon']==12 and r['cause']==1}
    for r in frequency:
        r['main_nonzero_frequency_12m_termination']=nonzero.get((r['outer_cutoff'],r['feature_id']),0.)
    for name,rows in [('FEATURE_SELECTION_STABILITY',frequency),('INTERACTION_STABILITY',interactions),('MAIN_FUNCTION_STABILITY',mainrows),('LOCAL_EXPLANATION_CHANGES',local),('BOOTSTRAP_SUPPORT_GATES',gates)]:pd.DataFrame(rows).to_csv(run/f'reports/{name}.csv',index=False)
    print(dict(bootstrap_ensembles=len(boot),actual_bootstrap_refits=len(boot)*3,selection_gate_passes=0,interaction_summary_rows=len(interactions)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();analyze(a.run)
