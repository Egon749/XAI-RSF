"""All prespecified repetitions, including unfavorable estimates; no model selection."""
from pathlib import Path
import argparse,json,shutil
import numpy as np
import pandas as pd
from scipy.stats import t,norm
from .common import *

IDS=['scenario','rep','arm','variant','seed','day','cause']

def stats(values):
    x=np.asarray(values,float);x=x[np.isfinite(x)];n=len(x)
    if not n:return dict(n=0,mean=np.nan,median=np.nan,sd=np.nan,mcse=np.nan,low95=np.nan,high95=np.nan,p025=np.nan,p975=np.nan)
    mean=float(x.mean());sd=float(x.std(ddof=1)) if n>1 else np.nan;se=sd/np.sqrt(n);width=t.ppf(.975,n-1)*se if n>1 else np.nan
    return dict(n=n,mean=mean,median=float(np.median(x)),sd=sd,mcse=se,low95=mean-width,high95=mean+width,p025=float(np.quantile(x,.025)),p975=float(np.quantile(x,.975)))

def summarize(frame,keys,metrics,rate_intervals=True):
    rows=[]
    for key,g in frame.groupby(keys,dropna=False,sort=True):
        if not isinstance(key,tuple):key=(key,)
        base=dict(zip(keys,key))
        for m in metrics:
            row=dict(**base,metric=m,**stats(g[m]))
            if rate_intervals and m in ['detected','positive_detected','sign_correct','true_pair_detected','true_pair_positive_detected','true_pair_sign_correct','pair_recall']:
                x=np.asarray(g[m],float);x=x[np.isfinite(x)]
                if len(x) and np.isin(x,[0,1]).all():
                    n=len(x);p=x.mean();z=norm.ppf(.975);den=1+z*z/n
                    center=(p+z*z/(2*n))/den;width=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
                    row.update(rate_wilson_low95=0. if p==0 else max(0.,center-width),rate_wilson_high95=1. if p==1 else min(1.,center+width))
            rows.append(row)
    return pd.DataFrame(rows)

def role(row):
    key=tuple(map(int,str(row.features).split(',')));s=list(map(int,signals(int(row.rep))))
    if len(key)>1:return 'true_pair' if len(key)==2 else 'true_triple'
    return f'z{s.index(key[0])}'

def paired(frame,metrics):
    keys=[k for k in IDS if k!='arm'];parts=[]
    a=frame[frame.arm=='q2'].set_index(keys);b=frame[frame.arm=='unrestricted'].set_index(keys)
    assert a.index.equals(b.index) or set(a.index)==set(b.index)
    d=(a[metrics]-b[metrics]).reset_index()
    return summarize(d,[k for k in keys if k!='rep'],metrics,rate_intervals=False)

def analyze(run):
    gate=json.loads((run/'audit/formal_gate.json').read_text());assert gate['status']=='PASS'
    assert sha(run/'input/protocol.json')==sha(run/'audit/protocol.json')
    outputs=[];identities=[];tasks=[];frames={k:[] for k in ['signals','recovery','prediction','models','validation']};trees=[];stability=[];energy=[]
    for i,line in enumerate((run/'formal.jsonl').read_text().splitlines()):
        task=json.loads(line);argv=task['argv'];scenario=argv[argv.index('--scenario')+1];rep=int(argv[argv.index('--rep')+1])
        folder=run/'results/formal'/str(gate['job'])/f'task-{i}/output';outputs.append(folder)
        ident=json.loads((folder/'data_identity.json').read_text());summary=json.loads((folder/'summary.json').read_text())
        assert ident['scenario']==scenario and ident['rep']==rep and summary['new_forest_fits']==6
        identities.append(ident);tasks.append(dict(task=i,**summary,output=str(folder),hashes_sha256=sha(folder/'hashes.json')))
        for name in frames:frames[name].append(pd.read_csv(folder/f'{name}.csv',dtype={'seed':str,'features':str}))
        try:tr=pd.read_csv(folder/'tree_signal.csv.gz',dtype={'seed':str,'features':str})
        except pd.errors.EmptyDataError:tr=pd.DataFrame()  # additive control has NO true interaction
        if len(tr):
            assert not (tr.nonzero & ~tr.eligible).any()
            tr['role']=tr.apply(role,axis=1)
            for (arm,variant,day,order,feature_role),g in tr.groupby(['arm','variant','day','order','role']):
                theta=float(g.true_coefficient.iloc[0]);el=g[g.eligible];nz=g[g.nonzero]
                ef=float(g.eligible.mean());coef=float(g.coefficient.mean());ec=float(el.coefficient.mean()) if len(el) else np.nan
                if len(el):np.testing.assert_allclose(coef,ef*ec,atol=1e-13,rtol=0)
                trees.append(dict(scenario=scenario,rep=rep,arm=arm,variant=variant,day=day,order=order,role=feature_role,
                    tree_count=len(g),eligible_trees=len(el),nonzero_trees=len(nz),eligible_fraction=ef,nonzero_fraction=float(g.nonzero.mean()),
                    eligible_nonzero_rate=float(el.nonzero.mean()) if len(el) else np.nan,coefficient=coef,amplitude_ratio=coef/theta,
                    eligible_amplitude_ratio=ec/theta,nonzero_amplitude_ratio=float(nz.coefficient.mean())/theta if len(nz) else np.nan))
        # Set stability includes absent selections; three seeds are algorithmic variation.
        c=pd.read_csv(folder/'components.csv.gz',usecols=['arm','variant','seed','day','cause','features','order','estimated_rms','function_l2_error','true'],dtype={'seed':str,'features':str})
        c['squared_error']=c.function_l2_error**2
        for keys,g in c.groupby(['arm','variant','seed','day','cause','order','true']):
            energy.append(dict(scenario=scenario,rep=rep,**dict(zip(['arm','variant','seed','day','cause','order','true'],keys)),component_squared_error=float(g.squared_error.sum()),components=len(g)))
        c=c[(c.day==730)&(c.cause==1)&(c.seed!='ensemble')]
        for (arm,variant),g in c.groupby(['arm','variant']):
            for order in [1,2]:
                selected={s:set(g[(g.seed==str(s))&(g.order==order)&(g.estimated_rms>=.005)].features) for s in SEEDS}
                vals=[];empty=0
                for a,b in [(42,101),(42,202),(101,202)]:
                    union=selected[a]|selected[b]
                    if union:vals.append(len(selected[a]&selected[b])/len(union))
                    else:empty+=1
                stability.append(dict(scenario=scenario,rep=rep,arm=arm,variant=variant,order=order,
                    mean_jaccard_nonempty=float(np.mean(vals)) if vals else np.nan,empty_both_pairs=empty,defined_pairs=len(vals)))
    df={k:pd.concat(v,ignore_index=True) for k,v in frames.items()};ident=pd.DataFrame(identities);taskdf=pd.DataFrame(tasks)
    assert len(ident)==450 and not ident.duplicated(['scenario','rep']).any()
    assert set(ident.scenario)==set(SCENARIOS) and all(len(g)==REPETITIONS for _,g in ident.groupby('scenario'))
    assert taskdf.new_forest_fits.sum()==2700 and taskdf.derived_shrunk_models.sum()==2700
    sig=df['signals'];sig['role']=sig.apply(role,axis=1);sig['rms_ratio']=sig.estimated_rms/sig.true_rms
    sig['detected']=sig.estimated_rms>=.005;sig['positive_detected']=sig.detected&sig.sign_correct
    rec=df['recovery'];base=df['prediction'].copy();base['mean_risk_error']=base.mean_prediction-base.true_mean
    for focus in ['all','main','pair','higher']:
        r=rec[(rec.threshold==.005)&(rec.focus==focus)]
        cols=['tp','fp','fn','precision','recall','f1']
        if focus=='all':cols+=['exact_product_risk_mse','baseline','true_baseline','higher_order_energy']
        ren={k:f'{focus}_{k}' for k in cols if k in ['tp','fp','fn','precision','recall','f1']}
        base=base.merge(r[IDS+cols].rename(columns=ren),on=IDS,validate='one_to_one')
    pair=sig[sig.role=='true_pair'];cols=['amplitude_ratio','rms_ratio','coefficient','true_coefficient','function_l2_error','estimated_rms','sign_correct','detected','positive_detected']
    base=base.merge(pair[IDS+cols].rename(columns={k:'true_pair_'+k for k in cols}),on=IDS,how='left',validate='one_to_one')
    base=base.merge(df['models'][['scenario','rep','arm','variant','seed','spatial_cells_including_missing','order1_structural_functions','order2_structural_functions','order3_structural_functions','order4_structural_functions','order1_nonzero_target_24m','order2_nonzero_target_24m']],on=['scenario','rep','arm','variant','seed'],validate='many_to_one')
    # Python bool mixed with missing values is converted once for numerical summaries.
    for k in ['true_pair_sign_correct','true_pair_detected','true_pair_positive_detected']:base[k]=pd.to_numeric(base[k],errors='coerce')
    metrics=[k for k in base if k not in IDS and pd.api.types.is_numeric_dtype(base[k])]
    summary=summarize(base,[k for k in IDS if k!='rep'],metrics)
    differences=paired(base,metrics)
    signal_summary=summarize(sig,['scenario','arm','variant','seed','day','cause','role'],['amplitude_ratio','rms_ratio','coefficient','true_coefficient','function_l2_error','estimated_rms','shape_cosine','sign_correct','detected','positive_detected'])
    tree=pd.DataFrame(trees);tree_summary=summarize(tree,['scenario','arm','variant','day','order','role'],['eligible_trees','nonzero_trees','eligible_fraction','nonzero_fraction','eligible_nonzero_rate','amplitude_ratio','eligible_amplitude_ratio','nonzero_amplitude_ratio'])
    st=pd.DataFrame(stability);st_summary=summarize(st,['scenario','arm','variant','order'],['mean_jaccard_nonempty','empty_both_pairs'])
    # Paired leaf-estimator changes use exactly the SAME fitted forest partitions.
    ix=[k for k in IDS if k!='variant'];raw=base[base.variant=='raw'].set_index(ix);shr=base[base.variant=='shrunk'].set_index(ix)
    ds=(shr[metrics]-raw[metrics]).reset_index();shrink_differences=summarize(ds,[k for k in ix if k!='rep'],metrics,rate_intervals=False)
    seed_variation=[]
    for key,g in sig[(sig.seed!='ensemble')].groupby(['scenario','rep','arm','variant','day','cause','role']):
        assert len(g)==3
        seed_variation.append(dict(zip(['scenario','rep','arm','variant','day','cause','role'],key),coefficient_seed_sd=float(g.coefficient.std(ddof=1)),amplitude_ratio_seed_sd=float(g.amplitude_ratio.std(ddof=1))))
    sv=pd.DataFrame(seed_variation);sv_summary=summarize(sv,['scenario','arm','variant','day','cause','role'],['coefficient_seed_sd','amplitude_ratio_seed_sd'])
    en=pd.DataFrame(energy);en_summary=summarize(en,['scenario','arm','variant','seed','day','cause','order','true'],['component_squared_error','components'])
    out=run/'reports'
    for name,frame in [('metric_summary',summary),('paired_q2_minus_unrestricted',differences),('signal_summary',signal_summary),('tree_signal_summary',tree_summary),('structure_stability_summary',st_summary),('shrinkage_paired_changes',shrink_differences),('seed_variation_summary',sv_summary),('function_error_by_order',en_summary),('task_manifest',taskdf),('data_identity',ident)]:frame.to_csv(out/f'{name}.csv',index=False)
    for name,frame in [('metrics_per_repetition',base),('signals_per_repetition',sig),('models_per_repetition',df['models']),('tree_signal_per_repetition',tree),('structure_stability_per_repetition',st),('validation',df['validation']),('threshold_sensitivity',rec)]:frame.to_csv(out/f'{name}.csv.gz',index=False,compression='gzip')
    write(out/'execution_summary.json',dict(status='COMPLETE',tasks=len(taskdf),formal_fits=int(taskdf.new_forest_fits.sum()),derived_shrunk_models=int(taskdf.derived_shrunk_models.sum()),
        pilot_fits=4,independent_repetitions_per_scenario=50,formal_wall_seconds=gate['seconds'],by_host=gate['by_host'],
        max_rss_MiB=float(taskdf.peak_rss_MiB.max()),max_reconstruction_error=float(taskdf.max_validation_error.max()),
        failed_formal_tasks=gate['failed_tasks'],code_commit=json.loads((run/'audit/pre_fit_freeze.json').read_text())['code_commit'],
        source_commit='65776adaac9fa0ad65625d5ef5dc90415dd8adc3',private_run=str(run),reference_changed=False,
        parameters_selected_using_results=False,real_enterprise_fits=0))
    from .plots import plots
    plots(run,summary,signal_summary)
    print((out/'execution_summary.json').read_text())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();analyze(a.run)
