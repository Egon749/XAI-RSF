"""Paired follow-up simulation analysis; no selection or additional fitting."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .common import *
from experiments.post_st_structure_recovery_v1.analyze import summarize,role,IDS
from experiments.post_st_structure_recovery_v1.common import SCENARIOS,generate,data_hash

def analyze_simulation(run,scenario_filter=None,destination=None):
    folders=outputs(run,'formal')[:450]
    if scenario_filter is not None:
        folders=[p for p in folders if json.loads((p/'data_identity.json').read_text())['scenario']==scenario_filter]
    expected=450 if scenario_filter is None else 50
    frames={n:[] for n in ['signals','recovery','prediction','models','validation']};trees=[];stability=[];identities=[];energy=[]
    for folder in folders:
        identity=json.loads((folder/'data_identity.json').read_text());scenario,rep=identity['scenario'],identity['rep']
        link=json.loads((run/'input/sim_reference'/f'{scenario}_{rep}.json').read_text());old=Path(link['old_path'])
        assert sha(old/'hashes.json')==link['hashes_sha256']
        tr,test=generate(scenario,rep)
        assert data_hash(tr)==identity['train_sha256']==link['identity']['train_sha256']
        assert data_hash({'X':test})==identity['test_sha256']==link['identity']['test_sha256']
        identities.append(identity)
        for path,source_arm,dest_arm in [(old,'q2','reference'),(folder,'wide83','wide83')]:
            hh=json.loads((path/'hashes.json').read_text())
            for name in frames:
                assert sha(path/f'{name}.csv')==hh[f'{name}.csv']
                f=pd.read_csv(path/f'{name}.csv',dtype={'seed':str,'features':str})
                f=f[f.arm==source_arm].copy();f['arm']=dest_arm;frames[name].append(f)
            assert sha(path/'components.csv.gz')==hh['components.csv.gz']
            c=pd.read_csv(path/'components.csv.gz',dtype={'seed':str,'features':str});c=c[c.arm==source_arm].copy();c['arm']=dest_arm
            c['squared_error']=c.function_l2_error**2
            for keys,g in c.groupby(['arm','variant','seed','day','cause','order','true']):
                energy.append(dict(scenario=scenario,rep=rep,**dict(zip(['arm','variant','seed','day','cause','order','true'],keys)),component_squared_error=float(g.squared_error.sum()),components=len(g)))
            d=c[(c.day==730)&(c.cause==1)&(c.seed!='ensemble')]
            for variant,g in d.groupby('variant'):
                for order in [1,2]:
                    ss={s:set(g[(g.seed==str(s))&(g.order==order)&(g.estimated_rms>=.005)].features) for s in SEEDS}
                    vv=[];empty=0
                    for a,b in [(42,101),(42,202),(101,202)]:
                        union=ss[a]|ss[b]
                        if union:vv.append(len(ss[a]&ss[b])/len(union))
                        else:empty+=1
                    stability.append(dict(scenario=scenario,rep=rep,arm=dest_arm,variant=variant,order=order,
                        jaccard=float(np.mean(vv)) if vv else np.nan,empty_pairs=empty,defined_pairs=len(vv)))
            assert sha(path/'tree_signal.csv.gz')==hh['tree_signal.csv.gz']
            try:tree=pd.read_csv(path/'tree_signal.csv.gz',dtype={'seed':str,'features':str})
            except pd.errors.EmptyDataError:continue
            tree=tree[tree.arm==source_arm].copy();tree['arm']=dest_arm
            tree['role']=tree.apply(role,axis=1)
            assert not (tree.nonzero&~tree.eligible).any()
            for key,g in tree.groupby(['arm','variant','day','order','role']):
                theta=float(g.true_coefficient.iloc[0]);nz=g[g.nonzero];el=g[g.eligible]
                trees.append(dict(scenario=scenario,rep=rep,**dict(zip(['arm','variant','day','order','role'],key)),
                    eligible_trees=len(el),nonzero_trees=len(nz),tree_count=len(g),eligible_fraction=float(g.eligible.mean()),
                    amplitude_ratio=float(g.coefficient.mean()/theta),eligible_nonzero_rate=float(el.nonzero.mean()) if len(el) else np.nan,
                    nonzero_amplitude_ratio=float(nz.coefficient.mean()/theta) if len(nz) else np.nan))
    df={k:pd.concat(v,ignore_index=True) for k,v in frames.items()}
    assert len(identities)==expected and len(set((x['scenario'],x['rep']) for x in identities))==expected
    sig=df['signals'];sig['role']=sig.apply(role,axis=1);sig['rms_ratio']=sig.estimated_rms/sig.true_rms
    sig['detected']=(sig.estimated_rms>=.005).astype(int);sig['positive_detected']=(sig.detected.astype(bool)&sig.sign_correct).astype(int)
    pred=df['prediction'].copy();pred['mean_risk_error']=pred.mean_prediction-pred.true_mean
    rec=df['recovery']
    for focus in ['all','main','pair','higher']:
        rr=rec[(rec.threshold==.005)&(rec.focus==focus)]
        cols=['tp','fp','fn','precision','recall','f1']
        if focus=='all':cols+=['exact_product_risk_mse','baseline','true_baseline','higher_order_energy']
        pred=pred.merge(rr[IDS+cols].rename(columns={k:f'{focus}_{k}' for k in ['tp','fp','fn','precision','recall','f1']}),on=IDS,validate='one_to_one')
    ps=sig[sig.role=='true_pair'];cols=['amplitude_ratio','rms_ratio','function_l2_error','estimated_rms','shape_cosine','detected']
    pred=pred.merge(ps[IDS+cols].rename(columns={k:f'pair_{k}' for k in cols}),on=IDS,how='left',validate='one_to_one')
    pred=pred.merge(df['models'][['scenario','rep','arm','variant','seed','spatial_cells_including_missing','order1_structural_functions','order2_structural_functions','function_bytes']],on=IDS[:5],validate='many_to_one')
    numeric=[k for k in pred if k not in IDS and pd.api.types.is_numeric_dtype(pred[k])]
    summary=summarize(pred,[k for k in IDS if k!='rep'],numeric)
    ix=[k for k in IDS if k!='arm'];a=pred[pred.arm=='wide83'].set_index(ix);b=pred[pred.arm=='reference'].set_index(ix)
    assert set(a.index)==set(b.index)
    delta=(a[numeric]-b[numeric]).reset_index()
    paired=summarize(delta,[k for k in ix if k!='rep'],numeric,rate_intervals=False)
    sigsummary=summarize(sig,['scenario','arm','variant','seed','day','cause','role'],['amplitude_ratio','rms_ratio','function_l2_error','estimated_rms','shape_cosine','detected','positive_detected'])
    tree=pd.DataFrame(trees,columns=['scenario','rep','arm','variant','day','order','role','eligible_trees','nonzero_trees','tree_count','eligible_fraction','amplitude_ratio','eligible_nonzero_rate','nonzero_amplitude_ratio']);st=pd.DataFrame(stability)
    out=(run if destination is None else destination)/'reports';out.mkdir(parents=True,exist_ok=True)
    for name,frame in [('simulation_metric_summary',summary),('simulation_paired_changes',paired),('simulation_signal_summary',sigsummary),
        ('simulation_tree_summary',summarize(tree,['scenario','arm','variant','day','order','role'],['eligible_trees','nonzero_trees','eligible_fraction','eligible_nonzero_rate','amplitude_ratio','nonzero_amplitude_ratio'])),
        ('simulation_stability',summarize(st,['scenario','arm','variant','order'],['jaccard','empty_pairs','defined_pairs'])),
        ('simulation_model_summary',summarize(df['models'],['scenario','arm','variant','seed'],[k for k in df['models'] if k not in IDS and pd.api.types.is_numeric_dtype(df['models'][k])]))]:frame.to_csv(out/f'{name}.csv',index=False)
    for name,frame in [('simulation_metrics',pred),('simulation_signals',sig),('simulation_models',df['models']),('simulation_validation',df['validation']),('simulation_tree_signal',tree),('simulation_stability_per_rep',st),('simulation_function_error',pd.DataFrame(energy)),('simulation_threshold_sensitivity',rec)]:frame.to_csv(out/f'{name}.csv.gz',index=False,compression='gzip')
    pd.DataFrame(identities).drop(columns=['signal_coordinates','configuration_differences']).to_csv(out/'simulation_data_identity.csv',index=False)
    primary=pred[(pred.variant=='shrunk')&(pred.seed=='ensemble')&(pred.day==730)&(pred.cause==1)]
    primary.groupby(['scenario','arm'])[['pair_amplitude_ratio','pair_function_l2_error','pair_detected','joint_risk_mse','expected_brier','all_fp','pair_recall','spatial_cells_including_missing','function_bytes']].mean().reset_index().to_csv(out/'simulation_main.csv',index=False)
    write(out/'simulation_validation.json',dict(status='PASS',datasets=expected,new_fits=3*expected,reused_reference_fits=3*expected,
        data_regenerated_and_paired=True,maximum_reconstruction_error=float(df['validation'][['reload','training_reference','population_reference','explicit_sum','function_reload','native_time']].max().max()),
        note='Same data as prior exposed simulation; paired follow-up mechanism study, not independent confirmation.'))
