"""Gate all 84 publications and analyse 50 paired refits, preserving pair dependence."""
from pathlib import Path
import argparse,json,hashlib
from itertools import combinations
import numpy as np
import pandas as pd
from experiments.interpretable_baselines_v1.evaluate import compare
from experiments.bootstrap_stability_50_v1.intervals import paired_u_interval,anchored_interval

METRICS=['normalized_rmse','rmse','importance_spearman','top10_jaccard','sign_agreement','median_anchor_correlation']
MODELS=['q2','unrestricted']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,data):
    with p.open('x') as f:json.dump(data,f,indent=2,allow_nan=False)

def main(run,job):
    out=run/'reports';assert not (out/'execution_receipt.json').exists()
    freeze=json.loads((run/'audit/freeze.json').read_text())
    for name,h in freeze['input_hashes'].items():assert sha(run/'input'/name)==h
    report=json.loads((run/f'formal.jsonl.results-{job}.json').read_text());items=report['completed']
    assert len(items)==84 and {x['id'] for x in items}==set(range(84))
    assert all(x['returncode']==0 and x['compute_completed'] and x['publication_completed'] for x in items)
    tasks=[json.loads(x) for x in (run/'formal.jsonl').read_text().splitlines()]
    meta=pd.read_parquet(run/'input/final_train_metadata_PRIVATE.parquet')
    assert len(meta)==56458 and meta.group.nunique()==2577 and meta.loc[meta.event==1,'event_id'].nunique()==177
    paths={};diag=[];hashes=[];phi={};original={};base={}
    reused=json.loads((run/'audit/reused_outputs.json').read_text())
    reused_hashes={(x['model'],x['replicate']):x['hashes'] for x in reused}
    plan=json.loads((run/'input/plan.json').read_text())
    locations=[(Path(x['output']),True,None) for x in reused]
    for x in items:
        p=run/f"results/formal/{job}/task-{x['id']}/output";s=json.loads((p/'summary.json').read_text());argv=tasks[x['id']]['argv']
        assert s['model']==argv[argv.index('--model')+1] and s['replicate']==int(argv[argv.index('--replicate')+1])
        locations.append((p,False,x['id']))
    for p,reuse,task_id in locations:
        s=json.loads((p/'summary.json').read_text());key=(s['model'],s['replicate']);assert key not in paths
        assert s['kind']=='bootstrap' and s['status']=='COMPLETE' and s['seed']==42 and not s['pilot']
        assert s['reload_error']<1e-12
        if reuse:assert json.loads((p/'hashes.json').read_text())==reused_hashes[key]
        fitted_config=json.loads((p/'forest_PRIVATE.json').read_text())['config']
        assert all(fitted_config[k]==v for k,v in plan['forest_configs'][s['model']].items())
        for name,h in json.loads((p/'hashes.json').read_text()).items():assert sha(p/name)==h
        assert s['resample_sha256']==sha(run/f"input/bootstrap_{s['replicate']}_PRIVATE.npz")
        z=dict(np.load(p/'explanations_PRIVATE.npz',allow_pickle=False));assert z['exact'].shape==(32,75)
        assert all(np.isfinite(v).all() for v in z.values())
        err=float(abs(z['base']+z['exact'].sum(1)-z['prediction']).max());assert err<1e-10
        assert ((z['prediction']>=0)&(z['prediction']<=1)).all()
        if s['model']=='q2':assert s['reconstruction_error']<1e-10
        bg=dict(np.load(p/'background_PRIVATE.npz',allow_pickle=False))
        if s['model'] not in original:original[s['model']]=bg['fixed']
        else:np.testing.assert_allclose(original[s['model']],bg['fixed'],atol=1e-14,rtol=0)
        paths[key]=p;phi[key]=z['exact'];base[key]=float(z['base'])
        diag.append(dict(**s,reused=reuse,task_id=task_id,shap_additivity_error=err,mean_absolute_attribution=float(abs(z['exact']).mean()),anchor_mean_risk=float(z['prediction'].mean())))
        hashes.append(dict(model=key[0],replicate=key[1],reused=reuse,model_sha256=sha(p/'forest_PRIVATE.json'),explanation_sha256=sha(p/'explanations_PRIVATE.npz'),resample_sha256=s['resample_sha256']))
    assert set(paths)=={(m,r) for m in MODELS for r in range(50)}
    pairs=[];anchored=[];background=[];native=[]
    for ra,rb in combinations(range(50),2):
        cmp={m:compare(phi[m,ra],phi[m,rb]) for m in MODELS}
        row=dict(replicate_a=ra,replicate_b=rb)
        for metric in METRICS:
            assert all(cmp[m][metric] is not None and np.isfinite(cmp[m][metric]) for m in MODELS)
            row.update({f'{m}_{metric}':cmp[m][metric] for m in MODELS});row[f'difference_{metric}']=cmp['q2'][metric]-cmp['unrestricted'][metric]
        pairs.append(row)
        na={}
        for r in [ra,rb]:
            z=np.load(paths['q2',r]/'native_PRIVATE.npz');na[r]={str(k):z['terms'][:,i] for i,k in enumerate(z['keys'])}
        a,b=set(na[ra]),set(na[rb]);pa={k for k in a if ',' in k};pb={k for k in b if ',' in k}
        cor=[float(np.corrcoef(na[ra][k],na[rb][k])[0,1]) for k in a&b if na[ra][k].std()>1e-12 and na[rb][k].std()>1e-12]
        native.append(dict(replicate_a=ra,replicate_b=rb,function_jaccard=len(a&b)/len(a|b),pair_jaccard=len(pa&pb)/len(pa|pb) if pa|pb else None,common_nonconstant=len(cor),median_shape_correlation=float(np.median(cor)) if cor else None))
    for r in range(50):
        row=dict(replicate=r);bgrow=dict(replicate=r)
        for m in MODELS:
            for metric,value in compare(phi[m,r],original[m]).items():row[f'{m}_{metric}']=value
            z=np.load(paths[m,r]/'background_PRIVATE.npz')
            for metric,value in compare(z['fixed'],z['resampled']).items():bgrow[f'{m}_{metric}']=value
        for metric in METRICS:
            row[f'difference_{metric}']=row[f'q2_{metric}']-row[f'unrestricted_{metric}']
            bgrow[f'difference_{metric}']=bgrow[f'q2_{metric}']-bgrow[f'unrestricted_{metric}']
        anchored.append(row);background.append(bgrow)
    pairs=pd.DataFrame(pairs);anchored=pd.DataFrame(anchored);background=pd.DataFrame(background);summaries=[];anchored_summaries=[];background_summaries=[]
    for n in [8,30,50]:
        sub=pairs[pairs.replicate_b<n]
        for metric in METRICS:
            kernel=np.zeros((n,n));ia=sub.replicate_a.to_numpy();ib=sub.replicate_b.to_numpy();kernel[ia,ib]=sub[f'difference_{metric}'];kernel[ib,ia]=sub[f'difference_{metric}']
            ci=paired_u_interval(kernel);assert ci['jackknife_se']>0,'Degenerate jackknife: report explicitly before proceeding'
            summaries.append(dict(metric=metric,primary=metric=='normalized_rmse',favorable_direction='negative' if metric in ['rmse','normalized_rmse'] else 'positive',
                q2_mean=float(sub[f'q2_{metric}'].mean()),blackbox_mean=float(sub[f'unrestricted_{metric}'].mean()),q2_median=float(sub[f'q2_{metric}'].median()),blackbox_median=float(sub[f'unrestricted_{metric}'].median()),paired_difference_median=float(sub[f'difference_{metric}'].median()),**ci))
            for data,target in [(anchored,anchored_summaries),(background,background_summaries)]:
                ss=data[data.replicate<n];res=anchored_interval(ss[f'difference_{metric}']);target.append(dict(metric=metric,q2_mean=float(ss[f'q2_{metric}'].mean()),blackbox_mean=float(ss[f'unrestricted_{metric}'].mean()),**res))
    for name,data in [('paired_stability',pairs),('original_anchor_stability',anchored),('background_stability',background),('stability_intervals',summaries),('original_anchor_intervals',anchored_summaries),('background_intervals',background_summaries),('native_q2_stability',native),('task_diagnostics',diag),('model_hashes',hashes)]:pd.DataFrame(data).to_csv(out/f'{name}.csv',index=False)
    # Reproduce old 8-replicate paired metrics exactly before accepting extension.
    old=pd.read_csv(Path(json.loads((run/'input/plan.json').read_text())['extension_parent'])/'reports/paired_stability_contrasts.csv').sort_values(['replicate_a','replicate_b'])
    first=pairs[pairs.replicate_b<8].sort_values(['replicate_a','replicate_b'])
    for metric in METRICS[:5]:np.testing.assert_allclose(first[f'difference_{metric}'],old[metric],atol=1e-14,rtol=0)
    gate=dict(status='PASS',job=job,new_tasks=84,reused_tasks=16,paired_replicates=50,dependent_pairs=1225,seconds=report['seconds'],by_host={h:sum(x['host']==h for x in items) for h in sorted({x['host'] for x in items})},failed_tasks=[],report_sha256=sha(run/f'formal.jsonl.results-{job}.json'))
    write(out/'formal_gate.json',gate)
    receipt=dict(**gate,run=str(run),code_commit=freeze['code_commit'],code_archive_sha256=freeze['code_archive_sha256'],manifest_sha256=freeze['manifest_sha256'],historical_parent_code_commit=plan['code_commit'],input_plan_code_commit_role='historical parent; execution code is freeze.json code_commit',reused_index_sha256=sha(run/'audit/reused_outputs.json'),reuse_protocol_reproduced=True,training_rows=56458,training_company_groups=2577,training_distinct_target_events=177,anchors=32,background_rows=16,seed=42,horizon_days=365,status_label='EXPLORATORY_PROVISIONAL',overall_research_decision='INCONCLUSIVE',estimand='conditional bootstrap expected stability difference; CI is Monte Carlo precision, not population uncertainty',max_reload_error=max(x['reload_error'] for x in diag),max_shap_additivity_error=max(x['shap_additivity_error'] for x in diag),max_q2_reconstruction_error=max(x['reconstruction_error'] for x in diag if x['model']=='q2'),analysis_source_hashes={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},results_hashes={p.name:sha(p) for p in out.iterdir() if p.is_file()})
    write(out/'execution_receipt.json',receipt);print(json.dumps(gate));print(pd.DataFrame(summaries).query('n_replicates==50')[['metric','q2_mean','blackbox_mean','estimate','ci_low','ci_high']].to_string(index=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);p.add_argument('--job',required=True,type=int);a=p.parse_args();main(a.run,a.job)
