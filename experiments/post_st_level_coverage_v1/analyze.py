"""Evaluate frozen predictions, separate calibration from wider eligibility."""
from pathlib import Path
import argparse,json,itertools,time
import numpy as np
import pandas as pd
from crforest import CompetingRiskForest
from experiments.post_st_ablation_v1.verification import merge
from experiments.post_st_ablation_v1.analyze import selected
from experiments.post_st_train2024_h12_v1.evaluate import metric
from experiments.post_st_train2024_h12_v1.validate import direct_metrics
from .common import *
from .worker import verify_export
from .affine import transform

def enterprise(run,destination=None):
    dest=run if destination is None else destination
    (dest/'reports').mkdir(parents=True,exist_ok=True);(dest/'private').mkdir(parents=True,exist_ok=True)
    formal=outputs(run,'formal');pred={};paths={};summaries=[]
    for y in YEARS:
        for s in SEEDS:
            folder=run/'input/reference'/str(y)/str(s);pred[(str(y),'reference',str(s))]=arrays(folder/'predictions_PRIVATE.npz');paths[(y,'reference',str(s))]=folder
    for folder in formal[450:456]:
        z=json.loads((folder/'summary.json').read_text());y,s=z['year'],str(z['seed']);summaries.append(z)
        for variant,arm in [('raw','wide83_raw'),('shrunk','wide83')]:
            pred[(str(y),arm,s)]=arrays(folder/variant/'predictions_PRIVATE.npz');paths[(y,arm,s)]=folder/variant
    for y,arm in itertools.product(YEARS,['reference','wide83','wide83_raw']):
        pp=[pred[(str(y),arm,str(s))] for s in SEEDS]
        for p in pp:np.testing.assert_array_equal(p['times'],TIMES)
        pred[(str(y),arm,'ensemble')]=dict(times=TIMES,sample_ids=pp[0]['sample_ids'],cif=np.mean([p['cif'] for p in pp],axis=0))
    data={str(y):arrays(run/'evaluation_input'/f'{y}_test_PRIVATE.npz') for y in YEARS}
    data['pooled']={k:np.concatenate([data[str(y)][k] for y in YEARS]) for k in data['2023']}
    risks={}
    for (y,arm,s),p in pred.items():
        np.testing.assert_array_equal(p['sample_ids'],data[y]['sample_ids'])
        for h in [12,24]:risks[(y,arm,s,h)]=horizon_predictions(p['cif'],p['times'],data[y][f'h{h}'])
    calibrators={}
    for folder in formal[456:]:
        cal=json.loads((folder/'calibrator_PRIVATE.json').read_text());y,h=cal['year'],cal['horizon_months'];calibrators[(y,h)]=cal
        summary=json.loads((folder/'summary.json').read_text());summaries.append(summary)
        z=arrays(folder/'calibrated/predictions_PRIVATE.npz')
        np.testing.assert_array_equal(z['sample_ids'],data[str(y)]['sample_ids'])
        expected=transform(pred[(str(y),'reference','ensemble')]['cif'],TIMES,cal)
        # Mean-of-seed predictions and concatenated96-tree predictions can
        # differ in the final floating-point bit because of summation order.
        np.testing.assert_allclose(expected,z['cif'],atol=1e-13,rtol=0)
        risks[(str(y),'affine','ensemble',h)]=horizon_predictions(z['cif'],TIMES,data[str(y)][f'h{h}'])
    for arm,s,h in {(a,s,h) for y,a,s,h in risks}:
        risks[('pooled',arm,s,h)]=np.concatenate([risks[(str(y),arm,s,h)] for y in YEARS])
    rows=[];warnings=[];bins=[]
    for (scope,arm,s,h),risk in risks.items():
        ev=data[scope]
        for c,cause in enumerate(['termination','recovery'],1):
            tags=dict(scope=scope,arm=arm,seed=s,horizon_months=h,cause=cause)
            result,w,y,known=metric(ev['analysis_time'],ev['event'],np.full(len(risk),h),risk[:,c-1],c,ev['label_eligible'])
            assert known.all();np.testing.assert_array_equal(w,np.ones(len(w)))
            np.testing.assert_allclose([result[k] for k in ['auc','ap','brier']],direct_metrics(y,risk[:,c-1]),atol=1e-12,rtol=0,equal_nan=True)
            rows.append(dict(**tags,**result,calibration_gap=result['risk_mean']-result['ipcw_observed_incidence']))
            if s=='ensemble':
                for i,ix in enumerate(np.array_split(np.argsort(risk[:,c-1],kind='stable'),4)):
                    bins.append(dict(**tags,bin=i+1,n=len(ix),predicted=float(risk[ix,c-1].mean()),observed=float(y[ix].mean()),events=int(y[ix].sum())))
            if c==1:
                for fraction in [.2,.25]:
                    sel=selected(risk[:,0],ev['year'],ev['sample_ids'],fraction)
                    warnings.append(dict(**tags,top_fraction=fraction,checked=int(sel.sum()),captured=int((sel&y).sum()),target_events=int(y.sum()),
                        coverage=float((sel&y).sum()/y.sum()) if y.any() else np.nan,known_non_target=int((sel&~y).sum()),unknown=0))
    out=dest/'reports';df=pd.DataFrame(rows);wf=pd.DataFrame(warnings)
    for n,f in [('enterprise_metrics',df),('enterprise_warnings',wf),('enterprise_calibration_bins',pd.DataFrame(bins))]:f.to_csv(out/f'{n}.csv',index=False)
    # Reproduce reference endpoints independently of prior report formatting.
    prior=pd.read_csv(Path(__file__).resolve().parents[2]/'experiments/post_st_leaf_shallow_v1/results/metrics.csv',dtype={'scope':str,'seed':str})
    checks=0
    for r in df[df.arm=='reference'].itertuples():
        old=prior[(prior.scope==r.scope)&(prior.arm=='shallow_parent')&(prior.seed==r.seed)&(prior.horizon_months==r.horizon_months)&(prior.cause==r.cause)]
        assert len(old)==1
        np.testing.assert_allclose([getattr(r,k) for k in ['auc','ap','brier','risk_mean']],old.iloc[0][['auc','ap','brier','risk_mean']].to_numpy(float),atol=1e-12,rtol=0,equal_nan=True);checks+=1
    for y,h,c in itertools.product(YEARS,[12,24],['termination','recovery']):
        g=df[(df.scope==str(y))&(df.seed=='ensemble')&(df.horizon_months==h)&(df.cause==c)].set_index('arm')
        np.testing.assert_allclose(g.loc['affine',['auc','ap']].to_numpy(float),g.loc['reference',['auc','ap']].to_numpy(float),atol=1e-12,rtol=0,equal_nan=True)
    for scope,h in itertools.product(list(map(str,YEARS))+['pooled'],[12,24]):
        g=wf[(wf.scope==scope)&(wf.seed=='ensemble')&(wf.horizon_months==h)]
        for f in [.2,.25]:assert g[(g.arm=='affine')&(g.top_fraction==f)].captured.iloc[0]==g[(g.arm=='reference')&(g.top_fraction==f)].captured.iloc[0]
    np.savez_compressed(dest/'private/risks_PRIVATE.npz',**{'_'.join(map(str,k)):v for k,v in risks.items()})
    write(out/'enterprise_metric_validation.json',dict(status='PASS',metric_rows=len(df),reference_rows_reproduced=checks,
        complete_endpoint_weights_one=True,within_year_affine_auc_ap_and_top20_25_unchanged=True,calibration_fits=4,wide_forest_fits=6,
        pooled_rank_note='Year-specific calibrators can change pooled AUC/AP; annual Top20 quota stays fixed.'))
    bootstrap(dest,risks,data,df,wf)
    function_checks(run,paths,pred,calibrators,dest)
    return df

def bootstrap(run,risks,data,df,wf):
    ev=data['pooled'];rng=np.random.default_rng(20261031)
    indices=np.stack([np.concatenate([rng.choice(np.flatnonzero(ev['year']==y),sum(ev['year']==y),replace=True) for y in YEARS]) for _ in range(1000)])
    np.savez_compressed(run/'private/bootstrap_indices_PRIVATE.npz',indices=indices)
    yy=(ev['event']==1)&(ev['analysis_time']<=24);p={a:risks[('pooled',a,'ensemble',24)][:,0] for a in ['reference','affine','wide83']}
    rows=[]
    def values(ix,arm):
        y=yy[ix];risk=p[arm][ix];auc,ap,brier=direct_metrics(y,risk)
        selection=selected(risk,ev['year'][ix],ev['sample_ids'][ix],.2)
        capture=int((selection&y).sum());coverage=capture/y.sum() if y.any() else np.nan
        return np.array([auc,ap,brier,float(risk.mean()-y.mean()),capture,coverage])
    for scope in ['2023','2024','pooled']:
        draws=indices if scope=='pooled' else [ix[ev['year'][ix]==int(scope)] for ix in indices]
        original=np.arange(len(yy)) if scope=='pooled' else np.flatnonzero(ev['year']==int(scope))
        base=np.stack([values(ix,'reference') for ix in draws]);base0=values(original,'reference')
        for arm in ['affine','wide83']:
            d=np.stack([values(ix,arm) for ix in draws])-base;point=values(original,arm)-base0
            for j,metric_name in enumerate(['auc','ap','brier','calibration_gap','top20_captured','top20_coverage']):
                z=d[:,j];ok=np.isfinite(z)
                rows.append(dict(scope=scope,contrast=f'{arm}_minus_reference',metric=metric_name,estimate=float(point[j]),
                    low95=float(np.quantile(z[ok],.025)) if ok.any() else np.nan,high95=float(np.quantile(z[ok],.975)) if ok.any() else np.nan,
                    successful_resamples=int(ok.sum()),undefined_resamples=int((~ok).sum()),resamples=1000))
    pd.DataFrame(rows).to_csv(run/'reports/enterprise_paired_intervals.csv',index=False)

def function_checks(run,paths,pred,calibrators,dest):
    root=dest/'private/ensemble_verification';root.mkdir();rows=[];stability=[];prediction_variation=[]
    for y in YEARS:
        tr=arrays(run/'input/folds'/str(y)/'train_PRIVATE.npz');ev=arrays(run/'input/folds'/str(y)/'eval_PRIVATE.npz')
        for arm in ['reference','wide83','wide83_raw']:
            fs=[CompetingRiskForest.load(paths[(y,arm,str(s))]/'forest_PRIVATE.json') for s in SEEDS];f=merge(fs)
            folder=root/f'{y}_{arm}';result=verify_export(f,tr,ev,folder)
            np.testing.assert_allclose(f.predict(ev['X'],TIMES),pred[(str(y),arm,'ensemble')]['cif'],atol=1e-13,rtol=0)
            rows.append(dict(year=y,arm=arm,seed='ensemble',**result))
            parts=[]
            for s in SEEDS:
                source=paths[(y,arm,str(s))]
                if arm=='reference':
                    checked=root/f'{y}_{arm}_{s}';ss=verify_export(fs[SEEDS.index(s)],tr,ev,checked);rows.append(dict(year=y,arm=arm,seed=str(s),**ss));source=checked
                else:
                    rows.append(dict(year=y,arm=arm,seed=str(s),**json.loads((source/'verification.json').read_text())))
                parts.append(arrays(source/'components_PRIVATE.npz'))
            pp=np.stack([pred[(str(y),arm,str(s))]['cif'] for s in SEEDS]);prediction_variation.append(dict(year=y,arm=arm,mean_prediction_seed_sd_24m=float(pp[:,:,2,0].std(0,ddof=1).mean())))
            for ia,ib in [(0,1),(0,2),(1,2)]:
                a,b=parts[ia],parts[ib];maps=[{k:j for j,k in enumerate(z['keys'])} for z in [a,b]]
                active=[]
                for z in [a,b]:active.append({k for j,k in enumerate(z['keys']) if np.mean(abs(z['terms'][:,j,2,0]))>1e-14})
                vv={}
                for order in [1,2]:
                    sa={k for k in active[0] if len(k.split(','))==order};sb={k for k in active[1] if len(k.split(','))==order}
                    vv[f'order{order}_jaccard']=len(sa&sb)/len(sa|sb) if sa|sb else np.nan
                cor=[];rms=[]
                for key in maps[0].keys()&maps[1].keys():
                    u=a['terms'][:,maps[0][key],2,0];v=b['terms'][:,maps[1][key],2,0];rms.append(float(np.sqrt(np.mean((u-v)**2))))
                    if np.std(u)>1e-12 and np.std(v)>1e-12:cor.append(float(np.corrcoef(u,v)[0,1]))
                stability.append(dict(year=y,arm=arm,seed_a=SEEDS[ia],seed_b=SEEDS[ib],**vv,common_nonconstant=len(cor),median_function_correlation=float(np.median(cor)) if cor else np.nan,mean_function_RMS_difference=float(np.mean(rms)) if rms else np.nan))
    for folder in outputs(run,'formal')[456:]:
        s=json.loads((folder/'summary.json').read_text());rows.append(dict(year=s['year'],arm='affine',seed='ensemble',horizon=s['horizon'],**s['checks']))
    pd.DataFrame(rows).to_csv(dest/'reports/enterprise_function_validation.csv',index=False)
    pd.DataFrame(stability).to_csv(dest/'reports/enterprise_function_stability.csv',index=False)
    pd.DataFrame(prediction_variation).to_csv(dest/'reports/enterprise_prediction_variation.csv',index=False)

def main(run):
    for n,h in json.loads((run/'audit/immutable_input_hashes.json').read_text()).items():assert sha(run/'input'/n)==h
    for p,h in json.loads((run/'audit/source_hashes.json').read_text()).items():assert sha(Path(p))==h
    assert sha(Path(__file__).resolve().parents[2]/'CURRENT_POST_ST_MODEL.json')==REFERENCE_SHA
    from .sim_analysis import analyze_simulation
    analyze_simulation(run);enterprise(run)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--stage',choices=['all','simulation','enterprise'],default='all');a=p.parse_args()
    if a.stage=='all':main(a.run)
    elif a.stage=='simulation':
        from .sim_analysis import analyze_simulation
        analyze_simulation(a.run)
    else:enterprise(a.run)
