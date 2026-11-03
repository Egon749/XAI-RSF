"""Fixed OOB tiers on outer cohorts, with complementary small-cell suppression."""
from pathlib import Path
import argparse,json,os,traceback
import numpy as np
import pandas as pd
from experiments.post_st_reporting_drift_v1.analyze import Reader,write_json,digest
from experiments.post_st_reporting_drift_v1.metrics import interval
from experiments.post_st_feature_permutation_v1.common import evaluation_context
from experiments.post_st_risk_index_v1.index import tiers

PUBLIC=Path(__file__).resolve().parent
LEVELS=['low','medium','high']


def suppress_event_cells(frame,keys):
    f=frame.copy();f['publication_status']='REPORTED'
    for _,block in f.groupby(keys,dropna=False):
        small_n=block.companies.between(1,4).any()
        if small_n:
            f.loc[block.index,['companies','events','observed_rate','lift']]=np.nan
            f.loc[block.index,'publication_status']='SUPPRESSED_SMALL_COMPANY_OR_COMPLEMENT'
        elif block.events.between(1,4).any():
            f.loc[block.index,['events','observed_rate','lift']]=np.nan
            f.loc[block.index,'publication_status']='SUPPRESSED_SMALL_EVENT_OR_COMPLEMENT'
    return f


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-name',default='analysis');args=parser.parse_args()
    if Path(args.output_name).name!=args.output_name:raise ValueError('Output must be one new directory name')
    loc=json.loads((PUBLIC/'run_location.json').read_text());run=Path(os.environ.get('RISK_INDEX_RUN',loc['private_run']))
    out=run/args.output_name;out.mkdir();pub=out/'public';pub.mkdir();reader=Reader()
    assert reader.json(run/'A_GATE.json')['status']=='PASS'
    regs={part:reader.json(run/f'registry_{part}.json') for part in ['A','B']}
    draws=reader.arrays(run/'input/bootstrap_indices_PRIVATE.npz')
    tables={};labs={};preds={};tiered={};qs={};rows=[];shares=[];overlap=[];monos=[];diffs=[];aux=[]
    for part in ['A','B']:
        for rec in regs[part]:
            path=Path(rec['path']);hashes=reader.json(path/'hashes.json')
            for name,h in hashes.items():assert digest(reader.path(path/name))==h
    for name in ['INDEX_SUMMARY','EXACTNESS_CHECKS','RANK_IDENTITY_CHECKS','GROUP_POINTS_SUMMARY']:
        tables[name]=pd.concat([pd.read_csv(Path(r['path'])/(name+'_PRIVATE.csv')) for r in regs['A']],ignore_index=True)
    tables['OOB_SUPPORT']=pd.concat([pd.read_csv(Path(r['path'])/'OOB_SUPPORT.csv') for r in regs['B']],ignore_index=True)
    for y in [2023,2024,2025]:
        labs[y]=reader.arrays(run/f'input/{y}/eval_PRIVATE.npz')
        a=next(r for r in regs['A'] if r['year']==y);b=next(r for r in regs['B'] if r['year']==y)
        preds[y]=reader.arrays(Path(a['path'])/'endpoints_PRIVATE.npz')
        q=reader.arrays(Path(b['path'])/'oob_PRIVATE.npz')['thresholds'];qs[y]=q
        for hi,h in enumerate([12,24]):
            lab=labs[y];n=len(lab['X']);supported=not(y==2025 and h==24)
            for c in [1,2]:
                threshold=q[hi,c-1];valid=bool(np.isfinite(threshold).all())
                group=tiers(preds[y][f'index_{h}'][:,c-1],threshold if valid else None);tiered[y,h,c]=group
                counts=np.array([(group==t).sum() for t in LEVELS])
                for ti,t in enumerate(LEVELS):shares.append(dict(year=y,months=h,index_cause=c,tier=t,companies=int(counts[ti]),cohort_companies=n,share=counts[ti]/n,nominal_share=[.5,.3,.2][ti],deviation=counts[ti]/n-[.5,.3,.2][ti],threshold_supported=valid))
                checked=int(np.ceil(.2*n));ids=lab['sample_ids'];p=preds[y][f'cif_{h}'][:,c-1]
                sel=np.lexsort((ids,-p))[:checked];top=np.zeros(n,bool);top[sel]=True;high=group=='high'
                overlap.append(dict(year=y,months=h,index_cause=c,high_companies=int(high.sum()),top20_companies=checked,intersection=int((top&high).sum()),high_only=int((high&~top).sum()),top20_only=int((top&~high).sum()),neither=int((~top&~high).sum()),jaccard=float((top&high).sum()/(top|high).sum()),threshold_supported=valid))
                for outcome in [1,2]:
                    yy=evaluation_context(lab,h,outcome)[0] if supported else None
                    if supported:
                        _,w,known,sup,_=evaluation_context(lab,h,outcome);assert sup and known.all() and (w==1).all()
                    rates=[]
                    for t in LEVELS:
                        mask=group==t;nc=int(mask.sum());ne=int(yy[mask].sum()) if supported and valid else np.nan
                        rate=ne/nc if nc else np.nan;overall=float(yy.mean()) if supported else np.nan
                        rates.append(rate)
                        rows.append(dict(year=y,months=h,index_cause=c,outcome_cause=outcome,tier=t,companies=nc,events=ne,observed_rate=rate,overall_observed=overall,lift=rate/overall if overall>0 else np.nan,supported=supported and valid))
                    finite=bool(np.isfinite(rates).all())
                    monos.append(dict(year=y,months=h,index_cause=c,outcome_cause=outcome,nondecreasing=bool(np.all(np.diff(rates)>=0)) if finite else np.nan,nonincreasing=bool(np.all(np.diff(rates)<=0)) if finite else np.nan,status='DEFINED' if finite else 'NA_EMPTY_TIER_OR_NO_FOLLOWUP'))
                    for t,mask in [('high',high),('not_high',~high)]:
                        nc=int(mask.sum());ne=int(yy[mask].sum()) if supported and valid else np.nan
                        rate=ne/nc if nc else np.nan;overall=float(yy.mean()) if supported else np.nan
                        aux.append(dict(year=y,months=h,index_cause=c,outcome_cause=outcome,tier=t,companies=nc,events=ne,observed_rate=rate,overall_observed=overall,lift=rate/overall if overall>0 else np.nan,share=nc/n,supported=supported and valid))
    # Paired draws are reused verbatim; no parameters/tiers re-estimated.
    for h in [12,24]:
        for scope in [2023,2024,2025,'pooled']:
            ys=([2023,2024,2025] if h==12 else [2023,2024]) if scope=='pooled' else [scope]
            for c in [1,2]:
                for outcome in [1,2]:
                    head=dict(scope=str(scope),months=h,index_cause=c,outcome_cause=outcome)
                    if scope==2025 and h==24:
                        diffs.append(dict(**head,estimate=np.nan,lower95=np.nan,upper95=np.nan,valid_repeats=np.nan,undefined_repeats=np.nan,status='NOT_RUN_NO_HORIZON_SUPPORT'));continue
                    yy=np.concatenate([evaluation_context(labs[y],h,outcome)[0] for y in ys]);g=np.concatenate([tiered[y,h,c] for y in ys])
                    offsets=np.r_[0,np.cumsum([len(labs[y]['X']) for y in ys])]
                    ix=np.concatenate([draws[str(y)]+offsets[i] for i,y in enumerate(ys)],axis=1)
                    def gap(z,group):
                        lo=group=='low';hi=group=='high'
                        return float(z[hi].mean()-z[lo].mean()) if lo.any() and hi.any() else np.nan
                    vals=[gap(yy[i],g[i]) for i in ix]
                    diffs.append(dict(**head,estimate=gap(yy,g),**interval(vals),status='FIXED_PREDICTIONS_AND_THRESHOLDS'))
    np.savez_compressed(out/'tiers_PRIVATE.npz',**{f'{y}_{h}_{c}':v for (y,h,c),v in tiered.items()})
    pd.DataFrame(rows).to_csv(out/'TIER_PERFORMANCE_PRIVATE.csv',index=False)
    pd.DataFrame(aux).to_csv(out/'HIGH_REST_PRIVATE.csv',index=False)
    pd.DataFrame(shares).to_csv(out/'TIER_SHARE_PRIVATE.csv',index=False)
    pd.DataFrame(overlap).to_csv(out/'TOP20_OVERLAP_PRIVATE.csv',index=False)
    pd.DataFrame(diffs).to_csv(out/'TIER_DIFFERENCE_PRIVATE.csv',index=False)
    # Complement suppression applies separately to each outcome and index cause.
    keys=['year','months','index_cause','outcome_cause']
    perf=suppress_event_cells(pd.DataFrame(rows),keys)
    highrest=suppress_event_cells(pd.DataFrame(aux),keys)
    highrest.loc[highrest.publication_status=='SUPPRESSED_SMALL_COMPANY_OR_COMPLEMENT','share']=np.nan
    share=pd.DataFrame(shares);share['publication_status']='REPORTED'
    for _,b in share.groupby(['year','months','index_cause']):
        if b.companies.between(1,4).any():
            share.loc[b.index,['companies','share','deviation']]=np.nan;share.loc[b.index,'publication_status']='SUPPRESSED_SMALL_COMPANY_OR_COMPLEMENT'
    ov=pd.DataFrame(overlap);ov['publication_status']='REPORTED'
    countcols=['high_companies','top20_companies','intersection','high_only','top20_only','neither']
    bad=ov[countcols].apply(lambda z:z.between(1,4)).any(axis=1)
    ov.loc[bad,countcols+['jaccard']]=np.nan;ov.loc[bad,'publication_status']='SUPPRESSED_SMALL_OVERLAP_OR_COMPLEMENT'
    differences=pd.DataFrame(diffs);differences['publication_status']='REPORTED'
    for i,r in differences.iterrows():
        ys=([2023,2024,2025] if r.months==12 else [2023,2024]) if r.scope=='pooled' else [int(r.scope)]
        b=perf[(perf.year.isin(ys))&(perf.months==r.months)&(perf.index_cause==r.index_cause)&(perf.outcome_cause==r.outcome_cause)]
        if b.publication_status.str.startswith('SUPPRESSED').any():
            differences.loc[i,['estimate','lower95','upper95']]=np.nan;differences.loc[i,'publication_status']='SUPPRESSED_LINKED_SMALL_CELLS'
    ranks=tables['RANK_IDENTITY_CHECKS'].copy()
    for col in ['captured_reference','captured_index','precision_reference','precision_index','coverage_reference','coverage_index']:ranks=ranks.drop(columns=[col],errors='ignore')
    ranks['annual_metrics_exact_identity']=np.where(ranks.supported,True,np.nan)
    tables['RANK_IDENTITY_CHECKS']=ranks
    tables.update(TIER_PERFORMANCE=perf,TIER_SHARE=share,TIER_DIFFERENCE_95=differences,HIGH_REST_PERFORMANCE=highrest,TIER_MONOTONICITY=pd.DataFrame(monos),TOP20_OVERLAP=ov)
    for name,frame in tables.items():frame.to_csv(pub/(name+'.csv'),index=False)
    pd.read_csv(run/'WITHIN_YEAR_IDENTITY.csv').to_csv(pub/'WITHIN_YEAR_IDENTITY.csv',index=False)
    reader.finish();write_json(out/'input_hashes_PRIVATE.json',reader.hashes)
    print(highrest.query("index_cause==1 and outcome_cause==1 and tier=='high'")[["year","months","companies","share","observed_rate","lift","publication_status"]].to_string(index=False))
    print(pd.DataFrame(monos).query('index_cause==1 and outcome_cause==1').to_string(index=False))


if __name__=='__main__':
    try:main()
    except Exception:
        loc=json.loads((PUBLIC/'run_location.json').read_text());run=Path(os.environ.get('RISK_INDEX_RUN',loc['private_run']))
        (run/'ANALYSIS_FAILURE.txt').write_text(traceback.format_exc());raise
