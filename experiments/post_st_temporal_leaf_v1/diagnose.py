"""Training-only entry-cohort AJ diagnostic, including horizon support."""
from pathlib import Path
import argparse,datetime,json
import numpy as np
import pandas as pd
from crforest.core import aalen_johansen
from experiments.post_st_prognosis_v1.common import arrays,calendar_elapsed,sha,write


def estimate(time,event,horizons):
    curve=aalen_johansen(time,event,np.asarray(horizons,float),2)
    support=np.array([np.any(time>=h) or curve[i].sum()>1-1e-12 for i,h in enumerate(horizons)])
    return curve,support


def run(root):
    freeze=json.loads((root/'audit/step0_freeze.json').read_text())
    old=Path(freeze['preserved_run']);horizons=freeze['diagnostic_horizons_months']
    rows=[];contrasts=[];source={};rrows=[];rng=np.random.default_rng(freeze['bootstrap_seed'])
    for year in [2023,2024]:
        mp=old/'evaluation_input'/f'{year}_fit_metadata_PRIVATE.parquet'
        ap=old/'input/folds'/str(year)/'train_PRIVATE.npz'
        source[str(mp)]=sha(mp);source[str(ap)]=sha(ap)
        meta=pd.read_parquet(mp);data=arrays(ap)
        np.testing.assert_array_equal(meta.sample_id,data['sample_ids'])
        np.testing.assert_array_equal(meta.group,data['groups'])
        np.testing.assert_array_equal(meta.event,data['event'])
        np.testing.assert_array_equal(meta.time,data['time'])
        assert meta.group.is_unique and meta.sample_id.is_unique and meta.label_eligible.all()
        assert (meta.stop<pd.Timestamp(f'{year}-01-01')).all()
        assert (meta.origin<meta.stop).all() and (meta.origin.dt.year<year).all()
        time=np.array([calendar_elapsed(a,b) for a,b in zip(meta.origin,meta.stop)])
        np.testing.assert_allclose(time,data['analysis_time'],atol=1e-12)
        assert time.max()<=24 and np.isin(data['event'],[0,1,2]).all()
        masks={str(y):meta.origin.dt.year.to_numpy()==y for y in sorted(meta.origin.dt.year.unique())}
        masks.update(pre_2021=meta.origin.dt.year.to_numpy()<2021,post_2020=meta.origin.dt.year.to_numpy()>=2021,all=np.ones(len(meta),bool))
        boots={}
        for cohort,mask in masks.items():
            t=time[mask];e=data['event'][mask];n=len(t)
            curve,support=estimate(t,e,horizons)
            sampled=[]
            for _ in range(freeze['bootstrap_repeats']):
                ix=rng.integers(n,size=n);c,s=estimate(t[ix],e[ix],horizons)
                sampled.append(np.where(s[:,None],c,np.nan))
            boot=np.array(sampled);boots[cohort]=boot
            for i,h in enumerate(horizons):
                finite=np.isfinite(boot[:,i,0]);b=boot[finite,i,0]
                row=dict(cutoff_year=year,cohort=cohort,horizon_months=h,companies=n,
                    all_observed_terminations=int((e==1).sum()),all_observed_recoveries=int((e==2).sum()),
                    target_events_by_h=int(((e==1)&(t<=h)).sum()),recovery_events_by_h=int(((e==2)&(t<=h)).sum()),
                    early_censored=int(((e==0)&(t<h)).sum()),risk_set_at_h=int((t>=h).sum()),
                    horizon_supported=bool(support[i]),max_followup_months=float(t.max()),
                    aj_termination=float(curve[i,0]) if support[i] else None,
                    aj_recovery=float(curve[i,1]) if support[i] else None,
                    lower_95=float(np.quantile(b,.025)) if support[i] and len(b) else None,
                    upper_95=float(np.quantile(b,.975)) if support[i] and len(b) else None,
                    supported_bootstraps=int(finite.sum()),bootstrap_repeats=freeze['bootstrap_repeats'],
                    note='no horizon support; no flat-tail extrapolation' if not support[i] else ('risk set below 5' if (t>=h).sum()<5 else ''))
                rows.append(row)
                # Aggregate/nonidentifying rows for the independent R AJ check.
                rrows.extend(dict(cutoff_year=year,cohort=cohort,time=float(tt),event=int(ee)) for tt,ee in zip(t,e)) if i==0 else None
            if cohort=='all':assert sum((e==1)) in [14,26]
        for i,h in enumerate(horizons):
            difference=boots['post_2020'][:,i,0]-boots['pre_2021'][:,i,0]
            valid=difference[np.isfinite(difference)]
            before=next(r for r in rows if r['cutoff_year']==year and r['cohort']=='pre_2021' and r['horizon_months']==h)
            after=next(r for r in rows if r['cutoff_year']==year and r['cohort']=='post_2020' and r['horizon_months']==h)
            supported=before['horizon_supported'] and after['horizon_supported']
            contrasts.append(dict(cutoff_year=year,horizon_months=h,horizon_supported=supported,
                post_minus_pre_pp=100*(after['aj_termination']-before['aj_termination']) if supported else None,
                lower_95_pp=float(np.quantile(valid,.025)*100) if supported and len(valid) else None,
                upper_95_pp=float(np.quantile(valid,.975)*100) if supported and len(valid) else None,
                valid_bootstrap_contrasts=len(valid),note='Descriptive entry-cohort contrast, not a causal policy effect; administrative censoring may differ by entry year'))
    pd.DataFrame(rows).to_csv(root/'reports/cohort_aj.csv',index=False)
    pd.DataFrame(contrasts).to_csv(root/'reports/cohort_aj_contrasts.csv',index=False)
    pd.DataFrame(rrows).to_csv(root/'private/aj_check_input_PRIVATE.csv',index=False)
    write(root/'audit/step0_receipt.json',dict(status='COMPLETE',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_sha256=source,script_sha256=sha(Path(__file__)),new_real_fits=0,test_metadata_read=False,
        bootstrap_unit='company; one record per company in frozen first-ST data',
        limits=['post_2020 means entry >=2021-01-01, a descriptive calendar contrast','No unsupported 12/24-month estimate','Bootstrap uncertainty conditional on source/cohort definitions, not policy causal inference']))
    print(pd.DataFrame(rows).query("cohort in ['pre_2021','post_2020','all']").to_string(index=False))
    print(pd.DataFrame(contrasts).to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);run(p.parse_args().run)
