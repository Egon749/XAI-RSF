"""Independently check all five paired AUC intervals and pooled seed/ensemble identities."""
from experiments.post_st_prognosis_v1.prepare import OLD
from sklearn.metrics import average_precision_score,roc_auc_score
from .metrics import DESIGNS
from .common import *
def verify():
    r=run();risk=arrays(r/'risk_PRIVATE.npz');draws=arrays(r/'eval/bootstrap_PRIVATE.npz')
    interval=pd.read_csv(D/'PAIRED_DIFFERENCES_95.csv');metric=pd.read_csv(D/'METRICS_ALL.csv')
    checks=[];maxerr=0.
    for variant in ['fixed25','proportional']:
        denom=np.zeros(1000);nums={a:np.zeros(1000) for a in SETS};observed=[];ranking={a:[] for a in SETS}
        for y in YEARS:
            dat=arrays(r/f'eval/{y}.npz');assert ((dat['event']>0)|(dat['time']>=dat['h12'])).all()
            yy=(dat['event']==1)&(dat['time']<=dat['h12']);observed.extend(yy)
            w=np.stack([np.bincount(z,minlength=len(yy)) for z in draws[str(y)]])
            pos,neg=w[:,yy],w[:,~yy];denom+=pos.sum(1)*neg.sum(1)
            for a in SETS:
                p=risk[f'{y}|{a}|{variant}|0|12|1'];ranking[a].extend(p)
                ps=np.stack([risk[f'{y}|{a}|{variant}|{s}|12|1'] for s in SEEDS])
                np.testing.assert_array_equal(ps.mean(0),p)
                m=(p[yy,None]>p[None,~yy]).astype(float)+.5*(p[yy,None]==p[None,~yy])
                nums[a]+=np.einsum('bi,ij,bj->b',pos,m,neg,optimize=True)
        for a in SETS:
            p=np.asarray(ranking[a]);z=metric.query("year=='pooled' and months==12 and cause==1 and seed==0 and variant==@variant and arm==@a").iloc[0]
            np.testing.assert_allclose([z.auc,z.ap,z.brier],[roc_auc_score(observed,p),average_precision_score(observed,p),np.mean((np.array(observed)-p)**2)],atol=1e-13,rtol=0)
        ok=denom>0
        for label,weights in DESIGNS.items():
            dist=sum(v*nums[a] for a,v in weights.items())[ok]/denom[ok]
            obs=interval.query("year=='pooled' and months==12 and cause==1 and variant==@variant and metric=='within_year_auc' and effect==@label").iloc[0]
            err=float(abs(np.quantile(dist,[.025,.975])-[obs.lower95,obs.upper95]).max());assert err<1e-12 and int(obs.valid)==len(dist)
            maxerr=max(err,maxerr);checks.append(dict(variant=variant,effect=label,valid=len(dist),interval_error=err))
    probe=pd.read_parquet(r/'audit_sources_PRIVATE.parquet')
    info=pd.read_parquet(CLOCK/'audit/information_alignment_PRIVATE.parquet').set_index('sample_id')
    events=pd.read_csv(OLD/'phase1_data_freeze_v1/03_event_history.csv',dtype=str,keep_default_na=False).set_index('event_id')
    gap=probe[~probe.original_before & ~probe.prior_fiscal_IAR_before]
    missing=[]
    for sid in gap.sample_id:
        f=info.loc[sid];e=events.loc[f.source_ids.split(';')];txt=' '.join(e.Content+' '+e.supplemental_content)
        if f.raw_prior_year_audit_opinions not in txt and '持续经营' not in txt:missing.append(sid)
    dump(D/'INDEPENDENT_CHECKS.json',dict(bootstrap_intervals=checks,max_interval_error=maxerr,company_seed_mean_exact=True,six_or_other_unresolved=len(missing),
         train_cutoffs_with_no_annual_or_explicit_entry_support={str(y):int(np.isin(arrays(r/f'input/folds/{y}/train_PRIVATE.npz')['sample_ids'],missing).sum()) for y in YEARS}))
if __name__=='__main__':verify()
