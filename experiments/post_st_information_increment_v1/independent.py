"""Independently reproduce the primary interval by event/control company multiplicities."""
from experiments.post_st_prognosis_v1.prepare import OLD
from .common import *
def main():
    r=run();risk=arrays(r/'risk_PRIVATE.npz');draws=arrays(r/'eval/bootstrap_PRIVATE.npz')
    nums=np.zeros((1000,2));denom=np.zeros(1000);source=source_rows=0
    for y in YEARS:
        data=arrays(r/f'eval/{y}.npz');target=(data['event']==1)&(data['time']<=data['h12'])
        known=(data['event']>0)|(data['time']>=data['h12']);assert known.all()
        rep=np.stack([np.bincount(row,minlength=len(target)) for row in draws[str(y)]])
        pos,neg=rep[:,target],rep[:,~target];denom+=pos.sum(1)*neg.sum(1)
        for j,a in enumerate(['M0','M2']):
            p=risk[f'{y}|{a}|fixed25|0|12|1'];v=(p[target,None]>p[None,~target]).astype(float)+.5*(p[target,None]==p[None,~target])
            nums[:,j]+=np.einsum('bi,ij,bj->b',pos,v,neg,optimize=True)
    valid=denom>0;diff=(nums[valid,1]-nums[valid,0])/denom[valid]
    table=pd.read_csv(D/'PAIRED_DIFFERENCES_95.csv')
    row=table.query("months==12 and cause==1 and year=='pooled' and effect=='market_vs_finance' and variant=='fixed25' and metric=='within_year_auc'").iloc[0]
    np.testing.assert_allclose([row.lower95,row.upper95],np.quantile(diff,[.025,.975]),rtol=0,atol=1e-12)
    assert row.valid==valid.sum()
    z=pd.read_parquet(r/'audit_sources_PRIVATE.parquet')
    info=pd.read_parquet(CLOCK/'audit/information_alignment_PRIVATE.parquet').set_index('sample_id')
    events=pd.read_csv(OLD/'phase1_data_freeze_v1/03_event_history.csv',dtype=str,keep_default_na=False).set_index('event_id')
    detail=[]
    for row in z[~z.original_before & ~z.prior_fiscal_IAR_before].itertuples():
        meta=info.loc[row.sample_id];entry=events.loc[meta.source_ids.split(';')]
        text=' '.join(entry.Content+' '+entry.supplemental_content)
        detail.append(dict(year=row.year,has_opinion_literal=meta.raw_prior_year_audit_opinions in text,
                           has_going_concern_mention='持续经营' in text,has_annual_report_mention=('年度报告' in text or '年报' in text)))
    screen=pd.DataFrame(detail)
    assert len(screen)>0
    by_year=[]
    for y in YEARS:
        train=arrays(r/f'input/folds/{y}/train_PRIVATE.npz');zz=z.set_index('sample_id').loc[train['sample_ids']]
        by_year.append(dict(year=y,train=len(zz),no_dated_original_or_IAR=int((~zz.original_before&~zz.prior_fiscal_IAR_before).sum())))
    dump(D/'INDEPENDENT_CHECKS.json',dict(primary_bootstrap_independent_concordance=True,valid=int(valid.sum()),interval=np.quantile(diff,[.025,.975]).tolist(),
        prior_fiscal_no_timely_original_or_IAR=len(screen),both_entry_and_supplement_lacking_both=int((~screen.has_opinion_literal & ~screen.has_going_concern_mention).sum()),per_training_cutoff=by_year,
        caution='Different report years or other separately dated filings are possible; no proof that a future outcome was used. A certified versioned source key is required.'))
if __name__=='__main__':main()
