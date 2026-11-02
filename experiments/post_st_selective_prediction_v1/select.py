"""Freeze every acceptance mask before reading any outer evaluation outcome."""
import pandas as pd
from .common import *

def history(diagnostics,year,h):
    membership=arrays(RUN/f'input/history/{year}/{h}/membership_PRIVATE.npz');parts=[]
    for cohort in cohorts(year,h):
        d=diagnostics[cohort];sel=membership['cohort']==cohort
        np.testing.assert_array_equal(d['sample_ids'],membership['sample_ids'][sel]);parts.append(d)
    return membership,parts

def load_diagnostics():
    records=load(RUN/'audit/registry.json');d={}
    for r in records:
        p=Path(r['path'])
        for n,h in load(p/'hashes.json').items():assert sha(p/n)==h
        assert r['new_forests']==0 and r['no_evaluation_outcomes_read'];d[r['year']]=arrays(p/'diagnostics_PRIVATE.npz')
    return d

def main():
    block_forest_fits();d=load_diagnostics();masks={};cuts=[];rep_masks={};summaries=[];distributions=[];historical_masks={}
    for y in YEARS:
        for h in [12,24]:
            membership,parts=history(d,y,h);histids=membership['sample_ids'];ids=d[y]['sample_ids']
            for c in [1,2]:
                reference=np.concatenate([matrix(p,h,c) for p in parts]);values=matrix(d[y],h,c)
                for j,name in enumerate(RAW):
                    for source,v in [('history',reference[:,j]),('evaluation',values[:,j])]:
                        distributions.append(dict(year=y,months=h,cause=c,proxy=name,source=source,n=len(v),mean=float(v.mean()),p10=float(np.quantile(v,.1)),median=float(np.median(v)),p90=float(np.quantile(v,.9))))
                for arm in ARMS:
                    rs=scores(reference,reference,arm);vs=scores(reference,values,arm)
                    for q in COVERAGES:
                        key=f'{y}|{h}|{c}|{arm}|{q}';cut=threshold(rs,histids,q);mask=accept(vs,ids,cut);hmask=accept(rs,histids,cut)
                        cuts.append(dict(year=y,months=h,cause=c,arm=arm,coverage=q,**cut));masks[key]=mask;historical_masks[key]=hmask
                        summaries.append(dict(year=y,months=h,cause=c,arm=arm,target_coverage=q,historical_companies=len(rs),historical_accepted=int(hmask.sum()),historical_coverage=float(hmask.mean()),
                            evaluation_companies=len(vs),accepted=int(mask.sum()),referred=int((~mask).sum()),realized_coverage=float(mask.mean()),empty_acceptance=not mask.any()))
                        rep_masks[key]=[]
                for rep in range(50):
                    indices=[]
                    for cohort in cohorts(y,h):
                        ix=np.flatnonzero(membership['cohort']==cohort);rng=np.random.default_rng(np.random.SeedSequence([HISTORY_SEED,rep,cohort]));indices.extend(ix[rng.integers(len(ix),size=len(ix))])
                    index=np.asarray(indices);rr=reference[index];ri=histids[index]
                    for arm in ARMS:
                        rs=scores(rr,rr,arm);vs=scores(rr,values,arm)
                        for q in COVERAGES:
                            rep_masks[f'{y}|{h}|{c}|{arm}|{q}'].append(accept(vs,ids,threshold(rs,ri,q)))
                # Same-count secondary controls depend on current score ranks,
                # unlike the individually applicable frozen historical threshold.
                for q in COVERAGES:
                    n=int(masks[f'{y}|{h}|{c}|combined|{q}'].sum())
                    for arm in ARMS:
                        masks[f'{y}|{h}|{c}|matched_{arm}|{q}']=matched(scores(reference,values,arm),ids,n)
                    np.testing.assert_array_equal(masks[f'{y}|{h}|{c}|matched_combined|{q}'],masks[f'{y}|{h}|{c}|combined|{q}'])
    np.savez_compressed(RUN/'private/masks_PRIVATE.npz',**masks);np.savez_compressed(RUN/'private/historical_masks_PRIVATE.npz',**historical_masks)
    np.savez_compressed(RUN/'private/history_bootstrap_masks_PRIVATE.npz',**{k:np.asarray(v) for k,v in rep_masks.items()})
    write(RUN/'private/thresholds_PRIVATE.json',cuts)
    stable=[]
    for key,rep in rep_masks.items():
        year,h,c,arm,q=key.split('|');r=np.asarray(rep);m=masks[key];intersection=(r&m).sum(1);union=(r|m).sum(1);jaccard=np.divide(intersection,union,out=np.ones(50),where=union>0)
        stable.append(dict(year=int(year),months=int(h),cause=int(c),arm=arm,target_coverage=float(q),replicates=50,mean_coverage=float(r.mean()),coverage_sd=float(r.mean(1).std(ddof=1)),
            mean_mask_jaccard=float(jaccard.mean()),mean_agreement_with_point=float((r==m).mean()),ambiguous_acceptance_companies=int(((r.mean(0)>.1)&(r.mean(0)<.9)).sum())))
    for name,rows in [('ACCEPTANCE_COUNTS',summaries),('GATE_ESTIMATION_STABILITY',stable),('RELIABILITY_PROXY_DISTRIBUTIONS',distributions)]:pd.DataFrame(rows).to_csv(RUN/f'reports/{name}.csv',index=False)
    write(RUN/'audit/selection_frozen.json',dict(status='PASS',point_thresholds=len(cuts),bootstrap_thresholds=len(cuts)*50,history_bootstraps=50,new_forests=0,new_calibrators=0,
        no_outer_outcome_file_opened=True,masks_sha256=sha(RUN/'private/masks_PRIVATE.npz'),historical_masks_sha256=sha(RUN/'private/historical_masks_PRIVATE.npz'),
        bootstrap_masks_sha256=sha(RUN/'private/history_bootstrap_masks_PRIVATE.npz'),threshold_sha256=sha(RUN/'private/thresholds_PRIVATE.json')))
    print('Frozen',len(cuts),'historical thresholds and',len(cuts)*50,'resampled thresholds; zero forest or probability fits.')

if __name__=='__main__':main()
