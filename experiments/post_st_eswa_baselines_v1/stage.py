import shutil
import pandas as pd
from experiments.post_st_paper_features_v1.common import history
from experiments.post_st_prognosis_v1.common import endpoint_weights
from .common import *
def main():
    root=old('post_st_seed_explanation_stability_v1')/'input';r=run();inp=r/'input';inp.mkdir();report=[];hashes={}
    for y in YEARS:
        folder=inp/str(y);folder.mkdir();tr=arrays(root/f'folds/{y}/train_PRIVATE.npz');ev=arrays(root/f'folds/{y}/eval_PRIVATE.npz');lab=arrays(root/f'evaluation_input/{y}_test_PRIVATE.npz')
        assert tr['X'].shape[1]==ev['X'].shape[1]==74 and set(ev)=={'X','sample_ids'}
        np.testing.assert_array_equal(ev['sample_ids'],lab['sample_ids']);assert not set(tr['groups'])&set(lab['groups'])
        m=pd.read_parquet(history(y)/f'evaluation_input/{y}_fit_metadata_PRIVATE.parquet')
        np.testing.assert_array_equal(m.sample_id,tr['sample_ids']);assert (m.origin<pd.Timestamp(y,1,1)).all() and (m.stop<pd.Timestamp(y,1,1)).all()
        assert (m.financial_available.isna()|(m.financial_available<m.origin)).all() and (m.market_available.isna()|(m.market_available<m.origin)).all()
        for role in ['train','eval']:
            p=root/f'folds/{y}/{role}_PRIVATE.npz';shutil.copyfile(p,folder/(role+'.npz'));hashes[f'{y}/{role}']=sha(p)
        hashes[f'{y}/labels']=sha(root/f'evaluation_input/{y}_test_PRIVATE.npz')
        for h in (12,24):
            for role,d in [('train',tr),('eval',lab)]:
                hd=Horizon(d,h,'IPCW');ix=np.arange(len(d['X']));w,known,_=endpoint_weights(hd.t,hd.e,h,d['label_eligible']);kk,yy,ww=hd.weighted(ix,ix);np.testing.assert_array_equal(kk,np.flatnonzero(known));np.testing.assert_allclose(ww,w[known],rtol=1e-14);assert ((hd.e==1)&(hd.t<=h)).sum()==sum(yy)
                cens=~known;reasons=m.loc[cens,'reason'].value_counts().to_dict() if role=='train' else {}
                report.append(dict(year=y,months=h,role=role,firms=len(ix),known=sum(known),termination=sum(yy),recovery=sum((hd.e==2)&(hd.t<=h)),event_free_at_H=sum(known)-sum((hd.e>0)&(hd.t<=h)),early_censored=sum(cens),early_censor_days_min=min(d['time'][cens]) if cens.any() else None,early_censor_days_median=np.median(d['time'][cens]) if cens.any() else None,early_censor_days_max=max(d['time'][cens]) if cens.any() else None,reason_counts=json.dumps(reasons),training_weight_max=ww.max() if role=='train' else None,supervised_rows=len(kk) if role=='train' else None,scored_evaluation=len(ix) if role=='eval' else None,formal_performance=role=='eval' and (y,h)!=(2025,24)))
    pd.DataFrame(report).to_csv(D/'CENSORING_AND_LABEL_AUDIT.csv',index=False)
    dump(D/'DATA_ALIGNMENT_AUDIT.json',dict(source='identical 74-input C039 models and Jan1 original training folds',hashes=hashes,one_record_per_company=True,groups_disjoint=True,evaluation_alignment=True,train_cutoff_verified=True,strict_vintage='Recorded market and financial clocks checked; inherited financial-vintage and historical audit-report availability are provisional. No audit-field timestamps invented.',training_KM_crosschecked=True,unknown_labels=-1,fit_receives_outer_evaluation_labels=False))
if __name__=='__main__':main()
