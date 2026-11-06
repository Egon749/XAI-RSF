"""Read-only checks of alignment, clocks, binary pairs and screening counts."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from experiments.post_st_prognosis_v1.common import arrays,horizon_predictions,calendar_elapsed
from .clock import endpoint
from .raw_scan import sha

def validate(run):
    d=pd.read_parquet(run/'private/landmarks_PRIVATE.parquet');z=arrays(run/'private/all_features_PRIVATE.npz')
    np.testing.assert_array_equal(z['sample_ids'],d.sample_id.to_numpy(str));assert d.sample_id.is_unique
    origins=d.origin.to_numpy(dtype='datetime64[D]')[:,None];finite=np.isfinite(z['X'][:,:48])
    assert (z['financial_available'][finite]<np.broadcast_to(origins,finite.shape)[finite]).all()
    assert (d.financial_available_at<d.origin).all() and (d.market_available_at<d.origin).all()
    positions={s:i for i,s in enumerate(d.sample_id)};forecasts=arrays(run/'private/evaluation_predictions_PRIVATE.npz')
    reported=pd.read_csv(run/'reports/METRICS_ALL.csv');errors=[];checks=[]
    for year in [2023,2024,2025]:
        for split in ['train','eval']:
            m=pd.read_parquet(run/f'evaluation_input/{year}_{split}_PRIVATE.parquet');data=arrays(run/f'prepared/folds/{year}/{split}_PRIVATE.npz')
            ix=[positions[s] for s in m.sample_id];np.testing.assert_array_equal(z['X'][ix],data['X'])
            np.testing.assert_array_equal(m.sample_id.to_numpy(str),data['sample_ids']);assert m.time.gt(0).all()
            if split=='train':
                cutoff=pd.Timestamp(year,1,1);assert m.origin.lt(cutoff).all() and m.stop.lt(cutoff).all()
                assert m.loc[m.event.gt(0),'event_date'].lt(cutoff).all()
            else:
                for h in [12,24]:
                    if year==2025 and h==24:continue
                    p=horizon_predictions(forecasts[f'{year}_ensemble_shrunk'],np.array([365.,366.,730.,731.]),m[f'h{h}'].to_numpy())
                    t=np.array([calendar_elapsed(o,s) for o,s in zip(m.origin,m.stop)])
                    for k in [1,2]:
                        known,y=endpoint(t,m.event.to_numpy(),h,k);a=p[known&y,k-1];b=p[known&~y,k-1]
                        score=float(np.sum(a[:,None]>b[None,:])+.5*np.sum(a[:,None]==b[None,:]))/(len(a)*len(b)) if len(a)*len(b) else np.nan
                        row=reported[(reported.scope==str(year))&(reported.seed=='ensemble')&(reported.version=='shrunk')&(reported.horizon_months==h)&(reported.cause==k)].iloc[0]
                        errors.append(abs(score-row.auc))
                        n=int(np.ceil(.2*len(m)));chosen=sorted(range(len(m)),key=lambda i:(-p[i,k-1],m.sample_id.iloc[i]))[:n]
                        assert int(y[chosen].sum())==int(row.top20_captured)
                        checks.append(dict(year=year,horizon_months=h,cause=k,auc_pair_count_error=abs(score-row.auc),top20_capture_matches=True))
    na=reported[(reported.scope=='2025')&(reported.horizon_months==24)]
    assert na[['auc','ap','ipcw_brier','top20_captured','observed_aj']].isna().all().all()
    assert max(errors)<1e-14
    result=dict(status='PASS',forest_fits=0,company_alignment=True,finite_financial_fields_before_origin=True,
        market_before_origin=True,training_stops_before_cutoff=True,unsupported_2025_H24_remains_NA=True,
        maximum_independent_pair_auc_error=max(errors),all_top20_counts_match=True,checks=checks)
    (run/'audit/final_numeric_validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='checks'}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);validate(p.parse_args().run)
