"""Assemble approved aggregate files and compute empirical map percentiles."""
from .common import *

def redact(frame,columns):
    for col in columns:
        frame[col]=frame[col].map(lambda v:'1–4' if 0<int(v)<5 else str(int(v)))
    return frame

def main():
    r=run();published=read(r/'analysis/published.json')
    def join(name):
        return pd.concat([pd.read_csv(Path(row['path'])/(name+'.csv')) for row in published if (Path(row['path'])/(name+'.csv')).is_file()],ignore_index=True)
    def save(name,df):
        df.to_csv(D/(name+'.csv'),index=False)
    metric=join('metrics')
    assert set(metric.map_id)==set(range(201)) and not metric.duplicated(['year','months','cause','map_id','level','shared','metric']).any()
    ran=metric[metric.map_id>0];assert len(ran)==200*3*4*4*3*7
    econ=metric[metric.map_id==0].copy()
    save('RANDOM_GROUP_BASELINE',ran.drop(columns=['firm_p05','firm_p95']))
    save('GROUP_STABILITY',econ)
    comp=[]
    for idx,values in ran.groupby(['year','months','cause','level','shared','metric']):
        key=dict(zip(['year','months','cause','level','shared','metric'],idx))
        z=econ
        for k,v in key.items():z=z[z[k]==v]
        assert len(z)==1
        a=z['mean'].iloc[0];v=values['mean'].dropna().to_numpy()
        comp.append(key|dict(economic=a,random_median=np.median(v) if len(v) else np.nan,random_p05=np.quantile(v,.05) if len(v) else np.nan,random_p95=np.quantile(v,.95) if len(v) else np.nan,valid_maps=len(v),
                             economic_percentile=100*((v<a).sum()+.5*(v==a).sum())/len(v) if len(v) and np.isfinite(a) else np.nan))
    comparison=pd.DataFrame(comp)
    for label in ['variable_full74','variable_main74','native_terms','financial_variables35']:
        ref=econ[econ.level==label].drop(columns='level')[['year','months','cause','shared','metric','mean']].rename(columns={'mean':label})
        comparison=comparison.merge(ref,on=['year','months','cause','shared','metric'],validate='many_to_one')
    save('GROUP_VS_FEATURE_COMPARISON',comparison)
    items=join('items');models=join('model_wise');support=join('support_PRIVATE')
    assert len(models)==132*3*4*(8+8+4+28)
    save('GROUP_MODEL_DISTRIBUTIONS',models)
    support=support.merge(pd.read_csv(r/'dominant_support_PRIVATE.csv'),on=['year','months','cause','group'],how='left')
    ii=support.level=='I28'
    np.testing.assert_array_equal(support.loc[ii,'any_consensus_firms'],support.loc[ii,'consensus_firms'])
    np.testing.assert_array_equal(support.loc[ii,'any_consensus_events'],support.loc[ii,'consensus_events'])
    support=support.drop(columns=['consensus_firms','consensus_events'])
    support['positive_supported_direction']=(support.consensus_positive_firms>=10)&(support.positive_events>=3)
    support['negative_supported_direction']=(support.consensus_negative_firms>=10)&(support.negative_events>=3)
    counts=['positive_events','negative_events','any_consensus_events','events_with_supported_dominant_pair']
    # Totals and directional or diagnostic subsets can be differenced. Suppress the
    # other positive totals as well when any of the linked counts is small.
    sensitive=((support[counts]>0)&(support[counts]<5)).any(axis=1)
    for column in counts:
        support[column]=['' if pd.isna(x) else '0' if x==0 else ('1–4' if x<5 else 'suppressed') if hide else str(int(x))
                         for x,hide in zip(support[column],sensitive)]
    details=items.merge(support,on=['year','months','cause','level','group'],validate='many_to_one')
    save('GROUP_MAIN_CONTRIBUTIONS',details[details.level.isin(['A8','C8'])])
    save('GROUP_INTERACTIONS',details[details.level=='I28'])
    save('FOUR_CAPABILITIES_STABILITY',details[details.level=='C4'])
    save('GROUP_CANCELLATION_AUDIT',join('cancel'))
    save('RANDOM_SLOT_AMPLITUDES',join('random_amplitudes'))
    save('LOCAL_CASE_BANDS',join('profiles'))
if __name__=='__main__':main()
