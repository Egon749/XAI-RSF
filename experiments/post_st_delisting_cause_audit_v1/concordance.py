"""Additive allocation of within-year concordance differences to disjoint realized grounds."""
from .common import *
def main():
    df=pd.read_parquet(run()/'linked_channels_PRIVATE.parquet');res=[]
    for h,d in df.groupby('months'):
        for a,b in [('M2','M0'),('M3','M1')]:
            parts=[]
            for y,ss in d.groupby('year'):
                controls=ss[ss.category.isin(['recovered','event_free'])]
                events=ss[~ss.category.isin(['recovered','event_free'])]
                for _,row in events.iterrows():
                    values=[]
                    for arm in [a,b]:
                        cr=controls['CIF_'+arm].to_numpy();v=row['CIF_'+arm];values.append(np.sum(v>cr)+.5*np.sum(v==cr))
                    parts.append(dict(price='price' in row.grounds,n_pairs=len(controls),net=values[0]-values[1]))
            p=pd.DataFrame(parts);den=p.n_pairs.sum()
            for flag,gg in p.groupby('price'):
                assert len(gg)>=5
                res.append(dict(months=h,contrast=a+' minus '+b,category='price cited (including dual)' if flag else 'no price cited',event_companies=len(gg),event_control_pairs=int(gg.n_pairs.sum()),denominator_all_year_pairs=int(den),net_concordant_pairs=gg.net.sum(),contribution_to_total_AUC_difference=gg.net.sum()/den,all_event_difference=p.net.sum()/den))
    pd.DataFrame(res).to_csv(D/'CHANNEL_CONCORDANCE_ACCOUNTING.csv',index=False)
if __name__=='__main__':main()
