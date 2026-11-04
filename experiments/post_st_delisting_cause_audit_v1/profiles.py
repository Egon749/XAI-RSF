"""Publish coarse raw-function profiles and observed flag-state contrasts with minimum cell support."""
from .common import *
from .integrity import sources,G
from experiments.post_st_grouped_explanation_v1.common import layout
def main():
    frame=pd.read_parquet(run()/'linked_channels_PRIVATE.parquet');names=pd.read_csv(G/'FEATURE_GROUP_MAPPING.csv').english_name.tolist();result=[];states=[]
    for (y,h),ss in frame.groupby(['year','months']):
        tr,ev,lab=layout(sources(y)[0],y);assert (ss.row.to_numpy()==np.arange(len(ss))).all()
        target=(tr['event']==1)&(tr['time']<=tr[f'h{h}'])
        for v in ('log_price_last','log_total_mktcap_last','prior_audit_standard','prior_audit_going_concern'):
            j=names.index(v);x=tr['X'][:,j];xe=ev['X'][:,j]
            if v.startswith('log'):
                cuts=np.quantile(x[np.isfinite(x)],[1/3,2/3])
                locate=lambda z:np.where(np.isfinite(z),np.searchsorted(cuts,z),-1)
                bins=range(3)
            else:
                locate=lambda z:np.where(np.isfinite(z),z,-1)
                bins=[0,1]
            for i in bins:
                train=locate(x)==i;test=locate(xe)==i;obs=ss[test]
                good=train.sum()>=10 and (train&target).sum()>=3 and test.sum()>=5
                if good:result.append(dict(year=y,months=h,feature=v,bin=int(i),train_groups=int(train.sum()),train_target_events=int((train&target).sum()),evaluation_companies=len(obs),mean_raw_main_pp=obs['main_'+v].mean(),mean_allocated_pp=obs['alloc_'+v].mean()))
            if not v.startswith('log'):
                states.extend(dict(year=y,months=h,feature=v,flag=x,main=m,allocated=a) for x,m,a in zip(xe,ss['main_'+v],ss['alloc_'+v]))
    pd.DataFrame(result).to_csv(D/'SUPPORTED_RAW_PROFILES.csv',index=False)
    flag=pd.DataFrame(states);o=[]
    for k,g in flag.groupby(['months','feature']):
        for v,gg in g.groupby('flag'):
            if len(gg)>=5:o.append(dict(months=k[0],feature=k[1],state=v,n=len(gg),mean_main_pp=gg.main.mean(),mean_allocated_pp=gg.allocated.mean(),scope='supported evaluation years, all known statuses'))
    pd.DataFrame(o).to_csv(D/'AUDIT_FLAG_STATES.csv',index=False)
if __name__=='__main__':main()
