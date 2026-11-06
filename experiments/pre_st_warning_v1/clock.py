"""Explicit annual landmark clocks; future outcomes cannot enter training."""
import numpy as np
import pandas as pd

def stop_at(origin, observed_end, event_date, event_type, cutoff=None, months=24):
    origin=pd.Timestamp(origin); end=min(pd.Timestamp(observed_end),origin+pd.DateOffset(months=months))
    if cutoff is not None:end=min(end,pd.Timestamp(cutoff)-pd.Timedelta(days=1))
    event_date=pd.Timestamp(event_date) if pd.notna(event_date) else pd.NaT
    event=0
    if pd.notna(event_date) and origin<event_date<=end:
        end=event_date;event=int(event_type)
    return max(0,(end-origin).days),event,end

def endpoint(time,event,horizon,cause):
    time,event,horizon=map(np.asarray,(time,event,horizon))
    happened=(event>0)&(time<=horizon)
    known=happened|(time>=horizon)
    target=happened&(event==cause)
    return known,target

def available_values(row,names,origin):
    v=row[names].to_numpy(float,copy=True)
    dates=np.full(len(names),np.datetime64(row.available,'D'),dtype='datetime64[D]')
    for j,n in enumerate(names):
        if n.endswith('_growth') or n in ['gross_margin_change','leverage_change']:
            if pd.isna(row.prev_available) or row.prev_available>=origin:v[j]=np.nan;dates[j]=np.datetime64('NaT')
            else:dates[j]=np.datetime64(max(row.available,row.prev_available),'D')
    v[~np.isfinite(v)]=np.nan
    assert (dates[np.isfinite(v)]<np.datetime64(origin,'D')).all()
    return v,dates
