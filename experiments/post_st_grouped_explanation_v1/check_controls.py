"""Independent real-row check of the first and last fixed negative controls."""
import pandas as pd
from scipy.stats import rankdata
from .common import *
def main():
    r=run();root=r/'analysis_input';maps=pd.read_csv(D/'RANDOM_MAPS.csv',index_col=0)
    published=pd.read_csv(D/'RANDOM_GROUP_BASELINE.csv').query("months==12 and cause==1 and shared==0 and level=='C8'")
    rows=[];sel=SHARED==0
    for y in YEARS:
        tr,ev,lab=layout(root,y);keys=np.load(root/f'{y}_keys.npy')
        x=np.load(root/f'{y}_terms.npy',mmap_mode='r')[:10,0,0,len(tr['X']):]
        var=np.zeros((*x.shape[:2],74))
        for d,(j,k) in enumerate(keys):
            if k<0:var[...,j]+=x[...,d]
            else:var[...,j]+=.5*x[...,d];var[...,k]+=.5*x[...,d]
        for mid in [1,200]:
            codes=maps.loc[mid].to_numpy()-1
            members=np.stack([var[...,codes==g].sum(-1) for g in range(8)],axis=-1)
            triples=np.stack([members[ids].mean(0) for ids in TRIPLES])
            a,b=(triples[idx[sel]].transpose(1,0,2) for idx in (LEFT,RIGHT))
            d=abs(a-b).sum(-1);den=abs(a).sum(-1)+abs(b).sum(-1)
            def ratio(x,n):return np.divide(x,n,out=np.full(np.shape(x),np.nan),where=n!=0)
            ar=rankdata(abs(a),axis=-1);br=rankdata(abs(b),axis=-1)
            ar-=ar.mean(-1,keepdims=True);br-=br.mean(-1,keepdims=True)
            rv=ratio((ar*br).sum(-1),np.linalg.norm(ar,axis=-1)*np.linalg.norm(br,axis=-1))
            mat=(abs(a)>=.001)&(abs(b)>=.001)
            for key,ans in [('relative_l1',ratio(d,den)),('rank_rho',rv),('sign_agreement',ratio((mat&(np.sign(a)==np.sign(b))).sum(-1),mat.sum(-1)))]:
                value=np.nanmean(np.nanmean(ans,axis=1))
                saved=published.query('year==@y and map_id==@mid and metric==@key')['mean'].item()
                diff=float(abs(value-saved));assert diff<1e-10
                rows.append(dict(year=y,map_id=mid,metric=key,error=diff))
    dump(D/'CONTROL_AUDIT.json',dict(method='Independent symmetric raw-variable formula and scipy ranks on real evaluation firms; first and last preregistered maps, all 2100 disjoint pairs, 12m termination',checks=rows,max_error=max(x['error'] for x in rows)))
if __name__=='__main__':main()
