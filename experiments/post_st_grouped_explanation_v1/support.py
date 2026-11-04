"""Supplement the prespecified empirical-row gate with support in actual raw-pair cells."""
from crforest import CIFExpansion
from .common import *

def run_checks():
    r=run();root=r/'analysis_input';records=[]
    for y in YEARS:
        tr,ev,labels=layout(root,y);n=len(tr['X']);X=tr['X'];assert len(np.unique(tr['groups']))==n
        keys=np.load(root/f'{y}_keys.npy');enc=bins(keys,mapping());values=np.load(root/f'{y}_terms.npy',mmap_mode='r')
        ex=CIFExpansion.load(r/f'input/{y}/ensemble320/functions_PRIVATE.npz')
        for pi,(g,h) in enumerate(PAIRS):
            allids=np.where(enc==8+pi)[0]
            ids=[j for j in allids if tuple(keys[j]) in ex.pairs]
            for u,months in enumerate([12,24]):
                events=tr['time']<=tr[f'h{months}']
                for k in (0,1):
                    target=events&(tr['event']==k+1);base=values[:,u,k,:n]
                    seed_sum=base[:10,:,allids].sum(-1)
                    v=(seed_sum[TRIPLES[:,0]]+seed_sum[TRIPLES[:,1]]+seed_sum[TRIPLES[:,2]])/3
                    consensus=((v>=.001).mean(0)>=.9)|((v<=-.001).mean(0)>=.9)
                    supported=np.zeros(n,bool)
                    if ids:
                        comp=base[11][:,ids];dominant=np.argmax(abs(comp),axis=1)
                        for ii,j in enumerate(ids):
                            key=tuple(keys[j]);bins_for_axes=[];dims=[]
                            for axis in key:
                                cuts=ex.cuts(key,axis);b=np.searchsorted(cuts,X[:,axis]);b[np.isnan(X[:,axis])]=len(cuts)+1
                                bins_for_axes.append(b);dims.append(len(cuts)+2)
                            cell=np.ravel_multi_index(tuple(bins_for_axes),tuple(dims))
                            counts=np.bincount(cell,minlength=np.prod(dims));event_counts=np.bincount(cell[target],minlength=np.prod(dims))
                            supported|=(dominant==ii)&(abs(comp[:,ii])>0)&(counts[cell]>=10)&(event_counts[cell]>=3)
                    records.append(dict(year=y,months=months,cause=k+1,group=INTER[pi],consensus_firms=int(consensus.sum()),consensus_events=int(target[consensus].sum()),firms_with_supported_dominant_pair=int((consensus&supported).sum()),events_with_supported_dominant_pair=int((consensus&target&supported).sum())))
    pd.DataFrame(records).to_csv(r/'dominant_support_PRIVATE.csv',index=False)
if __name__=='__main__':run_checks()
