"""Check compact grouped private endpoints against saved models and independent grouping."""
from crforest import CIFExpansion
from .common import *
from .grouped import grouped_components

def check():
    r=run();root=r/'input';worst=0.;group_error=0.;checked=0
    for rec in read(r/'analysis/published.json'):
        if rec['args'][2]!='0':continue
        y=int(rec['args'][1]);p=Path(rec['path']);tr,ev,lab=layout(root,y)
        records={(u,k):arrays(p/f'{h}_{k}_grouped_PRIVATE.npz') for u,h in enumerate([12,24]) for k in [1,2]}
        for idx,model in enumerate(ARMS+COMB):
            path=root/str(y)/(model+'.npz' if model in COMB else model+'/functions_PRIVATE.npz')
            ex=CIFExpansion.load(path);days=np.array([365,366,730,731]);pred=ex.predict(ev['X'],days)
            if model in ('reference96','ensemble320',COMB[0],COMB[59],COMB[119]):
                base,a,i,c=grouped_components(ex,ev['X'],days,mapping())
            for u,h in enumerate([12,24]):
                loc=np.searchsorted(days,lab[f'h{h}'])
                for k in (1,2):
                    block=records[u,k];mu=block['mu'][idx,len(tr['X']):]
                    get=lambda key:block[key][idx,len(tr['X']):]
                    actual=pred[np.arange(len(loc)),loc,k-1]
                    worst=max(worst,float(np.max(abs(actual-mu-get('C').sum(-1)))),float(np.max(abs(actual-mu-get('A').sum(-1)-get('I').sum(-1)))))
                    if model in ('reference96','ensemble320',COMB[0],COMB[59],COMB[119]):
                        for vals,key in [(a,'A'),(i,'I'),(c,'C')]:
                            err=vals[np.arange(len(loc)),loc,k-1]-get(key)
                            group_error=max(group_error,float(abs(err).max()))
            checked+=1
    assert checked==396 and max(worst,group_error)<=1e-10
    dump(D/'INDEPENDENT_ENDPOINT_AUDIT.json',dict(all_eval_models=checked,two_causes_two_calendar_endpoints=True,max_forest_to_saved_group_error=worst,independent_individual_blocks_models=15,max_individual_block_error=group_error))
if __name__=='__main__':check()
