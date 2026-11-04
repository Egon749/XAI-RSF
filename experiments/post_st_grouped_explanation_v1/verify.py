"""Check each saved shrunk predictor on both causes and its entire native grid."""
import time
from crforest import CompetingRiskForest,CIFExpansion
from .common import *

def worker(year,first,last,out,root):
    tr,ev,labels=layout(root,year);X=np.vstack((tr['X'],ev['X']))
    models={};report=[];groups=mapping()
    for label in (ARMS+COMB)[first:last]:
        triple=label in COMB
        ids=[str(SEEDS[j]) for j in TRIPLES[COMB.index(label)]] if triple else [label]
        for arm in ids:
            if arm not in models:
                m=CompetingRiskForest.load(root/str(year)/arm/'forest_PRIVATE.json')
                assert m.audit()['max_path_variables']<=2
                t=np.unique(np.r_[0,m.grid_,365,366,730,731]);pred=m.predict(X,t)
                q=arrays(root/str(year)/arm/'predictions_PRIVATE.npz')
                np.testing.assert_array_equal(ev['sample_ids'],q['sample_ids'])
                np.testing.assert_allclose(q['cif'],pred[len(tr['X']):,np.searchsorted(t,q['times'])],rtol=0,atol=1e-10)
                models[arm]=(m.grid_,t,pred)
        grid,t,_=models[ids[0]]
        orig=np.mean([models[i][2] for i in ids],axis=0)
        path=root/str(year)/(label+'.npz' if triple else label+'/functions_PRIVATE.npz')
        ex=CIFExpansion.load(path);np.testing.assert_array_equal(ex.grid,grid);assert ex.n_features==74 and ex.reference_n==len(tr['X'])
        keys=[(j,-1) for j in ex.mains]+list(ex.pairs);gi=bins(keys,groups)
        checks={n:0. for n in ['saved_function','original_sum','A_plus_I','sum_C']}
        hi=np.stack([np.r_[tr[f'h{h}'],labels[f'h{h}']] for h in [12,24]])
        assert np.isin(hi,t).all()
        cache=np.empty((2,2,len(X),len(keys))) if not triple else None
        baseline=ex._coeff(t)[1]
        # Limit working set to 48 rows across 36 categories and both event curves.
        for start in range(0,len(X),48):
            xx=X[start:start+48];pr=orig[start:start+len(xx)];native=np.zeros_like(pr)
            ai=np.zeros((*pr.shape,36))
            for j,(k,l) in enumerate(keys):
                term=ex.component(xx,(k,) if l<0 else (k,l),t)
                native+=term;ai[...,gi[j]]+=term
                if cache is not None:
                    for u in range(2):
                        loc=np.searchsorted(t,hi[u,start:start+len(xx)])
                        cache[u,:,start:start+len(xx),j]=term[np.arange(len(xx)),loc].T
            for kind,vals in [('saved_function',ex.predict(xx,t)),('original_sum',baseline+native),('A_plus_I',baseline+ai.sum(-1)),('sum_C',baseline+allocate(ai).sum(-1))]:
                checks[kind]=max(checks[kind],float(abs(pr-vals).max()))
        lo,inc,hi_sum=float(orig.min()),float(np.diff(orig,axis=1).min()),float(orig.sum(-1).max())
        assert max(checks.values())<=1e-10 and lo>=-1e-12 and inc>=-1e-12 and hi_sum<=1+1e-10
        if cache is not None:
            np.savez_compressed(out/f'{label}_PRIVATE.npz',terms=cache,keys=np.array(keys),baseline=baseline[np.searchsorted(t,hi)].transpose(0,2,1))
        report.append(dict(year=year,model=label,rows=len(X),grid_points=len(t),native_times=len(grid),causes=2,**checks,cif_min=lo,min_increment=inc,max_cause_sum=hi_sum))
    pd.DataFrame(report).to_csv(out/'exact.csv',index=False)
