"""Independently check reference identity, representative merged exports and support."""
import pandas as pd
from crforest import CompetingRiskForest,CIFExpansion
from experiments.post_st_ablation_v1.verification import merge
from .common import *

def run():
    r=run_path();root=r/'input';registry=read(source_path()/'input/old_registry.json')
    outputs={(z['args'][0],z['args'][1],k):Path(z['path']) for z in read(r/'verify/published.json') for k in range(int(z['args'][2]),int(z['args'][3]))}
    checked=[];max_explicit=0.
    for y in YEARS:
        tr=arrays(root/f'folds/{y}/train_PRIVATE.npz');ev=arrays(root/f'folds/{y}/eval_PRIVATE.npz');X=np.vstack([tr['X'],ev['X']])
        assert len(set(tr['groups'])&set(arrays(root/f'evaluation_input/{y}_test_PRIVATE.npz')['groups']))==0
        for v in VERSIONS:
            reference=CompetingRiskForest.load(root/str(y)/'reference96'/v/'forest_PRIVATE.json')
            members=[CompetingRiskForest.load(Path(registry[f'{y}|{s}'])/v/'forest_PRIVATE.json') for s in (42,101,202)]
            assert [m.config.seed for m in members]==[42,101,202]
            assert merge(members).trees_==reference.trees_
            for i in [0,59,119]:
                key=NAMES[i]; ex=CIFExpansion.load(outputs[str(y),v,i]/(key+'_PRIVATE.npz'))
                times=np.unique(np.r_[0.,ex.grid,TIMES])
                p=ex.predict(X,times);total=np.broadcast_to(ex._coeff(times)[1],p.shape).copy()
                for term in [(j,) for j in ex.mains]+ex.pairs:total+=ex.component(X,term,times)
                err=float(abs(total-p).max());max_explicit=max(max_explicit,err);assert err<1e-10
                checked.append(dict(year=y,version=v,combo=key,max_error=err))
    for rec in read(r/'explain/published.json'):
        y=int(rec['args'][0]);tr=arrays(root/f'folds/{y}/train_PRIVATE.npz');p=Path(rec['path']); raw=pd.read_csv(p/'cells_PRIVATE.csv')
        for (day,c),g in raw.groupby(['day','cause']):
            counts=g.groupby('term')[['companies','events']].sum()
            assert (counts.companies==len(tr['groups'])).all()
            assert (counts.events==np.sum((tr['event']==c)&(tr['time']<=day))).all()
    dump(HERE/'INDEPENDENT_AUDIT.json',dict(reference_components_verified=[42,101,202],reference_tree_identity_all_years_and_versions=True,
           full_merged_term_sum_spot_checks=checked,max_error=max_explicit,all_joint_support_tables_reconcile=True,disjoint_company_year_train_eval=True))
if __name__=='__main__':run()
