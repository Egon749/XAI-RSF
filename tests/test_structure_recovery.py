from dataclasses import replace
from itertools import product
import numpy as np
import pytest
from crforest import CompetingRiskForest
from crforest.shrinkage import shrink_leaf_cifs
from experiments.post_st_structure_recovery_v1.common import *
from experiments.post_st_structure_recovery_v1.recovery import population_expansion,weights,assess
from experiments.post_st_structure_recovery_v1.worker import verify,shrink_audit

def planted():
    rep=7;j,k=sorted(signals(rep)[:2]);f=CompetingRiskForest(configuration(42,'q2'))
    f.config=replace(f.config,n_trees=1);f.grid_=TIMES.copy();f.n_features_=P;f.candidate_evaluations_=0
    nodes=[]
    def grow(level,values):
        i=len(nodes);node=dict(ancestors=sorted([j,k][:level]),depth=level);nodes.append(node)
        if level==2:
            x,y=values;p=.18+.04*x+.03*y+.05*x*y
            node['cif']=(time_scale(TIMES)[:,None]*np.array([p,.4])).tolist()
        else:
            node.update(feature=[j,k][level],threshold=0.,missing_left=False)
            node['left']=grow(level+1,values+[-1]);node['right']=grow(level+1,values+[1])
        return i
    grow(0,[]);f.trees_=[nodes];f.audit_=f.audit();return f,j,k

@pytest.mark.parametrize('scenario',list(SCENARIOS))
def test_dgp_legal_known_functions_and_censoring(scenario):
    s=signals(7);xx=np.zeros((128,P));xx[:,s]=list(product([-1.,1.],repeat=7))
    p=truth(xx,scenario,7,[0,365,730,1000]);assert np.all(p>=0) and np.all(p.sum(-1)<=1)
    assert np.all(np.diff(p,axis=1)>=0);np.testing.assert_array_equal(p[:,0],0)
    tr,test=generate(scenario,7,100);tr2,test2=generate(scenario,7,100)
    assert data_hash(tr)==data_hash(tr2);np.testing.assert_array_equal(test,test2)
    assert tr['X'].shape[1]==83 and (tr['time']>0).all() and (tr['time']<=730).all()
    assert set(tr['event'])<=set([0,1,2]) and ((tr['event']==0)&(tr['time']<730)).any()

@pytest.mark.parametrize('kind',['binary','uniform'])
def test_population_L2_against_independent_quadrature(kind):
    f,j,k=planted();X=np.zeros((16,P))
    if kind=='binary':z=np.array([-1.,1.]);pairs=list(product(z,repeat=2));X=X[:4]
    else:
        # Two-point Gaussian quadrature on EACH half of [-1,1].
        z=np.r_[-.5+np.array([-1,1])/(2*np.sqrt(3)),.5+np.array([-1,1])/(2*np.sqrt(3))]
        pairs=list(product(z,repeat=2))
    X[:,[j,k]]=pairs;ex=population_expansion(f,X,kind)
    total=np.broadcast_to(ex.mu,(len(X),2,2)).copy()
    for key in ex.keys:
        c=ex.component(X,key);_,_,_,norm,inner=weights(ex,key,kind)
        np.testing.assert_allclose(norm,np.mean(c*c,axis=0),atol=1e-15,rtol=0)
        np.testing.assert_allclose(inner,np.mean(c*np.prod(X[:,list(key)],axis=1)[:,None,None],axis=0),atol=1e-15,rtol=0)
        np.testing.assert_allclose(c.mean(0),0,atol=1e-15);total+=c
    np.testing.assert_allclose(total,f.predict(X,TIMES),atol=1e-14,rtol=0)
    _,_,_,norm,inner=weights(ex,(j,k),kind)
    np.testing.assert_allclose(norm[-1,0],.05**2,atol=1e-15)
    np.testing.assert_allclose(inner[-1,0],.05*(1 if kind=='binary' else .25),atol=1e-15)

def test_real_config_pairing_and_full_pipeline(tmp_path):
    assert config_differences()=={'path_budget':[2,None]}
    tr,test=generate('pair_small',10001,30);tr={k:v[:120] for k,v in tr.items()}
    a=replace(configuration(42,'q2'),n_trees=2);b=replace(configuration(42,'unrestricted'),n_trees=2)
    assert pairing(tr['groups'],a)==pairing(tr['groups'],b)
    for cfg in [a,b]:
        f=CompetingRiskForest(cfg).fit(tr['X'],tr['time'],tr['event'],tr['groups'])
        sh=shrink_leaf_cifs(f,tr['X'],tr['time'],tr['event'],tr['groups'],10);shrink_audit(f,sh)
        out=tmp_path/str(cfg.path_budget);out.mkdir();ex=population_expansion(sh,tr['X'],'binary')
        checks=verify(sh,np.vstack([tr['X'],test]),tr,ex,out)
        assert checks['explicit_sum']<1e-12
        records,signals_,summaries=assess(ex,'pair_small',10001)
        assert len(signals_)==8 and all(r['exact_product_risk_mse']>=0 for r in summaries)

def test_exact_total_recovery_error_includes_missing_true_triple():
    f,j,k=planted();X=np.zeros((128,P));X[:,signals(7)]=list(product([-1.,1.],repeat=7))
    ex=population_expansion(f,X,'binary');_,signal_rows,rows=assess(ex,'triple_large',7)
    exact=np.mean((f.predict(X,TIMES)-truth(X,'triple_large',7))**2,axis=0)
    for row in rows:
        h=list(TIMES).index(row['day'])
        np.testing.assert_allclose(row['exact_product_risk_mse'],exact[h,row['cause']-1],atol=1e-15,rtol=0)
    triples=[r for r in signal_rows if r['order']==3]
    assert len(triples)==2 and all(r['coefficient']==0 for r in triples)

def test_MC_summary_and_matched_differences():
    import pandas as pd
    from experiments.post_st_structure_recovery_v1.analyze import stats,paired,summarize
    assert stats([np.nan])['n']==0
    assert stats([1.,2.,3.])['mean']==2.
    rows=[]
    for arm,shift in [('q2',.1),('unrestricted',0.)]:
        for rep in range(5):rows.append(dict(scenario='test',rep=rep,arm=arm,variant='shrunk',seed='ensemble',day=730,cause=1,metric=rep+shift))
    df=paired(pd.DataFrame(rows).sample(frac=1,random_state=44),['metric'])
    assert len(df)==1 and df.n.iloc[0]==5
    np.testing.assert_allclose(df['mean'].iloc[0],.1,atol=1e-15)
    rates=summarize(pd.DataFrame({'group':['A']*50,'detected':[False]*50}),['group'],['detected'])
    assert rates.rate_wilson_low95.iloc[0]==0 and .07<rates.rate_wilson_high95.iloc[0]<.08
