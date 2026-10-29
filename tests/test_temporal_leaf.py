import copy,json,os,subprocess
from pathlib import Path
import numpy as np
import pytest
from crforest import CompetingRiskForest,ForestConfig,CIFExpansion,aalen_johansen
from crforest.shrinkage import shrink_leaf_cifs
from crforest.temporal_leaf import weighted_aalen_johansen,temporal_leaf_cifs


def test_weighted_aj_integer_replication_and_rescaling():
    t=np.array([1.,1.,2.,3.,3.,4.]);e=np.array([1,2,0,1,0,2]);w=np.array([1.,3.,2.,4.,1.,2.]);grid=np.arange(6.)
    ix=np.repeat(np.arange(len(t)),w.astype(int))
    expected=aalen_johansen(t[ix],e[ix],grid)
    np.testing.assert_allclose(weighted_aalen_johansen(t,e,grid,w),expected,atol=1e-14,rtol=0)
    np.testing.assert_allclose(weighted_aalen_johansen(t,e,grid,w/7.3),expected,atol=1e-14,rtol=0)
    np.testing.assert_array_equal(weighted_aalen_johansen(t,e,grid,np.ones(6)*4),aalen_johansen(t,e,grid))
    np.testing.assert_allclose(expected[1],[1/13,3/13],atol=1e-14,rtol=0)


@pytest.mark.parametrize('w',[[0,1],[-1,1],[np.nan,1],[np.inf,1],[1]])
def test_weighted_aj_bad_weights(w):
    with pytest.raises(ValueError):weighted_aalen_johansen([1.,2.],[1,0],[1.,2.],w)


def test_weighted_forest_matches_unit_baseline_and_exact_full_functions(tmp_path):
    rng=np.random.default_rng(113);X=rng.normal(size=(95,5));X[::8,0]=np.nan
    t=rng.integers(1,61,len(X)).astype(float);e=rng.integers(0,3,len(X));g=np.arange(len(X))
    f=CompetingRiskForest(ForestConfig(n_trees=4,max_depth=4,path_budget=2,min_leaf=5,min_events=1,min_groups=3,
        mtry=5,n_thresholds=3,cut_strategy='shared_quantile',seed=42)).fit(X,t,e,g)
    before=copy.deepcopy(f.trees_);grid=np.r_[0.,f.grid_,100.]
    unit=temporal_leaf_cifs(f,X,t,e,g,np.ones(len(X)),10)
    standard=shrink_leaf_cifs(f,X,t,e,g,10)
    np.testing.assert_array_equal(unit.predict(X,grid),standard.predict(X,grid))
    z=temporal_leaf_cifs(f,X,t,e,g,np.where(g>50,4.,1.),10)
    assert before==f.trees_ and z.audit_==f.audit_
    p=z.predict(X,grid);assert p.min()>=0 and p.sum(-1).max()<=1+1e-12 and np.diff(p,axis=1).min()>=-1e-12
    ex=CIFExpansion.from_forest(z,X);terms=sum((ex.component(X,k,grid) for k in [(j,) for j in ex.mains]+ex.pairs),np.zeros_like(p))
    np.testing.assert_allclose(ex._coeff(grid)[1]+terms,p,atol=1e-12,rtol=0)
    z.save(tmp_path/'f.json');ex.save(tmp_path/'e.npz')
    np.testing.assert_array_equal(CompetingRiskForest.load(tmp_path/'f.json').predict(X,grid),p)
    np.testing.assert_allclose(CIFExpansion.load(tmp_path/'e.npz').predict(X,grid),p,atol=1e-12,rtol=0)
    with pytest.raises(ValueError):temporal_leaf_cifs(z,X,t,e,g,np.ones(len(X)),10)


def test_weighted_aj_independent_r_survival(tmp_path):
    exe=os.environ.get('CRFOREST_RSCRIPT')
    if not exe:pytest.skip('independent R runtime not configured')
    t=np.array([1.,1.,2.,3.,3.,4.]);e=np.array([1,2,0,1,0,2]);w=np.array([.7,3.,2.1,4.,1.,2.3]);grid=np.arange(6.)
    np.savetxt(tmp_path/'data.csv',np.column_stack([t,e,w]),delimiter=',',header='time,event,weight',comments='')
    script=tmp_path/'check.R';script.write_text('''args <- commandArgs(TRUE)
.libPaths(c(Sys.getenv("EW_RLIB"),.libPaths()))
library(survival)
d <- read.csv(args[1]); d$event <- factor(d$event, levels=c(0,1,2))
fit <- survfit(Surv(time,event)~1,data=d,weights=weight)
s <- summary(fit,times=0:5,extend=TRUE)
write.table(s$pstate[,c("1","2")],file=args[2],sep=",",row.names=FALSE,col.names=FALSE)
''')
    result=subprocess.run([exe,'--vanilla',str(script),str(tmp_path/'data.csv'),str(tmp_path/'result.csv')],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    np.testing.assert_allclose(weighted_aalen_johansen(t,e,grid,w),np.loadtxt(tmp_path/'result.csv',delimiter=','),atol=1e-12,rtol=0)
