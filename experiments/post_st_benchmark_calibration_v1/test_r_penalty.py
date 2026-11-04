import os,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from .estimators import estimate

def test_independent_survival_finegray_expanded_risk_sets(tmp_path):
    r=os.environ.get('CRFOREST_RSCRIPT')
    if not r:pytest.skip('independent R environment not configured')
    rng=np.random.default_rng(8);n=48;x=rng.normal(size=(n,2));t=np.arange(1,n+1,dtype=float);e=np.tile([0,1,2,1,0,2],8)
    t[9]=t[7]  # tied failures at distinct covariates
    fit=[estimate(x,t,e,1,.1,fg) for fg in (True,False)]
    pd.DataFrame(dict(x1=x[:,0],x2=x[:,1],t=t,event=e)).to_csv(tmp_path/'data',index=False)
    np.savetxt(tmp_path/'beta',np.concatenate([m['beta'] for m in fit]))
    subprocess.run([r,'--vanilla',str(Path(__file__).with_name('check_partial_likelihood.R')),str(tmp_path/'data'),str(tmp_path/'beta'),str(tmp_path/'score')],check=True,capture_output=True)
    assert np.abs(np.loadtxt(tmp_path/'score')).max()<1e-6
