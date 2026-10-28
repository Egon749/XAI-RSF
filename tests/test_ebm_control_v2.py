"""API-contract tests with an explicit fake; not real InterpretML validation."""
import sys
import types
import numpy as np
import pytest
from experiments.transparent_cif_v2.ebm_control import fit_ebm_control


def test_external_company_holdout_selects_rounds_without_refit(monkeypatch):
    calls=[]
    class FakeEBM:
        def __init__(self,**kwargs):self.options=kwargs;calls.append(self)
        def fit(self,X,y,sample_weight):
            self.training_X=X.copy();self.training_y=y.copy();return self
        def predict_proba(self,X):
            p=np.where(X[:,0]>0,.8,.2) if self.options['max_rounds']==200 else np.full(len(X),.5)
            return np.column_stack([1-p,p])
    fake=types.ModuleType('interpret.glassbox');fake.ExplainableBoostingClassifier=FakeEBM
    monkeypatch.setitem(sys.modules,'interpret',types.ModuleType('interpret'))
    monkeypatch.setitem(sys.modules,'interpret.glassbox',fake)
    X=np.array([[-1.],[1.],[-1.],[1.]])
    model,receipt=fit_ebm_control(X,[0,1,0,1],[1,1,0,1],['a','b','c','d'],X,[0,1,0,1],
                                 np.ones(4),['e','f','g','h'])
    assert receipt['selected_rounds']==200 and receipt['refit_on_holdout'] is False
    assert len(calls)==3 and all(len(m.training_y)==3 for m in calls)
    assert all(m.options['validation_size']==0 for m in calls)
    with pytest.raises(ValueError,match='disjoint'):
        fit_ebm_control(X,[0,1,0,1],np.ones(4),['a','b','c','d'],X,[0,1,0,1],np.ones(4),['a','f','g','h'])
