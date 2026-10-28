"""Run under python -I in a standalone directory containing no forest code."""
from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from crforest.transparent import TransparentCIF

def main():
    assert not (ROOT/'crforest/core.py').exists()
    assert 'crforest.core' not in sys.modules
    with np.load(ROOT/'inputs_PRIVATE.npz') as z:X=z['X'];h=z['horizon'];ids=z['sample_ids']
    if h.ndim==1:h=h[:,None]
    rows=[]
    for item in json.loads((ROOT/'index.json').read_text()):
        folder=ROOT/item['path'];m=TransparentCIF.load(folder)
        with np.load(folder/'expected_PRIVATE.npz') as z:
            expected=z['cif'];np.testing.assert_array_equal(z['sample_ids'],ids)
        pred=m.predict_horizons(X,h);error=float(np.max(abs(pred-expected)))
        assert error<1e-10 and pred.min()>-1e-10 and pred.sum(2).max()<=1+1e-10
        # Every time on the fitted grid, on deterministic evenly spaced records
        # plus an all-missing input; full-cohort check above covers actual horizons.
        anchors=np.r_[np.linspace(0,len(X)-1,64,dtype=int)]
        xx=np.vstack([X[anchors],np.full((1,X.shape[1]),np.nan)])
        allp=m.predict(xx);assert np.min(np.diff(allp,axis=1),initial=0)>-1e-10
        assert allp.min()>-1e-10 and allp.sum(2).max()<=1+1e-10
        ex=m.explain(xx,[90,180,365,366]);assert ex['max_error']<1e-10
        rows.append(dict(model=item['model'],seed=item['seed'],records=len(X),deployment_error=error,full_explanation_error=ex['max_error'],full_grid_records=len(xx),native_grid_times=len(m.expansion.grid),minimum_cif=float(pred.min()),maximum_total_cif=float(pred.sum(2).max()),forest_imported='crforest.core' in sys.modules))
    assert all(not r['forest_imported'] for r in rows)
    with (ROOT/'deployment_receipt.json').open('x') as f:json.dump(dict(status='PASS',runtime_files=['crforest/functions.py','crforest/transparent.py'],models=rows),f,indent=2)
if __name__=='__main__':main()
