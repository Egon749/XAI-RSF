"""Independent saved-output and aggregation checks; never fit a model."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from .common import *

def validate(run,repo):
    out=run/'reports';receipt=json.loads((out/'execution_summary.json').read_text())
    assert receipt['formal_fits']==2700 and receipt['tasks']==450 and receipt['pilot_fits']==4
    assert set(receipt['by_host'])=={'oliverdesktop','lemon'} and not receipt['failed_formal_tasks']
    assert sha(repo/'CURRENT_POST_ST_MODEL.json')==json.loads((run/'input/protocol.json').read_text())['current_reference_sha256']
    checked=0
    for name,digest in json.loads((run/'audit/immutable_input_hashes.json').read_text()).items():
        assert sha(run/'input'/name)==digest,name;checked+=1
    ident=pd.read_csv(out/'data_identity.csv')
    for row in ident.itertuples():
        train,test=generate(row.scenario,row.rep)
        assert data_hash(train)==row.train_sha256 and data_hash({'X':test})==row.test_sha256
    keys=['scenario','arm','variant','seed','day','cause']
    values=pd.read_csv(out/'metrics_per_repetition.csv.gz',dtype={'seed':str});summary=pd.read_csv(out/'metric_summary.csv',dtype={'seed':str})
    assert values.groupby(keys).size().eq(50).all() and not values.duplicated(keys+['rep']).any()
    indexed=values.set_index(keys).sort_index();maximum=0.
    for row in summary.itertuples():
        key=tuple(getattr(row,k) for k in keys);data=np.asarray(indexed.loc[key,row.metric],float);data=data[np.isfinite(data)]
        assert len(data)==row.n
        if len(data):maximum=max(maximum,abs(data.mean()-row.mean))
    assert maximum<1e-10
    pairkeys=[k for k in keys if k!='arm']+['rep']
    q=values[values.arm=='q2'].set_index(pairkeys);u=values[values.arm=='unrestricted'].set_index(pairkeys)
    np.testing.assert_allclose(q.expected_brier-u.expected_brier,q.joint_risk_mse-u.joint_risk_mse,atol=1e-13,rtol=0)
    models=pd.read_csv(out/'models_per_repetition.csv.gz',dtype={'seed':str})
    assert models[models.arm=='q2'].max_path_variables.le(2).all() and models.max_path_variables.le(4).all()
    structure=['spatial_cells_including_missing']+[f'order{i}_structural_functions' for i in range(1,5)]
    mk=['scenario','rep','arm','seed'];raw=models[models.variant=='raw'].set_index(mk)[structure].sort_index();shr=models[models.variant=='shrunk'].set_index(mk)[structure].sort_index()
    pd.testing.assert_frame_equal(raw,shr)
    v=pd.read_csv(out/'validation.csv.gz');errcols=['reload','training_reference','population_reference','explicit_sum','function_reload','native_time']
    assert v[errcols].max().max()<=1e-10 and v.reload.eq(0).all() and v.function_reload.eq(0).all()
    assert v.cif_minimum.min()>=-1e-12 and v.cif_sum_maximum.max()<=1+1e-12 and v.increment_minimum.min()>=-1e-12
    report=dict(status='PASS',immutable_files_rehashed=checked,independently_regenerated_datasets=len(ident),predictors_checked=len(v),
        formal_fits=2700,pilot_fits=4,max_aggregate_roundtrip_error=float(maximum),max_reconstruction_error=float(v[errcols].max().max()),
        all_scenarios_and_seeds_preserved=True,current_reference_unchanged=True,source_hashes_unchanged=True,
        expected_brier_difference_equals_true_risk_MSE_difference=True,shrinkage_preserves_structural_representation=True)
    write(out/'independent_validation.json',report);print(json.dumps(report))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[2]);a=p.parse_args();validate(a.run,a.repo)
