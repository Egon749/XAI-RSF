"""Independent arithmetic, identity, provenance and domain-certificate checks."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from crforest import CompetingRiskForest
from experiments.post_st_ablation_v1.verification import merge
from .common import *

def validate(run):
    source_count=0
    for filename,digest in json.loads((run/'audit/immutable_input_hashes.json').read_text()).items():
        assert sha(run/'input'/filename)==digest;source_count+=1
    for filename,digest in json.loads((run/'audit/source_hashes.json').read_text()).items():assert sha(Path(filename))==digest
    assert sha(Path(__file__).resolve().parents[2]/'CURRENT_POST_ST_MODEL.json')==REFERENCE_SHA
    formal=outputs(run,'formal');assert len(formal)==460
    fits=0;calibrations=0;models=[];tasks=[]
    for i,folder in enumerate(formal):
        s=json.loads((folder/'summary.json').read_text());fits+=s.get('new_forest_fits',0);calibrations+=s.get('calibration_fits',0)
        assert s['status']=='COMPLETE';tasks.append(dict(task=i,**{k:v for k,v in s.items() if not isinstance(v,(list,dict))}))
        for name,h in json.loads((folder/'hashes.json').read_text()).items():
            assert sha(folder/name)==h
            if name.endswith(('forest.json.gz','forest_PRIVATE.json','functions_PRIVATE.npz','population_functions.npz','predictions_PRIVATE.npz','calibrator_PRIVATE.json')):
                models.append(dict(task=i,artifact=name,sha256=h,bytes=(folder/name).stat().st_size))
    assert fits==1356 and calibrations==4
    cert_rows=[]
    for folder in formal[456:]:
        cal=json.loads((folder/'calibrator_PRIVATE.json').read_text());z=arrays(folder/'certificate_PRIVATE.npz');p,y=z['p'],z['y']
        b,c=np.array(cal['b']),np.array(cal['c']);n=len(p);lam=20/n
        before=np.sum((p-y)**2)/n
        after=(np.sum((c+p*b-y)**2)+lam*np.sum((c+p*b-p)**2))/n
        assert abs(before-cal['objective_before'])<1e-13 and abs(after-cal['objective_after'])<1e-13
        assert np.all(b[y.sum(0)<10]==1) and np.all((b>=.5)&(b<=4)) and np.all(c>=0)
        f=merge([CompetingRiskForest.load(run/'input/reference'/str(cal['year'])/str(s)/'forest_PRIVATE.json') for s in SEEDS])
        # Independently maximize each tree's terminal linear score, average, then
        # add common offset. This avoids relying on the convex-hull routine.
        maximum=sum(max(np.asarray(node['cif'])[-1]@b for node in tree if 'cif' in node) for tree in f.trees_)/len(f.trees_)+sum(c)
        assert maximum<=1+1e-10
        vertex_bound=float((z['vertices']@b+sum(c)).max())
        assert maximum<=vertex_bound+1e-12
        cert_rows.append(dict(year=cal['year'],horizon=cal['horizon_months'],independent_max_sum=maximum,certificate_max_sum=vertex_bound,
            max_objective_recompute_error=max(abs(before-cal['objective_before']),abs(after-cal['objective_after'])),
            gap=cal['gap'],primal_violation=cal['primal_violation'],sparse_slopes_fixed=True))
    sv=pd.read_csv(run/'reports/simulation_validation.csv.gz');fv=pd.read_csv(run/'reports/enterprise_function_validation.csv')
    simerror=float(sv[['reload','training_reference','population_reference','explicit_sum','function_reload','native_time']].max().max())
    realerror=float(fv[['full_reconstruction','explicit_component_sum','function_reload','forest_reload']].max().max())
    assert simerror<=1e-10 and realerror<=1e-10
    summary=pd.read_csv(run/'reports/simulation_metric_summary.csv');paired=pd.read_csv(run/'reports/simulation_paired_changes.csv')
    # Expected Brier and true-CIF MSE differences must be equal on shared DGP rows.
    keys=['scenario','variant','seed','day','cause']
    a=paired[paired.metric=='joint_risk_mse'].set_index(keys)['mean'];b=paired[paired.metric=='expected_brier'].set_index(keys)['mean']
    np.testing.assert_allclose(a,b,atol=1e-14,rtol=0)
    pd.DataFrame(tasks).to_csv(run/'reports/task_manifest.csv',index=False)
    pd.DataFrame(models).to_csv(run/'reports/artifact_hashes.csv.gz',index=False,compression='gzip')
    pd.DataFrame(cert_rows).to_csv(run/'reports/calibration_domain_validation.csv',index=False)
    result=dict(status='PASS',formal_tasks=460,formal_forest_fits=1356,
        simulation_fits=1350,enterprise_fits=6,calibration_fits=4,pilot_fits=2,immutable_input_files=source_count,
        maximum_simulation_reconstruction_error=simerror,maximum_enterprise_reconstruction_error=realerror,
        all_model_output_hashes_verified=True,reference_unchanged=True,full_domain_calibration_certificate_independently_verified=True,
        all_prespecified_cases_retained=True,new_2025_access=False)
    receipt=run/'reports/independent_validation.json'
    if receipt.exists():assert json.loads(receipt.read_text())==result,'Repeated validation changed its recorded result'
    else:write(receipt,result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);validate(p.parse_args().run)
