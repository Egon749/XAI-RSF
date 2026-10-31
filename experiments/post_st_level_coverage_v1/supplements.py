"""Post-result descriptive tables from saved artifacts only; no fitting/selection."""
from pathlib import Path
from dataclasses import asdict
import argparse,hashlib,json
import numpy as np
import pandas as pd
from crforest import CompetingRiskForest,CIFExpansion
from .common import *

def typed_array_hash(value):
    a=np.ascontiguousarray(value)
    if a.dtype.hasobject:raise ValueError('Object arrays are not a stable serialization')
    h=hashlib.sha256();h.update(a.dtype.str.encode());h.update(json.dumps(a.shape).encode());h.update(a.tobytes());return h.hexdigest()

def supplement(run):
    out=run/'reports';calrows=[];files=[];support=[];cost=[];geometry=[]
    for folder in outputs(run,'formal')[456:]:
        cal=json.loads((folder/'calibrator_PRIVATE.json').read_text());s=json.loads((folder/'summary.json').read_text())
        z=arrays(folder/'certificate_PRIVATE.npz');p,y=z['p'],z['y'];q=p*np.asarray(cal['b'])+cal['c']
        for j,cause in enumerate(['termination','recovery']):
            calrows.append(dict(year=cal['year'],horizon_months=cal['horizon_months'],cause=cause,n=len(y),events=int(y[:,j].sum()),
                observed=float(y[:,j].mean()),original_mean=float(p[:,j].mean()),calibrated_mean=float(q[:,j].mean()),
                original_brier=float(np.mean((p[:,j]-y[:,j])**2)),calibrated_brier=float(np.mean((q[:,j]-y[:,j])**2)),
                slope_fixed=cal['frozen_slope_for_sparse_cause'][j],slope_direction=int(np.sign(cal['b'][j]-1)),positive_offset=cal['c'][j]>1e-12,
                lower_bound_active=abs(cal['b'][j]-.5)<1e-10,upper_bound_active=abs(cal['b'][j]-4)<1e-10,
                sum_constraint_active=abs(cal['maximum_terminal_sum_bound']-1)<1e-10,iterations=cal['iterations'],gap=cal['gap'],
                primal_violation=cal['primal_violation'],solver_seconds=s['solver_seconds'],peak_rss_MiB=s['peak_rss_MiB']))
    for y in YEARS:
        tr=arrays(run/'input/folds'/str(y)/'train_PRIVATE.npz');ev=arrays(run/'evaluation_input'/f'{y}_test_PRIVATE.npz')
        for role,z in [('train',tr),('test',ev)]:
            for k in ['X','sample_ids','groups','time','analysis_time','event','h12','h24']:
                if k in z:files.append(dict(year=y,role=role,array=k,rows=len(z[k]),columns=z[k].shape[1] if z[k].ndim>1 else 1,sha256=typed_array_hash(z[k])))
    pd.DataFrame(files).to_csv(out/'data_array_hashes.csv',index=False)
    rolling=[]
    for p in sorted((run/'input/rolling').glob('*/*/forest_PRIVATE.json')):
        f=CompetingRiskForest.load(p);assert asdict(f.config)==asdict(configuration(f.config.seed,'reference'))
        assert all(n['leaf_shrinkage']['strength']==10 for t in f.trees_ for n in t if 'cif' in n)
        rolling.append(dict(year=int(p.parent.parent.name),seed=f.config.seed,configuration_matches_reference=True,shrinkage_matches=True,forest_sha256=sha(p)))
    pd.DataFrame(rolling).to_csv(out/'rolling_config_validation.csv',index=False)
    # Aggregates disclose no variable indices, company rows or fitted coefficients.
    for y in YEARS:
        for seed in SEEDS:
            f=CompetingRiskForest.load(run/'input/reference'/str(y)/str(seed)/'forest_PRIVATE.json')
            for arm,model in [('reference',f)]+[(a,CompetingRiskForest.load(folder/v/'forest_PRIVATE.json')) for folder in outputs(run,'formal')[450:456]
                if (lambda s:s['year']==y and s['seed']==seed)(json.loads((folder/'summary.json').read_text())) for v,a in [('raw','wide83_raw'),('shrunk','wide83')]]:
                leaves=[n for t in model.trees_ for n in t if 'cif' in n];ns=np.array([n['groups'] for n in leaves]);events=np.array([n['cause_company_support'] for n in leaves]);zero=events[:,0]==0
                support.append(dict(year=y,seed=seed,arm=arm,leaves=len(leaves),unique_companies_min=int(ns.min()),unique_companies_median=float(np.median(ns)),
                    target_companies_median=float(np.median(events[:,0])),recovery_companies_median=float(np.median(events[:,1])),
                    zero_target_leaves=int(zero.sum()),zero_target_fraction=float(zero.mean()),
                    zero_target_borrow_positive=int(sum(z and np.asarray(n['cif'])[:,0].max()>0 for z,n in zip(zero,leaves)))))
    for folder in outputs(run,'formal')[450:456]:
        s=json.loads((folder/'summary.json').read_text());node=pd.read_csv(folder/'node_search_PRIVATE.csv');feature=pd.read_csv(folder/'feature_coverage_PRIVATE.csv');pair=pd.read_csv(folder/'pair_coverage_PRIVATE.csv')
        cost.append({k:s[k] for k in ['year','seed','host','fit_seconds','shrink_seconds','seconds','cpu_seconds','peak_rss_MiB','actual_scored_candidates'] }|
            {k:int(node[k].sum()) for k in ['features_considered','finite_cuts','missing_only_cuts','routed_attempts','scored_candidates','features_with_fewer_than_five_cuts'] }|
            dict(eligible_min=int(feature.eligible_trees.min()),eligible_max=int(feature.eligible_trees.max()),used_variables=int((feature.split_trees>0).sum()),
                eligible_pairs=int((pair.eligible_trees>0).sum()),used_pairs=int((pair.co_path_trees>0).sum())))
    # Scalar storage counts follow the actual exported grid, also for affine maps.
    modeldirs=[]
    for p in sorted((run/'private/ensemble_verification').iterdir()):
        if not p.is_dir():continue
        bits=p.name.split('_');year=int(bits[0]);seed='ensemble'
        if bits[-1].isdigit():seed=bits.pop()
        arm='_'.join(bits[1:]);modeldirs.append((year,arm,seed,0,p))
    for folder in outputs(run,'formal')[450:456]:
        s=json.loads((folder/'summary.json').read_text())
        for v,a in [('raw','wide83_raw'),('shrunk','wide83')]:modeldirs.append((s['year'],a,str(s['seed']),0,folder/v))
    for folder in outputs(run,'formal')[456:]:
        s=json.loads((folder/'summary.json').read_text());modeldirs.append((s['year'],'affine','ensemble',s['horizon'],folder/'calibrated'))
    for y,a,seed,h,p in modeldirs:
        c=json.loads((p/'complexity_PRIVATE.json').read_text());ex=CIFExpansion.load(p/'functions_PRIVATE.npz');n=len(ex.grid)
        geometry.append(dict(year=y,arm=a,seed=seed,horizon=h,exported_grid_size=n,structural_main_functions=sum(r['order']==1 for r in c['functions']),
            structural_pair_functions=sum(r['order']==2 for r in c['functions']),nonzero_main_functions=c['main_functions'],nonzero_pair_functions=c['interaction_functions'],
            spatial_cells=c['spatial_table_cells'],scalar_function_cells=c['spatial_table_cells']*n*2,baseline_scalars=n*2,
            factored_coefficients=ex.coefficients.size,function_bytes=(p/'functions_PRIVATE.npz').stat().st_size,sha256=sha(p/'functions_PRIVATE.npz')))
    pd.DataFrame(calrows).to_csv(out/'calibration_historical_diagnostics.csv',index=False)
    pd.DataFrame(support).to_csv(out/'enterprise_leaf_support.csv',index=False)
    pd.DataFrame(cost).to_csv(out/'enterprise_search_cost.csv',index=False)
    pd.DataFrame(geometry).to_csv(out/'enterprise_export_geometry.csv',index=False)
    # Exact weighted decomposition; product of unweighted means is NOT an identity.
    d=pd.read_csv(out/'simulation_tree_signal.csv.gz');rows=[]
    for keys,g in d.groupby(['scenario','arm','variant','day','order','role']):
        n=g.tree_count.sum();eligible=g.eligible_trees.sum();nz=g.nonzero_trees.sum();amplitude=float((g.amplitude_ratio*g.tree_count).sum()/n)
        weighted=amplitude*n/nz if nz else np.nan
        rows.append(dict(zip(['scenario','arm','variant','day','order','role'],keys))|
            dict(eligible_fraction=eligible/n,nonzero_given_eligible=nz/eligible if eligible else np.nan,nonzero_weighted_amplitude=weighted,
                full_amplitude=amplitude,identity_error=abs(eligible/n*nz/eligible*weighted-amplitude) if nz and eligible else 0.))
    pd.DataFrame(rows).to_csv(out/'simulation_tree_weighted_identity.csv',index=False)
    # Record fixed-input prediction timing separately from export/validation wall time.
    # Native timings and counts are already in the immutable worker summaries.
    write(out/'supplement_scope.json',dict(new_fits=0,new_hyperparameters=0,post_result_descriptive=True,
        purposes=['actual export grid/storage counts','historical calibration diagnostics','support and search aggregates','weighted tree identity'],
        parameters_and_enterprise_details_public=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);supplement(p.parse_args().run)
