"""No-fit checks of this completed real run, independent of displayed conclusions."""
import os,json
from pathlib import Path
import numpy as np,pandas as pd
R=Path(os.environ['CIF_L1_RUN'])
def test_all_weights_converged_and_simplex():
 d=pd.read_csv(R/'reports/weight_diagnostics.csv');assert len(d)==9 and d.status.eq('CONVERGED').all();assert d.frank_wolfe_gap.max()<=1e-8;assert d.minimum_weight.min()>=0 and d.constraint_residual.max()<1e-12

def test_native_all_models_full_input_all_terms():
 d=pd.read_csv(R/'reports/native_complexity.csv');assert len(d)==24 and d.full_cohort_records.eq(23799).all();assert d.weighted_tree_max_error.max()<1e-10 and d.explicit_all_function_max_error.max()<1e-10 and d.reload_error.max()==0
 assert d.minimum_cif.min()>=-1e-10 and d.maximum_cause_sum.max()<=1+1e-10 and d.minimum_time_increment.min()>=-1e-10
 assert d.missing_cells.gt(0).all()
 for name,g in d.groupby('model'):assert g[g.seed==-1].spatial_cells.iloc[0]>=g[g.seed!=-1].spatial_cells.max()

def test_bootstrap_complete_and_point_agreement():
 d=json.loads((R/'audit/bootstrap_check.json').read_text());assert d['replicates']==1000 and d['all_ids_unique'] and d['point_max_error']<1e-10 and d['fits']==0
 contrasts=pd.read_csv(R/'reports/paired_differences.csv');assert contrasts.valid_replicates.eq(1000).all();assert len(contrasts.contrast.unique())==7

def test_reference_inclusion_and_unknown_workload():
 d=pd.read_csv(R/'reports/metrics_by_year_seed.csv');assert len(d.model.unique())==11 and set(d.records)=={11686,12113,23799}
 w=pd.read_csv(R/'reports/warning_utility.csv');np.testing.assert_array_equal(w.alert_records,w.hit_records+w.known_no_target_alerts+w.unknown_outcome_alerts)
 assert w[(w.scope=='pooled')&(w.top_fraction==.2)].alert_records.nunique()==1
 assert w[(w.scope=='pooled')].unknown_outcome_alerts.gt(0).all()

def test_independent_runtime_and_roles():
 x=json.loads((R/'standalone_deployment/deployment_receipt.json').read_text());assert len(x['models'])==24 and all(not m['forest_imported'] for m in x['models'])
 d=pd.read_csv(R/'reports/role_support.csv');assert d[d.role=='structure'].groups.iloc[0]==2061 and d[d.role=='weight_holdout'].groups.iloc[0]==516
