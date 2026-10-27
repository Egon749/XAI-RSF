from pathlib import Path
import argparse,json,hashlib,shutil,sys
import numpy as np
import pandas as pd
from experiments.interpretable_baselines_v1.methods import resample_groups,group_reference,SCENARIOS


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    with Path(p).open('x') as f:json.dump(d,f,indent=2,ensure_ascii=False,allow_nan=False)
def main(run,parent,order):
    inp=run/'input';assert not (inp/'plan.json').exists();sources={}
    def copy(src,dst):
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);assert sha(src)==sha(dst);sources[str(src)]=sha(src)
    for name in ['final_train_PRIVATE.npz','final_eval_PRIVATE.npz','final_train_metadata_PRIVATE.parquet','final_eval_metadata_PRIVATE.parquet','M2_PRIVATE.npz']:
        copy(parent/'input'/name,inp/name)
    for p in (parent/'input/legacy').glob('*.py'):copy(p,inp/'legacy'/p.name)
    sys.path.insert(0,str(inp/'legacy'));from common import endpoint_weights
    from endpoint_guard import source_consistent_endpoint_weights
    tr=dict(np.load(inp/'final_train_PRIVATE.npz',allow_pickle=False));ev=dict(np.load(inp/'final_eval_PRIVATE.npz',allow_pickle=False));meta=pd.read_parquet(inp/'final_train_metadata_PRIVATE.parquet')
    np.testing.assert_array_equal(tr['sample_ids'],meta.sample_id)
    assert (meta.outcome_available.dropna()<pd.Timestamp('2022-01-01')).all()
    weights,audit=source_consistent_endpoint_weights(meta,endpoint_weights);np.save(inp/'train_weights_PRIVATE.npy',weights)
    for rep in range(8):
        rows,groups=resample_groups(tr['groups'],8000+rep);w,wa=source_consistent_endpoint_weights(meta.iloc[rows].reset_index(drop=True),endpoint_weights)
        np.savez_compressed(inp/f'bootstrap_{rep}_PRIVATE.npz',rows=rows,groups=groups,weights=w)
    np.savez_compressed(inp/'references_PRIVATE.npz',background=group_reference(tr['X'],tr['groups'],16,800),anchors=group_reference(ev['X'],ev['groups'],32,801))
    for source in [parent,order]:
        for p in (source/'results/assembled').glob('*/task-*/output/summary.json'):
            s=json.loads(p.read_text());model={'coverage':'q2'}.get(s['variant'],s['variant']);seed=s['seed']
            copy(p.parent/'predictions_PRIVATE.npz',inp/f'reference_{model}_{seed}_PRIVATE.npz')
            if source==parent and seed==42:
                copy(p.parent/'forest_PRIVATE.json',inp/f'reference_{model}_42_forest_PRIVATE.json')
    parentplan=json.loads((parent/'input/plan.json').read_text())
    configs={'q2':parentplan['configs']['final/42/coverage'],'unrestricted':parentplan['configs']['final/42/unrestricted']}
    plan=dict(status='FROZEN_POST_EXPOSURE_EXPLORATORY_PROVISIONAL',parent=str(parent),order_run=str(order),seeds=[42,101,202],
        baseline_models=['ebm_gam','ebm_ga2m10','ebm_ga2m30'],ebm_rounds=2000,bootstrap_replicates=8,bootstrap_models=['q2','unrestricted','ebm_ga2m10'],
        forest_configs=configs,simulation_scenarios=SCENARIOS,simulation_seeds=[42,101,202,303,404],simulation_models=['q2','q3','unrestricted','ebm_gam','ebm_ga2m10'],
        simulation_train_n=3000,simulation_eval_n=1000,simulation_ebm_rounds=1000,simulation_effect_threshold=.005,
        shap_models=['q2','unrestricted'],shap_seeds=[42,101,202],shap_permutations=[1,8,32],shap_anchors=12,exact_anchors=32,shap_background=16,
        formal_baseline_fits=9,bootstrap_fits=24,simulation_fits=175,shap_tasks=18,
        source_counts=parentplan['source_counts'],training_weight_audit=audit,source_hashes=sources,
        caveats=['Only fixed 12-calendar-month endpoint classification for EBMs, not full competing-risk curves',
                 'EBM-GAM and GA2M-style EBM are one method family, not independent unrelated algorithms',
                 'EBM explanations log-odds plus logistic link; forest explanations CIF probability',
                 'Exploratory same previously exposed manufacturing 2023/2024; no strict holdout claim',
                 'Eight training cluster bootstraps are descriptive pilot stability, not calibrated confidence intervals',
                 'Observed-distribution and uniform product reference structure recovery are distinct under correlation',
                 'Finite-budget baselines are not exhaustive optimized reference methods'],
        stopping='One frozen finite comparison, no outcome-selected extensions or dropped seeds',
        private_storage=str(run),resource_policy='32 physical workers across oliver/lemon, one thread/item, 6GiB address space limit, NFS')
    write(inp/'plan.json',plan);write(run/'audit/input_provenance.json',dict(source_hashes=sources,train_ipcw=audit,training_rows=len(meta),training_company_groups=meta.group.nunique(),unknown_weight_zero=int((weights==0).sum())))
    shutil.copy2(parent/'audit/data_alignment.json',run/'audit/data_alignment.json')
    print(json.dumps({k:plan[k] for k in ['formal_baseline_fits','bootstrap_fits','simulation_fits','shap_tasks','training_weight_audit']}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);p.add_argument('--order',type=Path,required=True);a=p.parse_args();main(a.run,a.parent,a.order)
