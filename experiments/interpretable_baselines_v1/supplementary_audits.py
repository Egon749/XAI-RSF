"""Read-only signal magnitude / ensemble dilution and EBM saturation audits."""
from pathlib import Path
import argparse,json,pickle
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from crforest import CompetingRiskForest
from experiments.interpretable_baselines_v1.methods import ScalarForest,binary_grid,walsh


def main(run):
    rows=[];dilution=[];grid=binary_grid();W=walsh(grid)
    for stage in ['formal','coverage']:
        for p in (run/f'results/{stage}').glob('*/task-*/output/summary.json'):
            d=json.loads(p.read_text())
            if d['kind']!='simulation':continue
            z=np.load(p.parent/'structure_PRIVATE.npz')
            for mask in np.flatnonzero(abs(z['truth'])>1e-10):
                if mask==0:continue
                rows.append(dict(scenario=d['scenario'],model=d['model'],seed=d['seed'],term='x'+'*x'.join(str(j) for j in range(8) if mask>>j&1),order=int(mask).bit_count(),true_coefficient=float(z['truth'][mask]),estimated_coefficient=float(z['coefficient'][mask])))
            if d['scenario']=='pair_cif' and d['model']=='q2_coverage':
                forest=CompetingRiskForest.load(p.parent/'forest_PRIVATE.json');values=[]
                for tree in forest.trees_:
                    one=CompetingRiskForest(forest.config);one.trees_=[tree];one.grid_=forest.grid_;one.n_features_=forest.n_features_
                    values.append(float(W[:,6]@ScalarForest(one,2.)(grid)/len(grid)))
                active=np.abs(values)>1e-10
                dilution.append(dict(seed=d['seed'],trees=len(values),trees_with_true_pair_effect=int(active.sum()),active_tree_fraction=float(active.mean()),active_tree_coefficient_mean=float(np.mean(np.array(values)[active])) if active.any() else None,ensemble_coefficient=float(np.mean(values)),true_coefficient=.05))
    d=pd.DataFrame(rows);s=d.groupby(['scenario','model','term','order']).agg(truth=('true_coefficient','first'),estimated_mean=('estimated_coefficient','mean'),estimated_min=('estimated_coefficient','min'),estimated_max=('estimated_coefficient','max')).reset_index();s.to_csv(run/'reports/simulation_signal_recovery.csv',index=False)
    pd.DataFrame(dilution).to_csv(run/'reports/simulation_pair_dilution.csv',index=False)
    tr=np.load(run/'input/final_train_PRIVATE.npz');ev=np.load(run/'input/final_eval_PRIVATE.npz');w=np.load(run/'input/train_weights_PRIVATE.npy');y=(tr['event']==1).astype(int);audit=[]
    for p in (run/'results/formal').glob('*/task-*/output/summary.json'):
        s=json.loads(p.read_text())
        if s['kind']!='baseline':continue
        with (p.parent/'model_PRIVATE.pkl').open('rb') as f:m=pickle.load(f)
        a=m.decision_function(tr['X']);b=m.decision_function(ev['X']);q=m.predict_proba(tr['X'])[:,1];v=m.predict_proba(ev['X'])[:,1]
        audit.append(dict(model=s['model'],seed=s['seed'],train_score_min=float(a.min()),train_score_max=float(a.max()),eval_score_min=float(b.min()),eval_score_max=float(b.max()),train_weighted_logloss=float(log_loss(y[w>0],q[w>0],sample_weight=w[w>0])),eval_risk_gt_99=int((v>.99).sum()),train_risk_gt_99=int((q>.99).sum())))
    pd.DataFrame(audit).to_csv(run/'reports/ebm_saturation.csv',index=False)
    print(pd.DataFrame(dilution).to_string(index=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);main(p.parse_args().run)
