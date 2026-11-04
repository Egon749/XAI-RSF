"""Archive the exact 96-tree C039 traversal, preserving all original source files."""
from crforest import CompetingRiskForest
from experiments.post_st_ablation_v1.verification import merge
from .common import *
def main():
    r=run();paths=read(r/'anchor_paths_PRIVATE.json');rows=[]
    for y in YEARS:
        members=[CompetingRiskForest.load(Path(paths[f'{y}|{s}'])/'forest_PRIVATE.json') for s in SEEDS]
        model=merge(members);d=r/f'C039/{y}';p=d/'forest_materialized_PRIVATE.json';model.save(p)
        reloaded=CompetingRiskForest.load(p);assert reloaded.trees_==model.trees_ and len(reloaded.trees_)==96
        ev=arrays(r/f'input/folds/{y}/eval_PRIVATE.npz');z=arrays(d/'predictions_PRIVATE.npz')
        err=float(abs(reloaded.predict(ev['X'],TIMES)-z['cif']).max());assert err<=1e-10
        rows.append(dict(year=y,model='C039',seed_order='17,503,22',trees=96,forest_sha256=sha(p),member_mean_prediction_sha256=sha(d/'predictions_PRIVATE.npz'),reload_vs_mean=err))
    pd.DataFrame(rows).to_csv(D/'M3_ENSEMBLE_HASHES.csv',index=False)
if __name__=='__main__':main()
