"""Read-only verification of the historical 74-column complete-model anchor."""
from crforest import CompetingRiskForest
from experiments.post_st_ablation_v1.verification import merge
from .common import *
def main():
    src=source();members=read(src/'input/old_registry.json');rows=[]
    for z in read(src/'input/old_ensemble_registry.json'):
        y,v=z['year'],z['version'];p=Path(z['path'])/'representation'
        expected=read(p.parent/'hashes.json')
        for name,h in expected.items():assert sha(p.parent/name)==h
        old=CompetingRiskForest.load(p/'forest_PRIVATE.json')
        ms=[CompetingRiskForest.load(Path(members[f'{y}|{s}'])/v/'forest_PRIVATE.json') for s in SEEDS]
        assert merge(ms).trees_==old.trees_ and old.n_features_==74
        z0=arrays(p/'predictions_PRIVATE.npz')
        ps=[arrays(Path(members[f'{y}|{s}'])/v/'predictions_PRIVATE.npz') for s in SEEDS]
        for pr in ps:np.testing.assert_array_equal(pr['sample_ids'],z0['sample_ids'])
        err=float(abs(np.mean([a['cif'] for a in ps],axis=0)-z0['cif']).max());assert err<1e-10
        k=read(p/'verification.json');rows.append(dict(year=y,version=v,forest_sha256=sha(p/'forest_PRIVATE.json'),prediction_sha256=sha(p/'predictions_PRIVATE.npz'),member_mean_error=err,
            main_functions=k['main_functions'],interaction_functions=k['interaction_functions'],spatial_table_cells=k['spatial_table_cells'],status='historical_only_not_eligible_as_cell'))
    pd.DataFrame(rows).to_csv(D/'HISTORICAL_M3_HASH_AND_COMPLEXITY.csv',index=False)
if __name__=='__main__':main()
