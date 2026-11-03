import pandas as pd
from crforest import CompetingRiskForest
from .common import *

def main():
    r=run_path();rows=[]
    for n,h in load(r/'audit/input_hashes.json').items():assert sha(r/'input'/n)==h
    for n,h in load(r/'audit/source_hashes.json').items():assert sha(Path(n))==h
    assert sha(REPO/'CURRENT_POST_ST_MODEL.json')==load(r/'audit/current_pointer.json')['sha256']
    old=load(r/'input/old_registry.json')
    for y in YEARS:
        tr=arrays(r/f'input/folds/{y}/train_PRIVATE.npz');ev=arrays(r/f'input/folds/{y}/eval_PRIVATE.npz')
        labels=arrays(r/f'evaluation_input/{y}_test_PRIVATE.npz')
        np.testing.assert_array_equal(ev['sample_ids'],labels['sample_ids'])
        np.testing.assert_array_equal(ev['X'],labels['X'][:,load(SOURCE/'audit/keep_PRIVATE.json')])
        assert not set(tr['groups'])&set(labels['groups'])
        for s in SEEDS:
            p=r/f'input/parents/{y}/{s}';original=Path(old[f'{y}|{s}'])
            for v in ['raw','shrunk']:
                model=CompetingRiskForest.load(p/f'{v}_PRIVATE.json');base=CompetingRiskForest.load(original/f'{v}/forest_PRIVATE.json')
                assert json.dumps(model.trees_[:32])==json.dumps(base.trees_)
            np.testing.assert_equal(pairing(tr['groups'],configuration(s,40),74)[:32],load(original/'pairing.json'))
            rows.append(dict(year=y,seed=s,staged_trees=40,old32_raw_shrunk_identical=True,paired_schedule_unchanged=True,
                disjoint_train_test_companies=True,training_sha256=sha(r/f'input/folds/{y}/train_PRIVATE.npz'),forest_fits=0))
    pd.DataFrame(rows).to_csv(r/'reports/PAIRING_DATA_AUDIT.csv',index=False)
    write(r/'reports/data_validation.json',dict(status='PASS',forest_fits=0,new_trees=0,all_source_hashes_unchanged=True,
        original32_raw_shrunk_identical=True,disjoint_train_test_groups=True,evaluation74_columns_match=True,current_pointer_unchanged=True))

if __name__=='__main__':main()
