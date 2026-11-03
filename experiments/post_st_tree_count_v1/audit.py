"""Final immutable input, provenance, bootstrap and prefix accounting audit."""
import pandas as pd
from .common import *

def main():
    run=run_path();rows=[]
    for n,h in load(run/'audit/input_hashes.json').items():assert sha(run/'input'/n)==h
    for n,h in load(run/'audit/source_hashes.json').items():assert sha(Path(n))==h
    assert sha(REPO/'CURRENT_POST_ST_MODEL.json')==load(run/'audit/current_pointer.json')['sha256']
    grown=load(run/'audit/registry_grow.json');assert len(grown)==63
    for y in YEARS:
        tr=arrays(run/f'input/folds/{y}/train_PRIVATE.npz');ev=arrays(run/f'input/folds/{y}/eval_PRIVATE.npz')
        labels=arrays(run/f'evaluation_input/{y}_test_PRIVATE.npz')
        np.testing.assert_array_equal(ev['sample_ids'],labels['sample_ids'])
        np.testing.assert_array_equal(ev['X'],labels['X'][:,load(SOURCE/'audit/keep_PRIVATE.json')])
        assert not set(tr['groups'])&set(labels['groups'])
        for s in SEEDS:
            chunks=[load(Path(z['path'])/'trees_PRIVATE.json')for z in grown if z['year']==y and z['seed']==s]
            ids=sorted(t['tree_id']for z in chunks for t in z);assert ids==list(range(32,256))
            expected=pairing(tr['groups'],configuration(s,256),74)
            for z in chunks:
                for t in z:assert t['inbag_sha256']==expected[t['tree_id']]['multiplicity_sha256']
            rows.append(dict(year=y,seed=s,reused_trees=32,new_trees=224,total_trees=256,
                exact_disjoint_original_indices=True,no_train_test_group_overlap=True,
                bootstrap_and_feature_schedule_unchanged=True,evaluation74_columns_match_original=True,
                train_sha256=sha(run/f'input/folds/{y}/train_PRIVATE.npz'),evaluation_X_sha256=sha(run/f'input/folds/{y}/eval_PRIVATE.npz')))
    assert len(load(run/'audit/registry_assemble.json'))==9
    for z in load(run/'audit/registry_assemble.json'):
        summary=load(Path(z['path'])/'summary.json')
        assert summary['old32_raw_identical']and summary['old32_shrunk_identical']
    pd.DataFrame(rows).to_csv(run/'reports/PAIRING_DATA_AUDIT.csv',index=False)
    write(run/'reports/data_validation.json',dict(status='PASS',formal_new_trees=2016,reused_trees=288,
        all_source_hashes_unchanged=True,original32_raw_shrunk_identical=True,disjoint_train_test_groups=True,
        evaluation_label_container83_mapped_to_model74=True,current_pointer_unchanged=True))

if __name__=='__main__':main()
