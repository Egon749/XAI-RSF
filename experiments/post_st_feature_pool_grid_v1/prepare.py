"""Verify and copy frozen own-cutoff inputs; no outer outcomes read."""
import shutil
from .common import *
from experiments.post_st_shrinkage_fine_v1.prepare import verify_output

def main():
    r=run();old=Path(load(D.parent/'post_st_shrinkage_fine_v1/run_location.json')['private_run'])
    oldaudit=load(D.parent/'post_st_shrinkage_fine_v1/DATA_AUDIT.json')
    for name,digest in oldaudit['input_hashes'].items():assert sha(old/name)==digest
    inp=r/'input';inp.mkdir()
    for name in ['folds','raw','reference']:shutil.copytree(old/'input'/name,inp/name)
    shutil.copytree(old/'selection_input',r/'selection_input')
    cfg={str(c):{str(s):asdict(configuration(s,c)) for s in SEEDS} for c in COUNTS}
    write(inp/'configs.json',cfg);write(D/'FROZEN_CONFIGS.json',cfg)
    diffs=[]
    for c in COUNTS:
        actual=asdict(configuration(17,c));ref=asdict(configuration(17,REFERENCE))
        assert [k for k in actual if actual[k]!=ref[k]]==(['tree_feature_count'] if c!=REFERENCE else [])
        for k,v in actual.items():diffs.append(dict(pool_count=c,parameter=k,value=json.dumps(v),changed=v!=ref[k]))
    pd.DataFrame(diffs).to_csv(D/'PARAMETER_DIFFERENCES.csv',index=False)
    for y in ALL_YEARS:
        tr=arrays(inp/f'folds/{y}/train_PRIVATE.npz');ev=arrays(inp/f'folds/{y}/eval_PRIVATE.npz')
        assert set(ev)=={'X','sample_ids'} and tr['X'].shape[1]==74
        assert not set(tr['sample_ids'])&set(ev['sample_ids'])
        for s in SEEDS:
            raw=load(inp/f'raw/{y}/{s}/forest_PRIVATE.json')
            assert raw['config']==json.loads(json.dumps(asdict(configuration(s,REFERENCE))))
            assert all('leaf_shrinkage' not in n for tree in raw['trees'] for n in tree)
    pointer=sha(ROOT/'CURRENT_POST_ST_MODEL.json');assert pointer==oldaudit['current_pointer_sha256']
    for n in ['TRAINING_SUPPORT.csv','HISTORICAL_VALIDATION_SUPPORT.csv']:shutil.copyfile(D.parent/'post_st_shrinkage_fine_v1'/n,D/n)
    write(D/'DATA_AUDIT.json',dict(source_commit='f7d59c91086c17fa0aaee8f155139bfdef0c8d4f',current_pointer_sha256=pointer,
        input_hashes={str(p.relative_to(r)):sha(p) for p in r.rglob('*') if p.is_file()},outer_labels_read=False,
        forest_fits_planned=468,reference_reproductions=18,training_input_columns=74,own_cutoff_inputs_unchanged=True))
    print(json.dumps(dict(input_years=6,counts=26,planned_fits=468,outer_labels_read=False)))
if __name__=='__main__':main()
