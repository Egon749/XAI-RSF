"""Reuse immutable own-cutoff snapshots and checked raw forests; zero fits."""
import shutil
from dataclasses import asdict
from .common import *

def previous():
    return Path(load(D.parent/'post_st_shrinkage_strength_v1/run_location.json')['private_run'])

def verify_output(p):
    for name,digest in load(p/'hashes.json').items():assert sha(p/name)==digest
    for record in load(p.parent/'complete.json')['manifest']:
        if len(record)==2:assert record[1]=='directory' and (p/record[0]).is_dir()
        else:
            name,size,digest=record;assert (p/name).stat().st_size==size and sha(p/name)==digest

def main():
    r=run();old=previous();src=old/'derive_input';inp=r/'input';inp.mkdir()
    for name,digest in load(old/'derive_input_hashes.json').items():assert sha(src/name)==digest
    for name in ['folds','raw','reference']:shutil.copytree(src/name,inp/name)
    shutil.copyfile(src/'configs.json',inp/'configs.json')
    shutil.copytree(old/'selection_input',r/'selection_input')
    # copy hashes and reuse paths privately; neither company records nor models enter Git.
    records=[]
    for p in (old/'derive_published').glob('*/task-*/output'):
        s=load(p/'summary.json')
        if s['strength'] not in REUSED:continue
        assert s['status']=='COMPLETE' and s['forest_fits']==0 and s['year'] in ALL_YEARS
        verify_output(p)
        assert s['source_train_sha256']==sha(inp/f"folds/{s['year']}/train_PRIVATE.npz")
        records.append(dict(strength=s['strength'],year=s['year'],path=str(p),summary_sha256=sha(p/'summary.json')))
    assert len(records)==36 and len({(r['strength'],r['year']) for r in records})==36
    write(r/'reused_outputs_PRIVATE.json',records)
    for y in ALL_YEARS:
        tr=arrays(inp/f'folds/{y}/train_PRIVATE.npz');ev=arrays(inp/f'folds/{y}/eval_PRIVATE.npz')
        assert set(ev)=={'X','sample_ids'} and tr['X'].shape[1]==74
        assert not set(tr['sample_ids'])&set(ev['sample_ids'])
        for seed in SEEDS:
            raw=load(inp/f'raw/{y}/{seed}/forest_PRIVATE.json')
            assert raw['config']==json.loads(json.dumps(asdict(configuration(seed))))
            assert all('leaf_shrinkage' not in n for tree in raw['trees'] for n in tree)
    pointer=sha(ROOT/'CURRENT_POST_ST_MODEL.json')
    assert pointer==load(D.parent/'post_st_shrinkage_strength_v1/DATA_AUDIT.json')['current_pointer_sha256']
    for name in ['TRAINING_SUPPORT.csv','HISTORICAL_VALIDATION_SUPPORT.csv','FROZEN_CONFIGS.json']:
        shutil.copyfile(D.parent/'post_st_shrinkage_strength_v1'/name,D/name)
    hashes={str(p.relative_to(r)):sha(p) for p in r.rglob('*') if p.is_file()}
    write(D/'DATA_AUDIT.json',dict(current_pointer_sha256=pointer,source_experiment_commit='a88d84618fdb86fe8e5771b9d1d6999e598e23b6',
        input_hashes=hashes,own_cutoff_inputs_unchanged=True,outer_labels_read=False,raw_forests_reused=18,
        year_strength_outputs_reused=36,new_year_strength_outputs=90,forest_fits=0))
    rows=[]
    for s in STRENGTHS:
        for k,v in asdict(configuration(17)).items():rows.append(dict(strength=s,parameter=k,value=json.dumps(v),changed=False))
        rows.append(dict(strength=s,parameter='direct_parent_AJ_strength',value=s,changed=s!=10))
    pd.DataFrame(rows).to_csv(D/'PARAMETER_DIFFERENCES.csv',index=False)
    print(json.dumps(dict(raw_forests=18,reused_ensembles=36,new_ensembles=90,outer_labels_read=False)))

if __name__=='__main__':main()
