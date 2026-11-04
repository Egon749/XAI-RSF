"""Verify immutable source artefacts and stage an isolated shared input."""
from dataclasses import asdict
import shutil
import pandas as pd
from crforest import CompetingRiskForest
from experiments.post_st_requested_seeds10_v1.common import configuration
from .common import *

def catalog():
    source=source_path()
    paths={}
    for row in read(source/'input/reused_registry.json'):
        for version in VERSIONS:
            paths[row['year'],str(row['seed']),version]=Path(row['path'])/version
    for registry,arm,rep in [('input/old_ensemble_registry.json','reference96','representation'),('audit/registry_ensemble.json','ensemble320',None)]:
        for row in read(source/registry):
            paths[row['year'],arm,row['version']]=Path(row['path'])/(rep or row['version'])
    assert len(paths)==72
    return paths

def audit(stage=False):
    source=source_path(); paths=catalog(); entries={}; inputs={}
    frozen=pd.read_csv(BASE/'MODEL_HASHES.csv').set_index(['year','seed','version'])
    historical=read(source/'audit/input_hashes.json')
    for (year,arm,version),p in paths.items():
        oldroot=Path(str(p).split('/results/')[0])
        for name in ['features.json',f'folds/{year}/train_PRIVATE.npz',f'folds/{year}/eval_PRIVATE.npz']:
            digest=sha(source/'input'/name)
            assert historical[name]==digest==sha(oldroot/'input'/name),(oldroot,name)
            inputs[name]=digest
        label=f'evaluation_input/{year}_test_PRIVATE.npz'
        assert sha(source/label)==sha(oldroot/label)
        inputs[label]=sha(source/label)
        s=frozen.loc[(year,arm,version)]
        model=CompetingRiskForest.load(p/'forest_PRIVATE.json')
        actual=asdict(model.config); expected=asdict(configuration(actual['seed'],len(model.trees_)))
        assert actual==expected and model.n_features_==74
        if arm.isdigit(): assert actual['seed']==int(arm) and len(model.trees_)==32
        else: assert len(model.trees_)=={'reference96':96,'ensemble320':320}[arm]
        leaves=[node for tree in model.trees_ for node in tree if 'cif' in node]
        assert all(('leaf_shrinkage' in z)==(version=='shrunk') for z in leaves)
        if version=='shrunk':assert all(z['leaf_shrinkage']['strength']==10 for z in leaves)
        h={}
        for file,col in [('forest_PRIVATE.json','forest_sha256'),('functions_PRIVATE.npz','function_sha256'),('predictions_PRIVATE.npz','prediction_sha256')]:
            h[file]=sha(p/file);assert h[file]==s[col]
        entries[f'{year}/{arm}/{version}']=h
        if stage:
            dest=run_path()/'input'/str(year)/arm/version;dest.mkdir(parents=True)
            for file in h:shutil.copy2(p/file,dest/file)
    if stage:
        dest=run_path()/'input'
        for name in inputs:
            src=source/name if name.startswith('evaluation_input') else source/'input'/name
            (dest/name).parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(src,dest/name)
        pd.DataFrame([dict(combo=n,seed1=SEEDS[a],seed2=SEEDS[b],seed3=SEEDS[c]) for n,(a,b,c) in zip(NAMES,TRIPLES)]).to_csv(HERE/'COMBINATIONS_120.csv',index=False)
    return dict(models=entries,inputs=inputs,root_pointer_sha256=sha(ROOT/'CURRENT_POST_ST_MODEL.json'))

if __name__=='__main__':
    import sys
    if 'post' in sys.argv:
        assert audit()==read(run_path()/'hash_before.json')
    else:
        dump(run_path()/'hash_before.json',audit(True))
