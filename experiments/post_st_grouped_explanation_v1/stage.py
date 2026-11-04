"""Before/after hash check for the saved functions, source forests and immutable folds."""
import shutil
from .common import *

def inventory(stage=False):
    source=old();root=run()/'input';ref=read(PREV/'MODEL_HASH_AUDIT.json')['before']
    hashes={}
    def check(src,key,expected):
        assert sha(src)==expected,(key,src);hashes[key]=expected
        if stage:
            dest=root/key;dest.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(src,dest)
    for key,h in ref['inputs'].items():check(source/'input'/key,key,h)
    for y in YEARS:
        for a in ARMS:
            key=f'{y}/{a}/shrunk'
            for name,h in ref['models'][key].items():check(source/'input'/key/name,f'{y}/{a}/{name}',h)
    prior=pd.read_csv(PREV/'RECONSTRUCTION_DETAIL.csv').query("version=='shrunk'").set_index(['year','model'])
    for task in read(source/'verify/published.json'):
        year,v,first,last=task['args']
        if v!='shrunk':continue
        for name in COMB[int(first):int(last)]:
            check(Path(task['path'])/(name+'_PRIVATE.npz'),f'{year}/{name}.npz',prior.at[(int(year),name),'function_sha256'])
    # The previous fixed low/high case identities and scalar case-selection statistic stay private.
    for task in read(source/'explain/published.json'):
        year,v=task['args']
        if v!='shrunk':continue
        path=Path(task['path'])
        for name in ['firms_PRIVATE.csv',*[f'case_{h}_{c}_{q}_PRIVATE.npz' for h in [12,24] for c in [1,2] for q in [.1,.9]]]:
            # The older return is immutable. Record local files in addition to the prior public manifest.
            check(path/name,f'cases/{year}/{name}',sha(path/name))
    assert sha(REPO/'CURRENT_POST_ST_MODEL.json')==ref['root_pointer_sha256']
    return dict(hashes=hashes,current_model_pointer=ref['root_pointer_sha256'],main_bundle_features=sha(REPO/'主模型/配置/INPUT_FEATURES.csv'),main_bundle_spec=sha(REPO/'主模型/配置/MODEL_SPEC.json'))
if __name__=='__main__':
    import sys
    if 'post' in sys.argv:assert inventory()==read(run()/'source_hashes.json')
    else:dump(run()/'source_hashes.json',inventory(True))
