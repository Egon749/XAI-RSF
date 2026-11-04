import pandas as pd
from .common import *
from experiments.post_st_seed_explanation_stability_v1.common import SEEDS,TRIPLES,NAMES
def history_members():
    r=old('post_st_requested_seeds_v1')
    return {(j['year'],j['seed']):Path(j['path']) for n in ['pilot','formal'] for j in read(r/f'audit/registry_{n}.json')}
def stored(y):
    path=old('post_st_seed_explanation_stability_v1')/'input';root=path/str(y)
    expected=read(ROOT/'experiments/post_st_seed_explanation_stability_v1/MODEL_HASH_AUDIT.json')['before']['models']
    rows={};hashes={};ids=arrays(path/f'folds/{y}/eval_PRIVATE.npz')['sample_ids']
    for a in [*map(str,SEEDS),'reference96']:
        folder=root/a/'shrunk';hh=expected[f'{y}/{a}/shrunk']
        for fn,src in hh.items():assert sha(folder/fn)==src
        z=arrays(folder/'predictions_PRIVATE.npz');np.testing.assert_array_equal(z['times'],DAYS);np.testing.assert_array_equal(ids,z['sample_ids'])
        rows[a]=z['cif'];hashes[a]=hh
    results={f'XAI/{s}':rows[str(s)] for s in PICK}
    for s in ANCHOR:
        member=history_members()[y,s];assert sha(member/'shrunk/predictions_PRIVATE.npz')==read(member/'hashes.json')['shrunk/predictions_PRIVATE.npz']
        z=arrays(member/'shrunk/predictions_PRIVATE.npz');np.testing.assert_array_equal(ids,z['sample_ids']);results[f'XAI/{s}']=z['cif']
    np.testing.assert_allclose(rows['reference96'],np.mean([results[f'XAI/{s}'] for s in ANCHOR],axis=0),atol=1e-12,rtol=0)
    for name,ix in zip(NAMES,TRIPLES):results['triple/'+name]=np.mean([rows[str(SEEDS[i])] for i in ix],axis=0)
    results['XAI/C039']=results['triple/C039'];results['XAI/anchor']=rows['reference96']
    np.testing.assert_array_equal(results['XAI/C039'],np.mean([rows[str(s)] for s in PICK],axis=0))
    return results,hashes
def jobs():
    folders=list(run().glob('published/*/task-*/output/done.json'));out={}
    for done in folders:
        s=read(done);k=(s['arm'],s['year'],s['seed']);assert k not in out
        p=done.parent;assert (p.parent/'complete.json').exists()
        out[k]=(p,s)
    assert len(out)==33,len(out)
    return out
def collect(y,work):
    book,hashes=stored(y);ev=arrays(run()/f'input/{y}/eval.npz')
    for (arm,yy,seed),(folder,rec) in work.items():
        if yy!=y:continue
        z=arrays(folder/'predictions.npz');np.testing.assert_array_equal(z.pop('sample_ids'),ev['sample_ids'])
        for s,cif in z.items():book[arm if s=='0' else f'{arm}/{s}']=cif
    for arm in ('RF','unrestricted'):
        for name,ss in [('selected_seeds',PICK),('anchor_seeds',ANCHOR)]:
            book[arm+'/'+name]=np.mean([book[f'{arm}/{s}'] for s in ss],axis=0)
    return book,hashes
