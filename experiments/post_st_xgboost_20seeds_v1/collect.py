import numpy as np
from .common import *

def collect():
    found={};receipts=[];grouped={}
    for f in sorted(run().glob('published/*/task-*/output/result.json')):
        p=f.parent;r=read(f);key=r['year'],r['months'],r['batch']
        assert key not in found;found[key]=r
        for row in read(p.parent/'complete.json')['manifest']:
            if len(row)==2:assert (p/row[0]).is_dir();continue
            fn,size,hh=row;assert (p/fn).stat().st_size==size and sha(p/fn)==hh
        assert r['final_fits']==5 and r['tuning_fits']==0
        assert [m['seed'] for m in r['models']]==list(seed_batches()[r['batch']])
        for role in ('train','eval'):
            assert r['inputs'][role+'.npz']==sha(run()/f'input/{r["year"]}/{role}.npz')
        grouped.setdefault(key[:2],[]).append((p,r))
        receipts.append(dict(year=key[0],months=key[1],batch=key[2],prediction_hash=sha(p/'predictions_PRIVATE.npz'),manifest_hash=sha(p.parent/'complete.json'),job_id=r['job_id'],hostname=r['hostname'],task_seconds=r['seconds'],peak_RSS_MiB=r['peak_RSS_MiB'],successful_fits=5))
    assert set(found)=={(y,h,b) for y in YEARS for h in (12,24) for b in range(4)}
    result={}
    for (y,h),parts in grouped.items():
        out=run()/f'aggregate/{y}_{h}';out.mkdir(parents=True,exist_ok=True)
        models=[];pp={};train_pp={};ids=None;train_ids=None
        for path,r in parts:
            z=arrays(path/'predictions_PRIVATE.npz');tr=arrays(path/'training_predictions_PRIVATE.npz')
            if ids is None:ids=z['sample_ids'];train_ids=tr['sample_ids']
            np.testing.assert_array_equal(ids,z['sample_ids']);np.testing.assert_array_equal(train_ids,tr['sample_ids'])
            np.testing.assert_array_equal(z['ensemble'],np.mean([z[str(s)] for s in seed_batches()[r['batch']]],axis=0))
            for row in r['models']:
                s=row['seed'];assert str(s) not in pp
                pp[str(s)]=z[str(s)];train_pp[str(s)]=tr[str(s)]
                models.append(dict(**row,actual_node=r['hostname'],actual_task_seconds=r['seconds'],actual_peak_RSS_MiB=r['peak_RSS_MiB'],batch=r['batch']))
        assert set(pp)==set(map(str,SEEDS))
        pp['ensemble']=np.mean([pp[str(s)] for s in SEEDS],axis=0)
        np.savez_compressed(out/'predictions_PRIVATE.npz',sample_ids=ids,**pp)
        np.savez_compressed(out/'training_predictions_PRIVATE.npz',sample_ids=train_ids,**train_pp)
        rr=dict(year=y,months=h,models=models,final_fits=20,tuning_fits=0,seconds=sum(r['seconds'] for _,r in parts),peak_RSS_MiB=max(r['peak_RSS_MiB'] for _,r in parts),hostname=';'.join(sorted({r['hostname'] for _,r in parts})))
        write(out/'result.json',rr);result[y,h]=out,rr
    return result,receipts
