"""Audit and stage own-cutoff snapshots; no outer outcome access."""
from dataclasses import asdict
import shutil
from .common import *
from experiments.post_st_paper_features_v1.common import history

def main():
    r=run();inp=r/'input';inp.mkdir();audit=[];hashes={};configs={}
    pointer=load(ROOT/'CURRENT_POST_ST_MODEL.json')
    assert pointer['arm']=='C039' and pointer['configuration']['seeds']==list(SEEDS)
    for year in YEARS:
        src=source()/f'input/folds/{year}';data=arrays(src/'train_PRIVATE.npz')
        meta_path=history(year)/f'evaluation_input/{year}_fit_metadata_PRIVATE.parquet'
        meta=pd.read_parquet(meta_path)
        np.testing.assert_array_equal(meta.event,data['event']);np.testing.assert_allclose(meta.time,data['time'],rtol=0,atol=1e-12)
        assert data['X'].shape[1]==74
        ev=arrays(src/'eval_PRIVATE.npz');assert set(ev)=={'X','sample_ids'}
        assert not set(ev['sample_ids']) & set(data['sample_ids'])
        fold=inp/f'folds/{year}';fold.mkdir(parents=True)
        shutil.copyfile(src/'eval_PRIVATE.npz',fold/'eval_PRIVATE.npz')
        hashes[f'{year}/source_train']=sha(src/'train_PRIVATE.npz')
        hashes[f'{year}/metadata']=sha(meta_path);hashes[f'{year}/eval_X']=sha(src/'eval_PRIVATE.npz')
        for start in STARTS:
            tr,mask=filter_history(data,meta,start,year)
            dst=fold/str(start);dst.mkdir();np.savez_compressed(dst/'train_PRIVATE.npz',**tr)
            meta.loc[mask].to_parquet(dst/'metadata_PRIVATE.parquet',index=False)
            hashes[f'{year}/{start}/train']=sha(dst/'train_PRIVATE.npz')
            for h in (12,24):
                e=tr['event'];t=tr['time'];hh=tr[f'h{h}'];known=(e>0)|(t>=hh)
                audit.append(dict(year=year,earliest_entry_year=start,months=h,companies=len(e),
                    termination=int(((e==1)&(t<=hh)).sum()),recovery=int(((e==2)&(t<=hh)).sum()),
                    no_event_at_horizon=int((t>hh).sum()+((t==hh)&(e==0)).sum()),
                    censored_before_horizon=int(((e==0)&(t<hh)).sum()),
                    all_observed_termination=int((e==1).sum()),all_observed_recovery=int((e==2).sum()),
                    train_sha256=hashes[f'{year}/{start}/train']))
        for seed in SEEDS:
            p=source()/f'input/{year}/{seed}/shrunk/forest_PRIVATE.json';v=load(p)
            assert v['config']==json.loads(json.dumps(asdict(configuration(seed))))
            item=next(x for x in pointer['saved_models'] if x['test_year']==year and x['seed']==seed)
            assert sha(p)==item['forest_sha256']
            target=inp/f'reference/{year}/{seed}';target.mkdir(parents=True)
            for name,key in [('forest_PRIVATE.json','forest_sha256'),('predictions_PRIVATE.npz','predictions_sha256')]:
                assert sha(p.parent/name)==item[key];shutil.copyfile(p.parent/name,target/name)
                hashes[f'{year}/{seed}/{name}']=sha(target/name)
            configs[str(seed)]=v['config']
    write(inp/'configs.json',configs)
    write(D/'DATA_AUDIT.json',dict(input_hashes=hashes,current_pointer_sha256=sha(ROOT/'CURRENT_POST_ST_MODEL.json'),
        evaluation_labels_read=False,feature_count=74,company_disjoint=True,clocks_and_censoring_unchanged=True))
    pd.DataFrame(audit).to_csv(D/'TRAINING_SUPPORT.csv',index=False)
    rows=[]
    for start in STARTS:
        for key,value in configs[str(SEEDS[0])].items():
            rows.append(dict(earliest_entry_year=start,parameter=key,value=json.dumps(value),changed_from_reference=False))
        rows.append(dict(earliest_entry_year=start,parameter='earliest_training_entry_year',value=start,changed_from_reference=start!=2012))
    pd.DataFrame(rows).to_csv(D/'PARAMETER_DIFFERENCES.csv',index=False)
    print(json.dumps({'staged_windows':len(STARTS)*len(YEARS),'outer_labels_read':False}))

if __name__=='__main__':main()
