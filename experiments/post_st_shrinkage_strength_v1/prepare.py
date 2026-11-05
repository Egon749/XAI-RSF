"""Read own-cutoff inputs and stage history without external test labels."""
from dataclasses import asdict
import shutil
from .common import *
from experiments.post_st_paper_features_v1.common import history
from experiments.post_st_drop9_v1.common import kept_columns, reduced, HISTORY

def copy(src, dst, hashes):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    hashes[str(dst.relative_to(run()))] = sha(src)

def main():
    inp=run()/'input';inp.mkdir();audit=[];hashes={};current={};metas={}
    pointer=load(ROOT/'CURRENT_POST_ST_MODEL.json')
    assert pointer['arm']=='C039' and pointer['configuration']['seeds']==list(SEEDS)
    prior=Path(load(D.parent/'post_st_training_start_v1/run_location.json')['private_run'])
    raw_registry={}
    for p in [prior/'pilot', *list((prior/'published').glob('*/task-*/output'))]:
        s=load(p/'summary.json')
        if s['earliest_entry_year']==2012:
            assert s['status']=='COMPLETE'
            for name,digest in load(p/'hashes.json').items():assert sha(p/name)==digest
            raw_registry[s['year']]=p
    assert set(raw_registry)==set(YEARS)
    for year in YEARS:
        src=source()/f'input/folds/{year}';data=arrays(src/'train_PRIVATE.npz')
        mp=history(year)/f'evaluation_input/{year}_fit_metadata_PRIVATE.parquet';meta=pd.read_parquet(mp)
        np.testing.assert_array_equal(meta.event,data['event']);np.testing.assert_allclose(meta.time,data['time'],atol=1e-12,rtol=0)
        tr,_=filter_history(data,meta,2012,year)
        for k in tr:np.testing.assert_array_equal(tr[k],data[k])
        current[year]=data;metas[year]=meta
        assert tr['X'].shape[1]==74
        fold=inp/f'folds/{year}'
        for name in ['train_PRIVATE.npz','eval_PRIVATE.npz']:copy(src/name,fold/name,hashes)
        assert set(arrays(fold/'eval_PRIVATE.npz'))=={'X','sample_ids'}
        hashes[f'{year}_metadata']=sha(mp)
        for seed in SEEDS:
            ref=source()/f'input/{year}/{seed}/shrunk'
            p=next(x for x in pointer['saved_models'] if x['test_year']==year and x['seed']==seed)
            for name,key in [('forest_PRIVATE.json','forest_sha256'),('predictions_PRIVATE.npz','predictions_sha256')]:
                assert sha(ref/name)==p[key]
                copy(ref/name,inp/f'reference/{year}/{seed}'/name,hashes)
            raw=raw_registry[year]/str(seed)/'raw/forest_PRIVATE.json'
            assert load(raw)['config']==json.loads(json.dumps(asdict(configuration(seed))))
            copy(raw,inp/f'raw/{year}/{seed}/forest_PRIVATE.json',hashes)
    keep=kept_columns(load(HISTORY/'input/folds/2023/plan.json')['feature_names'])
    roll=Path('/home/oliver/MainDiskData/xai_rsf_temporal_leaf_20261028__kvhg19v')
    tr=current[2023];meta=metas[2023]
    for year in HISTORICAL:
        cut=pd.Timestamp(year,1,1)-pd.Timedelta(days=1)
        fit=meta.origin.dt.year.to_numpy()<year;val=meta.origin.dt.year.to_numpy()==year
        times=meta.time.to_numpy().copy();events=meta.event.to_numpy().copy();after=(meta.stop>cut).to_numpy()
        times[after]=(cut-meta.origin[after]).dt.days.to_numpy();events[after]=0
        derived={k:a[fit].copy() for k,a in tr.items()};derived['time']=times[fit];derived['event']=events[fit]
        derived['analysis_time']=np.array([calendar_elapsed(o,o+pd.Timedelta(days=int(t))) for o,t in zip(meta.origin[fit],times[fit])])
        old=roll/f'input/validation/{year}'
        saved=reduced(arrays(old/'train_PRIVATE.npz'),keep)
        assert set(saved)==set(derived)
        for k in derived:np.testing.assert_array_equal(derived[k],saved[k])
        ev=dict(X=tr['X'][val],sample_ids=tr['sample_ids'][val]);old_ev=reduced(arrays(old/'eval_PRIVATE.npz'),keep)
        for k in ev:np.testing.assert_array_equal(ev[k],old_ev[k])
        fold=inp/f'folds/{year}';fold.mkdir(parents=True)
        np.savez_compressed(fold/'train_PRIVATE.npz',**saved);np.savez_compressed(fold/'eval_PRIVATE.npz',**ev)
        hashes[f'historical_original83_{year}']=sha(old/'train_PRIVATE.npz')
        current[year]=saved
    # Selection labels come from each OUTER cutoff's training snapshot only.
    # Full entry cohorts must have completed the nominal calendar horizon.
    support=[]
    for outer in YEARS:
        tr=current[outer];meta=metas[outer]
        for h in [12,24]:
            for v in range(2020,outer):
                mask=meta.origin.dt.year.to_numpy()==v
                if not mask.any():continue
                mature=meta.origin[mask].map(lambda o:o+pd.DateOffset(months=h))<pd.Timestamp(outer,1,1)
                if not mature.all():continue
                labels={k:a[mask].copy() for k,a in tr.items()}
                ev=arrays(inp/f'folds/{v}/eval_PRIVATE.npz')
                np.testing.assert_array_equal(ev['sample_ids'],labels['sample_ids']);np.testing.assert_array_equal(ev['X'],labels['X'])
                target=run()/f'selection_input/{outer}/{h}';target.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(target/f'{v}_labels_PRIVATE.npz',**labels)
                e=labels['event'];t=labels['time'];hh=labels[f'h{h}'];known=(e>0)|(t>=hh)
                support.append(dict(outer_cutoff=outer,months=h,validation_entry_year=v,companies=len(e),
                    termination=int(((e==1)&(t<=hh)).sum()),recovery=int(((e==2)&(t<=hh)).sum()),
                    unknown=int((~known).sum()),snapshot_sha256=sha(inp/f'folds/{outer}/train_PRIVATE.npz')))
    for year,tr in current.items():
        audit.append(dict(year=year,companies=len(tr['event']),termination=int((tr['event']==1).sum()),
            recovery=int((tr['event']==2).sum()),censored=int((tr['event']==0).sum()),features=tr['X'].shape[1]))
    pd.DataFrame(audit).sort_values('year').to_csv(D/'TRAINING_SUPPORT.csv',index=False)
    pd.DataFrame(support).to_csv(D/'HISTORICAL_VALIDATION_SUPPORT.csv',index=False)
    write(inp/'configs.json',{str(s):asdict(configuration(s)) for s in SEEDS})
    write(D/'DATA_AUDIT.json',dict(input_hashes=hashes,current_pointer_sha256=sha(ROOT/'CURRENT_POST_ST_MODEL.json'),
        outer_evaluation_labels_read=False,historical_snapshot_reproduction_exact=True,
        source_raw_forests=9,new_historical_fits_planned=9))
    print(json.dumps(dict(staged_years=list(ALL_YEARS),outer_labels_read=False)))

if __name__=='__main__':main()
