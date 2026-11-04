import shutil
import pandas as pd
from .common import *
from experiments.post_st_paper_features_v1.common import history
from experiments.post_st_prognosis_v1.common import calendar_elapsed
def main():
    r=run(); inp=r/'input';inp.mkdir(exist_ok=True)
    src=old('post_st_seed_explanation_stability_v1')/'input'; hashes={};support=[]
    for y in YEARS:
        folder=inp/str(y);folder.mkdir(exist_ok=True);tr=arrays(src/f'folds/{y}/train_PRIVATE.npz');ev=arrays(src/f'folds/{y}/eval_PRIVATE.npz')
        mpath=history(y)/f'evaluation_input/{y}_fit_metadata_PRIVATE.parquet';m=pd.read_parquet(mpath);np.testing.assert_array_equal(m.sample_id,tr['sample_ids'])
        assert (m.stop<pd.Timestamp(y,1,1)).all() and (m.origin<pd.Timestamp(y,1,1)).all() and len(set(tr['groups']))==len(m)
        for role in ['eval','train']:
            p=src/f'folds/{y}/{role}_PRIVATE.npz';shutil.copyfile(p,folder/(role+'.npz'));hashes[f'{y}/{role}']=sha(p)
        hashes[f'{y}/metadata']=sha(mpath)
        cfg=read(src/f'{y}/17/shrunk/forest_PRIVATE.json')['config'];write(folder/'config.json',cfg)
        for v in (y-3,y-2):
            before=m.origin<pd.Timestamp(v,1,1);valid=m.origin.dt.year.eq(v);cap=pd.Timestamp(v,1,1)-pd.Timedelta(days=1)
            stop=m.stop.where(m.stop<=cap,cap);a={k:z[before] for k,z in tr.items()};a['event']=np.where(m.stop[before]<=cap,tr['event'][before],0);a['time']=(stop[before]-m.origin[before]).dt.days.to_numpy()
            a['analysis_time']=np.array([calendar_elapsed(o,s) for o,s in zip(m.origin[before],stop[before])]);b={k:z[valid] for k,z in tr.items()}
            assert a['time'].min()>0 and not set(a['groups'])&set(b['groups']) and (m.origin[valid]+pd.DateOffset(months=12)<pd.Timestamp(y,1,1)).all()
            np.savez_compressed(folder/f'{v}_fit.npz',**a);np.savez_compressed(folder/f'{v}_val.npz',**b)
            support.append(dict(outer=y,validation_year=v,internal_train=len(a['X']),validation=len(b['X']),train_delist=int(sum(a['event']==1)),train_recovery=int(sum(a['event']==2)),validation_delist_H12=int(sum((b['event']==1)&(b['time']<=b['h12'])))))
    write(r/'source_hashes.json',hashes);pd.DataFrame(support).to_csv(D/'FORWARD_FOLDS.csv',index=False)
if __name__=='__main__':main()
