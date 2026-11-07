"""Stage only own-cutoff historical outcomes and label-free outer X for workers."""
import shutil
import pandas as pd
from experiments.post_st_paper_features_v1.common import history
from experiments.post_st_eswa_baselines_v1.common import Horizon
from experiments.post_st_prognosis_v1.common import endpoint_weights
from .common import *

def main():
    r=run();inp=r/'input';inp.mkdir();src=source()/'input';hashes={};records=[]
    cols=pd.read_csv(ROOT/'experiments/post_st_grouped_explanation_v1/FEATURE_GROUP_MAPPING.csv').sort_values('input_position').original_feature_id.to_numpy()
    assert len(set(cols))==74
    expected=read(ROOT/'experiments/post_st_seed_explanation_stability_v1/MODEL_HASH_AUDIT.json')['before']['models']
    for y in YEARS:
        dest=inp/str(y);dest.mkdir()
        tr=arrays(src/f'folds/{y}/train_PRIVATE.npz');ev=arrays(src/f'folds/{y}/eval_PRIVATE.npz')
        assert tr['X'].shape[1]==ev['X'].shape[1]==74 and set(ev)=={'X','sample_ids'}
        assert len(set(tr['groups']))==len(tr['X'])
        orig=arrays(history(y)/f'input/folds/{y}/train_PRIVATE.npz')
        np.testing.assert_array_equal(orig['sample_ids'],tr['sample_ids']);np.testing.assert_array_equal(orig['X'][:,cols],tr['X'])
        for k in ('time','event','groups','h12','h24','analysis_time'):np.testing.assert_array_equal(tr[k],orig[k])
        mp=history(y)/f'evaluation_input/{y}_fit_metadata_PRIVATE.parquet';m=pd.read_parquet(mp)
        np.testing.assert_array_equal(m.sample_id,tr['sample_ids'])
        assert (m.origin<pd.Timestamp(y,1,1)).all() and (m.stop<pd.Timestamp(y,1,1)).all()
        for col in ('financial_available','market_available'):
            assert (m[col].isna()|(m[col]<m.origin)).all()
        hashes[f'{y}/metadata']=sha(mp)
        for role in ('train','eval'):
            p=src/f'folds/{y}/{role}_PRIVATE.npz';shutil.copyfile(p,dest/(role+'.npz'));hashes[f'{y}/{role}']=sha(p)
        for s in SEEDS:
            folder=src/f'{y}/{s}/shrunk'
            for fn,hh in expected[f'{y}/{s}/shrunk'].items():
                assert sha(folder/fn)==hh;hashes[f'{y}/{s}/{fn}']=hh
        folds=[('outer',tr)]
        for v in (y-3,y-2):
            train=recensor(tr,m,v);valid=(m.origin.dt.year==v).to_numpy();val={k:z[valid] for k,z in tr.items()}
            assert not set(train['groups'])&set(val['groups'])
            np.savez_compressed(dest/f'{v}_fit.npz',**train);np.savez_compressed(dest/f'{v}_val.npz',**val)
            folds.extend([(f'{v}_fit',train),(f'{v}_val',val)])
        for role,d in folds:
            for h in (12,24):
                hd=Horizon(d,h,'IPCW');ix=np.arange(len(d['X']));kk,yy,ww=hd.weighted(ix,ix)
                w,known,_=endpoint_weights(hd.t,hd.e,h,d['label_eligible'])
                np.testing.assert_array_equal(hd.y>=0,known);np.testing.assert_allclose(ww,w[known],rtol=1e-14)
                records.append(dict(year=y,partition=role,months=h,companies=len(ix),known=len(kk),delisting=int(sum(yy)),recovery=int(sum((hd.e==2)&(hd.t<=h))),early_censored=int(sum(~known)),max_weight=float(ww.max())))
    write(HERE/'SOURCE_HASHES.json',hashes)
    pd.DataFrame(records).to_csv(HERE/'TRAINING_SUPPORT.csv',index=False)
    write(r/'source_locations_PRIVATE.json',dict(source=str(source()),metadata=[str(history(y)) for y in YEARS]))

if __name__=='__main__':main()
