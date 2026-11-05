"""Stage only own-cutoff training snapshots and evaluation X, never test labels."""
import shutil,tempfile
from dataclasses import asdict
from experiments.post_st_paper_features_v1.common import history
from .common import *
def main():
    parent=Path(load(D.parent/'post_st_shrinkage_fine_v1/run_location.json')['private_run'])
    r=Path(tempfile.mkdtemp(prefix='shrinkage_stability_v1_',dir=parent.parent))
    write(D/'run_location.json',dict(private_run=str(r)))
    inp=r/'input';inp.mkdir();hashes={};audit=[]
    pointer=load(ROOT/'CURRENT_POST_ST_MODEL.json')
    assert pointer['arm']=='C039' and pointer['configuration']['seeds']==list(SEEDS)
    for year in YEARS:
        fold=inp/f'folds/{year}';fold.mkdir(parents=True)
        for name in ['train_PRIVATE.npz','eval_PRIVATE.npz']:
            src=parent/f'input/folds/{year}/{name}';shutil.copyfile(src,fold/name)
            hashes[str((fold/name).relative_to(inp))]=sha(src)
        tr=arrays(fold/'train_PRIVATE.npz');ev=arrays(fold/'eval_PRIVATE.npz')
        assert set(ev)=={'X','sample_ids'} and not set(tr['sample_ids'])&set(ev['sample_ids'])
        mp=history(year)/f'evaluation_input/{year}_fit_metadata_PRIVATE.parquet'
        meta=pd.read_parquet(mp)
        checked,_=filter_history(tr,meta,2012,year)
        for k in tr:np.testing.assert_array_equal(tr[k],checked[k])
        np.testing.assert_array_equal(meta.event,tr['event'])
        np.testing.assert_allclose(meta.time,tr['time'],atol=1e-12,rtol=0)
        audit.append(dict(year=year,training_companies=len(tr['X']),features=tr['X'].shape[1],
          termination=int((tr['event']==1).sum()),recovery=int((tr['event']==2).sum()),
          evaluation_companies=len(ev['X']),metadata_sha256=sha(mp),
          pre_cutoff=True,training_cap_months=24,train_sha256=sha(fold/'train_PRIVATE.npz')))
    write(inp/'configs.json',{str(s):asdict(configuration(s)) for s in SEEDS})
    hashes['configs.json']=sha(inp/'configs.json')
    write(D/'FROZEN_CONFIGS.json',load(inp/'configs.json'))
    pd.DataFrame(audit).to_csv(D/'DATA_AUDIT.csv',index=False)
    write(D/'INPUT_AUDIT.json',dict(source_run=str(parent),input_sha256=hashes,
      pointer_sha256=sha(ROOT/'CURRENT_POST_ST_MODEL.json'),evaluation_labels_staged=False,
      fixed_original_training_reference=True,source_vintage='conservative C; inherited PIT limitations'))
    print(r)
if __name__=='__main__':main()

