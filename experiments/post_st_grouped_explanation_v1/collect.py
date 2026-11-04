"""Assemble member term caches on the common frozen variable dictionary."""
import shutil
from .common import *
def main():
    r=run();records=read(r/'verify/published.json');target=r/'analysis_input';target.mkdir()
    for folder in ['folds','evaluation_input','cases']:shutil.copytree(r/'input'/folder,target/folder)
    rows=[];files={}
    for z in records:
        p=Path(z['path']);y=int(z['args'][1]);df=pd.read_csv(p/'exact.csv')
        rows.append(df)
        for name in df.model.astype(str):
            if name in ARMS:files[y,name]=p/(name+'_PRIVATE.npz')
    f=pd.concat(rows,ignore_index=True)
    assert len(f)==396 and not f.duplicated(['year','model']).any()
    f.to_csv(D/'EXACT_MODEL_CHECKS.csv',index=False)
    dump(D/'EXACT_RECONSTRUCTION_AUDIT.json',dict(models=len(f),annual_memberships=132,versions=['shrunk'],max_errors=f[['original_sum','A_plus_I','sum_C','saved_function']].max().to_dict(),min_CIF=float(f.cif_min.min()),min_increment=float(f.min_increment.min()),max_cause_sum=float(f.max_cause_sum.max()),all_rows_times_causes=True))
    for y in YEARS:
        members=[arrays(files[y,s]) for s in ARMS]
        keys=sorted({(j,-1) for j in range(74)} | {tuple(k) for a in members for k in a['keys']})
        index={k:i for i,k in enumerate(keys)}
        vals=np.zeros((len(ARMS),*members[0]['terms'].shape[:-1],len(keys)))
        for i,a in enumerate(members):
            vals[i][...,[index[tuple(k)] for k in a['keys']]]=a['terms']
        np.save(target/f'{y}_terms.npy',vals);np.save(target/f'{y}_keys.npy',np.array(keys))
        np.save(target/f'{y}_base.npy',np.stack([a['baseline'] for a in members]))
        # Verify all-member mean against the separate 320-tree export on every cached coordinate.
        np.testing.assert_allclose(vals[:10].mean(0),vals[11],atol=1e-10,rtol=0)
if __name__=='__main__':main()
