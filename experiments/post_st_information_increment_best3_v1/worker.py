"""Fresh reduced-input fits, or an equal-tree ensemble of their saved shrunk predictors."""
import time,socket,os,sys
from dataclasses import asdict
from crforest import CompetingRiskForest
from crforest.shrinkage import shrink_leaf_cifs
from experiments.post_st_level_coverage_v1.worker import verify_export
from experiments.post_st_ablation_v1.verification import merge
from experiments.post_st_ablation_v1.common import pairing,array_hash
from experiments.post_st_cause_support_v1.verification import replay
from .common import *

def execute(stage,y,arm,variant,seed,root,out):
    gate=read(root/'admissibility.json');assert arm in fit_arms(gate)
    spec=pd.read_csv(root/'features.csv');keep=spec[arm].to_numpy(bool)
    tr=arrays(root/f'folds/{y}/train_PRIVATE.npz');ev=arrays(root/f'folds/{y}/eval_PRIVATE.npz')
    assert set(ev)=={'X','sample_ids'}
    tr['X']=tr['X'][:,keep];ev['X']=ev['X'][:,keep];p=int(keep.sum());start=time.perf_counter()
    if stage=='fit':
        config=cfg(seed,p,variant);paired=pairing(tr['groups'],config,p)
        t=time.perf_counter();raw=CompetingRiskForest(config).fit(tr['X'],tr['time'],tr['event'],tr['groups'],n_jobs=1);duration=time.perf_counter()-t
        for actual,z in zip(raw.inbag_,paired):assert array_hash(actual)==z['multiplicity_sha256']
        shr=shrink_leaf_cifs(raw,tr['X'],tr['time'],tr['event'],tr['groups'],10)
        checks={k:verify_export(m,tr,ev,out/k) for k,m in [('raw',raw),('shrunk',shr)]}
        nodes,usage,_,leaves,other=replay(shr,tr)
        fields=pd.DataFrame(usage);fields['name']=spec[keep].english_name.to_numpy();fields['info']=spec[keep]['info'].to_numpy();fields.to_csv(out/'usage.csv',index=False)
        leaf=pd.DataFrame(leaves);leaf.to_csv(out/'leaves_PRIVATE.csv',index=False)
        dump(out/'bootstrap.json',paired)
        profile=dict(fit_seconds=duration,attempted=shr.candidate_evaluations_,scored=sum(n['scored_candidates'] for n in nodes),guard_rejected=sum(n['guard_rejected'] for n in nodes),leaf_company_median=float(leaf.unique_companies.median()),leaf_target_median=float(leaf.target_companies.median()),zero_target_leaves=int((leaf.target_companies==0).sum()))
    else:
        assert stage=='ensemble' and seed==0
        records=read(root/'members.json');paths=[Path(records[f'{y}|{arm}|{variant}|{s}'])/'shrunk' for s in SEEDS]
        members=[CompetingRiskForest.load(path/'forest_PRIVATE.json') for path in paths];f=merge(members)
        checks={'shrunk':verify_export(f,tr,ev,out/'shrunk')}
        pred=arrays(out/'shrunk/predictions_PRIVATE.npz');orig=[arrays(path/'predictions_PRIVATE.npz') for path in paths]
        for z in orig:np.testing.assert_array_equal(z['sample_ids'],pred['sample_ids'])
        np.testing.assert_allclose(np.mean([z['cif'] for z in orig],axis=0),pred['cif'],atol=1e-12,rtol=0)
        profile=dict(fit_seconds=0)
    dump(out/'summary.json',dict(stage=stage,year=y,arm=arm,variant=variant,seed=seed,features=p,pool=pool_size(p,variant),checks=checks,forest_fits=int(stage=='fit'),seconds=time.perf_counter()-start,host=socket.gethostname(),**profile))
    dump(out/'hashes.json',{str(f.relative_to(out)):sha(f) for f in out.rglob('*') if f.is_file()})
if __name__=='__main__':
    stage,y,arm,v,seed=sys.argv[1:];root=Path(os.environ['DATA_DIR']);out=Path(os.environ['OUTPUT_DIR']);out.mkdir(exist_ok=True,parents=True)
    execute(stage,int(y),arm,v,int(seed),root,out)
