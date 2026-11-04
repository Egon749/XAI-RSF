"""Read-only reconstruction worker; no calls to a fitting interface."""
import os,socket,sys,time
from dataclasses import asdict
import pandas as pd
from crforest import CompetingRiskForest,CIFExpansion
from experiments.post_st_ablation_v1.verification import merge
from .common import *

def difference(a,b):
    return float(np.max(np.abs(a-b)))

def worker(year,version,first,last,out,root):
    train=arrays(root/f'folds/{year}/train_PRIVATE.npz'); ev=arrays(root/f'folds/{year}/eval_PRIVATE.npz')
    X=np.vstack([train['X'],ev['X']]); members={}; forests={}; explicit={}; summary=[]; masks=[]
    for idx in sorted(set(j for i in range(first,last) for j in TRIPLES[i])):
        d=root/str(year)/str(SEEDS[idx])/version
        f=CompetingRiskForest.load(d/'forest_PRIVATE.json')
        member=CIFExpansion.from_forest(f,train['X']); stored=CIFExpansion.load(d/'functions_PRIVATE.npz')
        for field in ('mass','features','missing','high','low','mu','coefficients','grid'):
            np.testing.assert_array_equal(getattr(member,field),getattr(stored,field))
        times=np.unique(np.r_[0.,f.grid_,TIMES]); forecasts=f.predict(X,times)
        np.testing.assert_allclose(f.predict(ev['X'],TIMES),arrays(d/'predictions_PRIVATE.npz')['cif'],atol=1e-12,rtol=0)
        components=np.broadcast_to(stored._coeff(times)[1],forecasts.shape).copy()
        for key in [(v,) for v in stored.mains]+stored.pairs:
            components+=stored.component(X,key,times)
        assert difference(forecasts,components)<1e-10
        members[idx]=f; forests[idx]=forecasts; explicit[idx]=components
    for i in range(first,last):
        ids=TRIPLES[i];name=NAMES[i]
        start=time.perf_counter(); f=merge([members[j] for j in ids]);ex=CIFExpansion.from_forest(f,train['X'])
        compile_time=time.perf_counter()-start
        path=out/(name+'_PRIVATE.npz');ex.save(path);ex2=CIFExpansion.load(path)
        start=time.perf_counter(); direct=f.predict(X,times)
        errs=dict(member_mean=difference(direct,np.mean([forests[j] for j in ids],axis=0)),
                  reconstruction=difference(direct,ex.predict(X,times)),reload=difference(direct,ex2.predict(X,times)),
                  explicit_sum=difference(direct,np.mean([explicit[j] for j in ids],axis=0)))
        assert max(errs.values())<=1e-10 and f.audit_['max_path_variables']<=2
        lo,inc,hi=float(direct.min()),float(np.diff(direct,axis=1).min()),float(direct.sum(-1).max())
        assert lo>=-1e-12 and inc>=-1e-12 and hi<=1+1e-10
        verification=time.perf_counter()-start
        cx=ex.complexity(train['X']);terms=cx.pop('functions')
        masks.append([name,[','.join(map(str,t['features'])) for t in terms if t['nonzero']]])
        summary.append(dict(year=year,version=version,model=name,trees=96,rows=len(X),times=len(times),native_times=len(ex.grid),
                            max_path=f.audit_['max_path_variables'],probability_min=lo,min_increment=inc,max_sum=hi,**errs,**cx,
                            compile_seconds=compile_time,verification_seconds=verification,function_bytes=path.stat().st_size,
                            function_sha256=sha(path)))
    pd.DataFrame(summary).to_csv(out/'verify.csv',index=False);dump(out/'masks.json',masks)

if __name__=='__main__':
    out=Path(os.environ['OUTPUT_DIR']);out.mkdir(exist_ok=True,parents=True)
    worker(int(sys.argv[1]),sys.argv[2],int(sys.argv[3]),int(sys.argv[4]),out,Path(os.environ['DATA_DIR']))
    dump(out/'done.json',dict(host=socket.gethostname(),job=os.environ.get('SLURM_JOB_ID'),args=sys.argv[1:]))
