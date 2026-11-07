import os,sys,time,socket,resource
import numpy as np
import xgboost as xgb
from experiments.post_st_c061_xgboost_v1.pipeline import fit,predict
from .common import *

def main(y,h,batch):
    start=time.perf_counter();inp=Path(os.environ['DATA_DIR'])/str(y);out=Path(os.environ['OUTPUT_DIR']);out.mkdir(parents=True,exist_ok=True)
    tr=arrays(inp/'train.npz');ev=arrays(inp/'eval.npz');assert set(ev)=={'X','sample_ids'}
    c=config(y,h);pp={};train_pp={};rows=[]
    for seed in seed_batches()[batch]:
        model,info=fit(tr,h,c,seed);p=predict(model,ev['X']);tp=predict(model,tr['X'])
        path=out/f'model_{seed}.ubj';model.save_model(path);reloaded=xgb.Booster();reloaded.load_model(path)
        np.testing.assert_array_equal(p,predict(reloaded,ev['X']));np.testing.assert_array_equal(tp,predict(reloaded,tr['X']))
        pp[str(seed)]=p;train_pp[str(seed)]=tp
        rows.append(dict(seed=seed,**c,**info,reload_error=0.,model_sha256=sha(path),model_bytes=path.stat().st_size))
    pp['ensemble']=np.mean([pp[str(s)] for s in seed_batches()[batch]],axis=0)
    np.savez_compressed(out/'predictions_PRIVATE.npz',sample_ids=ev['sample_ids'],**pp)
    np.savez_compressed(out/'training_predictions_PRIVATE.npz',sample_ids=tr['sample_ids'],**train_pp)
    write(out/'result.json',dict(year=y,months=h,batch=batch,models=rows,final_fits=len(rows),tuning_fits=0,seconds=time.perf_counter()-start,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,hostname=socket.gethostname(),job_id=os.environ.get('SLURM_JOB_ID'),inputs={p.name:sha(p) for p in inp.glob('*.npz')},xgboost=xgb.__version__))

if __name__=='__main__':main(*map(int,sys.argv[1:]))
