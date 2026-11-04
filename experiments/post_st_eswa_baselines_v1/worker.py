import time,resource,socket,sys
import pandas as pd
from .common import *
def main():
    arm,y,h,mode=sys.argv[1:];y,h=int(y),int(h);root=Path(os.environ['DATA_DIR']);out=Path(os.environ['OUTPUT_DIR']);out.mkdir(exist_ok=True,parents=True)
    folder=root/str(y);tr=arrays(folder/'train.npz');ev=arrays(folder/'eval.npz');assert set(ev)=={'X','sample_ids'} and tr['X'].shape[1]==74
    from . import proboost,stacking
    hd=Horizon(tr,h,mode);start=time.perf_counter();p,logs,info=(proboost if arm=='PROBoost' else stacking).execute(hd,ev['X'],out);elapsed=time.perf_counter()-start
    assert p.shape==(len(ev['X']),) and np.isfinite(p).all() and p.min()>=0 and p.max()<=1
    np.savez_compressed(out/'predictions.npz',ids=ev['sample_ids'],score=p)
    pd.DataFrame(logs).to_csv(out/'training.csv',index=False)
    dump(out/'result.json',dict(arm=arm,year=y,months=h,mode=mode,train=len(tr['X']),known=int(sum(hd.y>=0)),events=int(sum(hd.y==1)),score_rows=len(p),elapsed=elapsed,peak_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,host=socket.gethostname(),job=os.environ.get('SLURM_JOB_ID'),**info))
if __name__=='__main__':main()
