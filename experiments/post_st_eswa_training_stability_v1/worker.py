import os,sys,time,socket,hashlib,traceback
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.post_st_eswa_baselines_v1 import proboost,stacking
from experiments.post_st_eswa_baselines_v1.common import arrays,dump
from .resampling import BootstrapHorizon,BootstrapSearch,draw_rows
def main(year,h,rep,model):
    root=Path(os.environ['DATA_DIR']);out=Path(os.environ['OUTPUT_DIR']);out.mkdir(parents=True,exist_ok=True)
    tr=arrays(root/str(year)/'train.npz');ev=arrays(root/str(year)/'eval.npz');assert set(ev)=={'sample_ids','X'}
    rows=draw_rows(year,rep,len(tr['X']));oob=np.setdiff1d(np.arange(len(rows)),rows);hd=BootstrapHorizon(tr,h,rows)
    unique=np.unique(rows,return_index=True)[1];n_pos=int(np.sum(hd.y[unique]==1));n_neg=int(np.sum(hd.y[unique]==0))
    rec=dict(year=year,months=h,rep=rep,model=model,training_rows=len(rows),training_companies=len(unique),oob_companies=len(oob),train_distinct_positive=n_pos,train_distinct_negative=n_neg,draw_sha256=hashlib.sha256(rows.astype('<i8').tobytes()).hexdigest(),job=os.environ.get('SLURM_JOB_ID'),host=socket.gethostname())
    np.savez_compressed(out/'draw.npz',rows=rows,oob=oob);start=time.perf_counter()
    if min(n_pos,n_neg)<2:
        rec.update(status='unsupported',reason='fewer than two distinct known positive or negative companies for required outer cross-validation',fit_count=0)
    else:
        try:
            x=np.vstack([tr['X'][oob],ev['X']])
            if model=='PROBoost':p,logs,info=proboost.execute(hd,x,out)
            else:p,logs,info=stacking.execute(hd,x,out,search_class=BootstrapSearch)
            assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
            df=pd.DataFrame(logs);df.to_csv(out/'pipeline_log.csv',index=False);dump(out/'fit_info.json',info)
            fits=df.stage.isin(['IPSO_proxy_fit','Hyperopt_fit','base_refit','GA_meta_fit','final_meta_fit','PROBoost_tune','PROBoost_final'])
            rec.update(fit_count=int(sum(fits)),status='ok',selected_columns=len(info['selected']),reload_error=info['reload_error'])
            if model=='PROBoost':
                j=int(df.loc[df.stage=='PROBoost_final','candidate'].iloc[0]);rec.update(selected_config=j,selected_cv_mean_ap=float(df.loc[(df.stage=='PROBoost_tune')&(df.candidate==j),'score'].mean()))
            else:
                rec.update(deployment_GA_F1=None,avg_deployment_base_F1=None)
                # plan() logs GA just before the deployment tuned_base(), followed by the final plan record.
                events=df[df.stage.isin(['GA_search','GA_support_fallback'])]
                if len(events) and events.iloc[-1].stage=='GA_search':rec['deployment_GA_F1']=float(events.iloc[-1].score)
                import pickle
                with (out/'stack.pkl').open('rb') as f:bundle,_=pickle.load(f)
                if bundle['fold_F1']:rec['avg_deployment_base_F1']=float(np.mean(bundle['fold_F1']))
                aug=df[df.synthetic>0];assert (aug.distinct_positive_firms>=6).all() and (aug.synthetic<=aug.distinct_positive_firms).all()
            np.savez_compressed(out/'scores.npz',oob=p[:len(oob)],test=p[len(oob):],oob_ids=tr['sample_ids'][oob],test_ids=ev['sample_ids'])
        except Exception as exc:
            (out/'failure.txt').write_text(traceback.format_exc());rec.update(status='fit_failed',reason=type(exc).__name__+': '+str(exc)[:180],fit_count=None)
    rec['elapsed_seconds']=time.perf_counter()-start;dump(out/'done.json',rec)
if __name__=='__main__':
    y,h,r,m=sys.argv[1:];main(int(y),int(h),int(r),m)
