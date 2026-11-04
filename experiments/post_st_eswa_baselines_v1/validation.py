"""Check reference CIF probability, exact representation and untouched archive from the prior study."""
import time
import pandas as pd
from crforest import CompetingRiskForest,CIFExpansion
from experiments.post_st_benchmark_calibration_v1.models import stored
from experiments.post_st_benchmark_calibration_v1.common import endpoints
from experiments.post_st_paper_features_v1.common import history
from .common import *
def main():
    d=old('post_st_seed_explanation_stability_v1')/'input';g=old('post_st_grouped_explanation_v1')/'input';rows=[];maturity=[];r=run()
    prior=pd.read_csv(ROOT/'experiments/post_st_benchmark_calibration_v1/COMPLEXITY.csv');verified=load(ROOT/'experiments/post_st_delisting_cause_audit_v1/MODEL_INTEGRITY_AUDIT.json')['hashes']
    for y in YEARS:
        tr=arrays(d/f'folds/{y}/train_PRIVATE.npz');ev=arrays(d/f'folds/{y}/eval_PRIVATE.npz');X=np.vstack([tr['X'],ev['X']]);ref,_=stored(y)
        for name,folders,file in [('C039',[d/f'{y}/{s}/shrunk' for s in (17,503,22)],g/f'{y}/C039.npz'),('anchor',[d/f'{y}/reference96/shrunk'],d/f'{y}/reference96/shrunk/functions_PRIVATE.npz')]:
            fs=[CompetingRiskForest.load(f/'forest_PRIVATE.json') for f in folders];ex=CIFExpansion.load(file);ts=np.unique(np.r_[0,ex.grid,DAYS]);z=np.mean([f.predict(X,ts) for f in fs],axis=0);rec=ex.predict(X,ts)
            start=time.perf_counter();p=np.mean([f.predict(ev['X'],DAYS) for f in fs],axis=0);pt=time.perf_counter()-start
            err=float(abs(z-rec).max());assert err<1e-10 and z.min()>=-1e-12 and np.diff(z,axis=1).min()>=-1e-12 and z.sum(2).max()<=1+1e-10
            np.testing.assert_allclose(p,ref['XAI/'+name],atol=1e-12,rtol=0)
            if name=='C039':
                assert sha(file)==verified[f'{y}/C039/functions']
                np.testing.assert_array_equal(p,ref['XAI/C039'])
                for s,folder in zip((17,503,22),folders):
                    for f in ['forest_PRIVATE.json','functions_PRIVATE.npz','predictions_PRIVATE.npz']:assert sha(folder/f)==verified[f'{y}/{s}/shrunk/{f}']
            row=prior.query("year==@y and model==@a",local_dict={'y':y,'a':'XAI/'+name}).iloc[0]
            rows.append(dict(year=y,model='XAI/'+name,exact_error=err,min_cif=float(z.min()),min_increment=float(np.diff(z,axis=1).min()),max_joint_sum=float(z.sum(2).max()),native_times=len(ex.grid),function_sha256=sha(file),model_bytes=sum((f/'forest_PRIVATE.json').stat().st_size for f in folders),function_bytes=file.stat().st_size,forecast_seconds_joint_H12_H24=pt,historical_fit_seconds=row.historical_fit_seconds,main_functions=row.observed_main,pair_functions=row.observed_pair,spatial_cells=row.cells,new_fits=0))
        meta=pd.read_parquet(history(y)/f'evaluation_input/{y}_fit_metadata_PRIVATE.parquet')
        for h in (12,24):
            mature=(meta.origin+pd.DateOffset(months=h)<pd.Timestamp(y,1,1)).to_numpy();hd=Horizon(tr,h,'IPCW')
            maturity.append(dict(year=y,months=h,training=len(mature),calendar_window_mature_by_cutoff=int(mature.sum()),calendar_immature=int(sum(~mature)),known_despite_calendar_immaturity=int(sum(~mature&(hd.y>=0))),unknown=int(sum(hd.y<0)),training_end=str(meta.stop.max().date())))
    pd.DataFrame(rows).to_csv(D/'REFERENCE_VALIDATION.csv',index=False);pd.DataFrame(maturity).to_csv(D/'MATURITY.csv',index=False)
if __name__=='__main__':main()
