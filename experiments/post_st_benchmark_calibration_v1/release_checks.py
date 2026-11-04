"""Read-only independent audit of historical predictions, numerical bounds and task publication."""
import pickle,subprocess
import pandas as pd
from .models import *
from .estimators import transform,regression_predict
def main():
    r=run();pub=[];prob=[]
    for (arm,y,seed),(p,j) in jobs().items():
        manifest=read(p.parent/'complete.json')['manifest']
        for entry in manifest:
            if len(entry)==2:
                assert entry[1]=='directory' and (p/entry[0]).is_dir();continue
            file,size,digest=entry
            assert (p/file).stat().st_size==size and sha(p/file)==digest
        pub.append(dict(year=y,arm=arm,seed=seed,host=j['host'],published_files=sum(len(entry)==3 for entry in manifest),private_prediction_sha256=sha(p/'predictions.npz'),complete_manifest_sha256=sha(p.parent/'complete.json')))
        if arm!='FG':continue
        tr=arrays(r/f'input/{y}/train.npz');ev=arrays(r/f'input/{y}/eval.npz');_,test,_=transform(tr['X'],ev['X'])
        with (p/'outer_0/model.pkl').open('rb') as stream:fit=pickle.load(stream)
        z=arrays(p/'predictions.npz')['0'];lab=arrays(old('post_st_seed_explanation_stability_v1')/f'input/evaluation_input/{y}_test_PRIVATE.npz')
        for h in (12,24):
            values=endpoints(z,lab[f'h{h}']).sum(1);prob.append(dict(year=y,months=h,companies=len(values),joint_sum_exceeds_one=int(sum(values>1+1e-10)),maximum_joint_sum=float(values.max())))
        ts=np.unique(np.r_[DAYS,*[m['time'] for m in fit]]);v=regression_predict(fit,test,ts,True);assert np.diff(v,axis=1).min()>=-1e-12 and v.max()<=1 and v.min()>=0
    pd.DataFrame(pub).to_csv(D/'PUBLICATION_CHECKS.csv',index=False);pd.DataFrame(prob).to_csv(D/'FG_JOINT_PROBABILITY.csv',index=False)
    # Verify identity against the previously scored combined C039 four-cell comparison.
    commit='614eb0be80ff5f7b8377ce7b4041cb884ebf98db'
    ext=Path(json.loads(subprocess.check_output(['git','show',commit+':experiments/post_st_information_increment_best3_v1/run_location.json'],cwd=ROOT))['private_run'])
    table=pd.read_parquet(ext/'rankings_PRIVATE.parquet');mx=0.;n=0
    for y in YEARS:
        z=stored(y)[0]['XAI/C039'];lab=arrays(old('post_st_seed_explanation_stability_v1')/f'input/evaluation_input/{y}_test_PRIVATE.npz')
        for h in (12,24):
            for k in (1,2):
                rows=table.query("year==@y and months==@h and cause==@k and arm=='M3' and variant=='fixed25' and seed==0").sort_values('row')
                if not len(rows):assert (y,h)==(2025,24);continue
                np.testing.assert_array_equal(rows.sample_id,lab['sample_ids']);v=endpoints(z,lab[f'h{h}'])[:,k-1];err=float(np.max(abs(rows.risk-v)));mx=max(mx,err);n+=len(v)
    assert mx<=1e-12
    frozen=pd.read_csv(ROOT/'experiments/post_st_seed_explanation_stability_v1/ALL_METRICS.csv',dtype={'scope':str})
    original=frozen.query("model=='C039' and version=='shrunk' and supported").set_index(['scope','months','cause'])
    new=pd.read_csv(D/'METRICS_ALL.csv',dtype={'scope':str}).query("model=='XAI/C039' and supported").set_index(['scope','months','cause']).loc[original.index]
    for lhs,rhs in [('auc','auc'),('ap','ap'),('brier','ipcw_brier'),('within_year_auc','within_year_auc'),('captured','top20_captured'),('risk_mean','mean_predicted')]:
        np.testing.assert_allclose(original[lhs],new[rhs],rtol=0,atol=1e-12,equal_nan=True)
    write(D/'SOURCE_CHECKS.json',dict(historical_c039_saved_combined_source=commit,combined_checked_row_endpoints=n,combined_max_difference=mx,original_metric_cells=len(original)*6,workers_all_output_hashes_checked=True,snapshot_hashes=read(r/'source_hashes.json')))
if __name__=='__main__':main()
