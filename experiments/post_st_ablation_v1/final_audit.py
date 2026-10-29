"""Final cross-artifact pairing, counters, environments and delivery validation."""
from pathlib import Path
from dataclasses import asdict
import argparse,json,os,subprocess,shutil
import numpy as np
import pandas as pd
from crforest import CompetingRiskForest,cluster_bootstrap
from .common import *
from .stages import validation_outputs

def audit(run):
    out=run/'evaluation';reg=registry(run);models={};hashes=[];cost=[]
    # Independent saved-RSF reload/CIF verification, never a reference refit.
    from experiments.post_st_prognosis_v1.worker import RSCRIPT,RLIB
    rp=run/'private/standard_rsf_check'
    if not rp.exists():
        rp.mkdir();write(rp/'manifest.json',[r for r in reg if r['arm']=='standard_rsf'])
        env=os.environ.copy();env.update(EW_RLIB=RLIB,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        with (run/'audit/standard_rsf_validation.log').open('x') as log:
            subprocess.run([RSCRIPT,'--vanilla',str(Path(__file__).with_name('verify_rsf.R')),str(rp/'manifest.json'),str(rp)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    refs=pd.read_csv(rp/'standard_rsf_validation.csv')
    assert len(refs)==6 and refs.new_fits.sum()==0 and refs.source_prediction_error.max()<=1e-12 and refs.reload_error.max()<=1e-12
    assert refs.min_cif.min()>=-1e-12 and refs.max_cause_sum.max()<=1+1e-12 and refs.min_increment.min()>=-1e-12
    if not (out/'standard_rsf_validation.csv').exists():shutil.copy2(rp/'standard_rsf_validation.csv',out/'standard_rsf_validation.csv')
    for r in reg:
        p=Path(r['path']);a=r['arm'];y=r['year'];s=r['seed'];su=json.loads((p/'summary.json').read_text())
        forest=p/('forest_PRIVATE.rds' if a=='standard_rsf' else 'forest_PRIVATE.json')
        hashes.append(dict(year=y,arm=a,seed=s,reused=r['reuse'],model_sha256=sha(forest),predictions_sha256=sha(p/'predictions_PRIVATE.npz'),model_bytes=forest.stat().st_size))
        cost.append(dict(year=y,arm=a,seed=s,reused=r['reuse'],host=su.get('host'),
            forest_fit_seconds=su.get('forest_fit_seconds',su.get('base_fit_seconds',su.get('fit_seconds'))),
            shrinkage_seconds=su.get('shrinkage_seconds',su.get('leaf_update_seconds')),
            prediction_seconds=su.get('prediction_seconds'),source_total_seconds=su.get('seconds'),source_peak_rss_MiB=su.get('peak_rss_MiB'),
            interpretation='historical wall times may be absent or have different hardware/accounting; no timing refits' if r['reuse'] else 'raw and shrunk share one fit; do not sum forest_fit_seconds twice'))
        tr=arrays(run/'input/folds'/str(y)/'train_PRIVATE.npz')
        if a=='standard_rsf':
            expected=np.column_stack([np.bincount(cluster_bootstrap(tr['groups'],s,i),minlength=len(tr['groups'])) for i in range(32)])
            np.testing.assert_array_equal(np.loadtxt(p/'inbag_PRIVATE.csv',delimiter=','),expected)
            continue
        f=CompetingRiskForest.load(forest);assert asdict(f.config)==asdict(configuration(s,a));models[(y,a,s)]=f
    geometry=[]
    strip=lambda f:[[{k:v for k,v in n.items() if k not in ['cif','leaf_shrinkage']} for n in t] for t in f.trees_]
    pairs=[('baseline','parent_shrink'),('shallow','shallow_parent')]+[(a+'_raw',a) for a in NEW]
    for y,s in [(y,s) for y in YEARS for s in SEEDS]:
        for raw,sh in pairs:
            l,r=models[(y,raw,s)],models[(y,sh,s)]
            assert asdict(l.config)==asdict(r.config) and strip(l)==strip(r)
            assert not any('leaf_shrinkage' in n for t in l.trees_ for n in t)
            assert all('leaf_shrinkage' in n for t in r.trees_ for n in t if 'cif' in n)
            geometry.append(dict(year=y,seed=s,raw=raw,shrunk=sh,configuration_same=True,all_split_geometry_same=True,raw_unregularized=True))
    pd.DataFrame(geometry).to_csv(out/'raw_shrunk_geometry.csv',index=False)
    pd.DataFrame(hashes).to_csv(out/'model_hashes.csv',index=False);pd.DataFrame(cost).to_csv(out/'model_costs.csv',index=False)
    executions=[]
    for stage in ['formal','verify']:
        gate=json.loads((run/f'audit/{stage}_gate.json').read_text())
        report=json.loads((run/f'{stage}.jsonl.results-{gate["job"]}.json').read_text())
        for task in report['completed']:
            folder=run/'results'/stage/str(gate['job'])/f'task-{task["id"]}/output'
            z=json.loads((folder/'summary.json').read_text())
            executions.append(dict(stage=stage,task_id=task['id'],job=gate['job'],year=z['year'],arm=z.get('arm',z.get('design_arm')),seed=z['seed'],host=z['host'],seconds=z['seconds'],cpu_seconds=z['cpu_seconds'],peak_rss_MiB=z['peak_rss_MiB'],new_forest_fits=z['new_forest_fits'],publication_completed=task['publication_completed'],returncode=task['returncode']))
    ed=pd.DataFrame(executions);ed.to_csv(out/'execution_tasks.csv',index=False)
    assert ed.new_forest_fits.sum()==18 and len(ed)==98 and ed.returncode.max()==0
    fd=pd.read_csv(run/'private/function_stability_detail_PRIVATE.csv')
    fd.groupby(['year','arm','seed_a','seed_b','day','cause','order']).agg(common_terms=('key','size'),common_nonconstant=('correlation','count'),median_correlation=('correlation','median'),mean_rms_difference=('rms_difference','mean')).reset_index().to_csv(out/'function_stability_by_order.csv',index=False)
    m=pd.read_csv(out/'metrics.csv',dtype={'seed':str,'scope':str});ci=pd.read_csv(out/'paired_bootstrap_intervals.csv',dtype={'scope':str});diff=pd.read_csv(out/'paired_differences.csv',dtype={'scope':str,'seed':str})
    for row in ci.itertuples():
        z=diff[(diff.scope==row.scope)&(diff.seed=='ensemble')&(diff.horizon_months==24)&(diff.cause=='termination')&(diff.contrast==row.contrast)]
        assert len(z)==1
        np.testing.assert_allclose(row.estimate,float(z.iloc[0][row.metric]),atol=1e-13,rtol=0)
        assert row.valid_replicates+row.undefined_replicates==1000
    assert len(m)==528 and len(ci)==552
    for n,h in json.loads((run/'audit/source_hashes.json').read_text()).items():assert sha(Path(n))==h
    for n,h in json.loads((run/'audit/immutable_input_hashes.json').read_text()).items():assert sha(run/'input'/n)==h
    repo=Path(__file__).resolve().parents[2]
    assert sha(repo/'CURRENT_POST_ST_MODEL.json')==json.loads((run/'audit/baseline_audit.json').read_text())['current_reference_sha256']
    write(out/'final_validation.json',dict(status='PASS',new_formal_forest_fits=18,historical_reproduction_fits=0,new_leaf_updates=18,reused_internal_models=24,reused_standard_RSF_models=6,total_single_seed_models_with_RSF=66,complete_internal_mean_predictors=20,successful_verification_tasks=80,failed_fits=0,failed_publications=0,postprocessing_failures=1,geometry_pairs_verified=len(geometry),metric_rows=528,paired_interval_rows=552,reference_unchanged=True,originals_and_inputs_unchanged=True,standard_RSF_reload_checks=6,interpretation='exploratory; no reference replacement'))
    print(ed.groupby('stage')[['seconds','peak_rss_MiB']].agg(['min','median','max']).to_string())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);audit(p.parse_args().run)
