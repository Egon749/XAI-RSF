"""Independent accounting, source preservation, pairing and numerical audit."""
from pathlib import Path
import argparse,gzip,json
from dataclasses import asdict
import numpy as np
import pandas as pd
from experiments.post_st_dropcolumn_selection_v1.common import REFERENCE_SHA
from experiments.post_st_train2024_h12_v1.validate import direct_metrics
from .analyze import final_models
from .initial import ExistingStore
from .fitting import *

def validate(run):
    repo=Path(__file__).resolve().parents[2];parent=Path(load(run/'input/protocol.json')['parent_private_run'])
    assert sha(repo/'CURRENT_POST_ST_MODEL.json')==REFERENCE_SHA
    for key,h in load(parent/'audit/immutable_input_hashes.json').items():assert sha(parent/'input'/key)==h
    for f,h in load(parent/'audit/source_hashes.json').items():assert sha(Path(f))==h
    for m in (run/'audit').glob('code_snapshot*.json'):
        meta=load(m);root=run/'input'/meta.get('code_dir','code')
        for p,h in meta['code_hashes'].items():assert sha(root/p)==h
        assert sha(run/'input/protocol.json')==meta['protocol_sha256']
    tasks=[];bundles=[];checks=[];model_hashes=[];pair_reference={};configuration_count=0;configuration_checked=set()
    for file in sorted((run/'audit').glob('registry_*.json')):
        stage=file.stem.removeprefix('registry_');records=load(file)
        if not records:continue
        gate=load(run/f'audit/gate_{stage}.json')
        if records:assert gate['status']=='PASS'
        for r in records:
            out=Path(r['output']);s=load(out/'summary.json');assert s['status']=='COMPLETE'
            tasks.append(dict(stage=stage,slurm_job_id=gate['job'],year=r.get('year',r.get('outer')),repeat=r.get('repeat',-1),new_forests=s['new_forests'],seconds=s['seconds'],peak_rss_MiB=s['peak_rss_MiB'],host=s['host']))
            paths=[out/'bundle'] if r['kind']=='bundle' else [out/p for p in s['bundles']]+([out/s['final']['path']] if not s['final']['reused'] else [])
            assert len(paths)*3==s['new_forests']
            for p in paths:
                info=load(p/'summary.json');keep=info['keep'];label=p.name
                bundles.append(dict(stage=stage,slurm_job_id=gate['job'],outer=r.get('outer',r.get('year')),repeat=r.get('repeat',-1),year=info['year'],subset_hash=info['subset_hash'],features=len(keep),new_forests=3,
                    seconds=info['seconds'],fit_seconds=sum(info['fit_seconds']),shrink_seconds=sum(info['shrink_seconds']),training_rows=info['training_rows'],training_companies=info['training_companies'],unique_training_termination=info['unique_training_termination']))
                for seed,cc in zip(SEEDS,info['checks']):
                    pp=load(p/f'{seed}/pairing.json');key=(r.get('outer',0),r.get('repeat',-1),info['year'],seed)
                    pair=[(v['bootstrap_sha256'],v['multiplicity_sha256']) for v in pp]
                    if key in pair_reference:assert pair_reference[key]==pair
                    else:pair_reference[key]=pair
                    for version,c in cc.items():
                        folder=p/f'{seed}/{version}';configuration_key=(stage,info['year'],len(keep),seed,version)
                        if configuration_key not in configuration_checked:
                            raw=json.loads(gzip.decompress((folder/'forest_PRIVATE.json.gz').read_bytes()))
                            assert raw['config']==json.loads(json.dumps(asdict(configuration(seed,len(keep),'count'))))
                            configuration_checked.add(configuration_key);configuration_count+=1
                        model_hashes.append(dict(stage=stage,slurm_job_id=gate['job'],outer=r.get('outer',r.get('year')),repeat=r.get('repeat',-1),year=info['year'],subset_hash=info['subset_hash'],seed=seed,version=version,
                            forest_sha256=sha(folder/'forest_PRIVATE.json.gz'),functions_sha256=sha(folder/'functions_PRIVATE.npz'),predictions_sha256=sha(folder/'predictions_PRIVATE.npz')))
                        checks.append(dict(stage=stage,year=info['year'],subset_hash=info['subset_hash'],repeat=r.get('repeat',-1),seed=seed,version=version,**c))
                assert info['ensemble_mean_error']<=1e-13
                checks.append(dict(stage=stage,year=info['year'],subset_hash=info['subset_hash'],repeat=r.get('repeat',-1),seed='ensemble',version='shrunk',**info['ensemble_check']))
    for c in checks:
        assert c['max_path_variables']<=2 and c['path_violations']==0
        assert c['probability_min']>=-1e-12 and c['probability_sum_max']<=1+1e-10 and c['minimum_increment']>=-1e-12
        for k in ['full_reconstruction','explicit_component_sum','function_reload','forest_reload']:assert c[k]<=1e-10
    df=pd.read_csv(run/'reports/MODEL_COMPARISON.csv',dtype={'year':str,'seed':str});verified=0;reproduced_forests=0;reproduction_error=0.
    final_labels={};final_predictions={}
    for year,spec in final_models(run,parent).items():
        lab=arrays(parent/f'evaluation_input/{year}_test_PRIVATE.npz');z=arrays(spec['path']/'ensemble/predictions_PRIVATE.npz')
        final_labels[year]=lab;final_predictions[year]=z['cif']
        if year==2024 and spec['new']:
            a=arrays(parent/'input/folds/2024/train_PRIVATE.npz');b=arrays(parent/'input/outer/2024/train_PRIVATE.npz')
            for key in ['X','time','event','groups','sample_ids']:np.testing.assert_equal(a[key],b[key])
            previous=ExistingStore(parent).get(tuple(spec['keep']),year)
            reproduction_error=float(abs(z['cif']-previous['ensemble']).max());assert reproduction_error<=1e-13;reproduced_forests+=3
        for h in [12,24]:
            np.testing.assert_array_equal(lab['time']<=lab[f'h{h}'],lab['analysis_time']<=h)
            if not evaluation_context(lab,h,1)[3]:continue
            for cause in [1,2]:
                p=horizon_predictions(z['cif'],TIMES,lab[f'h{h}'])[:,cause-1];y,w,known,_,_=evaluation_context(lab,h,cause)
                row=df[(df.year==str(year))&(df.label=='selected_h24')&(df.horizon==h)&(df.cause==cause)&(df.seed=='ensemble')].iloc[0]
                if known.all():wanted=direct_metrics(y,p)
                else:
                    from sklearn.metrics import roc_auc_score,average_precision_score
                    active=w>0;wanted=[roc_auc_score(y[active],p[active],sample_weight=w[active]),average_precision_score(y[active],p[active],sample_weight=w[active]),np.sum(w*(p-y)**2)/len(y)]
                np.testing.assert_allclose(wanted,row[['auc','ap','brier']].to_numpy(float),atol=1e-12,rtol=0,equal_nan=True);verified+=1
    for h in [12,24]:
        years=[year for year,lab in final_labels.items() if evaluation_context(lab,h,1)[3]]
        lab={k:np.concatenate([final_labels[year][k] for year in years]) for k in final_labels[years[0]]}
        risks=np.concatenate([horizon_predictions(final_predictions[year],TIMES,final_labels[year][f'h{h}']) for year in years])
        for cause in [1,2]:
            y,w,known,_,_=evaluation_context(lab,h,cause);p=risks[:,cause-1]
            if known.all():wanted=direct_metrics(y,p)
            else:
                from sklearn.metrics import roc_auc_score,average_precision_score
                active=w>0;wanted=[roc_auc_score(y[active],p[active],sample_weight=w[active]),average_precision_score(y[active],p[active],sample_weight=w[active]),np.sum(w*(p-y)**2)/len(y)]
            row=df[(df.year=='pooled')&(df.label=='selected_h24')&(df.horizon==h)&(df.cause==cause)&(df.seed=='ensemble')].iloc[0]
            np.testing.assert_allclose(wanted,row[['auc','ap','brier']].to_numpy(float),atol=1e-12,rtol=0,equal_nan=True);verified+=1
    selections=pd.read_csv(run/'reports/FEATURE_SELECTION_STABILITY.csv');assert len(selections)==249
    boot=pd.read_csv(run/'reports/BOOTSTRAP_SUPPORT_AND_SELECTION.csv');assert len(boot)==150
    assert set(zip(boot.year,boot['repeat']))=={(y,r) for y in [2023,2024,2025] for r in range(50)}
    pd.DataFrame(tasks).to_csv(run/'reports/EXECUTION_TASKS.csv',index=False);pd.DataFrame(bundles).to_csv(run/'reports/FIT_BUNDLES.csv',index=False)
    pd.DataFrame(checks).to_csv(run/'reports/NUMERICAL_VALIDATION.csv.gz',index=False,compression='gzip');pd.DataFrame(model_hashes).to_csv(run/'reports/MODEL_HASHES.csv.gz',index=False,compression='gzip')
    error=max(c[k] for c in checks for k in ['full_reconstruction','explicit_component_sum','function_reload','forest_reload'])
    result=dict(status='PASS',actual_new_forests=sum(r['new_forests'] for r in tasks),new_bundles=len(bundles),tasks=len(tasks),configuration_checks=configuration_count,
        configuration_check_scope='Each distinct stage/year/input-count/seed/raw-shrunk configuration; every model separately passed full save/reload/function verification in worker',
        extra_outer_forests_reproducing_existing_subset=reproduced_forests,existing_subset_forecast_reproduction_error=reproduction_error,
        max_reconstruction_error=error,independent_outer_metric_rows=verified,failed_fits=0,slurm_job_ids=sorted({str(r['slurm_job_id']) for r in tasks}),company_pairing_verified=True,original_sources_unchanged=True,current_reference_unchanged=True,
        selection_replicates_per_cutoff=50,bootstrap_primary_support_passes=boot.groupby('year').support_passed.sum().to_dict(),bootstrap_actual_selection_scans=boot.groupby('year').selection_scan_executed.sum().to_dict())
    write(run/'audit/validation.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();validate(a.run)
