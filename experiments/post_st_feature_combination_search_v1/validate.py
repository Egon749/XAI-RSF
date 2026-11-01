"""Independent accounting, eligibility, immutable sources and metric checks."""
import argparse
from dataclasses import asdict
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from experiments.post_st_dropcolumn_selection_v1.common import (
    load, sha, write, configuration, REFERENCE_SHA, arrays, TIMES, SEEDS,
    horizon_predictions, evaluation_context)
from experiments.post_st_feature_combination_search_v1.analyze import model_specs, concatenate
from .search import FULL, anchors, lattice, legal, decide


def validate(run):
    repo=Path(__file__).resolve().parents[2]
    assert sha(repo/'CURRENT_POST_ST_MODEL.json')==REFERENCE_SHA
    for path,expected in load(run/'audit/source_hashes.json').items():
        assert sha(Path(path))==expected,path
    meta=load(run/'audit/code_snapshot.json')
    assert sha(run/'input/protocol.json')==meta['protocol_sha256']
    for path,expected in meta['code_hashes'].items():
        assert sha(run/'input/code'/path)==expected,path
    tasks={};bundles=[];numerical=[];hashes=[];pair_reference={};config_checked=set()
    for file in sorted((run/'audit').glob('registry_*.json')):
        stage=file.stem.removeprefix('registry_');records=load(file)
        if not records:continue
        gate=load(run/f'audit/gate_{stage}.json');assert gate['status']=='PASS'
        assert gate['new_forests']==3*len(records)
        for r in records:
            p=Path(r['path']);out=Path(r['output']);s=load(p/'summary.json')
            taskkey=(stage,r['task']);task=load(out/'summary.json')
            assert task['status']=='COMPLETE' and task['no_outer_labels_read']
            tasks[taskkey]=dict(stage=stage,job=r['job'],task=r['task'],new_forests=task['new_forests'],
                host=task['host'],seconds=task['seconds'],peak_rss_MiB=task['peak_rss_MiB'],affinity=str(task['affinity']))
            assert r['keep']==s['keep'] and r['year']==s['year']
            keep=tuple(r['keep']);assert legal(keep)
            bundles.append(dict(stage=stage,job=r['job'],task=r['task'],year=r['year'],repeat=r.get('repeat',-1),
                subset_hash=s['subset_hash'],features=len(keep),new_forests=3,
                seconds=s['seconds'],fit_seconds=sum(s['fit_seconds']),shrink_seconds=sum(s['shrink_times']) if 'shrink_times' in s else sum(s['shrink_seconds']),
                training_companies=s['training_companies'],training_rows=s['training_rows'],
                known_termination=s['unique_training_termination']))
            for seed,cc in zip(SEEDS,s['checks']):
                pp=load(p/f'{seed}/pairing.json')
                key=(r['year'],r.get('repeat',-1),seed)
                identity=[(v['bootstrap_sha256'],v['multiplicity_sha256']) for v in pp]
                if key in pair_reference:assert pair_reference[key]==identity
                else:pair_reference[key]=identity
                for version,c in cc.items():
                    folder=p/f'{seed}/{version}';config_key=(r['year'],len(keep),seed,version)
                    if config_key not in config_checked:
                        raw=json.loads(gzip.decompress((folder/'forest_PRIVATE.json.gz').read_bytes()))
                        assert raw['config']==json.loads(json.dumps(asdict(configuration(seed,len(keep),'count'))))
                        config_checked.add(config_key)
                    numerical.append(dict(stage=stage,year=r['year'],repeat=r.get('repeat',-1),subset_hash=s['subset_hash'],seed=seed,version=version,**c))
                    hashes.append(dict(stage=stage,year=r['year'],repeat=r.get('repeat',-1),subset_hash=s['subset_hash'],seed=seed,version=version,
                        forest_sha256=sha(folder/'forest_PRIVATE.json.gz'),functions_sha256=sha(folder/'functions_PRIVATE.npz'),predictions_sha256=sha(folder/'predictions_PRIVATE.npz')))
            assert s['ensemble_mean_error']<=1e-13
            numerical.append(dict(stage=stage,year=r['year'],repeat=r.get('repeat',-1),subset_hash=s['subset_hash'],seed='ensemble',version='shrunk',**s['ensemble_check']))
    for c in numerical:
        assert c['max_path_variables']<=2 and c['path_violations']==0
        assert c['probability_min']>=-1e-12 and c['probability_sum_max']<=1+1e-10 and c['minimum_increment']>=-1e-12
        assert max(c[k] for k in ['full_reconstruction','explicit_component_sum','function_reload','forest_reload'])<=1e-10
    candidates=pd.read_csv(run/'reports/CANDIDATE_SCORES.csv')
    decisions=load(run/'audit/selection_decisions.json')
    for year,g in candidates.groupby('year'):
        rows=[]
        for d in g.to_dict('records'):
            d['keep']=json.loads(d['keep']);d['failed_guards']=json.loads(d['failed_guards']);rows.append(d)
        universe={tuple(d['keep']) for d in rows}
        assert set(lattice(year))<=universe
        assert set(anchors(year))<=universe
        assert all(legal(k) for k in universe)
        choice=decide(rows,decisions[str(year)]['gate'])
        assert choice['keep']==decisions[str(year)]['keep']
        if not decisions[str(year)]['gate']['passed']:assert tuple(choice['keep'])==FULL
        assert {0,1,2}<=set(g.first_stage), 'All frozen exploration stages must be retained'
    parent,specs=model_specs(run)
    labels={y:arrays(parent/f'evaluation_input/{y}_test_PRIVATE.npz') for y in [2023,2024,2025]}
    df=pd.read_csv(run/'reports/MODEL_COMPARISON.csv',dtype={'year':str,'seed':str})
    checked=0
    for model in ['full83','legacy74','previous_auto','selected_search']:
        pred={y:arrays(Path(specs[(y,model)]['path'])/'ensemble/predictions_PRIVATE.npz')['cif'] for y in labels}
        for group in [[2023],[2024],[2025],[2023,2024,2025]]:
            for h in [12,24]:
                years=[y for y in group if evaluation_context(labels[y],h,1)[3]]
                keyyear=str(group[0]) if len(group)==1 else 'pooled'
                if not years:
                    rows=df[(df.year==keyyear)&(df.model==model)&(df.horizon==h)&(df.seed=='ensemble')]
                    assert rows[['auc','ap','brier','captured']].isna().all().all()
                    continue
                lab=concatenate(labels,years)
                risk=np.concatenate([horizon_predictions(pred[y],TIMES,labels[y][f'h{h}']) for y in years])
                for cause in [1,2]:
                    y,w,known,ok,_=evaluation_context(lab,h,cause);assert ok
                    active=w>0;p=risk[:,cause-1]
                    auc=roc_auc_score(y[active],p[active],sample_weight=w[active]) if len(np.unique(y[active]))==2 else np.nan
                    ap=average_precision_score(y[active],p[active],sample_weight=w[active]) if np.any(y[active]) else np.nan
                    brier=np.sum(w*(p-y)**2)/len(y)
                    row=df[(df.year==keyyear)&(df.model==model)&(df.horizon==h)&(df.cause==cause)&(df.seed=='ensemble')].iloc[0]
                    np.testing.assert_allclose([auc,ap,brier],row[['auc','ap','brier']].to_numpy(float),atol=1e-12,rtol=0,equal_nan=True)
                    captured=0;checked_count=0
                    for yy in years:
                        ll=labels[yy];target,_,known,_,_=evaluation_context(ll,h,cause)
                        rr=horizon_predictions(pred[yy],TIMES,ll[f'h{h}'])[:,cause-1]
                        n=int(np.ceil(.2*len(rr)));ix=np.lexsort((ll['sample_ids'],-rr))[:n]
                        captured+=int((target[ix]&known[ix]).sum());checked_count+=n
                    assert captured==row.captured and checked_count==row.checked
                    checked+=1
    boot=load(run/'audit/registry_bootstrap.json')
    for y,d in decisions.items():
        expected=set(range(50)) if tuple(d['keep'])!=FULL else set()
        assert {r['repeat'] for r in boot if r['year']==int(y)}==expected
    pd.DataFrame(tasks.values()).to_csv(run/'reports/EXECUTION_TASKS.csv',index=False)
    pd.DataFrame(bundles).to_csv(run/'reports/FIT_BUNDLES.csv',index=False)
    pd.DataFrame(numerical).to_csv(run/'reports/NUMERICAL_VALIDATION.csv.gz',index=False,compression='gzip')
    pd.DataFrame(hashes).to_csv(run/'reports/MODEL_HASHES.csv.gz',index=False,compression='gzip')
    error=max(c[k] for c in numerical for k in ['full_reconstruction','explicit_component_sum','function_reload','forest_reload'])
    result=dict(status='PASS',new_forest_fits=3*len(bundles),bundles=len(bundles),tasks=len(tasks),
        task_fit_count=sum(t['new_forests'] for t in tasks.values()),failed_fits=0,failed_publications=0,
        max_reconstruction_error=error,configuration_classes_checked=len(config_checked),
        independently_verified_metric_rows=checked,company_bootstrap_pairing=True,all_candidate_lattices_complete=True,
        final_decisions_recomputed=True,original_sources_unchanged=True,current_reference_unchanged=True,
        jobs=sorted({str(t['job']) for t in tasks.values()}),code_commit=meta['commit'])
    assert result['new_forest_fits']==result['task_fit_count']
    write(run/'audit/validation.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    validate(parser.parse_args().run)
