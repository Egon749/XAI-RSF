"""Supplement: replay out-of-bag predictions from immutable Python forests."""
from pathlib import Path
import argparse
import datetime
import json
import numpy as np
import pandas as pd
from crforest import cluster_bootstrap
from experiments.post_st_prognosis_v1.common import arrays,sha,write,horizon_predictions
from experiments.post_st_train2024_h12_v1.evaluate import metric
from experiments.post_st_cause_weights_h24_v1.reference import forecasts


def replay(run):
    # Registered after observing optimistic apparent training metrics and before
    # computing OOB results. This is an adaptive diagnostic, not preregistration.
    spec=dict(registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              rationale='Apparent training AUC near one prompted saved-tree OOB replay to separate resubstitution optimism from temporal loss',
              adaptive_post_exposure_diagnostic=True,new_fits=0,
              models=['equal','double_termination','termination_only','unrestricted'],
              rule='Within each seed, average only trees whose company bootstrap excluded the company. Then average the three seed OOB predictions equally. Report missing OOB coverage; never substitute in-bag trees.',
              caveat='Shared cut dictionary uses whole training X, no held-out labels. Few OOB trees and few target events; diagnostic, not full independent cross-validation or a causal policy estimate.')
    write(run/'audit/oob_supplement.json',spec)
    source=Path(json.loads((run/'audit/protocol.json').read_text())['source_run'])
    protocol=json.loads((source/'audit/protocol.json').read_text())
    old,paths,_=forecasts(protocol)
    gate=json.loads((source/'audit/formal_gate.json').read_text())
    jobs=[json.loads(s) for s in (source/'formal.jsonl').read_text().splitlines()]
    models={}
    for i,task in enumerate(jobs):
        a=task['argv'];key=(int(a[a.index('--year')+1]),a[a.index('--variant')+1],str(a[a.index('--seed')+1]))
        models[key]=source/'results/formal'/str(gate['job'])/f'task-{i}'/'output'
    models.update({k:p for k,p in paths.items() if k[1]=='unrestricted'})
    source_hashes=json.loads((run/'audit/source_hashes.json').read_text())
    rows=[];diagnostics=[];predictions={};valids={};checks=[]
    for (year,model,seed),folder in models.items():
        path=folder/'forest_PRIVATE.json';assert sha(path)==source_hashes[str(path)]
        f=json.loads(path.read_text());data=arrays(source/'input/folds'/str(year)/'equal/train_PRIVATE.npz')
        all_trees=[];is_oob=[]
        for tid,nodes in enumerate(f['trees']):
            bag=cluster_bootstrap(data['groups'],int(seed),tid)
            counts=np.bincount(bag,minlength=len(data['X']));oob=counts==0
            assert int(counts.sum())==nodes[0]['n']
            assert int(np.sum((counts>0)&(data['event']==1)))==nodes[0]['cause_company_support'][0]
            is_oob.append(oob);pred=[]
            for i,x in enumerate(data['X']):
                at=0
                while 'cif' not in nodes[at]:
                    n=nodes[at];v=x[n['feature']]
                    left=v<=n['threshold'] or (np.isnan(v) and n['missing_left'])
                    at=n['left' if left else 'right']
                ix=np.searchsorted(f['grid'],data['h24'][i],side='right')-1
                pred.append(nodes[at]['cif'][ix] if ix>=0 else [0.,0.])
            all_trees.append(pred)
        is_oob=np.asarray(is_oob);all_trees=np.asarray(all_trees)
        saved=arrays(folder/'predictions_PRIVATE.npz')
        expected=horizon_predictions(saved['training_cif'],saved['times'],data['h24'])
        error=float(np.max(abs(all_trees.mean(0)-expected)));assert error<1e-12
        n_oob=is_oob.sum(0);valid=n_oob>0
        result=np.full((len(valid),2),np.nan)
        result[valid]=(all_trees*is_oob[:,:,None]).sum(0)[valid]/n_oob[valid,None]
        predictions[(year,model,seed)]=result;valids[(year,model,seed)]=valid
        np.savez_compressed(run/'private'/f'oob_{year}_{model}_{seed}_PRIVATE.npz',
                            sample_ids=data['sample_ids'],cif=result,oob_trees=n_oob)
        diagnostics.append(dict(test_year=year,model=model,seed=seed,n=len(valid),no_oob_prediction=int((~valid).sum()),
            oob_trees_min=int(n_oob.min()),oob_trees_median=float(np.median(n_oob)),oob_trees_max=int(n_oob.max()),training_prediction_replay_error=error))
    for year in protocol['test_years']:
        for model in spec['models']:
            keys=[(year,model,str(s)) for s in protocol['seeds']]
            predictions[(year,model,'ensemble')]=np.mean([predictions[k] for k in keys],axis=0)
            valids[(year,model,'ensemble')]=np.all([valids[k] for k in keys],axis=0)
    for (year,model,seed),risk in predictions.items():
        data=arrays(source/'input/folds'/str(year)/'equal/train_PRIVATE.npz');use=valids[(year,model,seed)]
        for cause,name in [(1,'termination'),(2,'recovery')]:
            result,w,y,known=metric(data['analysis_time'][use],data['event'][use],24.,risk[use,cause-1],cause,data['label_eligible'][use])
            active=known&w.astype(bool);p=risk[use,cause-1]
            a=p[active&y];b=p[active&~y];wa=w[active&y];wb=w[active&~y]
            direct=float(np.sum(((a[:,None]>b)+.5*(a[:,None]==b))*wa[:,None]*wb)/(wa.sum()*wb.sum()))
            assert abs(direct-result['auc'])<1e-12
            rows.append(dict(test_year=year,model=model,seed=seed,cause=name,scope='saved_tree_out_of_bag_diagnostic',
                             training_n=len(use),missing_oob=int((~use).sum()),**result))
    pd.DataFrame(rows).to_csv(run/'results/oob_metrics.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(run/'results/oob_coverage.csv',index=False)
    for name,digest in source_hashes.items():assert sha(name)==digest
    write(run/'results/oob_receipt.json',dict(status='COMPLETE',new_fits=0,saved_forests=24,
        trees_replayed=sum(len(json.loads((d/'forest_PRIVATE.json').read_text())['trees']) for d in models.values()),
        max_training_replay_error=float(pd.DataFrame(diagnostics).training_prediction_replay_error.max()),
        missing_oob_predictions=int(pd.DataFrame(diagnostics).no_oob_prediction.sum()),
        all_original_hashes_unchanged=True,weighted_auc_independently_verified=True,
        spec_sha256=sha(run/'audit/oob_supplement.json'),script_sha256=sha(Path(__file__))))
    print(pd.DataFrame(rows).query("seed == 'ensemble' and cause == 'termination'")[['test_year','model','auc','ap','brier','missing_oob']].to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);replay(p.parse_args().run)
