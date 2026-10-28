"""Describe where fixed-forecast changes occur; no model/threshold selection."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from experiments.post_st_prognosis_v1.common import arrays,horizon_predictions
from .analyze import artifacts


def diagnostics(run):
    paths={(y,a,s):p for y,a,s,p in artifacts(run)};rows=[]
    for year in [2023,2024]:
        ev=arrays(run/'evaluation_input'/f'{year}_test_PRIVATE.npz')
        y=(ev['event']==1)&(ev['analysis_time']<=24);assert ev['label_eligible'].all()
        for arm in ['baseline','parent_shrink','shallow','shallow_parent','standard_rsf']:
            members=[]
            for seed in ['42','101','202']:
                p=arrays(paths[(year,arm,seed)]/'predictions_PRIVATE.npz')
                members.append(horizon_predictions(p['cif'],p['times'],ev['h24'])[:,0])
            members=np.stack(members);pred=members.mean(0);base=[]
            for seed in ['42','101','202']:
                p=arrays(paths[(year,'baseline',seed)]/'predictions_PRIVATE.npz')
                base.append(horizon_predictions(p['cif'],p['times'],ev['h24'])[:,0])
            baseline=np.mean(base,axis=0)
            for group,mask in [('all',np.ones(len(y),bool)),('termination',y),('other_known',~y)]:
                rows.append(dict(test_year=year,arm=arm,group=group,n=int(mask.sum()),
                    risk_mean=float(pred[mask].mean()),risk_min=float(pred[mask].min()),risk_max=float(pred[mask].max()),
                    mse=float(((pred[mask]-y[mask])**2).mean()),seed_std_mean=float(members[:,mask].std(0,ddof=1).mean()),
                    risk_change_from_baseline=float((pred[mask]-baseline[mask]).mean()),
                    brier_contribution_change=float(np.sum((pred[mask]-y[mask])**2-(baseline[mask]-y[mask])**2)/len(y))))
    pd.DataFrame(rows).to_csv(run/'evaluation/prediction_change_diagnostics.csv',index=False)
    jobs=[]
    for stage in ['pilot','formal']:
        gate=json.loads((run/f'audit/{stage}_gate.json').read_text())
        for folder in sorted((run/'results'/stage/str(gate['job'])).glob('task-*/output')):
            summary=json.loads((folder/'summary.json').read_text());summary.pop('artifacts')
            jobs.append(dict(stage=stage,task=folder.parent.name,**summary))
    pd.DataFrame(jobs).to_csv(run/'evaluation/execution_tasks.csv',index=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);diagnostics(p.parse_args().run)
