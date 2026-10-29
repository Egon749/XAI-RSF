from pathlib import Path
import argparse,datetime,json
import numpy as np
import pandas as pd
from experiments.post_st_prognosis_v1.common import arrays,sha,write,horizon_predictions
from experiments.post_st_train2024_h12_v1.evaluate import metric


def artifacts(root,stage):
    gate=json.loads((root/'audit'/f'{stage}_gate.json').read_text());assert gate['status']=='PASS'
    for i,line in enumerate((root/f'{stage}.jsonl').read_text().splitlines()):
        argv=json.loads(line)['argv'];arg=lambda key:argv[argv.index('--'+key)+1]
        folder=root/'results'/stage/str(gate['job'])/f'task-{i}'/'output'
        yield int(arg('year')),int(arg('validation-year')),str(arg('seed')),folder


def gate(root,stage,job,result):
    pool=json.loads(result.read_text());tasks=(root/f'{stage}.jsonl').read_text().splitlines()
    assert not pool['pending'] and pool['srun_returncode']==0 and len(pool['completed'])==len(tasks)
    assert all(t['returncode']==0 for t in pool['completed'])
    hosts={};summaries=[]
    for i,line in enumerate(tasks):
        out=root/'results'/stage/str(job)/f'task-{i}'/'output'
        for name,h in json.loads((out/'hashes.json').read_text()).items():assert sha(out/name)==h
        summary=json.loads((out/'summary.json').read_text());assert summary['status']=='COMPLETE'
        summaries.append(summary);hosts[summary['host']]=hosts.get(summary['host'],0)+1
    write(root/'audit'/f'{stage}_gate.json',dict(status='PASS',stage=stage,job=job,tasks=len(tasks),
        pool_report=str(result),pool_report_sha256=sha(result),new_forest_fits=sum(s['new_forest_fits'] for s in summaries),
        model_artifacts=sum(s['model_artifacts'] for s in summaries),actual_hosts=hosts,
        published_and_hashes_verified=True))
    pd.DataFrame(summaries).to_csv(root/'reports'/f'{stage}_tasks.csv',index=False)
    print(stage,'PASS',hosts)


def select(root):
    protocol=json.loads((root/'input/protocol.json').read_text());pred={};source={}
    for _,v,seed,folder in artifacts(root,'validation'):
        for arm in protocol['arms']:
            p=folder/arm/'predictions_PRIVATE.npz';source[str(p)]=sha(p);pred[(v,arm,seed)]=arrays(p)
    rows=[];gates=[]
    for outer in protocol['outer_years']:
        for v in protocol['rolling_validation_entry_years']:
            lp=root/'private'/f'validation_{outer}_{v}_labels_PRIVATE.npz';source[str(lp)]=sha(lp);ev=arrays(lp)
            for h in protocol['primary_horizons_calendar_months']:
                known=((ev['event']>0)&(ev['analysis_time']<=h))|(ev['analysis_time']>=h)
                n_risk=int((ev['analysis_time']>=h).sum());supported=bool(known.all() or n_risk>=5)
                for arm in protocol['arms']:
                    members=[pred[(v,arm,str(s))] for s in protocol['seeds']]
                    ensembles={str(s):p for s,p in zip(protocol['seeds'],members)}
                    ensembles['ensemble']=dict(sample_ids=members[0]['sample_ids'],times=members[0]['times'],cif=np.mean([p['cif'] for p in members],axis=0))
                    for seed,p in ensembles.items():
                        np.testing.assert_array_equal(p['sample_ids'],ev['sample_ids']);risk=horizon_predictions(p['cif'],p['times'],ev[f'h{h}'])
                        for k,cause in [(1,'termination'),(2,'recovery')]:
                            row=dict(outer_year=outer,validation_entry_year=v,arm=arm,seed=seed,cause=cause,horizon_months=h,
                                horizon_supported=supported,risk_set_at_h=n_risk,all_endpoints_known=bool(known.all()),
                                companies=len(known),known_target_events=int((known&(ev['event']==k)&(ev['analysis_time']<=h)).sum()),unknown_outcomes=int((~known).sum()))
                            if supported:
                                metrics,_,_,_=metric(ev['analysis_time'],ev['event'],np.full(len(known),h),risk[:,k-1],k,ev['label_eligible']);row.update(metrics)
                            else:row['metric_note']='UNSUPPORTED_HORIZON; no selection score and no flat-tail inference'
                            rows.append(row)
    df=pd.DataFrame(rows);df.to_csv(root/'reports/rolling_validation_metrics.csv',index=False)
    select_rows=df[(df.seed=='ensemble')&(df.cause=='termination')&df.horizon_supported]
    chosen={};scores=[]
    for outer in protocol['outer_years']:
        for arm,w in [('baseline',1),('time_w2',2),('time_w4',4)]:
            cells=select_rows[(select_rows.outer_year==outer)&(select_rows.arm==arm)]
            assert len(cells)>0
            scores.append(dict(outer_year=outer,arm=arm,recent_weight=w,mean_brier=float(cells.brier.mean()),evaluable_cells=len(cells)))
        subset=[r for r in scores if r['outer_year']==outer];best=min(r['mean_brier'] for r in subset)
        winner=min([r for r in subset if r['mean_brier']<=best+1e-12],key=lambda r:r['recent_weight'])
        chosen[str(outer)]=winner
        for h in protocol['primary_horizons_calendar_months']:
            eligible=select_rows[(select_rows.outer_year==outer)&(select_rows.horizon_months==h)&(select_rows.arm==winner['arm'])&select_rows.all_endpoints_known]
            latest=eligible.sort_values('validation_entry_year').iloc[-1]
            target=int(latest.known_target_events);other=int(latest.companies)-target;passed=target>=10 and other>=10
            gates.append(dict(outer_year=outer,horizon_months=h,calibration_entry_year=int(latest.validation_entry_year),target_events=target,other_outcomes=other,
                minimum_target_events=10,minimum_other_outcomes=10,gate_passed=passed,
                status='ELIGIBLE_REQUIRES_FURTHER_CHECKS' if passed else 'SKIPPED_INSUFFICIENT_EVENTS',calibration_fits=0))
    pd.DataFrame(scores).to_csv(root/'reports/weight_selection.csv',index=False);pd.DataFrame(gates).to_csv(root/'reports/calibration_gate.csv',index=False)
    assert not any(g['gate_passed'] for g in gates),'Calibration eligibility differs from pre-fit counts; do not silently skip a required stage'
    write(root/'audit/selection_freeze.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_sha256=source,protocol_sha256=sha(root/'input/protocol.json'),chosen=chosen,
        criterion='equal mean historical cohort-by-horizon IPCW Brier, termination, ensemble',test_outcomes_used=False,
        calibration_fits=0,calibration_gate=gates,no_independent_2025_claim=True))
    print(pd.DataFrame(scores).to_string(index=False));print(pd.DataFrame(gates).to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--stage',choices=['pilot','validation','formal','select'],required=True)
    p.add_argument('--job',type=int);p.add_argument('--pool-result',type=Path);a=p.parse_args()
    select(a.run) if a.stage=='select' else gate(a.run,a.stage,a.job,a.pool_result)
