"""Independent final checks of dates, publication, frozen inputs and rank metrics."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from experiments.post_st_prognosis_v1.common import arrays,sha,write
from .prepare import SOURCE


def validate(run):
    hashes=json.loads((run/'audit/immutable_input_hashes.json').read_text())
    for name,h in hashes.items():assert sha(run/'input'/name)==h,name
    sources=json.loads((run/'audit/data_alignment.json').read_text())['source_sha256']
    for name,h in sources.items():assert sha(name)==h,name
    events=pd.read_parquet(SOURCE/'private/endpoint_ledger_PRIVATE.parquet');dates=0
    for year in [2023,2024]:
        groups=[]
        for role,cut in [('fit',pd.Timestamp(f'{year-2}-01-01')),('calibration',pd.Timestamp(f'{year}-01-01')),('test',pd.Timestamp('2026-09-04'))]:
            folder=run/('evaluation_input' if role=='test' else 'input/folds')/str(year)
            m=pd.read_parquet(folder/f'{role}_metadata_PRIVATE.parquet');a=arrays(folder/f'{role}_PRIVATE.npz');groups.append(set(a['groups']))
            assert len(m)==len(a['X']) and np.array_equal(m.sample_id.to_numpy(str),a['sample_ids'])
            for row in m.itertuples():
                cap=min(cut-pd.Timedelta(days=1),row.origin+pd.DateOffset(months=12))
                ep=events[(events.group==row.group)&(events.announcement>row.origin)&(events.announcement<=cap)].sort_values('announcement')
                if len(ep):
                    first=ep.iloc[0];stop=first.announcement;event=1 if first.type.endswith('X') else 2
                    assert ep.loc[ep.announcement==stop,'type'].str[-1].nunique()==1
                else:stop=cap;event=0
                assert row.event==event and row.stop==stop and row.time==(stop-row.origin).days
                assert row.financial_available<row.origin and row.market_available<row.origin and stop<cut
                dates+=1
        assert not groups[0]&groups[1] and not groups[0]&groups[2] and not groups[1]&groups[2]
    model_hashes=[];hosts={}
    for stage,n in [('pilot',2),('formal',18)]:
        g=json.loads((run/f'audit/{stage}_gate.json').read_text());assert g['status']=='PASS'
        report=run/f'{stage}.jsonl.results-{g["job"]}.json';assert sha(report)==g['report_sha256']
        r=json.loads(report.read_text());assert len(r['completed'])==n and not r['pending'] and r['srun_returncode']==0
        assert {x['id'] for x in r['completed']}==set(range(n))
        for x in r['completed']:
            assert x['compute_completed'] and x['publication_completed'] and x['returncode']==0
            out=run/'results'/stage/str(g['job'])/f"task-{x['id']}"/'output'
            for name,h in json.loads((out/'hashes.json').read_text()).items():assert sha(out/name)==h
            hosts[x['host']]=hosts.get(x['host'],0)+1
            if stage=='formal':
                model=next(out.glob('forest_PRIVATE.*'));model_hashes.append(dict(task=x['id'],host=x['host'],
                    file=model.name,sha256=sha(model),prediction_sha256=sha(out/'predictions_PRIVATE.npz')))
    ev=run/'evaluation';metrics=pd.read_csv(ev/'metrics.csv',dtype={'seed':str})
    assert len(metrics)==104
    # No events means undefined AUC/AP, not zero or an arbitrary constant.
    zero=metrics.known_target_events.eq(0);assert metrics.loc[zero,['auc','ap']].isna().all().all()
    for _,d in metrics[metrics.state.isin(['uncalibrated','affine'])].groupby(['test_year','model','seed','cause']):
        d=d.set_index('state');np.testing.assert_allclose(d.loc['uncalibrated',['auc','ap']].to_numpy(float),d.loc['affine',['auc','ap']].to_numpy(float),rtol=0,atol=1e-12,equal_nan=True)
    w=pd.read_csv(ev/'warning_metrics.csv',dtype={'seed':str})
    cols=['checked','captured','coverage','known_non_target','unknown','lead3','lead6']
    for _,d in w[w.state.isin(['uncalibrated','affine'])].groupby(['test_year','model','seed','top_fraction']):
        d=d.set_index('state');np.testing.assert_array_equal(d.loc['uncalibrated',cols].to_numpy(float),d.loc['affine',cols].to_numpy(float))
    solve=pd.read_csv(run/'reports/calibration_solver.csv');assert len(solve)==24
    assert solve.success.all() and solve.gap.max()<=1e-8 and solve.primal_violation.max()<=1e-10
    check=pd.read_csv(ev/'mathematical_checks.csv');assert check.rank_unchanged.all() and check.reconstruction_error.max()<1e-10
    result=dict(status='PASS',date_records_checked=dates,immutable_input_files=len(hashes),source_files=len(sources),
        formal_fits=18,pilots=2,calibration_fits=24,metrics_verified=104,undefined_metrics_preserved=True,
        rankings_and_warning_counts_unchanged=True,solver_max_gap=float(solve.gap.max()),
        q2_reconstruction_max_error=float(check.reconstruction_error.max()),actual_fit_hosts=hosts,model_hashes=model_hashes)
    write(ev/'independent_validation.json',result);print(json.dumps({k:v for k,v in result.items() if k!='model_hashes'}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);validate(p.parse_args().run)
