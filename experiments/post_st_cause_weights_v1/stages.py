"""Freeze unchanged source folds and finite tasks; verify successful publication."""
from pathlib import Path
import argparse,datetime,json,shutil,subprocess,tarfile
import numpy as np
import pandas as pd
from experiments.post_st_prognosis_v1.common import arrays,sha,write,calendar_elapsed


def prepare(run):
    protocol=json.loads(Path(__file__).with_name('frozen_protocol.json').read_text())
    write(run/'audit/protocol.json',protocol)
    write(run/'audit/pre_fit_freeze.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        protocol_sha256=sha(Path(__file__).with_name('frozen_protocol.json')),no_new_test_performance_seen=True))
    source=Path(protocol['source_run']);oldhash=json.loads((source/'audit/immutable_input_hashes.json').read_text())
    sources={};counts=[]
    for year in protocol['test_years']:
        src=source/'input/folds'/str(year);oldplan=json.loads((src/'plan.json').read_text())
        for name in ['train_PRIVATE.npz','eval_PRIVATE.npz','fit_metadata_PRIVATE.parquet']:
            path=src/name
            assert sha(path)==oldhash[str(path.relative_to(source/'input'))]
            sources[str(path)]=sha(path)
        tr=arrays(src/'train_PRIVATE.npz');meta=pd.read_parquet(src/'fit_metadata_PRIVATE.parquet')
        assert np.array_equal(tr['sample_ids'],meta.sample_id.to_numpy(str)) and meta.group.is_unique
        assert (meta.stop<pd.Timestamp(f'{year-2}-01-01')).all()
        assert (meta.financial_available<meta.origin).all() and (meta.market_available<meta.origin).all()
        np.testing.assert_array_equal(tr['analysis_time'],[calendar_elapsed(o,s) for o,s in zip(meta.origin,meta.stop)])
        counts.append(dict(test_year=year,role='fit',companies=len(meta),termination=int((tr['event']==1).sum()),
            recovery=int((tr['event']==2).sum()),unknown=int(((tr['analysis_time']<12)&(tr['event']==0)).sum())))
        for variant,weights in protocol['weights'].items():
            folder=run/'input/folds'/str(year)/variant;folder.mkdir(parents=True)
            for name in ['train_PRIVATE.npz','eval_PRIVATE.npz']:shutil.copy2(src/name,folder/name)
            plan={k:oldplan[k] for k in ['forest_common','forest_variants','standard_rsf','feature_names','calibration_n','test_n']}
            plan['forest_variants']['xai_q2']['cause_weights']=weights
            plan.update(pilot={'trees':4},variant=variant,test_year=year,
                source_hashes={name:sha(folder/name) for name in ['train_PRIVATE.npz','eval_PRIVATE.npz']})
            write(folder/'plan.json',plan)
        target=run/'evaluation_input'/str(year);target.mkdir()
        for name in ['test_PRIVATE.npz','test_metadata_PRIVATE.parquet']:
            path=source/'evaluation_input'/str(year)/name;sources[str(path)]=sha(path);shutil.copy2(path,target/name)
        ev=arrays(target/'test_PRIVATE.npz');mm=pd.read_parquet(target/'test_metadata_PRIVATE.parquet')
        assert np.array_equal(ev['sample_ids'],mm.sample_id.to_numpy(str))
        assert not set(ev['groups'])&set(tr['groups'])
        counts.append(dict(test_year=year,role='test',companies=len(mm),termination=int((ev['event']==1).sum()),
            recovery=int((ev['event']==2).sum()),unknown=int(((ev['analysis_time']<12)&(ev['event']==0)).sum())))
    pd.DataFrame(counts).to_csv(run/'reports/cohort_counts.csv',index=False)
    write(run/'audit/data_alignment.json',dict(status='PROVISIONAL',unchanged_source_hashes=sources,
        original_financial_vintages_PIT_certified=False,rolling_records_added=False,calibration_refitted=False,
        test2025_used=False,source_coverage_end='2026-09-03',counts=counts))
    print(pd.DataFrame(counts).to_string(index=False))


def freeze(run,repo):
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True)
    commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    for p in (run/'input/folds').glob('*/*/plan.json'):
        data=json.loads(p.read_text());data['code_commit']=commit;p.write_text(json.dumps(data,indent=2)+'\n')
    with (run/'audit/code.tar').open('xb') as f:subprocess.run(['git','-C',str(repo),'archive',commit],stdout=f,check=True)
    with tarfile.open(run/'audit/code.tar') as f:f.extractall(run/'input/code',filter='data')
    protocol=json.loads((run/'audit/protocol.json').read_text())
    for stage in ['pilot','formal']:
        items=[(y,v,s) for y in protocol['test_years']
            for v in (['termination_only'] if stage=='pilot' else protocol['weights'])
            for s in ([42] if stage=='pilot' else protocol['seeds'])]
        tasks=[dict(kind='cpu',argv=['python3',str(run/'input/code/experiments/post_st_cause_weights_v1/launch.py'),
            '--year',str(y),'--variant',v,'--seed',str(s),*(['--pilot'] if stage=='pilot' else [])],
            cwd=str(run/'input/code'),timeout=3600,
            storage=dict(source=str(run/'input'),mode='nfs',results=str(run/'results'/stage))) for y,v,s in items]
        (run/f'{stage}.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in tasks))
        (run/f'{stage}.py').write_text('# Run with hetero-run; independent jobs declared in sidecar.\n')
        write(run/f'{stage}.py.hetero.json',dict(independent=True,manifest=f'{stage}.jsonl'))
    write(run/'audit/immutable_input_hashes.json',{str(p.relative_to(run/'input')):sha(p) for p in (run/'input').rglob('*') if p.is_file()})
    print(commit)


def gate(run,stage,job):
    report=run/f'{stage}.jsonl.results-{job}.json';r=json.loads(report.read_text())
    n=len((run/f'{stage}.jsonl').read_text().splitlines())
    assert r['expected']==n and len(r['completed'])==n and not r['pending'] and r['srun_returncode']==0
    assert {x['id'] for x in r['completed']}==set(range(n))
    for x in r['completed']:
        assert x['returncode']==0 and x['compute_completed'] and x['publication_completed'],x
        out=run/'results'/stage/str(job)/f"task-{x['id']}"/'output'
        for name,h in json.loads((out/'hashes.json').read_text()).items():assert sha(out/name)==h
        s=json.loads((out/'summary.json').read_text());assert s['status']=='COMPLETE'
    result=dict(status='PASS',job=job,tasks=n,seconds=r['seconds'],report_sha256=sha(report),
        by_host={h:sum(x['host']==h for x in r['completed']) for h in {x['host'] for x in r['completed']}})
    write(run/f'audit/{stage}_gate.json',result);print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--repo',type=Path)
    p.add_argument('--stage',choices=['prepare','freeze','pilot','formal'],required=True);p.add_argument('--job');a=p.parse_args()
    if a.stage=='prepare':prepare(a.run)
    elif a.stage=='freeze':freeze(a.run,a.repo)
    else:gate(a.run,a.stage,a.job)
