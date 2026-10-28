"""Restore immutable old-C data, without invoking the annual acceptance adapter."""
from pathlib import Path
import argparse,datetime,hashlib,json,shutil,subprocess,tarfile
import numpy as np
import pandas as pd
from experiments.post_st_prognosis_v1.common import arrays,sha,write,calendar_elapsed
from experiments.post_st_next_open_info_v1.analyze import outputs


def prepare(run):
    repo=Path(__file__).resolve().parents[2]
    pp=Path(__file__).with_name('frozen_protocol.json');protocol=json.loads(pp.read_text())
    write(run/'audit/protocol.json',protocol)
    write(run/'audit/pre_fit_freeze.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        protocol_sha256=sha(pp),existing_results_exposed=True,new_fit_or_prediction_seen=False))
    old=Path(protocol['reference_run']);sources={};matches=[];groups=set();events=set()
    def source(p):
        p=Path(p);sources[str(p)]=sha(p);return p
    prior=json.loads(source(old/'audit/immutable_input_hashes.json').read_text())
    for name,h in prior.items():assert sha(old/'input'/name)==h
    delivered=json.loads(source(repo/'experiments/post_st_next_open_info_v1/execution_receipt.json').read_text())
    assert sha(source(old/'audit/data_alignment.json'))==delivered['result_sha256']['results/data_alignment.json']
    for name in ['metrics.csv','warning_metrics.csv','seed_stability.csv','ensemble_complexity.csv']:
        assert sha(source(old/'evaluation'/name))==delivered['result_sha256']['results/'+name]
    for year in protocol['test_years']:
        v='announcement_fields';src=old/'input/folds'/str(year)/v;dest=run/'input/folds'/str(year)/v;dest.mkdir(parents=True)
        plan=json.loads(source(src/'plan.json').read_text())
        assert len(plan['feature_names'])==83
        for name in ['train_PRIVATE.npz','eval_PRIVATE.npz']:
            p=source(src/name);assert sha(p)==plan['source_hashes'][name]
            shutil.copy2(p,dest/name);assert sha(dest/name)==sha(p)
        ep=source(old/'evaluation_input'/f'{year}_{v}_test_PRIVATE.npz')
        shutil.copy2(ep,run/'evaluation_input'/ep.name)
        tr=arrays(dest/'train_PRIVATE.npz');ev=arrays(ep);prediction_input=arrays(dest/'eval_PRIVATE.npz')
        assert set(prediction_input)=={'X','sample_ids'}
        for name in prediction_input:np.testing.assert_array_equal(prediction_input[name],ev[name])
        for role,d in [('fit',tr),('test',ev)]:
            p=source(old/'evaluation_input'/f'{year}_{role}_metadata_PRIVATE.parquet')
            shutil.copy2(p,run/'evaluation_input'/p.name);meta=pd.read_parquet(p)
            np.testing.assert_array_equal(d['sample_ids'],meta.sample_id)
            np.testing.assert_array_equal(d['analysis_time'],[calendar_elapsed(o,s) for o,s in zip(meta.origin,meta.stop)])
            assert meta.group.is_unique and meta.label_eligible.all() and d['X'].shape==(len(meta),83)
            if role=='fit':
                assert (meta.origin<pd.Timestamp(f'{year}-01-01')).all() and (meta.stop<pd.Timestamp(f'{year}-01-01')).all()
            matches.append(dict(test_year=year,role=role,companies=len(meta),columns=83,
                termination=int((d['event']==1).sum()),recovery=int((d['event']==2).sum()),
                unknown_h24=int(((d['event']==0)&(d['analysis_time']<24)).sum()),
                changed_input_companies=0,changed_labels=0))
        assert not set(tr['groups'])&set(ev['groups'])
        groups.update(tr['groups']);events.update(tr['groups'][tr['event']==1])
        plan.pop('code_commit')
        plan.update(source_status=protocol['source_status'],new_training_authorized=True,
                    restored_reference_plan_sha256=sha(src/'plan.json'))
        write(dest/'plan.json',plan)
    assert len(groups)==345 and len(events)==26
    refs=[]
    for year,v,model,seed,p in outputs(old):
        if v!='announcement_fields':continue
        for name,h in json.loads(source(p/'hashes.json').read_text()).items():assert sha(source(p/name))==h
        refs.append(dict(year=year,model=model,seed=seed,path=str(p)))
    assert len(refs)==12;write(run/'audit/reference_models.json',refs)
    fit=json.loads((old/'input/folds/2023/announcement_fields/plan.json').read_text())['code_commit'];impl={}
    for p in list((repo/'crforest').glob('*.py'))+[repo/'experiments/post_st_prognosis_v1'/n for n in ['worker.py','reference.R']]:
        rel=str(p.relative_to(repo));blob=subprocess.check_output(['git','-C',str(repo),'show',fit+':'+rel])
        assert hashlib.sha256(blob).hexdigest()==sha(p);impl[rel]=sha(p)
    shutil.copy2(source(old/'reports/cohort_counts.csv'),run/'reports/cohort_counts.csv')
    pd.DataFrame(matches).to_csv(run/'reports/matched_data_audit.csv',index=False)
    write(run/'audit/data_alignment.json',dict(status='PASS',source_status=protocol['source_status'],
        input_byte_identical_to_original_C=True,all_labels_times_groups_order_identical=True,
        training_company_union=345,training_termination_union=26,test_companies=81,test_terminations=32,
        original_financial_PIT_independently_certified=False,annual_acceptance_adapter_called=False,
        model_implementation_identical_to_old_fit=fit,implementation_sha256=impl,
        unchanged_source_hashes=sources,data=matches,new_fits_at_prepare=0))
    print(pd.DataFrame(matches).to_string(index=False))


def freeze(run,repo):
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True)
    commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    for p in (run/'input/folds').glob('*/*/plan.json'):
        d=json.loads(p.read_text());d['code_commit']=commit;p.write_text(json.dumps(d,indent=2)+'\n')
    with (run/'audit/code.tar').open('xb') as f:subprocess.run(['git','-C',str(repo),'archive',commit],stdout=f,check=True)
    with tarfile.open(run/'audit/code.tar') as f:f.extractall(run/'input/code',filter='data')
    protocol=json.loads((run/'audit/protocol.json').read_text())
    tasks=[dict(kind='cpu',argv=['python3',str(run/'input/code/experiments/post_st_next_open_info_v1/launch.py'),
        '--year',str(y),'--variant','announcement_fields','--model',m,'--seed',str(s)],
        cwd=str(run/'input/code'),timeout=3600,
        storage=dict(source=str(run/'input'),mode='nfs',results=str(run/'results/formal')))
        for y in protocol['test_years'] for m in protocol['models'] for s in protocol['seeds']]
    assert len(tasks)==protocol['formal_fits']==12
    (run/'formal.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in tasks))
    (run/'formal.py').write_text('# Independent commands are in the heteropool sidecar.\n')
    write(run/'formal.py.hetero.json',dict(independent=True,manifest='formal.jsonl'))
    write(run/'audit/immutable_input_hashes.json',{str(p.relative_to(run/'input')):sha(p) for p in (run/'input').rglob('*') if p.is_file()})
    print(commit)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--stage',choices=['prepare','freeze'],required=True);p.add_argument('--repo',type=Path)
    a=p.parse_args();prepare(a.run) if a.stage=='prepare' else freeze(a.run,a.repo)
