from pathlib import Path
import argparse,datetime,json,shutil,subprocess,tarfile
import numpy as np
import pandas as pd
from experiments.post_st_prognosis_v1.common import arrays,sha,write
from experiments.post_st_next_open_info_v1.analyze import outputs


def prepare(run):
    repo=Path(__file__).resolve().parents[2];pp=Path(__file__).with_name('frozen_protocol.json')
    protocol=json.loads(pp.read_text());old=Path(protocol['reference_run']);sources={};counts=[]
    write(run/'audit/protocol.json',protocol)
    write(run/'audit/pre_fit_freeze.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        protocol_sha256=sha(pp),new_real_predictions_seen=False,test_outcomes_previously_exposed=True))
    def source(p):
        p=Path(p);sources[str(p)]=sha(p);return p
    receipt=json.loads(source(repo/'experiments/post_st_old_input_rerun_v1/execution_receipt.json').read_text())
    assert receipt['reproduction_status']=='EXACT_REPRODUCTION'
    for name,h in json.loads(source(old/'audit/immutable_input_hashes.json').read_text()).items():assert sha(old/'input'/name)==h
    for name in ['metrics.csv','warning_metrics.csv','ensemble_complexity.csv']:
        assert sha(source(old/'evaluation'/name))==receipt['result_sha256']['results/'+name]
    for year in [2023,2024]:
        src=old/'input/folds'/str(year)/'announcement_fields';dest=run/'input/folds'/str(year);dest.mkdir(parents=True)
        plan=json.loads(source(src/'plan.json').read_text())
        for name in ['train_PRIVATE.npz','eval_PRIVATE.npz']:
            p=source(src/name);assert sha(p)==plan['source_hashes'][name];shutil.copy2(p,dest/name)
            assert sha(dest/name)==sha(p)
        ep=source(old/'evaluation_input'/f'{year}_announcement_fields_test_PRIVATE.npz')
        shutil.copy2(ep,run/'evaluation_input'/f'{year}_test_PRIVATE.npz')
        tr=arrays(dest/'train_PRIVATE.npz');ev=arrays(ep);fit_ev=arrays(dest/'eval_PRIVATE.npz')
        assert set(fit_ev)=={'X','sample_ids'} and not set(tr['groups'])&set(ev['groups'])
        for k in fit_ev:np.testing.assert_array_equal(fit_ev[k],ev[k])
        for role,d in [('fit',tr),('test',ev)]:
            mp=source(old/'evaluation_input'/f'{year}_{role}_metadata_PRIVATE.parquet');shutil.copy2(mp,run/'evaluation_input'/mp.name)
            meta=pd.read_parquet(mp);np.testing.assert_array_equal(meta.sample_id,d['sample_ids'])
            assert d['X'].shape==(len(meta),83) and meta.group.is_unique
            if role=='fit':assert (meta.stop<pd.Timestamp(f'{year}-01-01')).all()
            counts.append(dict(year=year,role=role,companies=len(meta),termination=int((d['event']==1).sum()),
                recovery=int((d['event']==2).sum()),unknown24=int(((d['event']==0)&(d['analysis_time']<24)).sum()),changed_inputs=0,changed_labels=0))
        plan.pop('code_commit');plan['source_status']=protocol['source_status'];plan['pilot']['trees']=4
        write(dest/'plan.json',plan)
    refs=[]
    for y,v,m,s,p in outputs(old):
        for name,h in json.loads(source(p/'hashes.json').read_text()).items():assert sha(source(p/name))==h
        refs.append(dict(year=y,model=m,seed=s,path=str(p)))
        if m=='xai_q2':
            dest=run/'input/reference'/str(y)/str(s);dest.mkdir(parents=True)
            shutil.copy2(p/'forest_PRIVATE.json',dest/'forest_PRIVATE.json')
            write(dest/'identity.json',dict(source_sha256=sha(p/'forest_PRIVATE.json'),original_fitting_commit=json.loads((p/'summary.json').read_text())['code_commit']))
    assert len(refs)==12;write(run/'audit/reference_models.json',refs)
    pd.DataFrame(counts).to_csv(run/'reports/cohort_counts.csv',index=False)
    write(run/'audit/data_alignment.json',dict(status='PASS',source_status=protocol['source_status'],
        all_input_bytes_unchanged=True,all_labels_unchanged=True,training_company_union=345,training_termination_union=26,
        test_companies=81,test_terminations=32,unchanged_source_hashes=sources,data=counts))
    print(pd.DataFrame(counts).to_string(index=False))


def freeze(run,repo):
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True)
    commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    for p in (run/'input/folds').glob('*/plan.json'):
        d=json.loads(p.read_text());d['code_commit']=commit;p.write_text(json.dumps(d,indent=2)+'\n')
    shutil.copy2(run/'audit/protocol.json',run/'input/protocol.json')
    with (run/'audit/code.tar').open('xb') as f:subprocess.run(['git','-C',str(repo),'archive',commit],stdout=f,check=True)
    with tarfile.open(run/'audit/code.tar') as f:f.extractall(run/'input/code',filter='data')
    for stage in ['pilot','formal']:
        tasks=[dict(kind='cpu',argv=['python3',str(run/'input/code/experiments/post_st_leaf_shallow_v1/launch.py'),
            '--year',str(y),'--seed',str(s),'--depth',str(d),*(['--pilot'] if stage=='pilot' else [])],
            cwd=str(run/'input/code'),timeout=3600,
            storage=dict(source=str(run/'input'),mode='nfs',results=str(run/'results'/stage)))
            for y in [2023,2024] for d in [6,4] for s in ([42] if stage=='pilot' else [42,101,202])]
        (run/f'{stage}.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in tasks))
        (run/f'{stage}.py').write_text('# Independent jobs are declared in the heteropool sidecar.\n')
        write(run/f'{stage}.py.hetero.json',dict(independent=True,manifest=f'{stage}.jsonl'))
    write(run/'audit/immutable_input_hashes.json',{str(p.relative_to(run/'input')):sha(p) for p in (run/'input').rglob('*') if p.is_file()})
    print(commit)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--repo',type=Path)
    p.add_argument('--stage',choices=['prepare','freeze'],required=True);a=p.parse_args()
    prepare(a.run) if a.stage=='prepare' else freeze(a.run,a.repo)
