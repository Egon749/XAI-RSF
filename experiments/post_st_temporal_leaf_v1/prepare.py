from pathlib import Path
import argparse,datetime,json,shutil,subprocess,tarfile
import numpy as np
import pandas as pd
from experiments.post_st_prognosis_v1.common import arrays,calendar_elapsed,sha,write
from experiments.post_st_leaf_shallow_v1.analyze import artifacts


def prepare(root):
    protocol=json.loads(Path(__file__).with_name('frozen_protocol.json').read_text());old=Path(protocol['reference_run'])
    write(root/'audit/protocol.json',protocol);sources={};counts=[]
    def source(p):sources[str(p)]=sha(p);return p
    for year in protocol['outer_years']:
        src=old/'input/folds'/str(year);dest=root/'input/formal'/str(year);dest.mkdir(parents=True)
        plan=json.loads(source(src/'plan.json').read_text());tr=arrays(source(src/'train_PRIVATE.npz'))
        meta=pd.read_parquet(source(old/'evaluation_input'/f'{year}_fit_metadata_PRIVATE.parquet'))
        np.testing.assert_array_equal(meta.sample_id,tr['sample_ids']);assert (meta.stop<pd.Timestamp(f'{year}-01-01')).all()
        np.savez_compressed(dest/'train_PRIVATE.npz',**tr)
        shutil.copy2(source(src/'eval_PRIVATE.npz'),dest/'eval_PRIVATE.npz')
        assert set(arrays(dest/'eval_PRIVATE.npz'))=={'X','sample_ids'}
        write(dest/'config.json',{**plan['forest_common'],**plan['forest_variants']['xai_q2'],'max_depth':4})
        for v in protocol['rolling_validation_entry_years']:
            if v>=year:continue
            folder=root/'input/validation'/str(v);folder.mkdir(parents=True,exist_ok=True)
            cutoff=pd.Timestamp(f'{v}-01-01')-pd.Timedelta(days=1)
            fit=meta.origin.dt.year.to_numpy()<v;val=meta.origin.dt.year.to_numpy()==v
            times=meta.time.to_numpy().copy();events=meta.event.to_numpy().copy()
            after=(meta.stop>cutoff).to_numpy();times[after]=(cutoff-meta.origin[after]).dt.days.to_numpy();events[after]=0
            train={k:a[fit].copy() for k,a in tr.items()};train['time']=times[fit];train['event']=events[fit]
            train['analysis_time']=np.array([calendar_elapsed(o,o+pd.Timedelta(days=int(t))) for o,t in zip(meta.origin[fit],times[fit])])
            assert (train['time']>0).all() and not set(train['groups'])&set(tr['groups'][val])
            evaluation=dict(X=tr['X'][val],sample_ids=tr['sample_ids'][val])
            for name,values in [('train_PRIVATE.npz',train),('eval_PRIVATE.npz',evaluation)]:
                if (folder/name).exists():
                    saved=arrays(folder/name);assert set(saved)==set(values)
                    for k in values:np.testing.assert_array_equal(saved[k],values[k])
                else:np.savez_compressed(folder/name,**values)
            labels={k:a[val] for k,a in tr.items() if k!='X'}
            np.savez_compressed(root/'private'/f'validation_{year}_{v}_labels_PRIVATE.npz',**labels)
            if not (folder/'config.json').exists():shutil.copy2(dest/'config.json',folder/'config.json')
            else:assert (dest/'config.json').read_bytes()==(folder/'config.json').read_bytes()
            counts.append(dict(outer_year=year,validation_entry_year=v,training_companies=int(fit.sum()),training_termination=int((events[fit]==1).sum()),
                training_recovery=int((events[fit]==2).sum()),validation_companies=int(val.sum()),max_fit_stop=str(cutoff.date()),outcome_cutoff_exclusive=f'{year}-01-01'))
        for yy,arm,seed,path in artifacts(old):
            if yy!=year or arm not in ['shallow','shallow_parent']:continue
            target=dest/'reference'/str(seed)/arm;target.mkdir(parents=True)
            hashes=json.loads(source(path/'hashes.json').read_text())
            for name in ['forest_PRIVATE.json','predictions_PRIVATE.npz']:
                assert sha(source(path/name))==hashes[name];shutil.copy2(path/name,target/name)
    prior=Path(__file__).resolve().parents[2]/'experiments/post_st_train2024_h12_v1/execution_receipt.json'
    receipt=json.loads(source(prior).read_text());assert receipt['status']=='COMPLETE' and receipt['test_entry_year']==2025
    write(root/'audit/exposure_2025.json',dict(status='PREVIOUSLY_EXPOSED_NO_NEW_READ',prior_receipt=str(prior),prior_sha256=sha(prior),
        prior_status=receipt['status'],prior_training_commit=receipt['training_commit'],formal_fits_succeeded=receipt['formal_fits_succeeded'],new_2025_evaluation=False))
    pd.DataFrame(counts).to_csv(root/'reports/rolling_fold_counts.csv',index=False)
    write(root/'audit/pre_fit_freeze.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol_sha256=sha(Path(__file__).with_name('frozen_protocol.json')),
        step0_done=True,new_test_predictions_seen=False,new_test_metadata_read=False,source_sha256=sources))
    write(root/'input/protocol.json',protocol)
    print(pd.DataFrame(counts).to_string(index=False))


def freeze(root):
    repo=Path(__file__).resolve().parents[2]
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True)
    commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    archive=root/'audit/code.tar'
    paths=['crforest','experiments/post_st_temporal_leaf_v1','experiments/post_st_prognosis_v1/common.py','experiments/post_st_leaf_shallow_v1/worker.py']
    with archive.open('xb') as f:subprocess.run(['git','-C',str(repo),'archive',commit,*paths],stdout=f,check=True)
    with tarfile.open(archive) as t:t.extractall(root/'input/code',filter='data')
    write(root/'input/code_identity.json',dict(commit=commit,archive_sha256=sha(archive)))
    protocol=json.loads((root/'input/protocol.json').read_text())
    for stage in ['pilot','validation','formal']:
        tasks=[]
        for y in ([protocol['outer_years'][0]] if stage=='validation' else protocol['outer_years']):
            vals=protocol['rolling_validation_entry_years'] if stage=='validation' else [0]
            for v in vals:
                if v>=y:continue
                for seed in ([42] if stage=='pilot' else protocol['seeds']):
                    tasks.append(dict(kind='cpu',argv=['python3',str(root/'input/code/experiments/post_st_temporal_leaf_v1/launch.py'),
                        '--stage',stage,'--year',str(y),'--validation-year',str(v),'--seed',str(seed)],cwd=str(root/'input/code'),timeout=3600,
                        storage=dict(source=str(root/'input'),mode='nfs',results=str(root/'results'/stage))))
        (root/f'{stage}.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in tasks))
        (root/f'{stage}.py').write_text('# Independent commands in sidecar; no duplicate full-script execution.\n')
        write(root/f'{stage}.py.hetero.json',dict(independent=True,manifest=f'{stage}.jsonl'))
    write(root/'audit/immutable_input_hashes.json',{str(p.relative_to(root/'input')):sha(p) for p in (root/'input').rglob('*') if p.is_file()})
    print(commit)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--stage',choices=['prepare','freeze'],required=True);a=p.parse_args()
    prepare(a.run) if a.stage=='prepare' else freeze(a.run)
