import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest
from crforest.core import ForestConfig
from crforest.transparent import write_json,TransparentCIF
from experiments.transparent_cif_v2.run import training_contract,feature_contract,calendar_horizons


def make_source(root):
    root.mkdir();rng=np.random.default_rng(333);n=500;X=rng.choice([-1.,1.],(n,4))
    p=.2+.08*X[:,0]*X[:,1];u=rng.random(n)
    e=np.where(u<p,1,np.where(u<p+.1,2,0));t=np.where(e>0,rng.integers(10,300,n),365).astype(float)
    origins=pd.to_datetime(['2018-01-01']*n);end=origins+pd.to_timedelta(t,unit='D')
    groups=np.arange(n).astype(str);ids=np.array([f'id{i}' for i in range(n)])
    arrays=dict(X=X,time=t,event=e,groups=groups,sample_ids=ids,horizon=np.full(n,365))
    np.savez_compressed(root/'final_train_PRIVATE.npz',**arrays)
    d=pd.DataFrame(dict(sample_id=ids,group=groups,origin=origins,stop_date=end,
                        outcome_available=end+pd.Timedelta(days=1),event=e,label_eligible=True))
    d.to_csv(root/'final_train_metadata_PRIVATE.csv',index=False)
    cfg=ForestConfig(n_trees=5,path_budget=2,min_leaf=10,min_groups=5,min_events=1,
                     mtry=4,cut_strategy='shared_quantile',n_thresholds=3)
    write_json(root/'plan.json',{'forest_configs':{'q2':asdict(cfg)}})
    # Evaluation bundle intentionally contains NO outcome arrays.
    np.savez_compressed(root/'final_eval_PRIVATE.npz',X=X[:80],sample_ids=ids[:80],groups=groups[:80],horizon=np.full(80,365))
    d.loc[:79,['sample_id','origin']].to_csv(root/'final_eval_metadata_PRIVATE.csv',index=False)
    fields=[dict(name=f'feature{i}',unit='synthetic',definition='synthetic input',transform='identity') for i in range(4)]
    write_json(root/'features.json',fields)
    return arrays,d


def test_actual_calendar_not_30_day_months():
    assert calendar_horizons(['2019-12-31','2020-08-31'])[:,0].tolist()==[366.,365.]


def test_time_contract_rejects_future_and_normalized_duration(tmp_path):
    src=tmp_path/'src';a,d=make_source(src)
    training_contract(src,'2022-01-01','csv')
    d.loc[0,'outcome_available']=pd.Timestamp('2022-01-01')
    d.to_csv(src/'final_train_metadata_PRIVATE.csv',index=False)
    with pytest.raises(ValueError,match='cutoff'):training_contract(src,'2022-01-01','csv')
    d.loc[0,'outcome_available']=pd.Timestamp('2021-01-01');d.to_csv(src/'final_train_metadata_PRIVATE.csv',index=False)
    a['time']=a['time']*12/365;np.savez_compressed(src/'final_train_PRIVATE.npz',**a)
    with pytest.raises(ValueError,match='actual days'):training_contract(src,'2022-01-01','csv')


def test_unknown_endpoint_contract_preserved(tmp_path):
    src=tmp_path/'src';a,d=make_source(src)
    row=int(np.flatnonzero(a['event']==0)[0]);d.loc[row,'label_eligible']=False
    d.to_csv(src/'final_train_metadata_PRIVATE.csv',index=False)
    _,_,_,elig=training_contract(src,'2022-01-01','csv')
    assert not elig[row,0]


def test_feature_semantics_required(tmp_path):
    src=tmp_path/'src';a,d=make_source(src)
    feature_contract(src/'features.json',4)
    bad=json.loads((src/'features.json').read_text());bad[0]['unit']=''
    (src/'features.json').write_text(json.dumps(bad))
    with pytest.raises(ValueError):feature_contract(src/'features.json',4)


def test_complete_cli_training_prediction_without_evaluation_labels(tmp_path):
    src=tmp_path/'src';make_source(src)
    cfg=json.loads(Path('experiments/transparent_cif_v2/config.json').read_text())
    cfg['pair'].update(max_pairs=6,max_trees=3,max_cuts=2)
    write_json(tmp_path/'config.json',cfg)
    tr=tmp_path/'train';pred=tmp_path/'pred'
    cmd=[sys.executable,'-m','experiments.transparent_cif_v2.run','train','--source-input',str(src),
         '--metadata-format','csv','--features',str(src/'features.json'),'--config',str(tmp_path/'config.json'),'--output',str(tr)]
    result=subprocess.run(cmd,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    index=json.loads((tr/'model_index.json').read_text());assert len(index)==6
    receipt=json.loads((tr/'execution_receipt.json').read_text())
    assert receipt['weight_qp_solves']==3 and receipt['roles']['structure_groups']+receipt['roles']['weight_groups']==500
    result=subprocess.run([sys.executable,'-m','experiments.transparent_cif_v2.run','predict',
        '--source-input',str(src),'--metadata-format','csv','--trained',str(tr),'--output',str(pred)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    pr=json.loads((pred/'prediction_receipt.json').read_text());assert pr['labels_read'] is False
    for row in pr['models']:
        assert row['full_reconstruction_max_error']<1e-10
        with np.load(pred/f'{row["model"]}_PRIVATE.npz') as f:
            assert f['risk'].shape==(80,) and f['cif12'].shape==(80,2)
    result=subprocess.run(cmd,capture_output=True,text=True)
    assert result.returncode!=0  # no overwriting completed output


def test_parquet_matches_csv_if_engine_available(tmp_path):
    pytest.importorskip('pyarrow',reason='Parquet integration requires repository dependency pyarrow')
    src=tmp_path/'src';a,d=make_source(src)
    d.to_parquet(src/'final_train_metadata_PRIVATE.parquet',index=False)
    csv=training_contract(src,'2022-01-01','csv')
    parquet=training_contract(src,'2022-01-01','parquet')
    np.testing.assert_array_equal(csv[2],parquet[2])
    np.testing.assert_array_equal(csv[3],parquet[3])
