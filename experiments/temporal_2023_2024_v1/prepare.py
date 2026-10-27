"""Recover frozen final split and audited feature clocks; no fitting or tuning."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import numpy as np
import pandas as pd

LEGACY=Path('/home/oliver/MainDiskData/early_warning_12m_server_isolated_20261025')
DEV=Path('/home/oliver/MainDiskData/xai_rsf_search_coverage_v1_20261027_035011')
OLD=Path('/home/oliver/LLMworkspace/early-warning-12m-model-comparison-2012plus')


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    with Path(p).open('x') as f:json.dump(d,f,indent=2,ensure_ascii=False,allow_nan=False)


def main(run):
    inp=run/'input';inp.mkdir();(run/'audit').mkdir();(run/'reports').mkdir();(run/'private').mkdir()
    protocol=Path(__file__).with_name('frozen_protocol.json')
    shutil.copy2(protocol,run/'audit/frozen_protocol.json')
    write(run/'audit/pre_outcome_freeze.json',{'protocol_sha256':sha(protocol),'statement':'Copied frozen config/protocol before this program reads requested evaluation outcomes.'})
    src=LEGACY/'input'
    expected=json.loads((LEGACY/'reports/frozen_input_hashes.json').read_text())
    signatures={}
    for name in ['metadata.parquet','features.npy','quality.npy','snapshot_indices.npy','splits.json','model_matrix.json']:
        p=src/name;signatures[str(p)]=sha(p);assert signatures[str(p)]==expected[name]
    splits=json.loads((src/'splits.json').read_text());split=splits['final']
    assert split['fit_cutoff']=='2022-01-01'
    matrix=json.loads((src/'model_matrix.json').read_text())
    assert matrix['folds']['final']=={'train_end':2020,'cutoff':'2022-01-01','eval_years':[2022,2023,2024]}
    # Read only year/sample identity for the complete index; materialize no 2022 labels.
    identity=pd.read_parquet(src/'metadata.parquet',columns=['year','sample_id'])
    wanted=identity.year.le(2020)|identity.year.isin([2023,2024])
    global_rows=np.flatnonzero(wanted)
    d=pd.read_parquet(src/'metadata.parquet',filters=[[('year','<=',2020)],[('year','in',[2023,2024])]])
    assert np.array_equal(d.sample_id.to_numpy(),identity.loc[wanted,'sample_id'].to_numpy())
    d.index=global_rows
    train_ids=np.asarray(split['train']);eval_ids=np.asarray([i for i in split['eval'] if identity.iloc[i].year in [2023,2024]])
    assert set(train_ids)<=set(d.index) and set(eval_ids)<=set(d.index)
    ids=np.r_[train_ids,eval_ids];rows=d.loc[ids].reset_index(drop=True)
    ntrain=len(train_ids);tr=rows.iloc[:ntrain];ev=rows.iloc[ntrain:]
    assert tr.year.min()==2012 and tr.year.max()==2020 and set(ev.year)=={2023,2024}
    cutoff=pd.Timestamp('2022-01-01')
    assert (tr.stop_date<cutoff).all() and (tr.outcome_available<cutoff).all() and (tr.duration>0).all()
    assert not rows.duplicated(['group','quarter']).any()
    assert not set(tr.loc[tr.event==1,'event_id'])&set(ev.loc[ev.event==1,'event_id'])
    prior_hashes=json.loads(Path('/home/oliver/LLMworkspace/newRSF2/experiments/real_development_v1/source_artifact_hashes.json').read_text())
    paths=sorted((LEGACY/'results/market_features').glob('*/*/task-*/output/market_features.parquet'))
    assert len(paths)==16
    for p in [LEGACY/'prepared/financial.parquet',LEGACY/'prepared/financial_facts.parquet',*paths]:
        signatures[str(p)]=sha(p);assert signatures[str(p)]==prior_hashes[str(p)]
    finance=pd.read_parquet(LEGACY/'prepared/financial.parquet')
    market=pd.concat([pd.read_parquet(p) for p in paths],ignore_index=True)
    merged=finance.merge(market,on=['code','quarter'],validate='one_to_one').sort_values(['code','quarter']).reset_index(drop=True)
    snap=np.load(src/'snapshot_indices.npy',allow_pickle=False)
    values=np.load(src/'features.npy',mmap_mode='r',allow_pickle=False)
    qualities=np.load(src/'quality.npy',mmap_mode='r',allow_pickle=False)
    z=merged.iloc[snap[ids]].reset_index(drop=True)
    assert np.array_equal((z.code+'_'+z.quarter).to_numpy(),rows.sample_id.to_numpy())
    names=matrix['features'];X=np.array(values[snap[ids]],dtype=float)
    assert len(names)==72 and np.array_equal(X,z[names].to_numpy(np.float32),equal_nan=True)
    facts=pd.read_parquet(LEGACY/'prepared/financial_facts.parquet')
    dates=facts[['ann','correction_0','correction_1','correction_2']].apply(pd.to_datetime).max(axis=1)
    assert (dates==pd.to_datetime(facts.available)).all()
    lookup=facts.set_index(['Stkcd','Accper']);assert lookup.index.is_unique
    has=z.available.notna();cur=lookup.reindex(pd.MultiIndex.from_frame(z.loc[has,['code','Accper']]))
    assert np.array_equal(cur.available.to_numpy(dtype='datetime64[D]'),z.loc[has,'available'].to_numpy(dtype='datetime64[D]'))
    prevperiod=(pd.to_datetime(z.Accper)-pd.DateOffset(years=1)).dt.strftime('%Y-%m-%d')
    prev=lookup.reindex(pd.MultiIndex.from_arrays([z.code,prevperiod]))
    assert np.array_equal(prev.available.to_numpy(dtype='datetime64[D]'),z.prev_available.to_numpy(dtype='datetime64[D]'),equal_nan=True)
    fin=pd.to_datetime(z.available).to_numpy(dtype='datetime64[D]')
    prevdate=pd.to_datetime(z.prev_available).to_numpy(dtype='datetime64[D]')
    marketdate=pd.to_datetime(z.market_latest_date,errors='coerce').to_numpy(dtype='datetime64[D]')
    origin=rows.origin.to_numpy(dtype='datetime64[D]')
    clocks=np.full(X.shape,np.datetime64('NaT'),dtype='datetime64[D]')
    for j,name in enumerate(names):
        clocks[:,j]=np.maximum(fin,prevdate) if j<48 and name.endswith(('_growth','_change')) else fin if j<48 else marketdate
    assert not (np.isfinite(X)&np.isnat(clocks)).any(),'No invented content dates allowed'
    # Identical inherited masking semantics: actual source dates at/after origin are unavailable.
    future=(~np.isnat(clocks))&(clocks>=origin[:,None])
    masked=np.isfinite(X)&future;X[future]=np.nan
    assert not (np.isfinite(X)&(np.isnat(clocks)|future)).any()
    q=np.column_stack([np.minimum(z.fin_period_age_days.fillna(730)/365,2),np.minimum(z.fin_age_days.fillna(730)/365,2),1-z.market_coverage_3m.fillna(0).clip(0,1)]).astype(np.float32)
    assert np.array_equal(q,qualities[snap[ids]])
    qc=np.column_stack([fin,fin,marketdate]);assert not ((~np.isnat(qc))&(qc>=origin[:,None])).any()
    X=np.column_stack([X,q]);assert X.shape[1]==75 and not np.isinf(X).any()
    # Preprocessing parity against all previously used V20 training records.
    with np.load(DEV/'input/V20_train_PRIVATE.npz',allow_pickle=False) as old:
        positions=pd.Index(rows.sample_id).get_indexer(old['sample_ids'])
        assert (positions>=0).all()
        assert np.array_equal(X[positions],old['X'],equal_nan=True),'Prior frozen preprocessing changed'
        assert np.array_equal(rows.iloc[positions].event.to_numpy(),old['event'])
        assert np.array_equal((rows.iloc[positions].stop_date-rows.iloc[positions].origin).dt.days.to_numpy(),old['time'])
        parity_count=len(positions)
    assert (rows.outcome_available>=rows.stop_date).all()
    events=rows.event>0
    assert (rows.loc[events,'event_date']==rows.loc[events,'stop_date']).all()
    assert (rows.loc[events,'outcome_available']==rows.loc[events,['event_date','announcement_date']].max(axis=1)).all()
    counts={}
    for role,ix in [('train',np.arange(ntrain)),('eval',np.arange(ntrain,len(rows)))]:
        rr=rows.iloc[ix].reset_index(drop=True);t=(rr.stop_date-rr.origin).dt.days.to_numpy();h=(rr.horizon-rr.origin).dt.days.to_numpy()
        assert set(h)<={365,366} and (t>=0).all() and (t<=h).all()
        np.savez_compressed(inp/f'final_{role}_PRIVATE.npz',X=X[ix],time=t,event=rr.event.to_numpy(),groups=rr.group.to_numpy(dtype=str),sample_ids=rr.sample_id.to_numpy(dtype=str),horizon=h)
        rr.to_parquet(inp/f'final_{role}_metadata_PRIVATE.parquet',index=False)
        def count(dd):
            return dict(records=len(dd),groups=int(dd.group.nunique()),target_records=int((dd.event==1).sum()),target_events=int(dd.loc[dd.event==1,'event_id'].nunique()),competing_records=int((dd.event==2).sum()),competing_events=int(dd.loc[dd.event==2,'event_id'].nunique()),zero_followup=int(((dd.stop_date-dd.origin).dt.days==0).sum()),origin_unresolved=int((dd.origin_state!='N_observed').sum()),unknown_endpoint_records=int(((dd.event==0)&~dd.label_eligible).sum()))
        counts[f'final_{role}']=count(rr)
        if role=='eval':
            for year,dd in rr.groupby('year'):counts[str(year)]=count(dd)
    probs=[]
    for seed in [42,101,202]:
        pp=list((LEGACY/'results/final'/f'final_M2_{seed}').glob('*/task-*/output/predictions.npz'));assert len(pp)==1
        signatures[str(pp[0])]=sha(pp[0])
        with np.load(pp[0],allow_pickle=False) as a:
            assert np.array_equal(a['indices'],split['eval'])
            pos=pd.Index(a['indices']).get_indexer(eval_ids);assert (pos>=0).all()
            pred=a['probabilities'][pos];assert np.isfinite(pred).all() and pred.min()>=0 and np.max(abs(pred.sum(2)-1))<1e-6
            probs.append(pred[:,-1,1])
    np.savez_compressed(inp/'M2_PRIVATE.npz',sample_ids=ev.sample_id.to_numpy(dtype=str),probabilities=np.array(probs),ensemble=np.mean(probs,axis=0),seeds=np.array([42,101,202]))
    helpers=inp/'legacy';helpers.mkdir()
    for name in ['common.py','evaluate.py','artifacts.py']:
        source=OLD/'experiments/early_warning_12m'/name;shutil.copy2(source,helpers/name);signatures[str(source)]=sha(source)
    source=Path('/home/oliver/LLMworkspace/newRSF2/experiments/real_development_v1/endpoint_guard.py')
    shutil.copy2(source,helpers/source.name);signatures[str(source)]=sha(source)
    for p,h in signatures.items():assert sha(p)==h
    protocol=json.loads(protocol.read_text());protocol.update(source_counts=counts,folds=['final'],source_hashes={str(p.relative_to(inp)):sha(p) for p in inp.rglob('*') if p.is_file()},data_status='PROVISIONAL')
    write(inp/'plan.json',protocol)
    audit=dict(status='PROVISIONAL',source_counts=counts,fit_origin_years=[int(tr.year.min()),int(tr.year.max())],fit_cutoff='2022-01-01',latest_training_stop=str(tr.stop_date.max().date()),latest_training_outcome_available=str(tr.outcome_available.max().date()),quality_unknown_clock_cells=np.isnat(qc).sum(0).tolist(),masked_future_content_cells=int(masked.sum()),prior_V20_training_rows_exactly_reproduced=parity_count,source_hashes=signatures,original_splits_hash=sha(src/'splits.json'),test_years=[2023,2024],data_vintages_certified=False,events_previously_exposed=True)
    write(run/'audit/data_alignment.json',audit)
    print(json.dumps({k:v for k,v in audit.items() if k!='source_hashes'},ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);main(p.parse_args().run)
