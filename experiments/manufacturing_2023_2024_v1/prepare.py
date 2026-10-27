"""Read-only report-linked industry audit and manufacturing restriction.

Run with the existing system Python (pandas/pyarrow/python_calamine installed).
No field here asserts an industry publication date or certified PIT vintage.
"""
from pathlib import Path
import argparse,hashlib,json,shutil,datetime
import numpy as np
import pandas as pd
from python_calamine import CalamineWorkbook

PARENT=Path('/home/oliver/MainDiskData/xai_rsf_temporal_2023_2024_v1_20261027_sWV4S5D1')
LEGACY=Path('/home/oliver/MainDiskData/early_warning_12m_server_isolated_20261025')
CACHE=Path('/home/oliver/LLMworkspace/csmar_financial_diagnosis_audit_20261012/industry_raw_cache.parquet')
SOURCE=Path('/home/oliver/pylib/csmar/盈利能力/FI_T5.xlsx')
DICTIONARY=Path('/home/oliver/pylib/csmar/盈利能力/FI_T5[DES][xlsx].txt')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    with Path(p).open('x') as f:json.dump(d,f,indent=2,ensure_ascii=False,allow_nan=False)
def counts(d):
    return dict(records=len(d),groups=int(d.group.nunique()),target_records=int((d.event==1).sum()),
                target_events=int(d.loc[d.event==1,'event_id'].nunique()),competing_records=int((d.event==2).sum()),
                competing_events=int(d.loc[d.event==2,'event_id'].nunique()),
                unknown_endpoint_records=int(((d.event==0)&~d.label_eligible).sum()),
                origin_unresolved=int((d.origin_state!='N_observed').sum()),zero_followup=int((d.stop_date==d.origin).sum()))

def main(run):
    for name in ['input','audit','reports','private']:(run/name).mkdir()
    inp=run/'input';protocol=Path(__file__).with_name('frozen_protocol.json')
    shutil.copy2(protocol,run/'audit/frozen_protocol.json')
    write(run/'audit/pre_outcome_freeze.json',dict(protocol_sha256=sha(protocol),created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    signatures={str(p):sha(p) for p in [CACHE,SOURCE,DICTIONARY,LEGACY/'prepared/financial.parquet',PARENT/'input/plan.json']}
    cache=pd.read_parquet(CACHE);columns=list(cache.columns)
    # Reconcile the convenient cache against the original export, not its label.
    wb=CalamineWorkbook.from_path(str(SOURCE));it=wb.get_sheet_by_index(0).iter_rows()
    header=next(it);next(it);next(it);indices=[header.index(c) for c in columns]
    raw=pd.DataFrame.from_records((tuple(r[i] for i in indices) for r in it),columns=columns)
    raw=raw[raw.Typrep.eq('A')].reset_index(drop=True).fillna('').astype(str)
    industry=cache[cache.Typrep.eq('A')].reset_index(drop=True).fillna('').astype(str)
    assert raw.equals(industry), 'Industry cache differs from original consolidated export'
    del wb,raw,cache
    assert industry.Stkcd.str.fullmatch(r'\d{6}').all()
    assert not industry.duplicated(['Stkcd','Accper']).any()
    industry['report_date']=pd.to_datetime(industry.Accper,errors='raise')
    industry['industry_field']=np.where(industry.report_date<pd.Timestamp('2023-05-01'),'Indcd','Indcd1')
    industry['industry_code']=np.where(industry.industry_field.eq('Indcd'),industry.Indcd,industry.Indcd1)
    industry['valid_industry']=industry.industry_code.str.fullmatch(r'[A-S]\d{1,3}')
    industry['manufacturing']=industry.valid_industry&industry.industry_code.str.startswith('C')
    fin=pd.read_parquet(LEGACY/'prepared/financial.parquet',columns=['code','quarter','Accper','available'])
    fin['Accper']=pd.to_datetime(fin.Accper).dt.strftime('%Y-%m-%d')
    assert not fin.duplicated(['code','quarter']).any()
    plan=json.loads(protocol.read_text());source_counts={};selection={};exclusion=[]
    for role in ['train','eval']:
        for suffix in ['_PRIVATE.npz','_metadata_PRIVATE.parquet']:
            p=PARENT/'input'/f'final_{role}{suffix}';signatures[str(p)]=sha(p)
        d=pd.read_parquet(PARENT/'input'/f'final_{role}_metadata_PRIVATE.parquet').reset_index(drop=True)
        joined=d[['code','quarter','sample_id','origin','year']].merge(fin,on=['code','quarter'],how='left',validate='one_to_one',sort=False)
        joined=joined.merge(industry[['Stkcd','Accper','report_date','industry_field','industry_code','valid_industry','manufacturing']],left_on=['code','Accper'],right_on=['Stkcd','Accper'],how='left',validate='many_to_one',sort=False)
        assert np.array_equal(joined.sample_id,d.sample_id)
        report_known=joined.available.notna()&(joined.available<joined.origin)&(joined.report_date<joined.origin)
        class_known=report_known&joined.valid_industry.fillna(False)
        keep=class_known&joined.manufacturing.fillna(False);selection[role]=np.flatnonzero(keep)
        joined['selected']=keep;joined['report_known']=report_known;joined['class_known']=class_known
        joined.to_parquet(run/'private'/f'{role}_industry_alignment_PRIVATE.parquet',index=False)
        with np.load(PARENT/'input'/f'final_{role}_PRIVATE.npz',allow_pickle=False) as z:
            assert np.array_equal(z['sample_ids'],d.sample_id)
            np.savez_compressed(inp/f'final_{role}_PRIVATE.npz',**{k:z[k][keep] for k in z.files})
        dd=d.loc[keep].reset_index(drop=True);dd.to_parquet(inp/f'final_{role}_metadata_PRIVATE.parquet',index=False)
        source_counts[f'final_{role}']=counts(dd)
        if role=='train':
            assert dd.year.min()==2012 and dd.year.max()==2020
            assert (dd.outcome_available<pd.Timestamp('2022-01-01')).all() and (dd.stop_date<pd.Timestamp('2022-01-01')).all()
        else:
            assert set(dd.year)=={2023,2024}
            for y,g in dd.groupby('year'):source_counts[str(y)]=counts(g)
        for year,ix in d.groupby('year').indices.items():
            exclusion.append(dict(role=role,year=int(year),original_records=len(ix),selected_manufacturing=int(keep.iloc[ix].sum()),known_nonmanufacturing=int((class_known&~keep).iloc[ix].sum()),unclassified=int((~class_known).iloc[ix].sum())))
        assert len(dd)>0 and set(dd.event)=={0,1,2}
    pd.DataFrame(exclusion).to_csv(run/'audit/industry_selection_counts.csv',index=False)
    m=PARENT/'input/M2_PRIVATE.npz';signatures[str(m)]=sha(m)
    with np.load(m,allow_pickle=False) as z:
        ix=selection['eval'];np.savez_compressed(inp/'M2_PRIVATE.npz',sample_ids=z['sample_ids'][ix],probabilities=z['probabilities'][:,ix],ensemble=z['ensemble'][ix],seeds=z['seeds'])
    # Preserve all-sector predictions on the exact same cohort for the refit contrast.
    for p in (PARENT/'results/assembled').glob('*/task-*/output/summary.json'):
        s=json.loads(p.read_text());pred=p.parent/'predictions_PRIVATE.npz';signatures[str(pred)]=sha(pred)
        with np.load(pred,allow_pickle=False) as z:
            np.savez_compressed(inp/f'allsector_{s["variant"]}_{s["seed"]}_PRIVATE.npz',**{k:z[k][selection['eval']] for k in z.files})
    shutil.copytree(PARENT/'input/legacy',inp/'legacy')
    shutil.copy2(PARENT/'audit/M2_provenance.json',run/'audit/M2_provenance.json')
    for p,h in signatures.items():assert sha(p)==h
    plan.update(source_counts=source_counts,folds=['final'],source_hashes={str(p.relative_to(inp)):sha(p) for p in inp.rglob('*') if p.is_file()},data_status='PROVISIONAL_REPORT_LINKED_MANUFACTURING')
    write(inp/'plan.json',plan)
    audit=dict(status='PROVISIONAL_REPORT_LINKED_MANUFACTURING',source_counts=source_counts,source_hashes=signatures,
               industry_original_A_rows=len(industry),cache_exactly_reconciled_to_original=True,industry_availability_certified=False,
               rule=plan['industry_rule'],source_selection_parent=str(PARENT),selection_counts=exclusion,
               classification_not_model_feature=True,prior_input_columns=75,allsector_model_training_records=87134,
               disclosures='No true industry publication/effective-version timestamps. Financial availability belongs only to the matched report. No invented industry availability.',
               test_years=[2023,2024],strict_blind=False)
    write(run/'audit/data_alignment.json',audit)
    print(json.dumps(source_counts,ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);main(p.parse_args().run)
