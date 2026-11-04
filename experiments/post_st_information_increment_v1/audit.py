"""Rebuild actual frozen vectors from the recorded as-of sources; separate audit-document evidence."""
from dataclasses import asdict
import shutil,re
from crforest import CompetingRiskForest
from experiments.post_st_prognosis_v1.prepare import market_features,OLD,LEGACY
from experiments.post_st_next_open_info_v1.prepare import financial_record,announcement_values,next_open
from .common import *

def main():
    r=run();root=r/'input';root.mkdir();(r/'eval').mkdir()
    old=source();f=features();f.to_csv(D/'FEATURE_SETS.csv',index=False)
    schema=read(CLOCK/'input/features.json');names=[a['feature_name'] for a in schema]
    keep=f.original_feature_id.to_numpy();assert [names[j] for j in keep]==f.english_name.tolist()
    base=Path(read(ROOT/'experiments/post_st_2025_shallow_parent_v1/frozen_protocol.json')['historical_cohort_run'])
    prior=read(history(2025)/'audit/data_alignment.json')['unchanged_source_hashes']
    digests={}
    def record(p,expected=None):
        p=Path(p);h=sha(p)
        if expected is not None:assert h==expected,p
        if str(p) in prior:assert h==prior[str(p)]
        digests[str(p)]=h;return p
    raw=LEGACY/'prepared/financial_facts.parquet'
    fin=pd.read_parquet(record(raw));fin['report_date']=pd.to_datetime(fin.Accper)
    np.testing.assert_array_equal(pd.to_datetime(fin.available),fin[['ann','correction_0','correction_1','correction_2']].apply(pd.to_datetime).max(axis=1))
    for col in ['available','prev_available','ann']:fin[col]=pd.to_datetime(fin[col])
    meta=pd.concat([pd.read_parquet(record(base/f'input/{role}_metadata_PRIVATE.parquet')) for role in ['train','eval']],ignore_index=True)
    fg={k:g for k,g in fin[fin.Stkcd.isin(meta.code)].groupby('Stkcd')}
    mk=pd.read_parquet(record(base/'private/market_source_subset_PRIVATE.parquet')).sort_values(['Stkcd','Trddt'])
    mg=dict(list(mk.groupby('Stkcd')));cal=np.load(record(base/'private/market_calendar_through2025_PRIVATE.npy'))
    ep=OLD/'phase1_data_freeze_v1/03_event_history.csv'
    events=pd.read_csv(record(ep),dtype=str,keep_default_na=False).set_index('event_id')
    listing=Path('/mnt/SSD2TB/pylib/3th/outputs_annual_report_extraction_v2_universe/annual_report_fields_document_level.csv')
    annual=pd.read_csv(record(listing),dtype={'code':str});annual.code=annual.code.str.zfill(6)
    pool=dict(list(annual.groupby(['code','report_year'])))
    info=pd.read_parquet(record(CLOCK/'audit/information_alignment_PRIVATE.parquet')).set_index('sample_id')
    fc=arrays(record(CLOCK/'audit/financial_availability_PRIVATE.npz'));dates=dict(zip(fc['sample_ids'],fc['dates']))
    rebuilt={};audit=[];fin_values=0;valid_market=0;fin_periods=[]
    for m in meta.itertuples(index=False):
        origin=next_open(m.origin,cal);sid=m.code+'_'+origin.strftime('%Y-%m-%d')+'_preopen';row=info.loc[sid]
        assert origin==row.origin and row.entry_announcement==m.origin
        ff,clock,fm=financial_record(fg[m.code],names[:48],origin)
        np.testing.assert_array_equal(clock,dates[sid]);assert ((np.isnat(clock))|(clock<np.datetime64(origin,'D'))).all()
        assert (~np.isfinite(ff)|~np.isnat(clock)).all()
        for j in [i for i,z in enumerate(names[:48]) if z.endswith(('_growth','_change')) and np.isfinite(ff[i])]:
            assert clock[j]==np.datetime64(max(fm['financial_available'],fm['prev_available']),'D')
        fin_values+=int(np.isfinite(ff[keep[keep<48]]).sum());fin_periods.append(fm['report_date'])
        mm=market_features(mg[m.code],cal,origin);assert mm['market_latest_date']==row.market_available<origin
        vals=np.array([mm[z] for z in names[48:72]],float);vals[~np.isfinite(vals)]=np.nan
        valid_market+=int(np.isfinite(vals[keep[(keep>=48)&(keep<72)]-48]).sum())
        v=mm['market_coverage_3m'];quality=[min((origin-fm['report_date']).days/365,2),min((origin-fm['financial_available']).days/365,2),1-np.clip(v if pd.notna(v) else 0,0,1)]
        extra,evidence=announcement_values(m._asdict(),events);assert evidence['available_at']==m.origin<origin
        rebuilt[sid]=np.r_[ff,vals,quality,extra][keep]
        docs=pool.get((m.code,m.origin.year-1),annual.iloc[:0]);pub=pd.to_datetime(docs.announce_date)
        timely=docs[(pub<origin)&(docs.is_revised==0)]
        original_facts=fg[m.code];annual_fact=original_facts[original_facts.report_date.eq(pd.Timestamp(m.origin.year-1,12,31))]
        iar=annual_fact.ann.min()
        entry=events.loc[str(m.entry_source_ids).split(';')]
        content=' '.join(entry.Content);op=evidence['raw_prior_year_audit_opinions'];gc=evidence['raw_going_concern_flags']
        if len(timely):
            for val in timely.path:record(val)
        opinion=timely.audit_opinion.replace({'带强调事项段的无保留意见':'带有解释性说明的无保留意见'})
        agree=opinion.eq(op)&timely.going_concern_flag.astype(str).eq(gc)
        audit.append(dict(sample_id=sid,year=m.origin.year,matched_document=bool(len(docs)),any_doc_before=bool((pub<origin).any()),
            first_indexed_doc_not_before=bool(len(pub) and pub.min()>=origin),original_before=bool(len(timely)),
            timely_extraction_pair_agrees=bool(agree.any()),prior_fiscal_IAR_before=bool(pd.notna(iar) and iar<origin),
            entry_text_present=bool(content.strip()),explicit_opinion_literal=op in content if op else False,mentions_going_concern='持续经营' in content,
            no_annual_or_entry_text=bool((pd.isna(iar) or iar>=origin) and not len(timely) and not content.strip())))
    assert len(rebuilt)==len(meta)==443
    aad=pd.DataFrame(audit);aad.to_parquet(r/'audit_sources_PRIVATE.parquet',index=False)
    # Vendor metadata have no independent opinion release, fiscal key or version. Even literal
    # matches or an automated report-text extractor cannot alone certify all seven fields.
    unresolved=(~aad.timely_extraction_pair_agrees)&(~aad.explicit_opinion_literal|~aad.mentions_going_concern)
    assert unresolved.any()
    historical=read(old/'audit/input_hashes.json');cohorts=[]
    for y in YEARS:
        dest=root/f'folds/{y}';dest.mkdir(parents=True)
        for role in ['train','eval']:
            key=f'folds/{y}/{role}_PRIVATE.npz';src=record(old/'input'/key,historical[key]);z=arrays(src)
            assert len(np.unique(z['sample_ids']))==len(z['X']);np.testing.assert_array_equal(z['X'],np.array([rebuilt[x] for x in z['sample_ids']]))
            shutil.copy2(src,dest/src.name)
        labp=record(old/f'evaluation_input/{y}_test_PRIVATE.npz');shutil.copy2(labp,r/f'eval/{y}.npz')
        tr=arrays(dest/'train_PRIVATE.npz');ev=arrays(dest/'eval_PRIVATE.npz');lab=arrays(labp)
        np.testing.assert_array_equal(ev['sample_ids'],lab['sample_ids']);np.testing.assert_array_equal(ev['X'],lab['X'][:,keep])
        assert not set(tr['groups'])&set(lab['groups']);assert len(set(lab['groups']))==len(lab['X'])
        tmeta=pd.read_parquet(record(CLOCK/f'audit/{y}_training_metadata_PRIVATE.parquet'))
        np.testing.assert_array_equal(tmeta.sample_id,tr['sample_ids']);assert (tmeta.stop<pd.Timestamp(y,1,1)).all() and (tmeta.origin<pd.Timestamp(y,1,1)).all()
        cohorts.append(dict(year=y,train=len(tr['groups']),evaluation=len(ev['X']),train_termination=int((tr['event']==1).sum()),train_recovery=int((tr['event']==2).sum()),overlap=0,train_sha256=sha(dest/'train_PRIVATE.npz'),eval_sha256=sha(dest/'eval_PRIVATE.npz')))
    pd.DataFrame(cohorts).to_csv(D/'COHORT_AUDIT.csv',index=False)
    registry=read(record(old/'input/old_registry.json')); hashdf=pd.read_csv(ROOT/'experiments/post_st_drop9_support_v1/MODEL_HASHES.csv',dtype={'seed':str})
    for key,path in registry.items():
        y,seed=map(int,key.split('|'));p=Path(path); h=read(record(p/'hashes.json'))
        for filename,digest in h.items():record(p/filename,digest)
        for ver in ['raw','shrunk']:
            model=CompetingRiskForest.load(p/ver/'forest_PRIVATE.json');assert asdict(model.config)==asdict(cfg(seed,74,'fixed25')) and model.n_features_==74
            s=hashdf[(hashdf.year==y)&(hashdf.arm=='drop9_support')&(hashdf.seed==str(seed))&(hashdf.version==ver)]
            assert len(s)==1
            for fn,col in [('forest_PRIVATE.json','forest_sha256'),('predictions_PRIVATE.npz','prediction_sha256'),('functions_PRIVATE.npz','function_sha256')]:assert sha(p/ver/fn)==s.iloc[0][col]
    assert len(registry)==9
    record(ROOT/'CURRENT_POST_ST_MODEL.json');record(ROOT/'主模型/配置/MODEL_SPEC.json');record(ROOT/'experiments/post_st_grouped_explanation_v1/FEATURE_GROUP_MAPPING.csv')
    frozen_draws=record(old/'private/bootstrap_indices_PRIVATE.npz');shutil.copy2(frozen_draws,r/'eval/bootstrap_PRIVATE.npz')
    vintage=pd.read_csv(record(OLD/'phase1b_unblock24m_v1/06_financial_pit_vintage_audit.csv'))
    acceptance=record(OLD/'phase1b2_targeted_recovery_v2/03_financial_source_acceptance.md')
    rr=dict(financial_clock_pass=True,market_clock_pass=True,matched_rows_pass=True,audit_release_pass=False,unique_companies=len(meta),
        finite_selected_financial_values_with_valid_clock=fin_values,finite_selected_market_values_rebuilt_from_prior_trades=valid_market,
        vendor_fields_known=443,prior_fiscal_year_document_probe={k:int(aad[k].sum()) for k in aad if k not in ['sample_id','year']},
        no_match_in_either_screen=int(unresolved.sum()),model_hashes_verified=9,
        finance_scope='Recorded financial disclosure plus latest recorded correction, both periods on growth. Frozen conservative source convention; researcher-accepted FS/IAR source, no claim of independently authenticated original financial values.',
        original_value_vintage_registry=dict(rows=len(vintage),verified=int(vintage.original_values_verified.sum())),audit_scope='Post-ST population selected at entry. Event-table timing and content verified separately. No complete dated primary evidence connecting prior fiscal report, opinion, going-concern status and version for these 443 rows. No annual mismatch alone proves leakage; earlier report or separate filing may exist. Zero imputed/changed rows.',
        market_scope='Daily raw data sliced strictly before next exchange open; coverage recomputed in same window.', independent_confirmation_available=False, registry='Remote catalog tunnel unavailable; local registered source dictionaries, FS/IAR facts, annual document index and actual ST event records inspected.')
    dump(D/'SOURCE_AND_TIME_AUDIT.json',rr);dump(root/'admissibility.json',rr);f.to_csv(root/'features.csv',index=False)
    dump(r/'sources_PRIVATE.json',digests)
if __name__=='__main__':main()
