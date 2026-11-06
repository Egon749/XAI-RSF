"""Build annual normal-company warning cohorts from local CSMAR sources.

Historical audits are used only for raw-matched scope, announcement corrections
and already-adjudicated event reasons, never as a replacement feature matrix.
"""
from pathlib import Path
import argparse,json,re,time
import numpy as np
import pandas as pd
from .raw_scan import sha,MARKET
from .clock import stop_at,available_values
from experiments.post_st_prognosis_v1.prepare import market_features

NORMAL={1,4,7,10,13}
RISK={2,3,5,6,8,9,11,12,14,15,16}
FIN={'01','02','03','04','05','08'}

def dump(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,default=str))
def name_state(s):
    s=str(s).replace(' ','').upper()
    return 'D' if '*ST' in s else 'B' if 'ST' in s else 'T' if '退' in s else 'A'

def source_events(root,run):
    audit={};p=root/'特殊处理变动文件/BDT_Trdchg.json'
    raw=pd.read_json(p,lines=True,dtype=False).fillna('').astype(str)
    raw.Symbol=raw.Symbol.str.zfill(6);audit[str(p)]=sha(p)
    old=Path('/home/oliver/LLMworkspace/PhD_STARST_Prognosis/runs')
    sp=old/'method_freeze_input_build_v2/evidence/source_scope_confirmation.json'
    scope=json.loads(sp.read_text());audit[str(sp)]=sha(sp)
    original=Path(scope['parsed_source']['path']);assert sha(original)==scope['parsed_source']['sha256']
    oldraw=pd.read_json(original,lines=True,dtype=False).fillna('').astype(str);oldraw.Symbol=oldraw.Symbol.str.zfill(6)
    key=['Symbol','Chgtype','Execudt']
    x=raw.sort_values(key).reset_index(drop=True);y=oldraw.sort_values(key).reset_index(drop=True)
    for c in ['Symbol','Chgtype','Execudt','Annoudt','Content','Chgreas','Chgrsdis']:
        assert x[c].equals(y[c]),'Local raw snapshot differs from previously accepted scope: '+c
    assert len(raw)==4158 and scope['population']=='all_enterprises_excluding_financial_institutions'
    bound=pd.Timestamp(scope['primary_snapshot_execution_bound'])
    ep=old/'phase1_data_freeze_v1/03_event_history.csv';eh=pd.read_csv(ep,dtype=str,keep_default_na=False)
    eh=eh[eh.source_role.eq('latest_primary')];assert len(eh)==len(raw)
    eh=eh.sort_values(key).reset_index(drop=True)
    assert eh[key].equals(x[key]);assert eh.Content.equals(x.Content)
    x['announcement']=pd.to_datetime(eh.adjudicated_announcement_date.where(eh.adjudicated_announcement_date.ne(''),x.Annoudt),errors='coerce')
    x['execution']=pd.to_datetime(x.Execudt,errors='coerce');x['source']='local_BDT'
    audit[str(ep)]=sha(ep);audit['announcement_corrections']=int(eh.adjudicated_announcement_date.ne('').sum())
    ip=root/'上市公司上市状态变更表/STK_ITEMCHANGE.json';item=pd.read_json(ip,lines=True,dtype=False).fillna('').astype(str)
    item.Symbol=item.Symbol.str.zfill(6);audit[str(ip)]=sha(ip)
    parent={}
    def find(c):
        parent.setdefault(c,c)
        if parent[c]!=c:parent[c]=find(parent[c])
        return parent[c]
    for _,g in item[item.InstitutionID.ne('')].groupby('InstitutionID'):
        codes=sorted(set(g.Symbol));r=find(codes[0])
        for c in codes[1:]:parent[find(c)]=r
    for c in set(x.Symbol)|set(item.Symbol):find(c)
    mapping={c:find(c) for c in parent}
    known=set(zip(x.Symbol,x.Chgtype,x.Execudt));extra=[]
    for z in item.itertuples():
        if z.ChangedItem!='证券简称':continue
        typ=name_state(z.ValueBefore)+name_state(z.ValueAfter)
        if typ not in ['AB','AD'] or (z.Symbol,typ,z.ChangeDate) in known:continue
        # Item table has an explicit declaration date. Never substitute ChangeDate.
        r=re.findall(r'S15(\d{2})',z.ReasonID)
        extra.append(dict(Symbol=z.Symbol,Chgtype=typ,Execudt=z.ChangeDate,Annoudt=z.DeclareDate,
            Chgreas=','.join(r),Chgrsdis=z.Comments,Content=z.Comments,ShortNamebc=z.ValueBefore,ShortNameac=z.ValueAfter,
            announcement=pd.to_datetime(z.DeclareDate,errors='coerce'),execution=pd.to_datetime(z.ChangeDate,errors='coerce'),source='local_item_supplement'))
        known.add((z.Symbol,typ,z.ChangeDate))
    x=pd.concat([x,pd.DataFrame(extra)],ignore_index=True);x['group']=x.Symbol.map(lambda c:mapping.get(c,c))
    rp=Path('/home/oliver/MainDiskData/first_st_followup_audit_20261025_v1/input/review160_event_decisions.json')
    reviews=json.loads(rp.read_text())['decisions'];audit[str(rp)]=sha(rp)
    lookup={}
    for z in reviews:
        for c in z['symbols']:lookup[(c,z['type'],z['date'])]=z
    classified=[]
    for z in x.itertuples():
        if z.Chgtype in ['AB','AD']:
            codes=set(re.findall(r'\d{2}',z.Chgreas));r=lookup.get((z.Symbol,z.Chgtype,z.Execudt))
            if codes&FIN:kind=1;reason='explicit_financial_audit_code'
            elif r and r['reason'].strip()==z.Chgrsdis.strip():
                kind=1 if r['decision']=='include' else 2;reason='raw_matched_prior_reason_review_'+r['decision']
            else:kind=0;reason='unresolved_first_ST_reason'
        elif z.Chgtype.endswith('X'):kind=2;reason='direct_delisting'
        else:continue
        date=z.announcement
        if pd.isna(date) or (pd.notna(z.execution) and date>z.execution):
            kind=0;reason='unresolved_announcement_clock';date=z.execution
        classified.append(dict(group=z.group,code=z.Symbol,date=date,event=kind,reason=reason,execution=z.execution))
    endpoints=pd.DataFrame(classified)
    # Every ST history hint, including recovery from an unobserved earlier ST,
    # prevents a later row from being described as a first-ST risk set member.
    histories=[]
    for z in x.itertuples():
        if any(c in z.Chgtype for c in ['B','D','C']) or name_state(z.ShortNamebc) in ['B','D']:
            dates=[d for d in [z.announcement,z.execution] if pd.notna(d)]
            if dates:histories.append((z.group,min(dates)))
    hist=pd.DataFrame(histories,columns=['group','date']).groupby('group').date.min().to_dict()
    x.to_parquet(run/'private/raw_event_alignment_PRIVATE.parquet',index=False)
    endpoints.to_parquet(run/'private/event_endpoints_PRIVATE.parquet',index=False)
    dump(run/'private/company_mapping_PRIVATE.json',mapping)
    audit.update(raw_rows=len(raw),supplement_rows=len(extra),scope_bound=bound,scope_logical_raw_match=True,
        accepted_scope_not_independent_universal_certification=True,identity_basis='local STK_ITEMCHANGE InstitutionID',
        unresolved_first_ST_rows=int(endpoints.reason.eq('unresolved_first_ST_reason').sum()))
    dump(run/'audit/event_sources.json',audit)
    return endpoints,hist,mapping,bound

def main(run,root):
    started=time.perf_counter();schema=json.loads(Path(__file__).with_name('feature_schema.json').read_text());names=schema['names']
    receipts=sorted((run/'results/raw_scan').glob('*/task-*/output/receipt.json'));assert len(receipts)==20
    frames=[];industry=None;rawaudit=[]
    for p in receipts:
        r=json.loads(p.read_text());assert sha(p.parent/'data_PRIVATE.parquet')==r['output_sha256'];rawaudit.append(r)
        d=pd.read_parquet(p.parent/'data_PRIVATE.parquet')
        if r['kind']=='industry':industry=d;continue
        d['priority']=1 if r['source'].endswith('.csv') else 0;frames.append(d)
    market=pd.concat(frames,ignore_index=True);del frames
    market=market.sort_values(['Stkcd','Trddt','priority'],kind='stable')
    dup=market.duplicated(['Stkcd','Trddt'],keep=False)
    # Conflict audit precedes deterministic newer-export precedence.
    conflicts=market.loc[dup].groupby(['Stkcd','Trddt'])[['Clsprc','Trdsta']].nunique().gt(1).any(axis=1)
    du=int(market.duplicated(['Stkcd','Trddt']).sum());market=market.drop_duplicates(['Stkcd','Trddt'],keep='last').drop(columns='priority')
    cal=np.sort(market.Trddt.to_numpy(dtype='datetime64[D]').__array__());cal=np.unique(cal)
    endpoints,hist,mapping,bound=source_events(root,run)
    assert market.Trddt.max()<=bound
    # Conservative follow-up bound: both accepted event registry and daily export.
    bound=min(bound,market.Trddt.max())
    market['group']=market.Stkcd.map(lambda c:mapping.get(c,c))
    daily_risk=market.loc[market.Trdsta.isin(RISK)].groupby('group').Trddt.min().to_dict()
    eg={k:g.sort_values('date') for k,g in endpoints.groupby('group')}
    f=pd.read_parquet(run/'prepared/financial_facts_PRIVATE.parquet')
    assert industry is not None
    industry.Accper=industry.Accper.dt.strftime('%Y-%m-%d')
    idup=industry.duplicated(['Stkcd','Accper'],keep=False);industry=industry[~idup].copy()
    industry['industry']=np.where(pd.to_datetime(industry.Accper)<pd.Timestamp('2023-05-01'),industry.Indcd,industry.Indcd1)
    f=f.merge(industry[['Stkcd','Accper','industry']],on=['Stkcd','Accper'],how='left',validate='one_to_one')
    fg={k:g.sort_values('financial_period_end') for k,g in f.groupby('Stkcd')}
    rows=[];values=[];clockrows=[];excluded=[]
    for code,g in market.groupby('Stkcd',sort=True):
        group=mapping.get(code,code);g=g.sort_values('Trddt');dates=g.Trddt.to_numpy(dtype='datetime64[ns]')
        ff=fg.get(code,f.iloc[:0]);ge=eg.get(group,endpoints.iloc[:0]);first_history=hist.get(group,pd.Timestamp.max)
        first_daily=daily_risk.get(group,pd.Timestamp.max)
        for year in range(2012,2026):
            origin=pd.Timestamp(year,1,1);i=np.searchsorted(dates,np.datetime64(origin),side='left')
            reason=None
            if not i:reason='not_observed_before_origin'
            elif (origin-g.Trddt.iloc[i-1]).days>31:reason='no_recent_origin_status'
            elif g.Trdsta.iloc[i-1] not in NORMAL:reason='not_normal_at_origin'
            elif min(first_history,first_daily)<=origin:reason='known_prior_ST_history'
            available=ff[ff.available.lt(origin)&ff.financial_period_end.lt(origin)]
            if reason is None and not len(available):reason='no_dated_industry_financial_report'
            if reason is None:
                last=available.iloc[-1];ind=str(last.industry)
                if ind in ['','nan','None'] or not re.fullmatch(r'[A-Z]\d*',ind):reason='unknown_industry'
                elif ind.startswith('J'):reason='financial_industry'
            if reason is not None:excluded.append((year,reason));continue
            first=ge[ge.date.gt(origin)].iloc[0] if len(ge[ge.date.gt(origin)]) else None
            event_date=first.date if first is not None else pd.NaT;event=int(first.event) if first is not None else 0
            end=bound;stop_reason='accepted_registry_and_market_snapshot'
            if first is not None and event==0:end=min(end,event_date);stop_reason=first.reason
            # Unmatched daily risk or earlier history hint is an unresolved
            # transition, not evidence that the company remained normal.
            hint=min(first_history,first_daily)
            if origin<hint<end and (pd.isna(event_date) or hint<event_date):
                end=hint;event=0;event_date=pd.NaT;stop_reason='unmatched_prior_risk_transition'
            v,dt=available_values(last,names[:48],origin)
            mg=market_features(g[MARKET],cal,origin);mv=np.array([mg[n] for n in names[48:72]],float)
            quality=np.array([min((origin-last.financial_period_end).days/365,2),min((origin-last.available).days/365,2),1-min(max(mg['market_coverage_3m'],0),1)])
            full=np.r_[v,mv,quality];full[~np.isfinite(full)]=np.nan
            assert last.available<origin and mg['market_latest_date']<origin
            row=dict(sample_id=f'{group}_{year}',code=code,group=group,year=year,origin=origin,observed_end=end,
                event_date=event_date,event_type=event,stop_reason=stop_reason,financial_available_at=last.available,
                financial_period=last.financial_period_end,prev_available=last.prev_available,market_available_at=mg['market_latest_date'],industry=ind,
                h12=(origin+pd.DateOffset(months=12)-origin).days,h24=(origin+pd.DateOffset(months=24)-origin).days)
            rows.append(row);values.append(full);clockrows.append(dt)
    d=pd.DataFrame(rows);X=np.asarray(values);clocks=np.asarray(clockrows)
    # More than one contemporaneous A-share code in a linked company requires
    # explicit identity adjudication. Exclude the whole conflicting landmark.
    duplicates=d.duplicated(['group','year'],keep=False);n_duplicate=int(duplicates.sum())
    keep=~duplicates.to_numpy();d=d[keep].reset_index(drop=True);X=X[keep];clocks=clocks[keep]
    assert d.sample_id.is_unique and X.shape[1]==75
    d.to_parquet(run/'private/landmarks_PRIVATE.parquet',index=False)
    np.savez_compressed(run/'private/all_features_PRIVATE.npz',X=X,financial_available=clocks,sample_ids=d.sample_id.to_numpy(str))
    counts=[]
    folds=run/'prepared/folds';folds.mkdir()
    for Y in [2023,2024,2025]:
        out=folds/str(Y);out.mkdir();trix=np.flatnonzero(d.year.lt(Y));evix=np.flatnonzero(d.year.eq(Y))
        for split,ix in [('train',trix),('eval',evix)]:
            m=d.iloc[ix].copy();clock=[stop_at(r.origin,r.observed_end,r.event_date,r.event_type,pd.Timestamp(Y,1,1) if split=='train' else None) for r in m.itertuples()]
            m['time']=[z[0] for z in clock];m['event']=[z[1] for z in clock];m['stop']=[z[2] for z in clock]
            positive=m.time.gt(0).to_numpy();m=m[positive].reset_index(drop=True);xx=X[ix][positive]
            if split=='train':
                assert m.stop.lt(pd.Timestamp(Y,1,1)).all();np.savez_compressed(out/'train_PRIVATE.npz',X=xx,time=m.time.to_numpy(float),event=m.event.to_numpy(int),groups=m.group.to_numpy(str),sample_ids=m.sample_id.to_numpy(str))
            else:
                np.savez_compressed(out/'eval_PRIVATE.npz',X=xx,sample_ids=m.sample_id.to_numpy(str))
            m.to_parquet(run/'evaluation_input'/f'{Y}_{split}_PRIVATE.parquet',index=False)
            counts.append(dict(cutoff_year=Y,split=split,records=len(m),companies=m.group.nunique(),target_records=int(m.event.eq(1).sum()),
                target_companies=m.loc[m.event.eq(1),'group'].nunique(),competing_records=int(m.event.eq(2).sum()),
                competing_companies=m.loc[m.event.eq(2),'group'].nunique(),last_stop=str(m.stop.max().date()),
                unknown12=int(((m.event==0)&(m.time<m.h12)).sum()),unknown24=int(((m.event==0)&(m.time<m.h24)).sum())))
    pd.DataFrame(counts).to_csv(run/'reports/COHORT_COUNTS.csv',index=False)
    pd.DataFrame(excluded,columns=['year','reason']).value_counts().rename('n').reset_index().to_csv(run/'reports/EXCLUSIONS.csv',index=False)
    dump(run/'audit/raw_scan_manifest.json',rawaudit)
    dump(run/'audit/preparation.json',dict(landmarks=len(d),companies=d.group.nunique(),features=75,market_rows=len(market),
        overlapping_market_rows_removed=du,overlap_price_status_conflicts=int(conflicts.sum()),industry_duplicate_rows_excluded=int(idup.sum()),
        multi_code_landmarks_excluded=n_duplicate,followup_bound=bound,feature_future_violations=0,original_vintage_certified=False,
        seconds=time.perf_counter()-started,cohort_counts=counts))
    print(json.dumps(counts,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);p.add_argument('--root',required=True,type=Path);a=p.parse_args();main(a.run,a.root)
