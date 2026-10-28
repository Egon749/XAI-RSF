"""One read-only diagnostic entrypoint, independent of all model training code."""
from pathlib import Path
import sys,json,hashlib,re
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from rule_registry import build,board_of
from asof_policy_resolver import resolve
from event_rule_mapping import main as map_events
from evaluate_warning_utility import additional_utility
from error_attribution import overlap
BASE=Path('/home/oliver/MainDiskData/xai_rsf_interpretable_baselines_v1_20261027_hw9gdrol')
FIN=Path('/home/oliver/MainDiskData/early_warning_12m_server_isolated_20261025/prepared')
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str))
def main(run):
 run=Path(run);out=run/'results';inp=run/'input';rules=build(run,out);events=map_events(run,run/'private');eidx=events.set_index('event_id')
 sys.path.insert(0,str(inp/'legacy'))
 from common import endpoint_weights
 from endpoint_guard import source_consistent_endpoint_weights
 from evaluate import top_flags,warning_metrics,event_leads,lead_summary
 allrows=[]
 for role in ['train','eval']:
  d=pd.read_parquet(inp/f'final_{role}_metadata_PRIVATE.parquet');d['role']=role;allrows.append(d)
 allrows=pd.concat(allrows,ignore_index=True);allrows['board']=allrows.code.map(board_of)
 # Finance metadata only; no values or future event paths are used to select policy.
 f=pd.read_parquet(FIN/'financial.parquet',columns=['code','quarter','Accper','available','prev_available']);assert not f.duplicated(['code','quarter']).any()
 context=allrows[['sample_id','code','group','role','origin','horizon','quarter','board']].merge(f,on=['code','quarter'],how='left',validate='many_to_one')
 assert len(context)==len(allrows)
 context['report_period_end']=pd.to_datetime(context.Accper);context['report_type']=np.where(context.report_period_end.dt.month.eq(12),'annual',np.where(context.report_period_end.notna(),'interim','MISSING'))
 context['available_at']=context.available;context['revision_available_at']=pd.NaT
 context['metric_definition']='冻结财务值未获原始披露版本认证；政策量不生成';context['unit']='UNVERIFIED_RAW_FIELD_UNIT';context['value_use']='LATEST_AVAILABLE_PROXY_NOT_PIT_CERTIFIED'
 context['financial_clock_visible']=context.available.notna()&(context.available<context.origin)
 # Latest annual public-date proxy, based only on existing fact availability; revision lineage not certified.
 facts=pd.read_parquet(FIN/'financial_facts.parquet',columns=['Stkcd','Accper','available','ann','correction_0','correction_1','correction_2'])
 facts['Accper']=pd.to_datetime(facts.Accper);annual=facts[facts.Accper.dt.month.eq(12)&facts.available.notna()].copy();annual['latest_available_fiscal_year']=annual.Accper.dt.year
 annual['revision_at']=annual[['correction_0','correction_1','correction_2']].max(axis=1)
 revisions=facts.assign(revision_at=facts[['correction_0','correction_1','correction_2']].max(axis=1)).groupby(['Stkcd','Accper']).revision_at.max()
 context['revision_available_at']=[revisions.get((code,period),pd.NaT) for code,period in zip(context.code,context.report_period_end)]
 # Current fiscal-year ordering maintained, not selecting revised older period over a newer report.
 maps={}
 for code,g in annual.groupby('Stkcd'):
  g=g.sort_values('available');g['latest_available_fiscal_year']=g.latest_available_fiscal_year.cummax();maps[code]=g
 af=[]
 for code,g in context.groupby('code',sort=False):
  z=maps.get(code);left=g[['origin']].sort_values('origin').copy();left['row_index']=left.index
  if z is None:continue
  merged=pd.merge_asof(left,z[['available','latest_available_fiscal_year']].sort_values('available'),left_on='origin',right_on='available',allow_exact_matches=False,direction='backward');af.append(merged)
 context['latest_available_fiscal_year']=np.nan
 if af:
  af=pd.concat(af).set_index('row_index');context.loc[af.index,'latest_available_fiscal_year']=af.latest_available_fiscal_year
 keys={}
 for key,g in context.groupby(['board','origin','horizon']):
  result=resolve(rules,*key);keys[key]=result
 for col in ['known_rule_ids','current_candidate_ids','known_future_ids','known_fiscal_paths','RETROSPECTIVE_DIAGNOSTIC_ONLY_future_rules']:
  context[col]=[json.dumps(keys[(b,o,h)][col],default=str) for b,o,h in zip(context.board,context.origin,context.horizon)]
 context['applicability']='UNRESOLVED_TRANSITION_OR_COMPANY_FACTS';context['registry_coverage']='PARTIAL_VERSION_CHAIN'
 context.to_parquet(run/'private/origin_policy_context.parquet',index=False)
 # Explicit verified policy changes; governance-only edition changes not counted as ST trigger changes.
 changes=[]
 def change(cid,boards,pub,eff,note):
  for b in boards:changes.append(dict(change_id=cid,board=b,published=pd.Timestamp(pub),effective=pd.Timestamp(eff),meaning=note))
 change('MAIN_2012_NEW_ANNUAL_PATHS',['SSE_MAIN','SZSE_MAIN'],'2012-07-07','2012-07-07','新增净资产/收入/审计年度路径；首FY2012')
 change('MAIN_2020_REFORM',['SSE_MAIN','SZSE_MAIN'],'2020-12-31','2020-12-31','新财务组合及新旧ST衔接，首FY2020')
 change('CHINEXT_2020_REFORM',['SZSE_CHINEXT'],'2020-06-12','2020-06-12','2020年6月正式制度通知；完整细则适用仍未决')
 change('CHINEXT_ORDINARY_ST_START',['SZSE_CHINEXT'],'2020-06-12','2020-09-12','正式通知第8项，普通ST延至9月12日')
 change('CHINEXT_2020_DEC',['SZSE_CHINEXT'],'2020-12-31','2020-12-31','12月财务/内控/持续经营修订及衔接')
 change('2024_RULE_PUBLICATION',['SSE_MAIN','SZSE_MAIN','SZSE_CHINEXT','SSE_STAR'],'2024-04-30','2024-04-30','不是统一财务切换：新收入利润口径首FY2024，旧FY2023保留')
 change('STAR_OTHER_ST_6M',['SSE_STAR'],'2024-04-30','2024-10-30','普通ST部分条款6个月后；非所有条款')
 change('DIVIDEND_START',['SSE_MAIN','SZSE_MAIN','SZSE_CHINEXT','SSE_STAR'],'2024-04-30','2025-01-01','分红ST首3年FY2022–2024，例外逐板核对')
 trans=[];known=np.zeros(len(allrows),bool);unknown=known.copy()
 for ch in changes:
  mask=allrows.board.eq(ch['board'])&(allrows.origin<=ch['effective'])&(allrows.horizon>=ch['effective'])
  k=mask&(ch['published']<allrows.origin);u=mask&~k;known|=k.to_numpy();unknown|=u.to_numpy()
  for role in ['train','eval']:
   rmask=allrows.role.eq(role)
   trans.append(dict(**{k:str(v.date()) if isinstance(v,pd.Timestamp) else v for k,v in ch.items()},role=role,crossing_rows=int((mask&rmask).sum()),known_before_origin_rows=int((k&rmask).sum()),unknown_at_origin_rows=int((u&rmask).sum())))
 allrows['policy_crossing']=np.select([known&unknown,known,unknown],['KNOWN_AND_UNKNOWN','KNOWN_FUTURE','UNKNOWN_FUTURE'],default='NO_REGISTERED_CHANGE')
 pd.DataFrame(trans).to_csv(out/'policy_transition_windows.csv',index=False)
 write(run/'private/policy_asof_audit.json',dict(records=len(context),known_rules_public_strict=True,same_day_excluded=True,retrospective_future_columns_not_features=True,financial_clock_visible_rows=int(context.financial_clock_visible.sum()),selected_financial_clock_violations=int((context.available.notna()&~context.financial_clock_visible).sum()),annual_proxy_available_rows=int(context.latest_available_fiscal_year.notna().sum()),pit_value_certified_rows=0,transition_counts={role:dict(any_crossing=int(((known|unknown)&allrows.role.eq(role)).sum()),known=int((known&allrows.role.eq(role)).sum()),unknown=int((unknown&allrows.role.eq(role)).sum())) for role in ['train','eval']}))
 d=allrows[allrows.role.eq('eval')].reset_index(drop=True);preds={}
 for model in ['q2','unrestricted']:
  ps=[]
  for seed in [42,101,202]:
   z=np.load(inp/f'reference_{model}_{seed}_PRIVATE.npz');np.testing.assert_array_equal(z['sample_ids'],d.sample_id);ps.append(z['cif12'][:,0])
  preds[model]=np.mean(ps,axis=0)
 z=np.load(inp/'M2_PRIVATE.npz');np.testing.assert_array_equal(z['sample_ids'],d.sample_id);assert set(z['seeds'])=={42,101,202};preds['M2_allsector']=np.mean(z['probabilities'],axis=0)
 flags={(m,f):top_flags(d,p,f) for m,p in preds.items() for f in [.2,.25]};metrics=[];warnings=[];leads=[]
 for scope,idx in [('pooled',np.arange(len(d))),*[(str(y),v) for y,v in d.groupby('year').indices.items()]]:
  dd=d.iloc[idx].reset_index(drop=True);w,wa=source_consistent_endpoint_weights(dd,endpoint_weights);y=dd.event.eq(1).to_numpy(int)
  for m,p in preds.items():
   pp=p[idx]
   for weighting in ['record_equal','company_equal']:
    base=np.ones(len(dd)) if weighting=='record_equal' else 1/dd.groupby('group').group.transform('size').to_numpy();ww=w*base;k=ww>0
    metrics.append(dict(scope=scope,model=m,weighting=weighting,auc=roc_auc_score(y[k],pp[k],sample_weight=ww[k]),ap=average_precision_score(y[k],pp[k],sample_weight=ww[k]),ipcw_brier=np.sum(ww*(y-pp)**2)/base.sum(),prediction_mean=np.average(pp,weights=base),observed_ipcw_incidence=np.sum(ww*y)/base.sum()))
   for f in [.2,.25]:
    ff=flags[m,f][idx];ls=event_leads(dd,ff);warnings.append(dict(scope=scope,model=m,top_fraction=f,**warning_metrics(dd,ff),**lead_summary(ls),**additional_utility(dd,ff)))
    if scope=='pooled':ls['model']=m;ls['top_fraction']=f;leads.append(ls)
 mf=pd.DataFrame(metrics);wf=pd.DataFrame(warnings);old=pd.read_csv(BASE/'reports/metrics.csv');ow=pd.read_csv(BASE/'reports/warning_metrics.csv');maxdiff=0.
 for row in mf.to_dict('records'):
  prev=old[(old.model==row['model'])&(old.scope.astype(str)==row['scope'])&(old.weighting==row['weighting'])&(old.aggregation=='seed_ensemble')];assert len(prev)==1
  for k in ['auc','ap','ipcw_brier','prediction_mean']:diff=abs(row[k]-prev.iloc[0][k]);assert diff<1e-12,(row,k,diff);maxdiff=max(maxdiff,diff)
 for row in wf.to_dict('records'):
  prev=ow[(ow.model==row['model'])&(ow.scope.astype(str)==row['scope'])&(ow.top_fraction==row['top_fraction'])&(ow.aggregation=='seed_ensemble')];assert len(prev)==1
  for k in ['alert_records','detected_events','alert_company_groups','known_target_free_alerts','competing_alerts','unknown_outcome_alerts','announcement_detected_3m','announcement_detected_6m']:assert row[k]==prev.iloc[0][k],(k,row)
 mf.to_csv(out/'reproduced_metrics.csv',index=False);wf.to_csv(out/'reproduced_warning_metrics.csv',index=False);lead=pd.concat(leads,ignore_index=True);lead.to_parquet(run/'private/event_leads_PRIVATE.parquet',index=False)
 np.savez_compressed(run/'private/reproduced_flags_PRIVATE.npz',sample_ids=d.sample_id.to_numpy(str),**{m+'_top'+str(int(f*100)):v for (m,f),v in flags.items()})
 write(run/'audit/reproduction.json',dict(metrics_rows=len(mf),warning_rows=len(wf),max_abs_metric_difference=maxdiff,warning_counts_exact=True,list_rule='same original top_flags: quarter, descending risk, ascending sample_id, ceil',warning_list_sha256=hashlib.sha256((run/'private/reproduced_flags_PRIVATE.npz').read_bytes()).hexdigest(),aggregation='mean probabilities 42/101/202',new_training=0))
 overlap(d,flags).to_csv(out/'model_error_overlap.csv',index=False)
 # All origins retained. Origin policy partition never depends on future event causes.
 slices=[]
 for dim in ['year','quarter','board','policy_crossing']:
  for value,idx in d.groupby(dim).indices.items():
   dd=d.iloc[idx].reset_index(drop=True)
   for m in preds:
    for f in [.2,.25]:
     ff=flags[m,f][idx];ls=event_leads(dd,ff);slices.append(dict(dimension=dim,value=str(value),model=m,top_fraction=f,records=len(dd),**warning_metrics(dd,ff),**(lead_summary(ls) if len(ls) else {}),**additional_utility(dd,ff)))
 pd.DataFrame(slices).to_csv(out/'warning_utility_by_policy.csv',index=False)
 # Explicitly separate verified trigger families from keyword candidates (not legal conclusions).
 comps=[];errs=[]
 for e in events.to_dict('records'):
  verified=e['verification_status']=='VERIFIED';fams=json.loads(e['verified_families'] if verified else e['candidate_trigger_families']) or ['UNRESOLVED_CAUSE']
  for fam in fams:comps.append(dict(role=e['role'],event_type=e['frozen_event_type'],board=e['board'],family=fam,evidence='VERIFIED_IMPLEMENTATION_REASON' if verified else 'UNRESOLVED_TEXT_CANDIDATE',rule_version=e['verified_rule_version'] or 'UNRESOLVED',event_id=e['event_id'],quarter_rows=e['window_rows']))
  if e['role']=='eval' and e['frozen_event_type']==1:
   for m in preds:
    for f in [.2,.25]:
     rr=lead[(lead.event_id==e['event_id'])&(lead.model==m)&(lead.top_fraction==f)].iloc[0];det=pd.notna(rr.first_warning)
     for fam in fams:errs.append(dict(family=fam,evidence='VERIFIED_IMPLEMENTATION_REASON' if verified else 'UNRESOLVED_TEXT_CANDIDATE',board=e['board'],model=m,top_fraction=f,event_id=e['event_id'],missed=int(not det),detected=int(det),announcement_opportunity_3m=int(rr.announcement_opportunity_3m),announcement_detected_3m=int(rr.announcement_detected_3m),announcement_opportunity_6m=int(rr.announcement_opportunity_6m),announcement_detected_6m=int(rr.announcement_detected_6m)))
 co=pd.DataFrame(comps);co.groupby(['role','event_type','board','family','evidence','rule_version'],dropna=False).agg(independent_events=('event_id','nunique'),quarter_rows=('quarter_rows','sum')).reset_index().to_csv(out/'policy_event_composition.csv',index=False)
 er=pd.DataFrame(errs);er.groupby(['family','evidence','board','model','top_fraction'],dropna=False).agg(independent_events=('event_id','nunique'),missed=('missed','sum'),detected=('detected','sum'),announcement_opportunity_3m=('announcement_opportunity_3m','sum'),announcement_detected_3m=('announcement_detected_3m','sum'),announcement_opportunity_6m=('announcement_opportunity_6m','sum'),announcement_detected_6m=('announcement_detected_6m','sum')).reset_index().to_csv(out/'policy_error_slices.csv',index=False)
 events.groupby(['role','frozen_event_type','board','verification_status'],dropna=False).agg(independent_events=('event_id','nunique'),quarter_rows=('window_rows','sum'),multiple_triggers=('multiple_triggers','sum'),transition_applies=('transition_applies','sum')).reset_index().to_csv(out/'event_rule_summary.csv',index=False)
 # Unapplied suggestions; empty is not proof original labels correct.
 candidates=events[events.proposed_classification.notna()].copy()
 candidates[['event_id','role','frozen_event_type','proposed_classification','window_rows','review_reason','official_documents']].to_parquet(run/'private/label_revision_candidates_PRIVATE.parquet',index=False)
 candout=[]
 for confirmed,g in candidates.groupby('conflict_with_frozen_label'):
  candout.append(dict(candidate_type='target_to_competing_review',original_label=1,proposed_classification=2,independent_events=g.event_id.nunique(),affected_rows=int(g.window_rows.sum()),affected_role='train',evidence_status='OFFICIAL_IMPLEMENTATION_REASON_CONFLICT' if confirmed else 'SPONSOR_OPINION_ONLY_UNRESOLVED',action='NOT_APPLIED'))
 pd.DataFrame(candout).to_csv(out/'label_revision_candidates.csv',index=False)
 # Exact official-trigger review provides descriptive change consistency, not causal proof.
 cases=[]
 reviewed=events[events.verification_status.eq('VERIFIED')&events.role.eq('eval')]
 for e in reviewed.to_dict('records'):
  if 'REVENUE_PROFIT' not in json.loads(e['verified_families']) or e['verified_fiscal_year']!=2024 or e['board'] not in ['SSE_MAIN','SZSE_MAIN']:continue
  packet=next(x for x in json.loads((run/'private/review_packets_PRIVATE.json').read_text()) if x['event_id']==e['event_id']);text=packet['text']
  # Human-reviewed examples with disclosed deducted revenue in the newly covered [100m,300m) band are attached separately, not inferred from generic quoted rules.
  cases.append(dict(event_id=e['event_id'],candidate='2024_NEW_REVENUE_BOUNDARY',status='ACTUAL_NEW_TRIGGER_CONFIRMED_BUT_OLD_TRIGGER_COUNTERFACTUAL_NOT_ESTABLISHED',source=e['official_documents']))
 pd.DataFrame(cases).to_parquet(run/'private/policy_consistency_cases_PRIVATE.parquet',index=False)
 write(run/'audit/diagnostic_summary.json',dict(total_events=len(events),verified_implementation_reasons=int(events.verification_status.eq('VERIFIED').sum()),unresolved_events=int(events.verification_status.ne('VERIFIED').sum()),multiple_verified=int(events.multiple_triggers.sum()),transition_verified=int(events.transition_applies.sum()),conflicting_labels_confirmed=int(events.conflict_with_frozen_label.sum()),verified_eval=int(reviewed.shape[0]),new_revenue_trigger_eval_cases=len(cases),new_fits=0))
 print((run/'audit/diagnostic_summary.json').read_text());print(wf[wf.scope=='pooled'][['model','top_fraction','alert_company_groups','detected_events','company_confirmed_hit_fraction']].to_string(index=False))
if __name__=='__main__':main(sys.argv[1])
