"""Only mapping/time/evaluation tests. No predictor construction or fitting."""
import ast,copy,json,os,sys
from pathlib import Path
import pandas as pd
import pytest
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
from adjudicate_labels import adjudicate,implementation_document_eligible
from audit_reason_mapping import same_code_namespace
from verify_source_clocks import before_origin,eligible,asof_versions
from validate_label_patch import validate
from join_prediction_origins import information_state

def review():return dict(identity_matched=True,official_implementation_verified=True,historical_clock_verified=True,published_at='2019-04-09',multiple_trigger_checked=True,actual_trigger_families=['OCCUPATION_GUARANTEE'],frozen_policy_supports_competing=True)
@pytest.mark.parametrize('field',['source_table','field','version'])
def test_namespaces_never_mix(field):
 a=dict(source_table='A',field='code',version='v1');b=dict(a);b[field]='other';assert not same_code_namespace(a,b)
def test_missing_namespace_not_equivalence():assert not same_code_namespace({}, {})
def test_same_namespace():assert same_code_namespace(dict(source_table='A',field='c',version='v1'),dict(source_table='A',field='c',version='v1'))
def test_multitrigger_priority():
 r=review();r['actual_trigger_families']+=['FINANCIAL_AUDIT'];assert adjudicate(r,'2022-01-01')=='RETAIN_TARGET_MULTIPLE_TRIGGERS'
def test_no_unknown_policy():
 r=review();r['actual_trigger_families']=['UNREVIEWED'];assert adjudicate(r,'2022-01-01')=='UNRESOLVED_CLASSIFICATION_POLICY'
def test_missing_implementation():
 r=review();r['official_implementation_verified']=False;assert adjudicate(r,'2022-01-01')=='UNRESOLVED_IMPLEMENTATION_EVIDENCE'
def test_target_to_competing_decision():assert adjudicate(review(),'2022-01-01')=='CONFIRMED_TARGET_TO_COMPETING'
def test_stock_bond_formal_announcement_regression():assert implementation_document_eligible('关于公司股票交易实施其他风险警示暨股票和债券交易复牌的公告')
@pytest.mark.parametrize('title',['债券受托管理临时报告','保荐机构关于公司股票实施风险警示的核查意见','公司股票可能被实施退市风险警示的提示公告'])
def test_nonimplementation_rejected(title):assert not implementation_document_eligible(title)
def fixture():
 d=pd.DataFrame([dict(sample_id=f's{i}',event_id='E',event=1,event_date='2020-04-01',announcement_date='2020-03-31',outcome_available='2020-04-01',stop_date='2020-04-01',duration=12+i,origin='2019-06-30',horizon='2020-06-30',group='G') for i in range(4)])
 p=d.drop(columns='event').assign(old_event=1,new_event=2);return d,p
def test_all_event_windows_present():
 d,p=fixture();assert validate(p,d,'same','same')
def test_missing_window_rejected():
 d,p=fixture()
 with pytest.raises(ValueError,match='incomplete'):validate(p.iloc[:3],d,'same','same')
def test_duplicate_window_rejected():
 d,p=fixture()
 with pytest.raises(ValueError,match='duplicate'):validate(pd.concat([p,p.iloc[:1]]),d,'same','same')
def test_hash_mismatch_rejected():
 d,p=fixture()
 with pytest.raises(ValueError,match='hash'):validate(p,d,'changed','original')
def test_never_competing_to_normal():
 d,p=fixture();p['new_event']=0
 with pytest.raises(ValueError,match='never normal'):validate(p,d,'same','same')
def test_dates_and_followup_unchanged():
 d,p=fixture();p.loc[0,'stop_date']='2021-01-01'
 with pytest.raises(ValueError,match='contract'):validate(p,d,'same','same')
def test_signing_not_public():
 signed='2023-03-25';published='2023-04-10';origin='2023-03-31';assert before_origin(signed,origin) and not before_origin(published,origin)
def test_same_day_strict():assert not before_origin('2023-03-31','2023-03-31')
def test_unknown_public_date():assert not before_origin(None,'2023-03-31')
def test_future_event_reason_not_available():assert not before_origin('2024-04-30','2023-12-31')
def test_amended_not_backfilled():
 d=pd.DataFrame([dict(audit_report_fiscal_year=2022,audit_report_type='financial',available_at='2023-04-30',opinion='original'),dict(audit_report_fiscal_year=2022,audit_report_type='financial',available_at='2023-08-02',opinion='amended')]);assert asof_versions(d,'2023-06-30').iloc[0].opinion=='original';assert asof_versions(d,'2023-09-30').iloc[0].opinion=='amended'
@pytest.mark.parametrize('origin,event,months,expected',[('2023-01-31','2023-04-30',3,False),('2023-01-30','2023-04-30',3,True),('2023-02-01','2023-04-30',3,False),('2023-08-31','2024-02-29',6,False),('2023-08-29','2024-02-29',6,True),('2023-09-01','2024-02-29',6,False)])
def test_calendar_month_boundaries(origin,event,months,expected):assert eligible(origin,event,pd.Timestamp(origin)+pd.DateOffset(months=12),event,months)==expected
def test_outside_horizon_not_eligible():assert not eligible('2023-03-31','2024-04-01','2024-03-31','2024-03-31',6)
def test_opportunity_vs_miss():assert information_state(False,False)=='NO_ELIGIBLE_ORIGIN';assert information_state(True,False)=='NO_PREORIGIN_SIGNAL_FOUND_IN_SEARCHED_SOURCES'
def test_no_fit_calls_in_analysis_entrypoints():
 for p in HERE.glob('*.py'):
  tree=ast.parse(p.read_text())
  for node in ast.walk(tree):
   if isinstance(node,ast.Call):assert not(isinstance(node.func,ast.Attribute) and node.func.attr in ['fit','fit_transform','partial_fit']),str(p)

def runpath():
 value=os.environ.get('LABEL_AUDIT_RUN');assert value,'Set LABEL_AUDIT_RUN; integration tests are required, never skipped';return Path(value)
def test_private_patch_counts_and_no_application():
 r=runpath();x=json.loads((r/'audit/label_patch_validation.json').read_text());assert x['events']==3 and x['quarter_records']==12 and not x['patch_applied']
def test_frozen_lists_not_reranked():
 x=json.loads((runpath()/'audit/capture_reproduction.json').read_text());assert x['full_riskset_records']==23799 and x['all_six_full_riskset_lists_identical'] and not x['subset_reranking']
def test_missing_not_applicable_normal_separate():
 r=runpath();c=pd.read_parquet(r/'private/audit_coverage_PRIVATE.parquet');t=pd.read_parquet(r/'private/audit_event_timeline_PRIVATE.parquet');assert 'NO_VERIFIED_RECORD_IN_BOUNDED_REVIEW' in set(c.status);assert 'NOT_APPLICABLE_EXPLICIT' in set(c.status);assert 'STANDARD_UNQUALIFIED' in set(t.prior_public_signal_type)
def test_actual_origins_and_fixed_captures():
 r=runpath();x=json.loads((r/'audit/origin_join.json').read_text());assert x['frozen_event_origin_rows']==54 and x['all_frozen_capture_counts_reproduced'];d=pd.read_csv(r/'results/audit_timing_summary.csv');assert d[d.lead_months.eq(6)].eligible_events.eq(14).all()
def test_no_backfilled_key_opinion():
 r=runpath();d=pd.read_csv(r/'results/audit_timing_summary.csv');assert d.current_key_opinion_early_events.eq(0).all()
def test_originals_unchanged():assert json.loads((runpath()/'audit/immutability_verification.json').read_text())['all_original_sources_unchanged']

def test_legacy_month_end_direction_regression():
 from review_common import BASE
 sys.path.insert(0,str(BASE/'input/legacy'));from evaluate import event_leads
 import numpy as np
 for origin,target,months in [('2024-01-31','2024-04-30',3),('2023-08-31','2024-02-29',6),('2024-01-30','2024-04-30',3)]:
  d=pd.DataFrame(dict(event=[1],event_id=['example'],group=['g'],origin=pd.to_datetime([origin]),announcement_date=pd.to_datetime([target]),event_date=pd.to_datetime([target])))
  expected=event_leads(d,np.array([True])).iloc[0][f'announcement_opportunity_{months}m']
  assert eligible(origin,target,pd.Timestamp(origin)+pd.DateOffset(months=12),target,months)==expected
