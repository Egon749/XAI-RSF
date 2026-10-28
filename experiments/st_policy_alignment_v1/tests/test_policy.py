"""Pure policy/evaluation tests. NO fit, training, calibration or resampling."""
import sys,json,os
from pathlib import Path
import pandas as pd
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from asof_policy_resolver import resolve,fiscal_applicability,visible_revision
from policy_features import evaluate,yuan,dependencies,feature_status
from evaluate_warning_utility import additional_utility
from verify_policy_outputs import verify
RUN=Path(os.environ['POLICY_RUN'])
@pytest.fixture
def rules():return pd.read_csv(RUN/'results/st_rule_versions.csv')

def row(rules,board,version,family):return rules[(rules.board==board)&(rules.rule_version.astype(str)==version)&(rules.trigger_family==family)].iloc[0].to_dict()
def expr(rule):return json.loads(rule['logical_expression'])
# Official-rule boundary fixtures: metadata and hash point to original rule/notice.
def test_source_boundaries_main_vs_chinext(rules):
 main=row(rules,'SZSE_MAIN','2024','REVENUE_PROFIT');growth=row(rules,'SZSE_CHINEXT','2024','REVENUE_PROFIT')
 assert len(main['source_hash'])==64 and len(growth['source_hash'])==64
 values=dict(net_profit=-1,net_profit_ex_nonrecurring=2,profit_total=3,deducted_revenue=2e8)
 assert evaluate(expr(main),values) is True
 assert evaluate(expr(growth),values) is False

def test_no_new_rules_before_or_on_publication(rules):
 for origin in ['2024-03-31','2024-04-30']:
  z=resolve(rules,'SSE_MAIN',origin,'2025-03-31');assert not any('_2024_' in x for x in z['known_rule_ids'])
  assert any('_2024_' in x for x in z['RETROSPECTIVE_DIAGNOSTIC_ONLY_future_rules'])
def test_known_future_dividend(rules):
 z=resolve(rules,'SSE_MAIN','2024-06-30','2025-06-30');assert 'SSE_MAIN_2024_DIVIDEND' in z['known_future_ids']
def test_fiscal_transition(rules):
 r=row(rules,'SSE_MAIN','2024','REVENUE_PROFIT');assert fiscal_applicability(r,2023,'annual')=='NOT_APPLICABLE_FISCAL_YEAR';assert fiscal_applicability(r,2024,'annual')=='POTENTIALLY_APPLICABLE'
 z=resolve(rules,'SSE_MAIN','2024-06-30','2025-06-30');assert any(x['rule_id']=='SSE_MAIN_2023FEB_REVENUE_PROFIT' for x in z['known_fiscal_paths'])
def test_quarter_not_annual(rules):
 assert fiscal_applicability(row(rules,'SSE_MAIN','2024','REVENUE_PROFIT'),2024,'interim')=='NOT_APPLICABLE_REPORT_TYPE'
def test_strict_boundary_and_profit_definition(rules):
 r=row(rules,'SZSE_MAIN','2024','REVENUE_PROFIT');x=dict(net_profit=10,net_profit_ex_nonrecurring=10,profit_total=-1,deducted_revenue=3e8)
 assert evaluate(expr(r),x) is False
 x['deducted_revenue']=3e8-1;assert evaluate(expr(r),x) is True
 x['profit_total']=0;assert evaluate(expr(r),x) is False
 assert dependencies(expr(r))=={'net_profit','net_profit_ex_nonrecurring','profit_total','deducted_revenue'}
def test_audit_paths_do_not_alias(rules):
 r=row(rules,'SSE_MAIN','2024','FINANCIAL_AUDIT');assert evaluate(expr(r),{'internal_control_audit':'adverse','financial_audit_opinion':'unqualified'}) is False
 assert evaluate(expr(r),{'internal_control_audit':'adverse'}) is None
# Synthetic program fixtures, not assertions of unverified legal semantics.
@pytest.mark.parametrize('unit,x',[('yuan',1e8),('万元',1e4),('亿元',1)])
def test_units(unit,x):assert yuan(x,unit)==1e8

def test_unknown_units():
 with pytest.raises(ValueError):yuan(10,'USD')
def test_correction_clock():
 d=pd.DataFrame({'available_at':['2023-04-01','2023-08-01'],'value':[1,-1]});assert visible_revision(d,'2023-08-01').value.tolist()==[1];assert visible_revision(d,'2023-08-02').value.tolist()==[1,-1]
def test_three_distinct_states():
 r=dict(first_applicable_fiscal_year=2024,expression={'field':'x','op':'<','value':0})
 assert feature_status(r,{},2024,'annual')[0]=='MISSING'
 assert feature_status(r,{},2023,'annual')[0]=='NOT_APPLICABLE_FISCAL_YEAR'
 assert feature_status(r,{},2024,'annual',False)[0]=='UNDECIDED'
def test_unknown_is_not_negative():
 d=pd.DataFrame(dict(event=[0,0,1,2],label_eligible=[False,True,True,True],group=['a','b','c','d']));m=additional_utility(d,np.ones(4,bool));assert m['unknown_company_groups']==1 and m['followup_supported_non_target_records']==1 and m['confirmed_hit_company_groups']==1

def test_event_unique_and_multiple_evidence():
 e=pd.read_parquet(RUN/'private/event_rule_evidence.parquet');assert e.event_id.nunique()==323
 assert e.multiple_triggers.any();assert 'FINANCIAL_AUDIT' in ' '.join(e.verified_families.dropna()) and 'INTERNAL_CONTROL' in ' '.join(e.verified_families.dropna())
def test_calendar_leads_and_separate_dates():
 sys.path.insert(0,str(RUN/'input/legacy'));from evaluate import event_leads
 d=pd.DataFrame(dict(event=[1,1],event_id=['x','x'],group=['g','g'],origin=pd.to_datetime(['2024-01-31','2024-02-01']),announcement_date=pd.to_datetime(['2024-04-30']*2),event_date=pd.to_datetime(['2024-05-06']*2)))
 r=event_leads(d,np.array([True,True])).iloc[0];assert not r.announcement_detected_3m and r.implementation_detected_3m;assert len(event_leads(d,np.ones(2,bool)))==1

def test_original_hashes_and_reproduction():assert verify(RUN)['original_hashes_unchanged']

def test_chinext_2020_delayed_ordinary_st(rules):
 z=resolve(rules,'SZSE_CHINEXT','2020-06-30','2021-06-30');assert 'SZSE_CHINEXT_2020JUN_OTHER_ST_PATHS' in z['known_future_ids']
 r=row(rules,'SZSE_CHINEXT','2020JUN','REVENUE_PROFIT');assert fiscal_applicability(r,2019,'annual')=='NOT_APPLICABLE_FISCAL_YEAR'
def test_unresolved_board_preserved():
 c=pd.read_parquet(RUN/'private/origin_policy_context.parquet');u=c[c.board.eq('UNRESOLVED_BOARD')];assert len(u)==61;assert u.known_rule_ids.eq('[]').all()
def test_all_rules_retain_provenance(rules):
 assert rules.source_hash.str.fullmatch('[a-f0-9]{64}').all();assert rules.rule_id.is_unique
 for name in rules.source_id.unique():assert (RUN/'sources'/(name+'.txt')).exists()
def test_conflicts_not_applied():
 e=pd.read_parquet(RUN/'private/event_rule_evidence.parquet');g=e[e.conflict_with_frozen_label.eq(True)];assert len(g)==2 and g.frozen_event_type.eq(1).all() and g.proposed_classification.eq(2).all()
 orig=pd.read_parquet(RUN/'input/final_train_metadata_PRIVATE.parquet');assert orig[orig.event_id.isin(g.event_id)].event.eq(1).all()
def test_revision_metadata_restored():
 c=pd.read_parquet(RUN/'private/origin_policy_context.parquet');assert c.revision_available_at.notna().any();assert (c.loc[c.revision_available_at.notna(),'revision_available_at']<c.loc[c.revision_available_at.notna(),'origin']).all()

def test_draft_excluded(rules):
 draft=rules.iloc[[0]].copy();draft['rule_id']='SYNTHETIC_DRAFT';draft['rule_status']='DRAFT';draft['official_published_at']='2010-01-01'
 r=pd.concat([rules,draft],ignore_index=True);z=resolve(r,str(draft.board.iloc[0]),'2012-01-01','2013-01-01');assert 'SYNTHETIC_DRAFT' not in z['known_rule_ids']
def test_original_top_budget_ties_include_unknown():
 sys.path.insert(0,str(RUN/'input/legacy'));from evaluate import top_flags
 d=pd.DataFrame({'quarter':['2023Q1']*5,'sample_id':['e','d','c','b','a'],'event':[0,0,0,0,0],'label_eligible':[True,True,True,False,False]})
 f=top_flags(d,np.ones(5),.25);assert f.tolist()==[False,False,False,True,True]
