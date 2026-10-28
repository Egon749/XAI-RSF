"""Strict information-clock resolver. A retrospective event table is never an input."""
import json
import pandas as pd

def resolve(rules,board,origin,horizon):
 origin=pd.Timestamp(origin);horizon=pd.Timestamp(horizon)
 relevant=rules[rules.board.eq(board)&rules.rule_status.eq('FORMAL')].copy()
 pub=pd.to_datetime(relevant.official_published_at);eff=pd.to_datetime(relevant.effective_from)
 known=relevant[pub<origin].copy()
 active=known[pd.to_datetime(known.effective_from)<=origin]
 # All legally known potential fiscal-year paths; do not select using a future event.
 latest=active.official_published_at.max() if len(active) else None
 current=active[active.official_published_at.eq(latest)] if latest else active
 scheduled=known[(pd.to_datetime(known.effective_from)>origin)&(pd.to_datetime(known.effective_from)<=horizon)]
 future=relevant[(pub>=origin)&(pub<=horizon)]
 return dict(known_rule_ids=known.rule_id.tolist(),current_candidate_ids=current.rule_id.tolist(),known_future_ids=scheduled.rule_id.tolist(),known_fiscal_paths=known[['rule_id','first_applicable_fiscal_year']].to_dict('records'),applicability='UNRESOLVED_TRANSITION_OR_COMPANY_FACTS',version_scope='PARTIAL_VERIFIED_REGISTRY_NOT_COMPLETE_LAW',RETROSPECTIVE_DIAGNOSTIC_ONLY_future_rules=future.rule_id.tolist())

def fiscal_applicability(rule,fiscal_year,report_type,transition_resolved=True):
 if report_type!='annual':return 'NOT_APPLICABLE_REPORT_TYPE'
 first=rule.get('first_applicable_fiscal_year')
 if first is not None and pd.notna(first) and fiscal_year<int(first):return 'NOT_APPLICABLE_FISCAL_YEAR'
 if not transition_resolved:return 'UNDECIDED'
 return 'POTENTIALLY_APPLICABLE'

def visible_revision(facts,origin):
 if not len(facts):return facts
 return facts[pd.to_datetime(facts.available_at)<pd.Timestamp(origin)].sort_values('available_at')
