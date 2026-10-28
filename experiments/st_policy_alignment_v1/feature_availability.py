"""Read existing facts only. Report feasibility and diagnostic proxies, no new score."""
from pathlib import Path
import json
import pandas as pd
import numpy as np

def main(run):
 run=Path(run);out=run/'results';ctx=pd.read_parquet(run/'private/origin_policy_context.parquet');d=pd.read_parquet(run/'input/final_eval_metadata_PRIVATE.parquet');f=pd.read_parquet('/home/oliver/MainDiskData/early_warning_12m_server_isolated_20261025/prepared/financial.parquet')
 z=d[['code','quarter','origin','event','label_eligible','group']].merge(f[['code','quarter','available','Accper','loss_flag','negative_equity_flag','营业收入','净利润','所有者权益合计']],on=['code','quarter'],how='left',validate='many_to_one');assert len(z)==len(d)
 rows=[]
 specs=[
 ('deducted_annual_revenue_distance','(deducted_revenue - board_threshold)/board_threshold','deducted_revenue;rule_board;applicable_fiscal_year','2024 main 9.3.1/9.3.2; ChiNext10.3.1; STAR12.4.2','CNY yuan','MISSING_DEDUCTED_REVENUE_AND_PIT_LINEAGE',0),
 ('profit_definition_continuity','annual min(net_profit,net_profit_ex_nonrecurring[,profit_total]); consecutive annual sequence','all raw profits;year;audited scope','2020/2024 revenue-profit; ordinaryST three-loss AND going-concern','CNY yuan','MISSING_EX_NONRECURRING_AND_AUDIT_LINEAGE',0),
 ('net_asset_distance','annual_net_assets - 0','audited annual net assets;report_scope;release','financial warning (2)','CNY yuan','PROXY_ONLY_RAW_UNIT_AND_PIT_UNCERTIFIED',int(z['所有者权益合计'].notna().sum())),
 ('financial_audit_opinion','categorical financial-statement audit opinion','opinion;actual_public_date;version','financial warning (3)','category','MISSING_STRUCTURED_CLOCKED_OPINION',0),
 ('internal_control_audit','categorical internal-control opinion;not same as financial audit','opinion;actual_public_date;version','ordinary ST separate clause','category','MISSING_STRUCTURED_CLOCKED_OPINION',0),
 ('going_concern','auditor going-concern uncertainty separately encoded','auditor text;actual_public_date;year','three-year loss AND going-concern','category','MISSING_STRUCTURED_CLOCKED_OPINION',0),
 ('board_known_rule','board + formally published rule paths at origin','historical board;source publication;exceptions','versioned registry','category','PARTIAL_RULE_CHAIN;61_ORIGINS_BOARD_UNRESOLVED',int(ctx.board.ne('UNRESOLVED_BOARD').sum())),
 ('report_calendar_staleness','origin - report_available_at;quarter;annual/interim','origin;report_period_end;available_at','information clock;not an ST trigger','days','OBSERVED_DATE_PROXY;PIT_VALUES_UNCERTIFIED',int(ctx.financial_clock_visible.sum())),
 ('applicability_missing_undecided','separate three categorical statuses','rule applicability;raw missing;transition facts','all rules','category','IMPLEMENTED_NO_FALSE_DEFAULTS',len(ctx)),
 ]
 for name,formula,dep,cl,unit,status,n in specs:rows.append(dict(feature=name,formula=formula,raw_dependencies=dep,official_clause=cl,unit=unit,public_clock='all components actual published_at < origin; max for derived fields',future_information='forbidden;event reasons never inputs',proxy='LATEST_AVAILABLE_PROXY' if 'PROXY' in status else 'NO_VALUE_GENERATED',coverage_count=n,coverage_denominator=len(d) if name=='net_asset_distance' else len(ctx),coverage_meaning='field/date metadata presence; not certified deployable values',historical_constructibility=status,pit_certified_policy_value_rows=0))
 pd.DataFrame(rows).to_csv(out/'policy_feature_availability.csv',index=False)
 flags=np.load(run/'private/reproduced_flags_PRIVATE.npz');diag=[]
 for m in ['q2','unrestricted','M2_allsector']:
  for top in [20,25]:
   alert=flags[m+'_top'+str(top)];known=z.event.eq(0)&z.label_eligible;g=z[alert&known]
   for field in ['loss_flag','negative_equity_flag']:
    vals=g[field];diag.append(dict(model=m,top_fraction=top/100,diagnostic='LATEST_AVAILABLE_PROXY_NOT_PIT_CERTIFIED',outcome='followup_supported_no_target',field=field,alert_records=len(g),known_value_rows=int(vals.notna().sum()),positive_flag_rows=int(vals.eq(1).sum()),zero_flag_rows=int(vals.eq(0).sum()),missing_rows=int(vals.isna().sum()),annual_report_rows=int(pd.to_datetime(g.Accper).dt.month.eq(12).sum()),policy_noncompliance_inferred=False))
 pd.DataFrame(diag).to_csv(out/'nonhit_financial_proxy_summary.csv',index=False)
if __name__=='__main__':
 import sys;main(sys.argv[1])
