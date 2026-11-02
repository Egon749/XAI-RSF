"""User-approved available-paper-variable adaptation, never a54-field replica."""
from pathlib import Path
from dataclasses import replace
import numpy as np
from experiments.post_st_drop9_v1.common import (
    SEEDS,TIMES,REFERENCE_SHA,DROP_NAMES,configuration as reference_configuration,
    pairing,array_hash,arrays,sha,write,horizon_predictions,kept_columns,reduced)

NAME='post_st_paper_features_v1'
YEARS=[2023,2024,2025]
PAPER=Path('/home/oliver/.codex/attachments/01240ea3-4202-499e-8b7b-d5862f2578a4/Progressive gradient boosted trees for imbalanced financial distress prediction.pdf')
OLD=Path('/home/oliver/MainDiskData/xai_rsf_leaf_shallow_20261028_ep_9hazj')
LATER=Path('/home/oliver/MainDiskData/xai_rsf_2025_shallow_parent_20261029_pSED0uC5')
PAPER_NAMES=[
    'Return on Assets (ROA)','Net Profit Margin on Total Assets','Return on Equity (ROE)',
    'Return on Invested Capital (ROIC)','Return on Long-Term Capital','Financial Expense Ratio','Cost Expense Profit Margin',
    'Current Ratio','Quick Ratio','Cash Ratio','Interest Coverage Ratio','Operating Cash Flow/Current Liabilities',
    'Debt to Asset Ratio','Tangible Asset to Interest-Bearing Debt Ratio','EBITD/Total Debt Ratio','Tangible Net Asset Debt Ratio',
    'Capital Preservation and Value Added Rate','Capital Accumulation Rate','Fixed Asset Growth Rate','Total Asset Growth Rate',
    'Total Revenue Growth Rate','Management Expense Growth Rate','Sustainable Growth Rate','Accounts Receivable Turnover Ratio',
    'Inventory Turnover Ratio','Operating Cycle','Accounts Payable Turnover Ratio','Fixed Asset Turnover Ratio','Capital Intensity',
    'Total Asset Turnover Ratio','Shareholder Equity Turnover Ratio','Current Asset Ratio','Cash Asset Ratio','Receivables Asset Ratio',
    'Working Capital Ratio','Long-Term Asset Suitability Ratio','Operating Liabilities Ratio','Profit Share from Core Business',
    'Net Profit Cash Net Containment','Cash Content of Operating Revenue','Total Cash Recovery Rate','Operating Index',
    'Cash Suitability Ratio','Cash to Investment Satisfaction Ratio','Earnings per Share (EPS)','Net Asset per Share','Debt per Share',
    'Operating Cash Flow per Share','Net Cash Flow per Share','Price to Earnings Ratio (P/E)','Price to Sales Ratio (P/S)',
    'Price to Cash Flow Ratio (P/CF)','Market Value','Book to Market Ratio']
SELECTED={1:('roa','parent-attributable net profit, fallback total net profit / ending total assets; paper average/basis unspecified'),
    2:('net_profit_to_assets','exact duplicate of existing ROA; paper does not disambiguate X1/X2 formulas'),
    6:('financial_expense_to_revenue','financial expense / operating revenue, fallback total revenue'),
    8:('current_ratio','current assets / current liabilities'),9:('quick_ratio','(current assets-inventory)/current liabilities'),
    10:('cash_ratio','cash / current liabilities; paper cash-equivalent treatment unspecified'),
    11:('interest_coverage','(total profit+financial expense)/abs(interest expense, fallback financial expense); inherited EBIT proxy'),
    12:('cfo_to_current_liabilities','operating cash flow / current liabilities'),13:('asset_liability_ratio','total liabilities / total assets'),
    20:('total_assets_growth','same-quarter year-on-year (current-prior)/abs(prior); actual both-period availability verified'),
    21:('revenue_growth','operating revenue fallback total revenue; same-quarter year-on-year /abs(prior), paper total-revenue basis differs'),
    24:('ar_turnover','operating revenue / ending net receivables; no average denominator or annualization'),
    25:('inventory_turnover','operating cost / ending net inventory; no average denominator or annualization'),
    30:('total_asset_turnover','operating revenue / ending total assets; no average denominator or annualization'),
    33:('cash_to_assets','cash / total assets'),35:('working_capital_to_assets','(current assets-current liabilities)/total assets'),
    39:('cfo_to_net_income','operating cash flow / parent-attributable net profit, fallback total net profit'),
    40:('cfo_to_revenue','operating cash flow / operating revenue, fallback total revenue'),
    41:('cfo_to_assets','operationalized as operating cash flow / total assets; paper does not give exact formula'),
    53:('log_total_mktcap_last','invert existing natural log to total market value in CSMAR thousand-CNY units, at same pre-open market date')}
PAPER_IDS=list(SELECTED)


def configuration(seed):
    # Preserve the maximum25 policy; only20 available columns can be eligible.
    return replace(reference_configuration(seed,'shallow_parent'),tree_feature_count=min(25,len(PAPER_IDS)))


def columns(names):
    if len(names)!=83 or len(set(names))!=83:raise ValueError('Expected unchanged83-input schema')
    return [names.index(SELECTED[k][0]) for k in PAPER_IDS]


def transform(data,names):
    result=reduced(data,columns(names));j=PAPER_IDS.index(53)
    result['X'][:,j]=np.exp(result['X'][:,j])
    if np.isinf(result['X']).any():raise ValueError('Invalid market-value inverse transform')
    return result


def history(year):return LATER if year==2025 else OLD


def references(year,arm,repo):
    if arm=='reference83':
        if year==2025:
            from experiments.post_st_2025_shallow_parent_v1.analyze import outputs
            return {int(s):p for model,s,p in outputs(LATER) if model=='xai_q2'}
        from experiments.post_st_leaf_shallow_v1.analyze import artifacts
        return {int(s):p for y,a,s,p in artifacts(OLD) if y==year and a=='shallow_parent'}
    name='post_st_drop9_v1' if year==2024 else 'post_st_drop9_transport_v1'
    r=json_load(repo/'experiments'/name/'execution_receipt.json');run=Path(r['private_run']);job=r['slurm_job']
    offset=3 if year==2025 else 0
    return {s:run/'results/formal'/str(job)/f'task-{offset+i}/output/shrunk' for i,s in enumerate(SEEDS)}


def json_load(p):
    import json
    return json.loads(Path(p).read_text())
