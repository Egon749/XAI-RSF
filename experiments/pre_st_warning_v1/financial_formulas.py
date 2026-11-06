import pandas as pd
import numpy as np
def to_numeric_series(s): return pd.to_numeric(s,errors="coerce")

INCOME_VALUE_COLS = [
    "营业总收入",
    "营业收入",
    "营业成本",
    "营业利润",
    "利润总额",
    "净利润",
    "归属于母公司所有者的净利润",
    "财务费用",
    "其中：利息费用(财务费用)",
    "资产减值损失",
    "信用减值损失",
]

CASHFLOW_VALUE_COLS = [
    "经营活动产生的现金流量净额",
    "投资活动产生的现金流量净额",
    "筹资活动产生的现金流量净额",
]

BALANCE_VALUE_COLS = [
    "货币资金",
    "应收账款净额",
    "存货净额",
    "流动资产合计",
    "非流动资产合计",
    "固定资产净额",
    "资产总计",
    "短期借款",
    "应付账款",
    "一年内到期的非流动负债",
    "流动负债合计",
    "长期借款",
    "负债合计",
    "所有者权益合计",
]

def add_financial_metrics(fin: pd.DataFrame) -> pd.DataFrame:
    out = fin.copy()
    for col in INCOME_VALUE_COLS + CASHFLOW_VALUE_COLS + BALANCE_VALUE_COLS:
        if col in out.columns:
            out[col] = to_numeric_series(out[col])

    revenue = out["营业收入"].where(out["营业收入"].notna(), out.get("营业总收入"))
    cost = out.get("营业成本")
    op_profit = out.get("营业利润")
    profit_total = out.get("利润总额")
    net_profit = out["归属于母公司所有者的净利润"].where(
        out["归属于母公司所有者的净利润"].notna(), out.get("净利润")
    )
    total_assets = out.get("资产总计")
    current_assets = out.get("流动资产合计")
    inventory = out.get("存货净额")
    cash = out.get("货币资金")
    current_liab = out.get("流动负债合计")
    short_debt_components = pd.concat(
        [out.get("短期借款"), out.get("一年内到期的非流动负债")],
        axis=1,
    )
    short_debt = short_debt_components.sum(axis=1, min_count=2)
    long_debt = out.get("长期借款")
    total_liab = out.get("负债合计")
    equity = out.get("所有者权益合计")
    cfo = out.get("经营活动产生的现金流量净额")
    investing_cf = out.get("投资活动产生的现金流量净额")
    financing_cf = out.get("筹资活动产生的现金流量净额")
    fin_expense = out.get("财务费用")
    interest_expense = out.get("其中：利息费用(财务费用)").where(
        out.get("其中：利息费用(财务费用)").notna(), fin_expense
    )
    ar = out.get("应收账款净额")
    ap = out.get("应付账款")
    noncurrent_assets = out.get("非流动资产合计")
    fixed_assets = out.get("固定资产净额")
    asset_impairment = out.get("资产减值损失")
    credit_impairment = out.get("信用减值损失")

    out["gross_margin"] = (revenue - cost) / revenue.replace(0, np.nan)
    out["operating_margin"] = op_profit / revenue.replace(0, np.nan)
    out["net_margin"] = net_profit / revenue.replace(0, np.nan)
    out["roa"] = net_profit / total_assets.replace(0, np.nan)
    out["operating_profit_to_assets"] = op_profit / total_assets.replace(0, np.nan)
    out["net_profit_to_assets"] = out["roa"]
    out["financial_expense_to_revenue"] = fin_expense / revenue.replace(0, np.nan)
    out["loss_flag"] = net_profit.lt(0).astype(float).where(net_profit.notna())
    out["asset_impairment_loss_to_assets"] = asset_impairment / total_assets.replace(0, np.nan)

    out["current_ratio"] = current_assets / current_liab.replace(0, np.nan)
    out["quick_ratio"] = (current_assets - inventory) / current_liab.replace(0, np.nan)
    out["cash_ratio"] = cash / current_liab.replace(0, np.nan)
    out["asset_liability_ratio"] = total_liab / total_assets.replace(0, np.nan)
    out["net_debt_to_assets"] = (short_debt + long_debt - cash) / total_assets.replace(0, np.nan)
    out["cash_to_assets"] = cash / total_assets.replace(0, np.nan)
    out["short_debt_to_assets"] = short_debt / total_assets.replace(0, np.nan)
    out["long_debt_to_assets"] = long_debt / total_assets.replace(0, np.nan)
    out["short_debt_zero_flag"] = (short_debt.eq(0) & cash.notna()).astype(float).where(short_debt.notna())
    out["cash_to_short_debt_denominator_missing_flag"] = short_debt.isna().astype(float)
    out["cash_to_short_debt"] = cash / short_debt.replace(0, np.nan)
    ebit_proxy = profit_total + fin_expense
    out["interest_coverage"] = ebit_proxy / interest_expense.abs().replace(0, np.nan)
    out["cfo_to_assets"] = cfo / total_assets.replace(0, np.nan)
    out["cfo_to_current_liabilities"] = cfo / current_liab.replace(0, np.nan)
    out["cfo_to_net_income"] = cfo / net_profit.replace(0, np.nan)
    out["negative_equity_flag"] = equity.lt(0).astype(float).where(equity.notna())
    out["negative_cfo_flag"] = cfo.lt(0).astype(float).where(cfo.notna())

    out["ar_to_current_assets"] = ar / current_assets.replace(0, np.nan)
    out["inventory_to_current_assets"] = inventory / current_assets.replace(0, np.nan)
    out["working_capital_to_assets"] = (current_assets - current_liab) / total_assets.replace(0, np.nan)
    out["total_asset_turnover"] = revenue / total_assets.replace(0, np.nan)
    out["ar_turnover"] = revenue / ar.replace(0, np.nan)
    out["inventory_turnover"] = cost / inventory.replace(0, np.nan)
    out["cfo_to_revenue"] = cfo / revenue.replace(0, np.nan)
    out["accrual_intensity"] = (net_profit - cfo) / total_assets.replace(0, np.nan)
    out["credit_impairment_loss_to_assets"] = credit_impairment / total_assets.replace(0, np.nan)
    out["ap_to_assets"] = ap / total_assets.replace(0, np.nan)
    out["noncurrent_assets_to_assets"] = noncurrent_assets / total_assets.replace(0, np.nan)
    out["fixed_assets_to_assets"] = fixed_assets / total_assets.replace(0, np.nan)

    out["investing_cf_to_assets"] = investing_cf / total_assets.replace(0, np.nan)
    out["financing_cf_to_assets"] = financing_cf / total_assets.replace(0, np.nan)
    out["ln_total_assets"] = np.log(total_assets.where(total_assets > 0))

    out["revenue_for_growth"] = revenue
    out["net_profit_for_growth"] = net_profit
    out["total_assets_for_growth"] = total_assets
    out["operating_profit_for_growth"] = op_profit
    out["profit_total_for_growth"] = profit_total
    out["cfo_for_growth"] = cfo
    out["investing_cf_for_growth"] = investing_cf
    out["financing_cf_for_growth"] = financing_cf

    out["prev_yoy_period"] = out["financial_period_end"] - pd.DateOffset(years=1)
    prev_cols = [
        "stock_code",
        "financial_period_end",
        "revenue_for_growth",
        "total_assets_for_growth",
        "operating_profit_for_growth",
        "net_profit_for_growth",
        "profit_total_for_growth",
        "cfo_for_growth",
        "investing_cf_for_growth",
        "financing_cf_for_growth",
        "gross_margin",
        "asset_liability_ratio",
    ]
    prev = out[prev_cols].rename(columns={
        "financial_period_end": "prev_yoy_period",
        "revenue_for_growth": "prev_revenue",
        "total_assets_for_growth": "prev_total_assets",
        "operating_profit_for_growth": "prev_operating_profit",
        "net_profit_for_growth": "prev_net_profit",
        "profit_total_for_growth": "prev_profit_total",
        "cfo_for_growth": "prev_cfo",
        "investing_cf_for_growth": "prev_investing_cf",
        "financing_cf_for_growth": "prev_financing_cf",
        "gross_margin": "prev_gross_margin",
        "asset_liability_ratio": "prev_asset_liability_ratio",
    })
    out = out.merge(prev, on=["stock_code", "prev_yoy_period"], how="left")

    def growth(curr: pd.Series, prev_name: str) -> pd.Series:
        prev_value = out[prev_name]
        return (curr - prev_value) / prev_value.abs().replace(0, np.nan)

    out["revenue_growth"] = growth(out["revenue_for_growth"], "prev_revenue")
    out["total_assets_growth"] = growth(out["total_assets_for_growth"], "prev_total_assets")
    out["operating_profit_growth"] = growth(out["operating_profit_for_growth"], "prev_operating_profit")
    out["net_profit_growth"] = growth(out["net_profit_for_growth"], "prev_net_profit")
    out["profit_total_growth"] = growth(out["profit_total_for_growth"], "prev_profit_total")
    out["cfo_growth"] = growth(out["cfo_for_growth"], "prev_cfo")
    out["investing_cf_growth"] = growth(out["investing_cf_for_growth"], "prev_investing_cf")
    out["financing_cf_growth"] = growth(out["financing_cf_for_growth"], "prev_financing_cf")
    out["gross_margin_change"] = out["gross_margin"] - out["prev_gross_margin"]
    out["leverage_change"] = out["asset_liability_ratio"] - out["prev_asset_liability_ratio"]
    return out