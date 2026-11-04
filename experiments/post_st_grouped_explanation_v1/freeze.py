"""Freeze a disjoint dictionary by exact name before inspecting stability."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.post_st_seed_explanation_stability_v1.common import run_path as old_run,sha
D=Path(__file__).resolve().parent
SPEC=[
("盈利能力","Profitability","gross_margin operating_margin roa financial_expense_to_revenue loss_flag asset_impairment_loss_to_assets accrual_intensity","利润形成、费用侵蚀或盈利质量"),
("偿债能力","Solvency","current_ratio quick_ratio cash_ratio asset_liability_ratio net_debt_to_assets cash_to_assets short_debt_to_assets long_debt_to_assets log_cash_to_short_debt interest_coverage negative_equity_flag leverage_change","现金保障、债务负担、资本缓冲或其变化"),
("营运能力","Operating efficiency","ar_to_current_assets inventory_to_current_assets total_asset_turnover ar_turnover inventory_turnover ap_to_assets noncurrent_assets_to_assets fixed_assets_to_assets","资产配置、营运资本占用或周转效率"),
("发展能力","Growth","revenue_growth total_assets_growth operating_profit_growth net_profit_growth cfo_growth investing_cf_growth financing_cf_growth gross_margin_change","收入、资产、利润率或现金流随期变化"),
("现金流与融资","Cash flow and financing","cfo_to_assets cfo_to_net_income negative_cfo_flag cfo_to_revenue investing_cf_to_assets financing_cf_to_assets","经营、投资或筹资活动的现金流水平及转换"),
("市场信息","Market","log_price_last log_total_mktcap_last log_float_mktcap_last ret_cum_1m ret_vol_1m max_drawdown_1m log_amihud_illiq_1m ret_cum_3m ret_vol_3m max_drawdown_3m log_amihud_illiq_3m var_5pct_3m es_5pct_3m extreme_negative_days_3m turnover_mean_3m log_amount_mean_3m intraday_range_mean_3m zero_amount_ratio_3m ret_cum_12m ret_vol_12m max_drawdown_12m","由证券交易、估值、收益、流动性或尾部波动度量"),
("审计信息","Audit","prior_audit_standard prior_audit_qualified prior_audit_adverse prior_audit_disclaimer prior_audit_explanatory prior_audit_qualified_matter prior_audit_going_concern","上一期审计意见、事项或持续经营信号"),
("规模及数据状态","Size and data status","ln_total_assets fin_period_age_scaled fin_publication_age_scaled market_missing_coverage entry_is_star_st","企业规模、披露时效、数据覆盖或入组时的警示类别")]
NOTES={'financial_expense_to_revenue':'融资成本也关系偿债；按其对营业收入及盈利的侵蚀归入盈利。','accrual_intensity':'应计项目与现金流相关；作为盈利质量代理放在盈利。','ap_to_assets':'应付账款也是负债；此处描述经营往来形成的资金占用。','financing_cf_to_assets':'筹资现金流取决于融资行为；现金流量水平归入现金流与融资。','cfo_growth':'按增长而非现金流水平归入发展。','financing_cf_growth':'筹资现金流的跨期变化归发展。','investing_cf_growth':'投资现金流的跨期变化归发展。','gross_margin_change':'毛利率的变化归发展，毛利率水平归盈利。','leverage_change':'杠杆虽为变化量，其直接含义是债务负担变化。','cash_to_assets':'存量现金反映偿债缓冲，区别于现金流量。','noncurrent_assets_to_assets':'资产结构仅间接表征营运能力。','fixed_assets_to_assets':'固定资产占比是配置而非纯周转指标。','market_missing_coverage':'描述市场信息覆盖质量，归数据状态。','entry_is_star_st':'标识入组警示状态，并非一种财务能力。','asset_impairment_loss_to_assets':'减值也反映资产质量，按当期损失归盈利。','log_total_mktcap_last':'市值也刻画规模，按市场估值归市场信息。','log_float_mktcap_last':'流通市值按交易市场估值归类。','cfo_to_net_income':'也刻画盈利质量，此处侧重利润的现金转换。'}
def main():
    f=pd.DataFrame(json.loads((old_run()/'input/features.json').read_text()))
    bundle=pd.read_csv(D.parents[1]/'主模型/配置/INPUT_FEATURES.csv')
    pd.testing.assert_frame_equal(f,bundle[f.columns]);assert bundle.input_position.tolist()==list(range(74))
    names=[s for _,_,seq,_ in SPEC for s in seq.split()]
    assert len(set(names))==len(names)==74 and set(f.feature_name)==set(names)
    out=[]
    for i,row in f.iterrows():
        (g,(zh,en,_,reason)),= [(j,z) for j,z in enumerate(SPEC) if row.feature_name in z[2].split()]
        out.append(dict(input_position=i,original_feature_id=row.feature_id,english_name=row.feature_name,chinese_name=row.label,group=f'G{g+1}',group_zh=zh,group_en=en,inherited_field_type=row.dtype,reason=NOTES.get(row.feature_name,reason)))
    frame=pd.DataFrame(out);frame.to_csv(D/'FEATURE_GROUP_MAPPING.csv',index=False)
    codes=frame.group.str[1:].astype(int).to_numpy();rng=np.random.default_rng(2026110401)
    maps=np.stack([rng.permutation(codes) for _ in range(200)])
    assert len(np.unique(maps,axis=0))==200 and not np.any(np.all(maps==codes,axis=1))
    pd.DataFrame(maps,index=np.arange(1,201),columns=f.feature_name).rename_axis('random_id').to_csv(D/'RANDOM_MAPS.csv')
if __name__=='__main__':main()
