# 稳定入选主效应：全部15个函数形状

稳定指V19/V20三个种子共六模型都选择了该变量，不代表形状稳定。

每图上排V19、下排V20；每列对应90/180/365日，三条线为种子42/101/202。纵轴为目标ST的CIF贡献（百分点），不是风险率或因果影响。

横轴为各训练折非缺失输入的经验分位位置，同一百分位在两折不一定对应相同原始值；NA为单独的缺失状态，未接入数值曲线。每个函数的六面板共用纵轴尺度，不同函数尺度不同。没有平滑或补出新的切点。

90/180日仅为3/6个月近似展示；原始研究的12个日历月评价口径未改。365/366日数值均在JSON和CSV内。所有区间精确函数值见[JSON](stable_functions_percentile.json)。这15个主效应不是完整森林全部解释项。

## 毛利率：`gross_margin`

![gross_margin](figures/gross_margin.svg)

## 净利率：`net_margin`

![net_margin](figures/net_margin.svg)

## ROA：`roa`

![roa](figures/roa.svg)

## 营业利润/总资产：`operating_profit_to_assets`

![operating_profit_to_assets](figures/operating_profit_to_assets.svg)

## 净利润/总资产：`net_profit_to_assets`

![net_profit_to_assets](figures/net_profit_to_assets.svg)

## 财务费用/营业收入：`financial_expense_to_revenue`

![financial_expense_to_revenue](figures/financial_expense_to_revenue.svg)

## 亏损标记：`loss_flag`

![loss_flag](figures/loss_flag.svg)

## 净债务/总资产：`net_debt_to_assets`

![net_debt_to_assets](figures/net_debt_to_assets.svg)

## 利息保障倍数：`interest_coverage`

![interest_coverage](figures/interest_coverage.svg)

## 总资产增长：`total_assets_growth`

![total_assets_growth](figures/total_assets_growth.svg)

## 杠杆变化：`leverage_change`

![leverage_change](figures/leverage_change.svg)

## 近1个月累计收益率：`ret_cum_1m`

![ret_cum_1m](figures/ret_cum_1m.svg)

缺失点幅度较大，下面另给非缺失区间的放大图；缺失效应仍完整保留在上图。

![ret_cum_1m observed detail](figures/ret_cum_1m__observed_only.svg)

## 近1个月收益波动率：`ret_vol_1m`

![ret_vol_1m](figures/ret_vol_1m.svg)

缺失点幅度较大，下面另给非缺失区间的放大图；缺失效应仍完整保留在上图。

![ret_vol_1m observed detail](figures/ret_vol_1m__observed_only.svg)

## 近1个月最大回撤：`max_drawdown_1m`

![max_drawdown_1m](figures/max_drawdown_1m.svg)

缺失点幅度较大，下面另给非缺失区间的放大图；缺失效应仍完整保留在上图。

![max_drawdown_1m observed detail](figures/max_drawdown_1m__observed_only.svg)

## 近1个月Amihud非流动性对数：`log_amihud_illiq_1m`

![log_amihud_illiq_1m](figures/log_amihud_illiq_1m.svg)

缺失点幅度较大，下面另给非缺失区间的放大图；缺失效应仍完整保留在上图。

![log_amihud_illiq_1m observed detail](figures/log_amihud_illiq_1m__observed_only.svg)
