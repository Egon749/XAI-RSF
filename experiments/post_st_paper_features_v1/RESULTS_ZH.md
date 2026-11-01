# 论文可映射20变量：12个月实验结果

本轮新增9个森林全部成功（三年×三个种子），派生9个直接父节点AJ收缩模型；原83列及74列的18个参考森林直接复用，没有重训。三个种子先平均逐公司CIF再评价。

**这是论文表2中可映射20列的适配实验，尚不是完整54变量实验，也不是PROBoost算法复现。** 用户明确选择先运行可核验子集并记录缺项及定义差异。原研究模型继续保留。

## 变量范围

来源：Liu等，Progressive gradient boosted trees for imbalanced financial distress prediction，Expert Systems With Applications321(2026)132187，DOI10.1016/j.eswa.2026.132187，PDF第9页表2。原论文54项指标分8类，用于ST分类；本轮沿用ST后先最终退市/先恢复竞争风险任务，不能将两者AUC直接比较。

| paper_id | paper_name | existing_name | operational_definition |
| --- | --- | --- | --- |
| X1 | Return on Assets (ROA) | roa | parent-attributable net profit, fallback total net profit / ending total assets; paper average/basis unspecified |
| X2 | Net Profit Margin on Total Assets | net_profit_to_assets | exact duplicate of existing ROA; paper does not disambiguate X1/X2 formulas |
| X6 | Financial Expense Ratio | financial_expense_to_revenue | financial expense / operating revenue, fallback total revenue |
| X8 | Current Ratio | current_ratio | current assets / current liabilities |
| X9 | Quick Ratio | quick_ratio | (current assets-inventory)/current liabilities |
| X10 | Cash Ratio | cash_ratio | cash / current liabilities; paper cash-equivalent treatment unspecified |
| X11 | Interest Coverage Ratio | interest_coverage | (total profit+financial expense)/abs(interest expense, fallback financial expense); inherited EBIT proxy |
| X12 | Operating Cash Flow/Current Liabilities | cfo_to_current_liabilities | operating cash flow / current liabilities |
| X13 | Debt to Asset Ratio | asset_liability_ratio | total liabilities / total assets |
| X20 | Total Asset Growth Rate | total_assets_growth | same-quarter year-on-year (current-prior)/abs(prior); actual both-period availability verified |
| X21 | Total Revenue Growth Rate | revenue_growth | operating revenue fallback total revenue; same-quarter year-on-year /abs(prior), paper total-revenue basis differs |
| X24 | Accounts Receivable Turnover Ratio | ar_turnover | operating revenue / ending net receivables; no average denominator or annualization |
| X25 | Inventory Turnover Ratio | inventory_turnover | operating cost / ending net inventory; no average denominator or annualization |
| X30 | Total Asset Turnover Ratio | total_asset_turnover | operating revenue / ending total assets; no average denominator or annualization |
| X33 | Cash Asset Ratio | cash_to_assets | cash / total assets |
| X35 | Working Capital Ratio | working_capital_to_assets | (current assets-current liabilities)/total assets |
| X39 | Net Profit Cash Net Containment | cfo_to_net_income | operating cash flow / parent-attributable net profit, fallback total net profit |
| X40 | Cash Content of Operating Revenue | cfo_to_revenue | operating cash flow / operating revenue, fallback total revenue |
| X41 | Total Cash Recovery Rate | cfo_to_assets | operationalized as operating cash flow / total assets; paper does not give exact formula |
| X53 | Market Value | log_total_mktcap_last | invert existing natural log to total market value in CSMAR thousand-CNY units, at same pre-open market date |

这些字段按数据/定义可对应程度确定，没有按测试成绩筛选。论文没有提供逐项公式，故匹配是显式既有口径映射，不是作者原代码逐值复现。X1与X2在冻结输入中逐值相同，20列不能视为20种独立信息；保留两列并记录重复，未临时去冗余。周转率使用期末余额且不年化、收入优先营业收入、净利润优先归母、利息保障倍数为已有代理公式、X41采用经营现金流/总资产，均有对应说明。X53从原自然对数总市值反变换为CSMAR千元单位，日期仍为原可得市场日期。

| paper_id | paper_name | status |
| --- | --- | --- |
| X3 | Return on Equity (ROE) | NOT_IN_VERIFIED_FROZEN83 |
| X4 | Return on Invested Capital (ROIC) | NOT_IN_VERIFIED_FROZEN83 |
| X5 | Return on Long-Term Capital | NOT_IN_VERIFIED_FROZEN83 |
| X7 | Cost Expense Profit Margin | NOT_IN_VERIFIED_FROZEN83 |
| X14 | Tangible Asset to Interest-Bearing Debt Ratio | NOT_IN_VERIFIED_FROZEN83 |
| X15 | EBITD/Total Debt Ratio | NOT_IN_VERIFIED_FROZEN83 |
| X16 | Tangible Net Asset Debt Ratio | NOT_IN_VERIFIED_FROZEN83 |
| X17 | Capital Preservation and Value Added Rate | NOT_IN_VERIFIED_FROZEN83 |
| X18 | Capital Accumulation Rate | NOT_IN_VERIFIED_FROZEN83 |
| X19 | Fixed Asset Growth Rate | NOT_IN_VERIFIED_FROZEN83 |
| X22 | Management Expense Growth Rate | NOT_IN_VERIFIED_FROZEN83 |
| X23 | Sustainable Growth Rate | NOT_IN_VERIFIED_FROZEN83 |
| X26 | Operating Cycle | NOT_IN_VERIFIED_FROZEN83 |
| X27 | Accounts Payable Turnover Ratio | NOT_IN_VERIFIED_FROZEN83 |
| X28 | Fixed Asset Turnover Ratio | NOT_IN_VERIFIED_FROZEN83 |
| X29 | Capital Intensity | NOT_IN_VERIFIED_FROZEN83 |
| X31 | Shareholder Equity Turnover Ratio | NOT_IN_VERIFIED_FROZEN83 |
| X32 | Current Asset Ratio | NOT_IN_VERIFIED_FROZEN83 |
| X34 | Receivables Asset Ratio | NOT_IN_VERIFIED_FROZEN83 |
| X36 | Long-Term Asset Suitability Ratio | NOT_IN_VERIFIED_FROZEN83 |
| X37 | Operating Liabilities Ratio | NOT_IN_VERIFIED_FROZEN83 |
| X38 | Profit Share from Core Business | NOT_IN_VERIFIED_FROZEN83 |
| X42 | Operating Index | NOT_IN_VERIFIED_FROZEN83 |
| X43 | Cash Suitability Ratio | NOT_IN_VERIFIED_FROZEN83 |
| X44 | Cash to Investment Satisfaction Ratio | NOT_IN_VERIFIED_FROZEN83 |
| X45 | Earnings per Share (EPS) | NOT_IN_VERIFIED_FROZEN83 |
| X46 | Net Asset per Share | NOT_IN_VERIFIED_FROZEN83 |
| X47 | Debt per Share | NOT_IN_VERIFIED_FROZEN83 |
| X48 | Operating Cash Flow per Share | NOT_IN_VERIFIED_FROZEN83 |
| X49 | Net Cash Flow per Share | NOT_IN_VERIFIED_FROZEN83 |
| X50 | Price to Earnings Ratio (P/E) | NOT_IN_VERIFIED_FROZEN83 |
| X51 | Price to Sales Ratio (P/S) | NOT_IN_VERIFIED_FROZEN83 |
| X52 | Price to Cash Flow Ratio (P/CF) | NOT_IN_VERIFIED_FROZEN83 |
| X54 | Book to Market Ratio | NOT_IN_VERIFIED_FROZEN83 |

上述34项未进入本轮冻结输入，原因是尚未完成新字段构造、确切定义及历史版本/各组成数据可得日核验；不等于所有原始科目都不存在。盈利能力FI_T5等预计算表的Accper是报表期，不是公告日，不能直接将其视为PIT合格数据。未制造可得日、空白占位54列或未经说明的替代变量。

## 统一预测比较

| year | arm | companies | known_target_events | auc | ap | brier | risk_mean | ipcw_observed_incidence | checked | captured | coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2023 | reference83 | 36 | 4 | 0.695312 | 0.198962 | 0.106531 | 0.019082 | 0.111111 | 8 | 1 | 0.250000 |
| 2023 | drop9 | 36 | 4 | 0.718750 | 0.290923 | 0.105864 | 0.021031 | 0.111111 | 8 | 2 | 0.500000 |
| 2023 | paper20 | 36 | 4 | 0.687500 | 0.290165 | 0.105331 | 0.021234 | 0.111111 | 8 | 2 | 0.500000 |
| 2024 | reference83 | 45 | 16 | 0.806034 | 0.718365 | 0.328452 | 0.032367 | 0.355556 | 9 | 6 | 0.375000 |
| 2024 | drop9 | 45 | 16 | 0.849138 | 0.799152 | 0.328816 | 0.032370 | 0.355556 | 9 | 8 | 0.500000 |
| 2024 | paper20 | 45 | 16 | 0.642241 | 0.455888 | 0.328005 | 0.035371 | 0.355556 | 9 | 4 | 0.250000 |
| 2025 | reference83 | 53 | 5 | 0.700000 | 0.181317 | 0.084689 | 0.058547 | 0.094340 | 11 | 2 | 0.400000 |
| 2025 | drop9 | 53 | 5 | 0.733333 | 0.207953 | 0.083828 | 0.061081 | 0.094340 | 11 | 3 | 0.600000 |
| 2025 | paper20 | 53 | 5 | 0.670833 | 0.150063 | 0.086194 | 0.062143 | 0.094340 | 11 | 1 | 0.200000 |

reference83=原83变量；drop9=固定删9后的74变量；paper20=论文可映射20列。每年1月1日前可得历史信息训练，评价当年新入组企业各自12个日历月内先最终退市。Brier越低越好，其余排序/覆盖指标越高越好。Top20按当年公司数的20%向上取整，分别核查8、9、11家公司，保留已知竞争事件与未知删失区别。三个年度12个月终点均可评价，IPCW权重为1。

2023年：论文可映射20列AUC=0.687500、AP=0.290165、Brier=0.105331，Top20核查8家，捕获2/4。相对83列，AUC变化-0.007812、AP变化+0.091202、Brier变化-0.001200；相对74列，AUC变化-0.031250、AP变化-0.000758、Brier变化-0.000533。平均风险2.12%，观察先退市率11.11%。
2024年：论文可映射20列AUC=0.642241、AP=0.455888、Brier=0.328005，Top20核查9家，捕获4/16。相对83列，AUC变化-0.163793、AP变化-0.262477、Brier变化-0.000447；相对74列，AUC变化-0.206897、AP变化-0.343265、Brier变化-0.000811。平均风险3.54%，观察先退市率35.56%。
2025年：论文可映射20列AUC=0.670833、AP=0.150063、Brier=0.086194，Top20核查11家，捕获1/5。相对83列，AUC变化-0.029167、AP变化-0.031253、Brier变化+0.001505；相对74列，AUC变化-0.062500、AP变化-0.057890、Brier变化+0.002366。平均风险6.21%，观察先退市率9.43%。

综合判断：这20列适配方案没有表现出稳定的预测收益。2023年相对83列的AP和Top20有所改善，与74列的AP和捕获数基本相同；2024年AUC、AP和Top20捕获下降，2025年这三项均低于两个参考方案。2024年Brier仅略有降低，不能据此忽略排序和事件捕获的损失。20列模型的平均风险仍低于各年观察发生率，特别是2024年仍为3.54%对35.56%，风险低估没有解决。

完整表示的函数数和空间格数减少，是可观察到的表示收益；它伴随部分年度预测损失。本文不以这些已查看队列上的结果作确认性方法选择，也不将全部下降归因于某个被删字段：变量内容、每树覆盖以及市场变量尺度共同变化，尚未被单独分离。

这种结果只评价当前适配输入集。不能将paper20的表现外推为完整论文54变量的优劣，也不能把年度AP简单平均当作合并AP。当前研究参考不因某个指标变化自动替换。

## 计算预算与解释

三套方案都使用q2、深度4、32树/种子、共享分位切点、mtry75、min_leaf10/min_groups5/min_events3、原因权重(1,1)、直接raw父节点AJ强度10及原24个月训练随访上限。由于只有20列，paper20每树使用全部20列；原83/74方案每树使用25列，因此输入内容、覆盖与市值变换共同改变，不能声称实际候选搜索量完全匹配。保留所有阈值候选与实际评分次数。

| year | arm | main_functions | interaction_functions | spatial_table_cells | scalar_table_cells | distinct_thresholds |
| --- | --- | --- | --- | --- | --- | --- |
| 2023 | reference83 | 62 | 144 | 3146 | 773916 | NA |
| 2023 | drop9 | 60 | 138 | 3010 | 740460 | NA |
| 2023 | paper20 | 20 | 66 | 2126 | 522996 | 96.000000 |
| 2024 | reference83 | 65 | 146 | 3263 | 913640 | NA |
| 2024 | drop9 | 58 | 132 | 2933 | 821240 | NA |
| 2024 | paper20 | 20 | 69 | 2239 | 626920 | 99.000000 |
| 2025 | reference83 | 64 | 147 | 3194 | 1034856 | NA |
| 2025 | drop9 | 54 | 131 | 3001 | 972324 | NA |
| 2025 | paper20 | 19 | 70 | 2367 | 766908 | 93.000000 |

新9个raw与9个收缩版本，以及3个96树平均预测器，在全部训练/评价公司、所有原生时点及两原因上验证完整q2重构、保存重载和CIF合法性；最大误差1.998e-15，容差1e-10。旧参考模型哈希不变、逐公司预测复现，并重新算出48行原12个月指标一致。

空间格数包含缺失状态，不乘时间/原因维；标量格数另包含这些维度。数学上精确表示不等于真实作用恢复，X1/X2重复仍带来解释归属歧义；此实验不声称改善可辨识性。

## 配对差值

| year | reference_arm | metric | difference | lower95 | upper95 | undefined_bootstraps |
| --- | --- | --- | --- | --- | --- | --- |
| 2023 | reference83 | auc | -0.007812 | -0.391619 | 0.303030 | 11 |
| 2023 | reference83 | ap | 0.091202 | -0.133656 | 0.502639 | 11 |
| 2023 | reference83 | brier | -0.001200 | -0.004511 | 0.001394 | 0 |
| 2023 | reference83 | risk_mean | 0.002152 | -0.002237 | 0.006734 | 0 |
| 2023 | reference83 | captured | 1.000000 | -2.000000 | 4.000000 | 0 |
| 2023 | reference83 | coverage | 0.250000 | -0.500000 | 1.000000 | 11 |
| 2023 | drop9 | auc | -0.031250 | -0.338235 | 0.279412 | 11 |
| 2023 | drop9 | ap | -0.000758 | -0.460833 | 0.433417 | 11 |
| 2023 | drop9 | brier | -0.000533 | -0.002999 | 0.001531 | 0 |
| 2023 | drop9 | risk_mean | 0.000202 | -0.003779 | 0.004197 | 0 |
| 2023 | drop9 | captured | 0.000000 | -2.000000 | 2.000000 | 0 |
| 2023 | drop9 | coverage | 0.000000 | -0.500000 | 0.500000 | 11 |
| 2024 | reference83 | auc | -0.163793 | -0.330020 | 0.034532 | 0 |
| 2024 | reference83 | ap | -0.262477 | -0.393875 | -0.069554 | 0 |
| 2024 | reference83 | brier | -0.000447 | -0.006812 | 0.005358 | 0 |
| 2024 | reference83 | risk_mean | 0.003003 | -0.002020 | 0.008737 | 0 |
| 2024 | reference83 | captured | -2.000000 | -5.000000 | 2.000000 | 0 |
| 2024 | reference83 | coverage | -0.125000 | -0.363756 | 0.125000 | 0 |
| 2024 | drop9 | auc | -0.206897 | -0.415967 | 0.004008 | 0 |
| 2024 | drop9 | ap | -0.343265 | -0.537080 | -0.063861 | 0 |
| 2024 | drop9 | brier | -0.000811 | -0.007334 | 0.005566 | 0 |
| 2024 | drop9 | risk_mean | 0.003001 | -0.002776 | 0.009448 | 0 |
| 2024 | drop9 | captured | -4.000000 | -7.000000 | 1.000000 | 0 |
| 2024 | drop9 | coverage | -0.250000 | -0.538636 | 0.043582 | 0 |
| 2025 | reference83 | auc | -0.029167 | -0.208271 | 0.158792 | 10 |
| 2025 | reference83 | ap | -0.031253 | -0.227014 | 0.053145 | 10 |
| 2025 | reference83 | brier | 0.001505 | -0.001052 | 0.004196 | 0 |
| 2025 | reference83 | risk_mean | 0.003597 | -0.003391 | 0.011003 | 0 |
| 2025 | reference83 | captured | -1.000000 | -4.000000 | 1.000000 | 0 |
| 2025 | reference83 | coverage | -0.200000 | -0.724107 | 0.250000 | 10 |
| 2025 | drop9 | auc | -0.062500 | -0.235479 | 0.140021 | 10 |
| 2025 | drop9 | ap | -0.057890 | -0.242025 | 0.032166 | 10 |
| 2025 | drop9 | brier | 0.002366 | -0.000357 | 0.005373 | 0 |
| 2025 | drop9 | risk_mean | 0.001062 | -0.006611 | 0.008996 | 0 |
| 2025 | drop9 | captured | -2.000000 | -4.000000 | 0.000000 | 0 |
| 2025 | drop9 | coverage | -0.400000 | -1.000000 | 0.000000 | 10 |

差值为paper20减对应参考。每年1000次公司配对bootstrap，三模型使用同一抽样矩阵；无事件导致指标无法计算的次数完整记录。区间仅反映固定预测条件下测试样本波动，不包含训练或变量方案选择不确定性。所有单种子、恢复原因、校准分箱、风险水平与不利结果完整保留。
