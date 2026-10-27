# 制造业重训及2023/2024评价：已完成

6/6正式模型、2/2试运行完成；**INCONCLUSIVE，PROVISIONAL_REPORT_LINKED_MANUFACTURING**。训练原点2012—2020、拟合截止2022-01-01，2023/2024分别评价，没有用后期结果调参。

以下均在相同制造业评价子集上，记录等权、三个种子指标均值：

|年度|模型训练范围与配置|AUC|AP|IPCW Brier|Top20捕获独立事件|
|---|---|---:|---:|---:|---:|
|2023|制造业q2重训|0.849045|0.074258|0.008189|37.00/44|
|2023|原全行业q2|0.830958|0.081219|0.008174|36.33/44|
|2023|制造业不限阶数重训|0.880118|0.095495|0.009232|40.67/44|
|2023|原全行业M2|0.867136|0.112737|0.008434|38.67/44|
|2024|制造业q2重训|0.834813|0.095680|0.010774|42.67/60|
|2024|原全行业q2|0.827441|0.108379|0.010696|43.00/60|
|2024|制造业不限阶数重训|0.870234|0.165847|0.011770|50.00/60|
|2024|原全行业M2|0.859507|0.171206|0.010855|46.33/60|

**q2重训改善AUC，但AP下降，Top20基本不变。** 精确解释仍成立（最大误差2.220e-16），平均1304.33空间格，交互集合稳定性和低支持问题仍在。原全行业模型只复用预测；M2没有制造业重训，不能误标。

训练56,458条、177个独立目标事件；2023/2024分别11,686/12,113条、44/60个目标事件，合并去重79个。行业按原点选用财报匹配，但缺少行业版本日期，不能宣称正式PIT认证。

- [完整年度、合并、公司等权及集成结果](RESULTS_2023_2024_ZH.md)
- [研究判断与重训增量](METHOD_INNOVATION_DECISION_ZH.md)
- [行业与样本审计](DATA_ALIGNMENT_AUDIT_ZH.md)、[排除明细汇总](industry_selection_counts.csv)、[来源哈希](data_alignment.json)
- [训练诊断](TRAINING_DIAGNOSTICS_ZH.md)、[解释诊断](INTERPRETABILITY_COMPARISON_ZH.md)
- [冻结协议](PROTOCOL_ZH.md)、[冻结配置](frozen_protocol.json)、[运行入口](RUN_ZH.md)
- [执行回执与模型哈希](execution_receipt.json)、[指标](metrics.csv)、[指标种子汇总](metrics_seed_summary.csv)、[集成指标](metrics_ensemble.csv)
- [交付核验](delivery_validation.json)、[子集完整性](subset_integrity.json)、[完整测试](tests_delivery.txt)、[训练/评价环境](environment.json)、[原表读取环境](preparation_environment.json)
- [全部季度预警](warning_metrics.csv)、[预警种子汇总](warning_seed_summary.csv)、[集成预警](warning_ensemble.csv)、[配对差值](paired_contrasts.csv)
- [校准](calibration.csv)、[高风险随访支持](high_risk_followup_support.csv)、[跨种子稳定性](seed_stability.csv)

私有模型、函数参数、真实阈值、逐行行业分类、公司预测及个体解释均保留服务器，路径见执行回执。没有继续启动第二轮搜索。
