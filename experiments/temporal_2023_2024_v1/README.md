# 2023与2024分别评价：已完成

**6/6正式模型、2/2试运行成功；PROVISIONAL，回溯时间外评价，非严格盲测。** 冻结训练原点2012—2020、拟合截止2022-01-01；两年共用相同模型，未按年度重新训练、未用后期结局调参。

主要口径为记录等权、三个固定种子的指标均值：

|年度|配置|AUC|AP|IPCW Brier|Top20捕获独立事件|
|---|---|---:|---:|---:|---:|
|2023|当前coverage q2|0.827433|0.097264|0.010242|53.33/83|
|2023|同后端不限阶数|0.850379|0.123736|0.010775|58.67/83|
|2023|冻结M2|0.860482|0.135388|0.010284|58.33/83|
|2024|当前coverage q2|0.826456|0.094958|0.011949|48.67/99|
|2024|同后端不限阶数|0.832775|0.129032|0.012821|59.67/99|
|2024|冻结M2|0.859592|0.153232|0.012157|66.00/99|

小数捕获数来自种子平均。完整重构最大误差1.388e-16，平均1317空间格。**预测保留目标尚未达到；正式研究判断INCONCLUSIVE。** 公司等权Brier不占优，高风险区未知结局集中，不能用记录等权Brier略低宣布成功。读完整结论后再引用指标。

- [完整年度/合并、公司等权、集成、Top20/25及校准结果](RESULTS_2023_2024_ZH.md)
- [研究判断与负面发现](METHOD_INNOVATION_DECISION_ZH.md)
- [数据与来源时钟审计](DATA_ALIGNMENT_AUDIT_ZH.md)、[原始来源哈希](data_alignment.json)、[M2来源](M2_provenance.json)
- [训练诊断](TRAINING_DIAGNOSTICS_ZH.md)、[解释与稳定性](INTERPRETABILITY_COMPARISON_ZH.md)
- [冻结协议](PROTOCOL_ZH.md)、[完整固定配置](frozen_protocol.json)、[复核入口](RUN_ZH.md)
- [执行回执与模型/预测哈希](execution_receipt.json)、[评价环境](environment.json)、[两节点环境](node_environments.json)、[交付测试](tests_delivery.txt)、[交付一致性核验](delivery_validation.json)
- [逐种子指标](metrics.csv)、[种子均值与范围](metrics_seed_summary.csv)、[集成指标](metrics_ensemble.csv)
- [季度/年度预警](warning_metrics.csv)、[预警种子汇总](warning_seed_summary.csv)、[集成预警](warning_ensemble.csv)
- [校准](calibration.csv)、[高风险随访支持](high_risk_followup_support.csv)、[配对差值](paired_contrasts.csv)、[种子稳定性](seed_stability.csv)

训练集87,134条、283个独立目标事件；2023/2024分别18,162和18,606条，独立目标事件83和99个，跨年合并去重137个。原始模型、预测、函数系数、真实切点及公司案例保留服务器私有目录，见执行回执。
