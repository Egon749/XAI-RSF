# 首次财务／审计ST后恢复与退市：公告日实验

本轮已实际完成2个小规模试运行和9个正式模型；无拟合或发布失败。所有入组、恢复、最终退市日期均采用公告日。使用全部非金融行业，2012—2022年入组309家公司训练，拟合结局信息严格截止2023-01-01之前；同一模型评价2023年36家和2024年45家。主终点为入组后24个日历月内“先恢复还是先最终退市”，12个月为补充。

**本配置结论：PREDICTIVE_NO_GO。数据与结果性质：PROVISIONAL、已接触后期结果的探索性回顾评价。** 二阶模型可稳定运行并完整精确解释，但2024年退市区分度接近随机，恢复区分度较差，三个模型均严重低估退市概率。本轮不支持把该配置直接用于此困境后预测任务；不将这一结论外推成整个RSF框架失效。没有追加调参、测试集再校准或第二轮训练。

| 文件 | 内容 |
| --- | --- |
| [PROTOCOL_ZH.md](PROTOCOL_ZH.md) | 冻结目标、模型、时间与评价口径；复现入口 |
| [DATA_ALIGNMENT_AUDIT_ZH.md](DATA_ALIGNMENT_AUDIT_ZH.md) | 样本、公告时钟、随访来源、PIT和行业限制 |
| [RESULTS_ZH.md](RESULTS_ZH.md) | 两个结局、两个时限、分年与合并、三种子均值 |
| [PREDICTION_COMPARISON_ZH.md](PREDICTION_COMPARISON_ZH.md) | 集成概率、校准、Top20/25和提前识别 |
| [INTERPRETABILITY_COMPARISON_ZH.md](INTERPRETABILITY_COMPARISON_ZH.md) | 全部函数复杂度、重构和稳定性 |
| [TRAINING_DIAGNOSTICS_ZH.md](TRAINING_DIAGNOSTICS_ZH.md) | 实际资源、数值检查、失败与更正记录 |
| [METHOD_INNOVATION_DECISION_ZH.md](METHOD_INNOVATION_DECISION_ZH.md) | 负面结论及可以／不能归因的原因 |
| [EVALUATION_CORRECTION_ZH.md](EVALUATION_CORRECTION_ZH.md) | 730/731天混合导致的旧IPCW评价更正 |
| [execution_receipt.json](execution_receipt.json) | 任务、代码、输入和模型哈希 |
| [results/](results/) | 允许发布的全部汇总CSV、图和环境 |

本分支为 `research/xai-rsf-post-st-prognosis-v1`，不替换原首次ST预警实验。企业明细、真实预测、真实切点、完整模型与企业解释仍仅保存在服务器私有运行目录。新增标准RSF针对新终点重新拟合，不冒充此前首次ST预警M2。
