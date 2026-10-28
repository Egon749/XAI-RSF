# 训练至2024年、评价2025年：12个月ST后首次转归

用户于2026-10-27授权：将2024年入组纳入训练，以2025年入组作为评价，观察期改为12个日历月。本轮同时将**训练和评价随访**截到入组后12个月，训练结局信息严格早于2025-01-01。目标仍是首次财务／审计ST后，先完整恢复A还是先最终退市X；其他中间状态不改成终点。

冻结原75个输入、公告日起点、原因权重(1,1)、32棵树、深度6、三个种子42/101/202。q2、同后端无限阶、randomForestSRC标准RSF各3个正式拟合，另2个4树数值试运行。没有追加原因权重或超参数消融。两次改变（训练年份、最大随访）同时发生，与旧24个月拟合不是单因素比较；不能用2024与2025不同公司的指标差证明算法提升。

准备结果：训练390家，12个月内23个先退市、73个先恢复、263个完整时限无首次转归、31个提前删失；2025评价53家，5个先退市、3个先恢复、45个完整时限无首次转归，未知0。5个目标事件意味着结果非常有限，所有种子和不利指标均报告，不以结果好坏修改配置。2025此前未经过独立封存，不宣称严格盲测。

2025日行情从已有两个原始CSMAR工作簿恢复／核对，与其缓存逐行比较共1,303,971行；旧390家入组特征逐值精确匹配。所有市场日期、财务可得日期仍严格早于原公告起点。下一交易日起点、同日原版年报恢复及警示类型新增特征仍未实施；原财务和行业版本未完全PIT认证，结果状态为PROVISIONAL。

## 运行入口

先创建唯一私有`RUN`及空的`input audit private reports results`子目录，在隔离Python环境执行：

```bash
python -m experiments.post_st_train2024_h12_v1.prepare --run "$RUN"
# 原始行情校验使用verify_market_source.py，两个source独立作业，全部通过后才拟合。
python -m experiments.post_st_train2024_h12_v1.stages --run "$RUN" --repo "$PWD" --stage freeze
/home/oliver/.local/bin/hetero-submit "$RUN/pilot.jsonl" physical both
python -m experiments.post_st_train2024_h12_v1.stages --run "$RUN" --stage pilot --job "$PILOT_JOB"
/home/oliver/.local/bin/hetero-submit "$RUN/formal.jsonl" physical both
python -m experiments.post_st_train2024_h12_v1.stages --run "$RUN" --stage formal --job "$FORMAL_JOB"
python -m experiments.post_st_train2024_h12_v1.evaluate --run "$RUN" --output "$RUN/evaluation"
python -m experiments.post_st_train2024_h12_v1.validate --run "$RUN" --reports "$RUN/evaluation"
python -m experiments.post_st_train2024_h12_v1.audit_extra --run "$RUN"
# 完整 pytest（含独立 R 检查）的日志保存为 "$RUN/audit/tests_final.log" 后：
python -m experiments.post_st_train2024_h12_v1.deliver --run "$RUN" --destination "$DESTINATION"
```

`DESTINATION`为独立报告目录，先放入本轮README；报告和results目录必须尚不存在，交付脚本拒绝覆盖已有结果。该脚本只汇总已保存的预测，不重新拟合。

只在试运行数值门检通过后提交正式任务。全部任务独立单线程，单任务地址空间上限4GiB；使用现有Slurm两节点物理核池和NFS，回传及哈希核验通过才算成功。真实公司明细、预测、切点、参数与完整函数留在服务器，仓库仅交付代码、协议和汇总。

## 实际交付

9个正式模型＋2个试运行成功，83项测试通过。2025年退市AUC三种子均值：q2 0.544、无限阶0.590、标准RSF0.528；固定集成Top20/25均捕获1/5事件。q2 Brier约0.08765，未优于无变量AJ约0.08664。研究决策为 **INCONCLUSIVE**，少事件与数据证据限制不允许宣称方法成功。

- [完整预测结果](RESULTS_ZH.md)、[核查容量与校准](PREDICTION_COMPARISON_ZH.md)。
- [解释与稳定性](INTERPRETABILITY_COMPARISON_ZH.md)、[数据对齐审计](DATA_ALIGNMENT_AUDIT_ZH.md)。
- [执行诊断](TRAINING_DIAGNOSTICS_ZH.md)、[研究判断](METHOD_INNOVATION_DECISION_ZH.md)、[执行回执](execution_receipt.json)。
