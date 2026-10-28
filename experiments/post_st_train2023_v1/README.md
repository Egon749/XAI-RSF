# 扩充至2023入组的ST后预后实验

用户授权仅扩充训练信息：2012—2023年入组，训练结局严格早于2024-01-01；评价2024年入组公司。冻结345家训练、26个先退市、198个先恢复、77个完整24月无首次转归、44个行政删失；2024测试45家，24月20先退市、16先恢复、9无首次转归。2023不再作为测试集。

原75列入组时特征、公告日起点、(退市,恢复)分裂权重(1,1)、32树、最大深度6及三个种子42/101/202均保持不变。q2、同后端无限阶和randomForestSRC标准RSF各3个正式任务，另2个4树数值试运行。沿用原训练与R后端，不复制或改动森林核心；明确没有实施下一交易日起点、原版年报恢复、规则特征或原因权重消融。比较估计的是“延长老样本随访＋增加2023入组”的合并效果。

数据与结果始终为PROVISIONAL、后期结果已暴露的探索性比较。禁止按2024表现追加调参、选种子或再校准；不接触2025入组结局。原工作区、旧模型与旧结果保持不变，企业行、预测、切点、参数和完整函数留在服务器。

## 执行入口

在已提交且干净的工作区，使用新的私有`RUN`根目录（prepare创建其子目录），固定隔离Python环境：

```bash
python -m experiments.post_st_train2023_v1.prepare --run "$RUN"
python -m experiments.post_st_prognosis_v1.stages --run "$RUN" --repo "$PWD" --stage freeze
/home/oliver/.local/bin/hetero-submit "$RUN/pilot.jsonl" physical both
python -m experiments.post_st_prognosis_v1.stages --run "$RUN" --stage pilot --job "$PILOT_JOB"
# 仅试运行门检通过后执行正式清单
/home/oliver/.local/bin/hetero-submit "$RUN/formal.jsonl" physical both
python -m experiments.post_st_prognosis_v1.stages --run "$RUN" --stage formal --job "$FORMAL_JOB"
python -m experiments.post_st_train2023_v1.evaluate --run "$RUN" --output "$RUN/evaluation"
python -m experiments.post_st_train2023_v1.validate --run "$RUN" --reports "$RUN/evaluation"
```

使用现有两节点Slurm物理核池与NFS。每个任务独立单线程、地址空间上限4GiB，不将一个Python进程跨机器分割。清单同时有可用于hetero-run的sidecar；当前显式提交physical/both以保留原资源布局。等待任务计算、回传及哈希检查均成功后才报告完成。

评价保留原日历月IPCW、并列分数AP和完整端点检查，仅移除2023／合并测试范围；52组2024模型／时限／原因指标独立核算。种子均值和平均预测后的集成指标分别展示，训练表观指标不充当验证。新的共享切点和解释参考分布由345家训练公司重估，是扩充训练的自然结果，未使用测试特征或结局选择。

## 实际执行结果

已完成9个正式模型及2个试运行，无失败。研究判断为 **PREDICTIVE_NO_GO（当前冻结q2的24个月ST后转归配置）**；PROVISIONAL／探索性状态不变。

q2主要24月退市AUC均值0.542→0.550、AP 0.503→0.549、Brier 0.372→0.343；Top20集成捕获3/20→5/20，Top25仍5/20。概率仍明显低估：平均13.02%对观察44.44%。q2保持完整精确解释，但交互稳定性和稀少退市支持仍不足。没有追加训练或调参。

- [全部预测结果](RESULTS_ZH.md)、[集成、校准与Top20/25](PREDICTION_COMPARISON_ZH.md)。
- [解释、股价使用及事件支持](INTERPRETABILITY_COMPARISON_ZH.md)。
- [数据时钟与对齐](DATA_ALIGNMENT_AUDIT_ZH.md)、[执行诊断](TRAINING_DIAGNOSTICS_ZH.md)。
- [研究决策](METHOD_INNOVATION_DECISION_ZH.md)、[执行回执](execution_receipt.json)。
