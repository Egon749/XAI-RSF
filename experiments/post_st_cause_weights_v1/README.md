# ST后竞争风险森林：原因权重有限匹配实验

本轮只比较退市／恢复分裂权重 `(1,1)`、`(2,1)`、`(1,0)`，保持q2路径、共享切点、候选资格、公司bootstrap、种子、叶内Aalen–Johansen与森林平均不变。原因权重不是样本类别权重，也不是Focal Loss。预测仍是两个相容的12个月CIF，完整解释仍在概率尺度上。

沿用 `post_st_affine_calibration_v1` 的**森林拟合**分割及测试记录：2023测试对应2012—2020入组、信息严格截止2021年初；2024测试对应2012—2021入组、信息严格截止2022年初。训练各232／277家，仅2／4个退市事件。之前预留的校准样本不并入训练，也不重新拟合校准器。这是隔离权重影响的比较，不是截至2023年6月底的扩样实验，也不是季度滚动实验。

测试为2023年36家（4退市、0恢复），2024年45家（15退市、4恢复）；每家公司一行，记录等权与公司等权相同。评价各年和合并预测；合并AP重新计算。主结果为三个固定种子预测的等权平均，再计算指标。保留全部种子，Top20%为主、Top25%为补充。年度排名为事后预算评价，不称为在线预警策略。

这是已接触测试结局后的探索性比较。原始财务披露版本／行业历史版本尚未完全PIT认证，同日财报仍按旧公告日起点排除。未引入政策特征、动态窗口或2025测试。统计不确定性不因重复种子而消失；原因权重本身是已有方法，不能单独声称算法创新。

固定配置见 [frozen_protocol.json](frozen_protocol.json)。18个正式拟合、2个4树试运行，CPU任务单线程，地址空间上限4GiB；已有两节点Slurm物理核池分配独立任务，NFS输入与自动回传。模型、真实阈值、预测、函数数值和企业信息仅在私有服务器目录。

执行入口（`RUN`是新建私有目录，包含audit/input/evaluation_input/reports/results；不得覆盖已有运行）：

```bash
python -m experiments.post_st_cause_weights_v1.stages --stage prepare --run "$RUN"
# 完整测试通过并提交冻结代码后：
python -m experiments.post_st_cause_weights_v1.stages --stage freeze --run "$RUN" --repo "$REPO"
hetero-submit "$RUN/pilot.jsonl" physical both
python -m experiments.post_st_cause_weights_v1.stages --stage pilot --run "$RUN" --job "$PILOT_JOB"
hetero-submit "$RUN/formal.jsonl" physical both
python -m experiments.post_st_cause_weights_v1.stages --stage formal --run "$RUN" --job "$FORMAL_JOB"
python -m experiments.post_st_cause_weights_v1.analyze --run "$RUN"
```

必须等待实际作业完成并通过发布门禁，不能以返回作业号代替成功。`pilot.py`和`formal.py`有hetero-run兼容sidecar；无需修改原始数据或旧实验。评价器核对所有输入／输出哈希，逐数组及逐树验证六个等权q2旧模型完全复现，并独立计算AUC/AP/Brier。

拟合完成后的独立节点／96树集成审计入口为 `python -m experiments.post_st_cause_weights_v1.audit --run "$RUN"`，它只读取保存模型，绝不重训。

已完成结果：[分年与合并结果](RESULTS_ZH.md)、[标准RSF参照及恢复结局](PREDICTION_COMPARISON_ZH.md)、[完整解释比较](INTERPRETABILITY_COMPARISON_ZH.md)、[数据对齐](DATA_ALIGNMENT_AUDIT_ZH.md)、[训练诊断](TRAINING_DIAGNOSTICS_ZH.md)、[研究判断](METHOD_INNOVATION_DECISION_ZH.md)、[执行回执](execution_receipt.json)。
