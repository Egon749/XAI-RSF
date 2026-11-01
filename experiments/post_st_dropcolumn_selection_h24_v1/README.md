# 24个月为主目标的历史Drop-column筛选

用户在12个月支持门槛说明后明确要求追加24个月实验。本目录单独冻结24个月AP/Top20主要选择规则，保留上一轮以12个月为主的结果和当前模型。

原森林训练上限本来就是24个月，已有全83项逐一删除及分组森林可以按哈希复用。改变的是历史验证选择目标，不是用不同标签重命名已有拟合。2023截止的24个月支持不足；2024/2025达到执行门槛，但仍是小样本探索。

所有候选选择仅用对应截止之前的滚动历史验证，未用已暴露外层表现放宽约束。本轮已完成36546次新增森林拟合、每截止50次公司重抽样流程；2023/2024/2025分别保留83/82/81项。9项文件回传超时已在不重训的情况下恢复，拟合失败0。

结果并非一致改善：2024的24个月AP略升而Top20少捕获一例，合并24个月AP几乎不变；2025完整函数格数增加，主要删除决定缺少重复支持。当前83项参考模型保持不变。

- [完整中文结果与六项研究判断](RESULTS_ZH.md)
- [数据、信息时点与支持审计](DATA_AND_PROTOCOL_AUDIT_ZH.md)
- [逐年、逐种子及合并预测结果](MODEL_COMPARISON.csv)
- [特征选择稳定性](FEATURE_SELECTION_STABILITY.csv)
- [完整函数复杂度](EXPLANATION_COMPLEXITY.csv)
- [执行数量、数值验证和哈希回执](execution_receipt.json)
- [逐模型文件哈希](MODEL_HASHES.csv.gz)、[逐模型数值核验](NUMERICAL_VALIDATION.csv.gz)
- 沿用核验过的[oliver环境](../post_st_dropcolumn_selection_v1/audit/oliver-environment.txt)及[lemon环境](../post_st_dropcolumn_selection_v1/audit/lemon-environment.txt)

原12个月优先实验保留在[前轮实验目录](../post_st_dropcolumn_selection_v1/)。两轮合计39636次新增森林拟合；本轮初始全83项/相关组候选按哈希复用前轮真实拟合，没有重复记为新增训练。2025外层24个月随访不足，仍保留NA。

完整复现入口（要求私有父运行工件可访问、代码已提交且工作区干净；将执行真实训练）：

```bash
/home/oliver/.venvs/xai-baselines-20261027/bin/python -m experiments.post_st_dropcolumn_selection_h24_v1.run_pipeline --run /home/oliver/MainDiskData/xai_rsf_dropcolumn_h24_REPRO_UNIQUE
```

该入口创建新目录，执行完整测试、冻结、试运行、依赖明确的递归批次、50次公司重抽样、外层最终模型、统一评价和数值验证。不会覆盖既有公开报告。失败保留原记录并停止，不自动重跑拟合；仅发布失败应使用原节点的发布回执恢复。

每次重抽样在支持时重新执行全量候选及相同递归流程，计算量显著大于只对固定模型重抽样预测。所有独立计算使用现有两节点池，任务内部单线程。为减少存储，森林JSON无损gzip压缩；解压到新临时文件后可用原`CompetingRiskForest.load`读取，完整函数NPZ可直接由`CIFExpansion.load`读取。
