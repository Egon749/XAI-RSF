# 共享切点与全森林分区预算有限实验

本轮检验共享切点、1000格全森林预算两个开关。固定V19/V20、种子42/101/202、每次32棵树，共30个正式模型。数据继续为PROVISIONAL。

- [冻结协议](PROTOCOL_ZH.md)
- [研究结论](METHOD_INNOVATION_DECISION_ZH.md)
- [预测结果](V19_V20_RESULTS_ZH.md)
- [完整解释对照](INTERPRETABILITY_COMPARISON_ZH.md)
- [训练与失败恢复](TRAINING_DIAGNOSTICS_ZH.md)
- [执行回执](execution_receipt.json)

私有运行目录：`/home/oliver/MainDiskData/xai_rsf_shared_partition_v1_20261027_023501/`。真实阈值、模型参数、企业预测和完整匿名案例均留在服务器，未上传GitHub。

## 可重复入口

先阅读PROTOCOL_ZH.md，每次prepare使用新的唯一目录。复用的旧冻结运行由`--original`指定；prepare只复制并核验输入，不重建研究样本。

```bash
python -m experiments.shared_partition_v1.prepare --run RUN --repo REPO --original ORIGINAL_RUN
/home/oliver/.local/share/heteropool-20261005/submit RUN/pilot.jsonl physical both
# 验证5项pilot计算、回传、概率、重载、重构及资源门禁后：
/home/oliver/.local/share/heteropool-20261005/submit RUN/training.jsonl physical both
python -m experiments.shared_partition_v1.collect --run RUN --job TRAIN_JOB
/home/oliver/.local/share/heteropool-20261005/submit RUN/assembly.jsonl physical both
python -m experiments.shared_partition_v1.evaluate --run RUN
python -m experiments.shared_partition_v1.report --run RUN --repo REPO
```

本轮原训练快照是`25cce212f0bc1adf00ffaa3f4a4cf01ce51d5eeb`。日志序列化失败的恢复由`recover --run RUN --repo REPO --job TRAIN_JOB`生成独立清单；collect同时接收`--recovery-job`。该入口只允许已核实的缺失单独分裂日志保存错误，不能通用重试失败任务或按指标挑选重试。恢复快照及训练模块哈希一致证明见recovery_plan.json。

## 模型接口

```python
cfg = ForestConfig(path_budget=2, cut_strategy='shared_quantile',
                   n_thresholds=5, global_cell_budget=1000)
model = CompetingRiskForest(cfg).fit(X, time, event, groups)
```

默认`local_random`和无全局预算保持旧行为。全局预算要求路径至多二阶、固定树序，禁止一个模型内部的树级并行；独立模型可跨节点调度。没有预算的森林用固定树编号分片，汇总与整模型逐字节一致。

仍保留公司bootstrap、生存分裂、AJ叶估计、固定等权平均。全局预算耦合树的结构选择，因此是受结构约束的RSF原型，不声称与GC/randomForestSRC后端等价。共享切点只用训练分位值；预算计数包含缺失状态，计完整主效应/二元空间表，不是叶数或时间/原因标量总数。

没有引入可辨识性惩罚，未自动追加训练或超参数搜索。所有种子、失败尝试和恢复均保留。
