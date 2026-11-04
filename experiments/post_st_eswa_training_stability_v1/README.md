# ESWA 分类器对训练公司扰动的稳定性

[中文结果](RESULTS_ZH.md) · [冻结协议](FROZEN_PROTOCOL.json) · [逐次内部、测试与差距](ALL_REPLICATES.csv) · [分布及原始位置](DISTRIBUTIONS.csv) · [三种方法并列](SIDE_BY_SIDE.csv) · [作业执行回执](execution_receipt.json)

基于提交 05f4ae73a166a95d7f16f2a4bf966c9dae8aa88b，协议在 9843452 提交，训练代码为 02dae8e。两种方法均仅使用原 IPCW 分支。训练扰动包含原始公司的有放回重抽，以及完整方法对这批训练数据的响应（重新预处理、调参、特征选择、集成和拟合）。每个年度每个重复的抽样与两种方法和两个时限配对。每年都包含重复 1–50，无替补。

**内部验证。** 每次均用没有被抽入该次训练样本的原历史公司评价完整选定预测器，即袋外（OOB）验证；计算指标所需的删失生存函数只在有重复数的袋内样本上估计。ALL_REPLICATES 另外提供所选 PROBoost 配置的历史调参折平均 AP、Stacking 部署计划可计算的 GA 验证 F1 和基学习器平均折 F1。这些分数的目标和尺度不同，不能互相作差。正式 gap 列为同一指标的 OOB 减时间外测试。已知退市原始捕获数取整数，OOB 覆盖率、命中率、Lift 为训练-G 加权估计（窗口前删失公司不标成正常）；名额仍按该年度全部评分公司确定。OOB 与外层的公司数不同，捕获家数分列。相同公司可能出现在不同截止点的历史样本内；合并内部统计的单位是公司–截止点。2025 H24 仅保留外层评分离散度和内部验证。

**文件。** TRAINING_STATUS 含全部 600 个请求的状态、支持、抽样/发布摘要及实际运行位置。ALL_REPLICATES 含 600 个年度和 200 个合并记录。DISTRIBUTIONS 每个指标同步给出有效数、NA 数、分位与极值。original_midpercentile 对等值按半数计算。没有与单次原始拟合直接对应的 OOB 验证集合，因此其原始 OOB 位置留空。XAI_FIXED_COMBINATIONS 仅含固定训练集下各组合及参照的汇总；SIDE_BY_SIDE 区分两种扰动来源。完整企业预测、抽样索引、真实模型、详细嵌套折与失败堆栈仅存在私有输出。

**复现。** STABILITY_RUN 指向一个新的、独占的共享私有目录，ESWA_RUN 指向上一轮已核验的私有归档（含 deps）。复用原研究 Python/R 环境。初始 XAI-RSF 模型及本轮基线文件无需修改。以仓库根目录为工作目录运行：

```bash
export PYTHONPATH="$ESWA_RUN/deps:$PWD"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
m=experiments.post_st_eswa_training_stability_v1
python -m pytest -q
python -m "$m.run"
python -m "$m.evaluate"
python -m "$m.verify"
python -m "$m.figures"
python -m "$m.report"
```

run 将已提交的代码、只读训练数据和不含标签的外层 X 固定成快照，交给 Slurm 运行并等待回传。evaluate 检查完整 600 个身份和每个文件的发布哈希，然后再读取冻结外层标签。不以新种子替换不能拟合的重复。合并结果要求所有应纳入的年度都拥有预测；若有年度不可拟合，该重复的合并指标标作 NA。

**复核。** [独立核验](VERIFICATION.json) 使用事件–对照逐对比较、按阈值计算 AP 以及独立排序核对年度与合并指标。另检查保存的 Chen 各层验证折。完整测试（含独立 R 检验）见 [日志](pytest.log)。图示给出[12 个月](figures/distributions_12m.png)和[24 个月](figures/distributions_24m.png)的中位数、分位范围、极值和原始结果。
