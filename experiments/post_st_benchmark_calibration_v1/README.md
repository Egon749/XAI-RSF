# 论文第 4.3、4.4 节：竞争风险比较和风险水平

[中文结果](RESULTS_ZH.md) · [方法取舍与论文文字](METHOD_DECISION_ZH.md) · [冻结协议](FROZEN_PROTOCOL.json) · [执行回执](execution_receipt.json)

以 16ba26c66faea63852cd068c1e6f05513c41525e 为代码与已保存数据基线。主模型是逐公司平均三个 32 树成员的双原因 CIF。17/503/22 为 C039；原 42/101/202 为敏感性对照。仅在本研究分支更新根目录 CURRENT_POST_ST_MODEL.json。PREVIOUS_POST_ST_MODEL.json 保留旧设定与工件哈希，完整旧指针可由所列不可变 Git 对象读取。

## 论文表和图

- METRICS_ALL：两原因、两个时限、年度和合并指标，含逐种子、两套 96 树组合、120 组各自结果与按指标计算的中位数。cause 1 为先最终退市，cause 2 为先恢复；ipcw_brier 越低越好。within_year_auc 只计同入组年内配对，auc 为普通合并排序。
- SEED_DISTRIBUTION：120 个固定组合的中位数和 5%–95% 分位；重叠组合间的波动不能作为置信区间。PAIRED_DIFFERENCES：按“所列参照减对比模型”计算。对中位数参照，每次重抽都对各组合的指标取中位数。所有区间条件于已有固定预测。
- CALIBRATION：C039 逐年及合并 O/E、预测减观察、Brier 和配对样本重抽区间。单位为概率，转成百分点需乘 100。无事件时保留合法的 Brier、观察率；相应 AUC/AP 为 NA。2025 年 24 个月只导出平均预测，不评价性能。
- RISK_INDEX、INDEX_DISTRIBUTION、HIGH_VS_REMAINDER、TIER_DIFFERENCES：同年训练 OOB 阈值下的三档与辅助高／非高比较。小格与可反推的互补格一起抑制。INDEX_CHECKS 和 INDEX_METRIC_REPLAY 核对原始分解、得分分解与排序。
- COMPLEXITY、COMPUTE_COST、TUNING、FORWARD_FOLDS：函数数量、模型与作业耗时、超参数搜索明细和训练期时间切分。标准 RSF 给叶子总数，回归给两原因的非零线性项数，两者均不等于二阶 CIF 函数数。历史训练耗时、单任务合计耗时、两台机器的墙钟时间口径各异。
- FG_JOINT_PROBABILITY：分别拟合 Fine–Gray 的两个边际 CIF 之和；保留实际违规计数。SOURCE_CHECKS、PUBLICATION_CHECKS、METRIC_VALIDATION：输入、输出哈希、原 C039 逐公司等价性及独立统计校验。figures 提供 PNG/PDF。

## 复现

在具备原服务器私有归档和两个节点研究环境的此分支工作区运行。用 BENCH_RUN 指向共享存储中**新建、唯一**且对作业账户可访问的私有目录。训练作业读入的正式评价文件只含 X 和样本对齐键；正式评价标签在作业外读取。按仓库现有约定配置 CRFOREST_RSCRIPT、EW_RLIB，使用单线程 BLAS。运行顺序如下。

```bash
m=experiments.post_st_benchmark_calibration_v1
python -m "$m.stage"
python -m pytest -q
python -m "$m.schedule"
python -m "$m.evaluate"
python -m "$m.index_audit"
python -m "$m.tier_checks"
python -m "$m.complexity"
python -m "$m.release_checks"
python -m "$m.figures"
```

schedule 归档已提交的代码并生成 33 个独立任务，等待两节点计算及发布。release_checks 重新核对每个发布文件内容哈希。index_audit 只读取保存的森林并创建新的指数函数文件，末步按本轮授权写入已经核验的主模型指针。个体数据、模型参数、原始预测、OOB 阈值、真实切点与原始日志仅在服务器私有目录。pytest 日志的环境路径做了脱敏。未改变任何既有模型参数。
