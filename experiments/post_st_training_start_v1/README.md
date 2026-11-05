# C039 训练历史起始年消融（探索性）

保持 C039 的 74 项输入、17/503/22 各32树、q2、深度4、退市支持保护、共享切点、均衡25和直接父节点AJ收缩10。只把最早入组年从2012逐年移到2020；三个年度仍分别在当年1月1日截止，单公司训练随访上限仍是24个日历月。主模型指针不变。

独立私有目录见 `run_location.json`。`input/` 存放冻结快照及代码；`pilot/` 和 `published/` 保留全部模型、函数、逐公司预测与日志；公开目录仅有汇总。不允许向旧运行目录写入。

复现入口（在新工作树和新的唯一私有目录中，更新本目录run_location）：

```bash
python -m experiments.post_st_training_start_v1.prepare
# 完整pytest（需设置仓库已有R环境变量），提交冻结协议与代码后：
python -m experiments.post_st_training_start_v1.schedule
DATA_DIR=<private>/input OUTPUT_DIR=<private>/pilot PYTHONPATH=<private>/input/code \
  python <private>/input/code/experiments/post_st_training_start_v1/launch.py 2024 2012
/home/oliver/.local/bin/hetero-run <private>/tasks.jsonl
python -m experiments.post_st_training_start_v1.analyze
python -m experiments.post_st_training_start_v1.report
```

试运行是81次正式拟合中的3次全预算复现，不另加种子/树。只有试运行通过才提交其余26个独立年度窗口任务。两机各自使用已验证Python3.13环境，单任务单线程。指标使用三个种子的逐公司预测均值；训练起始年不是随访长度。2025年24月性能留空。

完成后检查入口：python -m experiments.post_st_training_start_v1.verify_results。

主要阅读：RESULTS_ZH.md（完整解读）、AUC_OVERVIEW.csv（年度AUC对照）、METRICS_ENSEMBLE.csv（全部年度/合并指标）、PAIRED_DIFFERENCES_95.csv（相对2012的条件区间）、NUMERICAL_COMPLEXITY.csv及FUNCTION_STABILITY.csv。两张图同时提供PNG与PDF。
