# 随机20种子XGBoost固定参数重训

报告见[RESULTS_ZH.md](RESULTS_ZH.md)，逐种子全部结果见[NEW_SEED_RESULTS.csv](NEW_SEED_RESULTS.csv)，三年度AUC/AP/Brier/捕获宽表以`SEED_`开头。`SEED_DISTRIBUTION.csv`是随机模型分布；`ENSEMBLE_COMPARISON.csv`是平均预测后重新计算的指标，两者不能互换。

随机名单、全部参数继承与预算在`FROZEN_PROTOCOL.json`预先冻结。120次真实拟合、零重新选参；不改C061。2025年24月性能NA。全部结果探索性，不选优或升级模型。

授权环境可按以下顺序复现；先为`run_location.json`指定另一个**新的唯一私有目录**，不得覆盖本次运行或向pool已有task目录写入。使用原实验同一研究Python、依赖目录和R环境。

```bash
python -m experiments.post_st_xgboost_20seeds_v1.stage
python -m pytest -q
# 先提交协议与代码，再调度
python -m experiments.post_st_xgboost_20seeds_v1.schedule
python -m experiments.post_st_xgboost_20seeds_v1.evaluate
python -m experiments.post_st_xgboost_20seeds_v1.report
python -m experiments.post_st_xgboost_20seeds_v1.release
python -m pytest -q
```

私有路径见`run_location.json`；输入通过NFS供两机只读访问，24个独立批次每批5种子，经池的哈希发布验证返回。`PUBLICATION_CHECKS.csv`记录真实节点和作业号；`TRAINING_LOG.csv`记录每个模型耗时。`actual_task_seconds`是所在批次共享值，不能在模型行中重复加总；`PUBLICATION_CHECKS.csv`为唯一任务记录。

所有基线、参数继承和评价函数直接复用此前实验。公司级输入/预测、模型、真实切点留服务器。`PAIRED_DIFFERENCES.csv`为固定预测条件下的原1000次公司配对区间，不是随机种子分位区间。冻结协议后测试与环境检查中的修复记录也保留。
