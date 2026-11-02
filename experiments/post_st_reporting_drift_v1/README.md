# 退市报告口径与水平漂移（只读）

先读[结论](RESULTS_ZH.md)、[论文修订建议](PAPER_PATCH_ZH.md)及[全部臂汇总](ALL_ARMS_ZH.md)。分支research/xai-rsf-reporting-drift-v1源自6ea32103d2b6d953ae25ab0b9fa4301c6cfbb434，不合并、不替换当前模型。

私有目录见[run_location.json](run_location.json)；现有cause-support目录只读。新私有目录为`/home/oliver/MainDiskData/xai_rsf_leaf_shallow_20261028_ep_9hazj/reporting_drift_20261102_ynl4yy9h`，完整公司风险历程、逐公司保存预测的读取校验、完整入组年×日历年表、完整83列漂移和重抽样数组都保留该目录。公开目录不含任何新增公司行、标识、切点、模型参数或新预测。此前rankings授权未扩展。

复现使用同仓库代码和原输入，先创建一个新的空目录（不要使用本次run路径），然后：

```bash
python -m experiments.post_st_reporting_drift_v1.analyze metrics --run NEW_PRIVATE_RUN --source CAUSE_SUPPORT_PRIVATE_RUN
python -m experiments.post_st_reporting_drift_v1.analyze drift --run NEW_PRIVATE_RUN --source CAUSE_SUPPORT_PRIVATE_RUN
```

每次调用自动新建带微秒时间戳的attempt目录；失败日志保留。用明确成功目录交付：

```bash
python -m experiments.post_st_reporting_drift_v1.deliver --run NEW_PRIVATE_RUN --metrics METRICS_ATTEMPT --drift DRIFT_ATTEMPT
```

交付脚本拒绝覆盖已有公开CSV/图。复算时留结果于新私有目录；需新一版公开交付时使用新的实验版本目录，不删除历史文件。完整测试命令与日志见[VALIDATION_ZH.md](VALIDATION_ZH.md)。

主文件：STRATIFIED_METRICS.csv、STRATIFIED_BOOTSTRAP_95.csv、OBSERVED_EXPECTED.csv、TOP20_LIFT.csv、NULL_AJ_BASELINE.csv。时间诊断：CALENDAR_INCIDENCE.csv、COHORT_CALENDAR_CROSS.csv、COHORT_AJ.csv、LEVEL_TIME_SERIES.csv、COVARIATE_SHIFT_SUMMARY.csv、COVARIATE_SHIFT_TOP.csv、DELISTING_REASON_SUMMARY.csv、POLICY_TIMELINE.csv、POLICY_EVENT_ALIGNMENT.csv。

[PNG](RISK_LEVEL_TIME_SERIES.png) · [PDF](RISK_LEVEL_TIME_SERIES.pdf)。图来自汇总CSV，采用可复现matplotlib静态输出，符合本轮指定科研图格式。

所有CSV的NA/空值均为未定义、未支持或隐私抑制，由supported/reason/status字段区分，不填0。年内等权平均仅对有配对的年度计算；所有区间均为固定模型条件下描述性95%百分位区间。原始数据版本仍PROVISIONAL，2023—2025均探索性。
