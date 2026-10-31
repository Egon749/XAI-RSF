# 复现与工件

服务器本次独立运行：`/home/oliver/MainDiskData/xai_rsf_level_coverage_20261031_ckCB3u1l`。所有旧模型、当前引用和旧结果保持原样。公开results仅汇总、哈希、冻结配置、执行日志及匿名总体图；公司资料、逐公司预测、真实阈值、森林/校准参数和完整企业函数导出保留私有目录。

拟合代码冻结提交：`c82f303f164bd9013be0bc720ceb4494fad6ce01`；首次分析提交：`3cda259`；并行分析快照：`df9d839f6110a4968739c61d4dd6adbd3ec975ed`。后续改动只涉及汇总、真实导出网格计数、图表和报告，没有修改已拟合森林或校准器。

## 新运行的入口

先创建唯一私有目录，不覆盖本次目录。安装环境见results/environment.json；两机均使用已核验Python3.13环境。下列RUN需替换为新的唯一绝对路径：

```bash
python -m experiments.post_st_level_coverage_v1.prepare --stage prepare --run RUN
python -m experiments.post_st_level_coverage_v1.prepare --stage freeze --run RUN
/home/oliver/.local/bin/hetero-run RUN/pilot.py
```

以`experiments.post_st_ablation_v1.stages.gate`核验pilot任务、签名、SHA256与发布成功后，再运行`hetero-run RUN/formal.py`并核验formal gate。gate需要实际job编号和工具返回的结果JSON路径；不可仅凭提交成功继续。拟合快照与协议必须在第一项新结果前冻结。原prepare输入依赖18个历史企业/滚动森林以及450份旧模拟工件，缺失时不得静默重训或替代。

```bash
python -m experiments.post_st_level_coverage_v1.analysis_pool --stage prepare --run RUN
/home/oliver/.local/bin/hetero-run RUN/analysis.py
```

核验analysis gate后：

```bash
python -m experiments.post_st_level_coverage_v1.analysis_pool --stage collect --run RUN
python -m experiments.post_st_level_coverage_v1.validate --run RUN
python -m experiments.post_st_level_coverage_v1.supplements --run RUN
python -m experiments.post_st_level_coverage_v1.plots --run RUN
python -m experiments.post_st_level_coverage_v1.report --run RUN
```

分析读取已固定森林，不重新拟合。pool输入与结果分目录，DATA_DIR/OUTPUT_DIR由调度器提供；任务独立输出，NFS发布核对SHA256。只在未运行的唯一目录执行冻结准备，不把这段复现命令误用于再次启动已完成实验。报告脚本中的结论针对本次冻结结果，其他研究设置须重新审阅文字。

## 表格索引

- enterprise_metrics / warnings / calibration_bins：年度、合并、各种子及集成，12/24个月两原因，raw诊断完整保留。
- enterprise_paired_intervals：1000次共同重采样的24个月主要性能差；正AUC/AP/捕获有利，负Brier有利；calibration_gap=预测减观察，正差表示概率上移而非必然更好。
- calibration_support / historical_diagnostics / domain_validation：数据支持、训练前校准效果、优化与域证书，不公开拟合参数。
- enterprise_function_validation / export_geometry / function_stability / leaf_support / search_cost：完整解释、实际网格、稳定性、真实事件支持与计算量。
- simulation_main / metric_summary / paired_changes / signal_summary / tree_weighted_identity：模拟主表、全部主次指标、配对差、真实信号恢复与逐树机制；csv.gz保存逐重复及各种子结果。
- frozen_input_hashes / data_array_hashes / artifact_hashes / reuse：来源、输入与模型哈希。simulation_threshold_sensitivity完整保存0.0025/0.005/0.01三个门槛。
- execution_receipt.json：拟合数量、复用、失败、数值状态与提交来源。Git最终提交与远端同步在交付时另外核验。
