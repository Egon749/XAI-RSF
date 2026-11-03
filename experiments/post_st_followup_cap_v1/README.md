# 训练随访上限：12、24、36个月与不限

固定当前74输入＋退市支持保护＋q2＋深度4＋共享切点＋均衡25变量池＋直接父节点AJ10。仅改变训练标签的行政随访上限，主要评价2023/2024/2025入组企业各自未来12个日历月的先退市与先恢复。三种子平均后计算主指标。

“不限”仅指训练公司的随访不设月数上限；仍严格限制在对应2023/2024/2025年1月1日之前。不能使用训练截止日之后的结局，也不能引入测试公司。评价数据保持冻结版本。

24个月组逐字段复现现有训练快照并复用9个森林。新增3组×3年度×3种子＝27次正式拟合；另有3次4树数值试运行，不用于配置选择。每个raw森林生成自身直接父节点收缩版本，保留全部版本和失败记录。新工件由run_location.json定位，旧模型和CURRENT_POST_ST_MODEL.json不变。

复现入口（新运行必须先建立新的唯一私有目录并更新run_location.json，不能重跑到既有目录）：

```bash
python -m experiments.post_st_followup_cap_v1.prepare
# 完整pytest包含R，通过后提交协议与代码
python -m experiments.post_st_followup_cap_v1.run freeze
python -m experiments.post_st_followup_cap_v1.run pilot
python -m experiments.post_st_followup_cap_v1.run formal
python -m experiments.post_st_followup_cap_v1.run freeze_analysis
python -m experiments.post_st_followup_cap_v1.run ensemble
python -m experiments.post_st_followup_cap_v1.analyze metrics
python -m experiments.post_st_followup_cap_v1.analyze functions
python -m experiments.post_st_followup_cap_v1.audit
# 完整pytest含独立R，日志保存到私有运行audit/final_pytest.log
# 提交分析与报告代码，再生成公开汇总
python -m experiments.post_st_followup_cap_v1.report
```

计算使用已有两节点Slurm/NFS，独立单线程任务，沿用管理器的P/E核和亲和性规则。每个进程地址空间上限4GiB；历史同配置单种子测得约280MiB峰值RSS，不将地址空间上限当作实际占用。代码、训练输入、随机种子、任务输出和发布回执均保留哈希。

12个月训练组不报告24个月预测性能：库的末端平台值不当作有效的两年外推。2025年所有组24个月性能均NA。24/36/不限组的2023/2024年24个月结果仅作次要对照。本轮是已暴露年份的探索性研究，不自动替换主模型。

交付导航：RESULTS_ZH.md为中文完整结论；METRICS_ENSEMBLE.csv为三种子平均后的主指标，METRICS_ALL.csv保留每个种子及raw版本；PAIRED_BOOTSTRAP_95.csv为固定预测条件下的配对差值区间。COHORT_AUDIT.csv和INDEPENDENT_DATA_CHECKS.csv核对训练截止和事件数，PAIRING_AUDIT.csv核对公司抽样、变量池和切点。NUMERICAL_COMPLEXITY.csv、FUNCTION_STABILITY.csv、LEAF_SUPPORT.csv、COMPUTE_COST.csv保存解释与计算诊断。MODEL_HASHES.csv只发布模型/函数/预测哈希，实际公司数据和模型文件在run_location.json所指的私有目录。

新增正式拟合27次、复用9次，另3次4树试运行；所有工件使用独立运行目录。代码入口依赖仓库既有实验与crforest实现，不能只复制本目录脱离仓库运行。各阶段使用不可变代码快照，训练工件的准确代码提交以execution_receipt.json为准，不用后续文档提交冒充训练版本。
