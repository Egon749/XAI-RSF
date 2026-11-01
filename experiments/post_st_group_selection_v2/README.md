# 有限分组筛选：真实分支与完整过程重抽样

本目录独立保存修正后的探索性筛选。上一轮真实执行了回退森林，但未实现通过门槛后的监督筛选，不将其重述为已完成的筛选。这里缺少独立确认期只限制结论；发现阶段事件支持不足仍回退83项。

运行入口依次为 `python -m experiments.post_st_group_selection_v2.run prepare`、完整pytest、提交冻结配置和代码、`...run freeze`、`...run pilot`、核验试运行、`...run batch`、`...deliver`、`...validate`。prepare只允许新目录。重新运行须先指定唯一私有目录，不能覆盖已经存在的协议或输出。首次实际运行目录见run_location.json。

每个通过发现门槛的任务重新拟合有限组替换候选及83基准，记录所有候选性能与交互拒绝原因，再重新拟合最后保留的模型；每个未通过门槛的任务保留83项并明确记录停止原因。50次公司重抽样重新计算分组、筛选与训练，不复用点估计的筛选名单。分组替换并非递归或全局寻优；无独立确认期不构成效果已确认。候选与保护规则详见frozen_protocol.json。

使用已有森林训练、原始直接父节点AJ收缩、全时间函数编译及概率检查接口。没有修改森林核心、当前模型指针、输入版本或历史结果。公司信息、完整函数表、阈值和预测留在私有目录；公开代码、汇总指标、实际筛选次数、完整拒绝证据和哈希。

本次初始试运行发生诊断JSON序列化错误，已在所有正式批次前修正。实际执行链额外包含freeze_fixed和pilot_recovered，按recovery_plan对已完成森林逐文件核验后恢复，不重新拟合。一般重新执行已修复代码时无需故意复现故障：pilot通过后直接batch即可。旧失败报告及两个代码快照在audit中保留。

交付的主要入口为RESULTS_ZH.md、POINT_CANDIDATES.csv、MODEL_COMPARISON.csv、GROUP_DISCOVERY_RESULTS.csv和execution_receipt.json。图形为selection_process.png及annual_prediction_comparison.png。完整代码和汇总工件哈希见deliverable_hashes.json；企业级数据与完整模型仍在run_location指定的私有目录。
