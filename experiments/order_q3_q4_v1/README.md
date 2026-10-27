# 制造业q3/q4有限匹配实验

已完成：q3、q4各3个固定种子，共6个新模型；另2个试运行。使用原制造业56,458条训练记录（177个独立目标事件），2023/2024分别评价，不改变数据或训练时钟。两节点48个四树训练片段全部成功，6个模型完整解释核验全部通过。51项测试通过，包含两项独立R检查。

**结论：精确高阶解释实现成功，但预测收益不足以证明巨大复杂度代价值得。** q4合并AUC仅从q2的0.8422升至0.8441、AP从0.0838升至0.0949，Top20从61.33升至62/79；空间格约增至63.9倍，高阶选择稳定性较弱。q3也未取得一致改善。本轮PROVISIONAL、INCONCLUSIVE、后暴露探索性比较，不能作为独立盲测证据。无可辨识性惩罚。

|内容|入口|
|---|---|
|冻结设计及用户授权边界|[PROTOCOL_ZH.md](PROTOCOL_ZH.md)、[frozen_protocol.json](frozen_protocol.json)|
|年度/合并、记录/公司等权、Top20/25和校准|[RESULTS_2023_2024_ZH.md](RESULTS_2023_2024_ZH.md)|
|完整解释、各阶函数、稳定性与截断误差|[INTERPRETABILITY_COMPARISON_ZH.md](INTERPRETABILITY_COMPARISON_ZH.md)|
|资源、求和/重载检查、叶事件支持|[TRAINING_DIAGNOSTICS_ZH.md](TRAINING_DIAGNOSTICS_ZH.md)|
|输入复用和数据时钟限制|[DATA_ALIGNMENT_AUDIT_ZH.md](DATA_ALIGNMENT_AUDIT_ZH.md)、[reuse_receipt.json](reuse_receipt.json)|
|研究判断|[METHOD_INNOVATION_DECISION_ZH.md](METHOD_INNOVATION_DECISION_ZH.md)|
|执行回执、环境、交付验证|[execution_receipt.json](execution_receipt.json)、[environment.json](environment.json)、[delivery_validation.json](delivery_validation.json)|
|原始指标及配对差值|[metrics.csv](metrics.csv)、[paired_contrasts.csv](paired_contrasts.csv)、[warning_metrics.csv](warning_metrics.csv)、[calibration.csv](calibration.csv)|
|复杂度/稳定性汇总|[complexity_comparison.csv](complexity_comparison.csv)、[seed_stability.csv](seed_stability.csv)|

q2为上一轮制造业固定模型，唯一匹配配置差异是path_budget。unrestricted为同后端制造业不限阶数参考，其他候选/切点设置不同；M2为原全行业训练的冻结参考。本轮均未重训。72行参考指标与上一轮完全一致。

实现：`crforest/higher_order.py`新增`HigherOrderCIFExpansion.from_forest(forest, training_reference)`，支持全部1—4阶项，`component(X, sorted_feature_tuple, times)`读取某项，`predict`返回完整CIF，`predict_by_order`返回基线及各阶合计。原`CIFExpansion`继续只支持二阶。森林配置允许`path_budget=3/4`，硬全森林分区预算仍仅支持q1/q2；本轮没有硬预算。共享切点字典、bootstrap和分裂目标不变。

复现顺序：在新的唯一私有目录运行prepare.py（--help查看参数），提交并冻结清洁代码；stages.py --stage freeze生成pilot/training任务；经已有hetero-submit或hetero-run运行，逐阶段gate通过后进入下一步；training gate生成assembly任务。最后运行evaluate.py、report.py和validate.py。代码和配置快照已保存在原私有运行目录的input/code，训练提交见回执，切勿覆盖已完成目录。新增报告脚本不改变冻结训练代码。

私有运行目录：`/home/oliver/MainDiskData/xai_rsf_order_q3_q4_v1_20261027_pvsZaIkY/`。Slurm作业2526/2527/2528，分别试运行/训练/组装。企业记录、预测、真实阈值、模型参数与完整个案留在私有目录；Git仅包含代码、汇总和哈希。
