# 数据对齐：PROVISIONAL，2023/2024后暴露探索性

本轮复用原制造业冻结输入，没有重建样本、改写时间或把未知结局设为正常。原始冻结数据、M2与q2参考预测、原统一评价代码逐文件SHA256核验；详见input_alignment.json与frozen_input_code_hashes.json。

原始输入位置：`/home/oliver/MainDiskData/xai_rsf_interpretable_baselines_v1_20261027_hw9gdrol/input`。本次唯一私有目录：`/home/oliver/MainDiskData/xai_rsf_transparent_cif_v2_20261028_p3k845e7`。旧工作区、旧模型、旧结果未改写。

训练原点2012—2020，信息截止2022-01-01。sample_id、group、origin、事件/随访及outcome_available_at与冻结NPZ/Parquet逐行对齐；期限是每条原点加12个日历月的实际365/366日。首次目标ST、首次竞争ST、删失与source eligibility沿用原合同。未知但名义达到12个月的记录仍由原endpoint_guard排除其结局权重，不能填正常。

| role | rows | groups | target_rows | target_events | competing_rows | competing_events | source_unknown_endpoint_rows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all | 56458 | 2577 | 654 | 177 | 117 | 31 | 17096 |
| structure | 45343 | 2061 | 532 | 144 | 98 | 26 | 13671 |
| internal_weight_selection | 11115 | 516 | 122 | 33 | 19 | 5 | 3425 |

公司组角色按固定哈希分割，与结局无关，组重叠0；三种子共用同一角色划分，不是三个独立开发集。内部留出只有33个独立目标事件与5个独立竞争事件（122/19条相关季度记录），必须区分记录与独立支持。

| scope | rows | groups | target_records | target_events | competing_records | competing_events | unknown_records |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2023 | 11686 | 3050 | 94 | 44 | 30 | 14 | 200 |
| 2024 | 12113 | 3091 | 129 | 60 | 69 | 33 | 513 |
| pooled | 23799 | 3138 | 223 | 79 | 99 | 36 | 713 |

年度目标事件数不能相加当合并事件数：同一首次ST可能被2023和2024的多个12月窗口同时覆盖，合并期重新按event_id去重为79。

75列为原72内容变量（NaN保留）加3质量字段：financial_period_age、financial_disclosure_age、market_missing_coverage。字段顺序与冻结model_matrix和既有适配代码核对；不是M2的72内容＋72缺失指示＋3质量的相同参数化。字段定义/转换及未核准单位见feature_contract.json。没有新增availability数组。

原始财务披露版本的PIT认证、派生字段全部组成信息的最晚可得日以及制造业历史分类时钟仍未补齐，本轮未从财报期伪造发布日期。冻结输入具有原审计可用时钟，但不能据此声称原始版本已经PIT认证。要解除PROVISIONAL，需要上游逐组成字段历史披露/修订版本与行业分类历史有效日证据。此次仅做明确探索性比较。

原全训练q2使用全部制造业训练记录；M2为原全行业训练模型在同一制造业验证记录上的冻结预测。本轮六方案与EBM仅用结构角色公司拟合，内部留出用于统一树权重或EBM轮数选择。前两者是描述性参照，不能作为公司角色匹配的机制消融。
