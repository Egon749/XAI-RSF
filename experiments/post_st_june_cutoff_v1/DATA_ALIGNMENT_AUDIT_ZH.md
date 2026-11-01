# 截点与数据核验

| cutoff_year | training_cutoff | test_entry_from | test_entry_through | training_companies | training_target_events | training_recovery_events | training_target_events_within12 | evaluation_companies_available | entry_cohort_complete | evaluation_status | maximum_training_stop | train_test_groups_disjoint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2023 | 2023-06-30 | 2023-07-01 | 2024-06-30 | 343 | 14 | 194 | 5 | 46 | True | SUPPORTED | 2023-06-30 | True |
| 2024 | 2024-06-30 | 2024-07-01 | 2025-06-30 | 390 | 32 | 206 | 9 | 52 | True | SUPPORTED | 2024-06-30 | True |
| 2025 | 2025-06-30 | 2025-07-01 | 2026-06-30 | 442 | 53 | 227 | 25 | 1 | False | BLOCKED_INCOMPLETE_ENTRY_COHORT | 2025-06-30 | True |

复用443家审核冻结企业的旧版conservative C83入组输入，未重新接受修订年报。使用真实financial_available、market_available及48项财务字段可得日数组核对不晚于预测原点；沿用历史保守版本及其PIT来源局限，这不是新增的逐份原始披露版本认证。保留变量逐值、缺失状态及训练共享切点一致。

结局从原始事件链、公告日、公司组映射及既有审核记录重建。训练结局截断到当年6月30日，不能沿用旧年初模型已经截断的标签；未发生/尚未知的未来结局不设为正常。测试标签单独存储在evaluation_input，正式训练DATA_DIR仅含输入X、sample_id与训练标签。

2023截点有1家在当日入组且截至当天随访长度为0，按既有label_record规则从两臂一致排除；有效训练343家。2024及2025训练分别390与442家。训练/测试公司不重叠，两个完整测试窗口也不重叠。公司bootstrap及重复次数逐树配对，共288对；固定九项名单及来源哈希见协议，保留其此前诊断使用截至2024年初信息的来源背景。

首次准备将2023训练数量预期写为344，审计拒绝后发现上述零随访边界；没有拟合。保留失败目录，在新的唯一目录修正预期为343并重新准备，未修改实际标签或强制正随访。详见audit/preparation_history.json。

2026审核队列缺失的上游来源详见audit/source_search.json。原始事件可见不等于已完成首次事件审核，不制造availability、资格或随访。所有测试结局此前已接触，本轮为固定设计比较。

旧27份模型、预测、函数工件以及CURRENT_POST_ST_MODEL.json哈希均保持不变。私有运行目录：`/home/oliver/MainDiskData/xai_rsf_june_cutoff_20261101_oc6r2y1v`。公司身份、逐公司结局/预测、参数和切点保留服务器；公开汇总不含2025截点唯一公司的任何风险或结局摘要。
