# 输入、时间与来源核验

| year | fit_information_before | training_companies | training_terminations | training_recoveries | evaluation_companies | paper_listed | paper_available_columns | paper_unmapped | known_duplicate_pair | source_vector_values_verified | pre_origin_clock_checks_passed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2023 | 2023-01-01 | 309 | 14 | 184 | 36 | 54 | 20 | 34 | X1/X2 | True | True |
| 2024 | 2024-01-01 | 345 | 26 | 198 | 45 | 54 | 20 | 34 | X1/X2 | True | True |
| 2025 | 2025-01-01 | 390 | 49 | 211 | 53 | 54 | 20 | 34 | X1/X2 | True | True |

冻结旧版conservative C83输入逐值与原443家入组向量核对。20列除X53市值反变换外保持原值和缺失状态，其余time/event/group/sample_id等数组逐值不变。金融48项真实可得日、增长变量既有双期可得日数组、市场日期均早于各自公告后下一交易日开盘前；继承原始披露版本PIT局限，不构成新版本认证。训练截止严格早于当年1月1日，24个月训练随访上限保持。测试标签位于独立evaluation_input，正式训练仅读取测试X及标识。

三个年度训练/测试公司互斥，测试为36/45/53家；训练公司数为309/345/390，训练先退市14/26/49例。训练集嵌套，不能相加当作独立企业数。每年96棵树公司bootstrap与重复次数逐树配对，共288对。保留列的共享切点不变，X53因变换重新由训练集构造切点。

可映射字段、20列输入质量和全部34项缺项见paper_variable_mapping.csv及results/input_quality.csv。完整54字段构造仍未完成，用户确认接受此子集范围；标签和目标没有改成论文的ST分类。

原83/74共18个模型及其预测/函数哈希保持，CURRENT_POST_ST_MODEL.json不变。私有运行：`/home/oliver/MainDiskData/xai_rsf_paper_features_20261101_4c1852xy`。论文原文、企业记录、模型参数、切点、逐公司预测和解释未提交Git；公开仅代码、变量映射、配置、环境和汇总证据。
