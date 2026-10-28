# 冻结L1及后续数据交接

L1已形成，作为后续透明模型研究工作基线；依据是3项已裁决证据，不是择优模型成绩。L0永久保留。本轮不启动transparent_cif_v2六方案或审计信息增量实验。

- 原始L1输入：`/home/oliver/MainDiskData/label_revision_sensitivity_v1_20261028T151804Z/input/L1_ADJUDICATED`。
- 完整数据版本manifest：`/home/oliver/MainDiskData/label_revision_sensitivity_v1_20261028T151804Z/input/data_version_manifest.json`。
- 可直接作为run.py的`--source-input`：`/home/oliver/MainDiskData/label_revision_sensitivity_v1_20261028T151804Z/handoff_L1`。
- 特征合同`--features`：`/home/oliver/MainDiskData/label_revision_sensitivity_v1_20261028T151804Z/handoff_L1/features.json`。
- 交接SHA256及文件清单：`/home/oliver/MainDiskData/label_revision_sensitivity_v1_20261028T151804Z/handoff_L1/handoff_manifest.json`。

映射：final_train_PRIVATE.npz/event及配套metadata使用L1；X/time/groups/sample_ids/horizon与L0一致。final_eval两文件逐字节保持原版。plan.json保留原forest_configs.q2，刷新L1计数及输入哈希，去掉原基线大实验/模拟任务清单，不能由旧plan误触发其它任务。features.json保留72内容+3质量字段及原缺失处理，单位未知继续UNVERIFIED，未追加审计特征或72个独立缺失列。

已实际调用transparent_cif_v2的training_contract和feature_contract，无fit；结果PASS，56,458行、75特征、12日历月365/366天、有效端点39,362行、历史截止2022-01-01。未来按原公司角色分配仍须先核验新原因支持；不能自动复用旧标签权重或拟合模型。

reference_L0_q2_{seed}_PRIVATE.npz为旧全训练q2，reference_L1_q2_{seed}_PRIVATE.npz为修订全训练q2；两者不能冒充未来80/20公司角色匹配的base_equal。M2_PRIVATE.npz始终是原全行业历史参考。旧server/prepare.py硬编码父输入白名单，应使用本次已验证的交接目录/manifest而非把旧父hash检查强行改成通过。

合并记录等权基准：q2 L0 AUC0.843991/AP0.092670/Brier0.009488；L1 AUC0.843814/AP0.091838/Brier0.009480。年度、单seed、配对区间和Top20/25见本目录CSV。不得只引用合并点估计隐藏不利年度/名单变化。

数据保持 **PROVISIONAL**；2023/2024为已接触结局后的探索性比较。原始财务披露版本PIT、行业分类历史时钟和47项未决事件没有因本轮补丁获得认证。 后续若获授权，必须在新的唯一运行目录执行；不得把本轮交接当成已完成下一阶段研究。
