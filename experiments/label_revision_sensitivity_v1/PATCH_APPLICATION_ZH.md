# 补丁应用和数据一致性

3项裁决已完整应用到新的L1副本，3公司组、3独立首次事件、12训练窗口全部1→2；没有改为0、寻找后续事件、重新入组或全局替换原因02。

| version | records | groups | target_events | competing_events | target_rows | competing_rows |
| --- | --- | --- | --- | --- | --- | --- |
| L0_ORIGINAL | 56458 | 2577 | 177 | 31 | 654 | 117 |
| L1_ADJUDICATED | 56458 | 2577 | 174 | 34 | 642 | 129 |


NPZ只event数组改变。元数据允许改变event、status、previous_label、label_observed_first、label_eligible、three_view_eligible、followup_reason，且只能在这12行；其余列逐值不变。实际变化逐列统计见label_dependency_map.json，逐行旧新值仅保留私有账本。

字段按原来源代码的定义处理：竞争事件的label_observed_first为NA，label_eligible/three_view_eligible为false；它们表示旧目标/存活二元视图，不能照名称解释为所有已知结局。有效端点资格是label_eligible OR event==2，修正前后完全一致。三视图有11行发生改变，另一行本来为false。原origin/first-history未知标记保留。

全部12个monthly_labels原因单元由1变2；target/competing掩码重建。任意事件(event>0)、时间网格、删失分布以及全训练集原端点IPCW权重逐值一致。未知或提前删失资格不变。sample_id顺序、组、X、缺失位置、特征顺序、首次事件ID、公告/实施日期、stop_date、时间长度和365/366日窗口不变。评价输入和元数据容器逐字节一致；没有读取后期结局选择标签。

重复应用、错误来源哈希、缺失窗口和重复窗口会拒绝。container SHA256与数组/元数据逐列语义哈希同时交付，不能用重序列化哈希变化冒充内容变化。补丁来源SHA256：abda33f9af34913211b9e485670ed62697134b3a4dc9ac43bac7f9f844747a6e。

数据保持 **PROVISIONAL**；2023/2024为已接触结局后的探索性比较。原始财务披露版本PIT、行业分类历史时钟和47项未决事件没有因本轮补丁获得认证。 L1作为未来工作基线由裁决证据决定；指标下降不构成恢复已确认错误标签的理由。
