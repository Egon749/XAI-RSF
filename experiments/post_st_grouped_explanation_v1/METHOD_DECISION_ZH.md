# 论文采用判断

**建议加入论文，定位为经济类别的精确解释及敏感性审计。未发现这套经济分组具有优于等规模随机分组的稳定性增量。**

| 问题 | 本轮判断 |
|---|---|
| 四能力比单变量稳定吗 | 在同一批35项财务变量上，四组完整贡献的归一化 L1 在三年均下降，可支持幅度波动有所减小。企业内排序相关却从0.587、0.405、0.470降至0.489、0.372、0.360，不支持四能力重要性次序更容易复现。 |
| 八类经济标签是否优于随机标签 | 以12个月退市为主，经济八组的相对 L1 在2023年好于随机中位数，2024和2025年较差。三个年度均未同时达到预设的随机 L1 第5百分位和排名相关第95百分位标准。不能从这一结果推断经济分组与随机分组等效。 |
| 分组是否隐藏内部不一致 | 存在明显的正负抵消。营运组内部约55%–57%的绝对贡献相互抵消；2024年市场组约68%。完整保留原函数、组内净额和跨组交互，才能解释某一小净额怎样形成。 |
| 哪些能力能跨年重复 | 偿债组的平均绝对幅度在约0.17–0.18个百分点，营运组在后两年最常排第一；四能力中这两类每年都占据前两位的中位排名。对具体企业的方向和顺序仍会变。审计组在八类中表现最一致，市场组次之，现金流重要方向证据较弱。 |
| 哪些跨组项有联合支持 | 在2025年12个月退市中，营运×审计、市场×审计均达到组级门槛，且主导原始项也有实际联合格事件支持。2023年的市场×审计仅过组级门槛，2024年没有跨组项通过。未观察到跨三个年度重复通过的交互。 |
| 可否增加“经济能力层面一致性”结论 | 可写部分组贡献幅度较原变量更易重复，尤其是审计信息在这批企业和固定种子上的重要性。将这种改善归因于四大能力的经济划分、认定对所有企业均成立或据以作因果推断，证据为 INCONCLUSIVE。 |

**建议的论文表述**

> 对保留原预测的八类经济信息归因，在三个已暴露年度队列中，12个月退市贡献向量相对原始变量归因的归一化 L1 距离更小。审计信息的平均绝对贡献在全部三成员组合中均居首位。然而，与200个等规模随机映射比较，预先指定的经济分组未显示跨年可重复的额外稳定性优势。分组有助于组织和核查模型预测，具体财务能力排名与组间作用仍需在新队列验证。

> Grouping the native contributions into eight prespecified economic categories preserved the original competing-risk forest predictions. Across the three previously examined cohorts, grouped attribution vectors had smaller normalized L1 differences between disjoint seed triples than the corresponding feature-level vectors. Audit information consistently ranked first by mean absolute contribution. Comparisons against 200 size-matched random mappings did not establish a reproducible stability advantage specific to the economic taxonomy.

仅固定训练数据和模型设置，更换森林成员；三套年度背景不同，三个年度不能被解读为同一批企业的纵向变化。对称各半分摊跨组项可保证加总，是明确披露的记账选择。当前分析不给出经济因果效应、统计显著性或独立样本性能结论。
