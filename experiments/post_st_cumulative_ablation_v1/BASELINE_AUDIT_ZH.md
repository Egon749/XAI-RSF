# 累积消融基线核验

工作区起始ad49e66c25f6e76ad8206ddaf7f334ac4fcbd88c，干净；CURRENT_POST_ST_MODEL.json不变。完整模型s0及无q限制/共享切点/均衡覆盖/收缩s1共12份模型的哈希、实际配置和强度10均通过。原始样本与输入逐字节复用，完整哈希在input_hashes.csv；各模型位置在artifact_reuse.csv。

| 阶段 | q上限 | 切点 | 每树25变量 | 收缩强度 | 本轮拟合 |
|---|---|---|---|---|---|
| s0 完整 | 2 | shared_quantile | balanced | 10 | 复用6份 |
| s1 解除q | None，深度仍4 | shared_quantile | balanced | 10 | 复用6份 |
| s2 再改切点 | None | local_random | balanced | 10 | 新拟合6次并派生 |
| s3 再改覆盖 | None | local_random | iid | 10 | 新拟合6次并派生 |
| s4 再取消收缩 | None | local_random | iid | 0 | 使用s3的raw，不增拟合 |

所有阶段深度4、每种子32树、种子42/101/202、min_leaf10/min_groups5/min_events3、mtry75、n_thresholds5、原因权重(1,1)、全局单元预算None、单位时间权重。强度是拟合后的叶估计操作，不是森林配置字段。完整相邻参数差异表在parameter_differences.csv。

2023训练309家/14例24个月先退市，测试36家/12例；2024训练345家/26例，测试45家/20例。公告后下一交易日开盘前为原点；先最终退市X与先完整恢复A为竞争终点。测试覆盖沿用2026-09-03，两个期限未知均0，训练删失保留；原始版本PIT及观察首次口径限制不变。本轮不读取2025数据。

12次新增拟合，12个收缩派生版本，最终链包含30个单种子模型/10个96树平均预测器。用于固定顺序的探索性机制分析，不更换参考，不扩展删除顺序。
