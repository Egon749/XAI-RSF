# 本轮消融的基线核验

设计来源为261eb15e95b01e4527e07648216bbee3974f8444；起始工作区3931d0c69141e294ec2fb5e744bb912c80fe2c1a且干净。两提交间仅增加2025实验、图表和相关测试/说明；核心森林、当前参考声明和历史深度收缩实验没有变化。本轮不读取2025数据。

| 项目 | 实际核验 |
|---|---|
| 参考 | post_st_leaf_shallow_v1 / shallow_parent / xai_q2 |
| 训练 | 2023:309公司14退市；2024:345公司26退市（24个月/训练截止） |
| 测试 | 2023:36公司12退市；2024:45公司20退市（24个月）；两个期限全部结局已知 |
| 配置 | 32树；种子42/101/202；q2；深度4；min_leaf10/min_groups5/min_events3；mtry75；5切点；共享分位数；每树均衡25/83变量；原因权重(1,1)；全局预算None |
| 派生估计 | raw直接父节点AJ、独立in-bag公司数权重n/(n+10)、单位时间权重；不是ForestConfig新增参数 |
| 可复用 | 四臂各6份模型与预测，共24；另6份标准RSF外部参照；全部原始manifest SHA256通过 |
| 当前参考 | 六份模型和预测逐项匹配CURRENT_POST_ST_MODEL.json哈希 |
| 数据 | 旧C输入逐字节复用；公告后的下一交易日开盘前；先X退市/先A恢复；日历12/24月；训练信息严格在外层年初前 |
| 数据限制 | 保守披露时钟不等于原始版本已PIT认证；历史观察首次口径；2023/2024多次暴露 |
| 差异 | 源plan公共深度6由原worker覆盖为4；保存模型确认为4；此次新增显式覆盖开关，旧字段缺失按balanced读取 |
| 必要重现拟合 | 0；模型、预测、函数、审计工件齐全 |

历史私有运行：`/home/oliver/MainDiskData/xai_rsf_leaf_shallow_20261028_ep_9hazj`。逐个模型位置与哈希见results/artifact_reuse.csv，逐参数表见results/parameter_differences.csv。本轮运行：`/home/oliver/MainDiskData/xai_rsf_post_st_ablation_20261029_uDFKagf2`。

| 年度 | 文件 | SHA256 |
|---|---|---|
| 2023 | train_PRIVATE.npz | `2b8ced70e0b865d9bb8e3ea0ac262be26e8da903a5910b5845f98ddc93b19460` |
| 2023 | eval_PRIVATE.npz | `753c94b0d5b4c2e5e6ea55dfbcc8273a9cf55184b441b7b66290ab83752fb016` |
| 2024 | train_PRIVATE.npz | `88fb1d83242d0b7457b123f1f99c20f860756d3b333f3e748ca7a32006aea24f` |
| 2024 | eval_PRIVATE.npz | `f75d7359b81a60d88e746a80a13e93aca32db8927c8ee7fdc758316e3f97493e` |
