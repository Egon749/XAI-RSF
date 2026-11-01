# 数据与配对核验

原输入来自post_st_leaf_shallow_v1的2024冻结折，旧conservative C83列。只删除下表9列，其余74列保持原顺序、原值与缺失状态；train/eval的全部非X数组逐项相等。原始训练标签、删失、时间起点和公司集合不变。

| original_index | name | label |
| --- | --- | --- |
| 2 | net_margin | 净利率 |
| 4 | operating_profit_to_assets | 营业利润/总资产 |
| 5 | net_profit_to_assets | 净利润/总资产 |
| 20 | cfo_to_current_liabilities | 经营现金流/流动负债 |
| 26 | working_capital_to_assets | 营运资本/总资产 |
| 39 | profit_total_growth | 利润总额增长 |
| 58 | total_mktcap_change_3m | 近3个月总市值变化 |
| 60 | downside_vol_3m | 近3个月下行波动率 |
| 71 | total_mktcap_change_12m | 近12个月总市值变化 |

名单来源：post_st_feature_permutation_v1/results/feature_classification.csv的outer_cutoff=2024且subtype=suspected_redundant；完整名单和源哈希在冻结协议中。没有用本轮2024指标进一步选择字段。

训练345个独立公司，26个先最终退市、198个先恢复，其余删失；入组年份2012—2023，2023年36家。信息截止严格早于2024-01-01，近期公司只贡献截止前的部分随访，不虚构24个月标签。2024评价45家公司，与训练公司不重叠。原点沿用入组公告后的下一交易日开盘前；事件为先最终退市/先恢复的公告日，12/24个日历月评价。数据的金融/市场可得日期均不晚于原点；原年报披露版本并未因此获得新的独立PIT认证。

保留列的训练共享分位数切点逐列完全一致。96棵树的公司bootstrap及重复次数哈希逐棵配对；新拟合的实际inbag重复次数也逐棵验证，旧森林通过原节点重放验证。具体候选变量集合随输入维数变化，见feature_eligibility.csv和tree_pairing.csv。

测试标签在冻结时存于独立evaluation_input目录，不在训练作业DATA_DIR内。训练worker仅接触评价X和sample_id。全部3次拟合和发布成功后，统一评价器才加载测试标签。本轮未接触2025数据。

当前参考JSON及2023/2024六个旧森林、六份预测、六份函数文件，共18个工件SHA256全部未变。输入源哈希与冻结输入哈希均在运行后复核。公司记录、模型参数、切点和预测保留在私有目录：`/home/oliver/MainDiskData/xai_rsf_drop9_20261031_vt0m70vr`。
