# 完整函数解释

路径始终最多2个原始输入变量，允许在已使用变量上继续分裂。全部主效应和二元交互精确重构两原因CIF，不是事后拟合代理模型，也没有未解释的高阶残差。这不自动证明统计可辨识、跨样本稳定或因果效应。

| seed | native_all_time_main_functions | native_all_time_pair_functions | spatial_table_cells | scalar_table_cells | reference_active_mains | reference_active_pairs | reference_pair_absolute_share | low_support_cell_fraction | zero_support_cell_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | 42 | 52 | 1665 | 196470 | 39 | 47 | 0.4128 | 0.4336 | 0.1267 |
| 101 | 43 | 59 | 1761 | 207798 | 37 | 50 | 0.4154 | 0.4208 | 0.1420 |
| 202 | 43 | 56 | 1772 | 209096 | 42 | 54 | 0.4488 | 0.4430 | 0.1614 |

全原生时间网格42—43个主效应、52—59个交互，1665—1772个空间格。原因／时限特定的参考非零项可少于全函数数量。两个原因、365/366天、原生网格及保存重载检查的最大重构误差 3.331e-16。贡献在CIF概率尺度相加，不是对数风险率或因果贡献。

## 跨种子稳定性

| seed_a | seed_b | main_jaccard | pair_jaccard | common_nonconstant_functions | median_function_correlation |
| --- | --- | --- | --- | --- | --- |
| 42 | 101 | 0.5455 | 0.0882 | 34 | 0.6038 |
| 42 | 202 | 0.5455 | 0.0485 | 32 | 0.5464 |
| 101 | 202 | 0.5088 | 0.0550 | 30 | 0.6301 |

交互集合Jaccard只有0.049—0.088，主效应约0.509—0.545。约42%—44%的函数格少于10家训练公司支持。完整重构依然精确，但“简单、可靠、低歧义”没有得到充分证明。跨种子方向相关基于相同390家参考、固定365天，三个种子范围不是置信区间；没有声称优于SHAP的稳定性，也没有测试可辨识性惩罚。

## 实际变量使用和原因支持

| model | seed | actual_used_inputs | leaves | zero_termination_leaf_fraction | fewer_than_three_termination_leaf_fraction | zero_recovery_leaf_fraction | price_eligible_trees | price_used_trees | root_price_trees |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| xai_q2 | 42 | 42 | 468 | 0.4573 | 0.8803 | 0.0321 | 11 | 8 | 3 |
| xai_q2 | 101 | 43 | 470 | 0.4468 | 0.8936 | 0.0234 | 10 | 5 | 3 |
| xai_q2 | 202 | 43 | 475 | 0.4695 | 0.9011 | 0.0295 | 11 | 9 | 6 |
| unrestricted | 42 | 69 | 437 | 0.5195 | 0.8696 | 0.0549 | 32 | 24 | 7 |
| unrestricted | 101 | 71 | 422 | 0.4739 | 0.8602 | 0.0640 | 32 | 24 | 4 |
| unrestricted | 202 | 70 | 429 | 0.4965 | 0.8671 | 0.0676 | 32 | 26 | 7 |

每叶至少3条全部原因事件不保证有3家独立退市企业。真实变量清单、切点、参数、全函数值和企业贡献均在私有目录；仓库保存汇总诊断。变化不能归因于新的解释机制，因为模型机制没有改变。
