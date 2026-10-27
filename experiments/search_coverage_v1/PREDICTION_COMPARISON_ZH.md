# 预测、事件覆盖与预设门槛

仅V19/V20开发。每配置3种子指标均值，不是种子融合；范围不是置信区间。M2复用冻结预测，输入表示和预算与当前后端不同。同后端unrestricted是主要匹配参照。数据PIT、质量、原点及随访证据仍PROVISIONAL，未使用2022—2024结局。

季度Top20独立事件捕获及公告前3/6月识别：

| scope | model | events_mean | independent_events | lead3_mean | lead6_mean |
| --- | --- | --- | --- | --- | --- |
| V19 | M2 | 75 | 88 | 47 | 46 |
| V19 | baseline | 70 | 88 | 44 | 39 |
| V19 | combined | 72 | 88 | 46 | 42.3333 |
| V19 | coverage | 72 | 88 | 46 | 43 |
| V19 | refined | 69 | 88 | 44.6667 | 38 |
| V19 | unrestricted | 74.6667 | 88 | 47.3333 | 45 |
| V20 | M2 | 80 | 98 | 37 | 32 |
| V20 | baseline | 66.6667 | 98 | 30.6667 | 27.6667 |
| V20 | combined | 73 | 98 | 33.3333 | 27.3333 |
| V20 | coverage | 73.3333 | 98 | 33.3333 | 26.6667 |
| V20 | refined | 66.3333 | 98 | 32 | 26.6667 |
| V20 | unrestricted | 79.3333 | 98 | 37 | 30.6667 |
| pooled | M2 | 114 | 135 | 82 | 78 |
| pooled | baseline | 101.667 | 135 | 73.6667 | 66.6667 |
| pooled | combined | 103.333 | 135 | 77.3333 | 69.6667 |
| pooled | coverage | 103.333 | 135 | 77.3333 | 69.6667 |
| pooled | refined | 101.333 | 135 | 75.6667 | 64.6667 |
| pooled | unrestricted | 111.333 | 135 | 82.3333 | 75.6667 |

完整Top25、每季度核查记录、年度去重公司、已知非目标/未知、重复预警见warning_metrics.csv。独立事件跨年度去重，年度计数不能简单相加。

配对差异（AUC/AP正值较好，Brier负值较好）：

| contrast | scope | auc_difference | ap_difference | brier_difference |
| --- | --- | --- | --- | --- |
| combined | V19 | 0.0496687 | 0.00178752 | -9.87834e-05 |
| combined | V20 | 0.0492435 | 0.017162 | -0.000107847 |
| combined | pooled | 0.0445998 | 0.00720327 | -0.000104366 |
| coverage_only | V19 | 0.0630432 | 0.0761324 | 7.48726e-05 |
| coverage_only | V20 | 0.0498134 | 0.0325354 | -0.000321103 |
| coverage_only | pooled | 0.0534127 | 0.0621571 | -0.000132388 |
| coverage_with_refined | V19 | 0.0414084 | 0.068158 | -0.00051128 |
| coverage_with_refined | V20 | 0.0614214 | 0.0346734 | -0.000459714 |
| coverage_with_refined | pooled | 0.051641 | 0.0463697 | -0.000484658 |
| cuts_only | V19 | 0.0082603 | -0.0663705 | 0.000412497 |
| cuts_only | V20 | -0.0121779 | -0.0175114 | 0.000351867 |
| cuts_only | pooled | -0.0070412 | -0.0391664 | 0.000380292 |
| cuts_with_coverage | V19 | -0.0133745 | -0.0743449 | -0.000173656 |
| cuts_with_coverage | V20 | -0.000569934 | -0.0153734 | 0.000213256 |
| cuts_with_coverage | pooled | -0.00881289 | -0.0549538 | 2.8022e-05 |

预设门槛：AUC差距<=0.02，AP保留>=95%，Brier比<=1.05，Top20平均少捕获<=3，分别对同后端不限阶数对照；非统计非劣效检验。

| model | scope | ap_retention | brier_ratio | mean_events_lost | max_spatial_cells | auc_gap | prediction_gate | complexity_gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | V19 | 0.716632 | 1.09082 | 4.66667 | 299 | 0.0830956 | False | True |
| baseline | V20 | 0.571717 | 1.07121 | 12.6667 | 299 | 0.0856048 | False | True |
| baseline | pooled | 0.668681 | 1.08054 | 9.66667 | 299 | 0.0798105 | False | True |
| coverage | V19 | 0.956852 | 1.09629 | 2.66667 | 1348 | 0.0200523 | False | False |
| coverage | V20 | 0.746215 | 1.04809 | 6 | 1348 | 0.0357914 | False | False |
| coverage | pooled | 0.930027 | 1.07094 | 8 | 1348 | 0.0263979 | False | False |
| refined | V19 | 0.507213 | 1.12097 | 5.66667 | 674 | 0.0748353 | False | True |
| refined | V20 | 0.477798 | 1.09655 | 13 | 674 | 0.0977827 | False | True |
| refined | pooled | 0.504002 | 1.10813 | 10 | 674 | 0.0868517 | False | True |
| combined | V19 | 0.722272 | 1.0836 | 2.66667 | 4433 | 0.0334268 | False | False |
| combined | V20 | 0.663763 | 1.06345 | 6.33333 | 4433 | 0.0363613 | False | False |
| combined | pooled | 0.698968 | 1.07297 | 8 | 4433 | 0.0352108 | False | False |
