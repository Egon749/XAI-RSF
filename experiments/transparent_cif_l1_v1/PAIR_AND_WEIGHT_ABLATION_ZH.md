# 成对候选与全局权重的独立贡献

本轮没有证据支持将联合成对候选或全局权重升级为整体更优机制。所有树/角色匹配及数值收敛已通过，因此当前不利结果不能归咎于未运行或优化失败。数据为PROVISIONAL；2023/2024是后暴露探索性比较，不能称独立盲测。原财务披露版本PIT、行业历史时钟和47项未决事件限制保留。

| model | auc | ap | ipcw_brier |
| --- | --- | --- | --- |
| base_equal | 0.838715 | 0.0824313 | 0.00946755 |
| base_weighted | 0.836422 | 0.0739659 | 0.0095336 |
| ordinary_extra_equal | 0.840887 | 0.0849627 | 0.00944706 |
| ordinary_extra_weighted | 0.841005 | 0.0854986 | 0.00944132 |
| pair_equal | 0.839708 | 0.0746329 | 0.009462 |
| pair_weighted | 0.836074 | 0.0743767 | 0.00953104 |


| contrast | metric | point_difference | conditional_ci_lower | conditional_ci_upper |
| --- | --- | --- | --- | --- |
| base_weight_effect | ap | -0.00846545 | -0.0402035 | 0.0197895 |
| base_weight_effect | ipcw_brier | 6.60503e-05 | 1.77464e-05 | 0.000120391 |
| ordinary_weight_effect | ap | 0.000535961 | -0.0212021 | 0.0251644 |
| ordinary_weight_effect | ipcw_brier | -5.73531e-06 | -5.07168e-05 | 4.46564e-05 |
| pair_weight_effect | ap | -0.00025628 | -0.0322444 | 0.0311403 |
| pair_weight_effect | ipcw_brier | 6.90333e-05 | 2.10584e-05 | 0.000122369 |
| pair_equal_effect | ap | -0.0103297 | -0.030114 | 0.000863326 |
| pair_equal_effect | ipcw_brier | 1.49455e-05 | -2.59933e-06 | 3.1886e-05 |
| PRIMARY_pair_weighted_effect | ap | -0.011122 | -0.029085 | 0.000721005 |
| PRIMARY_pair_weighted_effect | ipcw_brier | 8.97141e-05 | 7.03085e-05 | 0.000109583 |
| total_effect | ap | -0.00805469 | -0.0401642 | 0.0207823 |
| total_effect | ipcw_brier | 6.34828e-05 | 1.72141e-05 | 0.000116716 |
| practical_vs_fulltrain_q2 | ap | -0.0174612 | -0.0479625 | 0.00941013 |
| practical_vs_fulltrain_q2 | ipcw_brier | 5.09591e-05 | 1.18928e-05 | 9.2839e-05 |


主要机制pair_weighted−ordinary_extra_weighted：AP −0.011122，条件区间[−0.029085,0.000721]；Brier +0.000090，区间[0.000070,0.000110]。AP缺乏正向增量证据，Brier方向不利。pair_equal对同树数ordinary_equal的AP也较低，不能将单纯加树收益归为联合成对机制。

权重作用依赖银行：base/pair加权合并AP未改善且Brier更高；ordinary加权AP仅+0.000536、Brier约−0.000006，区间均含0。不能说权重普遍改进；也不能把区间含0解释为等效。pair_weighted相对base_equal的AP−0.008055、Brier+0.000063；不是只有Brier改善、排名没改善，而是该总体方案两项合并点估计均不利。

年度方向混合：pair_equal相对ordinary_equal在2023 AP更高（0.080625对0.071132），2024更低（0.076866对0.099365）。pair_weighted对ordinary_weighted在2023几乎相同，2024较低。三seed概率平均与单seed统计分别交付，未挑seed或年度。

所有区间均按训练前声明未作多重比较调整，只作固定预测的探索性条件评价；保留全部方向，不据单个端点显著性宣布方法首创或总体优越。
