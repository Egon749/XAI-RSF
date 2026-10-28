# 预测、核查负担与条件区间

状态：PROVISIONAL；2023/2024后暴露探索性。没有用评价结局选择树权重、EBM轮数或追加配置。六个新森林及两种EBM公司角色匹配；原全训练q2与全行业M2仅作描述性参考。EBM是12个月二分类端点模型，不提供本轮原生全时间竞争CIF；数值解释为logit经logistic，而森林是CIF概率恒等链接。

主要配对统计为同种子比较，再对三个种子的指标求均值。合并期AP首先在合并记录上计算，绝非年度AP平均。下面是**三个单种子指标均值**，不是概率集成：

| model | auc | ap | ipcw_brier | prediction_mean | observed_ipcw_incidence |
| --- | --- | --- | --- | --- | --- |
| M2_allsector_reference | 0.864435 | 0.143716 | 0.00965798 | 0.0185938 | 0.00950082 |
| base_equal | 0.837811 | 0.0798441 | 0.00947875 | 0.0174256 | 0.00950082 |
| base_weighted | 0.834114 | 0.0665797 | 0.00955088 | 0.0175478 | 0.00950082 |
| ebm_pair0 | 0.877009 | 0.139874 | 0.00936769 | 0.0123293 | 0.00950082 |
| ebm_pair10 | 0.835857 | 0.135515 | 0.00965921 | 0.0127998 | 0.00950082 |
| ordinary_extra_equal | 0.840332 | 0.0838579 | 0.00945572 | 0.0172007 | 0.00950082 |
| ordinary_extra_weighted | 0.83713 | 0.0778121 | 0.00945284 | 0.01692 | 0.00950082 |
| pair_equal | 0.83954 | 0.074225 | 0.00947086 | 0.0175716 | 0.00950082 |
| pair_weighted | 0.834405 | 0.0672085 | 0.0095462 | 0.0176023 | 0.00950082 |
| q2_fulltrain_reference | 0.842184 | 0.0837777 | 0.00949645 | 0.0177839 | 0.00950082 |

下面是**先平均三种子概率、再计算指标**，与上表不同：

| scope | model | auc | ap | ipcw_brier |
| --- | --- | --- | --- | --- |
| 2023 | M2_allsector_reference | 0.868098 | 0.121186 | 0.00842371 |
| 2023 | base_equal | 0.847189 | 0.0760544 | 0.00817429 |
| 2023 | base_weighted | 0.835371 | 0.0576706 | 0.00823947 |
| 2023 | ebm_pair0 | 0.881244 | 0.122468 | 0.00810111 |
| 2023 | ebm_pair10 | 0.843731 | 0.107832 | 0.00841347 |
| 2023 | ordinary_extra_equal | 0.849974 | 0.073438 | 0.00815761 |
| 2023 | ordinary_extra_weighted | 0.845133 | 0.0613947 | 0.00815557 |
| 2023 | pair_equal | 0.849569 | 0.0810482 | 0.00816637 |
| 2023 | pair_weighted | 0.833966 | 0.0611651 | 0.00823782 |
| 2023 | q2_fulltrain_reference | 0.850544 | 0.0788127 | 0.00818078 |
| 2024 | M2_allsector_reference | 0.860974 | 0.177029 | 0.01084 |
| 2024 | base_equal | 0.829977 | 0.0921107 | 0.0107392 |
| 2024 | base_weighted | 0.835139 | 0.0908479 | 0.0108108 |
| 2024 | ebm_pair0 | 0.872759 | 0.16344 | 0.0105741 |
| 2024 | ebm_pair10 | 0.82957 | 0.165554 | 0.0108283 |
| 2024 | ordinary_extra_equal | 0.831771 | 0.0991918 | 0.0107144 |
| 2024 | ordinary_extra_weighted | 0.835634 | 0.104842 | 0.0107062 |
| 2024 | pair_equal | 0.83021 | 0.0777265 | 0.0107373 |
| 2024 | pair_weighted | 0.836395 | 0.0919942 | 0.0108021 |
| 2024 | q2_fulltrain_reference | 0.836844 | 0.107005 | 0.0107648 |
| pooled | M2_allsector_reference | 0.865613 | 0.148242 | 0.00964491 |
| pooled | base_equal | 0.838855 | 0.0831873 | 0.00947202 |
| pooled | base_weighted | 0.836972 | 0.0737268 | 0.0095404 |
| pooled | ebm_pair0 | 0.877454 | 0.141391 | 0.00935076 |
| pooled | ebm_pair10 | 0.83681 | 0.136076 | 0.00963346 |
| pooled | ordinary_extra_equal | 0.84106 | 0.0854928 | 0.00945127 |
| pooled | ordinary_extra_weighted | 0.841371 | 0.0853506 | 0.00944608 |
| pooled | pair_equal | 0.840084 | 0.0753903 | 0.00946714 |
| pooled | pair_weighted | 0.836973 | 0.074587 | 0.0095352 |
| pooled | q2_fulltrain_reference | 0.843991 | 0.0926698 | 0.00948808 |

记录等权与公司等权完整结果分别见metrics.csv、mean_seed_metrics.csv；季度警报/年度去重、所有种子及概率平均详见warning_company_quarter.csv。观察目标率的IPCW估计也保留；合并记录等权约0.950%，公司等权约1.685%，不能混用。校准十分位箱见calibration.csv，既含所有记录又列未知数量。

1000次原协议公司组有放回抽样使用seed20261025和spawn_key=(replicate)，同一公司抽样共同用于所有模型/种子/年度；预测、名单、原删失权重固定，不重训、不重排。2400个点估计与原评价直接核对，最大差约2.39e-13，并以非平凡重复公司抽样与sklearn核验排序指标。它是条件评价区间，不含训练不确定性；不作多重比较校正、不据此挑选优胜配置。

| model_a | model_b | metric | point | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- |
| base_weighted | base_equal | ap | -0.0132644 | -0.0405657 | 0.00771205 |
| base_weighted | base_equal | auc | -0.00369675 | -0.0108343 | 0.00297845 |
| base_weighted | base_equal | ipcw_brier | 7.21255e-05 | 2.44789e-05 | 0.000126611 |
| ordinary_extra_weighted | ordinary_extra_equal | ap | -0.00604575 | -0.0287595 | 0.0125659 |
| ordinary_extra_weighted | ordinary_extra_equal | auc | -0.00320214 | -0.0120206 | 0.00409106 |
| ordinary_extra_weighted | ordinary_extra_equal | ipcw_brier | -2.88354e-06 | -4.70657e-05 | 4.69603e-05 |
| pair_weighted | pair_equal | ap | -0.0070166 | -0.0372217 | 0.0164122 |
| pair_weighted | pair_equal | auc | -0.00513532 | -0.0145904 | 0.00328566 |
| pair_weighted | pair_equal | ipcw_brier | 7.53333e-05 | 2.56876e-05 | 0.000128933 |
| ordinary_extra_equal | base_equal | ap | 0.00401384 | -0.00158449 | 0.0125112 |
| ordinary_extra_equal | base_equal | auc | 0.00252158 | 0.000388218 | 0.00488451 |
| ordinary_extra_equal | base_equal | ipcw_brier | -2.3028e-05 | -3.0409e-05 | -1.51345e-05 |
| pair_equal | ordinary_extra_equal | ap | -0.00963284 | -0.0263629 | 0.000169206 |
| pair_equal | ordinary_extra_equal | auc | -0.000792254 | -0.00405696 | 0.00218161 |
| pair_equal | ordinary_extra_equal | ipcw_brier | 1.51415e-05 | -2.39558e-06 | 3.20266e-05 |
| pair_weighted | ordinary_extra_weighted | ap | -0.0106037 | -0.0252228 | -0.00118451 |
| pair_weighted | ordinary_extra_weighted | auc | -0.00272543 | -0.00806775 | 0.00208934 |
| pair_weighted | ordinary_extra_weighted | ipcw_brier | 9.33583e-05 | 7.38754e-05 | 0.000113469 |
| pair_weighted | base_equal | ap | -0.0126356 | -0.0404458 | 0.00796196 |
| pair_weighted | base_equal | auc | -0.003406 | -0.0120849 | 0.00431817 |
| pair_weighted | base_equal | ipcw_brier | 6.74467e-05 | 2.11086e-05 | 0.000119577 |
| pair_weighted | ebm_pair0 | ap | -0.072666 | -0.130581 | -0.027631 |
| pair_weighted | ebm_pair0 | auc | -0.0426042 | -0.0630657 | -0.0221331 |
| pair_weighted | ebm_pair0 | ipcw_brier | 0.000178504 | -0.000446839 | 0.000799669 |
| pair_weighted | q2_fulltrain_reference | ap | -0.0165692 | -0.0420639 | 0.00316426 |
| pair_weighted | q2_fulltrain_reference | auc | -0.00777894 | -0.0161877 | 0.00011281 |
| pair_weighted | q2_fulltrain_reference | ipcw_brier | 4.97423e-05 | 9.85387e-06 | 9.19823e-05 |
| pair_weighted | M2_allsector_reference | ap | -0.0765077 | -0.119652 | -0.0286632 |
| pair_weighted | M2_allsector_reference | auc | -0.0300303 | -0.0450721 | -0.0166983 |
| pair_weighted | M2_allsector_reference | ipcw_brier | -0.000111781 | -0.000615067 | 0.000330742 |

联合候选＋权重相对基础等权，平均种子AP差−0.012636，区间跨零；Brier差+0.000067，区间[+0.000021,+0.000120]。相对同48树的普通扩展＋权重，AP差−0.010604，区间[−0.025223,−0.001185]；Brier亦较差。本轮证据不能支持联合候选或学习权重的独立预测增益。原M2/q2参照不角色匹配，差距不能全归于机制。

合并期**概率平均**的Top20/25核查结果：

| model | top_fraction | alert_company_groups | hit_company_groups | company_hit_fraction | independent_events | detected_events | event_coverage | announcement_detected_3m | announcement_detected_6m |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M2_allsector_reference | 0.2 | 1152 | 66 | 0.0572917 | 79 | 66 | 0.835443 | 49 | 47 |
| M2_allsector_reference | 0.25 | 1398 | 69 | 0.0493562 | 79 | 69 | 0.873418 | 50 | 50 |
| base_equal | 0.2 | 1166 | 61 | 0.0523156 | 79 | 61 | 0.772152 | 44 | 42 |
| base_equal | 0.25 | 1379 | 67 | 0.0485859 | 79 | 67 | 0.848101 | 50 | 47 |
| base_weighted | 0.2 | 1188 | 61 | 0.0513468 | 79 | 61 | 0.772152 | 43 | 42 |
| base_weighted | 0.25 | 1405 | 68 | 0.0483986 | 79 | 68 | 0.860759 | 49 | 46 |
| ebm_pair0 | 0.2 | 1149 | 68 | 0.0591819 | 79 | 68 | 0.860759 | 48 | 43 |
| ebm_pair0 | 0.25 | 1346 | 71 | 0.0527489 | 79 | 71 | 0.898734 | 51 | 50 |
| ebm_pair10 | 0.2 | 1149 | 64 | 0.0557006 | 79 | 64 | 0.810127 | 44 | 42 |
| ebm_pair10 | 0.25 | 1370 | 67 | 0.0489051 | 79 | 67 | 0.848101 | 48 | 46 |
| ordinary_extra_equal | 0.2 | 1162 | 61 | 0.0524957 | 79 | 61 | 0.772152 | 44 | 41 |
| ordinary_extra_equal | 0.25 | 1375 | 67 | 0.0487273 | 79 | 67 | 0.848101 | 50 | 47 |
| ordinary_extra_weighted | 0.2 | 1178 | 62 | 0.0526316 | 79 | 62 | 0.78481 | 44 | 41 |
| ordinary_extra_weighted | 0.25 | 1402 | 66 | 0.0470756 | 79 | 66 | 0.835443 | 49 | 48 |
| pair_equal | 0.2 | 1167 | 61 | 0.0522708 | 79 | 61 | 0.772152 | 44 | 42 |
| pair_equal | 0.25 | 1389 | 67 | 0.0482361 | 79 | 67 | 0.848101 | 50 | 47 |
| pair_weighted | 0.2 | 1187 | 60 | 0.0505476 | 79 | 60 | 0.759494 | 43 | 42 |
| pair_weighted | 0.25 | 1399 | 66 | 0.0471766 | 79 | 66 | 0.835443 | 49 | 46 |
| q2_fulltrain_reference | 0.2 | 1171 | 62 | 0.0529462 | 79 | 62 | 0.78481 | 45 | 42 |
| q2_fulltrain_reference | 0.25 | 1379 | 67 | 0.0485859 | 79 | 67 | 0.848101 | 49 | 46 |

企业级命中占比=至少一个被标记窗口命中目标事件的去重公司/去重预警公司。本队列每个目标事件是一家企业的首次事件，故本表事件/公司分子相同；实现仍分别计数。不能把未命中公司一律叫正常，也不能把这个含未知负担的比率叫成熟结局PPV。

| top_fraction | alert_records | repeated_alert_records | known_no_target_alerts | unknown_outcome_alerts | no_hit_with_unknown_groups | no_hit_all_alerts_known_non_target_groups | announcement_opportunity_3m | announcement_opportunity_6m |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.2 | 4761 | 3574 | 4388 | 216 | 93 | 1034 | 60 | 60 |
| 0.25 | 5953 | 4554 | 5521 | 261 | 107 | 1226 | 60 | 60 |

联合＋权重Top25核查1399家公司，确认66/79独立事件，企业命中占比4.718%；107家无命中但存在未知预警窗口，1226家所有预警窗口均为已知非目标。后者仅说明这些窗口，不等于企业永远正常。5953条季度警报中4554条为去重公司数之外的重复警报。

公告前3/6个月用原日历日期而非90/180天近似，合并可识别机会分母各60；联合＋权重Top25分别49与46，不能以79直接解释为机会覆盖率。年度、季度及实施日前提前量也完整输出。AP/AUC只说明排序；Brier兼顾概率偏差；事件覆盖与核查负担须同时看，不能用覆盖单项宣布成功。
