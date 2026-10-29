# 同口径配置比较及限制

| model | auc | ap | brier | risk_mean | ipcw_observed_incidence | mean_risk_minus_observed |
|---|---|---|---|---|---|---|
| xai_q2 | 0.700000 | 0.181317 | 0.084689 | 0.058547 | 0.094340 | -0.035793 |
| standard_rsf | 0.733333 | 0.223413 | 0.084976 | 0.053243 | 0.094340 | -0.041097 |


两个模型共享390家训练、53家评价、83输入、训练信息截止、测试截止、32树／种子及公司bootstrap。普通RSF保持此前depth6、训练中位数填补、随机局部切点；q2保持depth4、共享切点、每树25变量候选和父节点收缩。这是完整既定配置比较，不是单独q或收缩的因果消融，不是首次ST预警M2。

| model | horizon_months | cause | auc_mean | auc_min | auc_max | auc_std | ap_mean | ap_min | ap_max | ap_std | brier_mean | brier_min | brier_max | brier_std | risk_mean_mean | risk_mean_min | risk_mean_max | risk_mean_std |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| standard_rsf | 12 | recovery | 0.633333 | 0.606667 | 0.660000 | 0.026667 | 0.208874 | 0.110962 | 0.385641 | 0.153381 | 0.084403 | 0.080068 | 0.090609 | 0.005514 | 0.202923 | 0.195252 | 0.217540 | 0.012664 |
| standard_rsf | 12 | termination | 0.686111 | 0.645833 | 0.725000 | 0.039602 | 0.199018 | 0.149267 | 0.239736 | 0.045906 | 0.085242 | 0.083972 | 0.086700 | 0.001374 | 0.053243 | 0.052701 | 0.053909 | 0.000614 |
| xai_q2 | 12 | recovery | 0.744444 | 0.726667 | 0.766667 | 0.020367 | 0.275418 | 0.175926 | 0.409586 | 0.120628 | 0.068180 | 0.065088 | 0.070001 | 0.002692 | 0.187286 | 0.177193 | 0.193391 | 0.008805 |
| xai_q2 | 12 | termination | 0.673611 | 0.650000 | 0.687500 | 0.020554 | 0.172564 | 0.159647 | 0.181602 | 0.011480 | 0.084741 | 0.084612 | 0.084952 | 0.000185 | 0.058547 | 0.055882 | 0.062168 | 0.003250 |


单种子指标范围不是置信区间，集成指标是对平均预测重算，不是上表均值。全部种子均保留。12个月测试端点完整，IPCW权重全为1，公司等权与记录等权一致。

| year | model | seed | horizon_months | cause | observation_cutoff | weighting | risk_quartile_low_to_high | n | predicted_mean | ipcw_observed | events | unknown |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2025 | xai_q2 | ensemble | 12 | termination | 2026-06-30 | one_company_one_record | 1 | 14 | 0.027061 | 0.000000 | 0 | 0 |
| 2025 | xai_q2 | ensemble | 12 | termination | 2026-06-30 | one_company_one_record | 2 | 13 | 0.041762 | 0.076923 | 1 | 0 |
| 2025 | xai_q2 | ensemble | 12 | termination | 2026-06-30 | one_company_one_record | 3 | 13 | 0.070698 | 0.153846 | 2 | 0 |
| 2025 | xai_q2 | ensemble | 12 | termination | 2026-06-30 | one_company_one_record | 4 | 13 | 0.097087 | 0.153846 | 2 | 0 |
| 2025 | xai_q2 | ensemble | 12 | recovery | 2026-06-30 | one_company_one_record | 1 | 14 | 0.123134 | 0.000000 | 0 | 0 |
| 2025 | xai_q2 | ensemble | 12 | recovery | 2026-06-30 | one_company_one_record | 2 | 13 | 0.155042 | 0.000000 | 0 | 0 |
| 2025 | xai_q2 | ensemble | 12 | recovery | 2026-06-30 | one_company_one_record | 3 | 13 | 0.215879 | 0.076923 | 1 | 0 |
| 2025 | xai_q2 | ensemble | 12 | recovery | 2026-06-30 | one_company_one_record | 4 | 13 | 0.260026 | 0.153846 | 2 | 0 |
| 2025 | standard_rsf | ensemble | 12 | termination | 2026-06-30 | one_company_one_record | 1 | 14 | 0.000000 | 0.071429 | 1 | 0 |
| 2025 | standard_rsf | ensemble | 12 | termination | 2026-06-30 | one_company_one_record | 2 | 13 | 0.016093 | 0.000000 | 0 | 0 |
| 2025 | standard_rsf | ensemble | 12 | termination | 2026-06-30 | one_company_one_record | 3 | 13 | 0.073870 | 0.000000 | 0 | 0 |
| 2025 | standard_rsf | ensemble | 12 | termination | 2026-06-30 | one_company_one_record | 4 | 13 | 0.127104 | 0.307692 | 4 | 0 |
| 2025 | standard_rsf | ensemble | 12 | recovery | 2026-06-30 | one_company_one_record | 1 | 14 | 0.072722 | 0.000000 | 0 | 0 |
| 2025 | standard_rsf | ensemble | 12 | recovery | 2026-06-30 | one_company_one_record | 2 | 13 | 0.119257 | 0.076923 | 1 | 0 |
| 2025 | standard_rsf | ensemble | 12 | recovery | 2026-06-30 | one_company_one_record | 3 | 13 | 0.247743 | 0.076923 | 1 | 0 |
| 2025 | standard_rsf | ensemble | 12 | recovery | 2026-06-30 | one_company_one_record | 4 | 13 | 0.381985 | 0.076923 | 1 | 0 |


四分组校准只描述现有预测，没有用2025标签做概率映射。5例退市不能支持反复筛选配置；2025已暴露，不能称为新的严格盲测。
