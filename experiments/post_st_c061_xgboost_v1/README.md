# C061 + XGBoost：有限历史前向对比

[结果和局限](RESULTS_ZH.md) · [主表](MODEL_COMPARISON.csv) · [全部种子](METRICS_ALL.csv) · [配对区间](PAIRED_DIFFERENCES.csv) · [校准](CALIBRATION.csv) · [图](figures/performance_comparison.png) · [冻结协议](FROZEN_PROTOCOL.json)

[原有PROBoost、Chen Stacking与新增XGBoost汇总](ALL_COMPARATORS_CONTEXT.csv)：旧方法只复用已发布点估计，并核对相同输入/结局哈希；不冒充本轮新拟合。

参考模型是C061（17/155/8679），不是C039；不修改CURRENT_POST_ST_MODEL.json。新增的是普通IPCW-XGBoost固定时限退市分类器；不替代竞争风险模型。所有结果是已暴露2023–2025队列的探索性结果。

在具备原服务器存档的环境中，首先将run_location.json中的private_run改为**新建唯一目录**，保留source_run。不要对已有input或published目录重新stage/schedule。deps需要xgboost-cpu3.4.1及已有数值环境。只有source_run和evaluation阶段读取原始评价结局，训练工作输入的eval.npz仅含X和对齐键。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# PYTHONPATH包含run_location.json的deps目录和本仓库；研究环境Python3.13.13。
# 完整测试同时设置既有CRFOREST_RSCRIPT和EW_RLIB，保留独立R检验。
python -m pytest -q
python -m experiments.post_st_c061_xgboost_v1.stage
# 此前须已提交冻结协议与实现；schedule只分发已提交代码。
python -m experiments.post_st_c061_xgboost_v1.schedule
python -m experiments.post_st_c061_xgboost_v1.evaluate
python -m experiments.post_st_c061_xgboost_v1.report
```

已有成功私有输出时只运行evaluate/report可复核，不必重训。完整648次历史模型和18次最终模型均在published目录，个体预测不能上传Git。公共SOURCE_HASHES/EVALUATION_HASHES/PUBLICATION_CHECKS用于核对输入、原C061模型、统一抽样及每项结果回传。

TRAINING_SUPPORT记录历史监督支持；TRAINING_LOG和TUNING_ALL记录每一候选/折/种子及选择。COMPUTE_AND_PARAMETERS按最终种子列示实际树数、叶数、使用变量数和成本，task字段在三行中重复，汇总时按年度/时限去重。C061_EXPLANATION_AUDIT明确复用历史完整分解审计，不把历史计时写成本次新训练。

所有比较差值为XGBoost减C061。单种子AUC平均不等于先平均预测后计算的组合AUC。2025H24无性能。HORIZON_COHERENCE仅检查两套独立分类得分能否作为时间单调曲线，不能据此给2025H24计算性能。
