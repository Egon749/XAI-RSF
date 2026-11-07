# 固定参数XGBoost随机种子敏感性

[中文结果](RESULTS_ZH.md) · [新种子逐项结果](NEW_SEED_RESULTS.csv) · [新旧组合与C061](ENSEMBLE_COMPARISON.csv) · [全部新旧种子](METRICS_ALL.csv) · [配对区间](PAIRED_DIFFERENCES.csv) · [冻结协议](FROZEN_PROTOCOL.json)

42/101/202替换本轮训练随机性；17/155/8679原模型保留。六个年度—时限任务的超参数完全继承上一轮，只做18次最终拟合，无重新选参。C061不重训、不修改。2025H24性能NA。

在原研究环境中使用新的唯一私有目录，并更新本目录run_location.json。上一实验的run_location.json继续定位原数据和共享依赖，不能修改原运行目录。以下顺序在冻结并提交本轮代码/协议之后执行；PYTHONPATH包含原deps和当前仓库，BLAS线程固定1，完整测试配置CRFOREST_RSCRIPT和EW_RLIB。

```bash
python -m pytest -q
python -m experiments.post_st_xgboost_seeds_v1.stage
python -m experiments.post_st_xgboost_seeds_v1.schedule
python -m experiments.post_st_xgboost_seeds_v1.evaluate
python -m experiments.post_st_xgboost_seeds_v1.report
python -m experiments.post_st_xgboost_seeds_v1.release
```

已有成功输出时只做evaluate/report/release；不要重新stage或schedule覆盖结果。任务使用不可变代码快照、NFS输入与经过哈希确认的回传。PRIVATE文件、模型及公司行不进Git。

各年度固定三种子平均的是预测，不是评价指标；没有合并新旧六个种子。PAIRED_DIFFERENCES中的差值方向为新组合减旧组合或新组合减C061，AUC/AP/捕获越高越好、Brier越低越好。区间不含训练及选参不确定性。
