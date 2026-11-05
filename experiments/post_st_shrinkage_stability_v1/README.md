# 父节点收缩的配对训练扰动验证

固定 C039 的全部结构与预算，每年50次公司重抽样。每次先拟合三个32树原始森林，再由同一原始森林派生收缩0、10、18，最后等权合并为96树。0对10是主比较，18是事后提出的探索性对照，不搜索参数或更换主模型。

原公司身份在外层重复记录中保留。森林仍按已有 cluster_bootstrap 抽取出现过的独立公司组，再带入每组的外层重复记录。这是明确冻结的嵌套公司扰动；不会把重复记录算成新的独立退市支持。全部解释采用原始完整训练集的共同边际乘积参考。

复现入口：先运行本目录 prepare.py 模块创建唯一私有输入；完整测试并提交协议及代码；运行 schedule.py；通过 hetero-run 执行 pilot.jsonl，核验后执行 formal.jsonl；最后运行 analyze.py 和 report.py。全部脚本作为 experiments.post_st_shrinkage_stability_v1 下的 Python 模块调用。

run_location.json 只定位服务器私有工件。企业标签、预测、模型、函数与重抽样索引不提交。2023–2025均为已暴露队列，2025年24个月性能保持NA。跨训练扰动分位数不是总体置信区间，不能将方差下降自动解释为预测改善。

## 已完成结果

实际150/150配对任务、450次原始拟合、1800份完整表示验证。先读[RESULTS_ZH.md](RESULTS_ZH.md)，统计含义见[METHODS_ZH.md](METHODS_ZH.md)，结论边界见[METHOD_DECISION_ZH.md](METHOD_DECISION_ZH.md)。全部年度/原因/时限/重复记录在METRICS_ALL.csv，分布在METRIC_DISTRIBUTIONS.csv，配对差值在PAIRED_PERFORMANCE_DIFFERENCES.csv。预测和函数稳定性分别见PREDICTION_STABILITY.csv与FUNCTION_STABILITY.csv。

## 运行命令

在独立的新工作副本中创建新运行，不覆盖本次run_location.json或结果。prepare.py采用独占创建；已有定位文件时会拒绝覆盖。Python环境与完整测试所需R及历史ESWA依赖路径见ENVIRONMENT.json。

```bash
python -m experiments.post_st_shrinkage_stability_v1.prepare
python -m pytest -q
# 提交冻结协议和代码后：
python -m experiments.post_st_shrinkage_stability_v1.schedule
# 将新私有路径中的pilot.jsonl交给hetero-run，核验后执行formal.jsonl。
# 不重复试运行：它已计入150个任务。
python -m experiments.post_st_shrinkage_stability_v1.analyze metrics
python -m experiments.post_st_shrinkage_stability_v1.analyze functions --year 2023
python -m experiments.post_st_shrinkage_stability_v1.analyze functions --year 2024
python -m experiments.post_st_shrinkage_stability_v1.analyze functions --year 2025
python -m experiments.post_st_shrinkage_stability_v1.analyze collect
python -m experiments.post_st_shrinkage_stability_v1.report
```

私有snapshot.json保存拟合代码提交及所有输入、任务清单哈希；COMPUTE_RECEIPTS.json保存两个Slurm任务的回传核验；MODEL_HASHES.csv包含全部1800份模型及函数文件的哈希。FAILED记录包括一次环境预检和一次注册文件保护中止，均已解决，无真实拟合失败。原CURRENT_POST_ST_MODEL.json及crforest实现未变。
