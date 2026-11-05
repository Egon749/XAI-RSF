# 父节点收缩强度有限实验

研究公式为 w=m/(m+lambda)。m 是叶节点独立袋内公司数；lambda=0 保留原叶 AJ，数值越大越靠近原始直接父节点 AJ。同一权重用于两个原因和全部时间。当前参考值10与当前指针均保留。

冻结十个值：0、1、2、5、10、20、30、50、100、200。当前 C039 的2023—2025三种子原始森林直接复用；只补2020—2022相同配置的9个历史验证森林。先按各外层截止日自己的训练快照做前向选择，再读取外层结果。事件支持很少，所有选择与已暴露年度比较均为探索性。

复现入口（仓库根目录）：

```bash
python -m experiments.post_st_shrinkage_strength_v1.prepare
python -m experiments.post_st_shrinkage_strength_v1.schedule fit
hetero-run PRIVATE_RUN/fit.jsonl
python -m experiments.post_st_shrinkage_strength_v1.schedule derive
hetero-run PRIVATE_RUN/derive.jsonl
python -m experiments.post_st_shrinkage_strength_v1.analyze history
# Commit the historical selection tables before reading outer results.
python -m experiments.post_st_shrinkage_strength_v1.analyze outer
python -m experiments.post_st_shrinkage_strength_v1.report
```

应先复制 run_location.json 到一个新的唯一私有运行目录；prepare 为排他创建，不能覆盖已完成运行。schedule 要求已提交代码，输入快照在作业期间不可修改。使用已有两节点共享存储及逐任务输出回传。真实公司、原模型、派生模型、预测、完整函数表均在 run_location.json 指向的私有目录。公开仅包含汇总、代码、协议和哈希。

解释验证覆盖全部训练和评价行、原生时间点及两个原因；误差上限1e-10。2025年24个月性能NA；不截断预测、不拟合校准器、不修改森林结构、不自动替换参考模型。

结果入口为 RESULTS_ZH.md，简表 AUC_COMPARISON.csv；METRICS_ALL.csv 含全部强度、两原因、12/24个月、单种子和先平均预测后的结果。historical_selected 指各截止日前固定规则选出的强度，区别于回看评价结果后寻找最高值。方法判断见 METHOD_DECISION_ZH.md，曲线见 figures/。
