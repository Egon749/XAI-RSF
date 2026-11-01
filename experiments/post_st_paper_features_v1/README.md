# 论文可映射20变量：年度12个月匹配实验

用户上传Liu等《Progressive gradient boosted trees for imbalanced financial distress prediction》（ESWA321,132187, DOI10.1016/j.eswa.2026.132187），并明确选择先运行能核验的子集，记录缺项和口径差异。

论文表2列54项；本实验映射20列，34项未纳入。映射依据名称和已有可审计计算口径，不代表作者原代码或完全相同定义。X1与X2在旧输入中重复；利息保障倍数、周转率分母、收入与利润范围、现金比率等差异逐项记录。X53将原自然对数市值反变换为千元总市值。无根据预测成绩筛选变量，无占位全缺失列。

本轮仍预测首次财务/审计ST后先最终退市或先恢复。每年1月1日之前的信息训练，评价当年新入组企业各自12个日历月结局，三种子平均后计算指标。训练24个月上限及旧conservative输入保持。新增9个q2/depth4/shared/parentAJ10森林，复用18个83/74列参考森林。有效每树候选数20；其与原25/83、25/74不同，不能将全部差异归因于变量本身。

这不是完整54变量实验，也不是PROBoost复现，不能与论文ST分类AUC直接比较。结果、协议、变量映射、环境和哈希公开；论文原文件、企业数据、模型参数、切点与预测仅在私有运行目录。

中文变量清单见[VARIABLE_SUBSET_ZH.md](VARIABLE_SUBSET_ZH.md)，实际年度比较见[RESULTS_ZH.md](RESULTS_ZH.md)。

```bash
python -m experiments.post_st_paper_features_v1.prepare --stage prepare --run "$RUN"
# 完整测试（含R）通过、代码协议提交并工作区干净后
python -m experiments.post_st_paper_features_v1.prepare --stage freeze --run "$RUN"
/home/oliver/.local/bin/hetero-run "$RUN/formal.py"
python -m experiments.post_st_ablation_v1.stages --stage formal --run "$RUN" --job "$JOB" --report "$REPORT"
python -m experiments.post_st_paper_features_v1.analyze --run "$RUN"
python -m experiments.post_st_paper_features_v1.deliver --run "$RUN"
```
