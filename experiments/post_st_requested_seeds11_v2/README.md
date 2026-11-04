# 11个指定种子组合实验（第二轮352树）

本轮固定98、17、503、5001、2570、22、155、213、150、6720、8679。使用上一轮17种子实验的相同年度快照和已保存的原始、收缩版模型，每个种子32树。按企业平均11个成员的两个原因CIF。每个年度单独导出352树预测器和精确主效应、二元交互函数。三个年度共6项独立导出任务，Slurm使用共享输入和独立私有输出。

协议保留74变量、退市支持保护、q≤2、最大深度4、共享分位数切点、25项均衡候选池、直接父节点AJ收缩10和训练随访24个日历月。评价2023—2025年12个月、2023—2024年24个月，2025年24个月随访不足记NA。原96树用于固定预测配对区间，上一轮544树用于描述本轮名单和树数改变后的实际差异。本轮属于探索性评价。

```bash
python -m experiments.post_st_requested_seeds11_v2.prepare
# 完整测试含独立R通过后，提交并冻结协议及代码
for stage in freeze stage_ensemble ensemble; do
  python -m experiments.post_st_requested_seeds11_v2.run "$stage" || exit 1
done
python -m experiments.post_st_requested_seeds11_v2.cpu_placement
python -m experiments.post_st_requested_seeds11_v2.analyze
python -m experiments.post_st_requested_seeds11_v2.compare_previous
# 核对失败记录，保存 INTERPRETATION_ZH.md 后生成报告
python -m experiments.post_st_requested_seeds11_v2.report
python -m experiments.post_st_requested_seeds11_v2.publication_check
```
