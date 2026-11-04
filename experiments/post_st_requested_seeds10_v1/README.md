# 10个指定种子的320树组合

固定98、17、503、5001、2570、22、155、150、6720、8679，复用三个年度共30个已保存的32树森林。与上一轮第二组11种子相比，唯一的成员变化是移出213，将其余10份预测等权平均。本轮使用冻结的74变量、退市支持保护、q2、最大深度4、均衡25变量候选池、共享分位切点、直接父节点AJ收缩10和24个日历月训练随访上限。

三个年度各自按1月1日前的信息拟合过成员模型。本次仅合并已保存的原始和收缩预测器，分别核验企业预测、全部时间与两个原因的精确分解和重载。评价2023—2025年12个月及2023—2024年24个月，2025年24个月记NA。原96树与上一轮352树分别列为参照。本轮采用已查看队列上的探索性评价。所有原始企业记录与模型明细留在私有目录。

```bash
python -m experiments.post_st_requested_seeds10_v1.prepare
# 完整测试含独立R通过后，提交并冻结代码与协议
for stage in freeze stage_ensemble ensemble; do
  python -m experiments.post_st_requested_seeds10_v1.run "$stage" || exit 1
done
python -m experiments.post_st_requested_seeds10_v1.cpu_placement
python -m experiments.post_st_requested_seeds10_v1.analyze
python -m experiments.post_st_requested_seeds10_v1.compare_previous
# 核对失败记录、保存解读后生成报告
python -m experiments.post_st_requested_seeds10_v1.report
python -m experiments.post_st_requested_seeds10_v1.publication_check
```

PREVIOUS_352_COMPARISON.csv 记录删去213前后的同队列指标及前20%名单重叠。PREVIOUS_352_PAIRED_95.csv 记录320树减352树的1000次配对公司重抽样区间。DELETION_IDENTITY.json 记录等权删除公式在所有评价公司、两个原因、两个评价时限和原始/收缩版本上的误差。
