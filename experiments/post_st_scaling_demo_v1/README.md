# 原预测、归一化评分、固定3倍的对照

本轮为用户要求的两种变换展示，原森林不变，没有训练或校准拟合。沿用之前举例的3倍，在本轮计算误差前固定，不搜索更有利倍率。归一化在每年度、每期限的整批风险上做min-max，输出0—100分，明确它是批次内评分。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.post_st_scaling_demo_v1.run \
  --output /path/to/a/new/unique/private/run
```

命令拒绝已存在的输出目录。实际私有路径见execution_receipt.json。全部输入与原模型哈希保留，原始预测与当前参考指针不变；脚本实际运行8.84秒，属于本地小型检查，未启动Slurm或新森林任务。完整仓库测试另行运行。

- RESULTS_ZH.md、COMPARISON.csv：逐年度支持端点、误差、排序和合法性。
- POOLED.csv：跨年度重新计算，12月含2023—2025，24月仅2023/2024。
- RANK_CHECKS.csv：每年三种方法的完整排序一致；Top20名额向上取整。
- FUNCTION_PROBABILITY_CHECKS.csv：全函数数值恒等与概率合法性分开报告，不能混为一项。
- INTERPRETATION_ZH.md：实际收益、反向结果及为何不能直接采用。

匿名逐公司排序图、实际预测、归一化常数和派生函数保留服务器，不上传GitHub。公开CSV仅汇总。转换组raw_squared_error保留真实计算值；它描述把数值与0/1结局比较的误差，不表示转换已被验证为合法完整竞争风险模型。2025年24月预测可做代数/合法性核验，但结局性能仍NA。
