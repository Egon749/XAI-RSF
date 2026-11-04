# 信息组新增预测价值的匹配重训实验

研究基准为 f01e07e8855787d8832d59d33128e6b66cc11f10，冻结提交为 b2c0922a8b894c158900229d965a0f9b3293cb73。见 [中文结果](RESULTS_ZH.md)、[论文判断](METHOD_DECISION_ZH.md) 和 [时点核验](SOURCE_AND_TIME_AUDIT.json)。

七项审计字段未能全部满足预测前真实披露要求，含审计的正式比较停止；旧的完整96树仍可复现，另存有明确标识的历史汇总。此目录只对能够使用相同企业及冻结时钟的财务和市场输入做新拟合，不得利用本目录的历史完整模型行拼接被阻断的四格比较。

## 主要表

- FEATURE_SETS.csv 将74个原始索引、实际列位置、八个既有经济类别和本轮信息组逐一映射。F有45列，A有7列，M由21个市场数值及1个覆盖字段Q组成。完整归因与删除输入重训是两种不同度量。
- PREDICTIONS_AND_METRICS.csv 是预测评价的**汇总**。seed=0 表示三份公司预测先平均；真实公司预测、Top20名单仅留私有目录。风险、Brier与校准差按0–1计。原因1先最终退市，原因2先恢复。2025年24个月预测有值，性能无定义。target_events 对该窗口仅表示已观察数量。
- INCREMENTAL_VALUE、FACTORIAL_COMPLEMENTARITY、PAIRED_DIFFERENCES_95、YEAR_STABILITY、SEED_SENSITIVITY 覆盖全部预设对照，阻断时有具体原因。negative_brier 是负损失效用；正差表示Brier下降。OE和校准差直接相减，不作越大越好排序。各条件区间固定模型，按年公司等概率有放回抽样，1000次，不包含模型训练不确定性。
- TOP20_COMPARISONS、CALIBRATION_DIAGNOSTICS 为逐年名单变化和风险四分位汇总。按每年向上取整的20%名额选取，分数并列沿用冻结样本ID排序。合并时合计已分配年度名额。分位层是描述，没有用结局选择切点。
- MODEL_CONFIG_AUDIT、FEATURE_COVERAGE、COST_AND_SEARCH、LEAF_SUPPORT、INTERPRETABILITY_COMPLEXITY 记录实际模型维数、树候选数、搜索、事件支持和函数规模；完整数据与模型、节点和阈值均保存在私有运行。
- REGULATORY_CHANNEL_AUDIT 按既有终局公告归类，含未知及同一事件同时触及价格／市值的情形。联合触及归入低价一栏，不重复计入市值。低于5个事件的原因组不披露该组平均概率或其排序差贡献，也不估计细分AUC。配对数差贡献是各事件类别相对全部同年度可比较事件—非事件对的加权净变化。具体规则和归因局限见报告。
- COHORT_AUDIT、INDEPENDENT_CHECKS、EXACT_RECONSTRUCTION_AUDIT、HISTORICAL_M3_HASH_AND_COMPLEXITY 及执行回执保留核对证据。TRAIN_CROSS_GROUP_CORRELATIONS、INFORMATION_OVERLAP、AUDIT_ENTRY_TYPE 是历史训练样本的描述性关联。

## 服务器重现

需能读取原冻结私有源与两节点共享存储，在新的独立工作副本与新建空私有运行目录中执行。将本实验对应副本的 run_location.json 的 private_run 设为该空目录，避免碰触原工件。使用原 XAI 环境，OPENBLAS_NUM_THREADS 和 OMP_NUM_THREADS 固定为1，R测试沿用原 CRFOREST_RSCRIPT 和 EW_RLIB。源审计脚本针对本次证据快照，来源或门槛一旦变化，应另行冻结，不能把文件存在等同于准入。

```bash
m=experiments.post_st_information_increment_v1
python -m "$m.audit"
python -m pytest -q
python -m "$m.schedule" fit
python -m "$m.schedule" ensemble
python -m "$m.metrics"
python -m "$m.diagnostics"
python -m "$m.channels"
python -m "$m.historical"
python -m "$m.independent"
python -m "$m.figures"
python -m "$m.report"
python -m pytest -q
```

训练工人只接收本年训练快照及评价X与sample_id；评价标签存放在独立目录。合并导出复用同一年度的三个已拟合成员，不生成额外树。Slurm结果核对任务完成、回传和逐文件哈希。已有森林、数据和公开报告均未覆盖。
