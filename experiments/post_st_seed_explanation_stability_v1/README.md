# 十选三随机种子解释审计

结果见 [中文报告](RESULTS_ZH.md) 和 [论文判断](METHOD_DECISION_ZH.md)。基准提交 0a19aaeaf01def20a0a7cbd866b41a6e4486ea02。冻结提交 4f1b163637f8eb5027dbe7d1b54595accfd431b0。对十个已有成员作有限枚举，2023–2025 均属已暴露队列。

## 文件与单位

- COMBINATIONS_120.csv 是组合键；ALL_METRICS.csv、METRIC_DISTRIBUTIONS.csv、PREDICTION_STABILITY.csv 给出两原因、两个时间窗、两种叶估计的逐组合值及分布。同一年度的相同企业比较，2025年24个月仅查看模型输出。跨组合的5%和95%是描述性分位数。
- STRUCTURAL_FREQUENCIES.csv 含未选择变量及变量对；STRUCTURE_JACCARD.csv 按共享成员数报告。MAIN_EFFECT_STABILITY.csv、INTERACTION_STABILITY.csv 和 SUPPORT_AUDIT.csv 提供逐项的全网格存在性、实证联合支持与外推区的函数差异、正贡献区域和事件支持。恒定函数无法计算形状相关。形状相关和符号指标分别按可计算的组合对以及共同达到幅度阈值的坐标定义；both_material_mass 为两函数同时达到阈值的坐标权重。低支持门槛为10家独立企业。
- LOCAL_EXPLANATION_STABILITY.csv 先计算单家公司跨组合对中位数，再汇总公司分布。LOCAL_TERM_VARIABILITY.csv 含每项在同一企业的组合间标准差的汇总。顶部贡献按绝对值排序，符号单独测量，全部局部项用共同项字典、未出现记为零。所有分解及概率以0–1 CIF单位存储；图与报告中标明“百分点”的数值已乘100。
- COMPLEXITY_TRADEOFF.csv、FIXED_ANCHOR_COMPARISON.csv 包括各年度和配置。后者采用无共同成员的42/101/202锚点；它同时是单个固定预测器，比较距离并不证明该预测器代表真值。各模型以训练边际为背景，形状稳定不代表变量独立。编译耗时与验证耗时单位均为秒，function_bytes 为实际压缩存储量。
- 图1–6的 PDF 和 PNG 位于 figures/。函数带是组合分位数；每个点对应一个有至少5家企业的训练区间，省略原始切点。局部图用企业组展示。按冻结规则选出的真实个案在私有运行的 case_figures_PRIVATE/；完整个案贡献、完整二维网格的独立事件计数、公司跨组合极差与Top20频率均位于私有运行。逐公司的名次不发布。
- RECONSTRUCTION_DETAIL.csv、RECONSTRUCTION_AUDIT.json、INDEPENDENT_AUDIT.json、MODEL_HASH_AUDIT.json、execution_receipt.json 记录数值检查、模型来源、版本和校验信息。

## 重现

需要有权限读取原私有运行的服务器、公共源码、原 Python/R 环境及共享 Slurm 池。在新的工作副本中从上述基准开始，把本目录的 Python 文件和 FROZEN_PROTOCOL.json 复制到同名实验目录。为该副本的 run_location.json 写入新建空私有目录的绝对路径，键名 private_run。保留原运行和 Git 中的已交付结果。以下从该副本仓库根目录运行。使用已有 XAI Python 环境，设置 OPENBLAS_NUM_THREADS、OMP_NUM_THREADS、MKL_NUM_THREADS 为 1。独立 R 测试使用原实验的 CRFOREST_RSCRIPT 和 EW_RLIB。

```bash
module=experiments.post_st_seed_explanation_stability_v1
python -m "$module.stage"
python -m "$module.schedule" verify
python -m "$module.predict"
python -m "$module.collect"
python -m "$module.schedule" explain
python -m "$module.benchmark"
python -m "$module.independent"
python -m "$module.assemble"
python -m "$module.figures"
python -m "$module.report"
python -m "$module.stage" post
python -m pytest -q
```

stage 会比较每个历史来源的完整训练、评价、标签和输入说明哈希，校验72套源表示，隔离复制所需文件。verify 组装森林、重新导出完整函数，按成员缓存显式逐项和后取等权平均；independent 还在18个预定的合并导出上直接对每个函数求和。explain 在6个年度及版本分片中计算120组合的形状与局部稳定性；同一有序网格的训练公司与目标事件按真实记录计数。作业由 hetero-run 调度，只有任务完成且结果回传并核对后才进入下一阶段。既有冻结配置、模型及评价预测均只读。核验数值误差容差为 1e-10。

交付回执另记录代码和图表哈希。全部任务回传、最终全套测试完成之后，可以把测试日志复制为本目录 pytest_final.log，运行 `python -m experiments.post_st_seed_explanation_stability_v1.delivery` 在新的发布副本生成回执。原始运行保留全部任务日志和独立数值抽查。
