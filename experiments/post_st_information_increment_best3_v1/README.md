# C039 财务、审计和市场信息四格实验

[完整中文报告](RESULTS_ZH.md) · [论文判断](METHOD_DECISION_ZH.md) · [来源与披露时钟](SOURCE_TIME_AUDIT.json) · [六组图表](figures/)

本实验基于 f01e07e8855787d8832d59d33128e6b66cc11f10，冻结提交 65f66b3ce2acaa8327493affbcca87a71bcb7e4c。C039的成员固定为17、503、22。M1和M3按指定的数据库字段进行回顾性比较，时钟限制沿用原证据。不得将回顾性已计算数值作为完整的实时披露认证。主配置固定每树25项，比例配置固定15/18/23/25，比例配置的M3完全复用主配置的同一预测。本轮没有加载原42/101/202森林做主实验、种子选择或对照。

- METRICS_ALL / CALIBRATION_DIAGNOSTICS：seed=0 为逐企业三成员等权平均；按年/合并估计概率均值、发生率、O/E及预测减观察的校准差。模型为M0财务、M1财务和审计、M2财务和市场、M3三者全部。年度等预算对照和两个原因均保留。2025H24只允许输出预测，无正式性能。个别年度原因无事件时相应AUC/AP保留NA。
- INCREMENTAL_VALUE / FACTORIAL_COMPLEMENTARITY / PAIRED_DIFFERENCES_95：含五个同样本配对对照。negative_brier 是负Brier效用，正差表示原损失减小。AP/AUC的pooled值重新计算。within_year_auc采用年内事件—非事件配对数加权；Top20名额先按年确定再合计。单种子和年度方向分列。未定义抽样次数在区间表，1,000次公司重抽样不校正事后挑选C039。
- FEATURE_SETS / MODEL_CONFIG_AUDIT / FEATURE_COVERAGE / GROUP_COVERAGE / COST_AND_SEARCH / LEAF_SUPPORT / INTERPRETABILITY_COMPLEXITY：输入、真实入树资格、实际分裂、支持度、运算和函数规模。M3标为shared_both，因为两种候选预算完全相同。
- M3_MEMBER_HASHES / M3_ENSEMBLE_HASHES / RECONSTRUCTION_AUDIT / INDEPENDENT_CHECKS：9个成员的源哈希、3个合并预测器的版本及数值验证。新模型和预测哈希列在INTERPRETABILITY_COMPLEXITY。概率与贡献均以0–1 CIF为原始单位。
- SUBGROUP_ATTRIBUTION / MARKET_SUBGROUP_MAPPING：冻结的市场子组模型内贡献；先把每个跨变量原始交互各半分给两方，再按组加总。M_numerical与M_broad是额外合计行，不重复相加。不能把平均绝对贡献当作重新训练子组带来的性能增益。
- REGULATORY_CHANNEL_AUDIT：基于实际终局证据拆解M2−M0的排序/捕获变化，事件数较少的组不披露平均预测值。双重交易原因只计一次，未确定者保留。标签审查未供训练工人使用。

## 重现

使用独立工作副本与新建空私有目录，把副本run_location.json的private_run指向该目录。保留源分支、旧私有运行和旧结果只读。需要原实验的XAI Python环境和两个Slurm节点，OPENBLAS_NUM_THREADS、OMP_NUM_THREADS、MKL_NUM_THREADS设为1；完整测试使用原项目CRFOREST_RSCRIPT及EW_RLIB。脚本使用冻结的服务器来源路径，私有输入与企业记录不分发到GitHub。

```bash
m=experiments.post_st_information_increment_best3_v1
python -m "$m.audit"
python -m pytest -q
python -m "$m.schedule" fit
python -m "$m.schedule" ensemble
python -m "$m.metrics"
python -m "$m.diagnostics"
python -m "$m.attribution"
python -m "$m.channels"
python -m "$m.independent"
python -m "$m.materialize"
python -m "$m.figures"
python -m "$m.report"
python -m pytest -q
```

训练工人输入只有训练折、评价X与标识、配置和核验摘要。仅结果确认回传并校验哈希后才运行下一阶段。materialize从不可变的三个成员生成完整森林序列化文件并核验重载，不拟合新树。审计年份身份与源散列应与原运行相同。文档中的种子、时限、目标原因和子组边界不得根据重现实验结果更改。
