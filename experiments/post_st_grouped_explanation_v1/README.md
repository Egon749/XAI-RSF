# 经济能力分组解释审计

本目录对应 74 项输入、退市支持保护、深度 4、直接父节点 AJ 收缩 10 的 XAI-RSF。沿用种子审计提交 `6d94d3ea8aa706a7125915776dd5d45b4dbf0681`。八组映射及 200 次随机分组在 `270b03ee5db5dae19c1709bd4592482b0ffda671` 冻结。结论见 [RESULTS_ZH.md](RESULTS_ZH.md)，论文表述及证据边界见 [METHOD_DECISION_ZH.md](METHOD_DECISION_ZH.md)。

## 分组与计算

FEATURE_GROUP_MAPPING.csv 将 0–73 模型列位置和原 83 项字典的 feature_id 分开保存。冻结特征说明来自上一轮实际私有 features.json 及仓库“主模型/配置/INPUT_FEATURES.csv”，两者逐行一致。原生缺失值、训练边际乘积参考与模型结构不变。财务费用率、应计强度、应付账款结构、融资现金流等有多种经济归类；逐项理由见映射表。这里审计的是特征说明与经济归类，沿用上一轮数据版本和可得日限制。

每个原始函数恰好归入一项。A 包括组内主效应和组内二元函数，I 聚合两个不同经济组之间的所有原始二元函数，每项只计一次。C 将每个 I 各半分给两方。用户可将已保存的 CIFExpansion 对象、评价矩阵、时间、八组映射传给 grouped.py 中的 grouped_components，得到基准、A、I、C。分组只是同一预测的重排；C 的半分是核查账目的对称约定。与原始变量比较时，完整变量归因 V 同样将每个交互的一半给每个变量，从而 C 亦可由组内 V 直接相加。

## 文件及单位

| 文件 | 内容 |
|---|---|
| GROUP_MAIN_CONTRIBUTIONS / FOUR_CAPABILITIES_STABILITY / GROUP_INTERACTIONS | A、C、四能力、28 个 I 的幅度、符号、稳定性、训练公司与观察事件支持。按 0/1/2 个共享成员分层。 |
| GROUP_MODEL_DISTRIBUTIONS | 原96树、十个单成员、120组三成员、320树的逐年度、逐时限、逐原因、逐组汇总。 |
| GROUP_STABILITY / GROUP_VS_FEATURE_COMPARISON | 变量、原函数、A、C、四能力、跨组交互的企业层比较，及每项相对等规模随机映射的位置。 |
| RANDOM_MAPS / RANDOM_GROUP_BASELINE / RANDOM_SLOT_AMPLITUDES | 全部200个随机映射、每个对照分数与各槽位幅度。随机槽位只匹配大小。 |
| GROUP_CANCELLATION_AUDIT | 全部201个映射的组内原始绝对贡献总和、净值、留存和抵消。先平均三个成员的有符号原始项，再取绝对值。无总绝对贡献时比值 NA。 |
| EXACT_RECONSTRUCTION_AUDIT / EXACT_MODEL_CHECKS / INDEPENDENT_ENDPOINT_AUDIT | 全部396个年度预测器两种分组核算及独立终点校验。 |
| figures / LOCAL_CASE_BANDS | 六类公开汇总图。代表性真实企业贡献图和模型级完整 A、I、C 阵列只在私有运行。 |

所有 CSV 的贡献以 0–1 CIF 为单位，乘 100 即概率百分点。幅度是各模型在真实评价企业的平均绝对贡献，P5/P95 是指定组合的描述分位数。局部比较先在每家公司内部对定义良好的组合对取平均，再给公司等权；区分两个贡献均达到 0.001（0.1 个百分点）条件下的方向一致率与同时达到阈值的覆盖比例。相对 L1 是绝对差之和除以两个向量的绝对值总量，分母为零时记 NA。不同数量的 Top 项仅作补充，不能单凭名单重合度判断经济含义。

研究范围包括 2023、2024、2025、12 和 24 个日历月、退市与恢复。全部使用已保存的收缩版本。2025 年 24 个月只分析函数和预测表示，正式预测性能继续 NA。训练支持仅来自各年 1 月 1 日前快照中已实际观察的事件；没有假设全部训练企业的随访已经成熟。所有年度的测试结局已经暴露。

## 复现

在可读取旧私有运行的服务器上，使用独立工作副本和空的私有输出目录。保留版本化代码、FROZEN_PROTOCOL.json、RANDOM_MAPS.csv 和 FEATURE_GROUP_MAPPING.csv，把该副本 run_location.json 的 private_run 设为新目录，原源码基准和旧私有运行保持只读。使用 XAI 的 Python 环境，设置 OPENBLAS_NUM_THREADS、OMP_NUM_THREADS、MKL_NUM_THREADS 均为 1，独立 R 检验沿用源实验的 CRFOREST_RSCRIPT 和 EW_RLIB。

```bash
mod=experiments.post_st_grouped_explanation_v1
python -m "$mod.stage"
python -m "$mod.schedule" verify
python -m "$mod.collect"
python -m "$mod.schedule" analysis
python -m "$mod.support"
python -m "$mod.publish_tables"
python -m "$mod.independent"
python -m "$mod.check_controls"
python -m "$mod.figures"
python -m "$mod.report"
python -m "$mod.stage" post
python -m pytest -q
```

两节点按年度、模型和映射区块并行，只在结果已回传后汇总。固定数据、固定原始函数，每个随机映射只改变分组标签。长期存储以原始函数、冻结字典和 grouped_components 为完整时间表示；私有运行内另存原始项终点缓存和 re-export 的 A、I、C。核验实际覆盖全部原生时间和两个原因。全粒度原始函数保存在上一轮私有运行。
