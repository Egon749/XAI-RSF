# 具名解释结果与函数形状

按用户2026-10-27明确要求发布。读取六个冻结coverage模型（V19/V20×种子42/101/202），不重新训练，不修改原预测或实验结论。函数值已获发布授权，切点以训练分位位置表示；企业明细、真实切点、个体预测和原始森林文件不发布。

## 按要求阅读三项结果

1. **[稳定入选主效应名称及3、6、12个月阶梯图](MAIN_EFFECTS_ZH.md)**：15个变量，六模型均入选。每图分别展示两折、三个种子和三个时间截面。缺失状态单列；缺失点较大的函数另有非缺失区间放大图。共同入选不等于函数形状稳定。
2. **[稳定入选交互及贡献占比](STABLE_INTERACTIONS_ZH.md)**：8个变量对，六模型均入选。报告相对全部交互及全部函数的两种绝对贡献份额，含每折三个种子的均值和范围。
3. **[全部75变量的重要性与跨种子范围](FEATURE_IMPORTANCE_ZH.md)**：主排序为参考样本上的平均绝对主效应。未选中的变量保留并计零，不只列TopK。

当前“稳定”是严格的**六模型共同选择**定义，不是另做了稳定性筛选、显著性检验或重新训练。函数值的跨种子差异在图中完整呈现；它们不一定方向相同。15个主效应和8个交互只是共同选择部分，不是整个森林全部函数；不能仅用它们重构完整预测。

## 参考样本、时间和单位

- 每个训练折取固定512行，索引为`np.linspace(0, n-1, 512, dtype=int)`，与此前只读解释审计一致。不是随机抽取的新样本或独立验证集；三个种子共用同一折参考行。训练所得边际乘积参考分布不变。
- 图中3、6个月明确使用**90、180日近似展示**；12个月图为365日，同时在数值文件提供366日。它们是函数时间截面，不重新定义原研究12个日历月预测与评价。若要严格对应某个季度原点的日历月，应使用该原点的确切天数，不能把90/180日当作所有原点都相同的日历月跨度。
- 所有贡献针对**目标ST原因1的CIF**。JSON/CSV保存概率尺度，图与Markdown幅度表乘100为百分点尺度。负贡献合法，不是负概率，也不是因果效应。
- 横轴分位切点为`100 * mean(X_observed <= actual_cut)`，按各训练折非缺失样本计算，不包含缺失值。数值区间上端点包含，最后另有缺失状态。不同折的相同分位位置可能对应不同真实阈值。分位变换只改横轴展示，不改变函数值。
- 某些短期市场变量的缺失状态贡献较大。非缺失阶梯曲线和NA点必须分别阅读，不能直接把全部贡献归为实测市场走势。
- 三个种子的最小—最大值是条件算法波动范围，不是置信区间；六模型总体范围还混合了年度变化。
- 数据时点、质量默认值和随访限制仍为PROVISIONAL；发布函数不等于统计可辨识性或独立方法创新已成立。

## 指标公式

对固定参考样本，主效应重要性为：

\[
A_j(t)=\frac{1}{n}\sum_i|f_j(t,x_{ij})|.
\]

交互强度及主要份额为：

\[
B_{jr}(t)=\frac{1}{n}\sum_i|f_{jr}(t,x_{ij},x_{ir})|,
\qquad s_{jr}(t)=B_{jr}(t)/\sum_{a<b}B_{ab}(t).
\]

另给相对全部主效应与交互绝对强度的份额，分母为`sum(A)+sum(B)`，不含基准。先逐模型计算份额，再跨模型平均，未将8个共同交互归一化为100%。份额不是事件概率占比或删项后的预测损失。

变量CSV还提供两种辅助指标：

- `equal_pair_allocated_mean_abs`：先计算`f_j + 0.5 * sum incident f_jr`，再取平均绝对值，采用交互对两变量等分的约定。
- `absolute_magnitude_score`：`A_j + 0.5 * sum incident B_jr`，先取绝对幅度再相加，与上项不等价。

这两项与主效应平均绝对贡献不同；均不是置换/删除变量后的性能重要性。此前会话对全部验证记录的综合幅度排名使用不同参考样本和计分，不应与本发布的训练参考主效应排名直接混用。

## 数值文件与代码

|文件|内容|
|---|---|
|[stable_functions_percentile.json](stable_functions_percentile.json)|六模型的15个共同主效应、8个共同交互的完整区间函数值，90/180/365/366日；分位切点、缺失状态、基准及模型来源哈希|
|[selected_functions.json](selected_functions.json)|六模型各自实际使用的全部变量和变量对名称|
|[feature_importance_by_model.csv](feature_importance_by_model.csv)|全部75输入×6模型×4时点，含未入选零项|
|[feature_importance_summary.csv](feature_importance_summary.csv)|每折三个种子的均值/最小/最大，以及六模型等权汇总|
|[interaction_importance_by_model.csv](interaction_importance_by_model.csv)|全部107种交互并集×6模型×4时点，缺席模型计零|
|[interaction_importance_summary.csv](interaction_importance_summary.csv)|全部交互强度及两种份额的均值和范围|
|[stable_interaction_shares.csv](stable_interaction_shares.csv)|8组共同交互的全部时点份额及范围|
|[export_receipt.json](export_receipt.json)|输入/函数/模型哈希、只读检查、定义及重构核验|
|[render_receipt.json](render_receipt.json)|每张SVG及渲染源的哈希|
|[validation.json](validation.json)|份额、均值/范围、文件边界、图形及完整测试核验|

函数JSON的`values[cell_index][time_index]`按`spatial_shape`的C顺序排列；每个轴的最后一格是缺失状态。它没有原始切点，且只导出共同选择部分，不能作为原始输入上的完整预测模型。

导出入口：[export_review_functions.py](../export_review_functions.py)。该程序只读取冻结训练X、函数及模型哈希，不调用fit，不读取结局标签或2022—2024数据。输出目录必须不存在。

绘图入口：[render_review_functions.py](../render_review_functions.py)，只需公开JSON/CSV和NumPy/Matplotlib即可重绘，不需要私有企业数据。原始参考样本不在GitHub中，因此外部审阅者可复算逐种子汇总、查阅函数形状，不能仅凭本仓库重新计算基于私有参考行的贡献均值。

旧版“全部函数参数保留服务器”的说明反映当时授权。本次例外仅覆盖用户指定的具名结果和分位坐标函数；没有上传公司级数据或原始森林。
