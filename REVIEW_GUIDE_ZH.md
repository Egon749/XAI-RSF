# XAI-RSF全部已完成实验：审阅入口

**当前采用版本（2026-10-28用户决定）：原旧C输入、q2、深度4＋直接父节点AJ收缩，min_leaf10、单位时间权重、收缩强度10。** [已核对的模型引用](CURRENT_POST_ST_MODEL.json)。用户取消本轮时间加权/min_leaf20方案，已完成实验仅保留为历史证据，后续沿用原模型；[撤回采用说明](experiments/post_st_temporal_leaf_v1/DISPOSITION_ZH.md)。本次没有重新训练。

**2026-10-28已归档、撤回采用的ST后预后实验：** [训练队列AJ、时间加权叶与min_leaf20有限实验](experiments/post_st_temporal_leaf_v1/README.md)。20/20任务成功、34次新森林拟合，139项测试通过。2024的24个月权重4使Brier从0.3396降至0.3050，但AUC/AP从0.770/0.782降至0.728/0.739，Top20仍8/20；完整q2解释精确，但交互稳定性未改善。原深度4＋收缩结果完整保留。所有后期结果为描述性，2025已在历史任务中暴露，本轮未重新评价。判断INCONCLUSIVE；未拟合校准器或继续搜索。

本新增结果在 `research/xai-rsf-post-st-prognosis-v1` 分支；下文 `research/xai-rsf-development-v1` 说明对应此前首次ST预警研究。两类任务、终点和样本不可混用。

**最新：可解释基线、结构恢复与SHAP配对审计。** [完整结果和图](experiments/interpretable_baselines_v1/README.md)：33个真实数据拟合、210个模拟拟合、18个SHAP任务全部完成。q2在8次共同公司bootstrap下部分归因更稳，但模拟发现真实二元信号在等权森林中明显稀释；EBM固定轮数训练严重过拟合，不能据此证明优于强EBM基线。双方都可精确加回预测。正式判断仍为INCONCLUSIVE，后暴露探索性。

**此前q3/q4匹配实验：** [全部结果与解释代价](experiments/order_q3_q4_v1/README.md)：沿用制造业输入和q2设置，仅改变路径阶数，各3种子。完整高阶解释保持精确；q4合并AUC/AP为0.8441/0.0949，对照q2为0.8422/0.0838，空间格为q2的63.9倍，高阶交互集合稳定性较弱。q3/q4未取得跨年、跨指标一致收益，不自动替换q2。**本轮为明确授权的后暴露探索性算法比较**，不是独立后期验证；正式判断INCONCLUSIVE/PROVISIONAL。

**此前制造业限定重训：** [制造业实验](experiments/manufacturing_2023_2024_v1/README.md)新增6个最终模型，沿用固定配置，2023/2024分别评价。在相同制造业样本上，相对原全行业训练q2，AUC分别提高0.0181/0.0074，AP分别下降0.0070/0.0127，Top20基本不变。完整解释最大误差2.220e-16。行业历史版本日未认证，INCONCLUSIVE/PROVISIONAL；不能将评价子集变化当作重训效果。

本入口于2026-10-27整理，供组会及独立审阅（包括Claude）使用。最初交付只整理开发期结果；随后用户明确授权2023/2024分别评价，新增6个冻结最终模型，该阶段未根据后期结局调参或筛种子。之后新增制造业重训和本次q3/q4探索；本次不能继承“无后期模型开发”的表述。

**此前全行业后期评价：** [2023/2024完整结果](experiments/temporal_2023_2024_v1/README.md)。沿用原最终训练原点2012—2020、拟合截止2022-01-01；2023与2024共用相同模型。当前q2的年度AUC约0.827，M2约0.860，AP及Top20覆盖存在明显差距；最高风险十分位随访/源资格缺失集中。完整解释仍精确，正式结论INCONCLUSIVE，数据PROVISIONAL。这是已接触数据的回溯时间外评价，不是严格盲测。下文V19/V20表格属于此前开发阶段，不能当作本次后期结果。

仓库：`Egon749/XAI-RSF`；结果分支：`research/xai-rsf-development-v1`。请读取此分支，不能只读取main。当前讨论的是XAI-RSF森林，不是早期newRSF2风险率回归模型。

**后续明确授权的具名解释发布：** [15个共同主效应的3/6/12个月阶梯图、8个共同交互的份额、全部75变量的重要性与种子范围](experiments/search_coverage_v1/function_release/README.md)。切点用训练分位位置表示，函数值和各种子形状可直接审阅。稳定仅指六模型共同入选。90/180日为3/6个月近似展示；365/366日数值均保留。

## 1. 当前模型与结论

当前本轮表现最好的可解释配置是`search_coverage_v1/coverage`：q=2路径约束、原共享五切点字典、平衡树级变量子集。75个输入，每棵树25个候选输入，32棵树，最大深度6。保留公司组bootstrap、logrankCR分裂、叶节点Aalen–Johansen CIF及等权森林平均。

“当前配置”是有限开发实验中的描述性选择，不是完成外部验证的最终模型。它不启用全森林硬分区预算，也没有风险信息矩阵或其他可辨识性惩罚。正式研究结论为**INCONCLUSIVE**，数据仍为**PROVISIONAL**。

## 2. 全部实验按时间顺序阅读

|阶段|正式任务|主要内容|该阶段冻结结论|
|---|---|---|---|
|[首次实验](experiments/development_v1/METHOD_INNOVATION_DECISION_ZH.md)|8个企业模型、90个合成模型，另2个试运行|不限阶数、深度限制、路径q2、q1及精确解释核验|PREDICTIVE_NO_GO，仅针对该轮冻结配置|
|[共享切点与全森林预算](experiments/shared_partition_v1/README.md)|30个企业模型，另5个试运行|5版本×2折×3种子|INTERPRETABILITY_ONLY_GAIN；预测保留目标未达成|
|[变量覆盖与切点细化](experiments/search_coverage_v1/README.md)|30个企业模型，另5个试运行|2×2消融加同后端不限阶数参照|INCONCLUSIVE；覆盖有开发期收益，联合门槛未通过|
|[只读解释审计](experiments/search_coverage_v1/EXPLANATION_SIMPLIFICATION_ANALYSIS_ZH.md)|读取6个冻结coverage模型，无训练|精确压缩机会、变量对候选暴露、交互贡献集中程度|未实施压缩、删项、修改预测或新训练|
|[2023/2024后期评价](experiments/temporal_2023_2024_v1/README.md)|6个最终模型，另2个试运行|固定coverage和同后端参照各3种子，逐年评价，M2复用|INCONCLUSIVE；AP/预警保留未达成，未知高风险结果集中|
|[制造业限定重训](experiments/manufacturing_2023_2024_v1/README.md)|6个制造业最终模型，另2个试运行|相同制造业子集比较新拟合与原全行业模型，逐年评价|INCONCLUSIVE；q2 AUC提高但AP下降，Top20基本不变；行业时钟未认证|
|[q3/q4有限匹配实验](experiments/order_q3_q4_v1/README.md)|6个制造业新模型，另2个试运行|仅改变路径阶数，q2及原参考预测复用，2023/2024分别评价|INCONCLUSIVE；有限AP/Brier收益，AUC/覆盖不一致，复杂度大幅增加；后暴露探索性比较|
|[可解释基线/结构恢复/SHAP审计](experiments/interpretable_baselines_v1/README.md)|33个真实拟合、210模拟、18解释计算|EBM-GAM/GA²M、共同公司bootstrap、精确有限背景Shapley参考|INCONCLUSIVE；EBM过拟合限制强基线结论，q2部分归因更稳但交互幅度被稀释|

各轮独立保留，不把旧轮“格数下降96.7%”、新轮AUC提升、不同配置的事件覆盖拼接为同一个模型结果。旧轮的失败、中断、恢复说明见各自`TRAINING_DIAGNOSTICS_ZH.md`和执行回执，不能用本轮30/30成功覆盖历史失败记录。

## 3. V19/V20开发阶段核心结果

V19训练原点2012—2017，拟合信息截止2019-01-01，验证原点2019年；V20训练原点2012—2018，拟合截止2020-01-01，验证原点2020年。完整协议及数据限制以[冻结协议](experiments/search_coverage_v1/PROTOCOL_ZH.md)和[数据审计](experiments/search_coverage_v1/DATA_ALIGNMENT_AUDIT_ZH.md)为准。

两折训练记录合计112,513，跨折去重62,049，独立训练目标事件182个；验证25,689条记录、跨年去重135个独立目标事件。最新轮30/30正式模型及5/5试运行成功。仅使用V19/V20开发结果；未使用2022—2024结局。

以下为记录等权、三个种子的指标均值（M2只有冻结预测）。合并指标逐种子在合并预测上重算，然后跨种子平均，既不是年度平均，也不是融合预测。Top20为每季度核查前20%记录后，跨年去重的独立事件数；小数来自种子平均。

|配置|V19 AUC|V20 AUC|合并AUC|合并AP|合并IPCW Brier|Top20事件/135|
|---|---:|---:|---:|---:|---:|---:|
|原共享q2 baseline|0.805740|0.775067|0.793472|0.159035|0.014896|101.67|
|当前coverage|0.868783|0.824880|0.846885|0.221193|0.014763|103.33|
|仅切点细化 refined|0.814000|0.762889|0.786431|0.119869|0.015276|101.33|
|覆盖与细化 combined|0.855408|0.824310|0.838072|0.166239|0.014791|103.33|
|同后端不限阶数|0.888835|0.860672|0.873283|0.237835|0.013785|111.33|
|冻结M2|0.900230|0.872472|0.885848|0.238536|0.013831|114|

coverage相对baseline合并AUC增加0.053413、AP增加0.062158；但相对同后端不限阶数森林，AUC低0.026398、AP保留约93%、Brier高约7.09%、Top20平均少8个事件。M2的输入编码和训练预算不匹配，所以同后端不限阶数森林是主要机制参照。该参照和coverage还存在切点及候选资格差异，不能把全部剩余差距归因于q2。

V19的coverage AP为0.303252，M2为0.302834；V20分别为0.139133和0.198632。不能只用V19或最佳种子宣称保住RSF性能。Top25合并事件数baseline为111.33、coverage为108.33、M2为121。公司等权结果同样有AUC/AP收益，但合并Brier略变差。

### 完整预测与预警证据

- [年度及合并指标，含公司等权](experiments/search_coverage_v1/V19_V20_RESULTS_ZH.md)
- [逐种子指标CSV](experiments/search_coverage_v1/metrics.csv)、[指标汇总CSV](experiments/search_coverage_v1/metrics_summary.csv)
- [逐种子预警指标](experiments/search_coverage_v1/warning_metrics.csv)、[预警汇总](experiments/search_coverage_v1/warning_summary.csv)
- [校准汇总](experiments/search_coverage_v1/calibration.csv)
- [配对差值](experiments/search_coverage_v1/paired_contrasts.csv)、[2×2开关交互](experiments/search_coverage_v1/factorial_interaction.csv)
- [冻结门槛逐项结果](experiments/search_coverage_v1/acceptance_gates.csv)、[完整研究判断](experiments/search_coverage_v1/METHOD_INNOVATION_DECISION_ZH.md)

## 4. 完整解释、复杂度与稳定性

|配置|平均主效应数|平均交互数|平均空间格|
|---|---:|---:|---:|
|baseline|12.00|11.00|277.67|
|coverage|25.17|40.33|1232.50|
|refined|9.67|9.17|556.50|
|combined|25.83|45.00|3792.67|

均值覆盖2折×3种子。全部24个二阶模型最大完整重构误差4.996e-16。coverage空间格范围1031—1348，均超过1000格可读性参考；当前轮没有在训练时实施1000格硬限制。空间格不是所有时间×原因标量数，coverage完整标量表最大622,262项。

coverage跨种子交互集合Jaccard：V19由baseline的0.784降为0.292，V20由0.518降为0.283。共同选择交互的函数相关性只评价共同项，不能替代集合稳定性或公司重抽样稳定性。当前只能说解释完整精确，不能说解释已更简单、更可靠或统计可辨识性已解决。

- [解释比较](experiments/search_coverage_v1/INTERPRETABILITY_COMPARISON_ZH.md)、[复杂度CSV](experiments/search_coverage_v1/complexity_summary.csv)、[逐种子对稳定性](experiments/search_coverage_v1/seed_stability.csv)
- [训练诊断CSV](experiments/search_coverage_v1/training_diagnostics.csv)、[预算诊断](experiments/search_coverage_v1/budget_diagnostics.csv)
- [只读审计说明](experiments/search_coverage_v1/EXPLANATION_SIMPLIFICATION_ANALYSIS_ZH.md)、[精确压缩检查](experiments/search_coverage_v1/posthoc_explanation_audit/compression_audit.csv)、[变量对暴露](experiments/search_coverage_v1/posthoc_explanation_audit/pair_exposure_audit.csv)、[交互贡献集中度](experiments/search_coverage_v1/posthoc_explanation_audit/interaction_strength_audit.csv)

只读审计没有发现可直接删除的整轴相邻完全相同切片或严格零函数；这不证明所有等价压缩都不可能。平均绝对贡献占比不是风险概率占比，也不是删掉某交互后预测损失的估计。

## 5. 模型方法与代码对应

- [crforest/core.py](crforest/core.py)：公司bootstrap、logrankCR分裂、路径变量约束、叶节点AJ和森林平均。API未指定mtry时默认为ceil(sqrt(p))，但本轮冻结实验明确设置mtry=75；coverage路径未用满两变量时实际评估树子集的全部25个资格变量，用满后只评估路径已用的两个变量。该更正只修正文档，不修改已训练模型。缺失状态为原生路由，不是隐式填成正常值。
- [crforest/partition.py](crforest/partition.py)：训练共享字典、平衡变量子集、可选全森林空间格预算。当前coverage使用原五切点字典；`shared_refined`是单独消融，不应合并描述为当前最佳配置。
- [crforest/functions.py](crforest/functions.py)：基于叶区域、训练边际乘积参考的精确主效应和双变量函数编译；导出预测不依赖原树遍历，不拟合替代解释器。
- [冻结配置](experiments/search_coverage_v1/frozen_protocol.json)、[输入字段](experiments/search_coverage_v1/input_fields.json)、[运行入口](experiments/search_coverage_v1/launch.py)、[训练及汇总实现](experiments/search_coverage_v1/worker.py)

每条路径的不同输入变量数不超过2，不是树深只能为2，也不是整片森林只用两个变量。各叶区域最多涉及两个坐标，其线性CIF平均满足：

\[
\widehat F_k(t\mid x)=\mu_k(t)+\sum_j f_{j,k}(t,x_j)+\sum_{j<r}f_{jr,k}(t,x_j,x_r).
\]

中心化使用训练边际乘积参考，不能假定真实变量独立。分解位于CIF尺度，不是对数风险率贡献，不是因果影响。q2确实排除真实三阶以上交互；增加树数不解除该表达约束。变量覆盖可改善搜索不足，但不扩展最高交互阶数。

## 6. 文献与创新边界（讨论性分析，不是新增实验）

以下原始文献已经覆盖低阶树集成、精确纯化和生存函数分解的重要思想，应纳入独立创新审查：

1. [Random Planted Forest，2020首稿/2023修订](https://arxiv.org/abs/2012.14563)：限制叶内交互变量数的直接可解释树集成；采用planted trees，与本项目二叉生存树不同。
2. [Lengerich等，AISTATS 2020](https://proceedings.mlr.press/v108/lengerich20a.html)：分段常数函数的精确纯化，区分主效应与交互。
3. [Hiabu等，AISTATS 2023](https://proceedings.mlr.press/v206/hiabu23a.html)：低维结构集成的精确全局/局部函数分解。
4. [Monotone GAMI-Tree，2023预印本](https://arxiv.org/abs/2309.02426)：交互选择、约束XGBoost及解析纯化。
5. [SurvFD/SurvSHAP-IQ，2026预印本](https://arxiv.org/html/2602.16505v1)：生存函数分解及时间交互；文中设定明确排除竞争风险和复发事件。

这些文献不是本项目已经跑过的实验基线。本项目相对普通RSF的结构修改，不自动等于相对文献的原创贡献。当前可主张保留RSF机制的具体实现、完整CIF分解及匹配机制实验；不能声称首次低阶可解释森林、首次精确分解、独立可辨识性惩罚有效或已优于RSF+SHAP。尚未建立独立方法创新充分证据；不以检索未发现同名方法作为首创证据。

## 7. 建议独立审阅的问题

请结合完整代码、每种子数据和不利结果独立判断，不必接受本项目现有结论：

1. q2限制是否严格执行，精确CIF展开与缺失路由是否数学及实现一致？
2. logrankCR、AJ、删失评价、时间原点和独立事件去重是否正确？现有数据审计不足会怎样限制结论？
3. 覆盖收益能归因到哪些机制，现有匹配实验有哪些仍无法区分的因素？
4. AUC/AP收益、Brier及事件覆盖损失、解释规模和跨种子稳定性，是否支持当前研究判断？
5. 相对RPF、GAMI和函数分解文献，哪些属于已有思想、具体适配、实证发现或尚未证明的原创候选？
6. 若提出后续建议，请区分仅凭现有结果能确认的结论与需要新实验验证的假说；本交付不授权自动训练。

## 8. 复核、哈希与私有数据边界

- [最新执行回执及30个模型SHA256](experiments/search_coverage_v1/execution_receipt.json)、[发布核验](experiments/search_coverage_v1/publication_verification.json)、[训练诊断及失败说明](experiments/search_coverage_v1/TRAINING_DIAGNOSTICS_ZH.md)
- [历史控制模型一致性](experiments/search_coverage_v1/legacy_control_parity.json)、[指标一致性](experiments/search_coverage_v1/legacy_metric_parity.json)
- [oliver环境](experiments/search_coverage_v1/environment_oliver.txt)、[lemon环境](experiments/search_coverage_v1/environment_lemon.txt)
- [此次交付文件清单及SHA256](review_delivery/artifact_manifest.json)、[服务器报告一致性核对](review_delivery/source_alignment.json)、[此次完整测试输出](review_delivery/tests_full.txt)

训练代码版本以各轮执行回执为准；本次文档交付提交不是拟合模型所用代码提交。哈希清单覆盖当前版本化代码、测试和实验产物，排除清单自身，既不是数据可得日认证，也不是第三方复现了真实数据训练的证明。

按项目既定约定，原始数据、公司标识、逐企业预测、原始森林、真实阈值及公司完整解释图不进入Git，仍在服务器私有运行目录中。用户随后明确授权的例外是`function_release/`中的变量/交互名称、参考样本贡献统计和分位坐标上的函数值。仓库现在可以直接审阅这些函数形状，仍不能重新计算私有逐企业预测或独立认证原始PIT数据。没有将私有目录整体上传或向Claude自动发送信息。
