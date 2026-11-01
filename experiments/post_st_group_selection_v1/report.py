from pathlib import Path
import json,shutil
import pandas as pd
from .run import REPO as r, PUBLIC as e, RUN as run
receipt=json.loads((e/'execution_receipt.json').read_text());t=pd.read_csv(e/'PIPELINE_STABILITY_RUNS.csv');m=pd.read_csv(e/'MODEL_COMPARISON.csv');c=pd.read_csv(e/'EXPLANATION_COMPLEXITY.csv');g=pd.read_csv(e/'INTERACTION_GUARD_SUMMARY.csv')
def table(df):
 cols=list(df.columns);lines=['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']
 for row in df.itertuples(index=False,name=None):
  lines.append('| '+' | '.join('NA' if pd.isna(v) else f'{v:.6f}' if isinstance(v,float) else str(v) for v in row)+' |')
 return '\n'.join(lines)
point=m[(m.seed=='ensemble')&(m.cause==1)][['year','model','horizon','companies','target_events','auc','ap','brier','captured','risk_mean','observed','supported']]
s=t.groupby(['year','selection_status']).agg(repeats=('repeat','size'),min_unique_discovery_events=('discovery_targets','min'),max_unique_discovery_events=('discovery_targets','max')).reset_index()
comparison=c[[x for x in ['year','model','features','main_functions','interaction_functions','spatial_table_cells','scalar_table_cells'] if x in c]]
counts=g.groupby('year').agg(candidate_cohort_checks=('checks','size'),all_function_guards_pass=('all_function_checks_pass','sum')).reset_index()
(e/'RESULTS_ZH.md').write_text(f'''# 有限分组筛选：前向确认不足，三个截止点回退83项

本轮已完成实际计算和数值验证。结果为 **INCONCLUSIVE（删减集合的独立增益未验证）**。保留原83项及74项模型，不修改研究参考。2023–2025已暴露，本轮是探索性机制实验。

## 执行范围与重要区别

每截止点预设8/9/9个集合（含83项），只允许一个稳定相关组的一/双代表替代，不做组间笛卡尔组合。实际复用历史森林检查23个非基准候选，完成1408条按组—伙伴—时限—原因定义的函数形状/幅度检查。

两个节点执行150个公司重抽样流程（每截止点50次），实际新拟合450个32树raw森林、派生450个直接父节点AJ收缩森林，合成150个96树预测器。不是新拟合150个最终删减方案：全部流程先重建分组与时间门槛，因支持不足回退83项，再真实拟合该回退模型。新抽样种子与既有固定集合重训不同，按入组年分层，独立公司数决定支持，不以重复行增加事件证据。

**监督式删减搜索执行0次，独立确认通过0次。** 所以FEATURE_SELECTION_STABILITY中的选择频率为NA，回退保留率100%不等于变量被证明重要、筛选稳定或83项最优。支持成立后的完整监督搜索与确认分支尚未在真实数据上验证；运行器遇到意外支持状态会明确停止，禁止静默批准删除。

## 为什么成熟历史事件仍不足

| 训练截止 | 当前可用的成熟H24发现队列 | 发现事件数 | 独立前向确认 | 正式决策 |
|---|---|---:|---|---|
| 2023-01-01 | 2020 | 6 | 无；发现阶段本身不足 | 回退83 |
| 2024-01-01 | 2020、2021 | 13 | 无 | 回退83 |
| 2025-01-01 | 2020、2021、2022 | 22 | 无 | 回退83 |

例如，2022队列的24个月结果到2025截止已成熟，但预测2022入组企业的筛选器必须在2022年初冻结。当时2020队列的完整24个月标签尚未成熟，不能拿后来观察到的结果去挑2022模型的变量。对2020/2021确认候选同理。2019模型拟合时没有退市事件，未满足预设支持要求。三个可能确认年份的合格发现事件均为0。

这是标签成熟滞后带来的约束，不是重新提高事件门槛追求某个结果；发现门槛仍为至少两队列、合计10例、各队列至少3例且历史拟合至少1例。新增条件是确认队列的筛选必须真正发生在它之前。没有把年份打乱或把未满随访标为非退市。

## 12/24个月统一比较

原83项、旧74项复用18个保存的种子森林，逐公司预测重新汇总并用统一评价器重算；正式筛选流程回退后与83项预测逐值相同。新450个森林用于重抽样流程检查，不能替换此处三种子正式预测。下表包括所有支持情况，2025H24仍NA；6例仅为已观察的部分窗口事件。

{table(point)}

合并12个月使用2023–2025；合并24个月只用2023–2024。AUC/AP/Brier在合并预测上重新计算，Top20按各年ceil(20%×公司数)名额后相加。已支持的正式窗口无未知结局；不支持窗口不赋零。

## 完整流程重抽样

{table(s)}

这检验的是分组及时间门槛对公司扰动的稳健性，并提供回退模型的真实训练波动。它不提供被门槛阻断的监督选择频率。FUNCTION_STABILITY、INTERACTION_STABILITY和PREDICTION_STABILITY均按同一年度训练参考分布、相同归一化和日历时限比较，未出现的函数计零；不得将其中结构出现频率当作特征筛选频率。

## 交互保护与复杂度

原生二元项按相关组与外部伙伴聚合，允许组内代表替代后作用重新分配。诊断要求幅度比0.5–2、相关性至少0.7、相对L2差不超过0.5；近零参考项单列。相同函数参考仅来自对应历史训练集，两个原因和12/24个月均检查。没有只凭主效应弱删除变量，也没有将联合删列分数称为纯交互增量。

{table(counts)}

这些是历史候选的函数一致性诊断，包含多个依赖的检查，不是独立验证队列或真实交互检验。没有通过前向确认时，即使函数保护通过也不能批准删除。阈值已冻结，未因本轮结果改变。

{table(comparison)}

正式模型回退，因此复杂度没有改善。50次新重训的函数波动单独保留，不能称为筛选带来的稳定性增益。相关变量的边际乘积参考可包含现实中稀少组合，精确分解不建立因果或统计可辨识性。

## 数值与资源

完整测试215项通过，1条已有EBM缺失值展示警告，无跳过或失败。全部1050个raw/收缩/合并模型验证q≤2，CIF合法、保存重载及全部训练/评价行、所有原生时间、两个原因的完整显式函数重构。最大误差{receipt['maximum_reconstruction_error']:.3e}，门槛1e-10；没有裁剪预测或删除旧函数。

Slurm作业2605、2606全部计算与回传成功，失败拟合0。任务节点分布：{receipt['hosts']}。计算资源由既有池管理，不改CPU绑定或cgroup。执行代码提交：`{receipt['training_code_commit']}`。模型参数、公司行、预测和阈值只在私有运行目录。

## 研究判断

1. 未找到获准替代83项或74项的新集合，也未证明不存在更好的集合。
2. 分组候选和交互保护诊断已实现并实际计算，但其未来预测收益没有得到确认。
3. 全流程重抽样按门槛回退，不能声称完成了有支持的监督选择稳定性验证。
4. q2完整内生解释和概率性质保留；本轮没有解释复杂度或稳定性改善的证据。
5. 当前参考继续保留。不得依据2023–2025外层结果再调保护阈值或扩展搜索。
''')
(e/'DATA_AND_PROTOCOL_AUDIT_ZH.md').write_text('''# 数据与时间审计

源提交386cad4a741e91c743c89a56d168db3ae700806f，独立分支research/xai-rsf-group-selection-v1。旧conservative C83列、名称/顺序、NaN语义和原始版本PIT的PROVISIONAL限制沿用。未恢复或重新认证年报版本。

训练截止均为各年1月1日之前；入组为公告后下一交易日开盘前，竞争终点为公告日先最终退市或先完整恢复，训练随访上限24日历月。训练公司309/345/390，已知退市14/26/49；评价公司36/45/53，训练与评价公司无重叠。每次重抽样只取对应截止点训练企业，按入组年分层。

support_at重新按截止日截断事件，成熟窗口按origin加日历月判定；bootstrap重复公司只算一次支持。确认队列对应的选择门槛必须按其入组年初信息重算，不能按当前外层截止点回填成熟标签。HISTORICAL_SUPPORT及audit/point_gates给出实际数值。

分组只使用历史训练X：最近三个训练窗口、Spearman绝对值>=0.95、至少30家共同非缺失独立企业。采用全连接相关组，避免传递连通分量误称组内两两高相关。每组仅固定一/双代表，不挑最高监督分数；审计七字段完整保留。旧74项作为公开固定对照，2023诊断时间局限保留，不注入新筛选先验。

2025年12个月按既有观察截止2026-06-30具备支持；24个月不足，完整指标NA。原83/74的模型和预测按哈希复用，三种子先平均后评价；正式回退模型不因重抽样表现替换。

首次本地创建目录因NFS根目录租约等待，停止本任务准备进程后通过lemon创建唯一运行目录。无训练失败、无覆盖旧目录、无服务配置修改。运行输入和当前模型指针哈希在audit/source_hashes中，交付复核一致。
''')
(e/'METHOD_INNOVATION_DECISION_ZH.md').write_text('''# 方法结论：INCONCLUSIVE

本轮未建立“分组筛选加交互保护优于原83/74”的证据。实际完成有限历史候选诊断、150次从重建分组开始的流程重抽样、450次回退森林拟合及统一评价。所有正式流程因独立前向确认支持不足回退83项。

完整精确解释得到数值验证；预测改善、解释简化、交互稳定性提升、真实交互保护和删减选择稳定性均不能宣称成立。交互函数诊断是已有模型的表示比较，不是经济因果作用恢复。NA选择频率不能替换为100%稳定选择。

分组/筛选属于训练前模型选择，q2与共享切点属于森林结构，完整函数导出属于该结构的精确表示。此次增加时间核验和保护规则不构成已证实的新算法预测贡献。

保留CURRENT_POST_ST_MODEL.json及全部83/74历史结果。本轮不自动扩充搜索、不调参、不切换主终点。支持成立后的监督选择与独立确认分支仍需真实充分历史数据验证；未来未暴露且随访充分的队列才可作为确认性升级依据。
''')
(e/'PAPER_METHODS_ZH.md').write_text('''# 方法段落

在固定q2、深度4、每树25候选变量、共享分位数切点及强度10直接父节点AJ收缩的森林上，预设有限的单组代表替代方案。稳定相关组仅由各截止点以前的财务/市场输入识别；审计意见字段整体保留。特征选择需满足历史发现期事件支持，并由后续成熟队列进行前向确认；用于确认的预测器必须在该队列起始前完成整个筛选过程。历史标签的成熟时间而非回顾时的可见性决定其可用性。交互保护采用相同历史参考下的组间原生二元函数形状和幅度诊断。每截止点实施50次按入组年分层的公司重抽样，重新识别组、检查时间支持并拟合实际回退或选择模型。支持不足时保留83项，记录监督筛选未执行，选择频率为不可估计。
''')
(e/'PAPER_RESULTS_ZH.md').write_text(f'''# 结果段落

2023、2024、2025截止点的成熟发现期退市分别为6、13、22例，但均不存在满足预设条件、且其筛选可在队列起始前完成的独立前向确认期。因此三年度正式流程均回退83项，外层预测与原83项完全一致，未获得删减带来的预测或解释复杂度改善证据。150次流程重抽样均未进入监督选择阶段，不能估计有支持的选择稳定性；450个新森林用于这些流程的实际回退训练。全部1050个模型及派生预测器通过完整q2函数、概率合法性和保存重载核验，最大重构误差{receipt['maximum_reconstruction_error']:.3e}。2025年24个月评价保持不可估计。所有外层比较为已暴露队列上的探索性结果。
''')
(e/'README.md').write_text('''# 有限分组筛选实验

先读frozen_protocol.json、DATA_AND_PROTOCOL_AUDIT_ZH.md和RESULTS_ZH.md。执行入口：`python -m experiments.post_st_group_selection_v1.run prepare|freeze|pilot|batch|diagnostics`，汇总入口：`python -m experiments.post_st_group_selection_v1.deliver`。

私有路径见run_location.json，源数据沿用原实验，不复制公司文件到Git。复现必须先在run.py指定新的唯一RUN，prepare拒绝已有冻结目录；完成测试并提交后freeze建立不可变代码快照。pilot两项纳入总150项，batch只运行其余148项，避免重复。每阶段JSONL及hetero sidecar由运行器生成，既有两节点Slurm/NFS调度并核验回传。

支持门槛关闭时监督候选搜索不执行，不能把回退拟合写成筛选成功。当前实现只实际验证此已冻结数据下的门槛/回退分支；若新的数据触发支持通过，运行器明确停止，须补齐并验证支持分支后另行冻结，禁止静默使用未经核验的选择逻辑。历史23候选和1408函数诊断独立保存，不影响正式回退决定。
''')
print(s.to_string(index=False));print(counts.to_string(index=False));print('Reports written')
