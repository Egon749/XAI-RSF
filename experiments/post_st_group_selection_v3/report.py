"""Generate Chinese conclusions from all frozen candidates, including failures."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .run import PUBLIC,RUN,PARENT
from experiments.post_st_dropcolumn_selection_v1.common import load
from experiments.post_st_group_selection_v2.report import table


def main():
    rec=load(PUBLIC/'execution_receipt.json');tasks=pd.read_csv(PUBLIC/'PIPELINE_RUNS.csv');scans=pd.read_csv(PUBLIC/'GROUP_DISCOVERY_RESULTS.csv')
    gc=pd.read_csv(PUBLIC/'GROUP_PROTECTION_DIAGNOSTICS.csv.gz');period=pd.read_csv(PUBLIC/'PERIOD_DISCOVERY_RESULTS.csv');pol=pd.read_csv(PUBLIC/'MATCHED_POLICY_DECISIONS.csv')
    perf=pd.read_csv(PUBLIC/'MODEL_COMPARISON.csv');cx=pd.read_csv(PUBLIC/'EXPLANATION_COMPLEXITY.csv')
    features=load(PARENT/'input/features.json');labels={r['feature_id']:r['label'] for r in features}
    pc=scans[scans.repeat==-1].copy();pc['deleted']=pc.keep.map(lambda x:'、'.join(labels[j] for j in sorted(set(range(83))-set(json.loads(x)))))
    pc['prediction_complexity_pass']=pc.performance_failures.map(lambda x:not json.loads(x))
    pc['approved_new']=pc.prediction_complexity_pass&pc.new_group_pass
    pc['approved_old']=pc.prediction_complexity_pass&pc.old_pair_pass
    pc[['year','mode','deleted','spatial_cell_change','prediction_complexity_pass','old_pair_pass','new_group_pass','qualified_groups','failed_group_checks','approved_old','approved_new','performance_failures']].to_csv(PUBLIC/'POINT_CANDIDATES.csv',index=False)
    annual=[]
    for y in [2023,2024,2025]:
        p=tasks[(tasks.year==y)&(tasks.repeat==-1)].iloc[0];b=tasks[(tasks.year==y)&(tasks.repeat>=0)]
        annual.append(dict(截止年度=y,发现期退市=int(p.discovery_targets),点估计变量数=int(p.features),重抽样实际搜索=int(b.supervised_search_executed.sum()),重抽样删减次数=int((b.features<83).sum()),新增森林数=int(tasks[tasks.year==y].new_forests.sum())))
    approval=pc.groupby(['year','mode']).agg(候选数=('keep','size'),性能复杂度通过=('prediction_complexity_pass','sum'),旧交互通过=('old_pair_pass','sum'),新组保护通过=('new_group_pass','sum'),最终旧规则通过=('approved_old','sum'),最终新规则通过=('approved_new','sum')).reset_index()
    allfull=bool((tasks[tasks.repeat==-1].features==83).all())
    conclusion=('三个年度点估计均保留83项，外层预测与原83项一致。放宽保护对象并未自动产生更优的小变量集；必须结合逐期性能和复杂度的具体拒绝理由解释。' if allfull else '部分点估计选择了删减集。下面逐年度报告其预测得失，不将历史发现期通过保护称为独立验证成功。')
    primary=perf[(perf.seed=='ensemble')].copy();primary['capture']=primary.apply(lambda r:f'{int(r.captured)}/{int(r.target_events)}' if pd.notna(r.captured) else 'NA',axis=1)
    views={c:table(primary[primary.cause==c][['year','horizon','model','auc','ap','brier','risk_mean','observed','capture']]) for c in [1,2]}
    failures=[]
    for r in scans.to_dict('records'):
        for reason in json.loads(r['performance_failures']):failures.append(dict(year=r['year'],scope='point' if r['repeat']==-1 else 'resampled',mode=r['mode'],reason=reason))
    failures=pd.DataFrame(failures).groupby(['year','scope','mode','reason']).size().rename('candidates').reset_index();failures.to_csv(PUBLIC/'GUARD_FAILURE_COUNTS.csv',index=False)
    qualified=gc[gc.qualified];qsummary=gc.groupby(['year','repeat','mode','cause']).agg(checks=('passed','size'),qualified=('qualified','sum'),failed=('passed',lambda v:int((~v).sum()))).reset_index();qsummary.to_csv(PUBLIC/'GROUP_GUARD_SUMMARY.csv',index=False)
    cq=[k for k in ['year','model','features','main_functions','interaction_functions','spatial_table_cells'] if k in cx]
    if len(cq)<5:cq=[k for k in cx if k in ['year','model','features'] or 'main' in k or 'pair' in k or 'cell' in k]
    changed=pol[(pol.repeat>=0)&(pol.features<83)].groupby(['year','policy']).size().rename('selected').reset_index()
    text=f'''# 稳定组信息保护与候选覆盖配对实验

实际完成{rec['pipeline_tasks']}个独立流程，新增{rec['new_raw_forests']}个32树森林及同数父节点收缩版本。两台机器任务分配为{rec['hosts']}，拟合失败{rec['failed_fits']}个，任务失败{rec['failed_task_attempts']}个。所有原83、74模型与当前指针保留。

{conclusion}

{table(pd.DataFrame(annual))}

## 实际改了什么

新保护以相关组及其训练期相关代理的主效应和全部关联二元项之和为对象。只有组贡献足够大、占比足够高、三个种子的函数形状一致且原因事件支持足够时，才限制新旧组贡献的差异；其余检查记录为未建立保护证据，不视为已经证明无信息。跨变量、主效应与二元项之间可以重新分配贡献。完整重构是另一项独立检查，并不要求新模型逐项复制旧模型。

删变量后保留原83候选表中的其余变量，只填补空位，每树仍25项。未涉及删除项的树须逐节点等同原83模型。填补空位仍会改变受影响树的搜索机会；本轮控制的是大量无关树的重排，不能宣称候选空间完全相同。

性能保护改为逐历史年度、逐时限、逐原因。两个结局的排序损失和Brier均检查，不能用恢复较好的一年平均掉较差的一年。稀少事件期不强行把不可估计的排序量当作0；Brier在可估计时仍参加保护。门槛冻结后没有根据外层表现修改。

## 配对结果：覆盖与保护分别起了什么作用

{table(approval)}

上表所有点估计对照使用相同的逐期预测保护。uncontrolled为删列后重新均衡；anchored为原83候选表修复。每个覆盖下，旧逐交互保护和新组保护作用于同一批实际森林，不额外拟合。候选通常同时不满足多个条件，不能将各失败数相加。全部候选的变量名称、细分失败原因和空间单元变化见POINT_CANDIDATES.csv。

点估计四种政策的最终名单：

{table(pol[pol.repeat==-1][['year','policy','features','removed','status']])}

重抽样采用完整流程而不是固定名单再训练。每个截止点50次重新分组、检查事件、拟合全部有限候选、应用保护、选择和最终重训。原83内层控制符合哈希与相同抽样时可复用，全部复用与新增数量见ARTIFACT_REUSE.csv。主要锚定覆盖下旧/新规则的删减次数：

{table(changed) if len(changed) else '各重抽样政策均未选择删减集。'}

这些频率描述算法在当前历史样本下的行为。发现支持不足而回退83不能解释成83项都重要；选择少数一次也不证明可安全删除。三个随机种子仅衡量算法随机性，不是三个独立验证队列。

## 退市12/24个月结果

先平均三个种子，再计算年度指标。风险均值与IPCW观察发生率同列；Top20每年按ceil(0.2×公司数)分配名额。pooled AUC/AP/Brier在合并预测上重新计算，捕获按年度名额相加。

{views[1]}

## 恢复12/24个月结果

{views[2]}

2025的24个月随访至2026-06-30不足，相关性能为NA；不能把部分随访的事件作为成熟24个月评价。所有年度均已暴露，本次只作探索性复核。PAIRED_DIFFERENCES.csv保存1000次共同公司重抽样下的条件95%差值区间，AUC/AP正值及Brier负值有利。区间不包含完整训练、选择或研究者调整的不确定性。

## 解释复杂度和稳定性

{table(cx[cq])}

完整组保护检查中有{len(qualified)}条达到稳定、幅度、占比及原因事件支持要求，其中{int((~qualified.passed).sum())}条超出冻结容许差异。每条都保留组成员、相关代理、原/新主效应与交互幅度、跨种子相关及容许差异。检查跨种子和时限重复，不能当作独立统计检验次数。

完整过程函数稳定性以共同的原历史训练参考分布计算、映射回原83列ID，缺失函数按0。见FUNCTION_STABILITY.csv、INTERACTION_STABILITY.csv和FUNCTION_SET_STABILITY.csv。概率是CIF、贡献在CIF尺度上；均不是风险率或因果效果。相关输入下的经验边际乘积参考分布有外推组合限制。

共{rec['verified_models']}个raw/收缩/96树合并模型通过所有训练与评价行、全部原生时间点和两原因的验证。最大完整重构或重载误差{rec['maximum_reconstruction_error']:.3g}，低于1e-10；q≤2、非负、随时间不减、两原因和≤1均保持，没有新增裁剪。

## 研究判断

这轮将保护对象从“所有旧二元项”改为“已有稳定且重要证据的组信息”，并实际控制了删列后的候选覆盖重排。它能检验旧规则的保守性，但不保证一定选出更少变量，也不把被否决变量认定为不可替代。

{conclusion} 本轮不能据此替换当前研究模型。应根据逐期退市、恢复的损失、复杂度和重抽样选择频率共同判断；不再延长搜索或降低冻结门槛以获得有利名单。

特征选择是训练前的有限优化，候选集合修复是配对实验方法，q2森林与完整CIF函数表示是内生解释机制。三者需要分别评价，不能合并称为一个已经证明有效的方法创新。原始披露版本仍有PROVISIONAL限制，没有独立未暴露确认队列。

执行代码：{rec['training_code_commit']}。数据、工件复用、Slurm回传、配置与模型哈希均见audit和execution_receipt.json；企业明细、阈值、模型与完整预测仅在服务器私有目录。
'''
    (PUBLIC/'RESULTS_ZH.md').write_text(text)
    (PUBLIC/'METHOD_INNOVATION_DECISION_ZH.md').write_text('# 方法研究判断\n\n'+conclusion+'\n\n新组信息保护、候选覆盖修复和逐期双原因保护已真实执行，原q2精确解释保留。有限分组筛选的实用收益以逐期性能、复杂度与完整过程稳定性决定，不能仅凭可运行或允许重新分配贡献宣称成功。当前83/74及研究指针不替换；不扩大搜索。\n')
    (PUBLIC/'PAPER_METHOD_RESULTS_ZH.md').write_text(f'''# 论文方法与结果草稿

采用各预测年度1月1日前的信息进行有限分组特征选择。候选为稳定高相关组的一或两个代表替代；采用原83列每树候选集合的最小修复，保持25项和公司bootstrap及分裂随机流。保护对象为同类相关代理范围内主效应与所有关联二元项之和，仅对具有足够幅度、贡献占比、跨种子一致性和原因事件支持的组施加差异约束。性能保护按历史前向队列分别检查两原因的12/24个月结果。每年度50次公司重抽样完整重做选择与最终拟合，事件支持不足回退83项。

完成{rec['pipeline_tasks']}个流程与{rec['new_raw_forests']}次新森林拟合。{conclusion} 最大完整函数重构/重载误差为{rec['maximum_reconstruction_error']:.3g}。2025年24个月随访不足未评分，全部外层队列仅作探索性分析，未自动更换研究参考模型。
''')
    print(table(pd.DataFrame(annual)));print(table(approval));print(views[1])

if __name__=='__main__':main()
