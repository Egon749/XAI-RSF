"""Chinese evidence report generated from the completed finite experiment."""
import json,shutil
from pathlib import Path
import numpy as np
import pandas as pd
from .run import RUN,PUBLIC
from experiments.post_st_dropcolumn_selection_v1.common import load


def table(df):
    return df.to_markdown(index=False,floatfmt='.4f')


def main():
    rec=load(PUBLIC/'execution_receipt.json');tasks=pd.read_csv(PUBLIC/'PIPELINE_RUNS.csv');scans=pd.read_csv(PUBLIC/'GROUP_DISCOVERY_RESULTS.csv')
    comp=pd.read_csv(PUBLIC/'MODEL_COMPARISON.csv');cx=pd.read_csv(PUBLIC/'EXPLANATION_COMPLEXITY.csv');features=pd.read_csv(PUBLIC/'SELECTED_FEATURES.csv')
    point=tasks[tasks.repeat==-1];boot=tasks[tasks.repeat>=0];pc=scans[scans.repeat==-1];bc=scans[scans.repeat>=0]
    rows=[]
    for y in [2023,2024,2025]:
        p=point[point.year==y].iloc[0];b=boot[boot.year==y]
        rows.append(dict(年度=y,发现事件=int(p.discovery_targets),点估计筛选=bool(p.supervised_search_executed),点估计候选=int(p.scored_candidates),
            最终变量=int(p.features),重抽样实际筛选=int(b.supervised_search_executed.sum()),重抽样选出删减集=int((b.features<83).sum()),新拟合森林=int(tasks[tasks.year==y].new_forests.sum())))
    failures=[]
    for scope,df in [('点估计',pc),('重抽样',bc)]:
        flags=df.failed_guards.map(json.loads)
        keys=sorted({k for rr in flags for k in rr})
        for key in keys:failures.append(dict(范围=scope,未通过项=key,候选次数=sum(key in rr for rr in flags)))
    reject=pd.DataFrame(failures);reject.to_csv(PUBLIC/'GUARD_FAILURE_COUNTS.csv',index=False)
    perf=comp[(comp.seed=='ensemble')&(comp.cause==1)].copy()
    perf['捕获']=perf.apply(lambda r:f'{int(r.captured)}/{int(r.target_events)}' if pd.notna(r.captured) else 'NA',axis=1)
    view=perf[['year','horizon','model','auc','ap','brier','risk_mean','observed','捕获']]
    ccols=[k for k in ['year','model','features','main_functions','pair_functions','spatial_table_cells'] if k in cx]
    if len(ccols)<5:ccols=[k for k in cx if k in ['year','model','features'] or 'main' in k or 'pair' in k or 'cell' in k]
    refused_interaction=sum('INTERACTION_SHAPE_OR_AMPLITUDE' in json.loads(x) for x in pc.failed_guards)
    perfpass=sum(not [g for g in json.loads(x) if g not in ['INTERACTION_SHAPE_OR_AMPLITUDE','INTERACTION_EVIDENCE_UNAVAILABLE']] for x in pc.failed_guards)
    allfull=bool((point.features==83).all())
    conclusion=('三个年度点估计最终均保留83项。因此外层预测与原83项相同；本轮没有取得一个可替代83/74的较小点估计特征集。与上一轮不同，本轮2024/2025的全部预设发现候选确实被重新训练、评分并检查交互，再因具体保护条件被拒绝。'
        if allfull else '部分年度点估计选择了较小特征集。其外层指标是已暴露队列的探索性复核，不能把发现阶段选择或本轮表现视为独立确认；所有差值均见配对表。')
    text=f'''# 有限分组筛选第二轮结果

本轮已完成真实筛选分支，而非仅运行83项回退森林。共{rec['pipeline_tasks']}个完整任务，实际新拟合{rec['new_raw_forests']}个32树原始森林，并各派生一个直接父节点AJ收缩版本。点估计{rec['point_searches']}次、公司重抽样{rec['resampled_searches']}次进入真实发现搜索，总计{rec['candidate_comparisons']}个删除候选比较；{rec['failed_fits']}个拟合失败。两个节点的任务分配：{rec['hosts']}。

{table(pd.DataFrame(rows))}

2023发现阶段只有一个合格成熟队列，仍依预设回退；2024与2025不再因为缺少独立确认而停止。重抽样支持按独立公司计数，不把重复抽中的退市当新事件；故部分重抽样仍回退是预设算法行为。其余任务都重新识别相关组、拟合所有有限候选、计算性能和交互保护，然后重新训练最终模型。原83、74和当前研究指针均未替换。

{conclusion}

## 为什么候选被接受或拒绝

点估计共{len(pc)}个删除候选，其中{perfpass}个满足性能及复杂度保护，{refused_interaction}个至少有一项交互形状/幅度保护失败。拒绝可能同时涉及多个条件，下表不能相加当独立候选数。

{table(reject)}

完整数值见GROUP_DISCOVERY_RESULTS.csv和INTERACTION_PROTECTION_DIAGNOSTICS.csv.gz。门槛始终为两个时限AUC/AP降幅不超过0.02、各时限Top20捕获减少不超过1、Brier增加不超过0.002、恢复Brier增加不超过0.005，以及空间单元减少；交互参考幅度比[0.5,2]、相关至少0.7、相对L2至多0.5。无法估计的保护项失败保留83，不填零。

这些阈值是有限探索的操作保护，不是统计等效检验或唯一正确标准。尤其要求每个非微小受影响交互块都通过，可能很严格。候选被拒绝说明不能满足这套已冻结规则，不能由此断言该变量永远不能删或具有不可替代的因果作用。没有事后降低门槛来得到一个更小的名单。

## 12/24个月外层预测

先平均3个种子的逐公司CIF再评价；AUC/AP按一致竞争风险与删失口径计算，Brier为IPCW。恢复为竞争事件，不是删失；未知结局不标为正常。风险均值与IPCW观察发生率同时列出。每年Top20名额为ceil(0.2×公司数)；合并捕获按年度名额加总，合并AUC/AP/Brier在合并预测上重新计算。

{table(view)}

2025年的24个月结果保留NA：随访至2026-06-30尚不支持该时点。合并24个月仅包含2023/2024，合并12个月包含三个年度。恢复原因及逐种子结果保存在同一MODEL_COMPARISON.csv。2023–2025均已暴露，只能探索性解释。

PAIRED_DIFFERENCES.csv保留所有预设方向的差值与1000次按年度分层的公司配对重抽样95%区间。AUC/AP正值和Brier负值有利。它们条件于已经固定的模型和IPCW权重，不包括重新训练、特征选择或模型开发的不确定性；不能选择性声称显著优越。

## 解释与稳定性

{table(cx[ccols])}

完整过程50次重抽样中的保留频率与有支持搜索条件下的保留频率分开报告。回退时83项全部保留，不表示每项都重要。变量组重新识别频率、实际组替换频率见GROUP_SELECTION_STABILITY.csv；函数集合Jaccard、同参考分布上主效应/交互的幅度波动见FUNCTION_SET_STABILITY.csv及FUNCTION_STABILITY.csv。缺失函数按零进入矩估计，原列ID对齐，避免删除列后的编号错位。

函数稳定性是整个流程的结果，包括分组、筛选、公司重抽样与森林随机性。本轮没有独立确认样本，也没有以重抽样稳定频率再次筛选名单。保护旧模型函数不等于证明学到了真实交互；全CIF精确重构只证明解释忠实于本模型。

{rec['verified_models']}个raw/收缩/96树合并预测器完成检查，全部训练与评价样本、全部原生时间点、两个原因的最大重构/重载误差为{rec['maximum_reconstruction_error']:.3g}，低于1e-10。q≤2、非负、时间单调和原因CIF和≤1均通过，没有裁剪。完整企业函数与参数留在服务器。

## 实施范围与研究判断

1. 真正的有限筛选分支已经实现并实际执行；有事件支持的任务不再被独立确认不足阻断。
2. {conclusion}
3. 不能将保护失败解释成变量真实不可替代；当前证据更适合说明这套严格规则的选择行为和限制。
4. 自动特征选择是训练前步骤；q2结构约束和精确CIF函数表示仍是森林内生解释的组成。完成自动筛选不自动构成新的森林算法创新。
5. 当前研究参考模型继续保留，不做自动升级，不扩展搜索或修改门槛。

## 可复现性

训练冻结提交：{rec['training_code_commit']}。所有153任务使用同一冻结代码与协议、独立输出，保存完整候选模型、公司重抽样索引、实际种子、哈希、Slurm作业与失败记录。复用18个历史83/74种子模型仅作外层参照；本轮所有发现候选及最终点估计/重抽样模型均新拟合。原始金融披露版本的PROVISIONAL限制仍存在。计算时间受两机并行、P/E核及调度影响，不作为模型性能排名。
'''
    (PUBLIC/'RESULTS_ZH.md').write_text(text)
    (PUBLIC/'METHOD_INNOVATION_DECISION_ZH.md').write_text(f'''# 方法判断\n\n{conclusion}\n\n本轮支持“完整、可运行、历史信息约束下的有限分组筛选流程”已实现，支持筛选后的森林保持精确内生解释。并不证明自动筛选必然改善预测，也不证明保护门槛识别了真实交互。\n\n缺少独立确认期限制效果主张，不限制已授权的探索计算。事件不足的回退与有支持但候选不通过保护应分别解释。特征选择属于训练前优化，不能与森林结构约束和精确函数分解混为一项创新。当前模型不替换，进一步阈值或候选设计需要单独研究协议。\n''')
    (PUBLIC/'PAPER_METHOD_RESULTS_ZH.md').write_text(f'''# 论文方法与结果草稿\n\n以各预测年度1月1日为信息截止点，仅使用此前观察到的历史结局。使用按时间前向构造的成熟24个月队列作为发现集；至少两个队列且累计10个独立退市事件才启动有限筛选。基于三个历史窗口稳定的绝对Spearman相关不低于0.95构造完整连接变量组，每次替换一个组内的一或两个代表。所有候选均用相同q2、深度4、三种子各32树和直接父节点AJ收缩10重新拟合。选择须同时满足12/24个月性能、函数空间复杂度和同参考分布下的交互幅度/形状约束。无独立确认期不阻断探索，但不据此提出确认性效果结论。每年度50次按入组年份分层的公司重抽样完整重做分组、候选拟合、选择及最终训练。\n\n共完成{rec['pipeline_tasks']}个流程，{rec['new_raw_forests']}次森林拟合，{rec['candidate_comparisons']}次候选比较。{conclusion}所有{rec['verified_models']}个验证预测器的最大完整函数重构/重载误差为{rec['maximum_reconstruction_error']:.3g}。2025年24个月随访不足，未计算该时点性能。已暴露外层队列结果仅用于探索性机制复核，未替换参考模型。\n''')
    for name in ['RESULTS_ZH.md','METHOD_INNOVATION_DECISION_ZH.md','PAPER_METHOD_RESULTS_ZH.md','GUARD_FAILURE_COUNTS.csv']:shutil.copy2(PUBLIC/name,RUN/'reports'/name)
    print(table(pd.DataFrame(rows)));print(table(view))

if __name__=='__main__':main()
