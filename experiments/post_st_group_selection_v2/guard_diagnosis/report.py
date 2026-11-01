"""Summarize existing-model evidence, without modifying the selection decision."""
from pathlib import Path
import json,shutil
import numpy as np
import pandas as pd
from .analyze import ROOT,RUN,DROPS,GROUPS
from experiments.post_st_dropcolumn_selection_v1.common import load,write,sha
from experiments.post_st_group_selection_v2.run import RUN as SOURCE


def table(d):
    def cell(x):
        if pd.isna(x):return 'NA'
        if isinstance(x,(float,np.floating)):return f'{x:.4f}'
        return str(x).replace('|','/')
    return '\n'.join(['| '+' | '.join(d.columns)+' |','| '+' | '.join(['---']*len(d.columns))+' |']+['| '+' | '.join(map(cell,row))+' |' for row in d.itertuples(index=False,name=None)])


def main():
    pair=pd.read_csv(ROOT/'PAIR_DIAGNOSTICS.csv');agg=pd.read_csv(ROOT/'GROUP_AGGREGATE.csv');tree=pd.read_csv(ROOT/'TREE_ELIGIBILITY.csv');pfi=pd.read_csv(ROOT/'GROUP_PERMUTATION.csv')
    old=load(Path(load(SOURCE/'audit/selection_decisions.json')['2024']['output'])/'interaction_checks.json');matched=0
    for row in old:
        removed=sorted(set(range(83))-set(row['keep']))
        if len(removed)!=1 or removed[0] not in DROPS or row['partner']<0:continue
        r=pair[(pair.scope=='train')&(pair.removed==removed[0])&(pair.fold==row['cohort'])&(pair.partner==row['partner'])&(pair.horizon==row['horizon'])&(pair.cause==row['cause'])].iloc[0]
        assert bool(r.guard_pass)==row['passed']
        if row['reason']!='NEGLIGIBLE_BASE_COMPONENT':
            np.testing.assert_allclose([r.amplitude_ratio,r.correlation,r.relative_l2],[row['ratio'],row['correlation'] if row['correlation'] is not None else np.nan,row['relative_l2']],atol=1e-10,rtol=1e-10,equal_nan=True)
        matched+=1
    f=pair[(pair.scope=='train')&(~pair.guard_pass)&(pair.guard_reason!='NEGLIGIBLE_BASE_COMPONENT')]
    summary=[]
    for drop,rows in f.groupby('removed'):
        summary.append(dict(删除变量=rows.removed_name.iloc[0],失败检查=len(rows),仅一个种子有该作用=int((rows.base_seed_presence==1).sum()),
            原幅度小于0_1个百分点=int((rows.base_rms_pp<.1).sum()),变化小于原种子间波动=int((rows.change_over_base_seed_sd<=1).sum())))
    pd.DataFrame(summary).to_csv(ROOT/'GUARD_SENSITIVITY_SUMMARY.csv',index=False)
    z=agg[(agg.scope=='validation')&(agg.horizon==24)&(agg.cause==1)&(agg.component=='original_group_all')]
    z=z[['fold','removed','base_rms_pp','new_rms_pp','amplitude_ratio','correlation']].copy()
    names={int(r.removed):r.removed_name for r in pair.itertuples()};z['removed']=z.removed.map(names)
    z.columns=['历史验证队列','删除变量','原组贡献RMS百分点','新组贡献RMS百分点','幅度比','相关系数']
    c=tree.groupby(['fold','removed']).agg(相同候选集树数=('eligible_same','sum'),平均共同候选数=('eligible_intersection','mean'),相同根分裂数=('root_split_same','sum')).reset_index()
    c['removed']=c.removed.map(names)
    major=pair[(pair.scope=='train')&(pair.horizon==24)&(pair.cause==1)].sort_values('base_rms_pp',ascending=False).head(10)
    major.to_csv(ROOT/'LARGEST_TARGET_INTERACTIONS.csv',index=False)
    text=f'''# 三个删减候选的交互保护诊断（不重新训练）

结论：当前“任一非微小旧交互失败即否决”的规则明显偏保守，但三个候选并不都只是相关变量间重新分配贡献。应区分实际模型函数减少、已有证据支持的预测信息损失，以及重训时整个森林的结构重组。

## 证据范围

只加载2024截止前选择阶段保存的2020/2021历史前向模型与数据，共32个函数导出器（原83项与三个候选、两个年度、三个种子及合并模型），**新增训练0次**，运行时禁用fit接口。未读取2023–2025外层结局。两个历史训练集仅有2、3例退市；对应24个月验证有6、7例，合计13例。12个月主要保护仅有2020队列3例退市，2021的12个月仅作低支持诊断。

这里的“2020/2021”是历史验证队列，三名候选均来自2024截止的筛选，不是2024外层测试结果。原来的筛选名单、阈值、当前83项和74项模型均保持不变。

重新计算并逐项复现了{matched}条原交互检查，判定及数值一致。全部主效应与二元项重新相加，最大误差见execution_receipt.json（小于1e-10）。全部模型与数据哈希未改变。

## 原规则已经合并了同组变量

原规则比较的是相关组G与某伙伴l的总二元贡献 Σ(j∈G) f(j,l)，并非只盯着删除变量的旧名称。因此，不能简单解释为“旧变量换成相关变量，名字变了所以误判”。三组分别为：

- 近12个月总市值变化、近12个月累计收益率；删前者、留后者。
- 近3个月下行波动率、近3个月尾部平均收益；删前者、留后者。
- 近3个月总市值变化、近3个月累计收益率；删前者、留后者。

在各自历史训练集上，第一组Spearman约0.986/0.991，第二组约−0.974/−0.964，第三组约0.937/0.952。第三组在更早2020拟合信息中尚未达到0.95，不能把它视为所有时期都几乎相同的变量。共同非缺失样本数、缺失一致性见CORRELATIONS.csv。

## 保护条件为什么偏保守

原规则对RMS≥1e-8概率的旧块均要求幅度比[0.5,2]、相关≥0.7、相对L2≤0.5，并要求所有队列、原因和时限同时通过。1e-8概率等于0.000001个百分点，仅排除数值近零项，不等于筛选“重要交互”。

{table(pd.DataFrame(summary))}

合计{len(f)}条失败检查中，{int((f.base_seed_presence==1).sum())}条仅一个种子产生了非微小作用，{int((f.base_rms_pp<.1).sum())}条原贡献RMS小于0.1个百分点；{int((f.change_over_base_seed_sd<=1).sum())}条的新旧变化没有超过原三种子函数的RMS波动。失败涉及退市65条、恢复108条。每条是“队列×伙伴×时限×原因”检查，同一交互的多条记录不是独立样本。

这提示规则常在保护单个种子特有、尚未证明稳定的函数。三种子波动不是置信区间，三个种子不是独立队列；小幅度也不必然无预测价值。上述结论应与绝对幅度、验证表现和稳定性一起理解，不能只因幅度小便自动删除。

## 合并主效应与全部组交互后，是否保留原作用

下表将同组所有主效应、组内二元项和涉及该组的全部组外交互相加，在历史验证公司上计算24个月退市CIF贡献RMS。它是贡献幅度，既不是企业退市概率，也不是事件捕获率。比值表示新幅度/旧幅度；RMS不区分正负，另列相关性。

{table(z)}

### 1. 删除近12个月总市值变化：作用确实减少，重要性证据弱

2020模型中整个“12个月收益/市值变化”组的退市贡献降到0；2021保留约59%。主效应未补回消失的交互，扩大到|Spearman|≥0.8的市场代理组仍未找到其他变量来保持该组的完整函数。这不是简单换名或完全等价的贡献转移。

但原组的预测必要性证据弱：将组内两个字段一起置换，原模型24个月退市AP的平均损失在2020为−0.0047、2021为+0.0006，接近无正向依赖证据。候选在2021对保留的收益率组反而显示AP依赖约+0.0636，说明幅度变小不等于排序信息变差。置换具有扰动噪声及相关性局限，不能据此证明该变量永远无用。

确有需保留关注的例外：2021该组与总资产增长的旧交互RMS约0.0474个百分点，来自2/3种子、两个非零种子函数相关约0.853，候选中消失。它不是纯数值残差；但仅凭少量历史事件仍无法断言这是稳定且不可替代的真实机制。

### 2. 删除近3个月下行波动率：部分重分配证据最强

保留尾部平均收益后，原组“全部二元项加总”的退市RMS在两折仍保留约83%和84%，并非交互总体被压至零。原组主效应及总贡献有减少。扩展到同一历史训练集|Spearman|≥0.8的相关市场波动/尾部变量后，该块总贡献RMS达到原来的约130%和144%。2021扩大块的函数相关约0.852，2020约0.524；支持部分信息转移，不能称完全保持原函数。

原组主效应变化与交互变化没有形成清楚的逐公司抵消。因此更合适的解释是相关波动变量及不同交互伙伴之间重新分配，而不是“交互准确搬到了同组主效应”。旧组置换在两折未带来稳定的正向退市AP损失；这一候选最不适合仅凭每个旧伙伴交互的保形失败就一票否决。

也有局部变弱：2021该组与标准无保留意见标记的交互RMS约0.0444→0.0089个百分点，原有2/3种子、非零种子相关约0.821。总体信号重分配与个别较稳定交互削弱可以同时发生。

### 3. 删除近3个月总市值变化：不能仅按“冗余”处理

退市组总贡献在2020仅保留约24%，2021降到0；保留的累计收益率、同组主效应或扩大代理组都没有恢复原函数。这里存在实际的函数信号削弱。

恢复结局还出现明确的局部预测代价：2020历史验证中，该组恢复贡献RMS约2.510→0.583个百分点，主要由主效应减少带来；恢复AUC约0.8421→0.8105、AP约0.9127→0.8750、Brier约0.1994→0.2055。原组联合置换造成恢复AP平均下降约0.0239，说明这一段恢复信号不能一概当噪声。2021恢复指标部分改善，因此跨期平均保护仍可能通过。

原筛选保护恢复的跨期平均Brier，而没有要求每个队列的恢复AUC/AP都保持；平均会掩盖这一局部损失。但这些证据尚不能将恢复损失专门归因于“重要二元交互消失”：主效应也明显减少，且森林其他部分同时改变。三个候选中，此项应最谨慎。

## 一个重要混杂：删除变量也改变了森林的候选覆盖

每树仍是25项候选，但83→82会改变均衡覆盖排列。以下每行包含96棵配对树；公司bootstrap重复次数逐树完全一致。

{table(c)}

每个候选只有1–2棵树的可用变量集合完全相同，平均只共享约19.3–19.6/25项。原先无权使用被删变量的66–68棵树中，也有65–66棵改变了候选集合。根分裂约41–49棵发生改变。因此对比中包含明显的搜索和结构重组；不能把全部新旧函数差异归因于被删字段的独立信息，也不能仅凭相同种子就认为森林其他部分固定。

完整预测在24个月历史验证上的相关性仍约0.845–0.963，但这种整体相似性也不能证明所有局部交互都保持。当前数据无法在不新增严格配对训练的情况下完全分离“删变量效应”与“覆盖重排效应”。本轮没有启动这类训练。

## 对规则的判断

1. 这套“所有非微小旧交互逐项保形”的否决规则确实过于保守，缺少对重要性和原模型自身不稳定性的区分。不能将零删除解释成83项都重要。
2. 也不能反向断言三个删减都安全：第一个和第三个存在真实的模型函数消失；第三个还有恢复结局的局部损失。真实作用是否重要、可替代或可推广，现有小样本不足以完全判定。
3. 若后续修订，应以历史支持的稳定、重要信号为保护对象，同时检查组总作用、相关代理组、逐时间窗口和两个原因的性能；针对候选覆盖重排单独控制。这里未据诊断修改门槛或重新接受任何候选。
4. 精确内生解释是“每个模型自己的基线+主效应+交互精确等于自己的CIF”。要求新模型逐项复刻旧模型函数，是额外且更强的跨模型稳定性约束，两者不能混同。

本次所有新统计均为描述性诊断。100次置换的离散波动不是训练不确定性；组置换保留组内配对，但可能打破与其他字段的相关关系。没有证明因果效应、真交互或全局最优特征集。未更换参考模型，未使用外层结果决定新筛选规则。
'''
    (ROOT/'RESULTS_ZH.md').write_text(text);shutil.copy2(ROOT/'RESULTS_ZH.md',RUN/'reports/RESULTS_ZH.md')
    result=dict(status='PASS',reproduced_original_guard_checks=matched,new_fits=0,decision_unchanged=True)
    write(ROOT/'validation.json',result);write(RUN/'validation.json',result)
    for name in ['GUARD_SENSITIVITY_SUMMARY.csv','LARGEST_TARGET_INTERACTIONS.csv']:shutil.copy2(ROOT/name,RUN/'reports'/name)
    print(result)

if __name__=='__main__':main()
