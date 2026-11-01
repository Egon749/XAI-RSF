"""Publish only aggregate evidence, protocols, provenance and static figures."""
from pathlib import Path
import argparse,datetime,json,shutil,subprocess
import numpy as np
import pandas as pd
from experiments.post_st_drop9_v1.deliver import table
from .common import *

def deliver(run,repo):
    reports=run/'reports';public=repo/'experiments'/NAME;v=load(run/'audit/final_validation.json')
    model=pd.read_csv(reports/'MODEL_COMPARISON.csv',dtype={'year':str,'seed':str});dr=pd.read_csv(reports/'DROP_COLUMN_ALL_83.csv');cx=pd.read_csv(reports/'EXPLANATION_COMPLEXITY.csv')
    internal=pd.read_csv(reports/'INTERNAL_MODEL_COMPARISON.csv',dtype={'year':str,'seed':str});groups=pd.read_csv(reports/'GROUP_DROP_COLUMN_RESULTS.csv',dtype={'year':str,'seed':str})
    primary=model[(model.seed=='ensemble')&(model.cause==1)&model.label.isin(['full83','legacy74','automatic_fallback83'])&(model['mode']=='count')]
    random=[];candidate=[]
    for (year,h,cause,mode),g in model[(model.seed=='ensemble')&model.supported].groupby(['year','horizon','cause','mode']):
        rr=g[g.label.str.startswith('random9_')];old=g[g.label=='legacy74']
        if len(rr)!=5 or len(old)!=1:continue
        for metric in ['auc','ap','brier','captured']:
            x=rr[metric].dropna();ref=float(old.iloc[0][metric])
            random.append(dict(year=year,horizon=h,cause=cause,mode=mode,metric=metric,legacy74=ref,random_sets=len(x),random_mean=x.mean(),random_min=x.min(),random_max=x.max(),
                random_better_than_legacy=int((x<ref).sum()) if metric=='brier' else int((x>ref).sum()),interpretation='five fixed random deletion sets, descriptive exposed-cohort comparison'))
    for (year,label,h,cause),g in model[(model.seed=='ensemble')&model.supported].groupby(['year','label','horizon','cause']):
        if set(g['mode'])!={'count','ratio'}:continue
        a=g[g['mode']=='count'].iloc[0];b=g[g['mode']=='ratio'].iloc[0]
        candidate.append(dict(year=year,label=label,horizon=h,cause=cause,count_candidates=min(25,int(a.features)),ratio_candidates=int(np.floor(a.features*25/83+.5)),
            delta_auc=b.auc-a.auc,delta_ap=b.ap-a.ap,delta_brier=b.brier-a.brier,delta_capture20=b.captured-a.captured))
    pd.DataFrame(random).to_csv(reports/'RANDOM_DELETION_CONTROLS.csv',index=False);pd.DataFrame(candidate).to_csv(reports/'CANDIDATE_COUNT_ABLATION.csv',index=False)
    pd.DataFrame([dict(outer=y,checkpoint=p,status='REFERENCE_EVALUATED' if p in [83,74] else 'NOT_REACHED_SUPPORT_GATE_CLOSED') for y in YEARS for p in [83,74,65,55,45,35]]).to_csv(reports/'SIZE_CHECKPOINTS.csv',index=False)
    r=dr[(dr.outer_cutoff==2025)&(dr.cause==1)&(dr.horizon==12)&dr.mature&dr.supported&(dr.known_termination_events>=1)&(dr.validation_target_events>=3)]
    ranking=r.groupby(['feature_id','feature_name','feature_label']).agg(delta_ap=('delta_ap','mean'),delta_auc=('delta_auc','mean'),delta_brier=('delta_brier','mean'),delta_capture20=('delta_capture20','sum'),
        cell_change=('spatial_cell_change','mean'),interactions_change=('interaction_count_change','mean'),positive_cohort_fraction=('delta_ap',lambda x:(x>1e-12).mean())).reset_index().sort_values('delta_ap',ascending=False)
    ranking.to_csv(reports/'DESCRIPTIVE_FEATURE_RANKING.csv',index=False)
    blocks=load(run/'input/group_specs.json');features=load(run/'input/features.json')
    blocktable=pd.DataFrame([dict(cutoff=g['outer'],group=g['group_id'],variables='、'.join(features[j]['label'] for j in g['members']),one='、'.join(features[j]['label'] for j in g['representative_one']),two='、'.join(features[j]['label'] for j in g['representative_two']),audit_block=g['audit_block']) for g in blocks])
    (public/'REDUNDANCY_GROUPS_ZH.md').write_text('''# 历史相关组与联合删除

主阈值为|Spearman|≥0.95、共同非缺失样本至少30、至少三个内部训练窗口且覆盖最新三个窗口。组为稳定边构成的连通分量，不保证组内任意两变量都达到阈值。另单独加入7列审计信息块。相关性、可得性及经济含义共同用于解释，不能仅据相关性认定可删除。

'''+table(blocktable)+'''

A为全83列；B删除整组；C保留训练期观测覆盖最高的一项；D在C基础上加入与其相关程度最低的另一项，作为待检验的互补候选；所有单一代表另行拟合供E选择。C/D的启发式不等于已证明互补。E因目标事件支持条件未通过，保守返回A，未以外层结果选择代表。

所有保留列的值、NaN和审计状态逐值保持。删掉审计状态列只减少信息，不重写其余列或拼造审计意见。A及同一子集复用同组真实训练输出，不把同配置重复命名冒充不同模型。

阈值敏感性如下。这里检验的是组结构敏感性；0.90/0.99没有额外按其组结构做整套预测搜索，不能宣称预测选择对阈值稳健。

'''+table(pd.read_csv(reports/'group_sensitivity.csv'))+'\n\n逐年度、每种子、合并内部队列及全部四个结局/时点的实际重训比较见GROUP_DROP_COLUMN_RESULTS.csv。\n')
    frozen=load(run/'audit/freeze.json');gates={k:load(run/f'audit/gate_{k}.json') for k in ['pilot','inner','bootstrap','outer','aggregate_inner','aggregate_bootstrap','aggregate_outer']}
    testlog=(run/'audit/tests_final.log').read_text();assert '198 passed' in testlog and 'failed' not in testlog.lower()
    (public/'RESULTS_ZH.md').write_text(f'''# 全变量Drop-column重训与自动特征选择结果

## 结论

**全83项删除、相关组、随机删9项及候选比例对照均已真实运行；本轮没有批准新的自动删除集合。** 三个截止点的12个月历史有效事件均不足，递归程序按冻结门槛在第0轮停止，保留83列。不能把这一回退称为找到了最优83列，也不能据此证明不存在更好的子集。

新增森林拟合{v['actual_new_forests']}次：试运行3、内部2538、公司重抽样450、外层固定对照99；全部成功。复用18个外层83/74参考森林，另编译1035个三种子完整预测器。保留全部预设正面、负面及NA结果，没有额外调参或替换当前模型。

## 预测表现

主目标24个月先最终退市、次目标12个月；自动选择按预设12个月AP与Top20优先，24个月及恢复/Brier作保护。各年先平均3个种子的逐公司CIF后评分。Top20每年向上取整；pooled重新计算AUC/AP/Brier，Top20名额按年度分别确定再合计。

{table(primary[['year','label','horizon','companies','target_events','auc','ap','brier','risk_mean','observed','checked','captured','coverage','supported']])}

自动回退与原83列使用同一预测，未冒充新增拟合。legacy74是历史固定删9项对照；近3个月总市值变化的冗余诊断直到2024截止才成立，因此该名单不能作为2023严格前向自动筛选的先验。本轮自动筛选没有使用它。

2025观察截止2026-06-30，12个月全部可评价，24个月未获得时点支持。其24个月指标留NA，pooled24仅含2023/2024。未知结局不当作正常或误报。2023–2025均已暴露，只能描述性比较。概率低估没有通过调整测试概率掩盖。

## 全83项逐一删除

DROP_COLUMN_ALL_83.csv包含4980行：三个外层信息截止下的15个内部年度组合×83变量×两个时点×两个原因。全部83项覆盖，保留每行基准/删除后指标、事件支持、复杂度变化、种子波动和500次配对公司bootstrap区间。各内部模型只使用其入组年前的信息拟合，验证标签截到对应外层截止。没有用外层排名来决定删除。

以下为2025截止前有效历史队列的**年内AP差均值**，正值为删除有利；它不是合并AP，也不是安全删除名单。该比较仅有2020、2023两个小事件队列，不能凭点估计决定保留/删除。

{table(ranking.head(10))}

删除后AP下降最多的候选如下；这些结果提示信息损失的可能性，尚不足以认定全局不可替代。

{table(ranking.tail(10).sort_values('delta_ap'))}

同一变量删除后，全森林的候选覆盖和分裂会一起变化，因此Drop-column衡量的是当前训练算法在该信息集合下的表现变化，不是变量的因果作用。不能用“差异不显著”推断性能等效。

值得注意：2023/2024截止时唯一符合主要排名条件的2020验证队列只有3例退市，83种单项删除的AP点估计全部提高；2025截止加入2023队列后，75项均值提高、8项下降。这不是83项都冗余的证据。基础模型已与历史模型逐种子精确复现，删列仍会重新安排候选覆盖及后续搜索，因此本结果同时包含输入信息变化和训练算法变化，少事件排名尤其不能直接转成删除名单。

## 分组、随机删除和候选比例

主阈值0.95得到26个截止点—组定义（含审计块）。A/B/C/D及E候选均用真实森林；相同有效输入与预算允许复用。具体组、经济名称与0.90/0.99组结构敏感性见REDUNDANCY_GROUPS_ZH.md。

随机删9项使用5个预先固定的变量抽样种子，每组均用3个森林种子。下表为已暴露外层12个月先退市AP；完整四个时点/原因和全部指标见RANDOM_DELETION_CONTROLS.csv。

{table(pd.DataFrame(random).query("horizon==12 and cause==1 and metric=='ap' and mode=='count'"))}

固定候选数量为25；固定比例按round(p×25/83)取整，74列对应22。两种模式的差值见CANDIDATE_COUNT_ABLATION.csv。它可以显示覆盖预算的影响，不能把所有删变量收益都归为消除噪声。

实际结果没有给出统一胜出的删列方案：既有74列的12个月AP在2023/2024超过五个随机删除对照，但在2025有三个随机对照超过它。2024年把74列的每树候选从25改为22后，AP从0.799152降至0.686158，低于83列的0.718365，说明候选预算会明显改变结论。随机对照只限本轮五组，不能视为删列空间的完整分布。

虽然74列的12个月年内AUC/AP在三年均高于83列，重新合并三年后AUC却从0.645138变为0.641101，AP从0.246877变为0.246299，不能宣称跨年度排序改善。24个月合并AP从0.681719下降到0.623033，Top20捕获从12/32下降到11/32；这也是保留当前参考的重要描述性依据，未用其反向调整历史选择规则。

**执行偏差：**清单按“子集+模式”去重，未完全按取整后的有效配置去重。141个模式子集中实际为132个不同有效配置，9对等效模式多拟合162次。全部保留、计入成本，逐对预测差异已核验；未当作额外种子或独立证据。冻结协议原定的等效构型去重没有完整实现，见scheduling_deviation.json。

## 内生解释与稳定性

{table(cx[cx.label.isin(['full83','legacy74'])&(cx['mode']=='count')][['year','label','features','main_functions','interaction_functions','spatial_table_cells','scalar_table_cells','distinct_thresholds']])}

所有新raw、收缩森林及完整三种子预测器均验证q≤2、CIF非负/随时间不减/原因和≤1、保存重载与全函数重构；覆盖全部训练及评价行、所有原生时点与两原因，最大误差{v['max_exact_reconstruction_error']:.3e}，小于1e-10。没有删除已训练非零函数、裁剪概率或使用事后代理模型。

单种子与合并预测器复杂度分别保留；空间格数含缺失状态，不乘时间/原因维，标量格数另计这些维度。INTERACTION_STABILITY.csv报告50次真实重训中的函数出现频率、幅度、相对原模型形状相关与差异，均使用相同年度原训练集的经验边际乘积参考。相关特征下，该参考会包含现实中稀少的组合；精确重构不等于真实效应恢复或统计可辨识。

每年度50次公司重抽样、每次3个种子，重新拟合的是因支持不足而回退的83列模型。所有150次重抽样的支持门槛仍未通过。FEATURE_SELECTION_STABILITY.csv中的选择频率明确为NA；回退保留率100%不能证明变量不可删除。未执行100次扩展，也未强迫运行65/55/45/35列。单变量、分组、递归与稳定性筛选四种自动流程均在同一前置支持条件回退，不能声称复杂流程优于简单删除。

LOCAL_EXPLANATION_CHANGES.csv在各模型相同的8条历史输入上比较全贡献向量与CIF变化；逐公司值仅保留私有目录。它用于判断作用是否重新分配到其他变量，不把风险百分点贡献称为因果影响。

## 计算与可复现性

训练冻结提交`{frozen['commit']}`。198项完整测试（含R）通过。逐任务哈希、配置、子集哈希、种子、训练截止、作业、内存与耗时保留。首次准备因一行非入模chain_gap诊断差异停止，定位后沿用最早历史诊断，实际入模字段、结局与资格检查未放宽。未发生训练失败。

{table(pd.DataFrame([dict(stage=k,job=g['job'],tasks=g['tasks'],seconds=g['seconds'],hosts=json.dumps(g['by_host'])) for k,g in gates.items()]))}

用户授权两节点，任务单线程，由现有Slurm负责P/E核和SMT亲和性，不硬编码CPU。资源实测来自执行日志；不将不同节点时间当作严格算法速度比较。当前指针及所有历史源哈希不变。私有运行目录：`{run}`。

## 图表

![83项删除排名](figures/drop_column_all83_ranking.png)
![12个月性能与变量数](figures/feature_count_performance_12m.png)
![24个月性能与变量数](figures/feature_count_performance_24m.png)
![变量数与交互数](figures/feature_count_interaction_complexity.png)
![选择门槛与回退模型稳定性](figures/feature_selection_stability_heatmap.png)
![预测与复杂度前沿](figures/prediction_complexity_pareto.png)
![年度对照](figures/annual83_74_automatic_comparison.png)
''')
    (public/'METHOD_INNOVATION_DECISION_ZH.md').write_text('''# 方法研究判断

1. **是否找到优于83/74的自动集合？** 尚未得到符合冻结历史支持及保护条件的新集合，自动结果回退83。12/24个月、年内及pooled排序、概率误差均按真实预测报告，部分候选的单点提升不能替代支持。不存在“已经证明83最优”的结论。
2. **哪些变量稳定且不可替代？** 本轮全83删除和联合组试验提供了训练算法条件下的增量证据；12个月目标事件不足，尚不能给出可靠的全局不可替代名单。单项可替代不意味着整组无信息，组内单/双代表结果及交互幅度必须一起看。
3. **哪些可以安全删除？** 本轮没有新增获准安全删除名单。既有9项只是既有固定对照；确定性重复、稳定高相关与PFI低贡献都不能单独替代重复重训、时序支持及保护指标。置信区间跨零不等于等效。
4. **自动筛选是否增强内生解释？** 精确CIF表示、概率合法性与q2结构保留。已量化所有候选的复杂度及相同历史样本下的贡献再分配；但自动流程未批准删列，所以自动方法的简化和稳定性增量尚未证明。50次重训提供的是回退模型函数稳定性，选择频率仍NA。
5. **是否成为正式训练方法？** 目前保留为研究审计工具，不纳入自动删列的正式默认训练。Drop-column和后向选择属于训练前模型选择；q2、共享切点等属于森林内部约束；精确函数导出属于模型的代数表示，三者不能混作一种新算法。此次计算支持可审计的工程流程，不单独构成已证实的方法创新。核心价值仍是可完整重构的内生解释。
6. **是否替换当前模型？** 否。当前83列参考及历史74列结果全部保留。模型升级仍需要未用于选择且随访充分的独立时间外证据和研究者明确批准。

研究状态：**自动特征选择增量证据不足。** 已完成核心全量诊断及预设对照；递归后续轮次、局部恢复/交换、65/55/45/35规模与真正选择稳定性因事件支持门槛停止，未将这些未执行部分冒充完成。
''')
    (public/'METHODS_RESULTS_ZH.md').write_text(f'''# 论文方法与结果表述

在固定q2、深度4、每种子32树、共享分位切点和强度10直接父节点Aalen–Johansen收缩的竞争风险森林下，我们对83项输入逐一执行删除后完整重训。内部模型分别以2019–2024各年1月1日为信息截止，并在2023、2024、2025三个外层截止下截断内部验证标签。相关组由历史训练窗口的稳定Spearman关系定义，另设随机删除9项和候选数量/比例对照。所有模型保留同公司bootstrap随机流、缺失状态及全部函数项。每组三个种子先平均公司CIF再评分，分别评价12/24个月先最终退市与先恢复，未知或不支持终点保留NA。

共实际拟合3090个森林，包括3个试运行、2538个内部候选、450个公司重抽样模型和99个外层固定对照；其中162次内部拟合属于取整后等效配置的重复计算，完整保留且不作为独立证据。完整表示最大重构误差为{v['max_exact_reconstruction_error']:.3e}。历史12个月先退市事件未达到预设支持条件，因此递归在第0轮停止，三个截止均回退83列。此结果不证明不存在更优子集，也不能用回退保留率衡量统计选择稳定性。2023–2025已暴露结果仅用于描述性复核，当前冻结模型未替换。
''')
    # Public evidence contains no company rows, cutpoints or model parameters.
    for path in reports.glob('*.csv'):shutil.copy2(path,public/path.name)
    shutil.copytree(reports/'figures',public/'figures',dirs_exist_ok=True)
    aud=public/'audit';aud.mkdir(exist_ok=True)
    for path in (run/'audit').glob('*'):
        if path.suffix in ['.json','.log','.txt'] and path.name not in ['registry_inner.json','registry_bootstrap.json','registry_outer.json','registry_aggregate_inner.json','registry_aggregate_bootstrap.json','registry_aggregate_outer.json']:
            shutil.copy2(path,aud/path.name)
    for filename in ['features.json','subsets.json','arms.json','group_specs.json','support_gates.json']:shutil.copy2(run/'input'/filename,public/filename)
    receipt=dict(status='COMPLETE_CORE_DIAGNOSTICS_WITH_SUPPORT_GATED_SELECTION',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),private_run=str(run),branch='research/xai-rsf-dropcolumn-selection-v1',
        source_reference_commit=SOURCE_COMMIT,fitting_commit=frozen['commit'],validation=v,jobs=gates,current_reference_replaced=False,current_reference_sha256=REFERENCE_SHA,
        full_tests_passed=198,independent_R_checks=True,failed_fits=0,stopped_preparation_attempts=1,unexecuted_due_to_support=['recursive rounds after0','local restoration/swaps','65/55/45/35 checkpoints','genuine feature selection stability','100-repeat expansion'],
        public_csv_sha256={p.name:sha(p) for p in public.glob('*.csv')},source_model_predictions_private=True)
    write(public/'execution_receipt.json',receipt);write(run/'audit/execution_receipt.json',receipt)
    print(json.dumps(dict(status=receipt['status'],fits=v['actual_new_forests'],ranked_features=dr.feature_id.nunique(),public_csvs=len(receipt['public_csv_sha256']))),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();deliver(a.run,Path(__file__).resolve().parents[2])
