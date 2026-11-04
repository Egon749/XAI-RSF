"""Render the Chinese audit from the archived aggregate outputs."""
from .common import *
import pandas as pd
def rd(name):return pd.read_csv(HERE/(name+'.csv'))
def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in rows)
def make():
    m=rd('METRIC_DISTRIBUTIONS').query("version=='shrunk'").set_index(['scope','months','cause','metric'])
    scopes=[('2023',12),('2023',24),('2024',12),('2024',24),('2025',12),('pooled',12),('pooled',24)]
    def row(y,h,c,k):return m.loc[(y,h,c,k)]
    def interval(z):return 'NA' if not np.isfinite(z['median']) else f"{z['median']:.3f} [{z.p05:.3f}, {z.p95:.3f}]"
    def predtab(c):
        data=[]
        for y,h in scopes:
            z=row(y,h,c,'captured');a=row(y,h,c,'auc')
            data.append([y,h,interval(a),interval(row(y,h,c,'ap')),f"{row(y,h,c,'brier')['median']:.4f}",f"{int(z['median'])} [{int(z['min'])}–{int(z['max'])}]",f'{a.reference96:.3f} / {a.ensemble320:.3f}' if np.isfinite(a.reference96) else 'NA / NA'])
        return table(['范围','日历月','AUC 中位数 [P5,P95]','AP 中位数 [P5,P95]','Brier 中位数','Top20 捕获中位数 [全范围]','原96 / 320 AUC'],data)
    p=rd('PREDICTION_STABILITY').query("version=='shrunk' and cause==1")
    l=rd('LOCAL_EXPLANATION_STABILITY').query("version=='shrunk' and cause==1")
    struct=rd('STRUCTURE_JACCARD').query("version=='shrunk' and shared_members=='0'").set_index(['year','order'])
    def val(frame,year,h,metric):return float(frame[(frame.year==year)&(frame.months==h)&(frame.metric==metric)]['median'].iloc[0])
    srows=[]
    for ys,h in scopes[:5]:
        y=int(ys)
        srows.append([y,h,f"{val(p[p.shared_members=='0'],y,h,'rank_spearman'):.3f}",f"{val(p[p.shared_members=='0'],y,h,'top20_jaccard'):.3f}",f"{100*val(p,y,h,'company_risk_range'):.2f}",f"{val(l,y,h,'top5_jaccard_0'):.3f}",f"{val(l,y,h,'top10_jaccard_0'):.3f}",f"{100*val(l,y,h,'l1_distance_0'):.2f}"])
    stability=table(['年','月','整体排名 ρ','Top20 Jaccard','企业风险极差中位数（百分点）','局部 Top5 Jaccard','局部 Top10 Jaccard','局部贡献 L1（百分点）'],srows)
    cx=rd('COMPLEXITY_TRADEOFF').query("version=='shrunk'").copy()
    cx['group']=np.where(cx.model.str.startswith('C'),'三成员96',np.where(cx.model=='reference96','原96',np.where(cx.model=='ensemble320','全部320','单成员32')))
    crows=[]
    for y in YEARS:
        for g in ['单成员32','三成员96','原96','全部320']:
            t=cx[(cx.year==y)&(cx.group==g)];z=t.median(numeric_only=True)
            crows.append([y,g,f'{z.main_functions:g}',f'{z.interaction_functions:g}',f'{z.spatial_table_cells:g}',f'{z.scalar_table_cells:,.0f}',f'{100*z.zero_support_cell_fraction:.1f}%',f'{100*z.low_support_cell_fraction:.1f}%',f'{z.function_bytes/1024:.1f}'])
    complexity=table(['年','树预算','主函数','二元函数','空间格','含时间及原因的标量格','无训练支持','少于10家公司','压缩 KiB'],crows)
    exact=read(HERE/'RECONSTRUCTION_AUDIT.json')
    text=f"""# XAI-RSF 三种子组合稳健性审计

同一批企业的风险排序大体可重复，主要审计变量的贡献方向也能复现。具体二元搭配与个体的前几项解释对种子更敏感。每一套组合的函数表都能精确还原自身预测。以 2024 年 12 个月退市为例，完全不共享种子的两套组合，企业排序相关性中位数为 0.929，二元函数集合的 Jaccard 为 0.183，单家企业前五项贡献的 Jaccard 中位数为 0.500。固定 320 树提高部分预测指标，同时使该年二元函数数从原 96 树的 135 增至 317。

### 数据、参照与比较范围

从已保存的十个种子枚举 120 个三成员组合，每成员 32 棵。冻结提交为 `4f1b163637f8eb5027dbe7d1b54595accfd431b0`。年度训练截止日均为 1 月 1 日，训练企业数为 309、345、390，评价企业数为 36、45、53。训练使用旧版 74 项输入、下一交易日开盘前的入组信息、最多 24 个日历月的随访、先最终退市与先恢复的竞争编码。既有披露版本限制及删九项的历史诊断时间继续适用。原 96 树锚点固定为 74 项、退市支持保护、42/101/202 的组合。仓库根指针记录旧 83 项规格且未覆盖全部三个年度，两者的特征与支持约束不同，根指针版本排除出种子对照。已按哈希核对各来源的输入顺序、训练和评价快照、已观察标签，核对配置、树及直接父节点 AJ 收缩。全局函数统一使用当年训练边际乘积分布。任何评价企业均未参与确定背景。

三个年度和两个版本共 720 份新组合函数表示，无新增森林或校准器拟合。比较限于这十个已查看过年度结果的种子。120 个组合共 7,140 个无序比较，其中 2,100 对无共同成员，3,780 对共享一个，1,260 对共享两个。下列区间是有限集合分位数，各组合高度依赖，没有做独立样本推断或选优。

### Q1　退市预测随种子变化多少

{predtab(1)}

年度排序和风险水平是不同的问题。2024 年 12 个月各组合平均退市概率的 P5–P95 为 3.52%–3.83%，观察发生率为 35.56%。这一系统性低估远大于种子之间的水平变化。原 96 树在 2024 年 24 个月的 AUC 为 0.670，低于这十个指定种子的全部三成员组合（最低 0.720）。2025 年 12 个月则是原 96 树捕获 3/5，320 树捕获 2/5。成员名单形成于年度成绩已被查看之后，这些现象描述种子敏感性，不能据此更换主模型。

合并范围先合并企业预测并重算 AUC/AP/Brier，Top20 先按年向上取整再相加。按同一年事件—非事件配对数加权的退市 AUC 中位数在 12、24 个月分别为 {row('pooled',12,1,'within_year_auc')['median']:.3f}、{row('pooled',24,1,'within_year_auc')['median']:.3f}，跨年混合配对的 AUC 见表。年度参考位置见 METRIC_DISTRIBUTIONS.csv（并列计半权的百分位）。2025 年 24 个月仅检验预测函数和数值稳定性，性能 NA。

### 恢复和原始叶估计

{predtab(2)}

2023 年 12 个月没有恢复事件，该年恢复 AUC/AP 保留 NA。ALL_METRICS.csv 对 raw 和 shrunk 逐一提供同一套年度与合并指标、事件分母、均值、校准差和原因。raw 版本各项解释、数据支持和局部结果也在相应表中；正文统一使用当前父节点收缩版。

### Q2　模型对哪些风险因素看法一致

{stability}

上述排名和名单指标比较完全不共享种子的组合。风险极差为同一企业在全部 120 个组合中最高减最低，然后对企业取中位数。局部指标先在该企业的组合对中取中位数，再跨企业取中位数。Top5/10 排名依据贡献绝对值，仅纳入至少 0.1 个 CIF 百分点的项，可能不足五或十项。L1 是全部主效应和交互贡献变化绝对值之和，局部基准另列。2024 年 12 个月，两套组合常能认同某家企业风险较高，但对哪五项最能说明其风险只部分一致。预测差与贡献 L1 差的企业内相关性中位数约 0.24，正负项能相互抵消。稳定／不稳定真实个案按全部公司的 L1 第10／90百分位选取，明细及有符号图仅私有保存。公开图展示同年度最低和最高两个五分位组的聚合剖面。

不共享成员的主效应集合 Jaccard，三个年度分别为 {struct.loc[(2023,1),'median']:.3f}、{struct.loc[(2024,1),'median']:.3f}、{struct.loc[(2025,1),'median']:.3f}；二元交互分别为 {struct.loc[(2023,2),'median']:.3f}、{struct.loc[(2024,2),'median']:.3f}、{struct.loc[(2025,2),'median']:.3f}。2024 年两套组合共享两个成员时，交互 Jaccard 升至 0.601。因此，全组合两两比较会高估两个全新种子组的解释重合程度。交互在两个组合均存在且形状可计算时，仍要逐项看数值；365 天退市、实际联合训练支持上，各交互的形状相关中位数再次跨交互取中位数，三个年度为 0.503、0.525、0.463。在材料性门槛两侧移动的微弱贡献，不能只凭出现次数解释为主要因素。

### Q3　有训练支持的可重复贡献

在 365 天退市模型中，上一期标准审计意见、持续经营审计提示和近一月最大回撤的主效应均通过三个年度的预设描述筛查。2024 年，标准意见取 1 的居中贡献为负，取 0 的贡献为正；各组合在训练企业上的平均绝对贡献中位数约为 0.81 个百分点。持续经营提示取 1 对风险的贡献为正，平均绝对贡献约为 0.55 个百分点。二元指标只在少数状态上比较，形状相关接近 1 时，仍需报告幅度分布。

在 2023、2024 年，没有二元项通过 365 天退市的全部预设条件。2025 年有一个，最近股价对数与上一期持续经营审计提示。它出现在 116/120 个组合，实测联合支持上的平均绝对贡献中位数为 0.198 个百分点，P5–P95 为 0.059–0.361 个百分点。完全不共享成员且都有该项的 1,960 对组合，形状相关中位数为 0.996，主要正贡献区域 Jaccard 为 0.667。约 23% 的组合平均绝对贡献低于阈值。该项公共网格有 21 格，约 43% 没有实际训练公司，3 格同时有至少 10 家企业和 3 家已观察到该原因事件的公司。该交互在 2024 年的非零出现率为零。证据仅支持 2025 年训练快照中的可重复模型模式。各单元独立企业和已观察事件的完整计数私有保留。未观察到目标事件的训练企业可以包含删失，支持计数不能作发生率估计。

公开频率表覆盖全部 74 项及所有可能变量对，包括从未出现的项。主效应与交互表将全原生时间非零、目标时点幅度接近零、方向一致程度、非恒定共同形状与 RMS 差分开。真实训练联合单元按公司数加权；无支持单元单独按格等权，仅描述函数外推。当没有相应区域时不报告该行；共同函数恒定、无共同材料性单元或无正贡献高值区域时，相应比较可为 NA。边际乘积仅定义函数中心，不能替代真实企业与事件计数。

### Q4　32、96、320 树的取舍

{complexity}

单成员与三成员行对同预算模型取中位数。空间格含每一维的缺失状态，标量格乘以原生时间数与两类原因，存储为无损因子化文件。编译与验证耗时见 COMPLEXITY_TRADEOFF.csv。三成员验证复用成员级显式求和，历史运行的验证器与负载不同，耗时不宜直接做加速比。

以完全独立于这十个成员的固定原 96 树为锚，2024 年 12 个月退市整体排序相关中位数在 32、组合 96、320 树依次为 0.857、0.907、0.917；共同交互的形状相关则约为 0.633、0.609、0.622。320 树的预测更接近该锚点，交互形状的一致性未呈同样趋势。320 树只有一套，其与三树组共享成员，无法从中估计独立重复训练多个 320 树预测器的变异。对兼顾曲线可审计性和排序的论文，固定原 96 树作为参照有依据，32 树提供较简洁的对照，320 树用于说明集成上限下的权衡。没有面向未来队列的最优树数结论。

### Q5　论文可支持的结论

全部训练和评价样本、全部原生时间、0 天及四个日历终点天数、两个原因均已核验。组合森林与成员均值、导出与森林、重载与森林、显式逐项求和与森林的最大绝对差分别为 {exact['max_errors']['member_mean']:.3g}、{exact['max_errors']['reconstruction']:.3g}、{exact['max_errors']['reload']:.3g}、{exact['max_errors']['explicit_sum']:.3g}。所有路径不超过两个原始变量，CIF 非负、非递减、两原因之和不超过 1。另对每年度和版本的第 1、60、120 个合并导出单独进行完整逐项加总，核对原 96 树的逐树成员身份及实际联合格的公司和事件合计。未裁剪概率。

可支持完整数学分解与这些年度样本上总体排序及部分主效应的重复性。跨年度稳定的关键退市交互、所有企业解释项目的可复现性、统计显著性、经济因果机制以及独立未来数据上的性能增益均缺乏充分依据。审计保持原模型，未选择任何一个三成员组作为新主模型。
"""
    (HERE/'RESULTS_ZH.md').write_text(text)
if __name__=='__main__':make()
