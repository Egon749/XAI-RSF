"""Assemble the Chinese report from audited aggregate tables."""
from .common import *
def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n|'+'|'.join('---' for _ in headers)+'|\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in rows)
def read_table(name):return pd.read_csv(D/(name+'.csv'))
def main():
    compare=read_table('GROUP_VS_FEATURE_COMPARISON')
    comp=compare.query("months==12 and cause==1 and shared==0")
    get=lambda y,l,m: comp[(comp.year==y)&(comp.level==l)&(comp.metric==m)].iloc[0]
    labels=read_table('FEATURE_GROUP_MAPPING').set_index('group').group_zh.to_dict()
    crows=[]
    for y in YEARS:
        for lev,p in [('C8','variable_full74'),('C4','financial_variables35')]:
            a=get(y,lev,'relative_l1');b=get(y,lev,'rank_rho')
            crows.append([y,'八类 / 74变量' if lev=='C8' else '四能力 / 对应35变量',f'{a[p]:.3f}',f'{a.economic:.3f}',f'{a.random_median:.3f} [{a.random_p05:.3f}, {a.random_p95:.3f}]',f'{b[p]:.3f}',f'{b.economic:.3f}',f'{b.random_median:.3f}'])
    ctable=table(['年','范围','变量 L1','经济 L1','随机 L1 中位数 [P5,P95]','变量 ρ','经济 ρ','随机 ρ 中位数'],crows)
    groups=read_table('GROUP_MAIN_CONTRIBUTIONS').query("level=='C8' and shared==0 and months==12 and cause==1")
    four=read_table('FOUR_CAPABILITIES_STABILITY').query("shared==0 and months==12 and cause==1")
    c=[]
    for _,z in four.sort_values(['year','group']).iterrows():
        c.append([z.year,labels[z.group],f'{100*z.mean_abs_median:.3f} [{100*z.amp_p05:.3f}, {100*z.amp_p95:.3f}]',f'{100*z.first_rank_frequency:.1f}%',f'{100*z.signed_mean_median:+.3f}',f'{100*z.positive_material_fraction:.1f}% / {100*z.negative_material_fraction:.1f}%',f'{z.sign_agreement:.3f}' if np.isfinite(z.sign_agreement) else 'NA',f'{100*z.both_material_fraction:.1f}%'])
    ftable=table(['年','能力','平均绝对贡献中位数 [P5,P95]（百分点）','四项中排第一','平均有符号贡献中位数（百分点）','正 / 负实质贡献占比','条件方向一致率','均达阈值占比'],c)
    canc=read_table('GROUP_CANCELLATION_AUDIT').query("map_id==0 and cause==1 and months==12")
    ca=[]
    for g,a in canc.groupby('group'):
        a=a.sort_values('year');ca.append([labels[g],*[f'{100*x:.1f}%' for x in a.cancellation_mean],f'{100*a.gross_mean.iloc[1]:.3f}',f'{100*a.abs_net_mean.iloc[1]:.3f}'])
    canctable=table(['组别','2023抵消','2024抵消','2025抵消','2024平均绝对总和（百分点）','2024平均净额绝对值（百分点）'],ca)
    agg=[]
    for _,a in compare.query("shared==0 and level=='C8' and metric=='relative_l1'").sort_values(['cause','months','year']).iterrows():
        b=compare.query("shared==0 and level=='C8' and metric=='rank_rho' and year==@a.year and months==@a.months and cause==@a.cause").iloc[0]
        agg.append([a.year,a.months,'退市' if a.cause==1 else '恢复',f'{a.variable_full74:.3f}',f'{a.economic:.3f}',f'{a.random_median:.3f}',f'{b.economic:.3f}',f'{b.random_median:.3f}', '仅函数' if a.year==2025 and a.months==24 else '函数'])
    stable=read_table('GROUP_STABILITY');crow=[]
    for lev in ['native_terms','variable_full74','A8','C8','C4','I28']:
        d=stable.query("year==2024 and months==12 and cause==1 and level==@lev")
        row=[lev]
        for s in [0,1,2]:
            a=d[d.shared==s].set_index('metric')['mean']
            row.append(f"{a.rank_rho:.3f} / {a.relative_l1:.3f} / {a.top_jaccard:.3f}")
        crow.append(row)
    ir=read_table('GROUP_INTERACTIONS').query("months==12 and cause==1 and shared==0 and support_pass")
    intertable=table(['年','跨组项','评价幅度（百分点）','条件方向一致率','一致且有实质贡献的训练企业','已观察退市企业','主导原始变量对所在格亦有支持的企业'],[[z.year,' × '.join(labels[g] for g in z.group.split('×')),f'{100*z.mean_abs_median:.3f}',f'{z.sign_agreement:.3f}',z.any_consensus_firms,z.any_consensus_events,int(z.firms_with_supported_dominant_pair)] for _,z in ir.sort_values(['year','group']).iterrows()])
    equation=r'$$F=\mu+\sum_G A_G+\sum_{G<H}I_{GH}=\mu+\sum_G C_G,\qquad C_G=A_G+\tfrac12\sum_{H\ne G}I_{GH}.$$'
    report=f"""# 财务能力及经济信息分组解释稳定性审计

**建议在论文中加入分组解释，用于展示原预测如何由经济类别的贡献组成，同时报告随机分组对照。** 在 2023—2025 年的 12 个月退市模型中，八组完整归因的相对 L1 差异都低于 74 项原始变量归因；与 200 套等规模随机分组相比，没有一年同时达到预先固定的幅度与排名优势标准。审计信息的组贡献在三个年度的所有三种子组合中均排第一。四大能力的局部排序仍随种子变化，尤其不能把某一企业的最大财务风险驱动固定为某一种能力。

## 设计与分组账目

以种子审计的冻结 74 变量模型为起点，沿用旧输入、q≤2、深度4、每成员32树、公司级重采样、均衡25项、共享切点、退市支持保护和直接父节点AJ收缩10。年度截止1月1日，下一交易日开盘前入组，先退市和先恢复为两个竞争原因。原96树固定为42/101/202组合，与根目录旧83项指针分别保留。各年源材料、已保存的森林、预测、函数、企业快照和分组命名均校验哈希。既有数据版本和事后暴露限制继续适用。

120 组三种子组合共享十个已保存成员，以真实评价企业为单位比较。首要分析使用 2,100 对完全不共享成员的组合，共享一个和两个成员的组合对数分别为 3,780、1,260。先按企业平均有效配对指标，再给企业等权。随机对照固定 200 套，组规模依次为 7、12、8、8、6、21、7、5，每一套在所有年度、原因、时限和种子上复用。风险贡献依当年训练边际乘积分布居中，在真实训练或评价记录上读取。本研究没有在独立新队列验证分组，也没有拟合或选择森林。

{equation}

每个 A 包含组内主效应及组内原始交互，每个 I 包含两组之间所有原始二元函数。C 给跨组两方各分一半，固定约定可精确回算。对照变量 V 同样分得每个相关原始交互的一半，故组内 V 加总等于 C。原始函数保留，可逐项核查。增长类现金流指标归发展，现金流量水平归现金流组；现金存量归偿债。财务费用率与应计强度从费用侵蚀和盈利质量角度归盈利，应付账款结构从营运资金角度归营运。固定资产占比与周转效率含义不同。全部74项一一对应；未按稳定性调整映射。

## 1　同一企业的解释更稳定了吗

{ctable}

L1 已除以两向量的绝对贡献总量，越小表示贡献幅度越接近；ρ 衡量同一企业内部贡献绝对值排序的一致性，越大越好。经济四组相对匹配的 35 个变量，三个年度的 L1 较小，但排名相关均下降。八组相对74项的 L1 和排序相关均有改善；2024年的排序改善仅 0.514→0.523。原始变量排序含缺席和零值的并列秩，不应将跨不同维数的绝对相关系数理解为等价任务。

经济八组的 L1 在 2023 年好于随机中位数，但在 2024、2025 年比随机中位数高。四个财务组在三个年度的 L1 均高于同大小随机四槽的中位数，排名相关也均低于随机中位数。随机槽可混入强审计或市场信号，因此对照检验的是这套经济标签是否具有额外的聚合优势，随机槽位本身不能解释为财务能力。预注册的重复优势标准为经济 L1 不高于随机 P5 且 ρ 不低于随机 P95；八组和四组的 12 个月退市均未在任何年度同时通过。没有对 120 组三种子或 200 套映射做独立样本显著性推断。

2024 年 12 个月退市的重叠敏感性。每格依次列 ρ、相对 L1、Top 集合 Jaccard。

{table(['对象','共享0个','共享1个','共享2个'],crow)}

原始项取 Top5，八组取 Top3，四能力取 Top1。要求局部绝对贡献至少0.1个百分点，两份解释中均为零或名单为空时，相应比较按定义处理为 NA。完整 CSV 同时提供符号一致率、有效配对计数、联合达到阈值比例及非归一化 L1。不能仅根据 Top 集合重合度的变化判定分组有效。

## 2　四大能力、审计、市场和现金流

{ftable}

“排第一”按每个组合在当年全体评价企业上的平均绝对贡献排序。正负实质占比以企业与组合为单位。方向一致率仅对同一企业、两组合均达阈值的比较定义；最后一列揭示其适用范围。2023 年发展能力没有满足条件的不重叠配对。偿债贡献的平均绝对幅度三年接近（约0.17–0.18个百分点），但排第一的组合比例从86.7%降到38.3%、7.5%。营运在2024、2025年排第一的比例为61.7%、92.5%。盈利与发展整体幅度较小，方向覆盖有限。偿债、营运的跨企业平均净贡献中位数三年均为正，同时个体有正有负，净贡献也随队列构成改变，不能解读为“提高偿债或营运能力会增加退市风险”。

审计信息三个年度全部组合居八组之首，三个年度平均绝对贡献中位数分别为0.667、1.053、2.338个百分点，达到材料性阈值的两套组合间符号一致率为1.000、0.995、0.991，联合阈值覆盖分别为95.1%、92.2%、95.1%。市场组的中位名次三年均为第二，方向一致率为0.927、0.801、0.886。在同年原96树锚点上，审计组平均绝对贡献分别为0.651、0.921、2.022个百分点；320树上分别为0.667、1.051、2.340个百分点。两组固定森林也显示审计组有较大贡献，但它们各只有一个年度预测器，不能由单点估计其独立重训稳定性。现金流组的相应阈值覆盖仅0.9%、10.8%、12.9%，2025年方向一致率为0.653。2023年现金流条件一致率虽为1，不能据此推广到全体企业。

## 3　抵消与跨组分配

{canctable}

抵消比例是单家公司、单一组合中 1−|A|/原始项绝对贡献总量 后的均值。分母为零时舍去该比值，原始项仍在完整重构中保留。不同权重使此比例与表中两个均值的比值不必相等。营运组约55%–57%的组内原始绝对贡献发生抵消。2024年市场组抵消约68%。归并后较短的解释表可以隐藏较大的正负项及它们之间的变化。仅看内部 A，三年八组相对 L1 为0.193、0.243、0.202；分摊全部跨组交互后 C 为0.227、0.273、0.215。对这批模型，加入跨组交互的真实波动削弱了一部分表面稳定性。

## 4　有实际企业与退市事件支持的跨组交互

{intertable}

支持区域在实际训练企业上定义：某公司的跨组 I 在至少90%组合中同向且绝对值≥0.1个百分点；该区域还须有至少10家公司、至少3家截止日前已观察到对应终点的公司，并满足组合训练平均绝对幅度中位数≥0.1个百分点。没有把删失当作无事件。附加诊断用固定320树在该公司的最大绝对原始二元项，核查它所在的联合格是否也有≥10家公司和≥3例对应事件，不重复计同一企业。隐私规则下较小事件数以范围显示；分原因、方向和小格细项保留私有证据。

2023 年市场×审计虽通过组级门槛，该共识区域没有企业再通过主导原始项的联合格检验。2024 年没有组间交互通过主要门槛。2025 年市场×审计、营运×审计均获两层支持。另行关注的偿债×审计及盈利×偿债等在这三年的12个月退市主要分析中均未达到完整支持条件。组间 I 可能由多个原始项叠加，实际观察支持不等于估出了两种经济能力间的因果交互。

## 5　补充时限与恢复原因

{table(['年','日历月','原因','74变量L1','八组L1','随机L1中位','八组ρ','随机ρ中位','范围'],agg)}

2025年24个月可计算归因，预测性能仍为 NA。以上为固定预测器的函数比较，不借未成熟标签判断方法优劣。各年度使用不同历史训练与评价企业，不把跨年度变化全部归为随机种子。逐组、逐模型（原96树、10个32树模型、120组三成员及320树）与全部28种跨组交互见相应 CSV。

## 6　忠实性及论文取舍

396个年度预测器全部通过全原生时间、所有训练及评价企业、两原因检查。组内加跨组、半分后的总和，两者相对森林最大绝对差均为4.00×10⁻¹⁵。最小 CIF 和最小增量均为0，两原因之和最大0.94696。组合的森林基准由三个已保存成员直接遍历后等权平均产生；上一轮已全量核对其与合并96棵树遍历等价，本轮另逐一读取已保存的合并函数验证。原始项没有被筛除或裁剪。另对全部396个预测器私有终点导出核对，对每年度固定的5种配置独立重建每项 A、I、C，并用独立实现复核随机映射1和200的真实企业 L1、排名和符号一致率。

**建议公开可追溯到原始函数的组账目、随机对照和上述局限。** 支持在已暴露数据中以经济类别较简洁地说明预测、降低部分归因的相对波动、指出审计及部分有联合支持的组间关系。没有证据支持经济理论分组系统优于随机分组；四能力对同一家企业的排序或作用方向不具有普遍稳定性。八个 C 单独不能恢复原始函数的细节；审计依赖保留完整原始函数和 A、I 分账。模型精确性、经济标签可读性、统计可靠性和因果解释分别成立与否，应分开表述。
"""
    (D/'RESULTS_ZH.md').write_text(report)
if __name__=='__main__':main()
