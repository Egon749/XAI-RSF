"""Generate report tables from the four matched retrospective information sets."""
from .common import *
def table(header,rows):
    return '| '+' | '.join(header)+' |\n|'+'|'.join('---' for _ in header)+'|\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in rows)
def fmt(v,d=4):return f'{v:.{d}f}' if pd.notna(v) else 'NA'
def main():
    met=pd.read_csv(D/'METRICS_ALL.csv');ens=met.query("variant=='fixed25' and seed==0")
    ci=pd.read_csv(D/'PAIRED_DIFFERENCES_95.csv');seed=pd.read_csv(D/'SEED_SENSITIVITY.csv')
    names={'audit_vs_finance':'审计｜财务','market_vs_finance':'市场｜财务','audit_given_market':'审计｜已有市场','market_given_audit':'市场｜已有审计','complementarity':'性能互补项'}
    p=ci.query("months==12 and cause==1 and year=='pooled'")
    def deltas(variant):
        rows=[]
        for key,lab in names.items():
            z=p[(p.effect==key)&(p.variant==variant)].set_index('metric')
            def band(k,dec):
                row=z.loc[k];return f'{row.delta:+.{dec}f} [{row.lower95:+.{dec}f}, {row.upper95:+.{dec}f}]'
            rows.append([lab,band('within_year_auc',4),band('ap',4),band('negative_brier',4),band('capture',0)])
        return table(['配对对照','年内 AUC 差 [95%区间]','合并 AP 差 [95%区间]','负 Brier 效用差 [95%区间]','Top20净捕获差 [95%区间]'],rows)
    rows=[]
    for year in [*map(str,YEARS),'pooled']:
        for a in SETS:
            z=ens[(ens.year==year)&(ens.arm==a)&(ens.months==12)&(ens.cause==1)].iloc[0]
            rows.append([year,a,fmt(z.auc),fmt(z.within_year_auc),fmt(z.ap),fmt(z.brier),f'{int(z.captured)}/{int(z.target_events)}',f'{z.risk_mean:.2%}',f'{z.observed:.2%}',f'{100*z.calibration_gap:+.2f}',fmt(z.oe,2)])
    main_table=table(['范围','模型','AUC','年内AUC','AP','Brier','捕获/事件','平均预测','观察率','预测−观察（百分点）','O/E'],rows)
    sz=[]
    for y in [*map(str,YEARS),'pooled']:
        for key,lab in names.items():
            v=seed[(seed.year==y)&(seed.effect==key)&(seed.metric=='within_year_auc')&(seed.variant=='fixed25')&(seed.cause==1)&(seed.months==12)].set_index('seed')
            z=ci[(ci.year==y)&(ci.effect==key)&(ci.metric=='within_year_auc')&(ci.variant=='fixed25')&(ci.cause==1)&(ci.months==12)].iloc[0]
            sz.append([y,lab,*[f"{v.at[s,'delta']:+.3f}" for s in SEEDS],f'{z.delta:+.3f}'])
    seed_table=table(['范围','对照','17','503','22','三成员集成'],sz)
    more=[]
    for z in ens[(ens.year=='pooled')&((ens.months==24)|(ens.cause==2))].sort_values(['cause','months','arm']).itertuples():
        more.append([z.months,'恢复' if z.cause==2 else '退市',z.arm,fmt(z.within_year_auc),fmt(z.auc),fmt(z.ap),fmt(z.brier),f'{int(z.captured)}/{int(z.target_events)}'])
    complexity=pd.read_csv(D/'INTERPRETABILITY_COMPLEXITY.csv').query("version=='shrunk' and seed==0 and variant!='proportional'")
    cx=table(['年','模型','主效应','二元交互','空间格','低支持格(<10家)','无实际样本格'],[[z.year,z.arm,int(z.main_functions),int(z.interaction_functions),int(z.spatial_table_cells),f'{z.low_support_cell_fraction:.1%}',f'{z.zero_support_cell_fraction:.1%}'] for z in complexity.sort_values(['year','arm']).itertuples()])
    text=f"""# C039 四套信息集的回顾性增量实验

**H1 对冻结数据库中的审计字段有探索性支持；关于真实预测时点可获得的审计信息，仍不能确认。H2 结果混合。H3 的超加性互补证据不足。** 固定每树25项时，12个月先退市的年内分层AUC由财务模型0.5925，升至财务加审计0.7344、财务加市场0.6562和完整模型0.8377。完整模型的更高AUC不能单独证明两类信息互补。四项性能的差中之差为+0.0397，条件95%区间−0.0657至+0.1466；负Brier效用的互补项略为负。

## 设计、选择与时钟

以 f01e07e8855787d8832d59d33128e6b66cc11f10 为代码基准。指定17、503、22（C039），各32棵树。核实该组合在已查看的120组三成员组合中，唯一取得收缩版12个月退市年内AUC最高值0.837740。它只是在该有限集合、该指标、该批队列上的最高值。尤其对涉及M3的增量和互补项，这种按完整模型表现进行的事后选择可引入乐观偏差。配对区间没有校正该选择。2025同样已暴露。

M0为41项财务和4项共同控制，M1增加7项审计，M2增加21项市场数值与1项市场覆盖，M3含全部74项。共同控制为规模、两种报告陈旧度和入组警示状态。选择次数和分裂计算量可以随维数改变。主实验固定25个每树候选；比例敏感性分别使用15、18、23、25个，按25×p/74四舍五入。维持q≤2、深度4、最小10行、5家公司和3个事件、共享5分位切点、原因等权带符号分裂、退市子节点支持为0或至少2家公司、同一原始直接父节点AJ强度10。三个删减模型在两种预算下各重训，没有对已有M3函数置零。固定配置下27次新拟合，比例敏感性再27次；另导出18个年度96树集成。M3在两种预算下的每树设置完全相同，直接复用已核验的9个成员森林和3个年度合并函数。

公司于首次财务/审计ST公告后的下一交易日开盘前入组。当年1月1日以前的训练公司数为309/345/390，期间已观察退市14/26/49例。实际入模值从冻结源重算，检查财报及已记录更正、同比组成数据最晚日期、市场日序列的严格左截断和公告来源；重建443家输入，与年度快照逐值相等。各年度测试为36/45/53家，与当年训练企业无交叉。有限财务历史版本证据不等于逐个原始财报PDF认证。未把后来接受的替代输入混入本次实验。

**审计时点的两种结论分开保留。** 对7项字段逐行复现六种意见类型编码及持续经营标志；这些值在事后数据库内有记录。前一财年原文和日期索引仍不能对全体企业验证两条审计事实在原点前真实公开。443家中341家有起点前未标更新的候选文档，其中225家自动提取的意见与持续经营标志同时吻合。205家未得到年报双字段吻合或入组文本的双重线索，且仅有线索也不足以证明具体版本。至少6家在三个训练截止点均已入组，缺少及时的前一财年未标更新候选或相应财报公告日，其入组原文与补充文本亦均未载记录的意见字样和持续经营说明。这不是泄漏个数；单用ST变动公告日不能证明审计报告首次公开时间。本轮按指定冻结字段实算M1、M3回顾性效应，同时保留严格可用性未通过的标记。需要公司、审计财年、原始版本、首次公开日期、两字段原文定位才能解除该限制。

## 主结果：12个月先最终退市

{main_table}

年度AUC及AP按当年公司计算，跨年合并AUC和AP在合并预测上重算。年内分层AUC以同年事件数×非事件数加权，排除跨年配对。Top20先按各年8/9/11家确定名额再合计。先恢复为退市的已知竞争结局；全部有支持的终点核对了竞争事件编码与删失权重，实际IPCW权重为1。2025年24个月仅有预测，AUC、AP、Brier、校准及捕获等性能均为NA。2023年12个月无恢复事件，其恢复AUC/AP无定义。

### 五个增量与固定预测的条件区间

{deltas('fixed25')}

正的负Brier效用差表示原Brier下降。ΔA=M1−M0，ΔM=M2−M0，ΔA|M=M3−M2，ΔM|A=M3−M1，Δint=M3−M1−M2+M0。AP列为跨年合并AP。按年度分层、公司为抽样单位，在同一1,000组重抽样索引上计算全部模型的差；三成员等权平均企业预测后再计算指标。区间仅反映固定训练结果下测试企业的变化。2023年仅4例、2025年仅5例12个月退市，年度AUC分别有15及5个未定义的重抽样；合并主指标有1,000个有效重抽样。

### 同一信息集、按原比例缩放每树候选数

{deltas('proportional')}

H1的主指标点估计从+0.1418减至+0.0841，比例对照的区间跨0；审计在已有市场时的点估计仍为正。市场相对纯财务的增量由+0.0637减至+0.0048。市场在已有审计时的年内AUC增量较明显；AP的对应区间跨0。互补项在两种预算下AUC区间均跨0。不能据此选择最有利的候选池口径。2024年固定25的四模型真正评价的候选分别为49,923、46,020、53,040、49,232个。相同公司bootstrap和种子配对已核实，候选调度和树结构并不相同。

### 年度与种子

{seed_table}

相对财务加入审计，三个单种子的三年年内AUC差约+0.144、+0.131、+0.133；2023年种子22的年度差为负。加入市场，三个种子总体差约+0.038、+0.077、+0.018，单种子年度方向多次反转。2023年M2的年度AUC较M0高，但AP由0.3781降至0.1728且少抓1例。上述三个成员并非三个独立验证队列。每个种子的AP、Brier、Top20等配对结果另见SEED_SENSITIVITY.csv。

## 经济解释、监管渠道及概率水平

**原生贡献。** M3在三年的审计组平均绝对贡献依次为0.728、1.024、2.234个退市CIF百分点，分别重现既有精确分解中的C039数值。M3的广义市场组对应0.362、0.275、0.394个百分点。按冻结的互斥市场子组，计入分摊交互后的波动、回撤和尾部风险组在M2及M3的三个年度均具有最大的平均绝对贡献。例如2024年M3价格/市值/累计收益为0.150个百分点、波动与尾部为0.159个百分点，接近；流动与交易活动为0.033个百分点、覆盖状态约0.004个百分点。各子组先按符号加总再取绝对值，其幅度之间不能直接求和。该诊断说明保存模型如何使用变量，不能判断删去每一子组重训后的独立增量。没有增加市场子组训练或按组挑选模型。

**增量预测。** 可据M0–M3重训差异评价额外开放信息的系统表现，但不能把单个变量函数或组贡献当成信息组删除效应，更不能把AUC差写成条件互信息。2024年固定25棵候选口径下M3的审计字段共获得227个变量—树候选资格，涉及64个分裂节点；21项市场数值对应682个候选资格和241个分裂节点，覆盖字段另计33个资格和4个节点。这些计数不衡量因果强度。

**终局渠道。** 用已有公告正文、核读的交易所决定及原事件时间链作事后分类，保留未知和价格与市值同时触发的记录。12个月25例先退市中19例有低价依据，M0和M2均在Top20捕获其中6例。市场相对财务的年内AUC净增量0.0637，可按同年事件—非事件配对分子拆解；低价事件贡献约+0.0397，其余事件合计约+0.0240。低价事件占样本76%，故较大总贡献也受事件数量影响。既不能据此认定股价单变量解释了全部收益，也不能由模型变量归因判定实际退市原因。24个月两个测试队列有32例先退市，保留1例原因不明确；不估计小样本分原因AUC。ST/*ST警示类别本身是共同输入，上一年审计信息可能同时描述入组时已知的风险状态。仅在明确可得日与正式规则的条件下解释制度关联，不做经济机制或政策因果判断。

**排序与校准。** 合并实际退市率18.66%，四个模型预测均值仅4.02%–4.25%。M3的Brier从M0的0.1740降至0.1702，但其平均风险4.04%与实际仍差14.62个百分点，O/E约4.62。2025年从M2转为M3，AUC 0.721→0.775，平均预测却从6.64%降至5.62%，离9.43%的观察率更远。没有利用这些结果来拟合新校准器。校准表含合并/年度总体及逐年风险四分位。2025年24个月不主张概率校准有验证结果。

## 补充结局和时限

{table(['月','原因','模型','年内AUC','合并AUC','合并AP','Brier','捕获/事件'],more)}

24个月退市时，M1与M3的年内AUC很接近（约0.772与0.774），M1的合并AP约0.702，高于M3的0.687。12个月恢复时M2的年内AUC高于M3。不能将选择C039所用的12个月退市结果推广到恢复或其他预测时限。所有年度、组合、单种子、时限和原因保存在METRICS_ALL.csv中。

## 表示复杂度与结论

{cx}

54次新拟合均保留raw及直接父节点收缩版，新增126份表示（包括18份新集成），M3从已有工件核对并保留全部模型/函数哈希。对全体训练及评价公司、全部原生时间、两种原因核验；新模型完整重构的最大误差2.00×10⁻¹⁵、逐项显式求和2.78×10⁻¹⁵，远低于1e−10；概率非负、随时间非递减，两原因和不超过1。全部预测由原森林给出，未裁剪，未使用代理解释。

**论文建议。** 可以将四格结果纳入论文的探索性信息组消融，说明数据库审计字段的回顾性预测增量、市场信息的预算/指标依赖性和概率水平低估。H1在该数据库层面为EXPLORATORY_SUPPORT，对实时可得信息为INCONCLUSIVE；H2为MIXED，H3为INCONCLUSIVE。含审计字段的所有对照及事后选种都应在同一表注中披露。不能写成已证实审计与市场协同导致退市，也不能以这一次实验替换原主模型。
"""
    (D/'RESULTS_ZH.md').write_text(text.replace('固定25棵候选口径','每树固定25项候选的口径'))
if __name__=='__main__':main()
