"""Publish aggregate evidence only; private models/forecasts never copied to Git."""
from pathlib import Path
import argparse,datetime,json,shutil,subprocess
import numpy as np
import pandas as pd
from .common import *

LABELS={'baseline':'深度6 raw','parent_shrink':'深度6＋收缩','shallow':'深度4 raw','shallow_parent':'当前参考：深度4＋收缩','q_unrestricted':'解除q限制＋收缩','local_cuts':'局部随机切点＋收缩','iid_features':'IID25＋收缩','q_unrestricted_raw':'解除q限制 raw','local_cuts_raw':'局部随机切点 raw','iid_features_raw':'IID25 raw','standard_rsf':'标准RSF外部参照'}

def table(df):
    def cell(v):
        if isinstance(v,(float,np.floating)):return 'NA' if pd.isna(v) else f'{v:.6f}'
        return str(v).replace('|','／')
    return '\n'.join(['| '+' | '.join(map(str,df.columns))+' |','|'+'---|'*len(df.columns)]+['| '+' | '.join(cell(v) for v in row)+' |' for row in df.itertuples(index=False,name=None)])+'\n'

def render(run,dest):
    e=run/'evaluation';out=dest/'results';out.mkdir(exist_ok=True,parents=True)
    def copy(src,target=None):
        target=target or out/src.name
        if target.exists():assert sha(target)==sha(src),(src,target)
        else:shutil.copy2(src,target)
    for src in e.iterdir():
        if src.suffix in ['.csv','.json']:copy(src)
    for name in ['pre_fit_freeze.json','formal_gate.json','verify_gate.json','environment_oliver.json','environment_lemon.json','postprocessing_failure.json','full_tests_pre_fit.log','full_tests_final.log','bootstrap.log','bootstrap_fixed.log']:
        copy(run/'audit'/name)
    for stage in ['formal','verify']:copy(run/f'{stage}.jsonl',out/f'{stage}_manifest.jsonl')
    freeze=json.loads((run/'audit/pre_fit_freeze.json').read_text());valid=json.loads((e/'final_validation.json').read_text());num=json.loads((e/'function_validation.json').read_text())
    m=pd.read_csv(e/'metrics.csv',dtype={'scope':str,'seed':str});cc=pd.read_csv(e/'numerical_complexity_cost.csv',dtype={'seed':str});ci=pd.read_csv(e/'paired_bootstrap_intervals.csv',dtype={'scope':str})
    d=pd.read_csv(e/'paired_differences.csv',dtype={'scope':str,'seed':str});ps=pd.read_csv(e/'prediction_stability.csv',dtype={'scope':str});st=pd.read_csv(e/'function_stability.csv');so=pd.read_csv(e/'function_stability_by_order.csv')
    core=['shallow_parent']+NEW;order={a:i for i,a in enumerate(ARMS+['standard_rsf'])}
    main=m[(m.seed=='ensemble')&(m.horizon_months==24)&(m.cause=='termination')].copy();main['sort']=main.arm.map(order);main=main.sort_values(['scope','sort'])
    cols=['scope','arm','auc','ap','brier','risk_mean','ipcw_observed_incidence','top20_captured','top20_coverage']
    def pretty(frame):
        frame=frame.copy()
        if 'arm' in frame:frame['arm']=frame.arm.map(LABELS)
        return table(frame)
    def md(name,text):
        p=dest/name
        with p.open('x') as f:f.write(text)
    core_main=main[main.arm.isin(core)][cols]
    uncertainty=ci[(ci.scope=='pooled')&ci.contrast.isin([a+'_minus_reference' for a in NEW])&ci.metric.isin(['auc','ap','brier','top20_coverage'])][['contrast','metric','estimate','lower95','upper95']]
    md('RESULTS_ZH.md',f'''# 当前森林消融：明确的表示收益，混合的预测与稳定性证据

本轮已经实际完成18次新森林拟合、18个父节点收缩派生模型，并复用24个历史内部模型和6个标准RSF参照。没有基线重现拟合、额外种子、校准或超参数搜索。当前参考模型及所有历史工件不变。

主要比较是三个新增收缩臂减当前shallow_parent；2023/2024已多次暴露，全部结果属于探索性机制分析。数据仍有原始财报版本/PIT限制。下表均先平均三个种子的逐公司预测再评价；不是平均三个AUC/AP。

{pretty(core_main)}

2023测试36家、12例24个月先退市；2024测试45家、20例；合并81家32例。Top20逐年度向上取整，核查8＋9＝17家，不在合并队列重新定名额。当前参考捕获4/12、8/20，合计12/32；解除q限制4/12、7/20，合计11/32；局部切点5/12、7/20，合计12/32；IID25为4/12、7/20，合计11/32。未知0、重复0，已知非目标包含先恢复及窗口内无目标事件，不能称永久正常。

**q2的主要收益是表示规模控制。** 在深度4和当前搜索设置下解除q限制，允许实际四变量路径，合并AUC/AP由0.729592/0.681719变为0.722577/0.662682，Top20少1例；在主终点点估计上，未观察到保留q2的合并预测代价。但2023解除限制AP提高0.073562，所以不能断言限制q在所有年度都不损失预测信息。

**共享分位数切点降低表示复杂度，预测收益随年度和期限变化。** 局部随机切点的空间单元增加，2023 AP明显更高，而2024 AP更低；合并24个月AP降低0.055151。12个月合并AUC/AP则是局部切点更高，必须同时保留。这个对照同时改变共享方式与生成方式，不能单独识别“共享”因素。

**均衡覆盖没有获得一致的独立预测或稳定性收益。** IID25合并AUC略高0.003189、Brier低0.001815，但AP低0.031034、Top20少1例；其交互集合和部分共同函数的稳定性更好。均衡模式确实让变量获得资格的次数更均匀，这一机制事实不等于预测优越性。

**深度与收缩存在条件依赖。** 变浅减少约19%的空间单元，但不减少历史两折的主效应/二元交互数；收缩不改变结构，普遍降低预测随机波动。它没有解决退市风险水平低估，部分情形Brier变差或Top20减少。

主要合并差值及公司配对bootstrap描述性95%区间如下；正差对AUC/AP/覆盖有利，对Brier不利。

{table(uncertainty)}

三个主要对照的AUC/AP/Brier/Top20差值，在逐年度和合并分析中的区间均包含0。不是等效性证明，也不支持确认性优越结论。1000次按测试年度分层、公司为单位、所有模型共同抽样；只反映固定预测条件下测试样本波动，不包括训练、方法选择和多次研究接触的不确定性。没有进行多重比较校正。

研究判断：预测优势及解释稳定性的普遍改善仍为 **INCONCLUSIVE**；本轮对q2和共享切点的有限范围表示压缩提供了直接证据。保持当前参考，消融用于说明取舍，不自动选取新模型。

完整年度/种子/合并结果见results/metrics.csv（528行），配对差值见paired_differences.csv，552行区间见paired_bootstrap_intervals.csv；raw版本和全部不利结果均保留。
''')
    hist=d[(d.seed=='ensemble')&(d.horizon_months==24)&(d.cause=='termination')&d.contrast.isin(['depth_without_shrink','depth_with_shrink','shrink_at_depth6','shrink_at_depth4','depth_shrink_interaction'])][['scope','contrast','auc','ap','brier','top20_captured']]
    new_inter=d[(d.scope=='pooled')&(d.seed=='ensemble')&(d.horizon_months==24)&(d.cause=='termination')&d.contrast.str.contains('shrink_effect|shrink_interaction')][['contrast','auc','ap','brier','top20_captured']]
    md('DEPTH_SHRINKAGE_ZH.md',f'''# 历史2×2与新增结构的收缩交互

四个历史臂每年每种子的原始预测均重新读取；在统一评价器中240行历史及标准RSF指标与原报告逐项复现。复用的是已保存森林/派生模型，不是新拟合冒充复用。baseline深度6/raw；parent_shrink深度6/强度10；shallow深度4/raw；shallow_parent深度4/强度10。

{pretty(main[main.arm.isin(OLD)][cols])}

差值定义：depth_without_shrink=shallow−baseline；depth_with_shrink=shallow_parent−parent_shrink；shrink_at_depth6=parent_shrink−baseline；shrink_at_depth4=shallow_parent−shallow；交互=(shallow_parent−shallow)−(parent_shrink−baseline)。这描述算法设计的条件依赖，不是企业层面的因果效应。

{table(hist)}

1. 不收缩时变浅：2023 AUC/AP分别−0.006944/−0.068165、Top20少1；2024为＋0.008/＋0.021764、Top20多2。合并AP＋0.010316，但Brier＋0.001636变差。
2. 已收缩时变浅：两年AUC/AP点估计均提高；2023 Top20少1、2024多1，合并不变。合并Brier仍＋0.001268变差。
3. 深度6加入收缩：2023 AP−0.065789、2024 AP＋0.003503；合并AP几乎不变（−0.000182）。Top20不变，合并Brier变差。
4. 深度4加入收缩：两年AUC/AP提高，2024 Top20从9/20降为8/20；合并AP只提高0.001827，Brier变差0.000348。
5. 交互：2023 AP的差中差为＋0.079499，2024为−0.001573，合并＋0.002009。方向和大小依赖年度，不能用单一年度宣称稳定协同收益。完整描述性区间均保存在同一配对bootstrap表中。

新增臂的raw森林也全部保留。以下为收缩效应，以及相对当前q2深度4收缩效应的差中差（合并主指标）：

{table(new_inter)}

相同森林的raw和shrunk全部路径、切点、缺失路由和配置一致，共30对核验通过。直接父节点AJ由原始bootstrap重建，权重使用独立公司数；根叶不变，无递归收缩、重复收缩或后处理校准。收缩不增加公司或事件支持。
''')
    ensembles=cc[cc.seed=='ensemble'].copy();ensembles['complete_cells_with_baseline']=ensembles.spatial_table_cells+1
    ensembles['complete_scalars_with_baseline']=ensembles.scalar_table_cells+ensembles.native_time_count*2
    ensembles.to_csv(out/'ensemble_complexity.csv',index=False)
    cc[cc.seed!='ensemble'].to_csv(out/'single_seed_complexity.csv',index=False)
    cs=ensembles[ensembles.arm.isin(core)][['year','arm','max_path_variables','main_functions','interaction_functions','order3_functions','order4_functions','spatial_table_cells','scalar_table_cells','function_file_bytes']]
    stable=st[(st.day==730)&(st.cause==1)&st.arm.isin(core)].groupby(['year','arm'])[['order1_nonzero_jaccard','order2_nonzero_jaccard','median_correlation']].mean().reset_index()
    stableorder=so[(so.day==730)&(so.cause==1)&so.arm.isin(core)].groupby(['year','arm','order'])[['common_terms','median_correlation','mean_rms_difference']].mean().reset_index()
    sd=ps[(ps.horizon_months==24)&(ps.cause=='termination')&ps.arm.isin(core)][['scope','arm','prediction_sd_mean']]
    md('COMPLEXITY_STABILITY_ZH.md',f'''# 完整函数表示、稳定性与真实支持

每年度以相同完整训练样本的经验边际乘积为参考，编译所有阶次。主模型严格q2；解除q限制只在深度4下运行，实际最大阶次4。各变量的边际分布保留，但参考乘积打破变量间依赖，不能将所得贡献解释为条件效应或因果影响。

{pretty(cs)}

空间单元采用原导出器的一致口径：每个非空结构项的各坐标有限区间数加1个缺失状态，再取笛卡尔积求和；即使缺失状态训练支持为0也计入。不合并相邻等值格。上表spatial_table_cells不含单独基准曲线；含基准的完整空间数等于该数＋1，在ensemble_complexity.csv单独列出。scalar_table_cells再乘原生时间数和两个原因；含基准的标量量也另列。

函数文件采用无损因子化矩形存储，文件字节数与逻辑表格数不是同一个量。解除q限制虽然逻辑格数达76,668/89,517，压缩文件仍约237/259KB，不能声称物理存储也增加二十多倍。统计非零要求精确格点与所有原生时点中存在幅度>1e−14；这只是代数判据，不是显著性。

相对解除q限制，q2空间单元减少约95.90%/96.35%。相对局部随机切点，共享方案减少32.46%/36.27%。深度6变4从3,869/4,068降为3,146/3,263（18.69%/19.79%）；历史主效应62/65、交互144/146均不变。收缩不改变任何空间分区。

解除q限制后的三阶、四阶函数确有非零幅度。day730退市的三阶总贡献在合并训练/评价样本上的平均绝对值约0.221/0.248个百分点，四阶约0.051/0.060个百分点；高阶路径不自动代表幅度大或有统计支持。仅保留二阶会产生残差，两个原因/day365与730上的最大截断残差约2.882/2.851个百分点；这是近似诊断，不是本轮完整解释。完整导出没有删高阶项。

跨种子集合与共同函数诊断（每年3对种子平均）：

{pretty(stable)}

集合越稠密，重合度也可能机械升高。无q限制的主效应Jaccard很高，但其函数数和交互数显著增多，不能据此宣称可读性或稳定性全面改善。共同函数的比较必须按阶次区分：

{pretty(stableorder)}

当前q2的共同二元函数相关性约0.650/0.506；无q限制约0.182/0.238。IID25则约0.688/0.570，且交互Jaccard也高于均衡方案。这是均衡覆盖稳定性主张的反向证据。相关性仅对共同非恒定函数计算，恒定/缺失共同项数量单列；共同项集合和幅度不同，不能把RMS更低等同于更可靠。

预测跨种子波动（各公司样本标准差ddof=1，再平均，单位为概率）：

{pretty(sd)}

收缩在全部5种结构、两个年度均降低平均预测波动，但两种结构之间的稳定性无一致顺序。三个种子反映算法随机性，不是3个独立验证队列。函数表精确与结构稳定是两件事。

当前参考2023/2024分别有583/433个零退市支持叶，收缩后其中178/186个借入正退市风险；它们真实目标事件支持仍是0。解除q限制零支持叶721/601、低支持格比例约73.57%/72.94%；当前参考低支持格约43.58%/41.31%。详见支持度和复杂度CSV。所有模型、所有训练/评价行、全部原生时点和两个原因的最大完整项求和误差为{num['max_component_sum_error']:.3e}，低于1e−10；概率与保存重载检查均通过。
''')
    secondary=m[(m.seed=='ensemble')&m.arm.isin(core)&((m.horizon_months==12)|(m.cause=='recovery'))][['scope','arm','horizon_months','cause','auc','ap','brier','risk_mean','ipcw_observed_incidence']].sort_values(['horizon_months','cause','scope','arm'])
    bins=pd.read_csv(e/'calibration_bins.csv',dtype={'scope':str});b=bins[(bins.scope=='pooled')&(bins.arm=='shallow_parent')&(bins.horizon_months==24)&(bins.cause=='termination')][['bin','n','predicted','observed','events']]
    md('PREDICTION_CALIBRATION_ZH.md',f'''# 排序与概率水平分开报告

全部11个版本（含raw与外部参照）的24个月先退市结果：

{pretty(main[cols])}

合并指标由81家公司的合并预测重算。各年度的预测来自各自训练截止，pooled同时包含年度之间的水平差异，不能代替逐年度表现。外部randomForestSRC的缺失处理、切点、深度等实现不同；它不是q消融，也不是原先首次ST预警任务M2。

本数据两个测试期限结局完整，沿用IPCW评价器得到所有权重恰为1，因此Brier等于普通平方误差均值；记录等权与公司等权相同。不是把未知删失当正常。

三个新增结构均未解决风险水平低估：合并当前参考平均风险9.31%，观察先退市39.51%；解除q限制9.38%，局部切点9.22%，IID25为9.61%。当前参考的四等分校准如下，分箱依自身预测升序，仅作描述、不拟合校准器：

{table(b)}

最高组预测约15.15%、观察70%；样本仅20家，不能当作新的校准映射或概率倍率。当前参考2023低估约26.87个百分点、2024约32.86个百分点。概率合法并不意味着概率已校准。

次要期限与恢复原因的全部当前参照/新收缩臂结果：

{pretty(secondary)}

12个月合并退市中，局部切点和IID25的AUC均0.827869，高于当前0.799180；局部切点AP0.678134、IID25 AP0.663553，也高于当前0.634071。解除q限制的AUC0.812295较高但AP0.617681较低。这些与24个月主分析不同，不得省略，也不能据此改主指标或改模型。

恢复24个月：解除q限制合并AUC/AP/Brier为0.683871/0.650388/0.237131，当前为0.673548/0.627720/0.242559，点估计有利于高阶版本。这提示q限制的代价可能依结局而异。2023的12个月恢复没有阳性事件，AUC/AP保留NA，不填0或0.5。

主比较的95%区间包含0，仍不等于方案等效。所有次分析均为预先声明的描述性结果，没有追加结果导向搜索。
''')
    search=pd.read_csv(e/'search_counts.csv');search=search[search.arm.isin(core)].groupby(['year','arm'])[['finite_cuts','missing_only_cuts','routed_attempts','scored_candidates']].sum().reset_index()
    cv=pd.read_csv(e/'coverage_summary.csv');cv=cv[cv.arm.isin(['shallow_parent','iid_features'])].groupby(['year','arm'])[['eligible_min','eligible_max','eligible_sd','eligible_pairs','co_path_pairs']].mean().reset_index()
    tasks=pd.read_csv(e/'execution_tasks.csv');resources=tasks.groupby('stage')[['seconds','cpu_seconds','peak_rss_MiB']].agg(['min','median','max']);resources.columns=['_'.join(c) for c in resources.columns]
    md('TRAINING_DIAGNOSTICS_ZH.md',f'''# 实际执行、随机配对与数值诊断

冻结拟合提交：`{freeze['code_commit']}`。设计来源261eb15，保留起始3931d0c的全部后续工作。协议在新增真实结果前冻结；分析和交付代码随后按既定协议补齐，未改变指标、对照、样本、预算或随机种子。

正式Slurm作业2573：18/18拟合和18个派生模型完成，oliver16项/lemon2项；计算加发布约10.45秒，含排队启动约13.07秒。验证作业2574：80/80完成，oliver62项/lemon18项；约21.72秒，含排队启动约22.95秒。这里的作业时间不包含人工代码审计、报告编制；不是整个研究的总耗时，也不称调度方案已证明最快。

{table(resources.reset_index())}

单任务单线程，4GiB地址空间上限，实际峰值内存远低于上限。通过现有调度器分配P/E核和lemon线程，不绕过Slurm，不宣称单Python进程跨两机加速。CPU算法未使用GPU。完整训练、收缩、预测、函数编译/导出/验证耗时分列在model_costs.csv和numerical_complexity_cost.csv；历史时间缺失保持NA，并标明并发/硬件/计时口径差异。raw与shrunk共享一次拟合时间，不能将其重复相加。

通过同年/种子/树编号的原始公司bootstrap索引和重复次数哈希配对。非IID臂的每树25变量集合保持一致；IID使用独立SeedSequence[seed,tree,1731]，不消耗bootstrap[seed,tree,0]或节点[seed,tree,node_key,1]流。旧均衡调度[seed,block,731]未变。

{pretty(cv)}

均衡方案跨种子每变量入选8—14次，IID为3—16次；每个种子总资格数32×25=800相同。83不能整除25，均衡不表示每个变量次数完全一致。每变量资格次数、实际分裂树数/节点数，以及每变量对资格/同路径覆盖，均保存为一致定义的汇总表。入选资格不等于实际使用。

下表为三种子合计的候选重放计数。routed_attempts含未通过叶支持约束的候选，scored_candidates才是实际调用logrank评分的次数；两者均报告。所有候选的获胜结果及历史candidate_evaluations逐节点精确复现。

{pretty(search)}

共享字典会在部分节点不足5个有限切点，缺失路由及missing-only候选又改变次数。局部随机臂保持n_thresholds=5，但不拥有相同实际搜索量；它同时改变共享和生成规则。原代码中的candidate_evaluations实际指尝试次数，本轮没有把它误写为全部实际评分次数。

直接父节点AJ独立重放误差0；raw与shrunk共30对全部结构一致。完整函数重构最大误差{num['max_component_sum_error']:.3e}；60个单种子模型、20个96树平均预测器均检查全部训练/评价公司与原生时间、两个原因，含缺失状态；高阶模型使用完整四阶展开，不经二阶截断。

失败记录：正式拟合0，发布0，模型验证0；bootstrap汇总脚本发生1次字典迭代错误，在重采样指标计算前退出。保留bootstrap.log和postprocessing_failure.json，改为遍历values并增加完整管线测试后继续，抽样索引逐项一致，未重训、未换种子、未改协议。完整测试拟合前150通过，交付前153通过（独立R检查均执行）；唯一警告为已有EBM缺失值可视化提示，与森林结果无关。
''')
    counts=pd.read_csv(run/'reports/cohort_counts.csv')
    md('DATA_ALIGNMENT_AUDIT_ZH.md',f'''# 固定旧C输入与竞争风险时钟

{table(counts)}

逐字节复用历史conservative C的83列训练和评价X，不重建研究样本、不接受年报版本覆盖、不引入2025。逐项核对sample_id、公司group、字段顺序、事件类型、实际日随访与日历月随访、原点与停止日期；训练和测试公司不重叠。哈希见input_hashes.csv，模型位置见artifact_reuse.csv。

原点逐行根据已验证交易日历复核为入组公告之后的下一交易日开盘前；下一交易日当日尚未公开的信息不进入输入。竞争事件编码为先最终终止上市X=1、先完整恢复A=2，均用公告日期；先恢复再退市仍是恢复结局。训练信息严格早于外层2023/2024年初，随访上限24日历月。评价保持原覆盖截止2026-09-03，不将当前日期当成延长随访的新证据。

2023/2024训练分别309/345家、14/26个24个月截断后先退市事件；两折654条但去重345家公司、26个目标事件。18次新增拟合累计读取5,886条训练计算记录，不能当作新增样本。测试共81家、32例24个月先退市、31例先恢复；12个月目标为4＋16＝20，恢复0＋5＝5。两个评价期限未知均为0，训练未知维持删失。

保守财务时钟使用记录中的原公告及修订日期最大值，增长/变化类变量继承组成记录的较晚可得日；本轮仅核对既有冻结产物和生成代码，不宣称完成原始PDF版本PIT认证。数据来自历史观察首次财务/审计ST非金融企业；历史覆盖和行业时钟局限不消失。

2023/2024已经反复查看，本轮是探索性机制分析。保持CURRENT_POST_ST_MODEL.json逐字节不变；当前参考继续保留。
''')
    md('PAPER_METHODS_RESULTS_ZH.md',f'''# 论文可用的消融方法与结果段落

## 方法

为区分路径交互阶数、切点方案、变量覆盖调度、树深和叶节点估计的作用，我们在已暴露的2023与2024年ST后预后队列中开展探索性配对消融。模型预测公告后下一交易日开盘前起算的首次最终退市与首次完整恢复的竞争风险；主要终点为24日历月内先最终退市，12个月和恢复为次要终点。两折训练信息分别冻结在2023和2024年初之前，沿用83列conservative C输入和原有删失定义。

当前参照为路径至多使用两个不同输入变量的森林，深度4、每种子32树，种子42/101/202，叶节点最小记录数10、独立公司数5、事件数3；每树均衡选25个可用变量，mtry75，最多5个共享分位数切点候选，原因权重(1,1)，不设置全局单元预算。新增三个单因素对照分别解除路径阶数限制、改用节点局部随机切点、改为每树独立均匀无放回抽取25个变量。无q限制臂的深度仍为4，所以其范围只是当前深度与搜索设置下的解除限制。变量覆盖的IID随机流独立于bootstrap和节点搜索；所有对照共用同年、种子和树编号的公司bootstrap及其重复次数。

每次新拟合先保存raw AJ森林，再将每个叶节点CIF替换为alpha×raw叶AJ＋(1−alpha)×raw直接父节点AJ，其中alpha=n_unique/(n_unique+10)，n_unique为该叶独立in-bag公司数。权重对时间与原因相同，根叶不变。三个新增臂共18次森林拟合，得到18个raw和18个收缩版本。另复用历史深度6/4与强度0/10的四臂共24个模型；标准randomForestSRC只作为外部参照。

每年度先平均三个种子的逐公司预测，再计算AUC、AP、IPCW Brier、平均风险、观察发生率及Top20事件捕获。合并指标重新在拼接预测上计算；Top20在每年度取ceil(0.2n)后合计。解释以同年训练集经验边际乘积为参考，导出所有实际阶数的完整因子化函数。空间单元按各函数有限区间与缺失状态的笛卡尔积计算，另报告标量量、实际文件大小、结构/非零项数、跨种子集合与函数稳定性以及叶内事件支持。全部训练和评价公司、原生时间点及两个原因均核验完整重构，容差1e−10。

我们在固定预测条件下，以公司为单位、按测试年度分层开展1000次共同索引的配对bootstrap（分析种子2026102901），报告差值的描述性百分位95%区间及不可计算次数。这些区间不包含训练样本变化、方法选择或多次开发接触的不确定性，不作为确认性多重假设检验。

## 结果

18次新拟合与全部80项单种子/集成表示验证成功；完整函数项求和的最大绝对误差为{num['max_component_sum_error']:.3e}，CIF非负、单调和总和约束均满足。2023/2024分别36/45家，24个月先退市12/20例。当前q2参考的年度AUC为0.746528/0.770000，AP为0.522272/0.782032；合并AUC/AP/Brier为0.729592/0.681719/0.315749，Top20捕获12/32例。

解除q限制的合并AUC/AP/Brier为0.722577/0.662682/0.315829，捕获11/32例；空间单元则由3,146/3,263增至76,668/89,517。该结果支持q2在所测范围内控制完整表示规模，但2023年解除限制提高AP，且恢复结局与12个月部分指标有不同方向，不支持普遍无损的阶数限制结论。局部随机切点的空间单元为4,658/5,120，合并24个月AP为0.626568，但2023年AP及12个月合并排序改善，体现切点方案的取舍。IID25合并AUC和Brier略优于均衡覆盖，而AP与Top20较差；均衡覆盖资格次数更均匀，却未显示一致的函数稳定性优势。

历史深度×收缩分析表明，变浅减少约19%的分区单元而不减少非零主效应与二元函数数量。父节点收缩保留结构并降低预测的随机种子波动，但2024年深度4的Top20从9/20降至8/20，合并Brier略变差。主要配对差值的描述性95%区间均覆盖0。当前参考合并平均退市风险仅9.31%，观察发生率39.51%，所有消融均未修复显著的概率水平偏差。因此结果支持有限范围内的表示压缩与平滑机制，不支持各项设计均带来独立预测收益、解释统计可辨识性或普遍稳定性提升的主张。
''')
    with (dest/'TRAINING_DIAGNOSTICS_ZH.md').open('a') as f:f.write('\n补充核验6份冻结标准RSF：仅加载、重新保存和预测，没有调用训练。全部训练/评价公司、123/140个原生时间和两个原因的预测与历史二进制工件逐项一致，保存重载误差0，CIF约束通过；见standard_rsf_validation.csv。\n')
    code={str(p.relative_to(dest)):sha(p) for p in list(dest.glob('*.py'))+list(dest.glob('*.R'))}
    receipt=dict(experiment='post_st_ablation_v1',status='COMPLETE_EXPLORATORY',design_source_commit=SOURCE_COMMIT,fitting_code_commit=freeze['code_commit'],private_run=str(run),
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**{k:v for k,v in valid.items() if k!='status'},
        formal_slurm_job=2573,verification_slurm_job=2574,fit_task_hosts={'oliverdesktop':16,'lemon':2},verification_task_hosts={'oliverdesktop':62,'lemon':18},
        formal_repeated_training_rows=5886,unique_training_companies=345,unique_training_target_events=26,evaluation_companies=81,primary_target_events=32,
        main_horizon_months=24,secondary_horizon_months=12,bootstrap_replicates=1000,bootstrap_seed=2026102901,
        decision='INCONCLUSIVE for predictive superiority; verified representation tradeoffs; current reference retained',
        original_reference_unchanged=True,no_2025_access=True,no_calibration=True,no_prediction_clipping=True,
        max_complete_component_reconstruction_error=num['max_component_sum_error'],full_tests_before_fits=150,full_tests_final=153,
        analysis_code_sha256=code,protocol_sha256=sha(dest/'frozen_protocol.json'),result_sha256={str(p.relative_to(dest)):sha(p) for p in out.iterdir() if p.is_file()},
        privacy='No company records, predictions, raw thresholds, fitted functions or fitted models copied to repository',git_result_commit='See Git history/final delivery; self-referential commit hash not embedded')
    md('execution_receipt.json',json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ['status','fitting_code_commit','new_formal_forest_fits','reused_internal_models','max_complete_component_reconstruction_error']},ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--dest',type=Path,default=Path(__file__).parent);a=p.parse_args();render(a.run,a.dest)
