"""Publish fixed-order cumulative results, with aggregate evidence only."""
from pathlib import Path
import argparse,datetime,json,shutil
import numpy as np
import pandas as pd
from experiments.post_st_ablation_v1.report import table
from .common import *


def figure(main,complexity,dest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'font.size':10})
    fig,axs=plt.subplots(2,3,figsize=(15,8.5),layout='constrained')
    specs=[('auc','AUC ↑'),('ap','AP ↑'),('brier','IPCW Brier ↓'),('top20_coverage','Top20 事件覆盖率 ↑'),('risk_mean','平均预测退市概率'),('spatial_table_cells','完整函数空间单元（对数轴）')]
    for ax,(key,title) in zip(axs.flat,specs):
        for y,color in [('2023','#2878B5'),('2024','#D75336')]:
            z=complexity[complexity.year==int(y)] if key=='spatial_table_cells' else main[main.scope==y]
            values=z.set_index('arm').loc[STAGES,key].to_numpy()
            ax.plot(range(5),values,'o-',color=color,lw=2,label=y)
            if key=='risk_mean':ax.axhline(12/36 if y=='2023' else 20/45,color=color,ls='--',lw=1,alpha=.65)
        ax.set_title(title);ax.set_xticks(range(5),['完整','解除 q','再改局部\n切点','再改\nIID25','再取消\n收缩']);ax.grid(alpha=.2)
        if key=='spatial_table_cells':ax.set_yscale('log')
        if key in ['top20_coverage','risk_mean']:
            from matplotlib.ticker import PercentFormatter
            ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.spines[['top','right']].set_visible(False)
    axs[0,0].legend(frameon=False)
    fig.suptitle('固定顺序累积消融｜24 个日历月内先退市\n三种子逐公司预测先平均；2023/2024 均为探索性评价',fontsize=15)
    fig.text(.5,-.025,'每一步保留之前的移除；最大深度始终为 4。概率图虚线为观察发生率。完整空间单元不含单独基准曲线。',ha='center',fontproperties=font)
    for ext in ['png','pdf']:fig.savefig(dest/f'cumulative_comparison.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def render(run,dest):
    e=run/'evaluation';out=dest/'results';out.mkdir(exist_ok=True,parents=True)
    def copy(src,target=None):
        target=target or out/src.name
        if target.exists():assert sha(target)==sha(src),(src,target)
        else:shutil.copy2(src,target)
    def md(name,content):
        with (dest/name).open('x') as f:f.write(content)
    def pretty(frame):
        frame=frame.copy()
        if 'arm' in frame:frame['arm']=frame.arm.map(lambda a:f'{a} {LABELS[a]}')
        return table(frame)
    freeze=json.loads((run/'audit/pre_fit_freeze.json').read_text())
    valid=json.loads((e/'validation.json').read_text());num=json.loads((e/'function_validation.json').read_text())
    assert valid['status']==num['status']=='PASS'
    for n,h in json.loads((run/'audit/source_hashes.json').read_text()).items():assert sha(Path(n))==h
    for n,h in json.loads((run/'audit/immutable_input_hashes.json').read_text()).items():assert sha(run/'input'/n)==h
    reference=Path(__file__).resolve().parents[2]/'CURRENT_POST_ST_MODEL.json'
    assert sha(reference)==json.loads((run/'audit/baseline_audit.json').read_text())['reference_sha256']
    for src in e.iterdir():
        if src.suffix in ['.csv','.json']:copy(src)
    for name in ['pre_fit_freeze.json','formal_gate.json','verify_gate.json','environment_oliver.json','environment_lemon.json','full_tests_pre_fit.log','full_tests_final.log','analysis.log']:
        copy(run/'audit'/name)
    assert '155 passed' in (run/'audit/full_tests_final.log').read_text()
    assert 'skipped' not in (run/'audit/full_tests_final.log').read_text()
    for stage in ['formal','verify']:copy(run/f'{stage}.jsonl',out/f'{stage}_manifest.jsonl')
    m=pd.read_csv(e/'metrics.csv',dtype={'scope':str,'seed':str});cc=pd.read_csv(e/'ensemble_complexity.csv')
    single=pd.read_csv(e/'single_seed_complexity.csv',dtype={'seed':str})
    main=m.query("seed=='ensemble' and horizon_months==24 and cause=='termination'").sort_values(['scope','arm'])
    ci=pd.read_csv(e/'paired_bootstrap_intervals.csv',dtype={'scope':str})
    dif=pd.read_csv(e/'paired_differences.csv',dtype={'scope':str,'seed':str})
    ps=pd.read_csv(e/'prediction_stability.csv',dtype={'scope':str})
    st=pd.read_csv(e/'function_stability.csv')
    cols=['scope','arm','auc','ap','brier','risk_mean','ipcw_observed_incidence','top20_captured','top20_coverage']
    main[cols].to_csv(out/'primary_ensemble_metrics.csv',index=False)
    pd.DataFrame(registry(run)).to_csv(out/'artifact_inventory.csv',index=False)
    searches=single.groupby(['year','arm'])[['finite_cuts','missing_only_cuts','routed_attempts','scored_candidates','features_with_fewer_than_five_cuts']].sum().reset_index()
    searches.to_csv(out/'search_counts.csv',index=False)
    supports=single.groupby(['year','arm'])[['leaves','zero_target_leaves','one_or_zero_target_leaves','borrowed_positive_target_leaves']].sum().reset_index()
    supports.to_csv(out/'leaf_support_summary.csv',index=False)
    # These are aggregate variable-index/tree counts, never company rows or thresholds.
    feature_rows=[];pair_rows=[]
    for y,a,s,p in verification_outputs(run):
        if s=='ensemble':continue
        for name,target in [('feature_coverage',feature_rows),('pair_coverage',pair_rows)]:
            z=pd.read_csv(p/f'{name}_PRIVATE.csv');z.insert(0,'seed',s);z.insert(0,'arm',a);z.insert(0,'year',y);target.append(z)
    pd.concat(feature_rows).to_csv(out/'feature_index_coverage.csv',index=False)
    pd.concat(pair_rows).to_csv(out/'pair_index_coverage.csv',index=False)
    coverage=pd.read_csv(e/'coverage_summary.csv').groupby(['year','arm'])[['eligible_min','eligible_max','eligible_sd','actual_split_features']].mean().reset_index()
    stable=st.query('day==730 and cause==1').groupby(['year','arm','order'])[['nonzero_jaccard','common_terms','nonconstant_terms','median_correlation','mean_rms_difference']].mean().reset_index()
    stable.to_csv(out/'function_stability_summary.csv',index=False)
    adjacency=dif.query("seed=='ensemble' and horizon_months==24 and cause=='termination' and not contrast.str.contains('cumulative')")
    adjacency.to_csv(out/'adjacent_primary_differences.csv',index=False)
    interval=ci.query("scope=='pooled' and not contrast.str.contains('cumulative') and metric in ['auc','ap','brier','top20_captured']")
    figure(main,cc,out)
    md('RESULTS_ZH.md',f'''# 累积消融：完整模型保住紧凑表示，预测贡献并非步步同向

已按用户指定顺序完成：s0完整q2/共享/均衡/收缩 → s1解除q限制 → s2再改局部随机切点 → s3再改IID25 → s4再取消收缩。新增12次正式森林拟合、12个收缩派生版本；前两阶段复用12份历史模型；最后一步直接使用s3的raw森林，没有额外拟合。全部12次拟合和40项完整验证通过，失败0。当前研究参考和历史结果不变。

所有阶段均为深度4、每树25个候选变量、每种子32树、三个固定种子。下表先平均三个种子的逐公司预测，再计算24个日历月内先退市的指标。2023测试36家/12例，2024测试45家/20例；合并81家/32例。两年已被多次查看，本轮为**探索性机制分析**，继承原始财报版本/PIT的PROVISIONAL限制。

{pretty(main[cols])}

Top20每年向上取整，分别核查8家、9家；pooled合计17家，不重新按81家确定名额。合并AUC/AP/Brier按合并预测重算，未平均年度指标。

![累积消融](results/cumulative_comparison.png)

1. **解除q限制：表示代价清楚，预测效果分年度。** 2023 AUC不变、AP提高0.073562；2024 AUC/AP下降0.034000/0.038357，少捕获1例。合并AUC/AP下降0.007015/0.019037。函数空间从3,146/3,263升到76,668/89,517格。q2相对这一步减少95.90%/96.35%的逻辑空间单元，但不能据此宣称每年都没有预测代价。
2. **再改局部随机切点：本条链上24个月排序进一步变差。** 合并AP下降0.069580，Brier上升0.003686，Top20少1例；2023 AP却提高0.033899。总格数反而小降1.59%/0.90%，因为树结构及函数集合也随之改变。此前单项q2对照中共享方案减少格数，此处无q限制背景下没有同向总量证据。不能把共享切点的复杂度收益说成普遍定理；该对照也同时改变候选生成规则。
3. **再改IID25：出现对均衡覆盖有利性主张的反向证据。** 2024 AUC/AP从0.688/0.630994回升至0.728/0.701951；合并AP提高0.039218、Brier下降0.003774、Top20从10/32回到12/32。2023 AUC/AP下降，合并AUC几乎不变。均衡覆盖确实均衡了资格次数，但这里未表现为一致的预测收益。
4. **再取消收缩：排序变化小，概率质量和稳定性有权衡。** 2024 AUC不变、AP略降0.000863、Brier改善0.004488；2023 AUC下降0.010417、AP提高0.006817、Brier变差0.001401。合并Top20仍12/32，Brier改善0.001870。取消收缩不改变空间格数，但平均跨种子预测标准差从0.013675升到0.017115。收缩的稳定作用得到这个对照支持，不能同时宣称其Brier必然更优。

完整模型与全部移除版本相比：合并AUC为0.729592对0.699617、AP为0.681719对0.629898，Top20均12/32；Brier为0.315749对0.313871，后者略好。当前框架不是所有指标的全面胜者，保留它的证据主要是明显更小的完整函数表示，以及本轮24个月合并排序点估计的优势。

相邻步骤的合并配对差值及描述性95%区间：正差对AUC/AP/捕获数有利，对Brier不利。

{table(interval[['contrast','metric','estimate','lower95','upper95']])}

完整模型与全部移除版本的合并AUC/AP/Brier/Top20区间均包含0；部分中间步骤的AP或Brier区间不含0，也不能选择性升级为确认性结论。1000次公司配对、按年度分层重采样只反映固定预测条件下测试样本波动，不含重新训练、方法选择、多次研究接触的不确定性，未做多重比较校正。

**研究结论：q2的表示压缩价值有直接证据；四项设计各自普遍提升预测的主张不成立。** 新证据支持保留当前参考进行设计取舍说明，不足以宣称算法普遍优越，也不自动选择新参考。所有增量都是此固定删除顺序下的条件差异，不是顺序无关的独立贡献。

全部240行年度/种子/合并指标、384行配对差值、192行区间及完整数值诊断保存在results/；12个月和恢复原因的反向结果见PREDICTION_COMPARISON_ZH.md。
''')
    secondary=m.query("seed=='ensemble' and (horizon_months==12 or cause=='recovery')")
    bins=pd.read_csv(e/'calibration_bins.csv',dtype={'scope':str})
    md('PREDICTION_COMPARISON_ZH.md',f'''# 预测、概率水平与全部次要结果

主终点：入组后24个日历月内，先于完整恢复发生最终退市。恢复作为竞争结局进入同一CIF模型。完整主指标见RESULTS_ZH.md及primary_ensemble_metrics.csv；单种子和三种子平均均在metrics.csv。

{pretty(main[cols])}

所有测试公司在12/24个月终点都可判定，沿用的IPCW评价器权重全部为1，因此本次IPCW Brier等于普通均方误差。未知0，未将训练删失当作无事件。每家公司一条记录，记录等权与公司等权一致。Top20逐年度名额8＋9；Top25次分析9＋12，全部保留于warning_metrics.csv。已知非目标包含先恢复和窗口内没有先退市，不能称永久正常企业。三个种子不是三个独立验证队列。

风险低估仍然明显：合并观察退市39.51%，五阶段平均预测仅8.98%—9.51%。2023为6.03%—6.75%对33.33%，2024为11.34%—12.27%对44.44%。没有拟合或应用任何校准器、概率放大或clipping。合法的CIF不等于校准良好，AUC/AP也不能回答概率水平是否准确。

合并主终点各阶段四等分校准（各自预测升序，分箱人数固定20/21附近）：

{pretty(bins.query("scope=='pooled' and horizon_months==24 and cause=='termination'")[['arm','bin','n','risk_mean','observed','events']])}

全部次要终点结果：

{pretty(secondary[['scope','arm','horizon_months','cause','auc','ap','brier','risk_mean','ipcw_observed_incidence']])}

12个月合并先退市：全部移除后的AUC/AP为0.846721/0.751104，高于完整模型0.799180/0.634071，Brier也更低。这与24个月主分析排序不同，明确保留，不改变预先冻结的主终点。24个月恢复：s3合并AUC/AP/Brier为0.702581/0.652458/0.231997，完整模型为0.673548/0.627720/0.242559。设计取舍与评价期限、原因相关。2023的12个月恢复无阳性事件，AUC/AP保留NA。

所有相邻与相对s0差值见paired_differences.csv。pooled预测来自两个不同训练截止，含年度水平差异，必须与年度结果一起解释。未使用本轮结果改参数、增加拟合或替换参考。
''')
    md('COMPLEXITY_COMPARISON_ZH.md',f'''# 完整函数复杂度、稳定性与真实支持

每年用同一完整训练集的经验边际乘积作参考，编译各模型的全部非空阶次。q2只包含基准、主效应和二元函数；s1—s4实际路径上限均为4，必须包含三阶和四阶。下表为三个种子等权平均的完整96树预测器，不是平均三份模型的函数数量。

{pretty(cc[['year','arm','max_path_variables','main_functions','interaction_functions','order3_functions','order4_functions','spatial_table_cells','scalar_table_cells','function_file_bytes','low_support_cell_fraction']])}

spatial_table_cells为每个结构项对应各坐标有限区间加独立缺失状态后，笛卡尔积单元数之和；不合并相邻等值格，训练无缺失支持的缺失状态仍计数。单独基准曲线不在该列；含基准的完整单元数另列为该数＋1。标量数再乘原生时间点数和2个原因。单种子与96树完整预测器均各自重编译/重计数，见single_seed_complexity.csv、ensemble_complexity.csv。

各阶structural_functions与非零functions均报告；非零仅表示所有格点/原生时点中存在幅度>1e−14，不是统计显著。路径达到四阶不自动代表四阶贡献大。完整函数采用精确因子化存储，文件大小不等于展开逻辑格数。s3与s4空间完全相同，但raw文件更小，涉及系数可压缩性，不能当作函数分区变简单。

解除q后格数大幅增加是稳定出现的机制事实。共享改局部切点时，本条链空间反而小降；变IID25再降1.29%/4.66%，均远未回到q2规模。完整模型相对s4减少95.78%/96.14%的逻辑空间。高阶模型若强行仅保留二阶项，在365/730天、两原因、全训练/评价样本上的最大近似残差可达约4.80个百分点（s4/2024）；本轮没有使用该截断近似，全部高阶函数已纳入验证。

完整项求和最大绝对误差{num['max_component_sum_error']:.3e}，最大重构误差{num['max_reconstruction_error']:.3e}，均低于1e−10。每个模型在全部训练/评价公司、全部原生时间点、两原因检查；保存重载、CIF非负/单调/原因和不超过1均通过。没有删除项或裁剪概率来通过验证。

730天退市函数稳定性：每年三对种子统计再平均，相关性只对共同、非恒定函数计算。共同函数在同年全部训练行上比较，参考分布仍为经验边际乘积。

{pretty(stable)}

q2共同二元函数平均相关性0.650/0.506，四阶方案的二元函数相关性明显较低，但q2交互集合Jaccard也只有0.073/0.092，不能称其变量对选择已稳定。高阶版本主效应Jaccard较高可能部分来自集合更稠密；四阶变量集合在三对种子间均无共同项，其相关性NA是无可比函数，不是0。函数RMS与幅度有关，较低不自动等于可靠。

24个月退市预测跨种子波动（各公司标准差ddof=1再平均）：

{pretty(ps.query('horizon_months==24 and cause==1')[['scope','arm','seed_sd_mean','seed_sd_max']])}

取消收缩后两年的预测波动均增加，但s3与s4的函数集合不变。收缩改变系数，不能增加真实支持。各阶段跨种子叶支持合计如下，叶节点会重复涉及同一公司，不是独立样本量：

{pretty(supports)}

s3有676/556个原始零退市叶，其中211/204个从父节点借入正风险；s4仍有同样676/556个零退市叶，借入数为0。高阶模型低支持格（训练公司数<10）约73%—75%，q2约41%—44%；额外格数多数缺少强支持。参考乘积打破相关特征的联合依赖，解释不代表条件效应或因果影响，精确重构本身不证明统计可辨识性。
''')
    tasks=pd.read_csv(e/'execution_tasks.csv')
    resources=tasks.groupby(['stage','host'])[['seconds','peak_rss_MiB']].agg(['count','min','max']);resources.columns=['_'.join(z) for z in resources.columns]
    md('TRAINING_DIAGNOSTICS_ZH.md',f'''# 配对、成本与数值核验

设计源工作区提交ad49e66c25f6e76ad8206ddaf7f334ac4fcbd88c；预拟合代码/协议提交`{freeze['code_commit']}`。拟合前完整测试155通过、0跳过，独立R数值测试执行。交付前完整测试亦155通过；唯一警告为已有EBM缺失值可视化提示，与本轮森林无关。

正式Slurm2575在约7.73秒内完成12项计算及发布，12项实际均由oliver处理；两机池可用，任务短而未重复拟合以填满两机。验证Slurm2576共40项，oliver24、lemon16，计算及发布约21.79秒，含排队启动约22.41秒。没有宣称单进程跨两机加速或使用GPU。单任务单线程、4GiB地址空间上限，遵守调度器亲和性，不硬编码CPU或绕过P/E核配置。

{table(resources.reset_index())}

上述是作业耗时，不是代码审计、测试、分析和报告的总研究耗时。model_costs.csv记录拟合/收缩/预测成本，numerical_complexity_cost.csv记录编译、导出、验证等。历史与本次硬件并发和计时方式不同；缺失历史计时保留NA，raw/收缩共享一次拟合成本，不能重复相加。

同年/种子/树编号的公司bootstrap索引及重复次数哈希全部相同。s0/s1/s2每树变量资格集合相同；IID25独立无放回均匀抽83列中的25列，使用独立随机流，未消费bootstrap/节点随机流。s3/s4结构、阈值、缺失路由完全一致。独立直接AJ逐节点重放，收缩权重使用独立in-bag公司数n/(n+10)，原始直接父节点曲线，非递归/重复收缩，误差0。

资格覆盖汇总（先每种子统计再平均；更完整的每变量/变量对计数见索引覆盖CSV）：

{pretty(coverage)}

候选重放统计（三种子合计）：

{pretty(searches)}

finite_cuts是节点实际有效有限切点数；routed_attempts含被支持约束拒绝的候选，scored_candidates才是实际调用分裂评分次数。保持n_thresholds=5不等于相同搜索量，缺失路由也影响次数。逐节点获胜候选、保存评分和总计均重放一致；未更换搜索语义。

12个新raw森林、12个对应收缩版本均保存；主链包含30个单种子模型，另有6个s2 raw辅助工件保留。s0/s1复用12份历史模型并重新预测验证，96行历史指标精确再现，没有“基线重现”拟合。40项验证覆盖30个单种子及10个96树完整预测器。拟合、发布、数值验证、分析失败均0；未隐藏种子或追加训练。

1000次固定公司抽样索引与上一轮逐项一致，全部模型复用同组抽样；192个主要差值区间中不可计算重采样次数{int(ci.undefined_replicates.sum())}。区间条件于已拟合预测，不代表包含训练与研究选择的总不确定性。
''')
    counts=pd.read_csv(run/'reports/cohort_counts.csv')
    md('DATA_ALIGNMENT_AUDIT_ZH.md',f'''# 数据保持旧版conservative C，未重建样本

{table(counts)}

两折全部83列X、列顺序、公司/样本对齐、事件编码、原点、随访和训练截止沿用上一轮权威审计，逐字节哈希相同。输入哈希见input_hashes.csv，复用模型位置与哈希见artifact_reuse.csv、artifact_inventory.csv及model_hashes.csv。源工件在拟合前、统一评价前、交付时再次核对，当前参考配置文件哈希未变。

研究对象为全部非金融行业、观察历史内首次财务/审计类ST后预后；时间起点为入组公告后下一验证交易日开盘前。先最终终止上市X=1与先完整恢复A=2均按公告日期，先恢复后退市仍计先恢复。训练信息严格早于外层测试年1月1日，随访截断于24个日历月。评价数据覆盖截至2026-09-03，未随当前日期自动延长。派生特征使用旧保守可得时钟，不采纳此前的年报原始版本覆盖实验。

2023训练309家/14例退市/184例恢复，2024训练345家/26例退市/198例恢复；两折合计654行，去重345家公司、26个目标事件。新增12次拟合重复读取3,924行计算记录，不能称3,924个独立样本。测试36＋45＝81家公司，24个月先退市12＋20＝32，先恢复15＋16＝31；12个月先退市4＋16＝20，恢复0＋5＝5。测试两时点未知均0；训练未完成随访仍为删失。

原始财报披露版本/PIT及“观察历史内首次”的局限保留，未新获得独立原始文件认证，研究仍为PROVISIONAL数据口径。2023/2024已经反复接触，本轮为固定顺序探索性机制分析，不是独立盲测。没有读取2025、增加年度、改变研究样本或因结果调整评价记录。
''')
    md('PAPER_METHODS_RESULTS_ZH.md','''# 可用于论文的实验方法与结果表述

在既定非金融上市公司ST后竞争风险预后任务中，采用固定顺序累积消融，依次解除路径不同变量数上限q=2、将共享分位数切点改为节点局部随机切点、将均衡变量覆盖改为每树独立无放回抽取25个变量，以及取消直接父节点Aalen–Johansen收缩。各步骤保留此前的移除，其他设置固定为深度4、32树/种子、三个随机种子及同一83列输入。每年以三个种子逐公司预测的等权平均作为主要预测器。前两阶段复用已验证工件，中间两阶段各拟合6个森林，末阶段使用相同森林的raw叶估计，共新增12次拟合。训练公司bootstrap在年度/种子/树编号间配对，IID变量抽样使用独立随机流。

主要评价时点为入组后24个日历月，结局为先于恢复发生最终退市，次要分析包括12个月及恢复原因。2023、2024队列分别包含36、45家公司及12、20个主要目标事件。报告年度和重新合并计算的AUC、AP、IPCW Brier、平均风险与按年度Top20名额合计的事件覆盖。固定预测采用1000次按年度分层的公司配对bootstrap描述差值不确定性。函数表示采用训练经验边际乘积参考，按实际路径阶次完整展开，分别报告非零项、结构项、含缺失状态的逻辑空间格数及因子化文件大小；全训练/评价样本、所有原生时间和两原因均验证精确重构。

完整模型的合并24个月AUC/AP为0.7296/0.6817，全部移除版本为0.6996/0.6299，Top20均捕获12/32个目标事件；Brier分别为0.3157和0.3139。完整模型的函数空间为3,146/3,263格，全部移除版本为74,469/84,571格，减少95.78%/96.14%。累积效应并非单调：局部随机切点降低合并24个月AP，而其后改IID25改善AP、Brier和事件覆盖；取消收缩增加预测跨种子波动但改善合并Brier。2023部分移除步骤的AP较高，12个月合并结果亦偏向全部移除版本。因此结果支持低阶约束的表示压缩作用及收缩在当前对照中的稳定性作用，不支持所有设计在所有指标上均有独立正收益的概括。

所有完整函数通过数值重构，最大项求和误差6.33×10⁻¹⁵。该性质不意味着函数集合稳定、概率充分校准或解释具有因果含义；五阶段合并24个月平均退市风险约9%，均明显低于观察发生率39.5%。完整模型与全部移除版本的主要性能差值区间均包含0。两个测试年度已被多次查看，数据原始版本仍有PIT限制，以上应作为探索性设计取舍证据；累积对比仅对应既定移除顺序，不能解释为顺序无关的独立贡献或普遍优越性。
''')
    receipt=dict(status='COMPLETE',experiment='post_st_cumulative_ablation_v1',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),private_run=str(run),source_commit=SOURCE_COMMIT,training_code_commit=freeze['code_commit'],new_forest_fits=12,new_shrunk_models=12,reused_chain_models=12,baseline_reproduction_fits=0,chain_single_models=30,chain_ensemble_predictors=10,auxiliary_s2_raw_models=6,formal_job=2575,verification_job=2576,successful_fit_tasks=12,successful_verification_tasks=40,failed_tasks=[],full_tests_passed=155,full_tests_skipped=0,training_unique_companies=345,training_unique_target_events=26,training_fold_records=654,repeated_training_rows_across_new_fits=3924,test_companies=81,test_target_events_24m=32,reference_unchanged=True,reference_sha256=sha(reference),exploratory=True,provisional_data_clock=True,bootstrap_replicates=1000,numerical_validation=num,not_executed=['2025','other_removal_orders','additional_tuning','calibration','reference_replacement'],sync_evidence=str(run/'audit/remote_sync.json'),delivery_commit_note='Resolve the commit containing this receipt; remote SHA is recorded separately after push to avoid self-reference.')
    receipt['analysis_source_sha256']={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}
    write(dest/'execution_receipt.json',receipt);write(run/'execution_receipt.json',receipt)
    write(out/'delivery_hashes.json',{str(p.relative_to(dest)):sha(p) for p in dest.rglob('*') if p.is_file() and '__pycache__' not in str(p) and p.name!='delivery_hashes.json'})
    print(json.dumps(dict(status='COMPLETE',new_fits=12,verified_predictors=40,reports=str(dest))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--dest',type=Path,default=Path(__file__).parent);a=p.parse_args();render(a.run,a.dest)
