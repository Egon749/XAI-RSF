"""Publish aggregate tables, figures and evidence-backed Chinese reports."""
from pathlib import Path
import datetime,importlib.metadata,platform,shutil,subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import *

LABEL={'baseline':'原模型','quadratic':'原因平方分数','support':'退市支持保护','quadratic_support':'二者同时'}
PUBLIC=REPO/'experiments'/NAME

def table(df):
    def val(x):
        if pd.isna(x):return 'NA'
        if isinstance(x,(float,np.floating)):return f'{x:.4f}'
        return str(x).replace('|','/')
    lines=['| '+' | '.join(map(str,df.columns))+' |','| '+' | '.join(['---']*len(df.columns))+' |']
    return '\n'.join(lines+['| '+' | '.join(val(x) for x in r)+' |' for r in df.itertuples(index=False,name=None)])

def report(name,text):(PUBLIC/name).write_text(text,encoding='utf-8')

def generated_json(path,value):
    """Only derived report JSON is revisable; archive previous bytes first."""
    path=Path(path)
    if path.exists():
        archive=RUN/'audit/report_revisions';archive.mkdir(exist_ok=True)
        old=archive/(path.stem+'_'+sha(path)+'.json')
        if not old.exists():shutil.copy2(path,old)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    for p in (RUN/'reports').iterdir():
        if p.suffix in ['.csv','.json']:shutil.copy2(p,PUBLIC/p.name)
    m=pd.read_csv(RUN/'reports/METRICS_ENSEMBLE.csv');m=m[m.version=='shrunk'].copy()
    def metric_table(h,c):
        d=m[(m.months==h)&(m.cause==c)].copy();d['前20%捕获']=d.apply(lambda r:f'{int(r.captured)}/{int(r.target_events)}' if pd.notna(r.captured) else 'NA',axis=1)
        d['arm']=d.arm.map(LABEL)
        return table(d[['scope','arm','auc','ap','brier','前20%捕获','risk_mean','observed']].rename(columns={'scope':'队列','arm':'模型','auc':'AUC','ap':'AP','brier':'IPCW Brier','risk_mean':'平均预测风险','observed':'观察发生率'}))
    n=pd.read_csv(RUN/'reports/NUMERICAL_COMPLEXITY.csv');ens=n[(n.seed=='ensemble')&(n.version=='shrunk')].copy()
    ens=ens.sort_values(['year','arm']);ss=pd.read_csv(RUN/'reports/SEARCH_DIAGNOSTICS.csv');leaf=pd.read_csv(RUN/'reports/LEAF_SUPPORT.csv')
    root=pd.read_csv(RUN/'reports/ROOT_SCORE_MECHANISM.csv');fs=pd.read_csv(RUN/'reports/FUNCTION_STABILITY.csv');cost=pd.read_csv(RUN/'reports/COMPUTE_COST.csv')
    fv=load(RUN/'reports/function_validation.json');mv=load(RUN/'reports/metric_validation.json')
    ci=pd.read_csv(RUN/'reports/PAIRED_BOOTSTRAP_95.csv')
    report('RESULTS_ZH.md',f'''# 原因分裂准则与退市支持保护：真实匹配消融

结论：保持当前83变量、q2、深度4＋父节点AJ收缩模型。新分裂准则确实使森林更重视退市信号，且表示更小，但没有得到跨年度一致的预测收益；退市支持保护消除了单家公司支撑的退市叶，却没有证明稳定提高预测。两者同时加入也未形成可靠的协同收益。此次不自动升级参考模型。

## 执行与口径

完成36/36正式raw森林拟合（其中9次基线重现、27次新设计）、36个收缩派生版本，另2/2四树数值试运行。24个96树完整平均预测器由保存模型合并并验证，新增拟合数为0。三种子42/101/202先逐公司平均，再评分；单种子与raw结果全部保留。原参考9个模型的树结构与预测精确重现。正式作业2613，合并验证2614，两台机器均参与；无正式失败或拟合重试。

各年1月1日前训练，企业数309/345/390，已知退市14/26/49，恢复184/198/211；年度训练集相互嵌套，不能把这些事件相加当独立事件。三年评价36/45/53家公司。原始披露版本仍继承PROVISIONAL限制；旧元数据日期与字段级时钟的核对见DATA_ALIGNMENT_AUDIT_ZH.md。

四组仅改变冻结的分裂分数和退市支持开关。原因平方分数消除有符号抵消，同时改变逐原因方差标准化；本实验不能把其全部效果归因于“消除抵消”。支持保护允许0个退市，但不允许仅1家退市公司支撑的子节点，至少2家时才保留；不是要求两个子节点都有退市，也不是新增样本。

## 12个月：先最终退市

{metric_table(12,1)}

## 24个月：先最终退市

{metric_table(24,1)}

2025年24月的6例早期退市不能代替完整24月标签：尚25家未知、无人达到24月随访，因此所有该窗口性能NA。12月pooled为三年134家公司、25例退市；24月pooled仅2023/2024共81家公司、32例退市。合并AUC/AP/Brier从合并预测重算。Top20各年名额为8、9、11，pooled先按年确定名额再合计，不进行跨年统一截断。

## 12个月：先恢复

{metric_table(12,2)}

## 24个月：先恢复

{metric_table(24,2)}

2023年12月无恢复事件，AUC/AP不可估计；Brier、均值仍可评价。其他支持窗口结局均已知，竞争事件作为非该原因结局，IPCW权重实际均为1，Brier与普通二元平方误差一致。没有把未知结局当成正常。METRICS_ALL.csv保存512行年度/合并、种子/均值、raw/收缩、两原因双时点结果。

## 如何理解差异

1. 原因平方分数在2023年12月将退市AP从0.199提高到0.514，Top20从1/4到2/4；但仅4个事件，不能据此宣布成功。2024年12月AUC从0.806降到0.705、AP从0.718降到0.573；24月AUC0.770→0.714、AP0.782→0.728，Top20保持8/20。2025年12月也未改善AUC/AP。
2. 2024年24月Brier0.3396→0.3196，说明概率误差有所减小，与排序变差可以同时发生。平均退市风险仅11.58%→12.91%，仍远低于观察44.44%。新准则没有解决严重的整体风险低估。2023年24月均值还下降6.47%→5.88%。
3. 单独支持保护在2024年12月AUC0.806→0.817、Top20从6/16到7/16，但AP下降0.718→0.676；同年24月Top20从8/20降为7/20。2025年12月变化很小，不能称为稳健收益。
4. 恢复预测没有得到一致保护。原因平方分数令2024年24月恢复AUC0.707→0.653、AP0.566→0.503；两种改动的12月pooled恢复Brier均比原模型更差。退市和恢复的取舍必须一起报告。
5. 两开关的差分之差已经完整计算，没有稳定协同改善。raw版本和收缩版本另见PAIRED_DIFFERENCES.csv，不能只展示最有利种子或年度。

raw次分析同样不支持把问题完全归因于收缩：24月合并原模型raw AUC/AP为0.724/0.680，新分数为0.717/0.663；收缩后为0.730/0.682与0.722/0.660。新准则的24月合并排序损失在收缩前已经存在。收缩对各设计的AP、捕获和Brier作用也不同；没有因此追加新强度搜索。

## 不确定性与研究判断

固定模型、按年度分层、公司为单位的1000次共同重采样已完成。各臂减原模型的pooled退市AUC/AP/Brier/Top20差值95%区间均包含0。2024年原因平方分数的24月Brier差值约−0.0199，描述性区间[−0.0420,−0.0002]；同时两开关组合的12月AP差值约−0.1595，区间[−0.2775,−0.0209]。必须同时报告不利证据，不能在多指标、多对照中挑选某个区间作为确认性成功。完整结果及无事件重采样次数见PAIRED_BOOTSTRAP_95.csv。

这些区间不包含训练样本、筛选和历次调参的不确定性。2023—2025已被查看，本轮是探索性机制分析，不是独立盲测。没有未来标签参与本次配置选择。没有证据支持替换当前参考，也没有证据说明“类别不均衡已经解决”。

数值性质全部保留：q≤2，所有训练及评价样本、全部原生时间点、两个原因的完整展开最大显式重构误差{fv['max_explicit_sum_error']:.3e}，保存重载误差0，无裁剪、无代理解释。结构更小不自动等于预测更可靠；详见INTERPRETABILITY_COMPARISON_ZH.md。

![退市预测比较](PREDICTION_COMPARISON.png)
''')
    ctab=ens[['year','arm','main_functions','interaction_functions','spatial_table_cells','scalar_table_cells']].copy();ctab.arm=ctab.arm.map(LABEL)
    st=fs[(fs.version=='shrunk')&(fs.day==730)&(fs.cause==1)&(fs.order==2)].groupby(['year','arm'])[['order1_jaccard','order2_jaccard','median_correlation','mean_rms_difference']].mean().reset_index();st.arm=st.arm.map(LABEL)
    report('INTERPRETABILITY_COMPARISON_ZH.md',f'''# 完整表示、稳定性与支持

全部72个单种子raw/收缩模型与24个三种子合并预测器均完成完整函数导出、保存重载及全部训练/评价样本×原生时间×两原因检查。最大显式分量相加误差{fv['max_explicit_sum_error']:.3e}，最大路径变量数2，森林重载误差0。叶AJ仍是原估计器；分裂改动不改变精确q2表示的数学依据。收缩是叶节点唯一公司数n/(n+10)与该树raw直接父节点AJ的凸组合，两个原因/全部时间使用同一权重。

## 三种子平均预测器的完整复杂度

{table(ctab)}

函数数为整个原生时间网格、两个原因上真正非零的函数（既有阈值1e−14）。空间单元数含缺失状态与结构上存在但函数可能为零的单元；标量存储量再乘时间和原因维度。每年度是96树完整预测器，不是32树复杂度的算术平均；单种子数量、导出文件字节及时间均在NUMERICAL_COMPLEXITY.csv。

原因平方分数使三个年度合并空间单元分别减少8.8%、11.0%、8.1%，2024/2025二元函数也减少。但单独支持保护2023二元函数144→149、2024为146→150，不能说该保护一定简化全局函数表。

## 算法随机性

下表为三对种子比较的均值；函数相关性先在每对种子的共同二元函数中取中位数，再平均。参考分布为同年度训练数据的边际乘积分布；在同一年度训练样本上比较幅度/形状。相关特征下该分解不提供因果或唯一统计机制保证。

{table(st)}

原因平方分数提高了二元集合Jaccard和共同函数相关性，但Jaccard仍只有约0.095—0.127，交互选择并不高度稳定。它同时提高了退市贡献幅度：96树、730天的二元函数平均绝对贡献之和，三个年度由0.0294/0.0357/0.0431变为0.0381/0.0487/0.0682。这是模型函数幅度，不是真实作用恢复率，也不是风险变化百分点的直接总和。

退市24月逐企业预测的跨种子标准差反而增加：2023平均0.00834→0.01205，2024为0.01351→0.01741，2025预测为0.01313→0.01749（2025此项仅是预测波动诊断，不评价未成熟结局）。因此“共同函数形状较一致”不能替代“整体预测稳定”。单独支持保护对稳定性的影响混合。

## 事件支持保护的真实作用

{table(leaf.groupby('arm')[['leaves','zero_target_leaves','one_target_leaves','borrowed_positive_target_leaves']].sum().reset_index())}

表中叶节点是9个种子年度模型的合计，不能视为独立公司。保护将正退市支持仅1家公司叶由606/491个降到0，说明开关确实生效。但零退市叶仍大量存在，原模型1373个，单独保护1472个，二者同时1401个；保护并不创造事件或消除零事件叶。borrowed列是原叶无退市但由直接父节点借入正风险的叶数，收缩不增加真实支持。

本轮支持完整精确解释可兼容新分裂与支持约束；不支持“新的约束已经解决稀少事件、显著提升解释可靠性”这一更强主张。
''')
    searches=ss.groupby('arm')[['selected_nodes','opposed_selected','scored_candidates','guard_rejected','guard_changed_winner','guard_stopped_split']].sum().reset_index()
    roots=root.groupby('arm')[['mean_cancellation_fraction','mean_target_z','mean_recovery_z','mean_target_quadratic_share']].mean().reset_index()
    report('TRAINING_DIAGNOSTICS_ZH.md',f'''# 训练和机制诊断

完整测试244项通过，含独立R检验；唯一提示来自既有EBM缺失值可视化，与本轮森林无关。14项新增测试覆盖反向原因抵消、独立风险集计算、权重尺度不变、bootstrap唯一公司计数、无事件边、无效参数、旧模型兼容、四臂配对及q2精确函数。

正式36次拟合成功，两台节点oliverdesktop32项、lemon4项；32棵树/种子、各自单线程，未假设单训练跨机加速。pilot2项，合并函数验证24项（oliver10、lemon14，无新拟合）。Slurm2612/2613/2614均成功计算、回传并验证输出SHA256。正式执行墙钟约18.03秒（含队列18.92秒）；各任务fit时间2.26—3.49秒，累计fit工作时间103.30秒。正式任务峰值RSS最大334.12MiB，4GiB上限未触发。集群启动、结果分析及报告时间不计入森林fit时间；不宣称该布局已证明全局最快。

## 候选与支持

{table(searches)}

scored_candidates为实际计算分数的候选，包含随后被支持保护拒绝的候选；routed_attempts另在CSV保存，含早期行数/事件/公司限制过滤的候选。支持拒绝24525和24087个候选，分别改变669和550个节点的最佳候选，其中227和210个节点完全停止分裂。各臂路径改变，节点总数不同，不能把它们当作一一对应节点的因果实验。每个观察到的节点候选、RNG、胜出分裂已重放，最大独立分数误差{fv['max_independent_score_error']:.3e}。

## 根节点：新分数是否确实改变原因取向

{table(roots)}

cancel_fraction=1−|U退市+U恢复|/(|U退市|+|U恢复|)，仅用于描述旧有符号组合的抵消。根节点有相同bootstrap风险集，所以比不同深度节点更可比。新分数选出的根节点平均退市标准化幅度3.86→6.05，退市占平方分数份额19.0%→42.3%；相应有符号抵消比例29.3%→47.3%。这支持“新准则确实允许更多方向相反且退市更强的候选胜出”。不能推出这些训练内更强的信号在未来都是真信号；2024的外推退步和跨种子预测波动增加正是必要反证。原因独立标准化也提高了稀少原因相对影响，效应并非只有抵消这一处。

## 数值与工件

72个单种子版本＋24个完整平均预测器全部合法，最大显式重构误差{fv['max_explicit_sum_error']:.3e}；直接父节点AJ重放误差0、保存森林重载误差0。9次原模型重现树/预测完全相同。36组×32树的bootstrap索引/重复次数及25变量集合配对一致。全部SHA256、配置、输入哈希、资源、失败说明见execution_receipt.json、MODEL_HASHES.csv和私有audit目录。

未发生正式拟合失败或重训重试。预检曾因旧元数据日期断言停止；随后查证权威字段级证据，未改输入。一次只读汇总在合并工件就绪前启动，报缺少registry；等待作业完成后重跑该汇总，没有重训。运行目录创建预检与诊断脚本问题亦保留于回执，不隐藏为计算成功。
''')
    report('METHOD_INNOVATION_DECISION_ZH.md','''# 方法判断

决定：KEEP_CURRENT_REFERENCE；有限消融已完成，不自动替换模型，不追加搜索。

- 机制实现有效：原因平方分数去掉有符号交叉抵消，独立公司支持保护确实杜绝单公司正退市叶；两者保留森林bootstrap、q2路径、共享切点、原AJ与直接父节点收缩，全部精确CIF解释成立。
- 外推增量未确立：新分数在2023较好、2024排序较差、2025没有一致收益；保护的改善局部，且可能牺牲24月捕获。合并退市主要差值区间仍跨0，没有可靠协同增量。
- 校准问题仍在：2024概率误差改善不等于风险水平已准确，平均退市概率仍严重偏低；恢复结果也存在代价。
- 解释收益有限：新分数得到更小函数表、部分函数重合与形状一致性较高，但全局交互Jaccard仍低，预测种子波动反而更大。不能把精确重构当真实机制恢复，也不能把借入风险当新增事件。
- 创新定位：内生可解释森林的价值在于训练结构受限且完整可展开。新增原因平方分数与最小事件支持是可审计的候选改动；本轮并未证明它们是独立有效的方法创新，也不主张平方组合或最低事件数本身具有文献新颖性。
- 当前版本保留：依据现有探索性证据无法推荐新默认值。不同年度表现与已暴露队列限制必须在论文中完整说明，不能追随2023的正面AP或2024的Brier选择性宣告成功。
''')
    report('PAPER_METHODS_RESULTS_ZH.md',f'''# 可用于论文的消融方法与结果

## 方法

以83变量、路径变量数至多2、深度4、每种子32棵树、三随机种子及直接父节点Aalen–Johansen收缩强度10的内生可解释随机生存森林为参照，设置2×2有限匹配消融。第一因素将原因偏差的有符号加权logrank组合替换为逐原因标准化偏差平方的加权均值开方，避免相反方向的原因信号在组合时抵消。该替换同时改变逐原因标准化，因此其效果被解释为整个分裂准则的变化。第二因素禁止生成仅由1家独立退市公司支撑的子节点，允许0家或至少2家；支持按bootstrap内独立公司计数。其他参数、公司bootstrap及每树可用变量集合保持一致。

分别使用2023、2024、2025年1月1日前可得信息拟合，按首次ST公告后下一交易日开盘前为原点预测先最终退市与先恢复。全部设计在新拟合前冻结；每年先平均三个种子的公司CIF，再评价有支持的12/24个月AUC、AP、IPCW Brier和年内Top20捕获。2025年24个月因随访不足不评分。所有raw及收缩预测均保留。所有模型在全部训练及评价样本、全部原生时间和两原因上验证完整基线、主效应及二元函数展开，并按相同训练参考分布比较结构与稳定性。采用固定预测、按测试年分层的1000次配对公司bootstrap给出描述性区间。本研究队列已经暴露，结果属于探索性机制分析。

## 结果

36次正式拟合与36个收缩派生版本均成功，9个基线重现精确匹配保存森林。新分裂准则提高根节点退市标准化信号份额，并将三年度完整平均预测器的空间单元数减少8.8%、11.0%、8.1%。然而预测增益不一致：2023年12个月退市AP由0.199升至0.514，而2024年由0.718降至0.573；2024年24个月AUC由0.770降至0.714，Brier由0.3396改善至0.3196，Top20仍捕获8/20个退市事件。单独支持保护完全消除单公司退市叶，但年度、时点及恢复结果呈现混合变化，未观察到稳定联合收益。

12个月合并退市AUC/AP从0.645/0.247变为新准则0.661/0.273，24个月从0.730/0.682变为0.722/0.660；对应主要差值的条件bootstrap区间跨越0。全部模型的最大显式重构误差为{fv['max_explicit_sum_error']:.3e}，概率及保存重载检查通过。因此新增分裂与支持约束能兼容完整内生解释，但不能据本轮结果认定其改善了稀少退市事件的时间外预测或彻底解决概率低估。保留原参考模型。
''')
    # Export standalone scientific plots; all inputs are aggregate public results.
    plt.rcParams.update({'font.family':'Noto Sans CJK SC','axes.unicode_minus':False,'font.size':10})
    colors=['#2455a4','#de7a16','#36947a','#8b4ba8']
    fig,axs=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    for row,h in enumerate([12,24]):
        scopes=['2023','2024','2025','pooled'] if h==12 else ['2023','2024','pooled']
        for col,k in enumerate(['auc','ap','brier']):
            ax=axs[row,col]
            for a,color in zip(ARMS,colors):
                sub=m[(m.cause==1)&(m.months==h)&(m.arm==a)].set_index('scope')
                ax.plot(range(len(scopes)),[sub.loc[s,k] for s in scopes],marker='o',lw=1.6,color=color,label=LABEL[a])
            ax.set_xticks(range(len(scopes)),[s if s!='pooled' else '合并' for s in scopes]);ax.set_title(f'{h}个月 · '+{'auc':'AUC ↑','ap':'AP ↑','brier':'Brier ↓'}[k]);ax.grid(alpha=.2)
    handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4)
    fig.suptitle('先最终退市：三种子平均预测，有限匹配消融\n24个月不评价2025队列；合并指标从预测重新计算',fontsize=14)
    for ext in ['png','pdf']:fig.savefig(PUBLIC/f'PREDICTION_COMPARISON.{ext}',dpi=180)
    plt.close(fig)
    # Reproducibility receipt, never contains company records or fitted parameters.
    stages={s:load(RUN/f'audit/gate_{s}.json') for s in ['pilot','formal','ensemble']}
    environment=dict(python=platform.python_version(),platform=platform.platform(),packages={p:importlib.metadata.version(p) for p in ['numpy','scipy','pandas','scikit-learn','pytest','matplotlib']},
        node_interpreters={'oliver':'/home/oliver/.venvs/xai-baselines-20261027/bin/python','lemon':'/home/lemon/.venvs/xai-baselines-py313-20261027/bin/python'},
        lemon_verified_versions={'python':'3.13.13','numpy':'2.5.4','pandas':'3.1.0','scipy':'1.18.1','scikit-learn':'1.9.1'})
    generated_json(PUBLIC/'ENVIRONMENT.json',environment)
    for tag in ['frozen','analysis']:
        src=Path(f'/home/oliver/xai_cause_support_tests_{tag}.log');shutil.copy2(src,PUBLIC/f'TESTS_{tag}.log')
    failures=[dict(stage='run_directory_preflight',type='blocked_directory_creation',action='terminated only three own preflight processes; used new isolated nested directory; no fit'),
        dict(stage='data_audit',type='stale_metadata_date_assertion',action='reconciled authoritative feature clocks and exactX; original data unchanged'),
        dict(stage='read_only_diagnostic',type='metadata_attribute_name',action='used actual previous_component_available field'),
        dict(stage='function_aggregation',type='registry_not_yet_ready',action='waited for all ensemble outputs then repeated read-only aggregation; zero extra fits'),
        dict(stage='report_revision',type='exclusive_JSON_write_and_stale_plot_hash',action='archive prior derived JSON and regenerate report hashes after plot changes; no model or metric change')]
    generated_json(RUN/'audit/preflight_events.json',failures)
    pointer=load(RUN/'audit/current_pointer.json');assert sha(Path(pointer['path']))==pointer['sha256']
    receipt=dict(status='COMPLETE_EXPLORATORY',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),private_run=str(RUN),
        branch='research/xai-rsf-cause-support-v1',source_commit='51e58f011dae19caba932659d98209a8658a0158',
        fit_code_commit=load(RUN/'audit/code_snapshot.json')['commit'],ensemble_code_commit=load(RUN/'audit/analysis_code_snapshot.json')['commit'],
        frozen_protocol_sha256=sha(PUBLIC/'frozen_protocol.json'),formal_raw_forests=36,baseline_reproduction_fits=9,new_design_fits=27,
        pilot_fits=2,total_raw_fits=38,formal_shrunk_derivatives=36,mean_predictor_compilations=24,formal_failed=0,fit_retries=0,
        stages=stages,source_inputs_sha256=load(RUN/'audit/source_hashes.json'),input_manifest_sha256=sha(RUN/'audit/immutable_input_hashes.json'),
        data_status='PROVISIONAL inherited original-vintage limitations; field-level clocks verified; unchanged conservativeC83',
        unsupported_metrics=['2025_H24_both_causes','2023_H12_recovery_AUC_AP_no_events'],
        formal_validation=fv,metric_validation=mv,full_tests='244 passed, one preexisting EBM visualization warning',
        old_reference_preserved=pointer,preflight_events=failures,formal_jobs_used_both_nodes=True,
        decision='KEEP_CURRENT_REFERENCE; no consistent independent predictive gain; no further training',
        public_results_sha256={p.name:sha(p) for p in PUBLIC.iterdir() if p.suffix in ['csv','.csv','.png','.pdf']})
    generated_json(PUBLIC/'execution_receipt.json',receipt);generated_json(RUN/'audit/execution_receipt.json',receipt)
    print('Aggregate delivery complete',PUBLIC)

if __name__=='__main__':main()
