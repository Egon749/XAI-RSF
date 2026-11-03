"""Publish aggregate evidence only. Never load company predictions or fit models."""
import datetime,platform,subprocess,shutil
import pandas as pd
import scipy,sklearn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import *

LABEL={'cap12':'12个月','cap24':'24个月（原模型）','cap36':'36个月','uncapped':'不限（截止日前）'}

def table(df,columns):
    def fmt(x):
        if pd.isna(x):return 'NA'
        if isinstance(x,(float,np.floating)):return f'{x:.4f}'
        return str(x)
    rows=['| '+' | '.join(columns.values())+' |','|'+'|'.join(['---']*len(columns))+'|']
    for r in df[list(columns)].itertuples(index=False,name=None):rows.append('| '+' | '.join(map(fmt,r))+' |')
    return '\n'.join(rows)

def main():
    run=run_path();pub=REPO/'experiments'/NAME
    for n in ['metric','function','data']:assert load(run/f'reports/{n}_validation.json')['status']=='PASS'
    for p in sorted((run/'reports').glob('*')):
        assert p.suffix in ['.csv','.json'];dest=pub/p.name
        assert not dest.exists();shutil.copy2(p,dest)
    d=pd.read_csv(pub/'METRICS_ENSEMBLE.csv');d=d[d.version.eq('shrunk')].copy()
    d['训练上限']=d.arm.map(LABEL)
    d['捕获']=d.apply(lambda r:f'{int(r.captured)}/{int(r.target_events)}'if pd.notna(r.captured)else'NA',axis=1)
    complexity=pd.read_csv(pub/'NUMERICAL_COMPLEXITY.csv')
    complexity=complexity[(complexity.seed=='ensemble')&(complexity.version=='shrunk')].copy()
    complexity['训练上限']=complexity.arm.map(LABEL)
    complexity.sort_values(['year','arm'],key=lambda c:c.map({a:i for i,a in enumerate(ARMS)})if c.name=='arm'else c,inplace=True)
    stability=pd.read_csv(pub/'FUNCTION_STABILITY.csv')
    stable=stability[(stability.version=='shrunk')&(stability.cause==1)&(stability.day==365)].groupby(['arm','order'])[['jaccard','median_correlation','mean_rms_difference']].mean().reset_index()
    stable['训练上限']=stable.arm.map(LABEL)
    metrics_cols={'scope':'评价年度','训练上限':'训练上限','auc':'AUC','ap':'AP','brier':'IPCW Brier','捕获':'Top20捕获','risk_mean':'平均预测','observed':'观察发生率'}
    term12=d[(d.cause==1)&(d.months==12)]
    term24=d[(d.cause==1)&(d.months==24)&(~d.arm.eq('cap12'))&(~d.scope.eq('2025'))]
    pooled=term12[term12.scope.eq('pooled')]
    cohorts=pd.read_csv(pub/'COHORT_AUDIT.csv');cohorts['训练上限']=cohorts.arm.map(LABEL)
    fn=load(pub/'function_validation.json')
    report=f'''# 训练随访上限匹配实验：12、24、36个月与不限

结论：为了预测未来一年，并不需要把训练随访强制裁到一年。本轮12个月训练组没有整体优势；24个月组保留较好的年内排序和Top20捕获，36个月组在2024年及部分AP/Brier指标上改善，但在2023年有明显排序代价。“不限”也没有一致优于24或36个月。现有证据不支持把某个窗口宣布为普遍最优，原模型保持不变。

实际完成27个新增正式森林、复用9个原24个月森林，另完成3个4树数值试运行。所有正式结果均保留，无训练失败。共有96份单种子/合并、raw/收缩表示通过核验，最大完整重构误差{fn['max_reconstruction_error']:.3e}，独立逐项求和误差{fn['max_explicit_sum_error']:.3e}；路径最多2个不同变量，概率合法性和重载检查通过。

## 固定比较口径

使用当前“主模型”74输入＋退市支持保护版本：q2、深度4、每种子32树、种子42/101/202、每树均衡25变量池、共享分位数切点、直接raw父节点AJ收缩10。退市支持保护仍为子节点退市独立公司数为0或至少2，不是每个叶必须有2例退市。没有使用83变量旧参考指针代替本轮模型，也没有改写该指针。

分别在2023/2024/2025年1月1日前的信息上训练。四组训练公司与入组时点X完全相同，仅改变训练随访上限；不限组也在各自截止日前停止，不包含当年测试公司或截止日后的结局。所有公司从原入组公告后的下一交易日开盘前起算，先最终退市为1、先恢复为2、未发生首次结局按实际随访右删失。先恢复之后再退市不计为首次退市。

先按公司平均三个种子的CIF，再评价12个日历月（主）与24个日历月（次）。Top20逐年取ceil(0.2n)，2023/24/25分别核查8/9/11家；合并时先逐年选定再合计。竞争事件是对应原因的已知非目标结局；未完成随访者不当作负例。支持的本轮评价端点均已知，独立核验IPCW权重均为1，因此这些端点的IPCW Brier数值等于普通平方误差；这不表示训练中没有删失。

12个月训练组的24个月性能不报告：末端平台值不能冒充有支持的两年外推。2025年24个月全部性能NA，不用已观察到的部分事件代替完整验证。所有2023—2025结果已经暴露，本轮仅为探索性机制比较。

## 训练结局支持

{table(cohorts,{'year':'截止年','训练上限':'训练上限','training_companies':'公司','termination_events':'退市','recovery_events':'恢复','censored':'删失','termination_within12':'一年内退市','termination_after24':'两年后退市'})}

每年的公司数不随窗口变化。一年内退市分别始终只有5、9、23例；延长随访增加的是后期事件，不能把它们说成新增了一年内退市样本。36个月相对24个月新增10/11/11例首次退市，不限新增14/17/19例。训练集“截止年前所有年份”与“每家公司最长随访几个月”是两个不同维度；本轮只改变后者。

## 未来12个月退市：主要结果

{table(term12,metrics_cols)}

年度表现存在权衡：2023年12个月组AUC下降但AP上升，说明两指标并不等价，小样本只有4例退市，排名前段的变化可明显影响AP。2024年36个月组AUC/AP/Brier改善且仍捕获8/16；不限组捕获7/16；12个月组降为5/16。2025年36个月组AP升至0.3563，但Top20为2/5，低于24个月组3/5；不能把AP提高等同于固定核查预算下覆盖提高。

合并不同年份的原始风险会受年度风险尺度影响，必须同时报告年内分层AUC：

{table(pooled,{'训练上限':'训练上限','auc':'pooled AUC','within_year_auc':'年内配对加权AUC','equal_year_auc':'年度AUC等权均值','ap':'pooled AP','捕获':'年度名额合计捕获'})}

24个月组的年内AUC为0.7921、捕获13/25；36个月组为0.7740、12/25；不限为0.7596、11/25。不限的pooled AUC略高，不能因此说年内筛查更好。pooled AUC/AP/Brier均在合并预测上重算，没有平均年度AP。

## 未来24个月退市：次要结果

{table(term24,metrics_cols)}

36个月训练组对2024年两年期的改善较明显：AUC从0.6700到0.7440，AP从0.6704到0.7707，捕获从7/20到8/20；2023年AUC基本相近，捕获从4/12到5/12。该收益只属于本轮74输入＋支持保护的对应年度，不能套用到83变量旧模型或称为独立确认。

## 恢复原因：不能只保留有利的退市结果

{table(d[(d.cause==2)&(d.months==12)],metrics_cols)}

2023年一年内无恢复事件，AUC/AP不可估计并保留NA；Brier仍可计算。2025年恢复AUC在12个月训练组降至0.4733，原24个月组为0.7333。24个月恢复的合并AUC/AP/Brier依次为：

{table(d[(d.cause==2)&(d.months==24)&d.scope.eq('pooled')],metrics_cols)}

延长随访并未同时改善两个原因。36个月组两年期退市改善伴随恢复Brier由0.2390升至0.2482；不限为0.2441。完整年度、种子及raw结果见METRICS_ALL.csv。

## 风险水平仍低估

2024年一年期观察退市率35.56%，四组平均预测仅2.94%、3.24%、3.40%、3.38%；延长训练上限没有解决绝对水平低估。2023年观察11.11%，预测约1.93%—2.14%；2025年观察9.43%，预测约5.23%—6.41%。没有对概率作任何放大、截断或校准。

## 解释复杂度、稳定性及数值核验

下表为三种子合并96树的收缩版本，函数数按全部原生时间、两个原因中绝对幅度超过1e-14计为非零。空间单元包括数值区间和独立缺失状态；标量表格量另外乘原生时间数和原因数，不等于压缩文件实际大小。

{table(complexity,{'year':'年份','训练上限':'训练上限','main_functions':'非零主效应','interaction_functions':'非零交互','spatial_table_cells':'空间单元','scalar_table_cells':'标量单元','function_bytes':'压缩导出字节'})}

所有组保持完整q2函数表示，无高阶残差、代理解释或裁剪。较长随访增加原生事件时间点，因而标量表格和导出文件通常更大；空间分区数量并不随窗口单调变化。完整精确重构只证明解释忠实于模型，不证明作用被准确学到或具有因果性。

同年度、同一训练X参考分布下比较三种子的所有配对，以下为三年度/种子配对的描述性平均。Jaccard是全部时间的非零函数集合重合；相关性是共同函数在365天退市CIF上的形状相关中位数，不能把仅共同函数相关性提高解读为所有交互都稳定。

{table(stable,{'训练上限':'训练上限','order':'阶数','jaccard':'平均集合Jaccard','median_correlation':'平均共同函数相关中位数','mean_rms_difference':'平均RMS差'})}

36个月/不限的共同交互相关性有所改善，但交互集合Jaccard仍约0.10—0.12，证据不足以宣称交互选择已稳定。三个种子仅反映算法随机性。参考是训练边际乘积分布，相关变量的虚拟组合可能缺少真实支持；NUMERICAL_COMPLEXITY.csv保留低支持和零支持比例。

## 为什么只改变训练随访也会改变一年预测

bootstrap公司重复次数、同树可用变量集合及共享切点完全配对，但分裂分数使用训练窗口中的事件风险集，最小事件要求和退市支持保护也取决于可见事件。延长窗口改变结构学习信息，随后叶与父节点AJ在新分区内重新估计。因此它不只是把同一条预测曲线向后延长。本轮相对24个月组，12个月和不限组的288棵树均改变了至少一处分裂结构，36个月组286/288棵改变。这个诊断不能单独区分分裂分数、支持门槛和叶估计各自的贡献；本轮未额外改造机制或训练。

## 条件不确定性

使用原实验1000次同一公司抽样索引，按测试年分层，固定模型预测，seed2026110201。区间不包含训练样本变动、特征名单选择、窗口选择或多次研究决策的不确定性。未定义重抽样次数逐项保留；2023年恢复AUC/AP全部1000次不可估计。

例：2024年12个月36对24的AUC差+0.0108，95%描述区间[-0.0455,0.0733]；AP差+0.0711，区间[-0.0667,0.2296]。2024年24个月AUC差+0.0740，区间[0.0108,0.1443]；AP差+0.1003，区间[0.0046,0.2041]。这些是固定模型条件下、已暴露数据上的多项探索性比较，不能选择性作为确认性显著证据。完整结果见PAIRED_BOOTSTRAP_95.csv。

## 执行、失败与限制

Slurm2627完成3个试运行；2628完成27个正式拟合；2629完成18个新增合并预测器编译/全样本验证。前两批短任务实际均在oliver运行；合并验证13项在oliver、5项在lemon，两节点都实际完成工作。另复用6个原24个月合并raw/收缩预测器。耗时包括Slurm排队与结果发布，任务输出SHA256核验通过。新正式拟合约2.0—3.4秒/32树，单任务全过程约6.1—16.0秒，峰值RSS约242—322MiB；任务并行，不能把单任务时间相加当墙钟时间，也不把历史/新运行的耗时差解释为算法效率结论。

初始顶层NFS目录创建曾等待委托回收，在任何训练开始前终止该准备进程，改用已有私有研究路径的新唯一子目录。另一次只打印序列化结构字段的检查误把树列表当字典，修正检查后完成，无模型重跑。无训练、发布、数值验证失败，无遗漏种子或事后改变预算。测试与各阶段回执随交付保留。

继承conservative C字段可得日/披露版本以及历史事件链审核的局限，日期过滤并非独立PIT认证。2023—2025均已被查看，本轮没有新增独立测试；不更新主模型，不自动延长搜索。训练截止、所有输入/模型/函数哈希、额外拟合数与复用来源见execution_receipt.json及独立表格。
'''
    (pub/'RESULTS_ZH.md').write_text(report)
    method='''# 可用于论文的方法与结果草稿

为检验训练随访范围对短期竞争结局预测的影响，在保持74项基线变量、公司集合、入组时点、q≤2路径限制、最大深度4、共享切点、每树25项均衡变量覆盖、退市事件支持保护和直接父节点AJ收缩强度10不变的条件下，将每家训练企业的行政随访上限预设为12、24、36个日历月或不另设月数上限。各组均只使用预测年度1月1日前已经观察到的首次恢复、首次最终退市和删失信息；不设月数上限并不允许越过历史信息截止日。每年每组采用相同的三个随机种子及每种子32棵树，并配对公司bootstrap、特征池和切点字典。复用已验证的24个月组，另拟合27个森林。先平均三种子CIF，再评价年度与合并预测。主要端点为未来12个月，24个月仅作有支持的次要分析。

缩短训练随访至12个月没有提高整体短期预测。24个月组的12个月年内分层AUC为0.7921、Top20合计捕获13/25；12、36个月及不限组分别为0.7452、0.7740、0.7596与9/25、12/25、11/25。36个月组在2024年12个月AP和24个月AUC/AP上有所改善，但2023年12个月AUC下降，恢复原因未同步受益。所有组均保留完整二阶CIF函数表示，96份单种子及合并表示在全部训练与评价样本、原生时间及两个原因上核验，完整重构最大绝对误差2.78×10⁻¹⁵。较长训练随访没有消除退市概率低估。

本研究为已暴露2023—2025队列上的探索性匹配实验；条件bootstrap区间不涵盖模型训练、窗口选择和此前研究决策的不确定性。结果支持将训练随访上限视为影响结构学习的设计选择，不支持“越长越好”或“一年预测必须采用一年训练随访”的普遍命题。
'''
    (pub/'PAPER_METHODS_RESULTS_ZH.md').write_text(method)
    # Standalone scientific comparison chart. Values are cohort aggregates only.
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':10})
    fig,axs=plt.subplots(2,3,figsize=(13,7),sharex=True)
    colors=['#4477aa','#228833','#ee7733'];xs=np.arange(4)
    for col,(metric,title) in enumerate([('auc','AUC（越高越好）'),('ap','AP（越高越好）'),('brier','Brier（越低越好）')]):
        for year,color in zip(YEARS,colors):
            rows=term12[term12.scope.eq(str(year))].set_index('arm').loc[ARMS]
            axs[0,col].plot(xs,rows[metric],marker='o',label=str(year),color=color)
        axs[0,col].set_title(title);axs[0,col].grid(alpha=.2)
    for year,color in zip(YEARS,colors):
        rows=term12[term12.scope.eq(str(year))].set_index('arm').loc[ARMS]
        axs[1,0].plot(xs,rows.coverage*100,marker='o',label=str(year),color=color)
        axs[1,1].plot(xs,rows.risk_mean*100,marker='o',color=color)
        axs[1,1].axhline(rows.observed.iloc[0]*100,color=color,linestyle='--',alpha=.55)
        cc=complexity[complexity.year.eq(year)].set_index('arm').loc[ARMS]
        axs[1,2].plot(xs,cc.spatial_table_cells,marker='o',color=color)
    axs[1,0].set_title('Top20退市覆盖率（%）');axs[1,1].set_title('退市风险：预测实线、观察虚线（%）');axs[1,2].set_title('完整解释空间单元（96树）')
    for ax in axs.flat:
        ax.set_xticks(xs,['12个月','24个月','36个月','不限']);ax.grid(alpha=.2)
    axs[0,0].legend(title='评价年份');fig.suptitle('训练随访上限对照｜未来12个月先退市｜74变量＋支持保护＋父节点收缩',fontsize=14)
    fig.supxlabel('训练随访上限；所有组均严格截止于预测年度1月1日前',fontsize=11)
    fig.tight_layout(rect=[0,.03,1,.95])
    for ext in ['png','pdf']:fig.savefig(pub/f'FOLLOWUP_CAP_COMPARISON.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    log=run/'audit/final_pytest.log';assert '273 passed' in log.read_text() and 'skipped' not in log.read_text()
    shutil.copy2(log,pub/'pytest_final.log')
    jobs={s:load(run/f'audit/gate_{s}.json') for s in ['pilot','formal','ensemble']}
    jobs={s:{**r,'actual_nodes':dict(pd.Series([x['host']for x in load(run/f'audit/registry_{s}.json')if str(x.get('job'))==str(r['job'])]).value_counts())}for s,r in jobs.items()}
    for r in jobs.values():r['actual_nodes']={k:int(v)for k,v in r['actual_nodes'].items()}
    receipt=dict(status='COMPLETED_EXPLORATORY',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        branch='research/xai-rsf-followup-cap-v1',base_commit='7cc2d631ef4f3eda5af9246f6a7542640fb3cbc7',
        training_code_commit=load(run/'audit/code_snapshot.json')['commit'],
        analysis_snapshot_commit=load(run/'audit/analysis_code_snapshot.json')['commit'],
        report_code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        formal_new_forest_fits=27,pilot_forest_fits=3,pilot_trees=4,reused_formal_forests=9,
        retained_raw_and_shrunk=True,verified_representations=96,yearly_ensemble_trees=96,
        model_replaced=False,core_changed=False,scope='TRAINING follow-up only; all information before own cutoff; no test rows in fit',
        jobs=jobs,metric_validation=load(pub/'metric_validation.json'),function_validation=fn,
        data_validation=load(pub/'data_validation.json'),tests='273 passed, 1 existing EBM visualization warning, no skips, independent R included',
        environment=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,matplotlib=matplotlib.__version__),
        source_hash_manifest_sha256=sha(run/'audit/source_hashes.json'),immutable_training_hash_manifest_sha256=sha(run/'audit/immutable_input_hashes.json'),
        immutable_analysis_hash_manifest_sha256=sha(run/'audit/immutable_analysis_hashes.json'),
        private_run=str(run),current_pointer_sha256=sha(REPO/'CURRENT_POST_ST_MODEL.json'),
        failures=[dict(stage='initial_directory_preparation',action='own blocked NFS mkdir process terminated before fitting; unique child directory used'),
            dict(stage='read_only_schema_probe',error='AttributeError on serialized tree list .keys()',action='probe corrected; no model rerun')],
        formal_failures=0,publication_failures=0,unsupported=['all cap12 H24 performance','all2025 H24 performance'],
        public_output_hashes={p.name:sha(p)for p in pub.iterdir()if p.is_file()and p.name!='execution_receipt.json'})
    write(pub/'execution_receipt.json',receipt)

if __name__=='__main__':main()
