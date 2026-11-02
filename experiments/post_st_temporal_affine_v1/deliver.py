"""Publish only aggregate tables, figures and reproducible methods."""
import shutil,subprocess,platform
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import *

PUB=REPO/'experiments'/NAME
LABEL={'baseline':'原森林','independent':'独立校准','temporal':'跨期收缩'}

def table(d):
    def fmt(x):
        if pd.isna(x):return 'NA'
        if isinstance(x,(float,np.floating)):return f'{x:.4f}'
        return str(x)
    return '| '+' | '.join(d.columns)+' |\n|'+'|'.join(['---']*len(d.columns))+'|\n'+'\n'.join('| '+' | '.join(fmt(x) for x in row)+' |' for row in d.itertuples(index=False,name=None))+'\n'

def figures(m,sd):
    plt.rcParams.update({'font.family':'Noto Sans CJK SC','axes.unicode_minus':False,'font.size':10,'pdf.fonttype':42})
    dst=PUB/'figures';dst.mkdir(exist_ok=True);labels=['2023 / 12月','2023 / 24月','2024 / 12月','2024 / 24月','2025 / 12月'];comb=[(2023,12),(2023,24),(2024,12),(2024,24),(2025,12)]
    colors=['#65758b','#e49b35','#167eaa'];x=np.arange(5);fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for c in [1,2]:
        for j,a in enumerate(ARMS):
            d=m[(m.cause==c)&(m.arm==a)].set_index(['scope','months'])
            risk=[d.loc[(str(y),h),'risk_mean']*100 for y,h in comb];brier=[d.loc[(str(y),h),'brier'] for y,h in comb]
            axes[c-1,0].bar(x+(j-1)*.23,risk,.23,color=colors[j],label=LABEL[a]);axes[c-1,1].plot(x,brier,'o-',color=colors[j],label=LABEL[a])
        actual=[d.loc[(str(y),h),'observed']*100 for y,h in comb];axes[c-1,0].plot(x,actual,'kD--',label='观察发生率')
        for j in [0,1]:
            ax=axes[c-1,j];ax.set_xticks(x,labels,rotation=15);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
            ax.set_title(('先最终退市' if c==1 else '先恢复')+('：平均预测风险与观察结果' if j==0 else '：Brier（越低越好）'))
        axes[c-1,0].set_ylabel('发生概率（%）');axes[c-1,1].set_ylabel('Brier')
    axes[0,0].legend(ncol=2,fontsize=9);axes[0,1].legend(ncol=3,fontsize=9)
    fig.suptitle('固定原森林的历史仿射校准：风险水平改善不等于排序改善',fontsize=16)
    for ext in ['png','pdf']:fig.savefig(dst/f'CALIBRATION_COMPARISON.{ext}',dpi=170)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained');allcomb=comb+[(2025,24)];xx=np.arange(6)
    for c,ax in zip([1,2],axes):
        for j,a in enumerate(['independent','temporal']):
            d=sd[(sd.cause==c)&(sd.arm==a)].set_index(['year','months']);v=[d.loc[(y,h),'prediction_sd_mean']*100 for y,h in allcomb]
            ax.bar(xx+(j-.5)*.34,v,.34,color=colors[j+1],label=LABEL[a])
        ax.set_xticks(xx,labels+['2025 / 24月*'],rotation=30,ha='right');ax.set_title('先最终退市' if c==1 else '先恢复');ax.set_ylabel('平均逐公司预测标准差（百分点）');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend();ax.spines[['top','right']].set_visible(False)
    fig.suptitle('50次公司重抽样：每次重估完整校准链，原森林固定',fontsize=15)
    fig.supxlabel('* 2025年24月只评估校准估计波动，不评价结局性能',fontsize=10)
    for ext in ['png','pdf']:fig.savefig(dst/f'CALIBRATION_STABILITY.{ext}',dpi=170)
    plt.close(fig)

def main():
    block_forest_fits();r=RUN/'reports';m=pd.read_csv(r/'METRICS.csv',dtype={'scope':str});sd=pd.read_csv(r/'CALIBRATION_ESTIMATE_STABILITY.csv')
    n=load(r/'numerical_validation.json');audit=load(r/'preservation_execution_audit.json');v=load(r/'metric_validation.json');unc=pd.read_csv(r/'PAIRED_BOOTSTRAP_95.csv',dtype={'scope':str})
    assert all(x['status']=='PASS' for x in [n,audit,v])
    for p in r.iterdir():
        if p.suffix in ['.csv','.json'] and 'PRIVATE' not in p.name:shutil.copy2(p,PUB/p.name)
    figures(m,sd)
    annual=[]
    for c in [1,2]:
        for y,h in [(2023,12),(2023,24),(2024,12),(2024,24),(2025,12)]:
            z=m[(m.scope==str(y))&(m.months==h)&(m.cause==c)].set_index('arm');b=z.loc['baseline']
            annual.append({'原因':'退市' if c==1 else '恢复','年份':y,'月':h,'观察率%':100*b.observed,'原风险%':100*b.risk_mean,'独立%':100*z.loc['independent','risk_mean'],'跨期%':100*z.loc['temporal','risk_mean'],
                '原Brier':b.brier,'独立Brier':z.loc['independent','brier'],'跨期Brier':z.loc['temporal','brier'],'共同AUC':b.auc,'共同AP':b.ap,'共同Top20':f'{int(b.captured)}/{int(b.target_events)}'})
    pooled=m[m.scope=='pooled'][['months','cause','arm','companies','target_events','auc','ap','brier','risk_mean','observed','captured']].copy();pooled['arm']=pooled.arm.map(LABEL)
    pooled=pooled.rename(columns={'months':'月','cause':'原因1退市2恢复','arm':'方法','companies':'公司','target_events':'事件','risk_mean':'平均风险','observed':'观察率','captured':'捕获'})
    results=f'''# 历史仿射校准与跨期参数收缩：结果

保留原83变量、q2、深度4＋直接父节点AJ10参考模型；本轮没有新拟合森林。历史仿射校准改善了24个月的概率误差，但跨期收缩的独立增益有限且不一致，不能据此替换当前参考。年内AUC/AP/Top20完全保持，跨年合并退市排序反而下降。2025年24个月没有足够随访支持，性能NA。

## 实验实际完成情况

复用2019—2025七个历史截止点×三种子的21个森林，均验证输入、训练截止与模型哈希。正式18个点估计校准器（12个外层、6个初始化），另50次配对公司重抽样×两条完整时间链，共900个校准器。最终102条链全部成功，Slurm2617/2618使用oliver及lemon；0次新森林拟合。首轮1条链因数值证书失败，修复后统一复算，全部旧输出保留，详见SOLVER_REVISION_ZH.md与EXECUTION_LEDGER.csv。

三组为原预测、独立历史校准、向前一期参数收缩的历史校准；后两者除跨期惩罚外完全匹配。所有校准样本、形状函数及前一期参数均只用各截止点之前的信息，外层结局不参与拟合或选择。每年先平均三种子森林预测再校准。12月、24月是两套分别训练的全曲线映射，不能混称为一个联合校准预测器。

## 年度比较

表中概率为百分数；Brier越低越好。共同AUC/AP/Top20表示该年三种方法完全一致。Top20名额向上取整为8、9、11家公司；分母为该原因该窗口独立事件总数。2023年12月恢复没有事件，AUC/AP为NA，0/0不表示覆盖率0。

{table(pd.DataFrame(annual))}

2024年24月退市平均风险由11.58%提升至28.30%，观察率44.44%，低估有所缓解但仍约16.15个百分点；Brier从0.3396降到0.2473。相对于独立校准的0.2521，跨期收缩只额外降低0.00486。2023年24月跨期收缩Brier为0.2618，反而比独立校准0.2593高0.00250。

2024年12月成熟历史校准集只有1家退市，斜率固定1；跨期收缩平均风险仅从3.24%升到3.33%，仍远低于35.56%的观察率。此前数据中的有限信息无法可靠校正这次跃升。2025年12月退市风险5.85%→8.11%，接近观察9.43%，但独立校准8.60%的点估计更接近。不得把单个年度的贴近视为稳定泛化证明。

恢复方面，24月Brier两年均改善。2024/2025年12月恢复事件不足而固定斜率1，非负偏移约束不能向下平移；这两个年度恢复预测完全未变，仍存在高估。这是本次冻结校准族的实际限制。

## 跨年合并

12月合并2023—2025共134家公司；24月只合并2023/2024共81家公司。AUC/AP/Brier在合并逐公司预测上重新计算，Top20在各年选定名额后合计，不能把年度平均当pooled。

{table(pooled)}

退市12月合并AUC 0.6451→0.4807、AP 0.2469→0.1816；24月合并AUC 0.7296→0.6913、AP 0.6817→0.6623。正斜率仿射变换保证同年度排序，不能保证不同年度采用不同参数后的跨年排序。这里的跨年排序下降是真实代价，不能因Brier改善而省略。

## 新增跨期收缩是否更稳健

50次公司重抽样均重新估计完整校准链。退市平均逐公司预测标准差：2023年24月2.48→2.18个百分点；2024年24月5.14→5.40个百分点；2025年12月1.68→1.58个百分点。2024年12月也从0.60升至0.69个百分点。跨期收缩降低了若干时期的波动，但不普遍降低。

2024年24月原校准集有13家退市，但50次公司bootstrap中37次不足10个独立退市公司，斜率回退1；2025年24月也有17/50次回退。重复抽中的同一家企业不计为新增事件。斜率可估计与固定两种状态的切换、前期参数本身的不确定性，以及两原因概率约束中的偏移—斜率权衡，均能使增加平滑惩罚后总波动上升。不能把跨期参数变化变小直接当作风险更准确。

## 不确定性

固定点校准器与森林，另做1000次按测试年分层的配对公司bootstrap。2024年24月退市Brier“跨期−原模型”为−0.09228，描述性95%区间[−0.14269,−0.04258]；“跨期−独立”为−0.00486，[−0.00830,−0.00158]。2023年24月“跨期−独立”为+0.00250，[+0.00063,+0.00440]。24月合并新增跨期增益为−0.00159，[−0.00354,+0.00028]，不能据此声称普遍独立收益。

这些区间仅反映固定模型条件下测试公司的变化，不含完整森林训练、校准参数估计、此前多轮设计选择的不确定性。校准参数的不确定性另由50次历史公司重抽样报告。零事件重抽样的AUC/AP保留NA并统计次数；不能选择性把多个比较中的区间解读为确认性发现。

## 精确解释和结论

18套完整预测表示均通过全部训练及评价公司、全部原生与校准时间点、两个原因的检查。完整显式函数求和最大误差{n['max_component_sum']:.3e}，保存重载一致，q≤2及概率合法性保留。结构没有新学习：2023/2024/2025分别62/65/64个主效应、144/146/147个二元函数，3146/3263/3194个空间单元。仿射校准统一缩放各原因的函数，并调整基准；不修复交互选择或相对幅度学习不足。

结论：**历史仿射校准可作为概率水平修正的候选；本轮跨期参数收缩尚未证明稳定、普遍优于逐期独立校准。保留原参考，不自动升级。** 所有已查看队列仅作探索性机制分析；原始财报版本证据的PROVISIONAL限制延续。2025年24月评价仍未执行，不补造结局或扩大调参。

完整指标、逐种子原预测、分箱、配对区间、重抽样指标、函数复杂度、计算成本及哈希均随本目录发布，真实公司预测、参数和函数表仍在私有服务器。
'''
    (PUB/'RESULTS_ZH.md').write_text(results)
    (PUB/'CALIBRATION_STABILITY_ZH.md').write_text('''# 校准估计稳定性

50次公司重抽样在入组年份内进行，同一入组年在全部截止点、期限及两校准臂使用同一组抽样。每次都重新求解从初始化到2025的完整时间链，只固定原森林及其样本外预测。故以下标准差属于校准估计不确定性，不是森林重训或全研究流程稳定性；也不能把初始化年与重叠校准窗口当独立样本。

offset_sd/slope_sd为校准参数标准差（未公布具体参数）；prediction_sd_mean为逐公司预测标准差的均值；risk_mean_sd为每次样本平均风险的标准差。两者不能互换。2025年24月可计算预测波动，不代表结局已有随访支持。

'''+table(sd)+'''

STABILITY_DIFFERENCES.csv按“跨期−独立”计算，负值表示波动降低。PARAMETER_STEP_CHANGES.csv还报告相邻期参数距离，以及在同一当前历史OOF风险上的映射变化RMS，避免把不同输入分布造成的风险变化归为参数变化。点估计的7次相邻期共同探针变化均下降，但2024年24月斜率向量变化反而增大；更关键的是重复抽样的预测稳定性并非普遍改善。平滑的时间序列不等同于较低估计方差或更好校准。

2024年24月退市有37/50次斜率固定，2025年24月有17/50次固定；这不是失败任务，而是预先冻结的独立事件支持保护。两臂门槛状态完全相同。恢复原因分别检查自己的事件支持，不以恢复事件替代退市事件。
''')
    nc=pd.read_csv(r/'NUMERICAL_COMPLEXITY.csv');cost=pd.read_csv(r/'COMPUTE_COST.csv')
    (PUB/'NUMERICAL_VALIDATION_ZH.md').write_text(f'''# 数值、函数与工件核验

918个最终校准求解全部通过；最大凸二次规划线性最优性gap {n['max_gap']:.3e}，原始约束最大违反量{n['max_primal_violation']:.3e}，验收阈值分别1e-8和1e-10。没有放宽阈值，无预测裁剪，SLSQP状态全部成功。最初一次失败及修订见SOLVER_REVISION_ZH.md。

原森林保持不变。18个完整预测表示包括3年×2期限×(原森林、独立校准、跨期校准)。每年验证训练+评价公司345/390/443条，全部原生与形状时间点以及两个原因。完整展开最大误差{n['max_full_reconstruction']:.3e}；逐个基线、主效应、二元函数显式相加最大误差{n['max_component_sum']:.3e}；函数重载误差{n['max_function_reload']:.3e}，森林重载误差0。

全部检查样本CIF最小值0、最小时间增量0、同一时点两原因和最大{n['max_probability_sum']:.6f}。此外，每个校准器用终端叶凸包Minkowski和约束验证全输入概率上界；并非仅依赖这批样本的概率检查。常数正斜率与非负不减偏移形状保留每条映射的单调性。

12/24月分别是独立的全曲线校准映射；训练及评价公司在两个报告端点的跨期限倒置为0。此经验检查不等于对所有可能输入证明两套映射相容，也不能把它们称为一个联合校准模型。需要连续展示时必须明确采用哪一个映射。

空间单元数包含各函数真实分区及单独缺失状态，未乘时间和原因。scalar_table_cells为乘上实际时间与两原因后的展开存储量，不能与空间单元混用。增加偏移曲线可能增加时间网格点，但不增加主效应、二元函数或空间分区；完整函数文件、规模和耗时逐项见NUMERICAL_COMPLEXITY.csv。

原数据与输入快照共核对66个直接源文件、35个继承源文件及511个冻结输入文件，全部哈希未变。当前指针SHA256 {REFERENCE_SHA}。crforest核心源码相对研究来源提交未变。复用21个历史森林的哈希、18个完整预测表示及918个校准器哈希分别独立记录。

最终两条点校准链包含函数编译验证，任务用时{audit['preservation']['point_solver_v2']['execution_seconds']:.2f}秒；100条重抽样链任务用时{audit['preservation']['bootstrap_solver_v2']['execution_seconds']:.2f}秒。单任务峰值RSS最高{cost.peak_rss_MiB.max():.1f}MiB。两节点合计成功102条链，oliver69条、lemon33条；均为CPU单线程独立链，无GPU计算、无单进程跨机训练。历史初次运行和数值复算时间全部保留，不能只将最终批次当全流程耗时。

最终252项测试通过；1条已有EBM缺失值可视化警告。正式结果使用solver_v2代码快照{audit['source_revision_commit']}，冻结协议SHA256 {audit['protocol_sha256']}。
''')
    (PUB/'METHOD_INNOVATION_DECISION_ZH.md').write_text('''# 研究判断

保持当前原森林参考，不自动替换。

1. **风险低估**：24个月退市概率明显改善，但2024仍低于观察发生率约16个百分点；12个月2024几乎未修复。不能说风险低估问题已经解决。
2. **新增跨期收缩**：部分年度降低校准估计波动、部分年度增大；24个月2023比独立校准差、2024更好。合并Brier独立增量的描述性区间跨0，尚无普遍稳健优势证据。
3. **排序代价**：正斜率保留每年度排序及Top20，但不同年度使用不同映射，跨年退市AUC/AP下降。不能笼统宣称仿射校准“不改变AUC”。
4. **解释**：精确q2函数闭合及全部函数表保留；只是更新同一个预测器的概率尺度，没有恢复被学弱的交互，没有提高变量或交互选择稳定性的证据。
5. **方法定位**：这是固定森林上的、受概率约束的历史概率更新模块；跨期正则化不是新的森林分裂或结构机制。可作为论文的校准扩展与负面边界实验，不能据此宣称内生可解释森林的核心创新已获独立确认。
6. **后续边界**：本轮不追加强度搜索。是否进一步研究联合期限校准、支持门槛的连续收缩或双向水平调整，应先另定协议；不能用这轮外层结果继续择优。最终替换需要新的充分随访时间外证据和研究者批准。
''')
    (PUB/'PAPER_METHODS_RESULTS_ZH.md').write_text('''# 可用于论文的方法与结果段落

## 方法

在不重新训练森林的前提下，我们将各年度三种子集成的原因别累积发生函数进行受约束仿射更新。对于原因k和单独冻结的目标期限H，更新为 F*_k(t|x)=c_k g_H(t)+b_k F_k(t|x)。g_H由相应历史训练集的合并事件AJ曲线归一化得到，非负、不减且终端等于1；b_k为不随时间变化的正数。校准数据采用预测截止前已完全成熟的最近两个入组年份，其风险来自在相应入组年前训练的固定森林，避免使用拟合内预测。损失为联合两原因平方概率误差，加向原预测收缩的固定惩罚。对照组逐期独立求解；跨期组额外加入对上一期已估计参数的惩罚，参数差异按历史风险尺度衡量，惩罚强度固定20/n，不以外层表现选择。任一原因少于10家独立事件公司时固定该原因斜率为1。

约束包括非负偏移、0.5至4之间的斜率及终端预测凸包上两原因和不超过1。结合非负不减形状与正斜率，这些约束保证每套映射为合法竞争风险CIF。仿射更新仅修改基线并统一缩放对应原因全部主效应与二元交互，因此完整函数分解精确保留。12月和24月各自独立校准，不解释为单个联合期限估计器。我们以50次按历史入组年分层的公司bootstrap重估完整校准链，评价校准估计稳定性；以另1000次按测试年度分层的配对公司bootstrap描述固定模型性能差值。评价包括2023—2025有随访支持的端点，2025年24月不计算性能。

## 结果

跨期校准使2023、2024年24个月退市Brier分别由0.2860、0.3396降至0.2618、0.2473，平均风险分别由6.47%、11.58%升至11.43%、28.30%，但仍低于33.33%、44.44%的观察率。与逐期独立校准相比，跨期项在2023年使Brier增加0.00250，在2024年降低0.00486；合并增量为−0.00159，描述性95%区间[−0.00354,0.00028]。历史重抽样中2023年24个月退市平均逐公司风险标准差由2.48降至2.18个百分点，2024年反由5.14升至5.40个百分点，显示跨期收缩的稳定性收益并不一致。

三组每年度AUC、AP与Top20捕获完全一致；跨年合并退市排序下降，12个月AUC由0.6451降至0.4807，24个月由0.7296降至0.6913。完整显式函数求和最大绝对误差2.665×10^-15，全部原始函数结构保持。结果支持历史仿射更新能够改善部分端点的概率水平，但不支持跨期参数收缩普遍优于独立校准，也未证明它能修复交互学习或跨年排序。所有外层年度此前均已查看，结论限于探索性机制分析。
''')
    receipt=dict(status='COMPLETE_EXPLORATORY',experiment=NAME,run=str(RUN),**{k:audit[k] for k in ['frozen_commit','source_revision_commit','protocol_sha256','current_pointer_sha256','accepted_chain_tasks','accepted_point_calibrators','accepted_resample_calibrators','new_forest_fits','superseded_successful_calibrator_solves','diagnostic_successful_calibrator_solves','diagnostic_failed_solves','actual_successful_calibrator_solves_including_revision_and_diagnostics','actual_failed_calibrator_solves_including_diagnostics','actual_chain_tasks','successful_chain_tasks','failed_chain_tasks','both_nodes_used']},
        point_job=2617,bootstrap_job=2618,preserved_initial_jobs=[2615,2616],tests_passed=252,existing_warnings=1,
        numerical_validation=n,metric_validation=v,research_decision='KEEP_ORIGINAL_REFERENCE; TEMPORAL_SHRINKAGE_GAIN_MIXED',
        evaluation_2025_24_months='NOT_ESTIMABLE_NO_FOLLOWUP_SUPPORT',new_models_adopted=0,
        artifacts_private=True,git_branch='research/xai-rsf-temporal-affine-v1',delivery_commit='See Git commit containing this receipt; exact remote SHA verified in final response and private sync receipt')
    (PUB/'execution_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    env={'local':{'python':platform.python_version(),'platform':platform.platform()}}
    import numpy,scipy,sklearn
    env['local'].update(numpy=numpy.__version__,pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__)
    env['lemon']=json.loads(subprocess.check_output(['ssh','lemon',"/home/lemon/.venvs/xai-baselines-py313-20261027/bin/python -c 'import json,platform,numpy,pandas,scipy,sklearn;print(json.dumps(dict(python=platform.python_version(),numpy=numpy.__version__,pandas=pandas.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__)))'"],text=True))
    (PUB/'ENVIRONMENT.json').write_text(json.dumps(env,indent=2)+'\n')
    print('Aggregate tables, Chinese reports and figures generated; no private predictions/parameters published.')

if __name__=='__main__':main()
