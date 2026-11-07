"""Aggregate-only publication; no firm forecasts or fitted parameters exported."""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import *

def md_table(frame):
    def cell(v):
        if pd.isna(v):return 'NA'
        if isinstance(v,(float,np.floating)):return f'{v:.4f}'
        return str(v)
    return '| '+' | '.join(frame.columns)+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+''.join('| '+' | '.join(map(cell,row))+' |\n' for row in frame.itertuples(index=False,name=None))

def main():
    a=pd.read_csv(HERE/'MODEL_COMPARISON.csv');pair=pd.read_csv(HERE/'PAIRED_DIFFERENCES.csv');cost=pd.read_csv(HERE/'COMPUTE_AND_PARAMETERS.csv')
    # Context requested by "add a comparator": reuse old point estimates only,
    # after proving identical train/eval/label files. No old methods rerun/tuned.
    old=ROOT/'experiments/post_st_eswa_baselines_v1'
    hh=read(old/'DATA_ALIGNMENT_AUDIT.json')['hashes'];current=read(HERE/'SOURCE_HASHES.json')
    labels=read(HERE/'EVALUATION_HASHES.json')['labels']
    for y in YEARS:
        for role in ('train','eval'):assert hh[f'{y}/{role}']==current[f'{y}/{role}']
        assert hh[f'{y}/labels']==labels[f'{y}/labels']
    previous=pd.concat([pd.read_csv(old/'METRICS_BY_YEAR.csv'),pd.read_csv(old/'POOLED_METRICS.csv')])
    previous=previous[previous.model.isin(['PROBoost/IPCW','Stacking/IPCW'])].copy()
    previous['provenance']='previous ESWA frozen point estimates, not refitted here'
    fields=['scope','months','model','auc','within_year_auc','ap','top20_captured','target_events','provenance']
    fresh=a.copy();fresh['provenance']=np.where(fresh.model=='C061','archived C061 replay','new XGBoost benchmark')
    pd.concat([fresh[fields],previous[fields]]).to_csv(HERE/'ALL_COMPARATORS_CONTEXT.csv',index=False)
    src=pd.read_csv(ROOT/'experiments/post_st_seed_explanation_stability_v1/RECONSTRUCTION_DETAIL.csv')
    cx=src[(src.model=='C061')&(src.version=='shrunk')].sort_values('year').copy()
    cx['provenance']='Historical full reconstruction audit, current source hashes reverified; no new forest or function fit'
    cx.to_csv(HERE/'C061_EXPLANATION_AUDIT.csv',index=False)
    a[a.supported].loc[:,['scope','months','model','observed','mean_predicted','observed_expected','calibration_gap','ipcw_brier']].to_csv(HERE/'CALIBRATION.csv',index=False)
    keys=[('2023',12),('2024',12),('2025',12),('2023',24),('2024',24)]
    fig,axes=plt.subplots(1,3,figsize=(11,4.3),layout='constrained')
    colors={'C061':'#276A8A','XGBoost':'#C76C37'}
    for ax,metric,title in zip(axes,['auc','ap','ipcw_brier'],['AUC (higher is better)','AP (higher is better)','Brier (lower is better)']):
        vals={n:[float(a[(a.scope==y)&(a.months==h)&(a.model==n)][metric].iloc[0]) for y,h in keys] for n in colors}
        for i in range(len(keys)):ax.plot([vals['C061'][i],vals['XGBoost'][i]],[i,i],color='#CED4DA',lw=2,zorder=1)
        for n in colors:ax.scatter(vals[n],range(len(keys)),s=43,label=n,color=colors[n],zorder=2)
        ax.set_yticks(range(5),[f'{y} / {h} months' for y,h in keys]);ax.invert_yaxis();ax.set_title(title,fontsize=10)
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
        ax.set_xlim((0,1) if metric!='ipcw_brier' else (0,.4))
    axes[0].legend(loc='lower left',frameon=False);fig.suptitle('C061 vs XGBoost: same held-out companies; exploratory comparison',fontsize=12)
    (HERE/'figures').mkdir(exist_ok=True)
    for ext in ('png','pdf'):fig.savefig(HERE/'figures'/('performance_comparison.'+ext),dpi=210)
    plt.close(fig)
    table=a[a.supported].loc[:,['scope','months','model','auc','within_year_auc','ap','ipcw_brier','top20_captured','target_events']].copy()
    table.columns=['年度','月数','模型','AUC','年内配对AUC','AP','IPCW Brier','Top20捕获','目标事件']
    cal=a[a.supported].loc[:,['scope','months','model','mean_predicted','observed','observed_expected']].copy()
    cal.columns=['年度','月数','模型','平均预测','观察发生率','O/E']
    selected=cost[cost.seed==17][['year','months','max_depth','num_boost_round','lambda']]
    intervals=pair[(pair.scope=='pooled')&pair.metric.isin(['auc','within_year_auc','ap','ipcw_brier','top20_captured'])][['months','metric','difference','lower95','upper95','undefined']]
    text='''# C061 与 XGBoost 对比结果

本次实际新增了 XGBoost 对照。XGBoost 在2023/2024年12个月AUC略高，但2025年12个月以及两年24个月的AUC、AP和Top20捕获均低于C061。2024年12个月虽然XGBoost的AUC更高，AP和Top20捕获却更低。不能据此说任一模型全面胜出。两者仍低估退市发生率，XGBoost的O/E在五个可评价年度—时限组合均更高。所有逐年AUC和AP差值的95%条件区间都包含0；合并24个月年内AUC差值为负且区间不含0。这仍是已暴露队列上的探索性比较，并非新的确认性证据。C061及当前模型指针保持原样。

## 执行与可比性

- C061：74输入；17/155/8679各32棵、平均预测；q≤2、深度4、共享分位数切点、均衡25变量、退市支持保护、直接父节点AJ收缩10。重读保存预测并核对源哈希及旧表，零次XAI拟合。
- XGBoost：普通`binary:logistic`梯度提升树。每个年度与时限独立训练；三种子平均预测后评价。输入、公司、历史截止日、结局定义和评价口径一致。每个任务18个候选配置、2个前向历史折、3个种子：实际648次内部拟合及18次最终拟合，六个批处理任务全部成功。与森林的树数没有算法预算等价性。
- 74个原始输入保留NaN，由XGBoost学习缺失分支；没有填补、特征筛选、过采样或额外类别权重。IPCW处理提前删失，竞争恢复作为“时限内未先退市”的已知负例，不等于正常企业。提前删失未填为负例。
- 训练公司数依次309/345/390，评价公司数36/45/53。两个时限共享相同年度历史公司，但可用于监督损失的记录数不同。各截止点的12个月训练退市5/9/23例，24个月14/26/49例。原有财务版本和部分审计历史可得日限制继续保留；未重新声称完整PIT认证。
- 协议与实现首先提交于 `3178e02cd897f24411de6cd46da8f9d9a06b393f`，之后才拟合。Slurm2701申请两节点；6项任务实际均由oliver完成。lemon环境已验证，但本批任务短，调度器没有把任务交给它；没有为制造双机使用记录而重复计算。端到端约6.58秒，实际任务1.49–3.17秒，时间不能与不同历史条件下的C061训练耗时直接比较。

## 预测结果

下面均为先平均三个种子的公司预测，再计算指标。Top20按年度ceil(0.2n)，三个年度名额8/9/11；合并时先按年选定再相加。2025年24个月已生成私有预测，但所有性能为NA。

'''+md_table(table)+'''
12个月合并直接AUC为C061 0.6789、XGBoost 0.7989；年内配对加权AUC却为0.8185、0.8017。这两个计算对象不同：前者包含跨年公司配对，后者只比较同一年。XGBoost的跨年排序更好，不能写成年内区分能力整体提高。合并AP从0.2692到0.4300，但按年度确定Top20的总捕获从12/25到11/25。24个月总捕获从14/32到10/32。

## 固定预测条件下的不确定性

下表均为“XGBoost减C061”。AUC/AP/捕获正值有利于XGBoost；Brier负值有利于XGBoost。原始1000次按年分层公司重抽样逐项复现。2023年12个月15次、2025年12个月5次重抽样没有目标事件，AUC/AP差值不可定义，保留NA并计数；其他记录见完整CSV。

'''+md_table(intervals)+'''
年度差值详见PAIRED_DIFFERENCES.csv。多重探索性比较没有进行确认性显著性筛选。区间不包含训练重抽样、历史参数选择及C061种子组合选择的不确定性，不能用“区间包含0”宣称等效。

## 风险水平

'''+md_table(cal)+'''
O/E大于1表示平均风险偏低。XGBoost在五个可评价年度—时限上均未缓解低估。Brier与O/E衡量不同内容：例如2024年24个月，XGBoost的Brier较低，但平均风险仍低估、AUC/AP/捕获较差，不能合并为“全面校准改善”。没有用测试结果拟合校准器、放大预测或裁剪概率。

## 历史选参和支持限制

'''+md_table(selected)+'''
18个候选为深度1/2/3 ×32/96/192轮×L2强度1/10，学习率0.05，行/列抽样0.8，其他参数见冻结协议。每个候选先对三个种子的历史验证预测平均，以年度AP等权均值选参，Brier与更简单结构作固定次序的平局判定。

真正历史前向切分中，内层训练结局也重新截断到该内层截止日，未把2023年初已知的事件回填给2020年的训练。2023年12个月两个历史验证年度共有4例退市，2024年只有2021年度的1例对AP有信息，2025年只有2023年度4例对AP有信息；零事件年度仍参与可估计Brier。各外层24个月只有较早的一个验证年度获得时限支持，较近年度仍记录为支持不足。即使搜索范围有限，这些稀疏事件也不能保证参数选择稳定。TUNING_ALL/TRAINING_LOG保留全部候选和折结果，不只保留最优配置。

## 解释、概率与完整性

C061的完整二阶CIF表示没有改变。历史全样本/原生时间核验的重构误差约1.1e-15至1.9e-15；54/59/60个主效应函数，131/133/132个二元函数，2625/2819/2809个空间单元（2023/24/25）。这里明确是复用且核对哈希的历史验证，不冒充新拟合或新函数编译。

XGBoost仅输出两个独立时限的退市分类概率，没有恢复CIF或完整生存曲线，不继承C061的概率尺度精确二阶分解。即使若干最终模型只选择深度1，logistic链接和三种子概率平均也不构成同一种低阶CIF函数表示。可以对XGBoost另做SHAP，但本轮没有把它称作原生完整函数表。

XGBoost所有保存重载预测逐值相等，得分有限且在[0,1]内。独立时限模型出现12个月分数高于24个月的公司，数量2023/24/25分别2/3/4；这表明它不能直接解释为时间单调的CIF曲线，不做事后修补。2025这项仅是预测结构检查，并非24个月性能。

公开文件仅含汇总；公司标识、原始特征、内层折、模型、预测均保留在新的私有运行目录。没有更换模型指针、覆盖旧模型或合并分支。未发生训练或数值验证失败；最初在存储根目录创建新运行目录时发生文件租约等待，终止后改在既有私有存储父目录创建唯一子目录，彼时尚无任何拟合。

## 论文可以写什么

可以写：在相同时间外公司队列上，加入历史前向选择的IPCW-XGBoost固定时限对照；短期排序存在年度间取舍，C061的24个月排序与筛查点估计更好，同时保留完整二阶CIF表示。应同时报告XGBoost的2023/2024年12个月AUC与12个月合并AP优势以及C061尚未解决的概率低估。

不可以写：C061全面优于XGBoost、已经证明泛化优势、所有指标显著更好、XGBoost不可解释，或把固定时限分类器与竞争风险森林称作同一种模型。C061组合在已暴露结果后被选中，未用与XGBoost相同的历史选择程序重新选择种子；这项非对称性必须披露。

官方实现依据：[XGBoost参数](https://xgboost.readthedocs.io/en/stable/parameter.html)、[原生缺失处理](https://xgboost.readthedocs.io/en/stable/faq.html)。完整提交及环境、输入输出哈希见execution_receipt.json；独立结果提交由其记录，回执自身位于随后的提交中。
'''
    text+='\n## 与原有对照模型放在一起\n\nALL_COMPARATORS_CONTEXT.csv另列原有PROBoost/IPCW和Chen Stacking/IPCW点估计，便于在原表中加入XGBoost并以C061为参照。三个年度的训练、测试X和标签文件哈希均与原实验一致。这两种既有方法本轮未重训、未改参，也未把其旧结果写成本轮新执行。其历史内部选择采用原冻结分层交叉验证，而本轮XGBoost采用前向验证，选择程序和搜索预算并不相同；因此属于预测器比较，不能单独归因于提升算法本身。原有极高的小样本年度点估计和训练扰动稳定性限制继续保留。\n'
    (HERE/'RESULTS_ZH.md').write_text(text)

if __name__=='__main__':main()
