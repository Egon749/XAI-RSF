"""Deterministic aggregate tables and publication figures; no refitting."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import *

def table(frame, digits=4):
    def cell(v):
        if pd.isna(v):return 'NA'
        if isinstance(v,(float,np.floating)):return f'{v:.{digits}f}'
        return str(v)
    return '| '+' | '.join(map(str,frame.columns))+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+''.join('| '+' | '.join(cell(x) for x in row)+' |\n' for row in frame.itertuples(index=False,name=None))

def main():
    m=pd.read_csv(D/'METRICS_ENSEMBLE.csv');c=pd.read_csv(D/'NUMERICAL_COMPLEXITY.csv')
    cc=c[c.seed=='ensemble'];fit=pd.read_csv(D/'FIT_COST.csv');diff=pd.read_csv(D/'PAIRED_DIFFERENCES_95.csv')
    comp=cc.groupby('earliest_entry_year')[['main_functions','interaction_functions','spatial_table_cells','scalar_table_cells']].mean()
    stability=pd.read_csv(D/'FUNCTION_STABILITY.csv')
    st=stability[(stability.version=='shrunk')&(stability.cause==1)&(stability.order==2)&(stability.day==365)].groupby('earliest_entry_year')[['jaccard','mean_correlation']].mean()
    overview=[]
    for start in STARTS:
        row={'训练起始年':start}
        for h in (12,24):
            for y in YEARS:
                if y==2025 and h==24:continue
                a=m[(m.earliest_entry_year==start)&(m.scope==str(y))&(m.months==h)&(m.cause==1)].iloc[0]
                row[f'{y}/{h}月 AUC']=a.auc
        overview.append(row)
    pd.DataFrame(overview).to_csv(D/'AUC_OVERVIEW.csv',index=False)
    text='''# C039训练历史起始年消融结果（探索性）

**本轮没有发现缩短训练历史能同时改善退市排序、概率误差与解释结构。** 在固定C039种子与配置下，从2012开始训练的12个月退市AUC在2023、2024、2025三个评价年度均高于其余八个起始窗口。缩短历史可以提高平均退市预测、减轻概率低估并改善部分Brier，但往往牺牲AUC、AP或Top20捕获。恢复原因有部分改善，不能把退市排序下降推广成两个原因均无收益。主模型保持不变。

实际完成81次32树拟合，组成27个96树年度预测器；其中9次为2012全历史复现，72次为缩短历史的新拟合。没有森林拟合失败。三种子逐公司预测先平均再计算指标，不是平均三个AUC。2012复现与原C039树及全部原生时点预测完全相同。代码和协议冻结提交404ba86，收尾记录修复提交6a5431a；完整SHA见execution_receipt.json。

## 1. 口径与数据

“2016起始”意味着预测2023时用2016—2022入组公司及2023-01-01前已知随访，预测2024时用2016—2023，预测2025时用2016—2024。每家公司最长训练随访仍为24个日历月，未将未来结局回填历史。74项输入、q2、深度4、每树25项均衡覆盖、共享5分位切点、退市支持保护及直接父节点AJ收缩10均不变；种子17/503/22各32树。

三个原历史训练快照分别309、345、390家公司，含14、26、49例在各截止日前已观察的退市；缩短到2020起始后为131、167、212家，含12、24、47例。删去2012—2017入组队列不减少已观察退市，但会减少恢复、无事件与删失公司的风险集支持。详细的12/24月成熟与未成熟计数见TRAINING_SUPPORT.csv，不能把其多个重复训练窗口相加当成独立样本量。

评价集固定2023年36家、2024年45家、2025年53家。12月退市分别4、16、5例；24月2023/2024分别12、20例。2025年24月随访不足，性能NA。支持的评价时限结局已知，既有IPCW计算的权重均为1；训练中的右删失照旧保留。先恢复与先最终退市为竞争结局，不把先恢复后退市重新编码为目标。

## 2. 退市AUC全表

'''+table(pd.DataFrame(overview))+'''
2025年24月未参与排序。24月2023年2017起始的AUC为0.7813，略高于全历史0.7778，但AP从0.7002降至0.5830、捕获从5/12降至4/12，不能据此称整体更好。

## 3. 各年度退市全部主指标

'''
    for h in (12,24):
        for y in YEARS:
            if h==24 and y==2025:continue
            a=m[(m.cause==1)&(m.months==h)&(m.scope==str(y))].copy()
            a['捕获/事件']=a.top20_captured.astype(int).astype(str)+'/'+a.events.astype(int).astype(str)
            z=a[['earliest_entry_year','auc','ap','ipcw_brier','mean_predicted','observed_expected','捕获/事件']]
            text+=f'### {y}年，{h}个月\n\n'+table(z)+'\n'
    text+='''预测均值以概率为单位；O/E为观察率除以平均预测，>1表示总体低估，<1表示总体高估。Top20按各年ceil(0.2n)，即8、9、11家公司，而不是按结局数量配额。

## 4. 合并结果与概率水平

以下pooled AUC/AP直接在合并逐公司预测上重算；within_year_auc只计算同年事件—非事件配对，按各年配对数加权。Top20先逐年选定再相加，两者不能混同。

'''
    for k,name in [(1,'退市'),(2,'恢复')]:
        for h in (12,24):
            a=m[(m.cause==k)&(m.months==h)&(m.scope=='pooled')]
            text+=f'### {name}，{h}月\n\n'+table(a[['earliest_entry_year','within_year_auc','auc','ap','ipcw_brier','observed_expected','top20_captured']])+'\n'
    text+='''以2012→2020起始为例：12月退市年内加权AUC由0.8377降到0.7356，pooled AP由0.2883降到0.2160；Brier从0.1702改善至0.1661。24月退市年内AUC由0.7741降到0.6586，AP由0.6873降到0.5680，Brier从0.3116改善至0.2820。**概率误差改善与排序下降可以同时发生。**

2025年12月O/E从1.6776接近1.0145，但AUC从0.7750降至0.7000，Top20由3/5降至2/5；这只是平均风险水平更接近，不能声称每家公司概率已经校准。2024年12月O/E从9.8162降至6.4478，仍明显低估（平均预测5.51%，观察35.56%）。2023年12月仍为3.5449。缩短历史并没有解决所有年份的概率问题。

恢复原因的合并12月年内AUC从0.5571到0.6857，Brier从0.0787到0.0618；24月年内AUC从0.6739到0.7060，但AP从0.6357到0.6130，Top20从12/31到11/31。全部逐年度、逐种子、raw/shrunk指标均保存在METRICS_ALL.csv，不隐藏不利结果。2023年12月没有恢复事件，该原因AUC/AP不可估计，保持NA。

## 5. 固定预测配对区间

采用原1000次按年分层的公司重抽样，分析种子2026110201，所有臂共享抽样索引。下表为2020起始减2012起始，非自动采用门槛；AUC/AP正值有利，Brier负值有利。

'''
    dd=diff[(diff.earliest_entry_year==2020)&(diff.cause==1)&(diff.scope=='pooled')&diff.metric.isin(['within_year_auc','ap','ipcw_brier','top20_coverage'])]
    text+=table(dd[['months','metric','difference','lower95','upper95','undefined']])+'''
这些区间条件于已保存模型，不包含训练过程、此前挑选种子/输入或当前九个窗口的搜索不确定性。不能将探索性多重比较当成确认性显著性；2023/2025少事件的重抽样可能无事件，不可估计次数已逐项保留。

## 6. 解释复杂度与稳定性

以下为三个年度完整96树预测器的均值。函数数量是全空间、全原生时点绝对值超过1e-14的非零函数数；空间格数含缺失状态，标量数另乘原生时间点数与两个原因。实际导出采用无损因子化存储，文件字节数不等于逻辑网格数。

'''+table(comp.reset_index())+'''
下表交互集合Jaccard基于各对种子的结构函数键；函数相关性只对共同、非恒定项，在同一年度/窗口训练参考样本上计算。先平均三对种子，再平均三年度。不同窗口的参考分布发生改变，不能把这里解读为跨窗口相同函数的因果变化。

'''+table(st.reset_index())+'''
2012→2020，平均非零交互从125.7增至154.7；平均空间格数2668.3→2755.7。2023年的空间格数虽下降，但2024、2025上升。交互结构Jaccard约0.1430→0.0529，未发现稳定的解释简化或稳定性收益。候选特征覆盖预算未减少，删历史公司并不等于删变量或删交互。

全部模型仍为q≤2、深度≤4。所有训练/评价公司、全部原生时间点、两个原因上，完整函数重构、显式逐项求和、保存重载和概率合法性均通过；最大重构误差1.89e-15，最大逐项求和误差2.78e-15，远低于1e-10。三种子合并与逐公司预测平均一致。解释贡献仍是CIF加法贡献，不是风险率或因果效应。

## 7. 执行、失败与局限

两机Slurm作业2687完成26/26独立年度窗口任务（oliver20，lemon6），共78次森林拟合；加本地全预算试运行3次，共81次。Slurm执行约60.0秒，含排队/启动约62.5秒；这个数字不包含准备、试运行与分析，不能当整个项目用时。最大任务峰值RSS约374.5MiB。FIT_COST.csv保留81次实测拟合/收缩/回放耗时，COMPUTE_COST.csv保留任务总耗时；硬件和并发不同，不把单次耗时差解释为算法优劣。

试运行已完成所有计算、数值检查及首次summary写入后，收尾重复写summary被现有独占创建规则拒绝。保留原FAILED和完整工件，以只读校验及独立恢复回执确认完成，没有再次拟合。修复为一次最终写入后，后续26项全部成功。试运行完整墙钟和峰值内存未被保存，留空；每种子拟合/收缩/验证时间已保存。另有前置JSON列表/元组比较问题，规范化后确认配置没有差异，原暂存目录保留。不存在未完成的预设森林，2025年24月性能按协议不执行。

本实验同时改变历史样本量、恢复/删失构成、bootstrap抽样池、训练切点和函数参考分布，不能单独证明政策机制。C039本身来自已暴露测试集的种子组合选择，原全历史配置可能受益于这种选择，不能据本轮宣称2012是普适最优起始年。当前结论仅是：在这个固定C039配置、这些已暴露队列和有限窗口中，删除早期历史带来风险水平与排序之间的权衡，未提供替换当前主模型的充分依据。
'''
    (D/'RESULTS_ZH.md').write_text(text)
    (D/'PAPER_NOTE_ZH.md').write_text('''# 论文方法与结果草稿

为考察历史训练长度敏感性，我们固定74输入的C039森林（种子17、503、22，每种子32树），将最早入组年份依次设为2012至2020。每个外层年度2023—2025均仅使用当年1月1日前自己的训练快照，保留原24日历月随访上限、竞争结局和删失。除训练队列截取外，不改变候选覆盖、切点生成规则、退市支持保护、q2、树深或父节点收缩。训练分位数切点和函数参考分布由各窗口训练数据重新构造。主指标来自三种子逐公司CIF均值；12月主要、24月次要，2025年24月不评价。

共拟合81个森林、形成27个96树预测器。相对于2012起始，2020起始的退市12月年内配对加权AUC从0.8377降至0.7356，而Brier从0.1702降至0.1661；24月年内AUC从0.7741降至0.6586，而Brier从0.3116降至0.2820。缩短历史提高了平均退市预测并部分缓解总体低估，但没有同时保持排序质量。恢复原因出现不同方向的部分收益。完整二阶分解和竞争风险概率约束均保留，但交互数量和种子间结构重合未显示一致改善。

研究队列及C039种子组合此前已被用于分析，故本结果仅作为探索性机制与敏感性证据；配对区间条件于固定预测，不构成新独立验证，也不证明训练窗口越长越好或政策变化的因果效应。未更换主模型。
''')
    figdir=D/'figures';figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    colors={2023:'#3568a8',2024:'#c65b3b',2025:'#218570'}
    for i,h in enumerate((12,24)):
        for j,(metric,title) in enumerate([('auc','AUC'),('ap','Average precision'),('ipcw_brier','IPCW Brier (lower is better)')]):
            ax=axes[i,j]
            for year in YEARS:
                if h==24 and year==2025:continue
                a=m[(m.scope==str(year))&(m.months==h)&(m.cause==1)].sort_values('earliest_entry_year')
                ax.plot(a.earliest_entry_year,a[metric],'-o',ms=4,color=colors[year],label=str(year))
            ax.set_title(f'{h} months | {title}');ax.set_xlabel('Earliest training entry year');ax.grid(alpha=.2);ax.set_xticks(STARTS);ax.tick_params(axis='x',labelrotation=45)
            ax.legend(frameon=False)
    fig.suptitle('C039 fixed configuration: shortening training history\nExploratory; 2025 at 24 months is not evaluable',fontsize=14)
    for ext in ['png','pdf']:fig.savefig(figdir/f'PREDICTION_WINDOWS.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,3.9),layout='constrained')
    for ax,metric,title in zip(axes,['main_functions','interaction_functions','spatial_table_cells'],['Nonzero main functions','Nonzero pair functions','Logical spatial cells (incl. missing)']):
        for year in YEARS:
            a=cc[cc.year==year].sort_values('earliest_entry_year');ax.plot(a.earliest_entry_year,a[metric],'-o',ms=4,label=str(year),color=colors[year])
        ax.set_title(title);ax.set_xlabel('Earliest training entry year');ax.grid(alpha=.2);ax.set_xticks(STARTS);ax.tick_params(axis='x',labelrotation=45);ax.legend(frameon=False)
    fig.suptitle('Complete 96-tree predictor complexity | exact q2 representation')
    for ext in ['png','pdf']:fig.savefig(figdir/f'EXPLANATION_WINDOWS.{ext}',dpi=180)
    plt.close(fig)

if __name__=='__main__':main()
