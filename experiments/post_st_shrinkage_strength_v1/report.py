"""Aggregate-only Chinese report and publication figures from completed results."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import *

def table(frame):
    def value(v):
        if pd.isna(v):return 'NA'
        if isinstance(v,(float,np.floating)):return f'{v:.4f}'
        return str(v)
    lines=['| '+' | '.join(map(str,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
    lines += ['| '+' | '.join(value(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)]
    return '\n'.join(lines)

def main():
    m=pd.read_csv(D/'METRICS_ENSEMBLE.csv');grid=m[m.strength!='historical_selected'].copy();grid['strength']=grid.strength.astype(int)
    target=grid[(grid.cause==1)&grid.supported]
    wide=[]
    for s in STRENGTHS:
        row={'强度':s}
        for h,ys in [(12,YEARS),(24,YEARS[:2])]:
            for y in ys:
                r=target[(target.strength==s)&(target.scope==str(y))&(target.months==h)].iloc[0]
                row[f'{y}/{h}月 AUC']=r.auc
        wide.append(row)
    wide=pd.DataFrame(wide);wide.to_csv(D/'AUC_COMPARISON.csv',index=False)
    comparison=target[target.strength.isin([0,5,10,20,100,200])].copy()
    comparison['Top20捕获']=comparison.top20_captured.astype(int).astype(str)+'/'+comparison.events.astype(int).astype(str)
    comparison=comparison.rename(columns=dict(strength='强度',scope='年度',months='月数',auc='AUC',ap='AP',ipcw_brier='Brier',observed_expected='O/E'))
    comparison=comparison[['强度','年度','月数','AUC','AP','Brier','Top20捕获','O/E']]
    hist=pd.read_csv(D/'HISTORICAL_SELECTED_STRENGTHS.csv')
    hist=hist.rename(columns=dict(outer_cutoff='预测年度',selected_strength='历史选择强度',historical_termination_events='历史12月退市事件',
        historical_companies='验证公司',within_year_auc='历史年内AUC',ap='历史合并AP',ipcw_brier='历史Brier'))
    hist=hist.drop(columns=['reason'])
    selected=m[(m.strength=='historical_selected')&(m.cause==1)&m.supported]
    reference=m[(m.strength=='10')&(m.cause==1)&m.supported]
    change=selected.merge(reference,on=['scope','months'],suffixes=('_selected','_reference'))
    summary=[]
    for _,r in change.iterrows():
        summary.append({'年度':r.scope,'月数':r.months,'原AUC':r.auc_reference,'历史选择后AUC':r.auc_selected,
            '原AP':r.ap_reference,'历史选择后AP':r.ap_selected,'Brier变化':r.ipcw_brier_selected-r.ipcw_brier_reference,
            '原捕获':int(r.top20_captured_reference),'选择后捕获':int(r.top20_captured_selected)})
    numerical=pd.read_csv(D/'NUMERICAL_COMPLEXITY.csv');errs=numerical[['full_reconstruction','explicit_component_sum','function_reload','forest_reload']].max()
    ensemble=numerical[(numerical.member=='ensemble')&(numerical.year>=2023)]
    # Fixed structure and representation counts across the entire tested grid.
    for y,g in ensemble.groupby('year'):
        for c in ['main_functions','interaction_functions','spatial_table_cells','scalar_table_cells']:
            assert g[c].nunique()==1
    complexity=ensemble[ensemble.strength==10][['year','main_functions','interaction_functions','spatial_table_cells','scalar_table_cells']].sort_values('year')
    complexity.columns=['年份','主效应函数','二元函数','空间单元','时间与原因标量单元']
    dif=pd.read_csv(D/'PAIRED_DIFFERENCES_95.csv')
    ci=dif[(dif.scope=='pooled')&(dif.cause==1)&(dif.strength.isin(['0','20','100','200','historical_selected']))&
        dif.metric.isin(['within_year_auc','ap','ipcw_brier','top20_captured'])][['strength','months','metric','difference','lower95','upper95','undefined']]
    text='''# 父节点收缩强度实验结果（探索性）

**本轮没有找到可以可靠取代当前10的统一最优值。** 十个固定强度均完成：0、1、2、5、10、20、30、50、100、200。以已暴露年度的12个月年内配对加权AUC看，10为本轮网格最高（0.8377）；取消收缩的合并AP、Brier略好；20在2024年筛查略好，但2025年捕获减少。按截止日前历史验证真正选出的强度为200、200、100，后续12个月结果反而下降。该发现支持保留原参考值作为当前工作选择，不证明10在总体或未来数据中最优。未替换模型指针。

## 1. 比较对象与执行

固定C039：74项输入，17/503/22每种子32树，q≤2、深度4、每树均衡25变量、共享5切点、退市支持保护、全部2012年起历史与24个日历月随访上限。仅改变直接父节点AJ收缩强度。当前2023—2025九个raw森林复用；新拟合2020—2022历史验证森林9个；派生180个单种子版本并合并为60个96树预测器。没有重长当前模型的树，没有拟合校准器。

公式为 w=m/(m+lambda)，m为独立袋内公司数；父节点使用尚未收缩的直接父节点AJ。同一叶节点的权重用于两个原因及全部时间。lambda=0与原raw模型一致。lambda越大，叶估计越接近父节点，不保证所有公司的风险升高或降低。

协议先于新参数结果提交。历史选择规则代码先提交，随后将选择结果单独提交锁定，再读取外层评价标签。历史数据的74项选择与C039种子仍继承此前开发，因此这只是固定模型族条件下的前向选择，不能重新包装为完全未暴露的模型开发验证。

## 2. 全部强度的退市AUC

'''+table(wide)+'''

2025年24个月随访不充分，全部性能保持NA，预测和合法性核验保留。完整逐年、逐原因、逐时限和单种子结果在 METRICS_ALL.csv；先平均三个种子的企业预测后计算主指标。

## 3. 主要取舍

- 2024年12个月：10→20，AUC 0.8750→0.8772，AP 0.8596→0.8604，Top20 8/16→9/16；Brier 0.324317→0.324696，略差。
- 2025年12个月：10→20，AUC 0.7750→0.7708，AP 0.2703→0.2532，Top20 3/5→2/5。
- 2023年12个月：取消收缩的AUC/AP较好（0.8359/0.5458），但只有4例退市，Top20仍为2/4。不能将大幅AP变动解释为稳定增量。
- 全部年度12个月：10的年内AUC最高，但取消收缩的pooled AUC/AP/Brier略好；年内排序和跨年排序并非同一个目标。0→10的年内AUC为0.8341→0.8377，pooled AP为0.2933→0.2883，Brier为0.169880→0.170182。
- 24个月：200的年内AUC为0.7779，10为0.7741；但合并AP 0.6873→0.6801，Brier 0.311606→0.314644，捕获仍12/32。不能仅凭AUC宣布强收缩更好。
- 恢复原因存在不同权衡：12个月合并年内AUC在10→200时0.5571→0.5829、Brier 0.078665→0.077907。不能把退市一个指标的选择称为两原因全面改进；2023年12个月没有恢复事件，恢复AUC/AP保持NA。

'''+table(comparison)+'''

## 4. 截止日前的历史选择是否有效

'''+table(hist)+'''

主规则只最大化历史12个月退市年内配对加权AUC；并列依次用AP、Brier、距10的距离解决。24个月和恢复仅作次要检查，不另行择优。

2023/2024可用历史12个月退市事件都只有4例。2022验证队列没有该事件，对年内AUC贡献0对，因此2024年历史年内AUC没有新增有效事件配对；其公司仍参与合并AP、Brier和校准。2025仅8例。200触及预先网格上界，本轮不继续扩大搜索；不能据此认定真实最优值更大。

'''+table(pd.DataFrame(summary))+'''

历史选出的强收缩在后续12个月AUC/AP均下降；25例退市合计捕获13→12。固定预测条件下，合并年内AUC差-0.0168，95%区间[-0.0554,0.0194]；Brier差+0.000598，区间[0.000160,0.001084]。这说明本次历史选值未成功迁移；既可能受到稀少事件下选值波动影响，也可能涉及历史与后续队列差异，本实验没有将二者因果分离。不能根据区间包含0声称等效。

## 5. 概率水平与完整解释

收缩强度不能解决当前风险水平低估。2024年12个月退市O/E从10的9.8162变为20的9.8808、200的10.0499；即使0也为9.6865。父节点仍来自同一历史训练数据，向其靠拢不会自动获得新的年度发生水平。

全部240个单种子/合并表示通过全训练与评价行、全部原生时间及两个原因的重构、逐项求和、重载和概率检查。实际最大误差 '''+f'{errs.max():.3e}'+'''，远小于1e-10；森林重载误差为0。强度10的九个当前成员树及已保存预测精确复现。未裁剪概率。

'''+table(complexity)+'''

每一年度上述函数数量与单元数在全部十个强度下不变；单元含缺失状态，标量存储另外乘时间与原因维度。收缩改变幅度，并未简化结构。2023—2025、12个月退市、共同二元函数跨种子的平均相关性在0/10/200下约0.486/0.486/0.406；平均绝对函数差随收缩降低，但形状相关性没有同步改善。不能把幅度缩小等同于解释更稳定。详细幅度、函数稳定性与预测波动分别在 FUNCTION_AMPLITUDE.csv、FUNCTION_STABILITY.csv、PREDICTION_STABILITY.csv。

## 6. 区间、限制与执行问题

全部配对差值均为候选减10；AUC/AP/捕获的正值有利，Brier正值不利。使用原有1000次按年度分层的公司抽样、seed2026110201；所有候选共用索引。区间条件于固定预测，不包含训练集重抽、历史选值和此前种子筛选的不确定性；多个强度与指标的结果不能选择性作为确认性检验。以下为部分退市汇总，完整内容见 PAIRED_DIFFERENCES_95.csv。

'''+table(ci)+'''

2023—2025已经暴露；74项变量与C039三种子有既往选择偏差。本轮没有新证据证明历史财务版本完全PIT认证，沿用已冻结保守输入及其局限。任何候选均未更换参考模型。

代码暂存预检曾因Git对中文路径加引号而失败，当时没有提交Slurm作业或拟合森林；已改用NUL分隔解析并保留原失败记录。正式9个拟合和60个派生任务全部成功并完成哈希回传；无选择性重跑、删结果或放宽数值容差。
'''
    (D/'RESULTS_ZH.md').write_text(text)
    (D/'METHOD_DECISION_ZH.md').write_text('''# 方法判断

当前保留lambda=10，理由是本轮没有可迁移、兼顾两时限与两原因的替换证据，而非已证明10最优。lambda=0、20分别展现不同局部优势；历史选出的200/200/100未改善后续主时限。停止本轮有限搜索，不扩大网格、不换种子、不用未来结局重选每年强度。

收缩是叶概率估计的正则化模块，不是校准器或特征选择模块；其强度可以改变排序，但不能凭此恢复未知未来风险水平。精确二阶函数表示与合法CIF在整个网格中保留；结构复杂度不下降。论文可报告明确的收益与代价，不应宣称该模块必然改善所有预测、解释稳定性或绝对校准。
''')
    figdir=D/'figures';figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'Noto Sans CJK SC','axes.unicode_minus':False,'font.size':10})
    fig,axes=plt.subplots(2,3,figsize=(13.5,7),constrained_layout=True)
    colors={2023:'#2166ac',2024:'#d6604d',2025:'#21918c'}
    for row,h in enumerate([12,24]):
        for col,(metric,label) in enumerate([('auc','退市AUC ↑'),('ap','退市AP ↑'),('ipcw_brier','退市Brier ↓')]):
            ax=axes[row,col]
            for year in YEARS:
                z=target[(target.months==h)&(target.scope==str(year))].sort_values('strength')
                if z.empty:continue
                ax.plot(range(len(STRENGTHS)),z[metric],marker='o',markersize=4,label=str(year),color=colors[year])
            ax.axvline(STRENGTHS.index(10),color='#666',ls='--',lw=1,label='当前10' if row==0 and col==0 else None)
            ax.set_xticks(range(len(STRENGTHS)),STRENGTHS);ax.set_xlabel('收缩强度（等距显示候选值）');ax.set_ylabel(label)
            ax.set_title(f'{h}个月');ax.grid(alpha=.2)
    axes[0,0].legend(frameon=False);fig.suptitle('固定C039森林，仅改变父节点收缩强度｜探索性比较；2025年24个月NA',fontsize=14)
    for suffix in ['png','pdf']:fig.savefig(figdir/f'shrinkage_performance.{suffix}',dpi=180)
    plt.close(fig)

if __name__=='__main__':main()
