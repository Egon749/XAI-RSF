"""All-grid aggregate tables, descriptive best values and Chinese report."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import *
from experiments.post_st_shrinkage_strength_v1.report import table

def strengths_text(values):
    values=sorted(map(int,values));runs=[]
    if not values:return 'NA'
    begin=end=values[0]
    for v in values[1:]:
        if v==end+1:end=v
        else:runs.append(str(begin) if begin==end else f'{begin}–{end}');begin=end=v
    runs.append(str(begin) if begin==end else f'{begin}–{end}')
    return '、'.join(runs)

def main():
    m=pd.read_csv(D/'METRICS_ENSEMBLE.csv');grid=m[m.strength!='historical_selected'].copy();grid.strength=grid.strength.astype(int)
    target=grid[(grid.cause==1)&grid.supported].copy()
    extrema=[]
    for (scope,h,k),g in grid[grid.supported].groupby(['scope','months','cause']):
        for metric,maximize in [('auc',True),('within_year_auc',True),('ap',True),('top20_captured',True),('ipcw_brier',False)]:
            v=g[metric].max() if maximize else g[metric].min()
            ss=g.loc[(g[metric]-v).abs()<=1e-14,'strength'].tolist() if pd.notna(v) else []
            extrema.append(dict(scope=scope,months=h,cause=k,metric=metric,best_strengths=strengths_text(ss),best_value=v,
                supported=pd.notna(v),interpretation='EXPLORATORY_METRIC_SPECIFIC_EXTREMUM_NOT_ADOPTION'))
    pd.DataFrame(extrema).to_csv(D/'METRIC_SPECIFIC_BEST.csv',index=False)
    primary=target[(target.scope=='pooled')&(target.months==12)].to_dict('records')
    best,_=select_strength(primary)
    pbest=next(r for r in primary if r['strength']==best);p10=next(r for r in primary if r['strength']==10);p0=next(r for r in primary if r['strength']==0)
    ranked=sorted(primary,key=lambda r:(-r['within_year_auc'],-r['ap'],r['ipcw_brier'],abs(r['strength']-10),r['strength']))
    pd.DataFrame(ranked).to_csv(D/'PRIMARY_GRID_RANKING.csv',index=False)
    wide=[]
    for s in STRENGTHS:
        r={'强度':s}
        for h,ys in [(12,YEARS),(24,YEARS[:2])]:
            for y in ys:r[f'{y}/{h}月 AUC']=float(target[(target.strength==s)&(target.scope==str(y))&(target.months==h)].iloc[0].auc)
        wide.append(r)
    pd.DataFrame(wide).to_csv(D/'AUC_COMPARISON.csv',index=False)
    alltable=target[['strength','scope','months','auc','within_year_auc','ap','ipcw_brier','top20_captured','events','observed_expected']].copy()
    alltable.to_csv(D/'TERMINATION_COMPARISON.csv',index=False)
    besttable=[]
    for h,ys in [(12,YEARS),(24,YEARS[:2])]:
        for y in ys:
            row={'年度':y,'月数':h}
            for metric,label in [('auc','AUC最高强度'),('ap','AP最高强度'),('ipcw_brier','Brier最低强度'),('top20_captured','捕获最高强度')]:
                r=next(r for r in extrema if r['scope']==str(y) and r['months']==h and r['cause']==1 and r['metric']==metric)
                row[label]=r['best_strengths']
                if metric=='auc':row['最高AUC']=r['best_value']
            besttable.append(row)
    hist=pd.read_csv(D/'HISTORICAL_SELECTED_STRENGTHS.csv')
    selected=m[(m.strength=='historical_selected')&(m.cause==1)&m.supported]
    ref=m[(m.strength=='10')&(m.cause==1)&m.supported]
    pair=selected.merge(ref,on=['scope','months'],suffixes=('_selected','_reference'))
    hcompare=[]
    for _,r in pair.iterrows():
        hcompare.append({'年度':r.scope,'月数':r.months,'参考AUC':r.auc_reference,'历史选择AUC':r.auc_selected,
            '参考AP':r.ap_reference,'历史选择AP':r.ap_selected,'Brier变化':r.ipcw_brier_selected-r.ipcw_brier_reference,
            '参考捕获':int(r.top20_captured_reference),'历史选择捕获':int(r.top20_captured_selected)})
    num=pd.read_csv(D/'NUMERICAL_COMPLEXITY.csv');err=num[['full_reconstruction','explicit_component_sum','function_reload','forest_reload']].max().max()
    cp=num[(num.member=='ensemble')&(num.year>=2023)].groupby('year')[['main_functions','interaction_functions','spatial_table_cells','scalar_table_cells']].agg(['min','max'])
    cp.columns=['_'.join(v) for v in cp.columns];cp=cp.reset_index();cp.to_csv(D/'COMPLEXITY_RANGE.csv',index=False)
    dif=pd.read_csv(D/'PAIRED_DIFFERENCES_95.csv');d=dif[(dif.strength==str(best))&(dif.scope=='pooled')&(dif.months==12)&(dif.cause==1)]
    ci=d[['metric','difference','lower95','upper95','undefined']]
    summary=[{'强度':r['strength'],'年内AUC':r['within_year_auc'],'合并AUC':r['auc'],'合并AP':r['ap'],'Brier':r['ipcw_brier'],
        'Top20捕获':f"{int(r['top20_captured'])}/{int(r['events'])}"} for r in ranked]
    pair_rows=[]
    for y in YEARS:
        z=target[(target.scope==str(y))&(target.months==12)]
        a=z[z.strength==best].iloc[0];b=z[z.strength==10].iloc[0]
        pairs=int(a.events*(a.companies-a.events))
        pair_rows.append(dict(year=y,companies=int(a.companies),events=int(a.events),event_nonevent_pairs=pairs,
            best_strength=best,auc_delta=float(a.auc-b.auc),net_correct_pair_change=float((a.auc-b.auc)*pairs)))
    pd.DataFrame(pair_rows).to_csv(D/'PRIMARY_PAIR_COUNTS.csv',index=False)
    text=f'''# 父节点收缩强度0—20细网格结果（探索性）

按预先固定的主排序规则——12个月退市年内配对加权AUC，随后AP/Brier/距10距离——本轮描述性第一名是 **{best}**，年内AUC **{pbest['within_year_auc']:.6f}**；参考10为{p10['within_year_auc']:.6f}，不收缩0为{p0['within_year_auc']:.6f}。这表示已暴露队列上的网格极值，不是已确认的总体最优。21个整数全部保留；不同年度和指标的最佳强度并不需要一致。没有更换参考模型。

**主要取舍：** 18相对10，在2024年12个月AUC由0.8750升至0.8815，AP由0.8596升至0.8689，Top20由8/16增至9/16；2023年AUC由0.8203降至0.8125、AP由0.3791降至0.3375；2025年AUC同为0.7750，AP由0.2703降至0.2578，捕获同为3/5。合计捕获13/25→14/25，但合并AP 0.288290→0.287546、Brier 0.170182→0.170316。24个月年内AUC 0.774112→0.771574、AP 0.687342→0.685290，捕获均12/32。0仍为两个时限合并AP和Brier的最好值。

18在12个月年内AUC上仅提高0.002404，配对95%区间[-0.011242,0.019380]，包含0。三年共有832个年内事件—非事件配对；2024净改善3对、2023净损失1对、2025为0，总共净改善2对（并列计半对），因此不能将这个细网格极值称为明显、稳定的性能突破。以12个月筛查排序为唯一主目标可以把18列为候选；以合并AP和概率误差为目标则0更有优势，当前10保持不变。

## 实验与执行口径

这轮由研究者在看过粗网格结果后指定0—20步长1，因此是自适应探索。先冻结细网格与选择规则，不能消除此前的结局暴露和C039种子/74变量筛选偏差。

固定C039：种子17/503/22，每个32树共96树；74输入、q≤2、深度4、退市支持保护、均衡25变量、共享5切点；每年自身Jan1之前训练快照、2012起入组历史与24个日历月训练随访上限不变。只修改 w=m/(m+lambda)，m为独立袋内公司数；两个原因/全部时间共用同一叶权重，采用原始直接父节点AJ，不递归收缩、不裁剪。

新增森林/校准器拟合均为0。复用18个raw森林，既包括之前拟合的2020—2022历史森林，也包括当前2023—2025森林；复用6个强度的36组输出，新增15个强度的90组输出。总计126个96树版本及378个单种子版本。复用证据经输入/工件/发布哈希核对，不冒充新计算。当前21个强度全体均有完整函数/概率/重载证据。

主时限12个月，24个月与恢复原因次要；2025年24个月性能全部NA。逐公司先平均三个种子预测再算主指标。Top20每年ceil(0.2n)再合计，合并AUC/AP重新计算；年内AUC只计同年度事件—非事件配对。所有可评价外层队列已达到相应结局支持，IPCW权重为1。

## 全部强度的退市AUC

'''+table(pd.DataFrame(wide))+'''

## 不同年度与指标的极值

下表完整保留并列值，不能理解为为每个测试年度分别选择部署参数。

'''+table(pd.DataFrame(besttable))+'''

## 12个月退市主排序（全部21个强度）

'''+table(pd.DataFrame(summary))+f'''

第一名{best}相对参考10的固定预测配对差值如下。AUC/AP/捕获正值有利，Brier正值不利。无事件重抽样的AUC/AP为NA，次数完整记录。

'''+table(ci)+'''

区间沿用1000次按测试年分层的公司抽样，seed2026110201，同一索引。它们条件于固定预测；没有包括训练抽样、筛选过程或从21个值中选最大值的不确定性。多个对照不能选择性解读为确认性显著结果，包含0也不代表等效。

## 截止日前历史验证的选择

'''+table(hist)+'''

历史主规则与上一轮一致：只用各截止日之前、整入组队列已成熟的历史12个月退市年内AUC；依次以合并AP、Brier、距10距离解决并列。2023/2024/2025仅有4/4/8例历史12个月退市。2022历史验证没有退市，对年内AUC贡献0对，但仍参与AP、Brier和校准。历史标签从该外层截止日自己的训练快照读取，不用后来快照回填。74变量及C039种子继承原模型开发，不能把这一条件选值称为完全独立的模型开发。

历史选择在读取本轮外层结果之前单独锁定并提交。其实际后续表现为：

'''+table(pd.DataFrame(hcompare))+'''

必须区分历史选值和回看2023—2025后的网格最高值；二者不能互换。

## 解释、概率与模型保留

'''+table(cp)+f'''

上表给出每年21个强度的函数/空间单元数范围。空间单元含缺失状态，标量单元另乘原生时间与原因维度。树结构与分区完全不变；贡献的大小、形状及基准可以改变。幅度变小或跨种子绝对差变小，不自动表示形状更稳定。FUNCTION_AMPLITUDE.csv、FUNCTION_STABILITY.csv、PREDICTION_STABILITY.csv保留完整证据。

全部{len(num)}个单种子/合并表示证据通过全部训练和评价行、全部原生时间点、两个原因的完整展开、逐项求和、保存重载和合法CIF检查；本轮集合最大误差{err:.3e}，容差1e-10，没有提高容差或裁剪。旧0/1/2/5/10/20的汇总指标须重新计算后逐值复现，10的当前模型指针保持原样。

## 完整交付与局限

METRICS_ALL.csv保留全部单种子/合并结果；TERMINATION_COMPARISON.csv为退市明细汇总；METRIC_SPECIFIC_BEST.csv含两原因各指标极值；PAIRED_DIFFERENCES_95.csv含全部差值和未定义抽样次数；原始公司信息、模型、切点和逐公司预测只在私有目录。

该实验只回答固定森林的叶估计收缩强度敏感性，没有重训树或重新选择变量。原保守输入及披露版本限制保持原样。2023—2025已暴露，历史事件稀少；不能断言网格第一名未来仍最佳，也不通过更细连续搜索或新增年份选择继续追高。不自动替换现有模型。
'''
    (D/'RESULTS_ZH.md').write_text(text)
    figdir=D/'figures';figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'Noto Sans CJK SC','axes.unicode_minus':False,'font.size':10})
    fig,axes=plt.subplots(2,3,figsize=(13.5,7),constrained_layout=True)
    for row,h in enumerate([12,24]):
        for col,(metric,label) in enumerate([('auc','退市AUC ↑'),('ap','退市AP ↑'),('ipcw_brier','退市Brier ↓')]):
            ax=axes[row,col]
            for year,color in zip(YEARS,['#2166ac','#d6604d','#21918c']):
                z=target[(target.months==h)&(target.scope==str(year))].sort_values('strength')
                if len(z):ax.plot(z.strength,z[metric],marker='o',ms=3,label=str(year),color=color)
            ax.axvline(10,color='#666',ls='--',lw=1,label='参考10');ax.set_xticks(range(0,21,2));ax.set_xlabel('父节点收缩强度')
            ax.set_ylabel(label);ax.set_title(f'{h}个月');ax.grid(alpha=.2)
    axes[0,2].legend(frameon=False,loc='center left');fig.suptitle('C039固定森林｜收缩强度0—20，步长1｜探索性结果；2025年24个月NA',fontsize=14)
    for suffix in ['png','pdf']:fig.savefig(figdir/f'integer_grid_performance.{suffix}',dpi=180)
    plt.close(fig)
    print(json.dumps(dict(descriptive_primary_best=best,primary_within_year_auc=pbest['within_year_auc'],reference10=p10['within_year_auc'],no_shrink0=p0['within_year_auc'])))

if __name__=='__main__':main()
