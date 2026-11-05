"""Complete grid, metric-specific extrema, uncertainty and exact-function tradeoffs."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import *
from experiments.post_st_training_start_v1.report import table

def counts_text(values):return '、'.join(map(str,sorted(map(int,values)))) if len(values) else 'NA'

def main():
    m=pd.read_csv(D/'METRICS_ENSEMBLE.csv');grid=m[m.pool_count!='historical_selected'].copy();grid.pool_count=grid.pool_count.astype(int)
    target=grid[(grid.cause==1)&grid.supported];primary=target[(target.scope=='pooled')&(target.months==12)].to_dict('records')
    best,_=select_count(primary)
    ranked=sorted(primary,key=lambda r:(-r['within_year_auc'],-r['ap'],r['ipcw_brier'],abs(r['pool_count']-25),r['pool_count']))
    pd.DataFrame(ranked).to_csv(D/'PRIMARY_GRID_RANKING.csv',index=False)
    top=next(r for r in primary if r['pool_count']==best);ref=next(r for r in primary if r['pool_count']==25)
    extrema=[]
    for (scope,h,k),g in grid[grid.supported].groupby(['scope','months','cause']):
        for metric,maximize in [('auc',True),('within_year_auc',True),('ap',True),('top20_captured',True),('ipcw_brier',False)]:
            value=g[metric].max() if maximize else g[metric].min();counts=g.loc[(g[metric]-value).abs()<=1e-14,'pool_count'].tolist() if pd.notna(value) else []
            extrema.append(dict(scope=scope,months=h,cause=k,metric=metric,best_counts=counts_text(counts),best_value=value,
                interpretation='EXPLORATORY_METRIC_SPECIFIC_EXTREMUM_NOT_ADOPTION'))
    ex=pd.DataFrame(extrema);ex.to_csv(D/'METRIC_SPECIFIC_BEST.csv',index=False)
    wide=[]
    for count in COUNTS:
        row={'每树候选数':count}
        for h,years in [(12,YEARS),(24,YEARS[:2])]:
            for year in years:row[f'{year}/{h}月 AUC']=target[(target.pool_count==count)&(target.scope==str(year))&(target.months==h)].iloc[0].auc
        wide.append(row)
    pd.DataFrame(wide).to_csv(D/'AUC_COMPARISON.csv',index=False)
    target.to_csv(D/'TERMINATION_COMPARISON.csv',index=False)
    reference=target[target.pool_count==25];bestrows=target[target.pool_count==best]
    paired=bestrows.merge(reference,on=['scope','months'],suffixes=('_best','_reference'));annual=[]
    for _,r in paired.iterrows():
        annual.append({'年度':r.scope,'月数':r.months,'参考25 AUC':r.auc_reference,'最佳候选 AUC':r.auc_best,
            '参考25 AP':r.ap_reference,'最佳候选 AP':r.ap_best,'Brier变化':r.ipcw_brier_best-r.ipcw_brier_reference,
            '参考捕获':f'{int(r.top20_captured_reference)}/{int(r.events_reference)}','最佳候选捕获':f'{int(r.top20_captured_best)}/{int(r.events_best)}'})
    difference=pd.read_csv(D/'PAIRED_DIFFERENCES_95.csv');ci=difference[(difference.pool_count==str(best))&(difference.scope=='pooled')&(difference.cause==1)&difference.metric.isin(['within_year_auc','auc','ap','ipcw_brier','top20_captured'])]
    ci12=ci[(ci.months==12)&(ci.metric=='within_year_auc')].iloc[0]
    num=pd.read_csv(D/'NUMERICAL_COMPLEXITY.csv');complexity=num[(num.member=='ensemble')&(num.year.isin(YEARS))].copy()
    complexity.to_csv(D/'EXPLANATION_COMPLEXITY.csv',index=False)
    avg=complexity.groupby('pool_count')[['main_functions','interaction_functions','spatial_table_cells','scalar_table_cells','function_bytes']].mean().reset_index()
    avg.to_csv(D/'COMPLEXITY_MEAN.csv',index=False)
    stab=pd.read_csv(D/'FUNCTION_STABILITY.csv');ss=stab[(stab.year.isin(YEARS))&(stab.order==2)&(stab.day==365)&(stab.cause==1)].groupby('pool_count')[['jaccard','mean_correlation','mean_abs_function_difference']].mean().reset_index()
    ss.to_csv(D/'INTERACTION_STABILITY_MEAN.csv',index=False)
    fit=[];coverage=[];feature_rows=[];pair_rows=[]
    for out in sorted((run()/'published').glob('*/task-*/output')):
        summary=load(out/'summary.json')
        for member in summary['members']:
            ident=dict(year=summary['year'],pool_count=summary['pool_count'],seed=member['seed'])
            fit.append(dict(**ident,**{k:v for k,v in member.items() if k not in ['seed','checks','raw_checks','reproduction','independent_checks']},**member['independent_checks']))
            f=pd.read_csv(out/str(member['seed'])/'feature_coverage_PRIVATE.csv');p=pd.read_csv(out/str(member['seed'])/'pair_coverage_PRIVATE.csv')
            coverage.append(dict(**ident,balanced_block_length=(74+summary['pool_count']-1)//summary['pool_count'],candidate_fraction=summary['pool_count']/74,eligible_feature_count=int((f.eligible_trees>0).sum()),split_feature_count=int((f.split_nodes>0).sum()),
                eligible_tree_count_min=int(f.eligible_trees.min()),eligible_tree_count_max=int(f.eligible_trees.max()),eligible_tree_count_sd=float(f.eligible_trees.std()),
                eligible_pair_count=int((p.eligible_trees>0).sum()),co_path_pair_count=int((p.co_path_trees>0).sum()),
                mean_pair_eligibility=float(p.eligible_trees.mean()),zero_pair_coverage_fraction=float((p.eligible_trees==0).mean())))
            for rr in f.to_dict('records'):feature_rows.append(dict(**ident,**rr))
            for rr in p.to_dict('records'):pair_rows.append(dict(**ident,**rr))
    pd.DataFrame(fit).to_csv(D/'FIT_COST_SUPPORT.csv',index=False)
    pd.DataFrame(coverage).to_csv(D/'COVERAGE_SUMMARY.csv',index=False)
    # Feature/pair counts have no observations, company identifiers or thresholds.
    pd.DataFrame(feature_rows).to_csv(D/'FEATURE_COVERAGE.csv',index=False)
    # Full pair-level counts remain private; public summaries suffice for this question.
    write(run()/'pair_coverage_aggregate_PRIVATE.json',pair_rows)
    hist=pd.read_csv(D/'HISTORICAL_SELECTED_COUNTS.csv')
    selected=m[(m.pool_count=='historical_selected')&(m.cause==1)&m.supported]
    hc=selected.merge(reference,on=['scope','months'],suffixes=('_selected','_reference'))
    histrows=[{'年度':r.scope,'月数':r.months,'参考AUC':r.auc_reference,'历史选值AUC':r.auc_selected,
        '参考AP':r.ap_reference,'历史选值AP':r.ap_selected,'Brier变化':r.ipcw_brier_selected-r.ipcw_brier_reference,
        '捕获变化':r.top20_captured_selected-r.top20_captured_reference} for _,r in hc.iterrows()]
    summary=[{'候选数':r['pool_count'],'年内AUC':r['within_year_auc'],'合并AUC':r['auc'],'合并AP':r['ap'],'Brier':r['ipcw_brier'],
        'Top20捕获':f"{int(r['top20_captured'])}/{int(r['events'])}"} for r in ranked]
    errors=num[['full_reconstruction','explicit_component_sum','function_reload','forest_reload']].max();err=errors.max()
    pairs=[]
    for y in YEARS:
        a=bestrows[(bestrows.scope==str(y))&(bestrows.months==12)].iloc[0];b=reference[(reference.scope==str(y))&(reference.months==12)].iloc[0]
        n=int(a.events*(a.companies-a.events));pairs.append(dict(year=y,company_pairs=n,auc_change=a.auc-b.auc,net_correct_pair_change=(a.auc-b.auc)*n))
    pd.DataFrame(pairs).to_csv(D/'PRIMARY_PAIR_COUNTS.csv',index=False)
    report=f'''# 每树候选变量数10—35：完整匹配实验（探索性）

按冻结的主排序规则，12个月退市年内配对加权AUC最高的候选数是 **{best}**：{top['within_year_auc']:.6f}，原25为{ref['within_year_auc']:.6f}；差值{ci12.difference:+.6f}，固定预测配对95%区间[{ci12.lower95:.6f},{ci12.upper95:.6f}]。这是已暴露队列上的描述性最高值，不是已确认的总体最优。合并AP从{ref['ap']:.6f}到{top['ap']:.6f}，Brier从{ref['ipcw_brier']:.6f}到{top['ipcw_brier']:.6f}，Top20捕获从{int(ref['top20_captured'])}/25到{int(top['top20_captured'])}/25。原模型保持25，没有自动替换。

## 设计、数据与执行

当前74输入C039固定17/503/22三个种子，每个32树，共96树；q≤2、深度4、退市支持保护、共享5分位切点、均衡变量调度、直接父节点AJ收缩强度10。所有年份使用自身Jan1之前训练快照、2012起入组历史和24日历月训练随访上限。仅ForestConfig.tree_feature_count从10到35逐一变化；上一轮收缩网格的18没有被采用。

实际重新拟合26×6×3=468个原始森林，其中2023—2025为234次，2020—2022历史前向验证为234次；包含pool25的18次原配置复现。每个原始森林派生一次AJ10版本，不递归重复收缩；形成156个96树合并预测器。2024/pool25三种子全预算试运行计入468，不另行重复。输入、配置、代码及输出均有哈希；历史模型与外层模型使用各自截止日信息。

公司bootstrap和重复次数按同年/种子/树逐一配对；切点字典和参考样本在同年相同。原均衡覆盖使用打乱分块，块长随候选数改变，所以各候选池不是嵌套集合。mtry名义值75不变，当路径尚未达到q上限时实际候选容量随池大小变化，达到两种变量后只复用路径变量。变化包括变量搭配机会、树间多样性和实际搜索量，不足以单独归因于某一个机制。阈值数上限5不表示所有节点都有5个有效切点。

主评价为12个月退市；24个月及恢复为次要。三个种子的逐公司预测先平均再计算，单种子另报。2025年24个月预测保留但性能NA。先恢复与先最终退市为竞争结局。可评价队列按原评价器检查均有完整支持、IPCW权重为1；训练右删失保留。

## 退市AUC全26值

'''+table(pd.DataFrame(wide))+'''

## 12个月主排序与其他指标

年内AUC仅比较同年事件—非事件配对并按配对数量加权；合并AUC/AP从合并预测重新计算，两种口径不能混同。Top20按各年ceil(0.2n)后合计。

'''+table(pd.DataFrame(summary))+f'''

## 主规则最高值{best}与原25逐年比较

'''+table(pd.DataFrame(annual))+'''

## 各指标最高值

下表保留全部并列值，不能用每个测试年的最高值拼接部署模型。

'''+table(ex[(ex.cause==1)&ex.metric.isin(['auc','within_year_auc','ap','ipcw_brier','top20_captured'])][['scope','months','metric','best_counts','best_value']])+'''

## 配对不确定性

所有对照共用原1000次按测试年分层的公司抽样，seed2026110201。下表为主规则最高值减25；AUC/AP/捕获正值有利，Brier正值不利。NA重抽样次数保留。

'''+table(ci[['months','metric','difference','lower95','upper95','undefined']])+'''

区间条件于固定模型预测，不包含训练样本、此前变量/种子筛选、当前26值搜索的选择不确定性，也未作多重比较校正。包含0不等于性能等效；不能选择性报告有利年度或指标。

## 历史信息选择与外层最高值分开

只使用各外层截止日前、整入组队列随访已成熟的历史前向预测，先按12月退市年内AUC，再按AP/Brier/距离25/较小值确定。没有可估计AUC时回退25。标签从该截止日自己的训练快照提取，不用后来版本回填。历史选择先锁定提交，再读本轮外层指标。

'''+table(hist)+'''

历史12月仅4/4/8个退市事件，不能据此声称选值稳定。2022无退市，对年内AUC贡献0对，但仍参与其他指标。历史选值后续结果：

'''+table(pd.DataFrame(histrows))+'''

C039种子和74变量继承既有回顾性选择，这种条件于固定配置的前向选值不消除此前信息暴露。外层网格最高值与历史选值可能不同。

## 完整解释、支持与候选覆盖

以下为2023—2025三个年度96树合并模型的平均复杂度；各年原值在EXPLANATION_COMPLEXITY.csv。主效应/交互数量为现有导出器按全空间、全部原生时间的非零判定，空间格含缺失状态，标量格另乘时间和原因。采用无损因子化存储，文件大小不等于逻辑网格数。

'''+table(avg)+'''

三个种子之间的二元函数集合Jaccard、共同非恒定函数相关性和幅度差如下。先平均三个种子对再平均年度；不把三个种子当三个独立验证队列。函数比较使用同年相同训练行与边际乘积参考；相关变量下可有贡献重新分配，不意味着因果作用变化。

'''+table(ss)+f'''

全部{len(num)}个raw、shrunk和合并表示检查全部训练与评价样本、所有原生时间及两个原因。最大重构/逐项求和/重载误差{err:.3e}，低于1e-10；q≤2，CIF非负、随时间不减、两原因之和≤1，不裁剪概率。所有25候选的原始森林与来源逐字节一致，2023—2025的收缩树与保存预测完全一致。

FIT_COST_SUPPORT.csv含独立分裂分数/AJ回放、叶独立公司支持、借入风险和计算时间；COVERAGE_SUMMARY.csv含候选变量/变量对覆盖。收缩不增加真实事件支持。不同机器并发下实测耗时不能直接当作算法固有效率比较。

## 范围与局限

2023—2025均已暴露，本轮26值是研究者指定的有限探索。所有有利、不利和不可估计结果保留，没有自动采用最高值或延伸网格。输入的保守披露版本/PIT限制继承原研究。数值精确说明解释忠实于所拟合森林，不证明学到真实财务机制、解释唯一或因果关系。完整原始数据、公司标识、模型、真实切点及逐公司预测留在私有目录，公开仅代码、配置、哈希和聚合结果。
'''
    (D/'RESULTS_ZH.md').write_text(report)
    (D/'METHOD_DECISION_ZH.md').write_text(f'''# 候选变量池实验判断

主规则最高值为{best}，12个月退市年内AUC{top['within_year_auc']:.6f}；原25为{ref['within_year_auc']:.6f}。差值95%区间[{ci12.lower95:.6f},{ci12.upper95:.6f}]。完整逐年/时限/AP/Brier/捕获和历史选值见RESULTS_ZH.md，不将单一最高值宣称为全面最优。

该参数控制森林的候选覆盖与搜索容量，不是内生精确解释成立的必要条件。全部取值仍q≤2，精确重构误差≤{err:.3e}。数量变化可能同时改变预测、函数数量及稳定性，不能把候选池更大称为自动更好，或把它与q阶数混同。

保留当前25/收缩10/C039参考。没有更换指针、再选种子或追加取值；这些暴露数据上的网格极值不能作为独立验证。历史事件很少，固定预测配对区间也不覆盖训练/选择不确定性。
''')
    figdir=D/'figures';figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'Noto Sans CJK SC','axes.unicode_minus':False,'font.size':10,'pdf.fonttype':42})
    colors=['#2166ac','#d6604d','#21918c']
    fig,axes=plt.subplots(2,3,figsize=(14,7.3),constrained_layout=True)
    for i,h in enumerate([12,24]):
        for j,(metric,label) in enumerate([('auc','退市AUC ↑'),('ap','退市AP ↑'),('ipcw_brier','退市Brier ↓')]):
            ax=axes[i,j]
            for y,color in zip(YEARS,colors):
                z=target[(target.scope==str(y))&(target.months==h)].sort_values('pool_count')
                if len(z):ax.plot(z.pool_count,z[metric],'-o',ms=3,color=color,label=str(y))
            ax.axvline(25,color='#666',ls='--',lw=1,label='参考25');ax.set_xticks(range(10,36,5));ax.set_xlabel('每树候选变量数');ax.set_ylabel(label);ax.set_title(f'{h}个月');ax.grid(alpha=.2)
    axes[0,2].legend(frameon=False,loc='center left');fig.suptitle('C039｜每树候选变量数10—35，步长1｜探索性；2025年24个月NA',fontsize=14)
    for ext in ['png','pdf']:fig.savefig(figdir/f'FEATURE_POOL_PERFORMANCE.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for ax,metric,label in zip(axes,['main_functions','interaction_functions','spatial_table_cells'],['非零主效应数','非零二元交互数','完整逻辑空间单元数（含缺失）']):
        for y,color in zip(YEARS,colors):
            z=complexity[complexity.year==y].sort_values('pool_count');ax.plot(z.pool_count,z[metric],'-o',ms=3,color=color,label=str(y))
        ax.axvline(25,color='#666',ls='--',lw=1);ax.set_xlabel('每树候选变量数');ax.set_ylabel(label);ax.grid(alpha=.2)
    axes[0].legend(frameon=False);fig.suptitle('完整96树预测器的解释复杂度｜全部保留q2精确分解')
    for ext in ['png','pdf']:fig.savefig(figdir/f'FEATURE_POOL_COMPLEXITY.{ext}',dpi=180)
    plt.close(fig)
    print(json.dumps(dict(descriptive_primary_best=best,within_year_auc=top['within_year_auc'],reference=ref['within_year_auc'],interval=[ci12.lower95,ci12.upper95],max_numerical_error=err),ensure_ascii=False))
if __name__=='__main__':main()
