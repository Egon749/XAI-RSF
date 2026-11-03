"""Aggregate-only tables, figures and reproducible execution receipt."""
import datetime,platform,shutil,subprocess
import pandas as pd
import scipy,sklearn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import *

def table(df,cols):
    def fmt(v):
        if pd.isna(v):return 'NA'
        return f'{v:.4f}'if isinstance(v,(float,np.floating))else str(v)
    return '\n'.join(['| '+' | '.join(cols.values())+' |','|'+'|'.join(['---']*len(cols))+'|']+
        ['| '+' | '.join(map(fmt,r))+' |'for r in df[list(cols)].itertuples(index=False,name=None)])

def main():
    run=run_path();pub=REPO/'experiments'/NAME
    for name in ['metric','function','data']:assert load(run/f'reports/{name}_validation.json')['status']=='PASS'
    for file in (run/'reports').iterdir():
        assert file.suffix in ['.csv','.json'];assert not(pub/file.name).exists();shutil.copy2(file,pub/file.name)
    d=pd.read_csv(pub/'METRICS_ENSEMBLE.csv');d=d[d.version.eq('shrunk')].copy()
    d['trees_per_seed']=d.arm.str[1:].astype(int);d['combined_trees']=3*d.trees_per_seed
    d['capture']=d.apply(lambda r:f'{int(r.captured)}/{int(r.target_events)}'if pd.notna(r.captured)else'NA',axis=1)
    d=d.sort_values(['scope','months','cause','trees_per_seed']);term=d[d.cause.eq(1)]
    comp=pd.read_csv(pub/'NUMERICAL_COMPLEXITY.csv');comp=comp[comp.seed.eq('ensemble')&comp.version.eq('shrunk')].sort_values(['year','trees_per_seed'])
    co=comp.groupby(['trees_per_seed','combined_trees'])[['main_functions','interaction_functions','spatial_table_cells','scalar_table_cells','function_bytes']].mean().reset_index()
    ps=pd.read_csv(pub/'PREDICTION_STABILITY.csv');ps['trees_per_seed']=ps.arm.str[1:].astype(int)
    ps=ps[ps.version.eq('shrunk')&ps.months.eq(12)&ps.cause.eq(1)].groupby('trees_per_seed')[['mean_prediction_sd','max_prediction_sd']].mean().reset_index()
    st=pd.read_csv(pub/'FUNCTION_STABILITY.csv');st['trees_per_seed']=st.arm.str[1:].astype(int)
    st=st[st.version.eq('shrunk')&st.day.eq(365)&st.cause.eq(1)&st.order.eq(2)].groupby('trees_per_seed')[['jaccard','median_correlation','mean_rms_difference']].mean().reset_index()
    pooled=term[term.scope.eq('pooled')&term.months.eq(12)].copy()
    maxima=[]
    for (scope,h,cause),g in d[d.supported].groupby(['scope','months','cause']):
        for metric in ['auc','within_year_auc','ap','brier','captured']:
            if not g[metric].notna().any():continue
            best=g[metric].min()if metric=='brier'else g[metric].max()
            winners=g.loc[np.isclose(g[metric],best,rtol=0,atol=1e-12),'combined_trees'].tolist()
            maxima.append(dict(scope=scope,months=h,cause=cause,metric=metric,best_observed_value=best,combined_tree_counts=';'.join(map(str,winners)),status='EXPLORATORY_POINT_MAXIMUM_NOT_SELECTED_MODEL'))
    pd.DataFrame(maxima).to_csv(pub/'OBSERVED_METRIC_MAXIMA.csv',index=False)
    frontier=[]
    p=pooled.merge(co[['trees_per_seed','spatial_table_cells']],on='trees_per_seed')
    values=p[['within_year_auc','ap','captured','brier','spatial_table_cells']].to_numpy()*np.array([1,1,1,-1,-1])
    for i,row in enumerate(p.to_dict('records')):
        dominated=any(np.all(values[j]>=values[i])and np.any(values[j]>values[i])for j in range(len(p))if j!=i)
        frontier.append(dict(**row,dominated_point_estimates=dominated))
    pd.DataFrame(frontier).to_csv(pub/'EXPLORATORY_PARETO.csv',index=False)
    fn=load(pub/'function_validation.json');verdict=(pub/'INTERPRETATION_ZH.md').read_text()
    cols={'scope':'测试年份','combined_trees':'合并树数','auc':'AUC','ap':'AP','brier':'IPCW Brier','capture':'Top20捕获','risk_mean':'平均预测','observed':'观察率'}
    report=f'''# 树数有限匹配实验

{verdict}

## 实际设计与执行

当前保留模型为74变量＋退市支持保护＋q≤2＋深度4＋共享分位数切点＋均衡25变量池＋直接父节点AJ收缩10，训练随访仍为24个日历月。仅比较每种子8/16/32/64/128/256棵树，对应三种子合并24/48/96/192/384/768棵树。每个测试年度使用自己的历史训练快照，2023/24/25年分别309/345/390家，退市14/26/49例、恢复184/198/211例；原数据、结局、公司集合、预测起点及缺失处理均不变。

复用9个原32树森林，共288棵树，按原seed/tree_id增长32—255号树，共2016棵新树，分63个独立任务；另有2棵数值试运行。形成9个256树森林，较小模型取相同编号的前缀。全部原32树结构、raw AJ和收缩版本逐项相同。独立合成测试确认分块拼接、前缀与直接训练等价；没有把删除函数项当作改变树数。父节点收缩使用各叶自身的raw直接父节点和独立in-bag公司数，不重复施加收缩。3种子×6树数×3年度得到54个种子级预测器，18个三种子平均预测器；raw/收缩均保留，共144份完整表示。

结果为已暴露2023—2025队列上的探索性分析。训练未读取评价结局，观察结果后没有扩大树数网格、换种子、改变预算或替换主模型。这里的“指标最好”只表示固定网格中的点估计最大/最小，不能当作新的独立泛化证据，也不能按每个测试年份事后切换模型。

## 未来12个月先退市：主分析

{table(term[term.months.eq(12)&~term.scope.eq('pooled')],cols)}

先平均三个种子的逐公司预测，再计算指标。测试企业数36/45/53，Top20每年取ceil(0.2n)，即8/9/11家，合并时先逐年确定名额再相加。竞争事件作为对应原因已知非目标结局，未知随访不当作正常；支持端点的IPCW权重独立检查均为1。2025年24个月仅保留预测与数值验证，性能NA。

合并年度指标必须重新计算；年内AUC只计同年度事件—非事件配对，按配对数加权，区别于混合不同年份风险尺度的pooled AUC：

{table(pooled,{'trees_per_seed':'每种子树数','combined_trees':'合并树数','within_year_auc':'年内AUC','equal_year_auc':'年度等权AUC','auc':'pooled AUC','ap':'pooled AP','brier':'Brier','capture':'Top20捕获'})}

## 未来24个月先退市：次分析

{table(term[term.months.eq(24)&~term.scope.eq('2025')],cols)}

## 恢复原因与不利结果

{table(d[d.cause.eq(2)&d.months.eq(12)&~d.scope.eq('pooled')],cols)}

2023年一年内没有恢复事件，AUC/AP保留NA，Brier仍可计算。两年期恢复与raw版本、所有种子都保留在METRICS_ALL.csv，不能只报告退市或最有利树数。

## 解释复杂度与算法稳定性

下表为每年度合并预测器的复杂度，再在三个年度作描述性平均。非零函数按全部原生时间、两个原因幅度超过1e-14计数。空间单元包括每个变量独立的缺失状态；标量单元另乘时间点数和原因数。压缩文件大小不等于完整表格大小。

{table(co,{'trees_per_seed':'每种子树数','combined_trees':'合并树数','main_functions':'平均非零主效应','interaction_functions':'平均非零交互','spatial_table_cells':'平均空间单元','scalar_table_cells':'平均标量单元','function_bytes':'平均导出字节'})}

一年期退市预测的种子间标准差，先对同年度企业求平均，再平均三个年度：

{table(ps,{'trees_per_seed':'每种子树数','mean_prediction_sd':'平均预测标准差','max_prediction_sd':'各年度最大标准差的平均'})}

同年度训练X、同一边际乘积参考下比较三个种子的共同二元函数，以下平均三个种子配对和三个年度。集合Jaccard使用全部时间的非零函数集合；相关性仅针对共同函数在365天退市上的形状，不能单独证明整体交互稳定：

{table(st,{'trees_per_seed':'每种子树数','jaccard':'平均交互Jaccard','median_correlation':'共同函数相关中位数的平均','mean_rms_difference':'平均RMS差'})}

CONVERGENCE.csv保留相邻树数及相对256/种子的预测RMS、最大差、Spearman和Top20重合。树数增加可能降低随机波动，但新增变量对会增加完整解释规模；精确可分解不等于足够简单、统计作用准确或因果解释。LEAF_SUPPORT.csv记录真实叶支持，收缩没有增加真实事件数。

## 数值、配对区间与资源

全部{fn['representations']}份表示通过所有训练/评价样本、原生时间和两个原因的核验。完整重构最大误差{fn['max_full_reconstruction']:.3e}，独立逐项求和误差{fn['max_explicit_sum']:.3e}，路径变量数最多{fn['max_path_variables']}。保存重载、非负、单调和两原因和≤1均通过，没有概率裁剪。

使用1000次公司级配对bootstrap，按测试年度分层，seed2026110201，复用原抽样索引。所有树数与32/种子比较，未定义次数逐项记录。这些区间只包含固定模型条件下测试公司的波动，不包含训练样本、树数选择、此前特征选择等不确定性。多个指标与树数的区间均为描述性，不能择取有利区间宣称确认性显著。

利用现有两节点Slurm，生长任务独立单线程，管理器负责P/E核与亲和性。导出阶段因内存限制设置oliver最多12任务、lemon最多2任务，不绕过cgroups；内存等待和计算耗时分别记录。先验证最大768树导出，再完成剩余导出；这一步只检查资源和数值，没有按性能改变网格。完整任务数、主机分配、实际耗时、峰值内存和发布哈希见COMPUTE_COST.csv与execution_receipt.json。

保留现有conservative C原始披露版本、历史可得日及74变量名单选择时间的限制，不宣称完成新PIT认证。原CURRENT_POST_ST_MODEL.json和主模型目录不变；是否替换另行判断。失败记录和所有预设候选必须保留。
'''
    (pub/'RESULTS_ZH.md').write_text(report)
    (pub/'PAPER_METHODS_RESULTS_ZH.md').write_text('''# 论文方法与结果写作说明

树数敏感性分析固定74项输入及q≤2、最大深度4、共享分位数切点、均衡25变量池、退市事件支持保护和直接父节点AJ收缩强度10，只改变每个种子的树数。预先冻结8、16、32、64、128、256六个规模，三个种子预测等权平均，对应24至768棵树。树的公司bootstrap、特征覆盖和节点搜索随机流均以原始树编号确定，且不存在全局分区预算依赖，因此使用相同编号前缀进行配对比较。保留现有288棵树，仅拟合2016棵新增树；独立测试验证前缀与直接训练等价。

评价分别报告2023—2025年有随访支持的12个月及2023—2024年的24个月竞争结局。先对三个种子取平均，再计算年度AUC/AP、IPCW Brier、平均风险和年度Top20事件覆盖；同时报告年内分层AUC与重新计算的pooled指标。完整函数复杂度、种子间波动、共同函数一致性和数值重构分别评价。公司级固定预测配对bootstrap用于描述差值，不涵盖完整模型训练与树数选择不确定性。

结果段使用RESULTS_ZH.md中的实际值及INTERPRETATION_ZH.md的结论，必须同时报告预测、表示复杂度及不利结果。各年度队列已暴露，任何最佳点估计均为探索性，不能作为独立确认的最优树数。
''')
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':10})
    fig,axes=plt.subplots(2,3,figsize=(13,7));colors=['#4477aa','#228833','#ee7733']
    for y,color in zip(YEARS,colors):
        g=term[term.scope.eq(str(y))&term.months.eq(12)].sort_values('combined_trees')
        for ax,metric,title in zip(axes.flat[:4],['auc','ap','brier','coverage'],['AUC ↑','AP ↑','Brier ↓','Top20覆盖率 ↑']):
            ax.plot(g.combined_trees,g[metric],'-o',color=color,label=str(y));ax.set_title(title)
        c=comp[comp.year.eq(y)]
        axes[1,2].plot(c.combined_trees,c.spatial_table_cells,'-o',color=color,label=str(y))
    axes[1,1].plot(ps.trees_per_seed*3,ps.mean_prediction_sd,'-o',color='#884488');axes[1,1].set_title('平均种子间预测标准差 ↓')
    axes[1,2].set_title('完整解释空间单元')
    for ax in axes.flat:
        ax.set_xscale('log',base=2);ax.set_xticks([b*3 for b in COUNTS],[str(b*3)for b in COUNTS]);ax.grid(alpha=.2)
        ax.axvline(96,linestyle='--',color='gray',alpha=.5)
    axes[0,0].legend(title='评价年度');fig.suptitle('树数匹配实验｜未来12个月先退市｜虚线为当前96树版本',fontsize=14)
    fig.supxlabel('三种子合并后的总树数（每个种子树数 × 3）');fig.tight_layout(rect=[0,.02,1,.95])
    for ext in ['png','pdf']:fig.savefig(pub/f'TREE_COUNT_COMPARISON.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    log=run/'audit/final_pytest.log';assert '274 passed'in log.read_text()and'skipped'not in log.read_text();shutil.copy2(log,pub/'pytest_final.log')
    stages=['pilot','grow','assemble','export_pilot','inspect'];gates={s:load(run/f'audit/gate_{s}.json')for s in stages}
    receipt=dict(status='COMPLETED_EXPLORATORY',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        branch='research/xai-rsf-tree-count-v1',source_commit='69fa782d79a279a5b57e72d35f6f802b767838cb',
        training_code_commit=load(run/'audit/code_snapshot.json')['commit'],report_code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        formal_new_trees=2016,pilot_new_trees=2,reused_trees=288,independent_new_tree_batches=63,extended_forests=9,
        seed_models=54,ensemble_models=18,raw_and_shrunk_representations=144,automatic_adoption=False,core_changed=False,
        frozen_protocol_sha256=sha(pub/'frozen_protocol.json'),input_hash_manifest_sha256=sha(run/'audit/input_hashes.json'),
        source_hash_manifest_sha256=sha(run/'audit/source_hashes.json'),private_run=str(run),gates=gates,
        numerical_validation=fn,metric_validation=load(pub/'metric_validation.json'),data_validation=load(pub/'data_validation.json'),
        tests='274 passed; independent R checks included; 1 inherited EBM visualization warning; no skips',
        environment=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,matplotlib=matplotlib.__version__),
        failures=[],unsupported=['2025H24 performance','independent confirmatory optimal-tree selection'],
        current_pointer_sha256=sha(REPO/'CURRENT_POST_ST_MODEL.json'),
        public_output_hashes={p.name:sha(p)for p in pub.iterdir()if p.is_file()and p.name!='execution_receipt.json'})
    write(pub/'execution_receipt.json',receipt)

if __name__=='__main__':main()
