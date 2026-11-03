"""Publish aggregates only; preserve private exact models and all search points."""
import datetime,platform,subprocess,shutil
import pandas as pd
import scipy,sklearn,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.post_st_tree_count_v1.report import table
from .common import *

def main():
    run=run_path();pub=REPO/'experiments'/NAME
    for n in ['data','metric','function']:assert load(run/f'reports/{n}_validation.json')['status']=='PASS'
    for p in(run/'reports').iterdir():
        assert p.suffix in ['.csv','.json']and not(pub/p.name).exists();shutil.copy2(p,pub/p.name)
    d=pd.read_csv(pub/'METRICS_ENSEMBLE.csv');d=d[d.version.eq('shrunk')].copy()
    d['trees_per_seed']=d.arm.str[1:].astype(int);d['combined_trees']=d.trees_per_seed*3
    d['capture']=d.apply(lambda r:f'{int(r.captured)}/{int(r.target_events)}'if pd.notna(r.captured)else'NA',axis=1)
    d=d.sort_values(['scope','months','cause','trees_per_seed'])
    comp=pd.read_csv(pub/'NUMERICAL_COMPLEXITY.csv');comp=comp[comp.seed.eq('ensemble')&comp.version.eq('shrunk')]
    cx=comp.groupby(['trees_per_seed','combined_trees'])[['main_functions','interaction_functions','spatial_table_cells','function_bytes']].mean().reset_index()
    term=d[d.cause.eq(1)];pooled=term[term.scope.eq('pooled')&term.months.eq(12)]
    summary=pooled[['arm','trees_per_seed','combined_trees','within_year_auc','auc','ap','brier','captured','coverage']].copy()
    for y in YEARS:
        z=term[term.scope.eq(str(y))&term.months.eq(12)].set_index('arm')
        for k in ['auc','ap','brier','captured']:summary[f'{k}_{y}']=summary.arm.map(z[k])
    summary=summary.merge(cx,on=['trees_per_seed','combined_trees']).sort_values('combined_trees')
    summary.to_csv(pub/'LOCAL_SEARCH_SUMMARY.csv',index=False)
    ranks=pd.read_csv(pub/'EXPLORATORY_RANKING.csv');winner=ranks.iloc[0]
    maxima=[]
    for (scope,h,c),g in d[d.supported].groupby(['scope','months','cause']):
        for m in ['auc','within_year_auc','ap','brier','captured']:
            if not g[m].notna().any():continue
            value=g[m].min()if m=='brier'else g[m].max()
            maxima.append(dict(scope=scope,months=h,cause=c,metric=m,value=value,
                combined_trees=';'.join(map(str,g.loc[np.isclose(g[m],value,rtol=0,atol=1e-12),'combined_trees'])),
                status='EXPLORATORY_NOT_ADOPTED'))
    pd.DataFrame(maxima).to_csv(pub/'OBSERVED_METRIC_MAXIMA.csv',index=False)
    cols={'scope':'年度','combined_trees':'总树数','auc':'AUC','ap':'AP','brier':'Brier','capture':'Top20','risk_mean':'平均预测','observed':'观察率'}
    fn=load(pub/'function_validation.json')
    verdict=(pub/'INTERPRETATION_ZH.md').read_text()
    report=f'''# 80—120棵范围的局部树数搜索

{verdict}

## 冻结范围与实际执行

范围指三种子合并总树数。保持种子42/101/202各有相同树数，检查每种子27—40，对应81、84……120，共14档。80不是三等份整数树数，未改变权重或另选种子去凑80。仅改变树数，74输入、退市支持保护、q≤2、深度4、共享5切点、均衡25候选变量、直接父节点AJ收缩10、24个日历月训练随访及各年Jan1信息截止均不变。

复用此前9个256树森林的原编号前缀，零新森林拟合、零新树、零校准器拟合。为减少加载开销，事先保存前40棵，并逐节点确认raw/收缩值不变；未挑选“好树”，未再次施加收缩。旧96树的12份单种子/合并导出直接复用，新增156项独立导出，含一个120树资源试运行。126份种子模型、42份三种子合并预测器，raw/收缩共336份表示。

协议在计算新网格指标前提交。由于本轮是在查看上一轮24—768树结果后提出，且2023—2025已多次暴露，所有“最优”均指在已暴露评价样本上选出的探索性网格最高点，尽管森林拟合数据仍严格早于各年截止日。没有新的独立验证，不自动替换96树或按年度切换树数。

## 主分析：12个月先退市

{table(summary,{'trees_per_seed':'每种子树数','combined_trees':'总树数','auc_2023':'2023 AUC','auc_2024':'2024 AUC','auc_2025':'2025 AUC','within_year_auc':'年内AUC','auc':'pooled AUC','ap':'pooled AP','brier':'Brier','captured':'Top20捕获/25'})}

先平均三种子逐公司预测再计算指标。年内AUC只比较同一年度事件—非事件配对；pooled AUC/AP对合并预测重新计算。Top20按各年度ceil(0.2n)取8、9、11家公司后合计，不在混合年度上重新选前20%。未知结局不当负例，支持端点IPCW均为1且已独立检查。

预设点排序：12个月年内退市AUC优先，数值并列（1e-12）依次取Top20捕获更多、pooledAP更高、Brier更低、树数更少。对应首位是{int(winner.combined_trees)}棵；该排序仅报告探索性候选，其他年度、时限、原因及复杂度必须一并考察。完整排序在EXPLORATORY_RANKING.csv，所有指标各自极值在OBSERVED_METRIC_MAXIMA.csv。

逐年完整指标：

{table(term[term.months.eq(12)&~term.scope.eq('pooled')],cols)}

## 次分析：24个月退市和恢复原因

{table(term[term.months.eq(24)&~term.scope.eq('2025')],cols)}

2025年24个月仅保留预测、稳定性和数值检查，性能NA。恢复原因及raw版本全部保留在METRICS_ALL.csv；2023年12个月无恢复事件，AUC/AP为NA，不填0。不利结果不删除。

## 表示规模与稳定性

每年度三种子合并预测器的规模再跨三年度平均：

{table(cx,{'combined_trees':'总树数','main_functions':'平均非零主效应','interaction_functions':'平均非零二元项','spatial_table_cells':'平均空间单元','function_bytes':'平均导出字节'})}

非零按全原生时间、两个原因幅度大于1e-14判断；空间单元计入缺失状态，标量存储量另乘原因和时间维度。树数增加可覆盖更多变量对和分区，但每路径q≤2及深度约束不变。FUNCTION_STABILITY.csv在同年度训练边际乘积参考下比较种子间集合和共同函数形状；共同函数池随树数变化，不能把集合重合或相关性单独当成真实作用恢复。PREDICTION_STABILITY.csv报告三种子的预测波动，三个种子不是独立验证队列。CONVERGENCE.csv比较相邻档和120树；LEAF_SUPPORT.csv保留真实叶支持，收缩不增加真实事件。

## 数值、区间、资源与局限

336份表示全部通过训练/评价全部样本、原生时间、两个原因的完整核验。重构最大误差{fn['max_full_reconstruction']:.3e}，独立component逐项求和误差{fn['max_explicit_sum']:.3e}。保存重载一致，CIF非负、随时间不减、两原因和≤1；没有裁剪。原96树指标逐值复现，输入和公司对齐及源文件哈希均通过。

1000次公司为单位、按年度分层的配对bootstrap沿用seed2026110201及原抽样索引，所有新档减96树。区间条件于固定预测，未包含完整训练、此前变量选择、本次局部搜索或多重比较不确定性；选择出的最大值可能乐观，区间跨0也不是等效证明。原始披露版本及74变量选择来源限制继承，未宣称新PIT认证。

两节点Slurm仅承担独立导出与核验。原40树前缀已单独暂存，避免反复加载256树。内存门槛和并发冻结，实际节点、峰值内存、等待/计算时间、发布状态见COMPUTE_COST.csv和回执。CURRENT_POST_ST_MODEL.json、主模型文件夹及历史结果未改动。
'''
    (pub/'RESULTS_ZH.md').write_text(report)
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':10})
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for y,color in zip(YEARS,['#4477aa','#228833','#ee7733']):
        g=term[term.scope.eq(str(y))&term.months.eq(12)]
        for ax,m,title in zip(axes.flat[:4],['auc','ap','brier','coverage'],['AUC ↑','AP ↑','Brier ↓','Top20覆盖率 ↑']):
            ax.plot(g.combined_trees,g[m],'-o',ms=4,color=color,label=str(y));ax.set_title(title)
        c=comp[comp.year.eq(y)].sort_values('trees_per_seed')
        axes[1,2].plot(c.combined_trees,c.spatial_table_cells,'-o',ms=4,color=color)
    ps=pd.read_csv(pub/'PREDICTION_STABILITY.csv');ps=ps[ps.version.eq('shrunk')&ps.months.eq(12)&ps.cause.eq(1)].copy()
    ps['combined_trees']=ps.arm.str[1:].astype(int)*3;ps=ps.groupby('combined_trees').mean_prediction_sd.mean()
    axes[1,1].plot(ps.index,ps.values,'-o',ms=4,color='#884488');axes[1,1].set_title('平均种子间预测标准差 ↓')
    axes[1,2].set_title('完整解释空间单元')
    for ax in axes.flat:
        ax.axvline(96,color='gray',ls='--',alpha=.6);ax.set_xticks([81,90,96,102,111,120]);ax.grid(alpha=.2)
    axes[0,0].legend(title='评价年度');fig.suptitle('81—120树局部搜索｜12个月先退市｜虚线为当前96树',fontsize=14)
    fig.supxlabel('三种子合并总树数（每种子树数 × 3）');fig.tight_layout(rect=[0,.02,1,.95])
    for ext in ['png','pdf']:fig.savefig(pub/f'LOCAL_TREE_SEARCH.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    log=run/'audit/pytest_final.log';assert '277 passed'in log.read_text()and'skipped'not in log.read_text();shutil.copy2(log,pub/'pytest_final.log')
    write(pub/'execution_receipt.json',dict(status='COMPLETED_EXPLORATORY_LOCAL_SEARCH',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_commit='0e5f864cc7d5e85ba491ac6784fb3f9e1536c397',analysis_code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        worker_code_commit=load(run/'audit/code_snapshot.json')['commit'],branch='research/xai-rsf-tree-count-local-v1',private_run=str(run),
        forest_fits=0,new_trees=0,calibration_fits=0,counts=[3*b for b in COUNTS],exports=156,reused_exports=12,representations=336,
        automatic_adoption=False,core_changed=False,exploratory_primary_point_winner=int(winner.combined_trees),
        gates={s:load(run/f'audit/gate_{s}.json')for s in ['pilot','formal']},
        validation={s:load(pub/f'{s}_validation.json')for s in ['metric','function','data']},
        input_manifest_sha256=sha(run/'audit/input_hashes.json'),source_manifest_sha256=sha(run/'audit/source_hashes.json'),
        tests='277 passed, independent R included;1 inherited EBM warning;no skips',failures=[],
        environment=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__),
        current_pointer_sha256=sha(REPO/'CURRENT_POST_ST_MODEL.json'),
        public_output_hashes={p.name:sha(p)for p in pub.iterdir()if p.is_file()and p.name!='execution_receipt.json'}))

if __name__=='__main__':main()
