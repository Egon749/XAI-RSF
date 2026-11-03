"""Unified24-month comparison; no composite winner or automatic replacement."""
import datetime,platform,subprocess,shutil
import pandas as pd
import scipy,sklearn,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.post_st_tree_count_v1.report import table
from .common import *

def main():
    r=run_path();pub=REPO/'experiments'/NAME
    for n in ['data','metric','function']:assert load(r/f'reports/{n}_validation.json')['status']=='PASS'
    for p in(r/'reports').iterdir():
        assert p.suffix in ['.csv','.json']and not(pub/p.name).exists();shutil.copy2(p,pub/p.name)
    d=pd.read_csv(pub/'METRICS_ENSEMBLE.csv');d=d[d.version.eq('shrunk')].copy()
    d['trees_per_seed']=d.arm.str[1:].astype(int);d['combined_trees']=d.trees_per_seed*3
    d['capture']=d.apply(lambda x:f'{int(x.captured)}/{int(x.target_events)}'if pd.notna(x.captured)else'NA',axis=1)
    d=d.sort_values(['scope','months','cause','trees_per_seed'])
    target=d[d.cause.eq(1)&d.months.eq(24)];pooled=target[target.scope.eq('pooled')]
    rows=[]
    for b in COUNTS:
        row=dict(trees_per_seed=b,combined_trees=3*b)
        for year in [2023,2024]:
            z=target[target.scope.eq(str(year))&target.trees_per_seed.eq(b)].iloc[0]
            for metric in ['auc','ap','brier','capture','risk_mean','observed']:row[f'{metric}_{year}']=z[metric]
        z=pooled[pooled.trees_per_seed.eq(b)].iloc[0]
        for metric in ['auc','within_year_auc','ap','brier','capture','risk_mean','observed']:row['pooled_'+metric]=z[metric]
        rows.append(row)
    comparison=pd.DataFrame(rows);comparison.to_csv(pub/'H24_COMPARISON.csv',index=False)
    extrema=[]
    for (scope,h,c),g in d[d.supported].groupby(['scope','months','cause']):
        for metric in ['auc','within_year_auc','ap','brier','captured']:
            if not g[metric].notna().any():continue
            value=g[metric].min()if metric=='brier'else g[metric].max()
            extrema.append(dict(scope=scope,months=h,cause=c,metric=metric,value=value,
                total_trees=';'.join(map(str,g.loc[np.isclose(g[metric],value,atol=1e-12,rtol=0),'combined_trees'])),
                status='EXPOSED_DESCRIPTIVE_EXTREMUM_NOT_ADOPTED'))
    pd.DataFrame(extrema).to_csv(pub/'METRIC_SPECIFIC_EXTREMA.csv',index=False)
    comp=pd.read_csv(pub/'NUMERICAL_COMPLEXITY.csv');comp=comp[comp.seed.eq('ensemble')&comp.version.eq('shrunk')&comp.year.isin([2023,2024])]
    cx=comp.groupby('combined_trees')[['main_functions','interaction_functions','spatial_table_cells','scalar_table_cells']].mean().reset_index()
    fn=load(pub/'function_validation.json');verdict=(pub/'INTERPRETATION_ZH.md').read_text()
    cols={'combined_trees':'总树数','auc_2023':'2023 AUC','ap_2023':'2023 AP','brier_2023':'2023 Brier','capture_2023':'2023 Top20','auc_2024':'2024 AUC','ap_2024':'2024 AP','brier_2024':'2024 Brier','capture_2024':'2024 Top20'}
    report=f'''# 70—120棵统一比较：重点24个月先退市

{verdict}

## 范围和执行口径

固定74输入、退市支持保护、q≤2、深度4、共享5切点、均衡25变量池、直接父节点AJ收缩10、各年1月1日前历史数据以及24个日历月训练随访。比较的是森林树数，不是改变训练随访或预测时点。三个种子42/101/202树数相同、等权平均；70—120总树数范围中可用72、75……120共17档。70和71不能分成三个相同整数树数，没有改变权重来凑端点。

复用81—120全部336份表示，只补72、75、78的36个独立联合导出任务（72份raw/收缩表示），仍全部采用已保存森林的原编号前缀，无新森林、校准器或树拟合，无重复收缩。沿用上一轮120树资源试运行，未重复计算试运行。合计153份种子模型、51份合并预测器，raw/收缩共408份表示。旧81—120的全部年度、种子、版本、原因和时限指标逐值复现。

这是在此前已暴露树数结果后的有限扩展，协议先于新增72/75/78指标提交。24个月单独展示，不把上一轮以12个月排序得到的105树结论改写为24个月最优。只列不同指标各自极值，不构造事后综合评分，不自动替换96树，不按年度选不同树数拼接表现。

## 24个月逐年度：全部17档

{table(comparison,cols)}

Top20分别核查2023年的8/36家和2024年的9/45家；目标事件数分别12和20。比较的是24个月内“先最终退市”，恢复为竞争结局。未知结局不当正常。2025年24个月随访不足，全部性能NA，未混入下表的合并评价。

## 24个月合并：重新计算指标

{table(comparison,{'combined_trees':'总树数','pooled_within_year_auc':'年内AUC','pooled_auc':'pooled AUC','pooled_ap':'pooled AP','pooled_brier':'Brier','pooled_capture':'Top20合计','pooled_risk_mean':'平均预测','pooled_observed':'观察率'})}

年内AUC只比较同年事件—非事件配对，按配对数加权；pooled AUC/AP从两年合并逐公司预测重新计算，不是年度均值。Brier沿用统一IPCW评价，支持端点权重均为1且独立核对。Top20先逐年确定名额再相加。两年度均值预测仍需结合观察率看待，排序改变不能当作概率校准改善。

## 解释复杂度与完整性

以下为2023和2024年度合并预测器的平均规模，计算包含全部原生时间、两个原因与缺失状态，而非只截取24个月单个点：

{table(cx,{'combined_trees':'总树数','main_functions':'主效应','interaction_functions':'二元函数','spatial_table_cells':'空间单元','scalar_table_cells':'标量单元'})}

全408份表示的最大完整重构误差{fn['max_full_reconstruction']:.3e}，独立逐项求和误差{fn['max_explicit_sum']:.3e}；全部训练及评价公司、原生时间、两个原因通过重载和CIF非负/单调/总和≤1检查，没有概率裁剪。函数非零以全时间/原因幅度大于1e-14计。空间单元包括缺失状态，标量单元另乘时间与原因维度。解释精确描述模型，不代表真实作用恢复或因果效果。

FUNCTION_STABILITY.csv、PREDICTION_STABILITY.csv保留种子间集合、共同函数和预测波动；共同函数使用同年度训练边际乘积参考，函数池随树数变化，不可仅据相关性证明稳定。LEAF_SUPPORT.csv保留真实叶支持，收缩没有增加真实事件。

## 其他结果与不确定性

12个月、恢复原因、raw版本及单种子结果全部保留于METRICS_ALL.csv，主展示为三种子平均后计算。METRIC_SPECIFIC_EXTREMA.csv列所有指标各自极值，不能只挑有利结果。2023年12个月恢复无事件，AUC/AP为NA；2025年24个月性能NA。

1000次公司级配对bootstrap按年度分层，沿用分析种子2026110201与原抽样索引，比较每档减96树。区间条件于固定模型，不含训练、此前变量选择、多次已暴露树数搜索及挑最高点的不确定性。区间跨0不是性能等效；任何极值也不是确认性优越。原始披露版本/PIT及74变量选择来源限制仍保留。

计算由已有两节点Slurm完成，独立导出并验证发布哈希；未改变CPU亲和性管理或共享存储。所有未收敛/失败必须保留。CURRENT_POST_ST_MODEL.json、主模型文件夹、核心森林实现及所有旧实验未改动。企业标识、预测、模型及函数参数仍留私有目录。
'''
    (pub/'RESULTS_ZH.md').write_text(report)
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':10})
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for year,color in zip([2023,2024],['#4477aa','#ee7733']):
        g=target[target.scope.eq(str(year))]
        for ax,metric,title in zip(axes.flat[:4],['auc','ap','brier','coverage'],['AUC ↑','AP ↑','Brier ↓','Top20覆盖率 ↑']):
            ax.plot(g.combined_trees,g[metric],'-o',ms=4,color=color,label=str(year));ax.set_title(title)
    axes[1,1].plot(pooled.combined_trees,pooled.within_year_auc,'-o',ms=4,label='年内AUC')
    axes[1,1].plot(pooled.combined_trees,pooled.auc,'-s',ms=4,label='pooled AUC');axes[1,1].legend();axes[1,1].set_title('两年合并排序')
    axes[1,2].plot(cx.combined_trees,cx.spatial_table_cells,'-o',ms=4,color='#884488');axes[1,2].set_title('完整解释空间单元：两年平均')
    for ax in axes.flat:
        ax.axvline(96,color='gray',ls='--',alpha=.6);ax.set_xticks(list(range(72,121,6)));ax.grid(alpha=.2)
    axes[0,0].legend(title='评价年度');fig.suptitle('72—120树统一比较｜未来24个月先退市｜虚线为当前96树',fontsize=14)
    fig.supxlabel('三种子合并总树数');fig.tight_layout(rect=[0,.02,1,.95])
    for ext in ['png','pdf']:fig.savefig(pub/f'H24_TREE_COMPARISON.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    log=r/'audit/pytest_final.log';assert '278 passed'in log.read_text()and'skipped'not in log.read_text();shutil.copy2(log,pub/'pytest_final.log')
    write(pub/'execution_receipt.json',dict(status='COMPLETED_EXPLORATORY_COMPARISON',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_commit='53158f80ce2472bfee646af13c5ab94899f8cba9',analysis_code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        worker_code_commit=load(r/'audit/code_snapshot.json')['commit'],branch='research/xai-rsf-tree-count-70-120-v1',private_run=str(r),
        forest_fits=0,new_trees=0,calibration_fits=0,new_export_tasks=36,new_representations=72,reused_representations=336,total_representations=408,
        total_tree_grid=[3*b for b in COUNTS],automatic_adoption=False,core_changed=False,gate=load(r/'audit/gate_formal.json'),
        validation={s:load(pub/f'{s}_validation.json')for s in ['metric','function','data']},
        input_manifest_sha256=sha(r/'audit/input_hashes.json'),source_manifest_sha256=sha(r/'audit/source_hashes.json'),
        tests='278 passed,independent R included;1 inherited EBM warning;no skips',failures=[],
        environment=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__),
        current_pointer_sha256=sha(REPO/'CURRENT_POST_ST_MODEL.json'),
        public_output_hashes={p.name:sha(p)for p in pub.iterdir()if p.is_file()and p.name!='execution_receipt.json'}))

if __name__=='__main__':main()
