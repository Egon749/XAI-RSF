"""Publish aggregate-only reports from completed fixed artifacts; never fit."""
import datetime,platform,shutil,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import *

LABEL={'reference83':'83项原版','support83':'83项＋支持','drop9':'74项原版','drop9_support':'74项＋支持'}

def table(rows,headers):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)

def number(x,d=4):return 'NA' if pd.isna(x) else f'{x:.{d}f}'

def main():
    public=REPO/'experiments'/NAME;reports=RUN/'reports'
    a=pd.read_csv(reports/'METRICS_ENSEMBLE.csv',dtype={'scope':str});a=a[a.version.eq('shrunk')]
    cc=pd.read_csv(reports/'NUMERICAL_COMPLEXITY.csv');bb=pd.read_csv(reports/'PAIRED_BOOTSTRAP_95.csv',dtype={'scope':str})
    checks=[]
    # Reproduce published historical numbers from source prediction files.
    old=pd.read_csv(REPO/'experiments/post_st_cause_support_v1/METRICS_ENSEMBLE.csv',dtype={'scope':str})
    for r in a[a.arm.isin(['reference83','support83'])].itertuples():
        arm='baseline' if r.arm=='reference83' else 'support'
        z=old[(old.scope==r.scope)&(old.arm==arm)&(old.version=='shrunk')&(old.months==r.months)&(old.cause==r.cause)].iloc[0]
        for field in ['auc','ap','brier','risk_mean','captured']:
            np.testing.assert_allclose(getattr(r,field),z[field],atol=1e-12,rtol=0,equal_nan=True)
        checks.append(dict(scope=r.scope,arm=r.arm,months=r.months,cause=r.cause,source_report_match=True))
    for name in ['post_st_drop9_v1','post_st_drop9_transport_v1']:
        old=pd.read_csv(REPO/'experiments'/name/'results/metrics.csv')
        old=old[(old.seed=='ensemble')&(old.arm=='drop9')]
        for r in old.itertuples():
            z=a[(a.scope==str(r.year))&(a.arm=='drop9')&(a.months==r.horizon_months)&(a.cause==r.cause)].iloc[0]
            for field in ['auc','ap','brier','risk_mean']:
                np.testing.assert_allclose(z[field],getattr(r,field),atol=1e-12,rtol=0,equal_nan=True)
            checks.append(dict(scope=r.year,arm='drop9',months=r.horizon_months,cause=r.cause,source_report_match=True))
    pd.DataFrame(checks).to_csv(reports/'HISTORICAL_REPRODUCTION.csv',index=False)
    q=a[(a.cause==1)&a.scope.ne('pooled')&a.supported].copy()
    q['observed_expected']=q.observed/q.risk_mean;q.to_csv(reports/'TERMINATION_COMPARISON.csv',index=False)
    for path in reports.glob('*.csv'):shutil.copy2(path,public/path.name)
    for path in reports.glob('*.json'):shutil.copy2(path,public/path.name)
    rows=[]
    for year,h in [(2023,12),(2023,24),(2024,12),(2024,24),(2025,12)]:
        g=q[(q.scope==str(year))&(q.months==h)].set_index('arm');o,n=g.loc['drop9'],g.loc['drop9_support']
        rows.append([f'{year}/{h}月',f'{o.auc:.4f}→{n.auc:.4f}',f'{o.ap:.4f}→{n.ap:.4f}',f'{o.brier:.6f}→{n.brier:.6f}',f'{int(o.captured)}/{int(o.target_events)}→{int(n.captured)}/{int(n.target_events)}'])
    text='# 74项输入＋退市支持保护：探索性匹配结果\n\n'
    text+='**本轮没有得到“74项去冗余与退市支持保护叠加后全面更好”的证据。** 2023年AUC上升；2024年12/24月AUC、AP下降；2025年12月接近原74版。所有五个可评价年度/时限的Top20捕获均不变。完整精确解释保留，空间单元数减少，但交互数量与稳定性并非一致改善。当前参考模型和全部旧模型保持不变。\n\n'
    text+='## 1. 最直接的匹配比较：74项原版 → 74项＋支持\n\n'+table(rows,['年度/时限','AUC↑','AP↑','IPCW Brier↓','Top20捕获'])
    text+='\n先平均种子42/101/202的逐公司预测，再计算指标。每种子32树，完整预测器96树。Top20名额按年度ceil(0.2n)，2023/24/25为8/9/11家。2025年24月随访不足，预测已保存，性能NA。上述支持终点IPCW权重恰为1，已独立用sklearn和平方误差核对。\n\n'
    text+='## 2. 四个版本放在同一评价口径下\n\n'
    rows=[]
    for year,h in [(2023,12),(2023,24),(2024,12),(2024,24),(2025,12)]:
        for arm in LABEL:
            r=q[(q.scope==str(year))&(q.months==h)&(q.arm==arm)].iloc[0]
            rows.append([year,h,LABEL[arm],number(r.auc),number(r.ap),number(r.brier,6),f'{int(r.captured)}/{int(r.target_events)}'])
    text+=table(rows,['年度','月','版本','AUC','AP','Brier','Top20捕获'])
    text+='\n83项与74项之间同时存在内容删除和候选覆盖变化；74＋支持相对74原版只有支持保护开关变化。83项支持保护的局部收益不能直接移植为74项上的预期收益。2024年24月74＋支持AUC0.6700，低于74原版0.6940及83项原版0.7700，不隐去这一不利结果。\n\n'
    text+='## 3. 年内排序与跨年合并\n\n'
    rows=[]
    for h in [12,24]:
        for arm in LABEL:
            r=a[(a.scope=='pooled')&(a.months==h)&(a.cause==1)&(a.arm==arm)].iloc[0]
            rows.append([h,LABEL[arm],number(r.within_year_auc),number(r.equal_year_auc),number(r.auc),number(r.ap),number(r.brier,6),f'{int(r.captured)}/{int(r.target_events)}'])
    text+=table(rows,['月','版本','年内配对加权AUC','年AUC等权均值','pooled AUC','pooled AP','pooled Brier','年度选定后捕获合计'])
    text+='\n年内AUC只比较同年事件/非事件配对，以事件数×非事件数加权；等权均值是不同汇总口径。12月年内AUC由74原版0.7957降至0.7921，而年等权均值由0.7671升至0.7831，主要因为2023年的提升权重不同。pooled重新计算，不能与年内排序互换。12月合并2023—2025，24月只合并2023/2024。\n\n'
    text+='## 4. 不确定性与恢复结局\n\n'
    rows=[]
    z=bb[(bb.cause==1)&bb.contrast.eq('74support_minus_74')&bb.metric.isin(['auc','ap'])&bb.scope.ne('pooled')]
    for r in z.itertuples():rows.append([r.scope,r.months,r.metric,number(r.estimate),f'[{r.lower95:.4f}, {r.upper95:.4f}]',r.undefined_repeats])
    text+=table(rows,['年度','月','指标','新74支持−原74','描述性95%区间','未定义重抽样'])
    text+='\n复用cause-support的1000组按测试年分层公司抽样，seed2026110201，所有版本同组索引，固定模型。上述AUC/AP区间均跨0，不能认定稳定改善或性能等效。2023/2025的少事件导致部分抽样AUC/AP未定义，保留NA和计数，不填0。不包含训练样本、筛选或后续开发不确定性。\n\n'
    rows=[]
    for year,h in [(2023,12),(2023,24),(2024,12),(2024,24),(2025,12)]:
        g=a[(a.scope==str(year))&(a.months==h)&(a.cause==2)].set_index('arm');o,n=g.loc['drop9'],g.loc['drop9_support']
        rows.append([year,h,f'{number(o.auc)}→{number(n.auc)}',f'{number(o.ap)}→{number(n.ap)}',f'{number(o.brier,6)}→{number(n.brier,6)}'])
    text+=table(rows,['年度','月','恢复AUC','恢复AP','恢复Brier'])
    text+='\n2023年12月没有恢复事件，AUC/AP不可估计。恢复24月在2023/2024小幅改善，并不消除退市24月2024变差的代价。全部单种子、raw及收缩结果均在CSV中。\n\n'
    text+='## 5. 解释复杂度、稳定性及保护实际做了什么\n\n'
    rows=[]
    for year in YEARS:
        g=cc[(cc.year==year)&(cc.seed=='ensemble')&(cc.version=='shrunk')].set_index('arm');o,n=g.loc['drop9'],g.loc['drop9_support']
        rows.append([year,f'{int(o.main_functions)}→{int(n.main_functions)}',f'{int(o.interaction_functions)}→{int(n.interaction_functions)}',f'{int(o.spatial_table_cells)}→{int(n.spatial_table_cells)}',f'{100*(n.spatial_table_cells/o.spatial_table_cells-1):.1f}%',int(n.scalar_table_cells)])
    text+=table(rows,['年度','主效应','二元交互','空间单元','空间变化','新模型标量单元'])
    text+='\n计数沿用完整函数导出器，空间包括缺失状态，标量单元再乘原生时间与两个原因；96树合并后的规模不是三种子计数均值。主/交互计数沿用全单元/全时间非零约定。2024年交互反而由132增至135，不能把单元减少等同于交互必然减少。\n\n'
    support=pd.read_csv(reports/'LEAF_SUPPORT.csv').groupby('year')[['leaves','zero_target_leaves','one_target_leaves','borrowed_positive_target_leaves']].sum()
    search=pd.read_csv(reports/'SEARCH_DIAGNOSTICS.csv').groupby('year')[['guard_rejected','guard_changed_winner','guard_stopped_split']].sum()
    rows=[[y,*[int(support.loc[y,k]) for k in support],*[int(search.loc[y,k]) for k in search]] for y in YEARS]
    text+=table(rows,['年度','总叶数','零退市叶','仅1家退市叶','零退市叶借入正风险','拒绝候选数','改变胜者节点','停止分裂节点'])
    text+='\n同一节点的多个候选可重复计数，不是公司数。保护实际消除了本轮的单家退市叶，但大量零退市叶仍存在；父节点收缩借入风险不增加真实事件支持。它改变搜索结构而非校准概率。\n\n'
    st=pd.read_csv(reports/'FUNCTION_STABILITY.csv');rows=[]
    for year in YEARS:
        g=st[(st.year==year)&st.version.eq('shrunk')&st.day.eq(365)&st.cause.eq(1)&st['order'].eq(2)].groupby('arm')[['jaccard','median_correlation']].mean()
        rows.append([year,f'{g.loc["drop9","jaccard"]:.3f}→{g.loc["drop9_support","jaccard"]:.3f}',f'{g.loc["drop9","median_correlation"]:.3f}→{g.loc["drop9_support","median_correlation"]:.3f}'])
    text+=table(rows,['年度','交互集合Jaccard（种子对均值）','共同交互形状相关（各对中位数的均值）'])
    text+='\n以同年度训练边际乘积为参考，对共同函数在同一训练X上比较365天退市贡献；常数函数不计算相关。集合重合度与形状一致性并非一同改善，2023形状相关下降。三种子只描述算法随机性，不是三个独立验证集，也不证明真实作用恢复。完整数据提供730天和恢复原因。\n\n'
    text+='## 6. 概率水平与数值验证\n\n'
    rows=[]
    for r in q[q.arm.eq('drop9_support')].itertuples():rows.append([r.scope,r.months,f'{r.risk_mean:.2%}',f'{r.observed:.2%}',f'{r.observed_expected:.2f}'])
    text+=table(rows,['年度','月','新74支持平均退市预测','观察退市率','O/E'])
    fv=load(reports/'function_validation.json')
    text+=f'\n低估仍存在，支持保护不等于风险水平更新。新18个单种子表示及6个96树合并表示，全部训练与评价样本、全部原生时间、两个原因通过概率与q≤2检查。最大直接重构误差{fv["max_reconstruction_error"]:.3g}，逐项独立求和误差{fv["max_explicit_sum_error"]:.3g}，森林重载误差0，均满足1e-10。没有裁剪、校准或隐藏高阶残差。\n\n'
    text+='## 7. 执行、失败记录与研究判断\n\n9/9正式森林拟合、9个收缩派生、6/6合并导出成功；另1/1四树数值试运行。正式作业2625，oliver7项、lemon2项；试运行2624，合并导出2626。已核验计算完成、回传完成和文件哈希。正式批次计算约20.3秒、含排队约21.7秒；单森林拟合约1.31—3.08秒，峰值RSS不超过297MiB，不将不同机器时间直接当速度消融。全部源模型、输入与CURRENT指针未改动。\n\n1次准备代码检查失败已保留：比较不等长切点列表的方式错误，修复为逐变量精确比较后通过；无失败拟合、无筛选种子或结果后的追加实验。全pytest含独立R检查通过；既有EBM缺失值可视化警告与本实验无关。\n\n**判断：有局部结构简化与2023排序收益，没有稳定的预测增量证据。** 2024两时限表现下降、捕获不增、校准未解决，故本轮不支持自动替换83或74模型。解释精确性得以保持是已验证性质；更简单且更稳定的解释、少样本支持保护的普遍收益均不能由本轮推出。本实验是已暴露年度上的固定组合探索，不是新的独立盲测。\n'
    (public/'RESULTS_ZH.md').write_text(text)
    methods='''# 可用于论文的匹配实验描述

在固定83项与既有74项变量模型的基础上，补齐“74项＋退市支持保护”组合。每年按1月1日前快照训练，沿用下一交易日开盘前起点及先最终退市/先恢复定义，路径变量上限2、深度4、每树25项均衡候选、共享分位数切点、等原因权重和直接父节点AJ收缩10。每年三个固定种子各32树，仅将子节点中原因1独立公司数约束由关闭改为“0或至少2”。与原74项模型复用相同bootstrap、候选池及切点。所有新森林保留raw和收缩版本，先平均三个种子的预测，再评价支持的12/24日历月结局。

74项模型加入支持保护后，2023年12/24月退市AUC分别由0.7188/0.7569升至0.7969/0.7708；2024年相应AUC由0.8491/0.6940降至0.8233/0.6700；2025年12月AUC由0.7333变为0.7292。五个支持终点的Top20捕获均不变，固定模型公司bootstrap的相关AUC/AP差值区间均包含0。完整函数空间单元分别减少8.8%、4.2%、2.6%，但交互数量和跨种子稳定性没有一致改善。全预测器精确二阶分解、CIF合法性和保存重载均经全部样本及原生时间核验。结果支持局部权衡而非普遍优越；所有年度已被查看，属探索性分析，不改变冻结研究参考。
'''
    (public/'PAPER_NOTE_ZH.md').write_text(methods)
    fig,axes=plt.subplots(1,2,figsize=(11,4.2));positions=[('2023',12),('2023',24),('2024',12),('2024',24),('2025',12)]
    for arm,label in [('reference83','83 reference'),('support83','83 + support'),('drop9','74 reference'),('drop9_support','74 + support')]:
        for ax,metric in zip(axes,['auc','ap']):
            vals=[float(q[(q.scope==y)&(q.months==h)&(q.arm==arm)][metric].iloc[0]) for y,h in positions]
            ax.plot(range(5),vals,marker='o',label=label);ax.set_xticks(range(5),[f'{y}\n{h}m' for y,h in positions]);ax.set_ylabel(metric.upper());ax.grid(alpha=.2);ax.set_ylim(0,1)
    axes[0].legend(fontsize=8);fig.suptitle('Termination: fixed 96-tree predictors | exploratory comparison');fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(public/f'PERFORMANCE_COMPARISON.{ext}',dpi=180)
    plt.close(fig)
    # Full provenance and immutable source checks, never overwrite old artifacts.
    verified=0
    for f,h in load(RUN/'audit/source_hashes.json').items():assert sha(Path(f))==h;verified+=1
    for directory,manifest in [('input','immutable_input_hashes.json'),('analysis_input','immutable_analysis_hashes.json')]:
        for f,h in load(RUN/'audit'/manifest).items():assert sha(RUN/directory/f)==h
    pointer=load(RUN/'audit/current_pointer.json');assert sha(Path(pointer['path']))==pointer['sha256']
    regs=load(RUN/'audit/registry_formal.json');cost=pd.read_csv(reports/'COMPUTE_COST.csv')
    env={'python':platform.python_version(),'platform':platform.platform()}
    import importlib.metadata
    for n in ['numpy','pandas','scipy','scikit-learn','matplotlib','pytest']:env[n]=importlib.metadata.version(n)
    (public/'environment.json').write_text(json.dumps(env,indent=2))
    receipt=dict(status='COMPLETE',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),branch='research/xai-rsf-drop9-support-v1',source_commit='fcf98aac4c68ac4ac1cadac4227262e0937e94a3',fitting_commit=load(RUN/'audit/code_snapshot.json')['commit'],analysis_commit=load(RUN/'audit/analysis_code_snapshot.json')['commit'],private_run=str(RUN),formal_forest_fits=9,pilot_fits=1,forest_fits=10,formal_derived_shrunk_models=9,reused_forests=27,ensemble_exports=6,calibrator_fits=0,failed_fits=0,preparation_failures=1,failures=[dict(stage='preparation',reason='Ragged cut arrays compared incorrectly; repaired per-variable exact comparison before any fit',retained=str(RUN/'preparation_attempt1/failure.json'))],formal_execution_by_host=pd.Series([r['host'] for r in regs]).value_counts().to_dict(),slurm_jobs={s:load(RUN/f'audit/gate_{s}.json') for s in ['pilot','formal','ensemble']},training_snapshot_rows=1044,distinct_training_companies=390,latest_training_termination_companies=49,evaluation_companies=134,supported_12month_termination=25,supported_24month_termination=32,unsupported='2025H24 performanceNA',validation=fv,environment=env,source_files_unchanged=verified,current_pointer_sha256=pointer['sha256'],source_manifest_sha256=sha(RUN/'audit/source_hashes.json'),frozen_input_manifest_sha256=sha(RUN/'audit/immutable_input_hashes.json'),bootstrap_indices_sha256=sha(RUN/'private/bootstrap_indices_PRIVATE.npz'),public_contains='code, fixed protocol, hashes, aggregate evidence only',reference_replaced=False,remote_sync='verify after final commit; exact delivery SHA recorded in private git_delivery.json')
    receipt['public_output_sha256']={f.name:sha(f) for f in public.iterdir() if f.is_file() and f.name!='execution_receipt.json'}
    (public/'execution_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print('Aggregate reports complete; historical predictions reproduced and immutable sources unchanged.')

if __name__=='__main__':main()
