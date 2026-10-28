"""Create the 2025 12-month result report from verified fixed predictions."""
from pathlib import Path
import argparse
import json
import shutil
import numpy as np
import pandas as pd
from experiments.post_st_prognosis_v1.common import sha,write
from experiments.post_st_prognosis_v1.deliver import table,LABELS


def figure(df,metrics,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'font.size':11,'svg.hashsalt':'train2024-h12'})
    models=['xai_q2','unrestricted','standard_rsf'];names=['XAI-RSF q2','无限阶森林','标准RSF','无变量AJ']
    fig,axes=plt.subplots(2,3,figsize=(13,7.5))
    for row,cause in enumerate(['termination','recovery']):
        d=df[(df.scope.astype(str)=='2025')&(df.cause==cause)].set_index('model').loc[models]
        b=metrics[(metrics.scope.astype(str)=='2025')&(metrics.cause==cause)&(metrics.model=='training_AJ')].iloc[0]
        for col,key in enumerate(['auc','ap','brier']):
            ax=axes[row,col];values=np.r_[d[f'{key}_mean'].to_numpy(),b[key]]
            err=np.vstack([np.r_[d[f'{key}_mean']-d[f'{key}_min'],0],np.r_[d[f'{key}_max']-d[f'{key}_mean'],0]])
            bars=ax.bar(np.arange(4),values,yerr=err,capsize=4,color=['#3277a8','#6a9b85','#d68a56','#a6a9ae'],width=.65)
            ax.bar_label(bars,labels=[f'{v:.3f}' for v in values],padding=5,fontsize=9)
            ax.set_xticks(np.arange(4),names,rotation=15,fontsize=9)
            ax.set_ylim(0,1 if key=='auc' else .5 if key=='ap' else .14)
            if key=='auc':ax.axhline(.5,color='#777777',linestyle='--',linewidth=1)
            if key=='ap':ax.axhline(b.ipcw_observed_incidence,color='#777777',linestyle='--',linewidth=1)
            ax.set_title(('退市：5例' if row==0 else '恢复：3例')+' · '+{'auc':'AUC ↑','ap':'AP ↑','brier':'Brier ↓'}[key])
            ax.spines[['top','right']].set_visible(False)
    fig.suptitle('2025年53家企业 · ST后12个月首次转归',fontsize=16)
    fig.text(.5,.018,'训练2012—2024年入组，信息截止2025-01-01之前；训练和评价均截断于12个月。\n'
             '森林柱为三个固定种子的指标均值，误差线为最小—最大值，不是置信区间。目标事件稀少，结果为探索性。',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.075,1,.95));fig.savefig(out/'prediction_summary.png',dpi=170)
    fig.savefig(out/'prediction_summary.svg',metadata={'Date':None});plt.close(fig)
    p=out/'prediction_summary.svg';p.write_text('\n'.join(x.rstrip() for x in p.read_text().splitlines())+'\n')


def deliver(run,destination):
    reports=run/'evaluation';valid=json.loads((reports/'validation.json').read_text());extra=json.loads((reports/'additional_validation.json').read_text())
    assert valid['status']==extra['status']=='PASS'
    plan=json.loads((run/'input/plan.json').read_text())
    alignment=json.loads((run/'audit/data_alignment.json').read_text())
    raw=json.loads((run/'audit/market_source_verification.json').read_text());assert raw['status']=='PASS'
    out=destination/'results';out.mkdir(exist_ok=False)
    for p in reports.iterdir():
        assert p.suffix in ['.csv','.json'];shutil.copy2(p,out/p.name)
    for name in ['cohort_counts.csv','sample_selection.csv','feature_manifest.csv']:
        shutil.copy2(run/'reports'/name,out/name)
    for name in ['data_alignment.json','pre_fit_protocol_freeze.json','preparation_history.json','market_source_verification.json',
                 'pilot_gate.json','formal_gate.json','environment_oliver.json','environment_lemon.json','tests_final.log']:
        shutil.copy2(run/'audit'/name,out/name)
    assert '83 passed' in (out/'tests_final.log').read_text() and ' skipped' not in (out/'tests_final.log').read_text()
    d=pd.read_csv(out/'metrics.csv',dtype={'scope':str,'seed':str});s=pd.read_csv(out/'metrics_summary.csv',dtype={'scope':str})
    diag=pd.read_csv(out/'training_diagnostics.csv');interp=pd.read_csv(out/'interpretability.csv');stab=pd.read_csv(out/'seed_stability.csv')
    support=pd.read_csv(out/'tree_support.csv');warnings=pd.read_csv(out/'warning_metrics.csv',dtype={'seed':str})
    def doc(name,text):
        with (destination/name).open('x') as f:f.write(text.strip()+'\n')
    def means(cause):
        f=s[(s.scope=='2025')&(s.cause==cause)].copy();f['model']=f.model.map(LABELS)
        return table(f[['model','auc_mean','auc_min','auc_max','ap_mean','brier_mean','risk_mean_mean']])
    def ensembles(cause):
        f=d[(d.scope=='2025')&(d.cause==cause)&d.seed.isin(['ensemble','none'])].copy();f['model']=f.model.map(LABELS)
        return table(f[['model','auc','ap','brier','risk_mean','ipcw_observed_incidence']])
    figure(s,d,out)
    doc('RESULTS_ZH.md','''# 2025年12个月预测结果

本轮真正重训的是12个月随访模型，不是从24个月拟合结果中取一个12月截面。训练2012—2024入组，截止2025-01-01之前；评价2025年53家新入组公司。目标为先最终退市X或先完整恢复A，仍为两个相容的竞争风险CIF。

12个月已知结局：退市5/53（9.43%）、恢复3/53（5.66%）、无首次转归45/53。这里的45家不等于恢复正常，可能仍处于风险警示；未知结局0。每家公司一条记录，公司等权与记录等权相同，测试IPCW权重均为1。

## 退市：三种子指标均值与AUC范围

'''+means('termination')+'''

## 恢复：三种子指标均值与AUC范围

'''+means('recovery')+'''

范围是三个种子的最小—最大值，不是置信区间。未选择最好种子。样本只有5个退市和3个恢复事件，指标易受个别公司排序影响，不能宣称模型间差异显著。

## 固定三种子集成与无变量AJ：退市

'''+ensembles('termination')+'''

## 固定三种子集成与无变量AJ：恢复

'''+ensembles('recovery')+'''

集成先平均三个种子的CIF，再重算AUC/AP/Brier，不能混称为指标均值。AJ由训练集估计，不使用2025发生率拟合。

q2退市AUC均值0.544，固定集成AUC0.533；AP均值0.134，集成0.117。无变量AJ的AP为0.0943。q2 Brier均值0.08765、集成0.08754，均略高于AJ的0.08664；当前没有展示概率预测优于无变量基线。无限阶排序略高，但Brier也高于AJ。恢复仅3例，q2 AUC均值0.367，仍不支持可靠恢复预测。

q2平均退市预测5.64%，观察9.43%；平均恢复预测15.91%，观察5.66%。不能因为退市概率差距比旧实验小，就认定校准改进来自算法。

## 与前轮比较的边界

上一轮评价2024入组、主要观察24个月，本轮评价2025入组且训练／评价均改成12个月。公司、事件比例、训练年份、风险集和随访长度均改变。尤其Brier从旧约0.343变成约0.088，主要不能按数字大小直接解释为巨大性能改善；必须看本轮AJ基线。AP也受发生比例影响，不把旧0.549与本轮0.134直接比较为方法退步幅度。

![2025年预测指标](results/prediction_summary.png)

完整逐种子、集成及训练表观指标见 `results/metrics.csv`；表观训练指标不是验证结果。
''')
    w=warnings[warnings.seed=='ensemble'].copy();w['model']=w.model.map(LABELS)
    doc('PREDICTION_COMPARISON_ZH.md','''# 核查容量和校准

Top20%按2025年度53家公司向上取整，核查11家；Top25%核查14家。按入组时固定12个月CIF排序，并列按固定sample_id排序。以下为固定三种子集成，不是挑选最佳种子。

'''+table(w[['model','top_fraction','checked_companies','captured_events','all_target_events','event_coverage','known_non_target_checked','unknown_checked',
             'identified_at_least_3_calendar_months_before_termination','identified_at_least_6_calendar_months_before_termination']])+'''

三个模型在两个核查容量下都只捕获1/5个退市事件（20%覆盖）；捕获事件满足提前3个月及6个月条件。Top25增加核查公司，没有增加捕获数量。相同捕获数不表示三个模型选中了完全相同的公司。已知非目标包含恢复和时限内无首次转归，不能都称为正常；本队列未知0、重复预警0。

这是全年回顾性容量比较，不是假定年初可以预知全年入组公司池的在线预警政策。全部种子核查结果保存在CSV，未只留集成或成功种子。概率校准五等分表见 `results/calibration.csv`；只有5/3个事件，分箱发生率波动大，不实施基于本测试集的再校准。
''')
    q=diag[diag.model=='xai_q2'];a=interp[interp.cause=='termination']
    max_error=q[['reconstruction_error','function_reload_error','native_grid_reconstruction_error','complete_component_sum_error']].max().max()
    doc('INTERPRETABILITY_COMPARISON_ZH.md','''# 完整函数解释

路径始终最多2个原始输入变量，允许在已使用变量上继续分裂。全部主效应和二元交互精确重构两原因CIF，不是事后拟合代理模型，也没有未解释的高阶残差。这不自动证明统计可辨识、跨样本稳定或因果效应。

'''+table(a[['seed','native_all_time_main_functions','native_all_time_pair_functions','spatial_table_cells','scalar_table_cells',
            'reference_active_mains','reference_active_pairs','reference_pair_absolute_share','low_support_cell_fraction','zero_support_cell_fraction']])+f'''

全原生时间网格42—43个主效应、52—59个交互，1665—1772个空间格。原因／时限特定的参考非零项可少于全函数数量。两个原因、365/366天、原生网格及保存重载检查的最大重构误差 {max_error:.3e}。贡献在CIF概率尺度相加，不是对数风险率或因果贡献。

## 跨种子稳定性

'''+table(stab[stab.cause=='termination'][['seed_a','seed_b','main_jaccard','pair_jaccard','common_nonconstant_functions','median_function_correlation']])+'''

交互集合Jaccard只有0.049—0.088，主效应约0.509—0.545。约42%—44%的函数格少于10家训练公司支持。完整重构依然精确，但“简单、可靠、低歧义”没有得到充分证明。跨种子方向相关基于相同390家参考、固定365天，三个种子范围不是置信区间；没有声称优于SHAP的稳定性，也没有测试可辨识性惩罚。

## 实际变量使用和原因支持

'''+table(support)+'''

每叶至少3条全部原因事件不保证有3家独立退市企业。真实变量清单、切点、参数、全函数值和企业贡献均在私有目录；仓库保存汇总诊断。变化不能归因于新的解释机制，因为模型机制没有改变。
''')
    counts=pd.read_csv(out/'cohort_counts.csv')
    doc('DATA_ALIGNMENT_AUDIT_ZH.md','''# 样本、时钟与可得性

训练和评价最多观察12个日历月；训练另外受2025-01-01之前的信息截止限制。使用实际公告日，不把执行日、报告期或今天当成缺失日期替代。所有443条记录用独立日期逻辑核对首次A/X和删失截止，通过。

'''+table(counts)+'''

训练390家中事件0为294家，其中263家完整12月无首次转归、31家提前删失。23个退市和73个恢复均在训练截止前且发生于自身12个月窗内。不能把上一轮24月事件数直接搬进12月模型。2025年53家入组日期范围2025-04-19至2025-07-02，按继承权威源覆盖边界2026-09-03，其12月结果均可判定；无需把未成熟样本当成正常。

2025候选资格沿用既有首次ST审计和人工原因复核：原执行日期在2025年的85条候选中，53条财务／审计原因明确、32条原因需复核；既有人工复核将其中2条纳入、30条排除。纳入者继续经过首次事件审计及公告年、行业筛选，不能把执行年计数直接当作公告年最终样本。不存在本轮按模型预测或结局筛选公司。

“首次”仍有证据边界：53家中仅14家具有完整日度历史的首次支持，其余沿用已接受的观察期首次及历史缺口审计，不宣称全部已证明上市生命周期首次。

原市场缓存止于2024年底。本轮使用两个原始CSMAR日行情工作簿及既有2025缓存，重新读取1,940,826条原始行，逐值核对其中1,303,971条缓存交易记录，数值、代码、日期完全一致。以真实交易日期构造延长至2025-12-31的日历，按各公司入组日之前计算24个市场字段，未制造availability。两个源、缓存、代码和运行都有哈希。

财务48＋市场24＋质量3，共75输入；财务衍生增长／变化项同时检查上期数据可得日。重建旧390家入组特征与原冻结数组逐值精确一致。新53家的财务／市场日期均严格早于入组日，来源与逐变量缺失率可查。记录等权与公司等权一致，训练／测试公司组交集为0。

原始财报版本和行业历史版本仍未完全PIT认证。同日年报仍被原公告日起点排除，下一交易日起点和原始年报版本恢复尚未实施。真实日期检查不等于版本已获PIT认证；本轮仍标PROVISIONAL。2023/2024已被研究接触；2025本轮按用户授权评价，虽本轮配置先冻结，未实施独立封存，不能宣称严格盲测。

一次准备因行情日期字符串与Timestamp混用导致Parquet写入失败，在任何模型拟合前已修复；原目录、代码与日志保留，成功准备使用新目录。没有改变数据值、删失败公司或放宽检验。
''')
    gates={k:json.loads((out/f'{k}_gate.json').read_text()) for k in ['pilot','formal']}
    doc('TRAINING_DIAGNOSTICS_ZH.md',f'''# 执行和数值检查

冻结拟合代码：`{plan['code_commit']}`。9个正式拟合、2个4树试运行均计算及回传成功，无拟合失败、无发布失败。另2个原始行情核查任务完成。完整测试83项通过、0跳过，含独立R检查；唯一警告为既有EBM缺失值可视化提示。

现有Slurm两节点physical池：行情作业2537，试运行2538，正式2539。两机Python/R和共享目录访问均预检；实际所有短拟合由oliver领取，lemon未执行拟合，不宣称双机实测加速。每拟合单CPU、线程1、地址空间上限4GiB，未绕过P/E核及SMT cgroup限制，没有GPU计算。正式pool耗时 {gates['formal']['seconds']:.3f} 秒，单任务耗时之和 {diag.seconds.sum():.3f} 秒，不是完整研究流程耗时。

'''+table(diag[['model','seed','seconds','peak_rss_MiB','child_peak_rss_MiB','candidate_evaluations','leaves','max_path_variables','reload_error']])+'''

原森林核心、分裂准则、默认原因权重(1,1)、共享切点候选语义和三个模型预算不变。worker仅将预测／解释时点及支持诊断改为12月；训练和评价数据本身在12月截断。所有CIF有限、非负、单原因≤1、原因和≤1、随时间不减，保存重载一致，q2路径阶数／祖先检查通过，未裁剪预测。

独立逐对AUC、并列阈值AP与直接Brier复算26组2025指标，最大容许差5e-14，全部通过；常数AJ的退市AP严格等于5/53。536个冻结输入／代码文件和9套模型产物哈希全部通过；独立核验443条日期、12月上限及训练信息截止。森林无Hessian优化器，gap、系数触边不适用，不填0冒充收敛。

数据准备失败1次已留档，之后无拟合补跑、无测试集再校准、无追加配置或种子。所有模型及参数留在私有目录。
''')
    doc('METHOD_INNOVATION_DECISION_ZH.md','''# 本轮研究判断

**INCONCLUSIVE：本轮预测结果偏弱，且仅5个退市、3个恢复事件，尚不能可靠比较方法优劣。数据时钟与首次历史证据仍为PROVISIONAL。** 这不意味着结果支持当前模型投入应用；本轮没有显示足够的预测优势。

1. 训练与预测可运行：9/9正式模型和2/2试运行成功。完整解释继续精确。
2. 退市区分度弱：q2 AUC均值0.544、无限阶0.590、标准RSF0.528；均无明确、稳健的优势证据。没有统计显著性结论。
3. 概率预测没有超过无变量基线：三个模型的退市Brier均高于训练AJ；不能把比旧年份更小的Brier归为算法成功。
4. 核查覆盖不足：三个固定集成在Top20和Top25均只捕获1/5退市事件。没有因扩大核查量获得额外捕获。
5. 解释稳定性仍低：交互Jaccard约0.049—0.088，精确加和不等于稳定可靠。
6. 整体框架与独立机制分开：本轮只是训练年份及观察窗改变，没有新增方法，也没有测试可辨识性惩罚或目标原因权重；不支持宣称这些机制有独立贡献。
7. 不能把本轮结果与旧2024／24月结果差异单独归因于数据扩充，评价公司、时限及事件比例都改变。此前针对旧配置的负面判断保留，不覆盖旧报告。

保留全部种子和不利结果，本轮停止，不自动扩大超参数搜索、改变终点或启动新的模型。
''')
    with (destination/'README.md').open('a') as f:
        f.write('''
## 实际交付

9个正式模型＋2个试运行成功，83项测试通过。2025年退市AUC三种子均值：q2 0.544、无限阶0.590、标准RSF0.528；固定集成Top20/25均捕获1/5事件。q2 Brier约0.08765，未优于无变量AJ约0.08664。研究决策为 **INCONCLUSIVE**，少事件与数据证据限制不允许宣称方法成功。

- [完整预测结果](RESULTS_ZH.md)、[核查容量与校准](PREDICTION_COMPARISON_ZH.md)。
- [解释与稳定性](INTERPRETABILITY_COMPARISON_ZH.md)、[数据对齐审计](DATA_ALIGNMENT_AUDIT_ZH.md)。
- [执行诊断](TRAINING_DIAGNOSTICS_ZH.md)、[研究判断](METHOD_INNOVATION_DECISION_ZH.md)、[执行回执](execution_receipt.json)。
''')
    receipt=dict(experiment='post_st_train2024_h12_v1',status='COMPLETE',data_status=plan['status'],decision='INCONCLUSIVE',
        branch='research/xai-rsf-post-st-prognosis-v1',training_commit=plan['code_commit'],private_run=str(run),
        training_companies=390,training_termination_events=23,training_recovery_events=73,training_early_censored=31,
        test_companies=53,test_termination_events=5,test_recovery_events=3,test_no_first_outcome=45,test_unknown=0,
        training_entry_years=[2012,2024],test_entry_year=2025,horizon_calendar_months=12,maximum_training_followup_months=12,
        training_information_cutoff_exclusive='2025-01-01',formal_fits_expected=9,formal_fits_succeeded=9,
        formal_fit_rows_including_seed_repetition=3510,pilot_fits_succeeded=2,raw_verification_tasks_succeeded=2,
        failed_fits=0,failed_publications=0,failed_preparation_attempts=1,additional_tuning_trials=0,
        cause_weights_changed=False,next_trading_day_origin_applied=False,strict_blind_test_claim=False,
        original_M2_retrained=False,prior_experiments_preserved=True,model_hashes=valid['model_hashes'],jobs=gates,
        requested_nodes=['oliver','lemon'],actual_fit_hosts={'oliverdesktop':11},full_tests_passed=83,full_tests_skipped=0,
        independent_metric_rows=26,independent_date_records=443,model_input_hashes=plan['source_hashes'],
        source_sha256=alignment['original_source_sha256'],raw_market_verification=raw,
        immutable_input_manifest_sha256=sha(run/'audit/immutable_input_hashes.json'),
        public_results_sha256={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()},
        documents_sha256={p.name:sha(p) for p in sorted(destination.glob('*.md'))},
        source_code_sha256={p.name:sha(p) for p in sorted(destination.iterdir()) if p.suffix in ['.py','.json']},
        notes=['Seed ranges are not confidence intervals','No per-company details, raw forecasts, thresholds or fitted parameters published'])
    write(destination/'execution_receipt.json',receipt);shutil.copy2(destination/'execution_receipt.json',run/'execution_receipt.json')
    print(json.dumps(dict(status='PACKAGED',result_files=len(receipt['public_results_sha256']))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--destination',type=Path,required=True)
    a=p.parse_args();deliver(a.run,a.destination)
