"""Publish allowlisted aggregates; keep every company and fitted parameter private."""
from pathlib import Path
import argparse
import json
import shutil
import pandas as pd
from experiments.post_st_prognosis_v1.common import sha, write
from experiments.post_st_prognosis_v1.deliver import table, LABELS


def deliver(run, destination):
    reports=run/'evaluation';comparison=run/'comparison'
    valid=json.loads((reports/'validation.json').read_text())
    matched=json.loads((comparison/'comparison_receipt.json').read_text())
    assert valid['status']==matched['status']=='PASS'
    plan=json.loads((run/'input/plan.json').read_text())
    alignment=json.loads((run/'audit/data_alignment.json').read_text())
    output=destination/'results';output.mkdir(exist_ok=False)
    for p in reports.iterdir():
        assert p.suffix in ['.json','.csv'];shutil.copy2(p,output/p.name)
    for p in comparison.iterdir():
        assert p.suffix in ['.json','.csv','.png','.svg'];shutil.copy2(p,output/p.name)
    for name in ['cohort_counts.csv','training_by_year.csv','feature_manifest.csv']:
        shutil.copy2(run/'reports'/name,output/name)
    for name in ['data_alignment.json','pilot_gate.json','formal_gate.json','pre_fit_protocol_freeze.json',
                 'environment_oliver.json','environment_lemon.json','tests_final.log']:
        shutil.copy2(run/'audit'/name,output/name)
    tests=(output/'tests_final.log').read_text()
    assert '80 passed' in tests and ' skipped' not in tests
    metrics=pd.read_csv(output/'old_new_metrics.csv',dtype={'seed':str,'scope':str})
    means=pd.read_csv(output/'seed_mean_comparison.csv')
    warning=pd.read_csv(output/'old_new_warning_metrics.csv',dtype={'seed':str})
    diag=pd.read_csv(output/'training_diagnostics.csv')
    support=pd.read_csv(output/'tree_support_and_price_use.csv')
    interp=pd.read_csv(output/'old_new_interpretability.csv')
    stable=pd.read_csv(output/'old_new_seed_stability.csv')
    def doc(name,content):
        with (destination/name).open('x') as f:f.write(content.strip()+'\n')
    def mean_table(cause,months):
        d=means[(means.cause==cause)&(means.horizon_months==months)].copy()
        d['model']=d.model.map(LABELS);d['version']=d.version.map({'old':'原训练','new':'扩充至2023'})
        return table(d[['model','version','auc_mean','auc_min','auc_max','ap_mean','brier_mean','risk_mean_mean']])
    def ensemble_table(cause,months):
        d=metrics[(metrics.cause==cause)&(metrics.horizon_months==months)&metrics.seed.isin(['ensemble','none'])].copy()
        d['model']=d.model.map(LABELS)
        return table(d[['model','version','auc','ap','brier','risk_mean','ipcw_observed_incidence']])
    docs=[]
    for cause,title in [('termination','先退市'),('recovery','先完整恢复')]:
        for months in [24,12]:
            docs.append(f'## {months}个月{title}：三种子指标均值与AUC范围\n\n'+mean_table(cause,months))
    doc('RESULTS_ZH.md', '''# 新旧训练信息下的2024年结果

同一45家非金融企业，原入组特征、公告终点及评价规则逐项一致。24个月20个先退市、16个先完整恢复、9个无首次转归；12个月15个先退市、4个先完整恢复、26个时限内无首次转归。全部测试端点已知，IPCW权重均为1。每家公司一行，记录等权与公司等权相同。

下表是各自三个种子的指标均值，AUC范围是种子最小—最大值，并非置信区间。AP越高越好，Brier越低越好。`old`对应原309家／14退市，`new`对应345家／26退市。固定种子编号相同不代表新增公司后bootstrap样本相同。没有筛除种子、显著性检验或完整训练不确定性估计。

'''+'\n\n'.join(docs)+'''

## 如何解释变化

主终点24个月：q2 AUC均值0.5420→0.5500，改善很小；AP 0.5026→0.5493，Brier 0.3720→0.3434。新q2种子AUC为0.476—0.628，不能把均值变化当成稳健显著提升。无限阶森林AUC 0.5133→0.5987；标准RSF 0.5407→0.5600，但标准RSF的AP 0.5354→0.5041下降，结果并非所有指标一致改善。

12个月作为预先定义的补充终点：q2 AUC 0.5052→0.5919，无限阶0.5837→0.7519，标准RSF0.5867→0.6756。不能因补充终点改善较多就改称它为本次主要终点。q2与无限阶还同时不同在变量覆盖和切点组织，差距不能单独归因于二阶限制。

恢复24个月AUC均值：q2 0.4404→0.5079，仍接近随机；标准RSF 0.6552→0.7270。不能把退市与恢复表现合成一个笼统的“模型成功”结论。

![同一2024队列的旧新训练比较](results/prediction_change.png)

完整逐种子、集成、训练表观指标在 `results/metrics.csv`；旧新2024对照在 `results/old_new_metrics.csv`。训练表观结果不代表泛化；旧实验未经重训或覆盖。
''')
    alerts=warning[(warning.seed=='ensemble')&(warning.horizon_months==24)].copy()
    alerts['model']=alerts.model.map(LABELS)
    doc('PREDICTION_COMPARISON_ZH.md','''# 集成预测、校准与核查容量

以下集成是预先固定的三个种子CIF等权平均后重新计算指标，与三个指标的平均值不同。未按2024结果选择模型／种子或校准。训练AJ是由各自训练数据得到的无变量基线，没有用测试发生率拟合。

## 24个月退市：固定集成及AJ

'''+ensemble_table('termination',24)+'''

## 12个月退市：固定集成及AJ

'''+ensemble_table('termination',12)+'''

## 24个月恢复：固定集成及AJ

'''+ensemble_table('recovery',24)+'''

## Top20%／Top25%：24个月退市

'''+table(alerts[['model','version','top_fraction','checked_companies','captured_events','all_target_events','event_coverage',
                   'known_non_target_checked','unknown_checked','identified_at_least_3_calendar_months_before_termination',
                   'identified_at_least_6_calendar_months_before_termination']])+'''

Top20%核查9家，Top25%向上取整为12家。排序是2024整年入组的回顾性排序，不是可在年初部署的全年在线配额。每家只有一次起点，无重复预警。已知非目标包含竞争恢复与时限内无首次转归，不一律称为正常企业；本测试集未知结局为0。

q2的Top20捕获3/20→5/20，Top25仍为5/20。新增q2的Top20和Top25均没有捕获公告前至少6个月的目标事件，原来分别有1和2个，不能只报告捕获总数改善。

概率低估仍严重：q2 24个月平均预测由8.98%升至13.02%，观察退市44.44%；12个月平均预测约3.51%，观察33.33%。Brier下降表示本次平方概率误差减小，不等于校准已经合格。恢复24个月q2平均预测51.48%，观察35.56%，仍偏高。五等分校准详见 `results/calibration.csv`。
''')
    selected=interp[(interp.horizon_months==24)&(interp.cause=='termination')]
    st=stable[stable.cause=='termination']
    ss=support[support.model=='xai_q2']
    doc('INTERPRETABILITY_COMPARISON_ZH.md','''# 完整内生解释、支持度与稳定性

所有q2模型保持路径最多2变量，全部主效应和二元交互共同重构两原因CIF，不丢弃高复杂度残差。没有因果解释，也没有启用newRSF2可辨识性惩罚。贡献直接在CIF尺度相加，不能称为风险率贡献。

## 全部原生时间网格的函数复杂度

'''+table(selected[['version','seed','native_all_time_main_functions','native_all_time_pair_functions','spatial_table_cells','scalar_table_cells',
                    'reference_active_mains','reference_active_pairs','reference_pair_absolute_share','low_support_cell_fraction','zero_support_cell_fraction']])+f'''

新增q2实际使用42—45个输入；全原生网格有42—45个主效应函数、54—59个二元交互，1885—2083个空间格。不同原因／时限的参考样本非零函数可少于全部网格函数，不能只展示较少的端点数冒充模型总复杂度。12/24月完整分量加和误差最大 {selected[selected.version=='new'].reconstruction_error.max():.3e}，原生时间网格重构误差最大 {diag.native_grid_reconstruction_error.max():.3e}；两原因检查通过，保存重载一致。解释“精确”继续成立。

## 跨种子稳定性

'''+table(st[['version','seed_a','seed_b','main_jaccard','pair_jaccard','common_nonconstant_functions','median_function_correlation']])+'''

交互集合Jaccard由原约0.057—0.077变为约0.078—0.130，有所增加但仍低。函数相关按各版本自己的固定训练参考、730天退市CIF计算，参考公司从309变345；不能将不同参考上的变化直接解释为纯函数形状改善。仅三个种子的描述结果，没有稳定性置信区间。切点由新增训练数据重估，允许数学表示变化，但未使用评价公司确定切点。

## 稀少退市与股价实际使用

'''+table(ss[['version','seed','leaves','zero_termination_leaf_fraction','fewer_than_three_termination_leaf_fraction',
             'price_eligible_trees','price_used_trees','root_price_trees','root_termination_company_min','root_termination_company_max']])+'''

q2零退市叶比例由约65%降至49%—53%，支持有所增加；约91%—92%的叶仍少于3家独立退市企业。每叶至少3条全部原因事件不等于退市原因有3家支持。股价资格仍为32/96棵树，实际使用4/96→3/96，根节点依旧0次；扩充数据并未使股价经常被选中。这不能单独证明抵消是主因，也不能从此次扩充实验识别改原因权重的效果。

真实切点、全函数参数、实际变量清单及企业贡献均留在服务器私有目录。汇总函数数量和精确加和不等于解释已经简单、稳定或统计可辨识。
''')
    cohort=pd.read_csv(output/'cohort_counts.csv')
    doc('DATA_ALIGNMENT_AUDIT_ZH.md','''# 训练扩充与数据时钟审计

训练入组2012—2023年，结局信息严格早于2024-01-01。原309家公司延长随访后，退市14→22、恢复184→198；加入2023年的36家公司，再增加4个已知先退市。2023队列其余32家截至训练截止仍删失，未使用其后来完整观察到的11退市／15恢复标签补齐训练。

'''+table(cohort)+'''

训练24个月：26退市＋198恢复＋77完整时限无首次转归＋44提前删失＝345家公司。拟合中的事件0为121，不可把它们都叫作正常。测试24个月45家，20退市、16恢复、9无首次转归，全部已知。

准备脚本先用原截止时间从审计端点账本重建309个旧标签，时间和事件逐项精确匹配，再统一延长训练截止。只允许在新截止前公告的端点及相关链审计参与训练；同日冲突或未解决链缺口会阻断。每家公司固定原入组时75列特征，未用2023年底财报更新早年入组特征。各模型训练内重新估计共享切点／标准RSF中位数，测试信息未进入这些计算。

2024数组的sample_id、公司组、X、时间、事件、时限与资格等所有字段均与原评价数组相同子集逐项核对。345训练公司与45测试公司无交集。旧输入、账本及新不可变输入/代码均有哈希；核对487个冻结文件。按用户新授权修改的是训练年份与训练结局截止，没有静默采用旧实验禁止滚动吸收2023的规则。

仍采用原入组公告日、严格早于该日的财务／市场信息。尚未实施此前讨论的下一交易日开盘前起点，也未恢复被后续修订覆盖的原始年报。这是为本轮保持可比性而明确保留的限制，不代表这些时钟问题已经解决。原财务版本、行业历史版本和首次事件历史完整性仍继承原审计限制，PROVISIONAL状态不变。

没有使用2024测试结局训练、调参、选种子或再校准；2024结果已在前次研究接触，不宣称严格盲测。没有读取或训练2025入组公司的结局。测试2024入组后的2025／2026端点仅属于其固定随访，不能与2025年新入组队列混为一谈。
''')
    gates={k:json.loads((output/f'{k}_gate.json').read_text()) for k in ['pilot','formal']}
    doc('TRAINING_DIAGNOSTICS_ZH.md',f'''# 训练与验证

冻结拟合代码 `{plan['code_commit']}`。与原拟合提交相比，`crforest/core.py`、`partition.py`、原worker及R训练入口内容相同；没有更改分裂公式、原因权重、候选语义、叶约束或预算。9个正式任务全部成功，另2个4树试运行成功，失败拟合0、失败发布0，没有补跑或额外调参。

试运行作业2535；正式作业2536。两节点环境、R依赖和NFS访问预检通过，Slurm申请physical/both。实际11个短任务全部由oliver领取，lemon未执行拟合，不能声称实测双机加速。每任务CPU亲和性1、BLAS/OMP/MKL线程1、地址空间上限4GiB。既有P/E核与cgroup配置未更改，未启用GPU。

正式pool耗时 {gates['formal']['seconds']:.3f} 秒，单任务墙钟时间总和 {diag.seconds.sum():.3f} 秒；两者均不是包含分析、审计和报告的全流程时间。独立任务共享CPU而非把单个训练进程跨两机拆分。

'''+table(diag[['model','seed','seconds','peak_rss_MiB','child_peak_rss_MiB','reload_error','candidate_evaluations','leaves','max_path_variables']])+'''

父进程与R子进程峰值分开报告，不将两个峰值直接相加为同时内存。全部CIF有限、非负、单原因≤1、两原因和≤1、随时间不减；保存重载通过。q2路径阶数及祖先检查通过，完整函数在两个原因、四个日历天数及原生网格精确重构。无预测裁剪。森林不使用Hessian凸优化，数值秩、最优性gap与系数触边不适用，不能伪填0。

完整测试80项通过、0跳过；唯一警告为既有EBM缺失值可视化提示。独立R检查已启用。另用逐对AUC、按并列分数组AP与直接Brier复算52组新2024指标，并从原保存预测独立复算52组旧2024指标；均在5e-14内一致。常数AJ的2024退市AP等于20/45，完整端点IPCW权重均为1。完整输入和模型文件哈希核验通过。执行中没有改变评价方法以提高结果。
''')
    doc('METHOD_INNOVATION_DECISION_ZH.md','''# 研究判断

**PREDICTIVE_NO_GO，限定当前输入、默认组合分裂和冻结q2配置的24个月ST后转归任务。** 数据证据为PROVISIONAL、后期结果已暴露的探索性比较。它不是对整个低阶RSF框架的永久否定，也不是对原因加权方案的检验。

1. 实际训练／预测成功：9/9正式拟合、2/2试运行及完整精确解释均通过。
2. 扩充训练信息有部分收益：q2 AP、Brier和Top20捕获有所改善，但主要24月AUC只有0.550，平均退市风险13.02%仍远低于观察44.44%。Top25捕获未增加，提前6个月捕获反而减少，不能称为全面改善。
3. 更灵活基线也受益，12个月无限阶AUC均值0.752较明显；仍不能用补充终点替换主终点，且其概率也严重低估。标准RSF24月AP下降，不隐去不利结果。
4. 解释精确性保留；交互稳定性略有提高，但Jaccard仍低，多数叶缺少足够退市支持。未达到解释简单、可靠已经充分证明的程度。
5. 数据扩充没有解决股价少被选中的现象，也没有分离“新增2023公司”和“延长旧公司随访”的贡献。不能从本轮推断抵消机制是唯一或主要原因。
6. 整体方法价值与可辨识性惩罚贡献分开：本轮没有新方法组件、没有可辨识性惩罚，不提供其独立贡献证据。精准函数分解仍是已验证属性，但不能据此宣布预测应用或论文创新成功。

本次有限实验到此结束；保留所有种子和不利结果，不自动扩大超参数搜索、改权重或改变研究终点。
''')
    # Append a self-contained result block to the pre-fit protocol README.
    with (destination/'README.md').open('a') as f:
        f.write('''
## 实际执行结果

已完成9个正式模型及2个试运行，无失败。研究判断为 **PREDICTIVE_NO_GO（当前冻结q2的24个月ST后转归配置）**；PROVISIONAL／探索性状态不变。

q2主要24月退市AUC均值0.542→0.550、AP 0.503→0.549、Brier 0.372→0.343；Top20集成捕获3/20→5/20，Top25仍5/20。概率仍明显低估：平均13.02%对观察44.44%。q2保持完整精确解释，但交互稳定性和稀少退市支持仍不足。没有追加训练或调参。

- [全部预测结果](RESULTS_ZH.md)、[集成、校准与Top20/25](PREDICTION_COMPARISON_ZH.md)。
- [解释、股价使用及事件支持](INTERPRETABILITY_COMPARISON_ZH.md)。
- [数据时钟与对齐](DATA_ALIGNMENT_AUDIT_ZH.md)、[执行诊断](TRAINING_DIAGNOSTICS_ZH.md)。
- [研究决策](METHOD_INNOVATION_DECISION_ZH.md)、[执行回执](execution_receipt.json)。
''')
    receipt=dict(experiment='post_st_train2023_v1',status='COMPLETE',data_status=plan['status'],decision='PREDICTIVE_NO_GO',
        decision_scope='Frozen q2 post-ST 24-month first-outcome prediction; not universal framework invalidation',
        branch='research/xai-rsf-post-st-prognosis-v1',training_commit=plan['code_commit'],private_run=str(run),
        original_training_commit=plan['original_fit_commit'],formal_fits_expected=9,formal_fits_succeeded=9,
        pilot_fits_expected=2,pilot_fits_succeeded=2,failed_fits=0,failed_publications=0,additional_tuning_trials=0,
        training_companies=345,unique_training_termination_events=26,unique_training_recovery_events=198,
        formal_fit_rows_including_seed_repetition=3105,test_2024_companies=45,test_h24_termination_events=20,
        test_h24_recovery_events=16,test_h24_unknown=0,entry_2025_outcomes_used=False,
        original_M2_retrained=False,all_original_models_preserved=True,training_information_cutoff_exclusive='2024-01-01',
        next_trading_day_origin_applied=False,cause_weights_changed=False,model_hashes=valid['model_hashes'],jobs=gates,
        requested_nodes=['oliver','lemon'],actual_fit_hosts={'oliverdesktop':11},
        full_tests_passed=80,full_tests_skipped=0,independent_metric_rows_new=52,independent_metric_rows_old=52,
        model_input_hashes=plan['source_hashes'],source_sha256=alignment['source_sha256'],
        immutable_input_manifest_sha256=sha(run/'audit/immutable_input_hashes.json'),
        public_results_sha256={p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()},
        documents_sha256={p.name:sha(p) for p in sorted(destination.glob('*.md'))},
        source_code_sha256={p.name:sha(p) for p in sorted(destination.iterdir()) if p.suffix in ['.py','.json']},
        note='Seed range is not confidence interval; no claimed temporal blindness or certified PIT; company details/models private.')
    write(destination/'execution_receipt.json',receipt)
    shutil.copy2(destination/'execution_receipt.json',run/'execution_receipt.json')
    print(json.dumps(dict(status='PACKAGED',files=len(receipt['public_results_sha256']),destination=str(destination))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--destination',type=Path,required=True)
    a=p.parse_args();deliver(a.run,a.destination)
