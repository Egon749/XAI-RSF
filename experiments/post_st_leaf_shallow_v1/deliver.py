"""Publish aggregate evidence; keep company forecasts and fitted functions private."""
from pathlib import Path
import argparse,datetime,json,shutil
import pandas as pd
from experiments.post_st_prognosis_v1.common import sha,write
from experiments.post_st_next_open_info_v1.deliver import table

NAMES={'baseline':'原基线深度6','parent_shrink':'深度6+父节点收缩','shallow':'深度4','shallow_parent':'深度4+父节点收缩','standard_rsf':'标准RSF','training_AJ':'训练AJ'}


def metric_table(d):
    d=d[['scope','arm','auc','ap','brier','risk_mean','ipcw_observed_incidence']].copy();d['arm']=d.arm.map(NAMES)
    d.columns=['年度','版本','AUC','AP','IPCW Brier','平均风险','观察发生率'];return table(d)


def deliver(run,dest):
    for name,h in json.loads((run/'audit/immutable_input_hashes.json').read_text()).items():assert sha(run/'input'/name)==h
    alignment=json.loads((run/'audit/data_alignment.json').read_text())
    for name,h in alignment['unchanged_source_hashes'].items():assert sha(Path(name))==h
    audit=json.loads((run/'evaluation/independent_audit.json').read_text());val=json.loads((run/'evaluation/validation.json').read_text())
    formal=json.loads((run/'audit/formal_gate.json').read_text());pilot=json.loads((run/'audit/pilot_gate.json').read_text())
    assert all(x['status']=='PASS' for x in [audit,val,formal,pilot,alignment])
    out=dest/'results';out.mkdir(exist_ok=False)
    for p in (run/'evaluation').iterdir():
        assert p.suffix in ['.csv','.json'] and 'PRIVATE' not in p.name;shutil.copy2(p,out/p.name)
    for p in (run/'reports').glob('*.csv'):shutil.copy2(p,out/p.name)
    for name in ['data_alignment.json','pre_fit_freeze.json','formal_gate.json','pilot_gate.json',
        'environment_oliver.json','environment_lemon.json','full_tests_pre_fit.log','tests_final.log','shrinkage_unit_tests.log']:
        shutil.copy2(run/'audit'/name,out/name)
    plan=json.loads((run/'input/folds/2023/plan.json').read_text());protocol=json.loads((run/'audit/protocol.json').read_text())
    write(out/'frozen_model_configurations.json',dict(forest_common=plan['forest_common'],xai_q2=plan['forest_variants']['xai_q2'],
        standard_rsf_reference=plan['standard_rsf'],input_columns=len(plan['feature_names']),seeds=protocol['seeds'],arms=protocol['arms']))
    m=pd.read_csv(out/'metrics.csv',dtype={'scope':str,'seed':str});w=pd.read_csv(out/'warning_metrics.csv',dtype={'scope':str,'seed':str})
    main=m[(m.seed.isin(['ensemble','none']))&(m.horizon_months==24)&(m.cause=='termination')].sort_values(['scope','arm'])
    warn=w[(w.seed=='ensemble')&(w.horizon_months==24)][['scope','arm','top_fraction','checked','captured','target_events','coverage','known_non_target','unknown','lead3','lead6']].sort_values(['scope','arm','top_fraction'])
    complexity=pd.read_csv(out/'ensemble_complexity.csv');st=pd.read_csv(out/'seed_stability.csv')
    st=st[st.cause=='termination'].groupby(['test_year','arm'])[['main_jaccard','pair_jaccard','median_function_correlation']].mean().reset_index()
    nodes=pd.read_csv(out/'node_audit.csv');support=nodes.groupby(['test_year','arm']).agg(leaves=('leaves','sum'),
        zero_target=('zero_target_leaves','sum'),at_most_one_target=('at_most_one_target_leaves','sum'),borrowed=('zero_target_leaves_with_borrowed_risk','sum')).reset_index()
    runtime=pd.read_csv(out/'execution_tasks.csv');fr=runtime[runtime.stage=='formal']
    assert len(fr)==12 and int(fr.new_forest_fits.sum())==6 and int(fr.leaf_updates.sum())==12 and int(fr.model_artifacts.sum())==18
    (dest/'RESULTS_ZH.md').write_text('''# 固定四组实验完成：局部改善，没有全面胜出的新版本

保留旧C输入、q2、三个固定种子和32树/种子，其余预算不变。4个数值小试和12个正式作业全部成功；正式部分实际新训练6个深度4森林，复用6个深度6森林，并生成12个对应收缩模型。共评价24个q2和6个冻结标准RSF参照，未重训参照。主结果先平均三个种子预测再计算指标。

## 24个月先最终退市

'''+metric_table(main)+'''
深度4+收缩在2024的AUC从0.752升至0.770、AP从0.758339升至0.782032；2023的AUC从0.736111升至0.746528，但AP从0.576727降至0.522272。合并AUC/AP从0.723852/0.669576变为0.729592/0.681719，Brier从0.313764变差为0.315749。因此不能把2024单年的排序提高写成整体改进成功。

单独收缩的合并AP几乎不变（0.669394），Brier未改善；单独深度4的合并AP为0.679892，Top20多捕获1例，但2023 AP和覆盖下降。三个改进版本均未通过训练前冻结的逐年度综合保留条件，见retention_gate.csv。该条件是实用阈值检查，不是显著性检验。

## 固定核查比例

'''+table(warn)+'''
Top20按年度取整，2023核查8家、2024核查9家：基线捕获5/12、7/20；单独收缩5/12、7/20；深度4为4/12、9/20；组合为4/12、8/20。合计依次12/32、12/32、13/32、12/32。没有因为排序较好而扩大核查人数。

Top25合计21家，四组依次捕获15、14、13、13例。所有测试结局均已知，未知0、重复0；已知非目标包含先恢复及窗口内未发生目标，不能解释为永久正常。年度完整队列回顾排序不等于在线政策。

本轮按范围结束，保留原基线作为参照，不自动替换默认配置或追加搜索。总体研究判断仍为INCONCLUSIVE。
''')
    recovery=m[(m.seed=='ensemble')&(m.horizon_months==24)&(m.cause=='recovery')].sort_values(['scope','arm'])
    secondary=m[(m.seed=='ensemble')&(m.horizon_months==12)&(m.cause=='termination')].sort_values(['scope','arm'])
    diag=pd.read_csv(out/'prediction_change_diagnostics.csv')
    (dest/'PREDICTION_COMPARISON_ZH.md').write_text('''# 排序、覆盖与概率质量分开判断

'''+metric_table(main)+'''
## AUC提高而Brier变差的实际来源

2024组合版本提高排序，但观察退市公司的平均预测从14.10%降至13.48%；非目标公司从10.20%降至10.07%。目标组的Brier贡献恶化约0.004614，非目标组改善约0.000298，合计恶化0.004316。AUC看相对次序，Brier也惩罚目标公司的概率低估。

父节点使用同一历史训练样本的AJ曲线，无法引入新制度下更高的发生率。2023四组平均退市预测仅6.27%—6.47%，观察33.33%；2024为11.58%—11.93%，观察44.44%。本实验没有校准层，没有把函数乘以趋零斜率。收缩不能解决整体风险水平漂移。

'''+table(diag[(diag.arm!='standard_rsf')&(diag.group!='all')][['test_year','arm','group','n','risk_mean','mse','brier_contribution_change']])+'''
## 同一公司的跨种子预测波动

'''+table(diag[(diag.arm!='standard_rsf')&(diag.group=='all')][['test_year','arm','seed_std_mean']])+'''
收缩降低预测跨种子波动，但更稳定不等于更准确、更好校准或交互选择更稳定。三个固定种子范围不是置信区间，也没有估计完整训练不确定性。

## 24个月先恢复

'''+metric_table(recovery)+'''
## 补充12个月先最终退市

'''+metric_table(secondary)+'''
252行指标包含全部种子、均值预测、年度/合并、两个原因和两个期限。合并AP在拼接预测上重算，独立AUC/AP/Brier逐行核验，96行基线/标准RSF参照指标精确复现。测试端点完整，IPCW权重为1；每家公司一行，记录等权和公司等权相同。

标准RSF是本ST后任务的randomForestSRC 3.9.0参照，不是首次ST预警M2。它与q2共用样本和公司bootstrap，但缺失处理、切点及阶数限制有差异，不能宣称后端等价。
''')
    (dest/'DATA_ALIGNMENT_AUDIT_ZH.md').write_text('''# 数据与标签均未变化

'''+table(pd.read_csv(out/'cohort_counts.csv'))+'''
全部83列训练/预测输入NPZ与刚完成的旧C复跑逐字节一致，字段顺序、sample_id、公司组、原点、事件、随访和信息截止未改。训练目录仅包含训练数据与测试X/sample_id，测试结局单独存储，不传入训练函数。原来源、冻结输入和参照模型均核验SHA256。

对象为全非金融行业历史观察中的首次财务/审计ST企业，起点为ST公告后下一交易日开盘前；终点为先恢复A或最终终止上市X，均按公告日。主窗口24日历月、12个月补充；先恢复再退市仍计首次恢复竞争结局。

2023模型训练2012—2022入组、信息严格早于2023-01-01：309家、14例退市；2024模型训练2012—2023入组、信息严格早于2024-01-01：345家、26例退市。两折654条，去重345家26个退市事件。6次新正式拟合共1962条重复训练计算记录，不是新增样本；叶估计更新也不增加独立事件。训练未知保持删失。

测试2023为36家12例退市，2024为45家20例退市；共81家32例退市、31例先恢复、18例窗口内无首次事件。两期限测试结局已知，每折训练测试无公司重叠。

保留年报接受前的保守旧输入，不重新生成财务值，不证明其比接受口径更真实。原年报接受实验、研究者确认和不利结果保留。保守时钟不等于原始PDF/PIT已独立认证；来源覆盖仍截至2026-09-03，没有扩展覆盖或评价2025入组队列。

2023/2024结果已暴露，本次在分析结果后提出算法变化，全部标为POST_EXPOSURE_EXPLORATORY_FIXED_TWO_BY_TWO，不是盲测或独立外部验证。
''')
    ctab=complexity[['test_year','arm','main_functions','interaction_functions','spatial_table_cells','scalar_table_cells','low_support_cell_fraction','zero_support_cell_fraction']]
    (dest/'INTERPRETABILITY_COMPARISON_ZH.md').write_text('''# 精确性保留，简单性局部改善，稳定性仍有局限

每年度以完整96树的全部主效应和二元交互重构两个原因的CIF，保留缺失分区及所有原生时间。不截取Top变量，不用解释代理。贡献是CIF概率贡献，不是风险率乘数或因果效应。

'''+table(ctab)+f'''
八个集成在全部训练/测试公司和原生时间重构/保存重载通过，最大重构误差{audit['max_ensemble_all_native_time_reconstruction_error']:.3e}。每棵树路径仍至多两个原始输入，无隐藏高阶残差。

深度4使2023分区格3869→3146（减少18.69%），2024为4068→3263（减少19.79%）。主效应仍62/65个，二元交互仍144/146个。已有函数在浅路径已进入，降低深度主要减少同一变量对内继续细分，不能称变量或交互数明显减少。单独收缩不改格数或切点。

低支持格比例50.81%/48.30%→43.58%/41.31%；零支持格占比20.44%/22.89%→22.12%/25.13%，但绝对零支持格数下降，占比升高与分母缩小有关。仍有大量样本稀少区域。

## 相同训练参考上的跨种子稳定性

'''+table(st)+'''
主效应和交互选择Jaccard在四组完全相同，交互仍约0.073/0.092。共同非恒定函数相关性2024从0.524升至0.643，2023从0.733降至0.663。共同函数相关性不能代替完整结构稳定性。

## 原始事件支持与借用信息分开

'''+table(support)+'''
深度6收缩给278/287个原本没有退市事件的叶节点借入正概率，深度4收缩为178/186个。它们仍没有本叶目标事件支持，正概率不是新证据。变浅后2023仍有812/1007叶节点至多1例退市，2024为649/1019。

本轮支持减少部分分区复杂度、降低预测随机波动和保持精确分解，不支持交互选择稳定性已解决或统计可辨识性已证明。没有移植newRSF2的Hessian惩罚。
''')
    (dest/'TRAINING_DIAGNOSTICS_ZH.md').write_text(f'''# 实际执行、数值与公平性

冻结拟合提交：{plan['code_commit']}。训练前及交付前完整131项测试通过，0跳过，含独立R检查；11项新增测试覆盖手工真值、重复bootstrap独立公司计数、原模型不变、强度0精确无操作、非法输入、缺失路由、完整函数和保存重载。唯一既有警告为EBM缺失值可视化提示。

只试预设强度10：w=n/(n+10)，叶CIF'=w×原叶AJ+(1−w)×原直接父AJ。n是该叶独立入袋公司数，重复行不虚增n；相同w用于所有原因及时间，根叶不变。这是一层直接父节点混合，不是递归层级收缩或校准。

凸组合保持非负、时间单调及原因之和≤1。父/叶AJ来自相同bootstrap，修改后的叶仍是至多两坐标的阶梯函数，因此q2展开保持精确。未裁剪预测或放宽测试。

4/4小试完成，Slurm {pilot['job']}，{pilot['by_host']}，计算/回传{pilot['seconds']:.3f}秒；仅检查数值与资源，不据小试评价选配置。12/12正式作业完成，Slurm {formal['job']}，{formal['by_host']}，计算/回传{formal['seconds']:.3f}秒。正式单任务{fr.seconds.min():.3f}—{fr.seconds.max():.3f}秒，最高RSS {fr.peak_rss_MiB.max():.2f} MiB。

正式部分6次新森林拟合、12次叶估计更新、18个新模型产物；复用6个q2基线和6个标准RSF。小试另有4次4树拟合和4次叶更新，不混入正式结果。失败0、选择性重跑0、发布失败0；任务ID、计算、回传和哈希全部核验。两节点环境匹配，p/both布局单线程/任务、4GiB地址空间限制，由既有调度管理CPU亲和。

独立核验24个q2模型的{audit['nodes_checked']}节点、{audit['selected_scores_checked']}个已选分裂、{audit['leaf_estimators_checked']}个叶估计。独立风险集递推重算AJ和混合，最大叶差{audit['max_leaf_estimator_error']:.3e}；独立分裂分数差{audit['max_score_error']:.3e}。6个标准RSF bootstrap和训练中位数核验通过。

6组匹配种子中，收缩前后全部结构相同；深度4公共路径前缀与深度6逐项相同，唯一截断在预设深度。核心分裂搜索未修改。森林无梯度求解器，Hessian秩、优化gap不适用。所有种子和不利结果保留，未追加搜索。
''')
    (dest/'METHOD_INNOVATION_DECISION_ZH.md').write_text('''# 研究判断INCONCLUSIVE；固定优化没有全面优于基线

1. 实现/计算成功：完成真实数据新训练、叶修改、预测和精确解释，全部测试与独立数值检查通过。
2. 简化局部有效：深度4减少约19%分区格，但函数数及交互选择稳定性不变。
3. 单独收缩没有稳定预测增量：合并AP几乎不变，2023 AP下降、2024 Brier恶化。它降低随机波动，没有解决风险低估。
4. 深度4和组合在2024排序更好，2023 AP/覆盖下降。三个新版本均未通过事前逐年度AUC/AP、Brier和Top20保留条件。这是本次固定设置的权衡，不证明所有收缩/浅树方法无效。
5. 解释可靠性部分改善：共同函数相关性2024提高、2023下降；交互Jaccard仍低，大量叶只有0—1例事件。借用父概率不增加独立事件。
6. 不替换旧基线、不宣布创新成功、不继续搜索。整体q2精确函数框架价值与本次叶估计调整的收益分开评价；可辨识性惩罚没有获得新支持。

父节点收缩属于已有统计正则化思路。这里是一层、共同原因/时间权重的AJ凸组合，不能宣称发明层级收缩或完整复现其递归算法。原始文献：[Agarwal等，Hierarchical Shrinkage，ICML 2022](https://proceedings.mlr.press/v162/agarwal22b.html)。概率约束和q2展开由代码与独立测试验证。

测试2023/2024已多次暴露，本轮不是新的独立验证；仅81家、32个先退市事件，三个种子范围不是置信区间。概率低估、历史数据版本和时间漂移仍限制判断，不能通过回退输入或挑选年度消除这些限制。
''')
    receipt=dict(status='COMPLETE',research_decision='INCONCLUSIVE',optimization_result='NO_UNIFORM_GAIN_PARTIAL_COMPLEXITY_REDUCTION',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),private_run=str(run),repository='Egon749/XAI-RSF',
        branch='research/xai-rsf-post-st-prognosis-v1',fitting_commit=plan['code_commit'],source_status=protocol['source_status'],evaluation_status=protocol['evaluation_status'],
        formal_tasks=12,new_formal_forest_fits=6,new_formal_leaf_updates=12,new_formal_model_artifacts=18,
        pilot_tasks=4,pilot_new_forest_fits=4,pilot_leaf_updates=4,q2_models_evaluated=24,standard_reference_models=6,
        reused_baseline_q2_models=6,failed_tasks=0,failed_publications=0,tests_passed=131,tests_skipped=0,
        slurm_jobs=dict(pilot=pilot['job'],formal=formal['job']),formal_by_host=formal['by_host'],
        training_fold_rows=654,training_company_union=345,training_termination_union=26,repeated_new_formal_fit_rows=1962,
        test_companies=81,test_terminations=32,test_recoveries=31,input_columns=83,changed_inputs=0,changed_labels=0,
        maximum_reconstruction_error=audit['max_ensemble_all_native_time_reconstruction_error'],main_horizon_months=24,secondary_horizon_months=12,
        calibration_fits=0,new2025_evaluation=False,default_changed=False,descriptive_gate_passed_arms=0,
        result_sha256={str(p.relative_to(dest)):sha(p) for p in out.iterdir() if p.is_file()},
        code_sha256={p.name:sha(p) for p in dest.glob('*.py')},document_sha256={p.name:sha(p) for p in dest.glob('*.md')},
        protocol_sha256=sha(dest/'frozen_protocol.json'),core_shrinkage_sha256=sha(dest.parents[1]/'crforest/shrinkage.py'))
    write(dest/'execution_receipt.json',receipt);write(run/'execution_receipt.json',receipt)
    print(json.dumps({k:receipt[k] for k in ['status','research_decision','formal_tasks','new_formal_forest_fits','new_formal_leaf_updates']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--dest',type=Path,required=True)
    a=p.parse_args();deliver(a.run,a.dest)
