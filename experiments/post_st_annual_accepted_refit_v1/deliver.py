"""Deliver aggregate matched-refit evidence; never export company forecasts."""
from pathlib import Path
import argparse
import datetime
import json
import shutil

import numpy as np
import pandas as pd

from experiments.post_st_prognosis_v1.common import sha, write
from experiments.post_st_next_open_info_v1.deliver import table


STATES = {'conservative_C': '旧C', 'accepted_annual_C': '年报确认后C', 'common': '共同基线'}
MODELS = {'xai_q2': 'q2', 'standard_rsf': '标准RSF', 'training_AJ': '训练AJ'}


def metric_table(frame):
    d = frame[['scope', 'source_state', 'model', 'auc', 'ap', 'brier',
               'risk_mean', 'ipcw_observed_incidence']].copy()
    d['source_state'] = d.source_state.map(STATES)
    d['model'] = d.model.map(MODELS)
    d.columns = ['年度', '来源口径', '模型', 'AUC', 'AP', 'IPCW Brier', '平均风险', '观察发生率']
    return table(d)


def deliver(run, dest):
    protocol = json.loads((run/'audit/protocol.json').read_text())
    old = Path(protocol['reference_run'])
    alignment = json.loads((run/'audit/data_alignment.json').read_text())
    for name, h in alignment['unchanged_source_hashes'].items():
        assert sha(Path(name)) == h
    for name, h in json.loads((run/'audit/immutable_input_hashes.json').read_text()).items():
        assert sha(run/'input'/name) == h
    old_c = pd.read_csv(old/'evaluation/ensemble_complexity.csv')
    old_c = old_c[old_c.variant == 'announcement_fields'].assign(source_state='conservative_C')
    new_c = pd.read_csv(run/'evaluation/ensemble_complexity.csv').assign(source_state='accepted_annual_C')
    pd.concat([old_c, new_c], ignore_index=True).to_csv(run/'evaluation/complexity_comparison.csv', index=False)
    out = dest/'results'
    out.mkdir(exist_ok=False)
    for p in (run/'evaluation').iterdir():
        assert p.suffix in ['.csv', '.json'] and 'PRIVATE' not in p.name
        shutil.copy2(p, out/p.name)
    for p in (run/'reports').glob('*.csv'):
        shutil.copy2(p, out/p.name)
    for name in ['data_alignment.json', 'pre_fit_freeze.json', 'pilot_gate.json', 'formal_gate.json',
                 'environment_oliver.json', 'environment_lemon.json', 'tests_pre_freeze.log', 'tests_final.log']:
        shutil.copy2(run/'audit'/name, out/name)
    policy = Path(__file__).resolve().parents[1]/'post_st_annual_source_acceptance_v1/source_policy.json'
    shutil.copy2(policy, out/'source_policy.json')
    plan = json.loads((run/'input/folds/2023/announcement_fields/plan.json').read_text())
    write(out/'frozen_model_configurations.json', dict(forest_common=plan['forest_common'],
        xai_q2=plan['forest_variants']['xai_q2'], standard_rsf=plan['standard_rsf'],
        feature_names=plan['feature_names'], seeds=protocol['seeds'], pilot=plan['pilot']))
    m = pd.read_csv(out/'metrics.csv', dtype={'scope': str, 'seed': str})
    w = pd.read_csv(out/'warning_metrics.csv', dtype={'scope': str, 'seed': str})
    s = pd.read_csv(out/'training_diagnostics.csv')
    audit = json.loads((out/'independent_audit.json').read_text())
    validation = json.loads((out/'validation.json').read_text())
    formal = json.loads((out/'formal_gate.json').read_text())
    pilot = json.loads((out/'pilot_gate.json').read_text())
    assert validation['status'] == audit['status'] == formal['status'] == pilot['status'] == 'PASS'
    assert formal['tasks'] == 12 and pilot['tasks'] == 4 and validation['metric_rows'] == 204
    # Reproduce all old-C warning counts as a second check on the paired comparison.
    prior_w = pd.read_csv(old/'evaluation/warning_metrics.csv', dtype={'scope': str, 'seed': str})
    prior_w = prior_w[prior_w.variant == 'announcement_fields']
    keys = ['scope', 'model', 'seed', 'cause', 'horizon_months', 'top_fraction']
    vals = ['checked', 'captured', 'target_events', 'known_non_target', 'unknown', 'repeat_alerts', 'lead3', 'lead6']
    got = w[w.source_state == 'conservative_C'].set_index(keys)[vals].sort_index()
    expected = prior_w.set_index(keys)[vals].sort_index()
    pd.testing.assert_frame_equal(got, expected, check_dtype=False)
    primary = m[(m.seed.isin(['ensemble', 'none'])) & (m.horizon_months == 24) & (m.cause == 'termination')]
    primary = primary.sort_values(['scope', 'model', 'source_state'])
    q = primary[primary.model == 'xai_q2']
    ww = w[(w.seed == 'ensemble') & (w.model == 'xai_q2') & (w.horizon_months == 24)]
    wt = ww[['scope', 'source_state', 'top_fraction', 'checked', 'captured', 'target_events', 'coverage', 'lead3', 'lead6']].copy()
    wt['source_state'] = wt.source_state.map(STATES)
    wt = wt.sort_values(['scope', 'top_fraction', 'source_state'])
    seed = pd.read_csv(out/'seed_summary.csv', dtype={'scope': str})
    seed = seed[(seed.model == 'xai_q2') & (seed.horizon_months == 24) & (seed.cause == 'termination') & (seed.scope != 'pooled')]
    seed = seed[['scope', 'source_state', 'auc_min', 'auc_max', 'ap_min', 'ap_max']].copy()
    seed.source_state = seed.source_state.map(STATES)
    (dest/'RESULTS_ZH.md').write_text('''# 原始年报确认口径重训：2024小幅改善，整体未一致改善

12/12正式模型和4/4数值试运行完成；旧C的12个模型预测只读复用。两年度各使用三个固定种子，先平均预测再计算主指标。唯一变化是按研究者确认，将现有年度财务数值视为原始版本并使用已记录的首次公告日；季度规则、83个输入的顺序、标签、预算及算法均不变。

## 24个月先最终退市：q2主结果

'''+metric_table(q)+'''
2024 AUC 0.752→0.760（+0.008），AP 0.758339→0.771273，Brier 0.335241→0.332689，均有小幅改善；2023 AUC 0.736111→0.715278、AP 0.576727→0.547778，Brier略变差。合并AUC 0.723852→0.718112，AP 0.669576→0.674651，Brier 0.313764→0.312517。合并AP重新计算，不是两年AP平均。不能把本轮概括为全面提升。

**风险低估仍存在。** 新q2在2023平均预测6.22%，实际先退市33.33%；2024平均12.28%，实际44.44%；合并9.58%对39.51%。AUC衡量排序，不能据此把这些概率当成校准准确的绝对风险。没有使用测试发生率校准、平移或裁剪预测。

## 固定核查比例下的事件覆盖

'''+table(wt)+'''
Top20：2023捕获5/12→4/12，2024维持7/20；合并核查17家、12/32→11/32（37.5%→34.375%）。Top25合并核查21家、15/32→14/32。新模型Top20至少提前3个月识别9例、6个月3例，旧版分别10/4例。未知结局和重复预警均为0。年度回顾排序需要收齐该年度队列，不能直接当成实时核查政策；比例取整后核查8/9或9/12家公司。

## 保留全部种子，不能只报告AUC最高的种子

'''+table(seed)+'''
2024新q2的种子AUC为0.732、0.820、0.652；旧版为0.746、0.768、0.718。三种子的波动范围扩大，两种子变差、一种子改善。主结果仍按预先约定使用全部种子平均预测，不能将0.820报告为本轮集成AUC。这些范围不是置信区间，也没有开展完整训练不确定性检验。

这次解除25份测试年报的来源限制，但实际只有4家测试企业输入改变，其他21家仍采用更新的第一季报；训练输入改变38/41家。结果说明这一具体口径调整没有带来稳定的总体增益，不能进一步推断政策变化或某个特征是剩余问题的唯一原因。
''')
    recovery = m[(m.seed.isin(['ensemble', 'none'])) & (m.horizon_months == 24) & (m.cause == 'recovery')]
    secondary = m[(m.seed.isin(['ensemble', 'none'])) & (m.horizon_months == 12) & (m.cause == 'termination')]
    (dest/'PREDICTION_COMPARISON_ZH.md').write_text('''# 与同口径标准RSF的比较

标准RSF为randomForestSRC 3.9.0，此次也重训六个模型，不是最早首次ST预警任务的M2。两者采用共同公司bootstrap和相同数值预算；q2保留二阶路径、共享切点、每树25输入及原生缺失路由，标准RSF不限阶、用训练中位数填补。后端和候选生成并不完全等价。

## 24个月先最终退市

'''+metric_table(primary)+'''
新口径下，q2两年的AUC/AP高于相同输入的标准RSF：2023为0.715278/0.547778对0.621528/0.475832；2024为0.760000/0.771273对0.594000/0.552142。但是q2的Brier两年均略差于标准RSF，合并0.312517对0.308111。排名优势与绝对概率质量必须分别报告。本次两个模型均在2023退化、2024改善，也不能据此断言来源调整对任意模型都有固定方向的效果。

## 24个月先恢复

'''+metric_table(recovery.sort_values(['scope', 'model', 'source_state']))+'''
q2的恢复AUC在2023/2024分别0.676190→0.660317、0.689655→0.672414，Brier也均略变差。新口径标准RSF的恢复预测在这两个年度的AUC/AP/Brier都优于q2。应保留两个结局的全部结果，不能只展示退市改善的一年。

## 补充12个月先最终退市

'''+metric_table(secondary.sort_values(['scope', 'model', 'source_state']))+'''
2024新q2的12个月AUC/AP为0.821121/0.777994，高于旧版0.771552/0.674612；2023为0.664062/0.187976。该期限预先指定为补充，不能因AUC更高而替换24个月主结果；2024平均12个月退市概率仍仅3.32%，观察35.56%。

全部204行指标均有独立AUC、并列分数AP、Brier复算，原12个模型的96行指标与历史值一致。测试的12/24个月结局均已知，IPCW权重均为1，因此Brier等于普通端点Brier。每家公司一行，记录等权与公司等权完全相同。完整各种子、两结局、两期限和两年度/合并表见metrics.csv，所有核查指标见warning_metrics.csv。
''')
    counts = pd.read_csv(out/'cohort_counts_UNCHANGED.csv')
    changes = pd.read_csv(out/'matched_data_audit.csv')
    (dest/'DATA_ALIGNMENT_AUDIT_ZH.md').write_text('''# 年报来源确认与匹配数据核验

按研究者明确确认“视为原始版本”执行：当前来源中的年报值按记录的首次公告日可得，更正日期保留作来源审计信息。该政策对训练与测试一致适用，也适用于增长/变化指标中的实际同比年度组成项；可得日取所有组成项的最晚日期。非年度报告继续采用原来的公告/更正时钟。没有改动原始财务数值，没有制造日期，也没有强制用年报覆盖更近的合格季报。

来源状态为ACCEPTED_BY_RESEARCHER_CONFIRMATION；这是研究者确认接受，未声称已逐值核对原始PDF。原来源、更正日期、旧保守时钟、旧模型和旧结果均保留。

'''+table(changes)+'''
2023/2024测试此前受版本限制的16/9份年报均解除该限制，但只有3/1家公司的实际输入改变，其余21家仍选择较新的第一季报。训练输入分别38/41家公司改变；训练中7份首次公告本身晚于起点或不可得的年度报告仍不能使用，这不是再次施加更正版本限制。

## 样本与时钟

'''+table(counts)+'''
总体为全非金融行业、观察历史中的首次财务/审计ST入组；起点为公告后下一交易日开盘前，事件为先恢复A或最终终止上市X的公告日。2023训练2012—2022入组且信息严格早于2023-01-01；2024训练2012—2023入组且信息严格早于2024-01-01。跟踪24个日历月为主、12个月为补充，原来源覆盖截至2026-09-03。先恢复再退市仍作为先恢复，不改成第二个退市事件。

两折训练309/345家、24个月先退市14/26例；合计654条折内记录，去重345家、26个独立先退市训练事件。12次正式拟合共3924条重复计算记录，不是3924个独立样本。测试36/45家、12/20例先退市，共81家、32例先退市、31例先恢复、18例两年内无首次事件。训练删失仍为删失，未填成正常。各折训练测试无公司交叉；2023测试公司进入2024历史训练时只使用当年截止前可得结局。

新旧C的sample_id、company group、origin、time、event、observed end、标签可用性及所有非X数组逐项一致。市场输入和新增警示/审计字段不变；83个字段名称及顺序一致。所有源文件、训练/预测输入、代码实现与旧预测文件均核验SHA256。没有新增样本、目标、警示字段或政策变量。

2023/2024结局已被多轮分析接触，本轮是POST_EXPOSURE_EXPLORATORY；不能宣称新盲测。没有评价2025入组队列。来源确认与本轮配对检查不取消历史样本覆盖、事件界定和行业分类的继承限制。
''')
    cc = pd.read_csv(out/'complexity_comparison.csv')
    ctab = cc[['test_year', 'source_state', 'main_functions', 'interaction_functions', 'spatial_table_cells',
               'scalar_table_cells', 'low_support_cell_fraction', 'zero_support_cell_fraction']].copy()
    ctab.source_state = ctab.source_state.map(STATES)
    stability = pd.read_csv(out/'seed_stability.csv')
    st = stability[stability.cause == 'termination'].groupby(['test_year', 'source_state'])[
        ['main_jaccard', 'pair_jaccard', 'median_function_correlation']].mean().reset_index()
    st.source_state = st.source_state.map(STATES)
    leaf = s[s.model == 'xai_q2'].groupby(['test_year', 'source_state'])['target_unsupported_leaf_fraction'].mean().reset_index()
    leaf.source_state = leaf.source_state.map(STATES)
    (dest/'INTERPRETABILITY_COMPARISON_ZH.md').write_text('''# 解释仍精确，复杂度和可靠性未同步改善

这里评价实际用于主结果的三种子96树集成，完整保留主效应、二元交互、缺失分区和两个结局。贡献在CIF尺度精确相加，不是log hazard，不是因果影响。未删除低贡献函数，未引入SHAP替代或三阶残差。

'''+table(ctab)+f'''
最大完整重构误差为{audit['max_ensemble_all_native_time_reconstruction_error']:.3e}，保存重载同样通过。2023空间格数3869→3978（+2.8%），2024为4068→4294（+5.6%）；主效应62→65、65→68；交互144→143、146→152。因此本轮不能称解释变得更简单。scalar_table_cells还包括原生时间和原因维度，不等同空间格数或独立参数自由度。

低支持格约52.5%/48.6%，零支持格均约21.7%；支持门槛为10家训练企业。相关输入和互斥审计哑变量使乘积参考产生一些未观察或不可实现的组合。精确函数分解不消除这种数据支持不足，也不证明统计可辨识性。

## 跨种子稳定性

'''+table(st)+'''
2024共同函数相关性改善，但交互选择Jaccard从0.0922降至0.0807；2023约0.073，仍低。主效应选择Jaccard两年均降低。相关性仅针对共同、非恒定的函数，在各自来源口径的训练参考样本上评价；输入及中心化参考变化，不是同一函数形状的因果比较。三个种子的描述不能替代训练重抽样推断。

## 缺少退市支持的叶节点

'''+table(leaf)+'''
新q2约64.3%/48.8%的叶节点没有退市公司支持。虽然整体能排序，很多局部风险及交互仍缺少事件证据。新增警示字段中的*ST指示依然没有被用于两折各96棵树的分裂，不能把本次变化归因于这一字段。全部字段使用次数、主效应和交互分摊的平均绝对贡献及种子范围已交付；企业记录、实际阈值和完整参数仍留私有目录。

这次没有修改可辨识性惩罚或解释复杂度约束，不能由结果声称其独立贡献得到增强。
''')
    fitted = s[s.source_state == 'accepted_annual_C']
    fitting = plan['code_commit']
    (dest/'TRAINING_DIAGNOSTICS_ZH.md').write_text(f'''# 训练与核验回执

冻结拟合代码：`{fitting}`。训练前完整120项测试通过、0跳过；交付前再次完整通过。独立R检查包含在内。唯一警告是既有EBM缺失值可视化提示。本轮没有修改crforest核心、旧worker或R实现；逐文件核验其与旧C拟合提交dc8c7f49b62750b2ae87058e293b935b0b1fcd34完全相同。

4/4数值试运行、12/12正式训练完成，无失败模型、发布失败或选择性重跑。Slurm作业{pilot['job']}与{formal['job']}；试运行4个短任务由oliver取走，正式分配{formal['by_host']}。使用现有p/both拓扑布局：oliver P核、lemon物理核，单任务单线程、4GiB地址空间限制，NFS输入及校验回传。不硬编码CPU，不重复训练凑利用率，不宣称实测最优布局。

正式任务池计算及回传用时{formal['seconds']:.3f}秒，不含数据准备、代码审查及完整解释审计。单任务{s[s.source_state == 'accepted_annual_C'].seconds.min():.3f}—{fitted.seconds.max():.3f}秒。Python父进程最高RSS {fitted.peak_rss_MiB.max():.3f}MiB，R子进程单独最高{fitted.child_peak_rss_MiB.max():.3f}MiB；不是同时进程树峰值的求和测量。样本仅309/345家、每个模型32棵深度受限树，因此拟合很快。模型哈希、候选评分次数、节点/叶数、阈值数量、支持和时间均在CSV中，不只报告训练成功。

独立重放6个q2模型的{audit['nodes_checked']}个节点、{audit['selected_scores_checked']}个已选分裂和{audit['leaf_aj_checked']}个叶内AJ；分裂分数最大差{audit['max_score_error']:.3e}，AJ差{audit['max_leaf_aj_error']:.3e}。6个标准RSF的共同bootstrap及训练中位数全部核验。原生时间CIF非负、时间单调、原因和不超过1（机器精度内），路径阶数不超过2，模型重载误差0。

两个96树完整集成在训练和测试全部公司、所有原生时刻重构并重载，最大误差{audit['max_ensemble_all_native_time_reconstruction_error']:.3e}。204行端点指标独立计算，旧C96行指标和全部核查计数与历史记录精确对齐。森林没有梯度求解器，Hessian秩、迭代收敛gap不适用。本轮没有裁剪、校准、叶收缩、改预算、筛种子或附加搜索。
''')
    (dest/'METHOD_INNOVATION_DECISION_ZH.md').write_text('''# 判断：INCONCLUSIVE；年报口径调整没有显示稳定总体增益

本轮回答的是“按研究者接受的原始年报口径重训C组，能否改善已有结果”。执行成功，q2在2024的排序/AP小幅改善，2023变差，合并Top20少捕获1例。两年的绝对退市概率仍明显低估，2024种子波动扩大。来源调整没有显示跨年度、一致指标的改善，不能宣布模型或数据问题已经解决。

整体框架的价值仍在：有限样本上可以实际训练并生成有效竞争风险曲线，保留完整精确的低阶函数解释；本轮q2退市AUC/AP仍高于同输入标准RSF。局限同样明确：标准RSF的Brier及恢复预测较好，q2完整表略变大、交互选择仍不稳定。这些是具体配置和已观察测试样本的结果，不是对RSF家族的普遍优越性证明。

本轮没有新算法、没有新增可辨识性惩罚，也没有检验惩罚的独立贡献。数学精确重构不等同统计可辨识性或解释可靠。来源口径的更新不能作为方法创新。

风险低估在未经校准的原始森林中就存在，本次没有近零斜率校准器，因此不能解释成“校准丢掉了学习内容”。缺少事件支持和训练/测试发生率差异仍在；本轮不能把残余差距因果归于政策变化。研究者已接受25份年报作为原始版本，这一确认已实际执行，不能继续把它列为本轮未执行的阻塞条件；同时也不声称独立原始PDF核验。

来源政策不根据本轮性能反向选择，旧C与新C均保留。按预定有限范围结束，不追加种子、校准、权重、叶收缩、2025评价或超参数搜索。继续宣称方法成功或校准可用，都需要本轮以外的独立证据。
''')
    receipt = dict(status='COMPLETE', decision='INCONCLUSIVE',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_status='ACCEPTED_BY_RESEARCHER_CONFIRMATION', evaluation_status='POST_EXPOSURE_EXPLORATORY',
        private_run=str(run), fitting_commit=fitting, repository='Egon749/XAI-RSF',
        branch='research/xai-rsf-post-st-prognosis-v1', formal_fits=12, pilots=4, reused_models=12,
        failed_fits=0, failed_publications=0, tests_passed=120, tests_skipped=0,
        training_fold_rows=654, training_company_union=345, training_termination_union=26,
        repeated_formal_fit_rows=3924, test_companies=81, test_terminations=32, test_recoveries=31,
        input_columns=83, training_input_changed_companies=[38,41], test_input_changed_companies=[3,1],
        changed_labels=0, annual_reports_accepted_by_confirmation=True,
        independent_original_PDF_verification=False, algorithm_changed=False,
        calibration_fits=0, leaf_shrinkage_fits=0, new2025_evaluation=False,
        formal_by_host=formal['by_host'], slurm_jobs=[pilot['job'],formal['job']],
        main_horizon_months=24, secondary_horizon_months=12, metric_rows=204,
        old_C_metric_rows_reproduced=96, old_C_warning_rows_reproduced=len(got),
        maximum_reconstruction_error=audit['max_ensemble_all_native_time_reconstruction_error'],
        result_sha256={str(p.relative_to(dest)):sha(p) for p in out.iterdir() if p.is_file()},
        code_sha256={p.name:sha(p) for p in dest.glob('*.py')},
        document_sha256={p.name:sha(p) for p in dest.glob('*.md')},
        protocol_sha256=sha(dest/'frozen_protocol.json'))
    write(dest/'execution_receipt.json', receipt)
    write(run/'execution_receipt.json', receipt)
    print(json.dumps({k:receipt[k] for k in ['status','decision','formal_fits','pilots','test_companies','test_terminations']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--dest', type=Path, required=True)
    a = p.parse_args()
    deliver(a.run, a.dest)
