"""Publish aggregate evidence only; never fit, select, or change a model."""
from pathlib import Path
import argparse, datetime, json, shutil
import numpy as np
import pandas as pd
from crforest import CompetingRiskForest
from experiments.post_st_ablation_v1.verification import replay
from experiments.post_st_leaf_shallow_v1.analyze import artifacts
from .common import *


def table(frame):
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return f'{v:.6f}' if np.isfinite(v) else 'NA'
        return str(v)
    return '\n'.join(['| ' + ' | '.join(frame.columns) + ' |',
        '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |'] +
        ['| ' + ' | '.join(map(cell, row)) + ' |' for row in frame.itertuples(index=False, name=None)])


def deliver(run, repo):
    public = repo / 'experiments' / NAME
    validation = json.loads((run/'audit/validation.json').read_text())
    assert validation['status'] == 'PASS'
    gate = json.loads((run/'audit/formal_gate.json').read_text())
    frozen = json.loads((run/'audit/freeze.json').read_text())
    ref = json.loads((repo/'CURRENT_POST_ST_MODEL.json').read_text())
    preservation = []
    for year, arm, seed, folder in artifacts(HISTORY):
        if arm != 'shallow_parent':
            continue
        entry = next(r for r in ref['saved_models'] if r['test_year'] == year and r['seed'] == int(seed))
        for name, key in [('forest_PRIVATE.json', 'forest_sha256'),
                          ('predictions_PRIVATE.npz', 'predictions_sha256'),
                          ('functions_PRIVATE.npz', 'functions_sha256')]:
            assert sha(folder/name) == entry[key]
            preservation.append(dict(year=year, seed=seed, artifact=name, sha256=entry[key], unchanged=True))
    assert len(preservation) == 18 and sha(repo/'CURRENT_POST_ST_MODEL.json') == REFERENCE_SHA
    pd.DataFrame(preservation).to_csv(run/'reports/reference_preservation.csv', index=False)

    # Saved old forests replay their original company bootstrap; no tree is fitted.
    support = []
    tr = arrays(run/'input/reference_train_PRIVATE.npz')
    for i, seed in enumerate(SEEDS):
        reference = CompetingRiskForest.load(run/'input/reference'/str(seed)/'forest_PRIVATE.json')
        _, _, _, old_leaves, check = replay(reference, tr)
        assert max(check.values()) <= 1e-10
        fresh = run/'results/formal'/str(gate['job'])/f'task-{i}/output'
        for arm, leaves in [('reference83', pd.DataFrame(old_leaves)),
                            ('drop9', pd.read_csv(fresh/'leaf_support_PRIVATE.csv'))]:
            support.append(dict(arm=arm, seed=seed, leaves=len(leaves),
                min_unique_companies=int(leaves.unique_companies.min()),
                median_unique_companies=float(leaves.unique_companies.median()),
                zero_target_leaves=int((leaves.target_companies == 0).sum()),
                borrowed_positive_target_leaves=int(leaves.borrowed_positive_target.sum())))
    pd.DataFrame(support).to_csv(run/'reports/leaf_support_summary.csv', index=False)

    risks = arrays(run/'private/risks_PRIVATE.npz')
    stability = []
    for arm in ['reference83', 'drop9']:
        for horizon in [12, 24]:
            p = np.stack([risks[f'{arm}_{s}_{horizon}'][:, 0] for s in SEEDS])
            sd = p.std(axis=0, ddof=1)
            stability.append(dict(arm=arm, horizon_months=horizon,
                company_seed_prediction_sd_mean=float(sd.mean()),
                company_seed_prediction_sd_max=float(sd.max())))
    pd.DataFrame(stability).to_csv(run/'reports/prediction_seed_variation.csv', index=False)

    m = pd.read_csv(run/'reports/metrics.csv', dtype={'seed': str})
    w = pd.read_csv(run/'reports/warnings.csv', dtype={'seed': str})
    cv = pd.read_csv(run/'reports/function_validation.csv', dtype={'seed': str})
    diff = pd.read_csv(run/'reports/paired_differences.csv')
    fit = pd.read_csv(run/'reports/execution_tasks.csv')
    main = m[(m.seed == 'ensemble') & (m.cause == 1)].merge(
        w[['arm', 'seed', 'horizon_months', 'checked', 'captured', 'target_events', 'coverage']],
        on=['arm', 'seed', 'horizon_months'])
    columns = ['arm', 'auc', 'ap', 'brier', 'risk_mean', 'captured', 'target_events', 'coverage']
    complex_cols = ['arm', 'seed', 'main_functions', 'interaction_functions',
                    'spatial_table_cells', 'scalar_table_cells', 'distinct_thresholds']
    main.to_csv(run/'reports/ensemble_comparison.csv', index=False)
    cv[complex_cols].to_csv(run/'reports/function_complexity.csv', index=False)
    names = pd.read_csv(run/'reports/feature_map.csv')
    removed = names[names.removed][['original_index', 'name', 'label']]
    reduction = 1 - cv[(cv.arm == 'drop9') & (cv.seed == 'ensemble')].spatial_table_cells.iloc[0] / cv[(cv.arm == 'reference83') & (cv.seed == 'ensemble')].spatial_table_cells.iloc[0]
    result = f'''# 删除九项疑似冗余变量：24个月主结果下降，保留原模型

本轮实际新增3次森林拟合（种子42、101、202，各32树），派生3个直接父节点AJ强度10模型，复用3个原83列参考森林；训练任务失败0，成功3/3。不是用历史指标充当新实验。全部预设种子均报告，没有补种子、调参或校准。

训练使用2012—2023年的同一冻结345家公司，含2023年入组36家；截至2024-01-01前可用的最终退市26例、恢复198例。原参考模型本来就包含这些2023年记录，本轮增加的是删变量对照。2024评价45家，24个月先最终退市20例，12个月16例。两臂先逐公司平均3个种子预测，再计算指标。

## 24个日历月，先最终退市（主要比较）

{table(main[main.horizon_months == 24][columns])}

AUC下降0.076，AP下降0.088399，Brier增加0.003500（越低越好）。Top20固定核查9家，捕获由8/20降为7/20，覆盖由40%降为35%；其中已知其他结局由1家增为2家，未知结局均0。这里按整年入组队列回顾排序，不是在线核查政策。

平均预测风险由11.581%略升至11.834%，但观察发生率为44.444%。这没有解决整体低估，且Brier变差；不能把平均预测值变高当作校准改善。

## 12个日历月，先最终退市（预先声明的次要比较）

{table(main[main.horizon_months == 12][columns])}

12个月AUC与AP有改善，Brier仍略差。必须保留这一反向结果：删除的影响依赖时点，不能说九项变量在所有任务上都有用，也不能把12个月的提升写成24个月主目标已改善。恢复原因及全部单种子结果见results/metrics.csv。

## 配对不确定性与种子差异

以下为“删除后减原模型”，固定两组预测，使用相同公司抽样索引重抽样1000次；独立分析种子202610319。

{table(diff[['metric', 'difference', 'lower95', 'upper95', 'undefined_bootstraps']])}

区间仅描述固定模型下这45家测试公司的波动，不包含训练样本、变量筛选和模型选择不确定性，也不能因为多个区间中某一项不跨零而作确认性成功/失败宣称。AUC和Brier的区间偏向退化，AP与捕获数差异仍不确定。

{table(m[(m.cause == 1) & (m.horizon_months == 24) & (m.seed != 'ensemble')][['arm', 'seed', 'auc', 'ap', 'brier']])}

三个种子的24个月AUC/AP均下降；Brier两差一好。完整跨种子预测波动见prediction_seed_variation.csv；三个种子不等于三个独立验证队列。

## 表示简化与解释核验

{table(cv[cv.seed == 'ensemble'][complex_cols])}

96树合并预测器的非零主效应65→58、二元交互146→132，空间单元3263→2933，减少{reduction:.2%}。空间单元包含每维单独缺失状态，不含时间和原因维；标量表量另乘140个原生时间点和2个原因。这里是完整未合并网格规模，实际保存的是等价的因子化函数表示；大小见function_validation.csv。

所有原/新单种子、raw/shrunk新模型及两组完整平均预测器，在345训练+45评价记录、全部原生时点和两个原因上完整验证。最大重构误差{validation['maximum_explanation_error']:.3e}，小于1e-10；CIF概率约束、保存重载、直接raw父节点AJ和独立公司数收缩权重核验通过。没有裁剪预测或删除解释项。

## 解释本轮证据

“疑似冗余”是此前小样本、相关变量置换诊断的候选结论，不是可安全删除的证明。删除后还会改变均衡覆盖调度：83→74列使调度块数4→3，每树仍25变量，但保留变量的入选比例25/83→25/74，具体变量组合和后续树结构也改变。公司bootstrap已严格配对，保留值与共享切点完全相同；不能进一步声称候选集合完全相同，或把性能变化全部归因于某一个被删变量。

本轮发现解释表变小，但24个月排序与捕获下降；不支持整体删除这九项后替换参考模型。12个月收益不自动改变预先固定的24个月目标。本次结果为已暴露2024队列上的探索性比较，财务原始版本PIT限制继承，证据状态PROVISIONAL。当前原模型继续保留，本轮到此结束，不追加筛选或搜索。
'''
    (public/'RESULTS_ZH.md').write_text(result)
    data = f'''# 数据与配对核验

原输入来自post_st_leaf_shallow_v1的2024冻结折，旧conservative C83列。只删除下表9列，其余74列保持原顺序、原值与缺失状态；train/eval的全部非X数组逐项相等。原始训练标签、删失、时间起点和公司集合不变。

{table(removed)}

名单来源：post_st_feature_permutation_v1/results/feature_classification.csv的outer_cutoff=2024且subtype=suspected_redundant；完整名单和源哈希在冻结协议中。没有用本轮2024指标进一步选择字段。

训练345个独立公司，26个先最终退市、198个先恢复，其余删失；入组年份2012—2023，2023年36家。信息截止严格早于2024-01-01，近期公司只贡献截止前的部分随访，不虚构24个月标签。2024评价45家公司，与训练公司不重叠。原点沿用入组公告后的下一交易日开盘前；事件为先最终退市/先恢复的公告日，12/24个日历月评价。数据的金融/市场可得日期均不晚于原点；原年报披露版本并未因此获得新的独立PIT认证。

保留列的训练共享分位数切点逐列完全一致。96棵树的公司bootstrap及重复次数哈希逐棵配对；新拟合的实际inbag重复次数也逐棵验证，旧森林通过原节点重放验证。具体候选变量集合随输入维数变化，见feature_eligibility.csv和tree_pairing.csv。

测试标签在冻结时存于独立evaluation_input目录，不在训练作业DATA_DIR内。训练worker仅接触评价X和sample_id。全部3次拟合和发布成功后，统一评价器才加载测试标签。本轮未接触2025数据。

当前参考JSON及2023/2024六个旧森林、六份预测、六份函数文件，共18个工件SHA256全部未变。输入源哈希与冻结输入哈希均在运行后复核。公司记录、模型参数、切点和预测保留在私有目录：`{run}`。
'''
    (public/'DATA_ALIGNMENT_AUDIT_ZH.md').write_text(data)
    diagnostics = f'''# 执行与数值核验

拟合前全套测试189通过（含独立R检查）；1条既有EBM缺失值可视化警告，与本次森林无关。唯一核心修改：mtry是候选数上限，允许名义75大于剩余74列，继续原有min(mtry,eligible)计算；实际每树25列。测试已验证与mtry74时树、预测、候选次数和公司抽样完全相同，旧有效配置保持行为兼容。没有改变数学目标、分裂规则、精度容差或预测值。

冻结提交：`{frozen['commit']}`，冻结UTC时间：{frozen['utc']}。原来源提交：31cc93d4bcad16d1e680a88cba5a4b186debcfe2。完整配置差异表见results/configuration_diff.csv。

Slurm作业{gate['job']}启用oliver/lemon两节点池。本批只有3个短任务，实际调度到{json.dumps(gate['by_host'])}；如实记录，不宣称lemon执行了拟合，也没有为使用另一台机器重复训练。每任务单线程、4GiB地址空间上限，服从调度器亲和性。计算及结果发布均验证成功，失败0。

{table(fit[['seed', 'host', 'fit_seconds', 'shrink_seconds', 'seconds', 'peak_rss_MiB', 'actual_scored_candidates']])}

池内总墙钟{gate['seconds']:.3f}秒；上表seconds还包含函数编译、数值核验及节点重放。不要把各任务耗时相加当作并行墙钟，也不与历史不同负载运行直接比较速度。

{table(pd.DataFrame(support))}

收缩使用本森林raw直接父节点AJ，权重为叶内独立inbag公司数/(该公司数+10)，对全部时间和原因相同；不存在递归收缩或再次收缩。借入风险不会创造真实退市事件支持。raw与shrunk都完整保存。

{table(cv[['arm', 'seed', 'full_reconstruction', 'explicit_component_sum', 'function_reload', 'forest_reload', 'verified_rows', 'verified_times']])}

概率约束与解释在每组390条记录、142个核验时点（含140个原生时点）及两个原因通过。原参考16行历史指标重新计算一致；本轮评价和独立指标实现一致。24个月与12个月的测试终点均已知，IPCW权重恰为1；仍沿用竞争风险评价器，未知结局没有改成正常。“终点已知”包括窗口内先恢复或先退市，不表示每家公司都随访满24个月。

12/24个月和两原因的完整结果、全部种子、1000个配对差值均公开保存；无失败种子丢弃。公开目录不包含逐公司预测或函数参数。
'''
    (public/'TRAINING_DIAGNOSTICS_ZH.md').write_text(diagnostics)
    methods = '''# 论文方法与结果段落

在包含2012—2023年入组公司的2024预测训练集中，我们根据仅使用历史信息的置换诊断预先确定九项疑似冗余输入，构造83列与74列输入的有限匹配比较。模型均采用q=2、最大深度4、共享分位数切点、每树均衡选择25个变量及强度10的直接父节点Aalen–Johansen收缩，种子为42、101和202，每种子32树。其余配置、随访定义和公司bootstrap保持一致。每年度逐公司平均三个种子的CIF后评价；主要终点为24个日历月内先最终退市，12个月为次要终点。原83列模型直接复用，新74列模型实际拟合三次。由于输入维数减少会改变均衡调度的变量组合，该比较反映整个固定删变量策略的效果，不能将差异解释为单个变量的因果贡献。

2024年45家公司中20家在24个月内先最终退市。删除九项后AUC由0.770下降至0.694，AP由0.782032下降至0.693633，IPCW Brier由0.339557增至0.343057；前20%核查9家公司，捕获从8/20降至7/20。合并预测器的主效应/二元交互由65/146减至58/132，空间单元由3263减至2933，完整函数重构误差低于1e-10。12个月AUC/AP相反有所提高，但Brier未改善。结果不支持将历史诊断中的疑似冗余直接视为安全删除依据，且仍存在明显概率低估。该实验为已暴露测试队列上的探索性机制分析，固定预测的配对公司bootstrap区间不涵盖模型训练与选择不确定性。原研究参考模型保留。
'''
    (public/'METHODS_RESULTS_ZH.md').write_text(methods)
    dest = public/'results'; dest.mkdir(exist_ok=True)
    for p in sorted((run/'reports').glob('*.csv')):
        shutil.copy2(p, dest/p.name)
    audits = public/'audit'; audits.mkdir(exist_ok=True)
    for name in ['freeze.json', 'source_hashes.json', 'data_alignment.json',
                 'formal_gate.json', 'validation.json', 'full_tests_pre_fit.log']:
        shutil.copy2(run/'audit'/name, audits/name)
    receipt = dict(status='COMPLETE', utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        private_run=str(run), frozen_commit=frozen['commit'], source_commit='31cc93d4bcad16d1e680a88cba5a4b186debcfe2',
        forest_fits=3, derived_shrunk_models=3, reused_reference_forests=3, failed_tasks=0,
        new_calibration_fits=0, pilot_fits=0, slurm_job=gate['job'], actual_execution_by_host=gate['by_host'],
        reference_preserved_files=18, current_reference_sha256=REFERENCE_SHA,
        training_companies=345, training_termination_events=26, training_2023_companies=36,
        evaluation_companies=45, evaluation_24m_termination_events=20, current_model_replaced=False,
        validation=validation, tests='189 passed, 1 existing EBM visualization warning; independent R checks enabled',
        evidence='PROVISIONAL/post-exposure exploratory', decision='KEEP_REFERENCE; no additional search',
        aggregate_manifest={str(p.relative_to(public)): sha(p) for p in sorted(dest.glob('*'))})
    write(public/'execution_receipt.json', receipt)
    print(json.dumps(dict(status='COMPLETE', public=str(public), reference_files_preserved=18)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--run', type=Path, required=True); a = p.parse_args()
    deliver(a.run, Path(__file__).resolve().parents[2])
