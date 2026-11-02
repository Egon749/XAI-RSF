"""Publish aggregates from explicit successful analysis attempts, never fits."""
from pathlib import Path
import argparse
import json
import shutil
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
from .analyze import PUBLIC, REPO, digest, write_json


def table(df, digits=4):
    def text(v):
        if pd.isna(v): return 'NA'
        if isinstance(v, (float, np.floating)): return f'{v:.{digits}f}'
        return str(v)
    return ('| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'
            +'\n'.join('| '+' | '.join(text(v) for v in row)+' |' for row in df.itertuples(index=False, name=None)))


def plot_level(series, destination):
    font = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    fontManager.addfont(font)
    plt.rcParams.update({'font.family': FontProperties(fname=font).get_name(), 'axes.unicode_minus': False,
                         'font.size': 11, 'pdf.fonttype': 3})
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.2), sharex=True)
    q = series[(series.arm == 'baseline') & (series.version == 'shrunk')]
    for ci, cause in enumerate([1, 2]):
        for hi, h in enumerate([12, 24]):
            ax = axes[ci, hi]
            d = q[(q.cause == cause) & (q.months == h)].sort_values('year')
            for key, label, color, marker, style in [
                ('observed', '评价队列观察发生率', '#BA3D35', 'o', '-'),
                ('mean_predicted', '原模型平均预测', '#245E9C', 's', '-'),
                ('training_aj', '截止日前训练 AJ', '#646464', '^', '--')]:
                ax.plot(d.year, 100*d[key], marker=marker, color=color, linestyle=style, label=label, lw=2)
            for r in d.itertuples():
                if r.supported:
                    offset = -17 if cause == 2 and h == 12 and r.year == 2024 else 9
                    ax.annotate(f'{100*r.observed:.1f}%', (r.year, 100*r.observed),
                                xytext=(0, offset), textcoords='offset points', ha='center', color='#963329', fontsize=10)
            ax.set_title(f'{"先最终退市" if cause == 1 else "先恢复"} · {h} 个月', loc='left', fontsize=13)
            ax.set_ylim(-3, 80)
            ax.set_xlim(2022.8, 2025.2)
            ax.set_xticks([2023, 2024, 2025])
            ax.set_ylabel('累计发生概率（%）')
            ax.grid(axis='y', alpha=.2)
            ax.spines[['top', 'right']].set_visible(False)
            if h == 24:
                ax.annotate('2025：观察不足，性能 NA', (2025, 75), ha='right', fontsize=10, color='#555555')
            if ci == 1: ax.set_xlabel('评价队列入组年（同年 1 月 1 日信息截止）')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .95), ncol=3, frameon=False)
    fig.suptitle('训练发生水平、原模型预测与评价队列观察水平', y=.995, fontsize=16)
    fig.text(.07, .012, '只读保存工件；三种子平均后评价。训练 AJ 使用日历月时钟。2023–2025 已暴露，均为探索性结果。\n'
             '连线仅辅助阅读；不是连续时间拟合趋势。2025 年 24 个月只显示原有预测与训练 AJ，不补造观察率。', fontsize=9.5, color='#444444')
    fig.tight_layout(rect=[0, .06, 1, .91])
    for ext in ['png', 'pdf']:
        path = destination/f'RISK_LEVEL_TIME_SERIES.{ext}'
        if path.exists(): raise FileExistsError(path)
        fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def build(run, metrics, drift):
    for source in [metrics, drift]:
        receipt = json.loads((source/'stage_receipt.json').read_text())
        assert receipt['status'] == 'PASS' and receipt['forest_fits'] == receipt['calibrator_fits'] == 0
        for name, h in receipt['output_hashes'].items():
            assert digest(source/name) == h
    names_m = ['STRATIFIED_METRICS', 'STRATIFIED_BOOTSTRAP_95', 'OBSERVED_EXPECTED',
               'TOP20_LIFT', 'INDEPENDENT_AUC_CHECKS', 'YEAR_PAIR_SUPPORT']
    names_d = ['NULL_AJ_BASELINE', 'CALENDAR_INCIDENCE', 'COVARIATE_SHIFT_SUMMARY',
               'DELISTING_REASON_SUMMARY', 'TRAINING_AJ', 'LEVEL_TIME_SERIES',
               'COHORT_CALENDAR_CROSS', 'COHORT_AJ', 'COVARIATE_SHIFT_TOP']
    for source, names in [(metrics, names_m), (drift, names_d)]:
        for name in names:
            target = PUBLIC/f'{name}.csv'
            assert not target.exists(), 'No overwrite of published output'
            shutil.copy2(source/target.name, target)
    for source, name in [(metrics, 'metric_validation.json'), (metrics, 'ACCEPTANCE_GATE.json'), (drift, 'drift_validation.json')]:
        assert not (PUBLIC/name).exists()
        shutil.copy2(source/name, PUBLIC/name)
    m = pd.read_csv(metrics/'STRATIFIED_METRICS.csv')
    decomposition = m[m.scope.eq('pooled')].copy()
    decomposition['all_pairs'] = decomposition.events*(decomposition.companies-decomposition.events)
    decomposition['cross_year_pairs'] = decomposition.all_pairs-decomposition.within_pairs
    decomposition['cross_year_auc'] = ((decomposition.pooled_auc*decomposition.all_pairs
        - decomposition.within_year_auc*decomposition.within_pairs)/decomposition.cross_year_pairs)
    np.testing.assert_allclose(
        (decomposition.within_year_auc*decomposition.within_pairs+decomposition.cross_year_auc*decomposition.cross_year_pairs)/decomposition.all_pairs,
        decomposition.pooled_auc, atol=1e-14, rtol=0)
    decomposition[['months','cause','version','arm','all_pairs','within_pairs','cross_year_pairs',
                   'within_year_auc','cross_year_auc','pooled_auc']].to_csv(PUBLIC/'PAIR_DECOMPOSITION.csv',index=False)
    b = pd.read_csv(metrics/'STRATIFIED_BOOTSTRAP_95.csv')
    oe = pd.read_csv(metrics/'OBSERVED_EXPECTED.csv')
    alert = pd.read_csv(metrics/'TOP20_LIFT.csv')
    level = pd.read_csv(drift/'LEVEL_TIME_SERIES.csv')
    null = pd.read_csv(drift/'NULL_AJ_BASELINE.csv')
    shift = pd.read_csv(drift/'COVARIATE_SHIFT_SUMMARY.csv')
    cal = pd.read_csv(drift/'CALENDAR_INCIDENCE.csv')
    reason = pd.read_csv(drift/'DELISTING_REASON_SUMMARY.csv')
    plot_level(level, PUBLIC)
    p2020 = 'https://www.sse.com.cn/lawandrules/sselawsrules/repeal/rules/c/c_20220107_5679292.shtml'
    p2024 = 'https://www.sse.com.cn/lawandrules/sselawsrules2025/repeal/rules/c/c_20240430_10785259.shtml'
    policy = pd.DataFrame([
        dict(date='2020-12-31', rule='上交所2020退市修订发布实施', applicability='存在旧暂停上市等过渡安排；财务标准FY2020起算', url=p2020),
        dict(date='2021-07-01', rule='2020修订的市值退市指标施行', applicability='上交所通知明确延后实施，不能把所有条款统一设为2020-12-31', url=p2020),
        dict(date='2024-04-30', rule='上交所2024修订发布实施', applicability='财务新条款以FY2024年报为首个适用年度；FY2023仍按旧规则及衔接安排', url=p2024),
        dict(date='2024-10-30', rule='2024新市值退市指标施行', applicability='保留跨实施日期的连续低市值旧规则衔接；不等同低于1元价格退市', url=p2024),
        dict(date='2025-01-01', rule='相关其他风险警示新条款施行', applicability='其他风险警示不是已经最终退市；只列政策背景，不据此修改事件', url=p2024),
    ])
    policy.to_csv(PUBLIC/'POLICY_TIMELINE.csv', index=False)
    parallel = cal[['calendar_year', 'termination_events', 'recovery_events', 'person_months',
                    'termination_rate_per100_person_months', 'recovery_rate_per100_person_months']].copy()
    parallel['official_policy_dates'] = parallel.calendar_year.map(
        lambda y: '; '.join(policy.loc[policy.date.str.startswith(str(y)), 'date'])).fillna('')
    parallel['interpretation'] = 'temporal alignment only; no causal claim;2026partial;fixed cohort first events'
    parallel.to_csv(PUBLIC/'POLICY_EVENT_ALIGNMENT.csv', index=False)
    # Publication hashes identify inputs without exporting any company-level data.
    hashes = {}
    for source in [metrics, drift]:
        for path, h in json.loads((source/'INPUT_HASHES_PRIVATE.json').read_text()).items():
            if path in hashes: assert hashes[path] == h
            hashes[path] = h
    pd.DataFrame([dict(path=p, sha256=h) for p, h in sorted(hashes.items())]).to_csv(PUBLIC/'INPUT_HASHES.csv', index=False)
    test_log = run/'logs/pytest_full_v1.log'
    assert test_log.exists() and 'failed' not in test_log.read_text().lower()
    assert 'passed' in test_log.read_text()
    shutil.copy2(test_log, PUBLIC/'TESTS_full.log')
    methods = m[(m.scope == 'pooled') & (m.cause == 1) & (m.version == 'shrunk')]
    contrast = b[(b.scope == 'pooled') & (b.cause == 1) & (b.version == 'shrunk') &
                 (b.kind == 'paired_difference') & (b.metric == 'within_year_auc')]
    baseoe = oe[(oe.arm == 'baseline') & (oe.version == 'shrunk') & (oe.cause == 1) & (oe.scope != 'pooled')]
    basenull = null[(null.arm == 'baseline') & (null.version == 'shrunk') & (null.cause == 1)]
    baselevel = level[(level.arm == 'baseline') & (level.version == 'shrunk') & (level.cause == 1)].copy()
    for k in ['training_aj', 'mean_predicted', 'observed']: baselevel[k] *= 100
    undef = b.groupby(['scope', 'months', 'cause']).undefined_repeats.max().reset_index()
    report = f'''# 退市预测报告口径与水平漂移：只读分析

**结论：原来的 pooled AUC 数值计算没有错误，但不能把它的上升表述为“年内筛查排序改善”。** 原因平方分数的12个月 pooled AUC由0.6451升到0.6613，年内AUC却由0.7584降到0.7019；两者同时加入也反向。单独支持保护的年内AUC在12/24个月均略升，但95%差值区间均跨0，尚不足以宣布稳定收益或替换参考模型。

**低估更符合“2023—2024发生水平集中上升、历史训练水平反应较慢”的描述。** 2025年12个月观察率回落，而训练AJ和模型均值继续上升，低估明显减轻。日历年风险在2024年集中升高，入组队列风险也变化；市场输入分布有明显漂移，但这些描述不足以在队列、日历环境和协变量之间作独立因果归因。

本轮森林拟合0、校准器拟合0、其他新研究模型0、倍率/参数搜索0。只读取现有保存工件；训练AJ是用户要求的无变量发生率统计。原模型、CURRENT指针、冻结协议和所有旧预测/报告未修改。2023—2025全部为**探索性**，继承财报原始披露版本PROVISIONAL限制。2025年24个月性能保持NA。

## 年内排序与跨年排序

年内AUC = 各年“胜出配对数＋0.5×并列配对数”之和 / 各年“事件数×非事件数”之和。只比较同一年度的公司。年度等权均值仅平均有事件且有非事件的年份；无事件年记0对，AUC保留NA，不按0加入均值。

12个月退市配对数为128、464、240，共832；24个月为288、500，共788。pooled AUC还包括跨年配对，因此不是这两个指标的另一种写法。12个月所有正负配对2725对，其中1893对跨年，占69.47%；24个月跨年占49.74%。不同年份风险水平和事件构成会明显影响pooled排序。

下表为三种子平均后的收缩版本；within_minus_pooled为年内减pooled。年度、raw、两原因、AP/Brier对照全部见CSV。

{table(methods[['months','arm','within_year_auc','equal_year_auc','pooled_auc','within_minus_pooled']])}

单独支持保护在24个月的方向也随口径改变：年内0.7614→0.7678，pooled0.7296→0.7283。原因平方分数及组合臂24个月在两种口径下都下降。恢复年内AUC在新增三臂下均未提高：12个月原模型0.6457，三臂0.6229/0.6286/0.6143；24个月原模型0.7022，三臂0.6765/0.6919/0.6791。不能只报告退市一侧。

{table(contrast[['months','contrast','estimate','lower95','upper95','undefined_repeats']])}

1000次、种子2026110201，按测试年分层并以公司为单位，**逐元素复用原cause-support抽样索引**。所有臂、raw/收缩版本使用相同抽样；差值为消融臂减基线。区间只反映固定模型下评价公司的波动，不含训练、选择或历次开发不确定性，多重探索不可当确认性显著检验。各scope未定义次数如下（所列是AUC家族的最大值）：

{table(undef, 0)}

2023年12个月恢复没有事件，1000次AUC均未定义；在跨年度年内指标中该年贡献0对。其O/E为0、bootstrap区间也退化为0，**不表示真实恢复风险确定为0**。完整逐scope/指标计数保留。

## 预警报告：风险水平与Top20

O/E是观察发生率除以平均预测概率，大于1表示总体低估，不能直接当作可部署的放大倍率。以下为原模型退市：

{table(baseoe[['scope','months','observed','mean_predicted','observed_expected','lower95','upper95']])}

Top20名额按年ceil(0.2n)为8、9、11；合并先按年度选定后相加。命中率=捕获/被检查数，覆盖率=捕获/该年事件总数，提升倍数=命中率/该年观察发生率。二者不能混称。例如2024年12个月支持保护捕获7/16，覆盖率43.75%，检查9家中7家发生退市，命中率77.78%，提升2.1875倍。

{table(alert[(alert.version == 'shrunk') & (alert.cause == 1) & (alert.scope != 'pooled')][['scope','months','arm','checked','captured','precision','coverage','lift']])}

所有两原因、raw/收缩版本及合并结果见TOP20_LIFT.csv。零事件年的lift与coverage为NA，未知结局不当非事件。所有被支持的原评价窗口均为完整终点，IPCW实际权重为1，沿用既有评价器。

## 无变量AJ基线与风险水平滞后

temporal-affine分支使用训练AJ构造校准偏移形状，但没有等价的“日历月常数预测Brier”表，因此这里直接从各截止日前保存的训练time/event和日历时钟计算AJ，不估计校准参数。每年度、时限、原因的AJ作为所有评价公司的同一常数；训练删失留在风险集中直到删失时点。嵌套训练集不能相加当独立样本。pooled逐公司重算，各年使用各自历史常数，不把年度Brier简单平均。

{table(basenull[['year','months','training_aj','null_brier','forest_brier','forest_minus_null_brier']])}

原森林在全部有支持的年度/时限/原因上Brier均优于该历史常数基线；因此“风险水平低估”不等于“森林完全没有预测信息”。但两者都依赖历史发生水平，森林在2024年24个月均值11.58%仍远低于44.44%，优于旧历史基线并不证明绝对概率准确。

下表概率为百分数：

{table(baselevel[['year','months','training_aj','mean_predicted','observed']], 2)}

![预测与观察水平](RISK_LEVEL_TIME_SERIES.png)

12个月训练AJ为1.73%、2.71%、5.97%，原模型均值1.91%、3.24%、5.85%，观察率11.11%、35.56%、9.43%。历史混合水平上升滞后于评价队列峰值，是有直接数值支持的描述；仅三个外层年份，不拟合趋势、不做显著性检验。既有仿射/3倍示例也显示统一放大不能保证跨年都受益；本轮不重估或试用任何新映射。

## 日历时间、队列与协变量

固定443家首次财务/审计ST企业，每公司只保留一条风险历程；2012—2022入组者用已有2025截止训练标签，2023—2025入组者用原冻结评价标签。累计207009风险天、62次先退市、251次先恢复。原点为公告后下一交易日开盘前；到首次事件/删失或24日历月即停止。**这不是全市场每年退市数，也不包括先恢复后再退市。** 风险公司数是当年贡献风险时间（或Jan1端点事件）的公司；人月=实际天数÷(365.25/12)。2026仅到6月30日，不能把半年度事件数与整年直接相比。

{table(cal.loc[cal.calendar_year.ge(2021), ['calendar_year','risk_companies','person_months','termination_events','recovery_events','termination_rate_per100_person_months']], 3)}

事件发生日历年2024有23次退市，粗率3.792/百人月，2023为2.049、2025为1.150。峰值并非只见于单一入组年的观察比例，支持关注日历环境；同时风险集构成与入组后时长在变化，不能把粗率视为已调整日历效应。精确“每个入组年×日历年”表在私有目录；公开COHORT_CALENDAR_CROSS按入组期合并，1—4事件及可由边际倒推的小格互补抑制，COHORT_AJ保留支持状态与稀少计数抑制标记。

协变量SMD=(评价均值−训练均值)/sqrt((训练方差＋评价方差)/2)，各列只用非缺失值；缺失率另列，不把NaN当0。未定义方差保留NA。完整83列结果留私有，公开五组汇总和每年最多5项大漂移变量；不公布原始均值、真实阈值或公司值。

{table(shift[['year','feature_group','features','median_abs_smd','above_abs_05','mean_train_missing','mean_eval_missing']])}

市场24变量的|SMD|中位数为0.333、0.690、0.541，2024有16/24项超过0.5；2024尾部收益、极端下跌、下行波动等变化明显。财务组中位数为0.219、0.173、0.156，较市场组小。缺失覆盖也不同。**存在协变量漂移的证据较强；它解释低估的份额不可识别**：没有重加权、条件风险模型或因果对照，且2025市场分布仍明显不同、观察退市却下降。不能据此说低估全由协变量、政策或模型交互衰减造成。

## 退市依据与政策时间

只复用post_st_rule_features_v1已有45例终止公告审计，按原sample_id映射到当前风险历程，不用ST/*ST简称推断终止原因，不重新对未审查文本分类。本次62次先退市中43次有已审查明确原因，19次不可得/文本不明确（包括全部2025入组原因缺口）。旧审计与当前下一开盘原点/随访窗口并非全覆盖，新出现事件保持不可得。

{table(reason, 0)}

为防小样本识别，原因按日历期与“已审查低于1元/其他已审查/不可得”合并；2023—2024日历期35次退市中19次具有已审查低价依据，6次其他依据，10次无明确依据。不可得部分不能推断成非低价。其他细原因与逐年小格保留私有。

政策分支研究的是制造业季度首次ST预警（首次ST为目标），本轮是全行业非金融ST后的首次恢复/最终退市，**其样本、标签、个体规则匹配和错误归因均不适用**。只引用已核查官方日期：2020-12-31发布退市修订，市值条款2021-07-01施行；2024-04-30发布修订，财务新条款以2024年报为首个适用年度，市值条款2024-10-30施行。低于1元价格路径不能混同市值路径，2024发生的退市不能全部归入新财务标准。[上交所2020通知]({p2020})、[上交所2024通知]({p2024})。

POLICY_EVENT_ALIGNMENT.csv将日期与年事件/暴露并列；政策表只展示上交所示例，不冒充逐公司、逐板块完整适用判断。网页现标“失效”表示历史版本，不抹去当时生效日期。本轮核查官方网页，深交所2020链接一次超时，保留失败，采用已有来源和已取得的上交所通知，不补抓任何公司结局。

## 证据边界、验证与失败

1. 不能独立分解队列效应、日历效应、入组后时长效应：日历时间=入组时间+随访时长，年份少且样本小。粗发生率和SMD均不是因果估计。
2. 不能确定2025年24个月的校准或排序；保持NA。没有为补齐时间线获取新公司标签或延长观察窗口。
3. 不能证明单家企业概率的“真值”；O/E描述群体平均。缺失的终止依据、全板块政策适用细节和财报原始版本仍未解决。
4. 竞争事件在另一原因的CIF二元评价中是已知非该原因事件，不是失访或未知。原pooled AUC保留为跨年排序对照，不将其删除或改写成年内指标。完整raw、收缩、恢复、不利结果和所有未定义重抽样均在交付表中。
5. 指定8个验收值全部通过；逐年sklearn、原fast_metrics和纯配对计数一致；三种子均值逐元素复现保存预测；原pooled AUC/AP/Brier全部复现；输入哈希前后不变，CURRENT指针保持。全pytest含独立R检查见TESTS_full.log。
6. 实际分析失败均保留：验收回执首次因numpy.bool_不能JSON序列化失败（8个数值已通过）；初次metrics/drift尝试误把仅含X/ID的worker评价文件当作含groups，分别报KeyError后在新attempt修正读取来源。未改数据或评价口径。补齐pooled AJ对照后又做一次成功drift汇总，旧成功结果也保留。无真实模型拟合、无倍率搜索。

本次主要修订是报告的比较对象，而不是以新结果选择模型。论文建议文本见PAPER_PATCH_ZH.md；原报告不改写，当前参考模型继续保留。
'''
    (PUBLIC/'RESULTS_ZH.md').write_text(report)
    paper = '''# 论文报告修订建议（不改写原文件）

目标原文件：experiments/post_st_cause_support_v1/PAPER_METHODS_RESULTS_ZH.md。

## 1. 最重要的结果句：区分年内与跨年排序

原文：
> 12个月合并退市AUC/AP从0.645/0.247变为新准则0.661/0.273，24个月从0.730/0.682变为0.722/0.660；对应主要差值的条件bootstrap区间跨越0。

建议替换：
> 以仅含同年度事件—非事件配对的年内AUC为主要排序汇总，原模型、原因平方分数、退市支持保护及组合臂的12个月AUC分别为0.7584、0.7019、0.7788、0.7055；24个月分别为0.7614、0.7360、0.7678、0.7221。原因平方分数相对基线的年内AUC差值为−0.0565（描述性95%区间−0.1492至0.0252）及−0.0254（−0.0942至0.0388）；支持保护为+0.0204（−0.0120至0.0592）及+0.0063（−0.0288至0.0403）。跨年pooled AUC保留为独立对照：12个月原模型0.6451、新准则0.6613；24个月0.7296、0.7219。12个月pooled上升与年内下降同时存在，不能将其表述为年度内筛查能力改善。pooled AP分别为0.2469/0.2725及0.6817/0.6602，它不是年内AP的加权替代品。

## 2. 方法句：明确估计对象和无事件年

原文片段：
> 每年先平均三个种子的公司CIF，再评价有支持的12/24个月AUC、AP、IPCW Brier和年内Top20捕获。

建议替换：
> 每年先平均三个种子的公司CIF，再计算年度AUC/AP/IPCW Brier及年度Top20。跨年度主要排序汇总为年内AUC，即以年度事件数乘非事件数为权重汇总年度AUC；无事件或无非事件的年份贡献0对，其年度AUC保持NA。另报告有效年度AUC的等权均值与包括跨年配对的pooled AUC，分别解释三者。12个月包含2023—2025，24个月仅2023/2024；2025年24个月性能NA。

补充不确定性：
> 固定模型，以公司为单位，按测试年分层进行1000次配对重抽样，分析种子2026110201。所有臂和raw/收缩版本使用与原分析完全相同的公司抽样索引，重新计算每次年度配对权重；记录每个scope未定义的重复次数。区间只刻画固定模型下的评价样本波动，不能视为确认性显著性检验。

## 3. 支持保护的结论避免过度否定或升级

原文：
> 单独支持保护完全消除单公司退市叶，但年度、时点及恢复结果呈现混合变化，未观察到稳定联合收益。

建议替换：
> 支持保护消除了单家公司支撑的退市叶，年内退市AUC的12/24个月点估计略升，但配对区间跨0，恢复排序未改善；2024年12个月Top20捕获由6/16提高到7/16，24个月由8/20降为7/20。因而该模块存在特定时限的有利信号，同时有明确权衡，尚不足以替换参考模型。

## 4. 加入风险水平与日历诊断，避免“低风险=无筛查能力”

原文片段：
> 不能据本轮结果认定其改善了稀少退市事件的时间外预测或彻底解决概率低估。

建议扩充：
> 原模型2023、2024、2025年12个月的观察/平均预测比为5.82、10.99、1.61，2023/2024年24个月为5.15、3.84；低估有明显年度差异。2024年12个月支持保护Top20命中率77.78%、覆盖率43.75%、提升2.19倍，说明绝对概率低估可以与年内筛查富集并存。固定队列的日历年退市粗发生率于2024升高、2025回落，而历史训练AJ和模型均值持续缓慢上升。该模式提示水平漂移及历史平均滞后；队列构成、入组后时长和协变量变化未被独立识别，不作政策因果结论。全部结论为已暴露队列的探索性分析。

## 5. 与仿射/归一化报告的衔接

不用改原仿射或3倍示例数值：相同年内正斜率变换保持年度排序，不同年度的变换可以改变pooled排序；年内分层AUC也保持。归一化不能自行获得概率含义；观察/预测比不是本轮选出的校准倍率。本轮不提出新的最优模型或映射，不把现有展示结果解释为前瞻校准证据。
'''
    (PUBLIC/'PAPER_PATCH_ZH.md').write_text(paper)
    # Full raw and recovery comparisons are accessible without private artifacts.
    appendix = '# 全部口径汇总\n\n仅支持的年度参与；2025H24保持NA。所有条目为探索性固定模型分析。\n\n'
    for cause in [1, 2]:
        for v in ['raw', 'shrunk']:
            part = m[(m.scope == 'pooled') & (m.cause == cause) & (m.version == v)]
            appendix += f'## 原因{cause} · {v}\n\n'+table(part[['months','arm','within_year_auc','equal_year_auc','pooled_auc','within_minus_pooled']])+'\n\n'
    (PUBLIC/'ALL_ARMS_ZH.md').write_text(appendix.rstrip()+'\n')
    (PUBLIC/'README.md').write_text(f'''# 退市报告口径与水平漂移（只读）

先读[结论](RESULTS_ZH.md)、[论文修订建议](PAPER_PATCH_ZH.md)及[全部臂汇总](ALL_ARMS_ZH.md)。分支research/xai-rsf-reporting-drift-v1源自6ea32103d2b6d953ae25ab0b9fa4301c6cfbb434，不合并、不替换当前模型。

私有目录见[run_location.json](run_location.json)；现有cause-support目录只读。新私有目录为`{run}`，完整公司风险历程、逐公司保存预测的读取校验、完整入组年×日历年表、完整83列漂移和重抽样数组都保留该目录。公开目录不含任何新增公司行、标识、切点、模型参数或新预测。此前rankings授权未扩展。

复现使用同仓库代码和原输入，先创建一个新的空目录（不要使用本次run路径），然后：

```bash
python -m experiments.post_st_reporting_drift_v1.analyze metrics --run NEW_PRIVATE_RUN --source CAUSE_SUPPORT_PRIVATE_RUN
python -m experiments.post_st_reporting_drift_v1.analyze drift --run NEW_PRIVATE_RUN --source CAUSE_SUPPORT_PRIVATE_RUN
```

每次调用自动新建带微秒时间戳的attempt目录；失败日志保留。用明确成功目录交付：

```bash
python -m experiments.post_st_reporting_drift_v1.deliver --run NEW_PRIVATE_RUN --metrics METRICS_ATTEMPT --drift DRIFT_ATTEMPT
```

交付脚本拒绝覆盖已有公开CSV/图。复算时留结果于新私有目录；需新一版公开交付时使用新的实验版本目录，不删除历史文件。完整测试命令与日志见[VALIDATION_ZH.md](VALIDATION_ZH.md)。

主文件：STRATIFIED_METRICS.csv、STRATIFIED_BOOTSTRAP_95.csv、OBSERVED_EXPECTED.csv、TOP20_LIFT.csv、NULL_AJ_BASELINE.csv。时间诊断：CALENDAR_INCIDENCE.csv、COHORT_CALENDAR_CROSS.csv、COHORT_AJ.csv、LEVEL_TIME_SERIES.csv、COVARIATE_SHIFT_SUMMARY.csv、COVARIATE_SHIFT_TOP.csv、DELISTING_REASON_SUMMARY.csv、POLICY_TIMELINE.csv、POLICY_EVENT_ALIGNMENT.csv。

[PNG](RISK_LEVEL_TIME_SERIES.png) · [PDF](RISK_LEVEL_TIME_SERIES.pdf)。图来自汇总CSV，采用可复现matplotlib静态输出，符合本轮指定科研图格式。

所有CSV的NA/空值均为未定义、未支持或隐私抑制，由supported/reason/status字段区分，不填0。年内等权平均仅对有配对的年度计算；所有区间均为固定模型条件下描述性95%百分位区间。原始数据版本仍PROVISIONAL，2023—2025均探索性。
''')
    write_json(PUBLIC/'analysis_attempts.json', dict(metrics=str(metrics), drift=str(drift),
                failed_attempts=[str(p.parent) for p in run.glob('*/FAILED.log')],
                superseded_successful_drift=[str(p) for p in run.glob('drift_*') if p != drift and (p/'stage_receipt.json').exists()],
                additional_failures=['Initial acceptance receipt numpy.bool_ JSON serialization; numerical8gates passed',
                                     'Official SZSE2020 webpage timeout; used preexisting evidence and SSE official source'],
                no_protocol_deviation=True, no_original_artifact_mutation=True))
    print('Aggregate delivery generated; inspect PNG/PDF and run final validation before commit.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--metrics', required=True, type=Path)
    parser.add_argument('--drift', required=True, type=Path)
    args = parser.parse_args()
    build(args.run, args.metrics, args.drift)
