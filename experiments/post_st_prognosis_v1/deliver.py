"""Package an allowlisted aggregate report; never export company/model records."""
from pathlib import Path
import argparse
import json
import shutil
import pandas as pd
from experiments.post_st_prognosis_v1.common import sha, write


LABELS = {'xai_q2': 'XAI-RSF q2', 'unrestricted': '同后端无限阶森林',
          'standard_rsf': '标准RSF（randomForestSRC）', 'training_AJ': '训练AJ无变量基线'}


def table(frame, digits=4):
    def fmt(x):
        if pd.isna(x): return 'NA'
        if isinstance(x, float): return f'{x:.{digits}f}'
        return str(x)
    lines = ['| ' + ' | '.join(map(str, frame.columns)) + ' |',
             '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |']
    lines += ['| ' + ' | '.join(fmt(x) for x in row) + ' |' for row in frame.itertuples(index=False, name=None)]
    return '\n'.join(lines)


def deliver(run, reports, destination):
    assert json.loads((reports/'validation.json').read_text())['status'] == 'PASS'
    destination.mkdir(exist_ok=True, parents=True)
    output = destination/'results'; output.mkdir(exist_ok=False)
    for name in ['metrics.csv', 'metrics_summary.csv', 'warning_metrics.csv', 'calibration.csv',
                 'training_diagnostics.csv', 'interpretability.csv', 'seed_stability.csv',
                 'evaluation_receipt.json', 'validation.json']:
        shutil.copy2(reports/name, output/name)
    for name in ['cohort_counts.csv', 'sample_selection.csv', 'feature_manifest.csv']:
        shutil.copy2(run/'reports'/name, output/name)
    for name in ['data_alignment.json', 'additional_checks.json', 'cohort_clock_comparison.json', 'independent_date_check.json',
                 'pilot_gate.json', 'formal_gate.json', 'pre_fit_protocol_freeze.json',
                 'preparation_history.json', 'tests_final.txt']:
        audit_name = 'tests_final_delivery.txt' if name == 'tests_final.txt' else name
        shutil.copy2(run/'audit'/audit_name, output/name)
    shutil.copytree(reports/'figures', output/'figures')
    env = {k: json.loads((run/'audit'/f'environment_{k}.json').read_text()) for k in ['oliver', 'lemon', 'r']}
    env['threads_per_fit'] = 1
    env['memory_limit_per_fit_GiB'] = 4
    env['note'] = 'Both node environments verified; all 11 short fits actually executed on oliver. No GPU.'
    write(output/'environment.json', env)
    d = pd.read_csv(output/'metrics.csv', dtype={'seed': str})
    s = pd.read_csv(output/'metrics_summary.csv')
    diag = pd.read_csv(output/'training_diagnostics.csv')
    warning = pd.read_csv(output/'warning_metrics.csv', dtype={'seed': str})
    interp = pd.read_csv(output/'interpretability.csv')
    stability = pd.read_csv(output/'seed_stability.csv')
    cohort = pd.read_csv(output/'cohort_counts.csv')
    alignment = json.loads((output/'data_alignment.json').read_text())
    validation = json.loads((output/'validation.json').read_text())
    plan = json.loads((run/'input/plan.json').read_text())
    source = Path(__file__).parent

    def doc(name, content):
        with (destination/name).open('x') as f: f.write(content.strip()+'\n')

    def means(cause, months):
        f = s[(s.scope!='train_apparent') & (s.cause==cause) & (s.horizon_months==months)].copy()
        f['model'] = f.model.map(LABELS)
        return table(f[['model','scope','auc_mean','ap_mean','brier_mean']].rename(columns={
            'model':'模型','scope':'入组期','auc_mean':'AUC均值','ap_mean':'AP均值','brier_mean':'Brier均值'}))

    def ensembles(cause, months):
        f = d[(d.scope!='train_apparent') & (d.cause==cause) & (d.horizon_months==months) & d.seed.isin(['ensemble','none'])].copy()
        f['model'] = f.model.map(LABELS)
        return table(f[['model','scope','auc','ap','brier','risk_mean','ipcw_observed_incidence']].rename(columns={
            'model':'模型','scope':'入组期','risk_mean':'平均预测概率','ipcw_observed_incidence':'观察事件比例'}))

    doc('README.md', '''# 首次财务／审计ST后恢复与退市：公告日实验

本轮已实际完成2个小规模试运行和9个正式模型；无拟合或发布失败。所有入组、恢复、最终退市日期均采用公告日。使用全部非金融行业，2012—2022年入组309家公司训练，拟合结局信息严格截止2023-01-01之前；同一模型评价2023年36家和2024年45家。主终点为入组后24个日历月内“先恢复还是先最终退市”，12个月为补充。

**本配置结论：PREDICTIVE_NO_GO。数据与结果性质：PROVISIONAL、已接触后期结果的探索性回顾评价。** 二阶模型可稳定运行并完整精确解释，但2024年退市区分度接近随机，恢复区分度较差，三个模型均严重低估退市概率。本轮不支持把该配置直接用于此困境后预测任务；不将这一结论外推成整个RSF框架失效。没有追加调参、测试集再校准或第二轮训练。

| 文件 | 内容 |
| --- | --- |
| [PROTOCOL_ZH.md](PROTOCOL_ZH.md) | 冻结目标、模型、时间与评价口径；复现入口 |
| [DATA_ALIGNMENT_AUDIT_ZH.md](DATA_ALIGNMENT_AUDIT_ZH.md) | 样本、公告时钟、随访来源、PIT和行业限制 |
| [RESULTS_ZH.md](RESULTS_ZH.md) | 两个结局、两个时限、分年与合并、三种子均值 |
| [PREDICTION_COMPARISON_ZH.md](PREDICTION_COMPARISON_ZH.md) | 集成概率、校准、Top20/25和提前识别 |
| [INTERPRETABILITY_COMPARISON_ZH.md](INTERPRETABILITY_COMPARISON_ZH.md) | 全部函数复杂度、重构和稳定性 |
| [TRAINING_DIAGNOSTICS_ZH.md](TRAINING_DIAGNOSTICS_ZH.md) | 实际资源、数值检查、失败与更正记录 |
| [METHOD_INNOVATION_DECISION_ZH.md](METHOD_INNOVATION_DECISION_ZH.md) | 负面结论及可以／不能归因的原因 |
| [EVALUATION_CORRECTION_ZH.md](EVALUATION_CORRECTION_ZH.md) | 730/731天混合导致的旧IPCW评价更正 |
| [execution_receipt.json](execution_receipt.json) | 任务、代码、输入和模型哈希 |
| [results/](results/) | 允许发布的全部汇总CSV、图和环境 |

本分支为 `research/xai-rsf-post-st-prognosis-v1`，不替换原首次ST预警实验。企业明细、真实预测、真实切点、完整模型与企业解释仍仅保存在服务器私有运行目录。新增标准RSF针对新终点重新拟合，不冒充此前首次ST预警M2。
''')
    doc('PROTOCOL_ZH.md', f'''# 冻结协议与复现

授权范围为全部非金融行业、首次因财务／审计原因被ST或*ST之后的恢复／退市预测，2023与2024入组分别测试。“首次”是已有历史来源中的审计接受首次，不是完整上市生命周期已证明首次。先前已有非目标ST的企业不当作新的首次目标ST加入。

起点为首次目标ST公告日。终点1为先于完整恢复的最终终止上市公告X（全原因）；终点2为先于最终终止上市的完整恢复公告A。B/D/C/S/T都是中间状态，T不等于X；先恢复后退市者属于首次恢复。预测的是这两个首次转归的CIF，不是“最终必然退市概率”或撤销*ST但仍为ST的概率。两个CIF之和不必为1，余量是时限内未发生首次转归的概率。

每家公司仅一个起点，主时限24日历月、次时限12日历月。固定随访来源边界2026-09-03；训练信息边界严格早于2023-01-01，因此近期训练病例可被右删失。禁止用训练公司2023年后的结局补齐训练。2024不滚动吸收2023测试结局。特征仅取入组公告日期之前已经记录可得的内容；无日内时间的同日披露一律排除。

## 有限模型比较

三个配置、三个固定种子42/101/202，共9个正式拟合。所有模型为32棵树、最大深度6、75个入模字段；共同公司bootstrap。训练309家公司中的原始观察事件为退市14、恢复184。因样本从数万季度记录变为数百家公司，在拟合前冻结叶最小记录数10、公司组数5（原100/20），两种Python森林同时调整；不是看测试表现后的调参。

| 配置 | 冻结设置 | 比较边界 |
| --- | --- | --- |
| XAI-RSF q2 | 路径最多2变量；每树25/75变量子空间；共享分位数切点；最多5候选阈值；每叶总事件至少3 | 完整二阶CIF函数；未启用全森林格数上限或信息惩罚 |
| 同后端无限阶 | 不限路径变量；75变量可用；局部随机切点；其余共同预算 | 与q2同时差在阶数、覆盖组织和切点，不能单独估计q的因果贡献 |
| 标准RSF | randomForestSRC 3.9.0；logrankCR，两原因权重(1,1)；nodesize10，nodedepth6，mtry75，nsplit5 | 与Python后端不逐位等价；原生分裂和节点约束不同；缺失用仅训练列中位数补齐 |

另有训练Aalen–Johansen无协变量基线。Python采用原生缺失路由；R补齐不新增缺失指示维度。标准RSF是新任务的标准实现参考，并非经过充分调优的性能上界。旧M2预测首次ST，不能用于本次ST后转归，未重训或覆盖旧M2。

## 评价

分2023、2024与合并期重算AUC、AP、IPCW Brier；合并AP不平均两个年份。主要比较三种子指标均值及范围，范围不是置信区间；另列预先等权平均三种子预测形成的集成指标，二者不可混用。当前测试81家公司在12/24月均有完整终点判定，IPCW权重均为1；训练表观指标仍正确处理删失，且不是泛化性能。

删失时钟用精确日历月，事件时取G左极限，已知竞争事件作为该原因的非目标结局，未知结局不给端点评价权重。未做权重下限裁剪。记录等权与公司等权完全相同：每公司仅1条。Top20/25按每个入组年风险从高到低、向上取整；并列以固定sample_id顺序打破。此为全年回顾性核查容量比较，不是当时可部署的全年在线排名。无重复预警；提前3/6个月用公告日至终止公告日的日历跨度。

## 执行入口

拟合不可变提交：`{plan['code_commit']}`。代码归档、输入哈希和模型参数保留于 `{run}`。`frozen_protocol.json` 在试运行前冻结，`results/pre_fit_protocol_freeze.json` 给出其哈希。新复现必须创建唯一私有目录，禁止覆盖本次运行。

```bash
# 在独立已提交的checkout与已存在隔离环境执行；RUN为新的私有目录
python -m experiments.post_st_prognosis_v1.prepare --run "$RUN"
python -m experiments.post_st_prognosis_v1.stages --run "$RUN" --repo "$REPO" --stage freeze
/home/oliver/.local/bin/hetero-submit "$RUN/pilot.jsonl" physical both
# 等待完成并使用真实作业号；先检查试运行再提交正式任务
python -m experiments.post_st_prognosis_v1.stages --run "$RUN" --stage pilot --job "$PILOT_JOB"
/home/oliver/.local/bin/hetero-submit "$RUN/formal.jsonl" physical both
python -m experiments.post_st_prognosis_v1.stages --run "$RUN" --stage formal --job "$FORMAL_JOB"
python -m experiments.post_st_prognosis_v1.evaluate --run "$RUN" --output "$RUN/reports_calendar_months"
python -m experiments.post_st_prognosis_v1.validate --run "$RUN" --reports "$RUN/reports_calendar_months"
python -m experiments.post_st_prognosis_v1.plot --reports "$RUN/reports_calendar_months"
```

`prepare` 要求调用前创建空的 `input/ audit/ private/ reports/ results/`；输入源路径和解释器为本服务器配置，不能假设在其他机器存在。`stages freeze` 要求干净Git工作区。若另用hetero-run sidecar，先核对物理核布局；本轮明确使用physical/both提交。只有计算完成、结果发布及哈希验证全通过才记成功。`validate` 有本轮样本及156条完整端点检查，换队列时需显式重新审计这些断言，不能删除失败断言强行通过。
''')
    doc('DATA_ALIGNMENT_AUDIT_ZH.md', '''# 数据对齐审计

性质为 **PROVISIONAL_POST_EXPOSURE_EXPLORATORY**。恢复了现有审计产物中的实际日期及覆盖确认，没有用执行日替代缺失公告日，也没有把报告期充当披露日。日期可核查不等于原始披露版本已经PIT认证。

## 样本与结局

公告日落在2012—2024的已接受候选406家；剔除金融1家、行业不明15家，保留非金融390家（制造236、其他154）。与前次按执行日统计的非金融390家是同一公司集合。行业分类代理因公告时点前可用报告选择而有一家公司分类变化，不影响本轮全非金融入组。训练309家；2023年36家；2024年45家；训练／测试公司组交集0。没有重复季度记录。

''' + table(cohort[['split','horizon_months','companies','termination_events','recovery_events','known_no_first_event','unknown_or_censored']]) + '''

每行的事件数、已知无首次转归、未知/删失之和等于公司数。测试24个月合计：31先退市、31先恢复、19未出现首次转归，无未知；这19家不能写成“正常企业”。训练24个月49家结局未完整可见，保留删失，未补成正常或恢复。训练拟合中事件0共有111家，包含完整时限无事件与提前删失，不能把111全部称为未知。

## 公告日与随访来源

使用 `first_st_followup_audit_20261025_v1/audit_v4_observed_first` 的入组裁决，以及已有 `PhD_STARST_Prognosis/runs/phase1_data_freeze_v1` 的事件历史与个案裁决。4158条主源解析记录在代码、状态、公告日、执行日和正文上与规范事件表逐项一致；规范表包括补充历史共4257条。每个选入公司均有主源记录。采用已有核实公告日覆盖原始公告日；否则保留原始Annoudt。公告日期缺失为0，选入起点有1条既有公告日裁决覆盖；执行日仅用于来源匹配和审计，不进入生存时钟。

独立逐公司复核390条起点及260条已观察首次转归：起点有389条公告日与执行日不同，实际均使用公告日；每条已观察结局的停止时间均等于来源ledger中的首个A/X公告日。无事件者停止于核实随访边界与24个月时限的较早者。训练停止时间均早于2023-01-01，财务与市场可得日均严格早于起点；见 `results/independent_date_check.json`。

随访上限2026-09-03来自已有来源范围确认：2026-09-08研究者明确接受“非金融企业全部年份”的已供快照，并保留其既有事件时间边界。它不是本次臆造的数组，也不是数据库提供商的精确查询时间；提供商query start/end/time仍缺失。无完整生命周期证据：390家中仅18家满足旧日频连续历史支持，372家不满足；全体仍只称“已提供历史中的审计接受首次”。未以“未查到更早ST”证明上市以来首次。

状态链异常标记涉及训练3家、2023年6家、2024年17家。按本轮“首次A或最终X”口径，分别由既有裁决、明确终止上市文本及ST名称、或仅缺中间状态解决；不合成A，不把T直接当X。未解决端点冲突为0；原异常和对应来源逐条保存在私有ledger。该处理不证明全历史状态链无瑕疵。6个关联证券组另有安全代码终止记录，核对后不改变这6组的首次A/X终点；不将证券代码自动等同独立企业。

## 75个输入与可得时钟

48财务内容、24市场内容、3缺失覆盖／陈旧度质量字段，列名与顺序见 `results/feature_manifest.csv`。继承原矩阵公式；本轮不引入MD&A。财务选入组公告日之前可得的最新报告；可得日取原披露及修订可得日的最大值。增长／变化字段同时检查上年组成记录，后者不明或晚于起点则该值缺失；同日日期无日内时间时排除。每个非缺失值必须有已记录的组成可得日约束。

市场只读取入组公告日前的交易日，24项公式来自原日频加工；独立重算32条原季度记录与原结果在1e-12内一致。财务与市场可得时钟缺失的选入记录均为0。财报与日频数据库仍是继承的当前版本，历史改写／原始发布版本未被独立证明；行业也是与报告关联的历史字段代理，行业版本时间未认证。因此不能声称正式PIT验证已通过。

发现训练内容 `roa` 与 `net_profit_to_assets` 完全重复，`zero_amount_ratio_3m` 在训练中为常量。为保持拟合前冻结的75列，不事后删除、改配或重训。重复内容会使函数归属及交互选择产生替代，须纳入解释稳定性的限制。模型原生缺失路由与R训练中位数补齐不同；不能说输入参数化完全一样。

原始来源文件及SHA256、模型输入哈希、公司数与事件数见 `results/data_alignment.json` 和 `execution_receipt.json`。sample_id、company group、起点、终点类型/公告日、实际随访cap、财务/市场可得日、端点来源都保留在私有元数据；未向Git发布企业记录。
''')
    doc('RESULTS_ZH.md', '''# 预测结果：三种子均值

以下主表为三个固定种子42/101/202的指标均值，不是挑选最好种子，也不是集成预测指标。完整种子值、最小／最大及标准差在CSV中；三个种子不构成置信区间。记录与公司等权完全一致。

## 24个月先退市（主终点之一）

''' + means('termination',24) + '''

2023有11/36个事件，2024有20/45个事件，合并31/81。q2在2023尚有一定排序能力，2024平均AUC仅0.542；合并AUC不能掩盖分年失效。AP的参照水平随事件发生率变化，2024 AP约0.503并不代表概率校准良好。

## 24个月先恢复（另一竞争转归）

''' + means('recovery',24) + '''

q2恢复AUC在2024为0.440，合并0.493，弱于本轮标准RSF；不能只展示退市的相对AUC而省略恢复的负面结果。

## 12个月先退市

''' + means('termination',12) + '''

## 12个月先恢复

''' + means('recovery',12) + '''

2023年12个月内没有观察到首次恢复，AUC与AP无定义，表中NA不是0，也不是计算任务失败。2024也仅4个恢复事件，不支持强结论。合并指标来自合并预测重新计算。

## 表观训练表现与评价边界

''' + table(s[(s.scope=='train_apparent')&(s.horizon_months==24)&(s.cause=='termination')][['model','auc_mean','ap_mean','brier_mean']]) + '''

以上为同样本表观指标，不是OOB或交叉验证。高训练AUC与弱时间外表现提示拟合适应性及年代迁移问题；不能据此区分过拟合、事件谱变化、数据版本误差各自占比。训练只有14个先退市事件，测试31个；不使用测试事件率反向调参或重新校准。

![三种子平均预测表现](results/figures/prediction_comparison.png)
''')
    ww = warning[(warning.seed=='ensemble')&(warning.horizon_months==24)].copy()
    ww['model'] = ww.model.map(LABELS)
    doc('PREDICTION_COMPARISON_ZH.md', '''# 集成预测、校准与核查覆盖

本页先将同配置三个种子的概率等权平均，再评价这个集成；主比较仍见 `RESULTS_ZH.md` 的种子均值。没有从三个种子中选优。原始预测不重新映射、缩放或按测试结局校准。

## 24个月先退市集成

''' + ensembles('termination',24) + '''

q2合并平均退市概率0.0751，实际31/81=0.3827；标准RSF约0.0704，无限阶约0.0814，同样明显低估。训练AJ基线约0.0516，反映训练端可见发生率较低。相对这个弱基线的Brier改善不等于已能正确估计新年代绝对概率。不能使用测试事件率重新标定后再把结果冒充原固定模型。

![平均风险与观察发生率](results/figures/calibration_in_the_large.png)

## 24个月先恢复集成

''' + ensembles('recovery',24) + '''

q2合并平均恢复概率约0.6000，观察发生率0.3827，存在过估。两种转归应同时检查；单独强调退市排序会遗漏这一问题。五组校准表在 `results/calibration.csv`，每组约7—17家公司，只作描述，未声称曲线稳定。

## Top20%及Top25%

''' + table(ww[['model','scope','top_fraction','checked_companies','captured_events','all_target_events','known_non_target_checked','unknown_checked',
               'identified_at_least_3_calendar_months_before_termination','identified_at_least_6_calendar_months_before_termination']].rename(columns={
                   'model':'模型','scope':'入组期','top_fraction':'比例','checked_companies':'核查数','captured_events':'捕获数',
                   'all_target_events':'总退市数','known_non_target_checked':'已知非目标','unknown_checked':'未知',
                   'identified_at_least_3_calendar_months_before_termination':'提前≥3月','identified_at_least_6_calendar_months_before_termination':'提前≥6月'})) + '''

合并Top20为各年分别排名后合并：17/81家被核查，q2捕获8/31（25.8%），无限阶9/31，标准RSF8/31。Top25时21家被核查，q2捕获10/31、无限阶11/31、标准RSF9/31。小样本向上取整使实际核查比例略大于名义比例。此处“已知非目标”含先恢复及到时限仍未出现首次转归，不能全称为误报或正常；本批未知为0。

每企业只在首次ST公告时预测一次，所以无重复预警、无季度核查机制。提前识别以公告日计算，不是退市执行日前的时间。年度排序使用事后完整入组名单，不能当成当时已部署的在线决策系统。逐种子和12个月结果均保存在 `warning_metrics.csv`。
''')
    qq = diag[diag.model=='xai_q2']
    doc('INTERPRETABILITY_COMPARISON_ZH.md', '''# 完整解释与可靠性

二阶森林直接预测原因别CIF，保留bootstrap、生存分裂、叶Aalen–Johansen和固定平均。路径最多用两个原始输入，允许重复细分已出现变量。对固定训练参考分布，完整预测精确分解为：

`F_k(t | x) = μ_k(t) + Σ_j f_jk(t, x_j) + Σ_{j<l} f_jlk(t, x_j, x_l)`。

全部主效应与交互必须一起加回；没有高阶残差或另训解释替身。贡献在CIF概率尺度上，可乘100成为相对固定基线的百分点贡献；不是对数风险率、风险率倍数、干预效果或因果解释。缺失作为原生路由状态保留，不能只展示非缺失阶梯曲线而宣称完整。

## 全时间、双原因的原生结构规模

''' + table(qq[['seed','main_functions','interaction_functions','distinct_thresholds','spatial_table_cells','scalar_table_cells','low_support_cell_fraction','zero_support_cell_fraction']]) + '''

75输入中实际形成41—42个非零主效应、53—59个二元函数；共享切点仍有183—196个实际不同阈值；全空间表1884—2009格。标量表425784—454034项包含113个原生时间点和两个原因。实际保存可采用叶分解表示，不能将压缩存储长度冒充数学函数表规模。无限阶同后端有485—510个阈值、最大路径6变量；其二阶表数未计算，记NA，不写成0。

12/24日历月、两原因、全部390家公司完整函数和的最大误差7.77e-16；原生完整时间网格检查最大9.99e-16。函数保存重载与模型保存重载通过，后者误差为0。没有裁剪预测制造通过。

## 按时限和原因的有效贡献

''' + table(interp[['seed','horizon_months','cause','reference_active_mains','reference_active_pairs','reference_pair_absolute_share']]) + '''

“参考样本有效”指固定训练参考上的平均绝对贡献超过1e-14，用作浮点活动描述，不是统计显著性筛选。全时间非零函数数与指定时限有效函数数是不同概念。二元绝对贡献占比不按正负抵消计算。

## 跨种子稳定性

''' + table(stability[['seed_a','seed_b','cause','main_jaccard','pair_jaccard','common_nonconstant_functions','median_function_correlation']]) + '''

主效应选择Jaccard约0.456—0.519；交互选择只有0.057—0.077。共有函数的相关系数使用同一训练参考、固定730天，避免把公司日历时限差别混成函数差别；只比较共有且非恒定函数，不能把它说成全模型稳定性。仅三个种子，不是重抽样置信区间。

约54.7%—55.9%的全空间格子不足10个训练公司支持，18.2%—20.4%格子无训练公司。乘积参考分布可能覆盖现实数据稀少的组合；完整可计算不等于这些组合已经可靠估计。ROA与净利润/总资产完全重复也使变量归属可替代。因此本轮只证实“解释精确”，没有证实“解释简单、稳定、无歧义”。本轮没有森林可辨识性惩罚，也没有Hessian信息秩意义下的可辨识性检验；不声称该机制产生独立增益。企业函数、真实阈值和逐企业解释保留私有。
''')
    doc('TRAINING_DIAGNOSTICS_ZH.md', f'''# 执行与数值诊断

训练代码固定为 `{plan['code_commit']}`，独立工作树基于 `26124b78c568213cca4a5cfaf562bb5d718b03af`。未覆盖原工作区、旧M2、main或历史结果。所有拟合读取归档代码和输入哈希，9个正式模型与2个试运行均计算成功、发布成功、逐文件SHA256通过。种子集合严格为42/101/202，各模型3个，没有删掉失败种子或只保留成功结果。

试运行作业2533（2任务）用4树，正式作业2534（9任务）用32树。两者分配oliver+lemon的physical池；每任务单CPU、BLAS/OMP/MKL线程1、地址空间上限4GiB、NFS输入／结果。oliver P核第二线程仍受cgroup限制，未绕过；没有GPU计算。实际全部11个短任务由oliver领取，lemon预检成功但未执行拟合。没有重复计算以制造双机利用率，也没有声称本批因双机加速。

正式作业pool计时9.63秒；各任务墙钟时间之和33.63秒。它不是从数据审计、排队到报告的全流程时间，更不是比较两机效率的测量。q2单任务约1.39—1.42秒，无限阶约8.27—8.41秒，R约1.24—1.97秒。小样本309家和32树解释了低CPU总用量，不是把森林任务自动跨机器拆分。

''' + table(diag[['model','seed','seconds','peak_rss_MiB','child_peak_rss_MiB','reload_error','candidate_evaluations','leaves','max_path_variables']]) + '''

Python进程峰值RSS约155—177MiB；R子进程峰值约237MiB，父子峰值单列，不能直接称为同一时刻联合峰值。`cpu_seconds`为Python父进程计时，不含R子进程。候选评分数q2约1.95万、无限阶约26.4—26.8万，均完成。没有组合搜索失控或内存溢出。森林无迭代凸求解器，Hessian秩、优化gap和系数触边不适用，不能伪填0表示收敛。

全部原生CIF有限、范围合法、随时间不减、两原因概率和≤1；全部模型保存重载误差0，q2路径阶数违规0、祖先变量检查错误0。精确解释检查覆盖两原因、四个实际天数时限及原生时间点；详细误差见CSV。未人为裁剪预测。每个树叶最小总事件约束不保证退市原因的叶内事件充分；只有14个训练退市事件仍是严重支持限制。

## 失败、更正及测试

第一次准备数据在任何拟合前因pandas3的只读NumPy视图报错，已用显式copy修复；失败脚本、错误记录和运行目录保留。第二次成功准备后增加同日入组与端点顺序冲突检查，在拟合前重新创建最终唯一目录。没有正式拟合失败，没有发布失败，无须补跑。

初次评价的原始天数IPCW混合了730/731天行政时限，引入伪删失权重。已经修复为精确日历月评价，原文件保留且废止，全部拟合及预测不变。详见 `EVALUATION_CORRECTION_ZH.md`。另外汇总列 `r_version` 更正为 `randomForestSRC_version`，因为原worker值是包版本3.9.0，实际R版本另见环境文件；不影响任何指标。

完整测试 **65 passed，0 skipped，1 warning**，包含独立R检查。唯一warning来自既有EBM可视化对缺失状态显示的提醒，不是本轮森林测试失败。独立验证程序重新实现成对AUC、按并列分数组AP和Brier，核对全部156个测试指标组合，最大容许浮点差5e-14，全部通过；常数AJ预测合并AP等于31/81。还核对368个不可变输入/代码文件、9个模型所有产物哈希、公司组不交叉及核查人数守恒。

版本见 `results/environment.json`，完整测试日志见 `results/tests_final.txt`。环境采集最初查询了未安装的`interpret`元包，随后改为实际已安装`interpret-core`；未安装或更改任何依赖，不影响训练。旧原始结果与失败准备目录见 `results/preparation_history.json`。
''')
    doc('EVALUATION_CORRECTION_ZH.md', f'''# 日历月评价更正

首次评价按实际天数合并反向Kaplan–Meier。当不同起点的24个月分别为730和731天时，一个已完整观察24个月的730天无事件记录会被错误用于减少另一个731天记录的G值。常数AJ基线合并AP变成0.344444，而81家公司都已知24个月结局，正确值应为31/81=0.382716。这一恒定预测自检暴露了评价问题。

已保留旧脚本、CSV及问题说明在 `{run}/audit/initial_calendar_day_evaluation/`。原 `reports/` 下旧评价文件不作为最终结果；本次正式交付仅使用 `reports_calendar_months_v2/`。

修复仅作用于评价：每个起点加12/24日历月定义时限，反向KM使用以该起点为锚的精确日历月时钟。之后发生的事件在较早时限评价为已知非目标，其权重使用G(时限)，不是更晚的事件日。各公司在365/366/730/731天的冻结CIF预测不变，森林仍以实际天数拟合；所有11次原拟合保留，**重训次数0**。

新增闰年回归测试保证完整24个月记录不产生伪IPCW，所有测试端点权重现在均为1。独立验证不依赖sklearn指标实现，用逐对排序和分组AP复算156组测试指标；通过。错误版本的合并AP、Brier不应再引用。本修复由端点定义与算术不变量决定，没有按照模型优劣选择评价方法。
''')
    doc('METHOD_INNOVATION_DECISION_ZH.md', '''# 本轮研究判断

**PREDICTIVE_NO_GO：仅针对当前冻结配置在“首次财务／审计ST后的首次恢复或退市”这个时间外任务。** 数据证据为PROVISIONAL，既有后期结果已接触，不宣称盲测或正式PIT认证。保留全部负面结果，本轮结束，没有开启第二阶段调参。

1. **能否真实训练？能。** 9/9正式拟合和2/2试运行成功，概率、保存重载和完整函数重构通过。执行成功不等于预测任务成功。
2. **预测是否足够？当前不够。** q2退市三种子均值AUC从2023的0.624降至2024的0.542；恢复AUC在2024仅0.440。合并退市平均预测7.5%，观察38.3%，绝对概率严重低估。Top20集成仅捕获8/31个退市事件。
3. **相对普通RSF如何？退市部分指标略好，恢复较差。** 合并退市AUC均值q2为0.614，标准RSF0.566，AP约0.480与0.473，Brier约0.327与0.331；但标准RSF本身较弱，不能以超过弱基线宣布成功。合并恢复AUC均值q2为0.493、标准RSF0.603。有限32树预算也不能当作标准RSF的优化性能上界。
4. **是否是二阶限制导致？证据不足以单独归因。** 无限阶森林与标准RSF都严重低估退市，q2也不是退市排序最差者。两种Python配置还同时改变变量覆盖和切点策略，不是孤立q消融。不能凭本轮宣布q2是唯一失败原因，或宣称去掉二阶限制即可解决。
5. **解释是否可靠？仅精确性已证实。** 41—42主效应、53—59交互可全量加回预测，但交互Jaccard仅0.057—0.077，超过一半空间格支持不足10家公司，存在重复输入。没有证实稳定、简单、低歧义。
6. **可辨识性惩罚贡献？没有测试。** 本森林未移植newRSF2的原因别Hessian惩罚；不能从本轮正交分解或精确加和推断统计可辨识性，也不能声称惩罚有独立价值。
7. **整体方法价值是否被否定？没有，但此应用不能直接进入推广。** 低阶森林的完整函数表示仍是可检验的工程和方法属性。它尚未解决小事件支持、时间外预测及解释稳定性；本轮不支持“迁移到ST后预测已经成功”或“论文创新已被证明”。

能从现有证据识别的限制：训练仅14个先退市事件、75输入和明显的年代间观察结局差别；训练表观AUC约0.987而测试较弱；当前版本与行业时钟仍有PIT限制；首ST生命周期历史及源范围依赖继承审计。这些事实与过拟合或分布迁移相容，但本轮没有分离各因素作用，也没有证明某个制度变化是原因。

不以测试发生率再校准，不删除困难年份，不改终点日期，不挑选最好种子。任何后续方案都应另立授权、数据审计和独立评价协议；本次未自动执行。
''')
    tests = (output/'tests_final.txt').read_text()
    assert '65 passed' in tests and ' skipped' not in tests
    receipt = dict(experiment='post_st_prognosis_v1',status='COMPLETE',data_status=alignment['status'],
        decision='PREDICTIVE_NO_GO',decision_scope='this frozen post-ST prognosis configuration, not universal framework invalidation',
        branch='research/xai-rsf-post-st-prognosis-v1',training_commit=plan['code_commit'],
        delivery_revision='Git commit containing this receipt; full SHA returned to user after remote verification',
        private_run=str(run),authoritative_evaluation_directory=str(reports),date_basis='announcement for entry and both endpoints',
        training_companies=309,test_2023_companies=36,test_2024_companies=45,
        unique_training_termination_events=14,unique_training_recovery_events=184,
        test_h24_termination_events=31,test_h24_recovery_events=31,test_h24_unknown=0,
        formal_fit_rows_including_seed_repetition=9*309,formal_fits_expected=9,formal_fits_succeeded=9,
        pilot_fits_expected=2,pilot_fits_succeeded=2,failed_fits=0,failed_publications=0,
        failed_preparation_attempts=1,evaluation_correction_without_retraining=True,additional_tuning_trials=0,
        test_recalibration=False,original_M2_retrained=False,full_tests_passed=65,full_tests_skipped=0,
        independent_test_metric_rows=156,model_hashes=validation['model_hashes'],
        jobs={k:json.loads((output/f'{k}_gate.json').read_text()) for k in ['pilot','formal']},
        requested_nodes=['oliver','lemon'],actual_fit_hosts={'oliverdesktop':11},
        source_hashes=alignment['original_source_sha256'],model_input_hashes=plan['source_hashes'],
        immutable_input_manifest_sha256=sha(run/'audit/immutable_input_hashes.json'),
        evaluation_and_delivery_source_sha256={p.name:sha(p) for p in sorted(source.iterdir())
                                               if p.suffix in ['.py','.R'] or p.name=='frozen_protocol.json'},
        public_results_sha256={str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file()},
        documents_sha256={p.name:sha(p) for p in sorted(destination.glob('*.md'))},
        private_artifacts_not_exported=['company records','raw predictions','raw thresholds','forest parameters','individual full explanations'],
        notes=['Seed ranges are not confidence intervals','One company equals one record','Initial raw-day IPCW reports superseded and preserved'])
    write(destination/'execution_receipt.json', receipt)
    print(json.dumps(dict(status='PACKAGED',destination=str(destination),formal_models=9,aggregate_files=len(receipt['public_results_sha256']))))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--reports',type=Path,required=True);p.add_argument('--destination',type=Path,required=True)
    a=p.parse_args();deliver(a.run,a.reports,a.destination)
