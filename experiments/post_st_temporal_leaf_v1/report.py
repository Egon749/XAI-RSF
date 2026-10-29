"""Build an aggregate-only research handoff from completed, audited outputs."""
from pathlib import Path
import argparse,datetime,importlib.metadata,json,platform,shutil,subprocess,sys
import numpy as np
import pandas as pd
from experiments.post_st_prognosis_v1.common import sha

LABEL={'baseline':'原深度4＋收缩','time_w2':'近期权重2','time_w4':'近期权重4','leaf20':'min_leaf20','selected_policy':'历史选择策略'}


def write(path,value):
    """Refresh only generated public report JSON; never private run outputs."""
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)


def report(root,public):
    public.mkdir(parents=True,exist_ok=True);result=public/'results';result.mkdir(exist_ok=True)
    audit=json.loads((root/'evaluation/independent_audit.json').read_text());assert audit['status']=='PASS'
    selection=json.loads((root/'audit/selection_freeze.json').read_text())
    protocol=json.loads((root/'input/protocol.json').read_text())
    identity=json.loads((root/'input/code_identity.json').read_text())
    preserved=json.loads((root/'audit/step0_freeze.json').read_text())
    for path,h in preserved['preserved_public_sha256'].items():assert sha(Path(path))==h
    copied=[]
    # These directories contain only checked aggregate tables. No recursive copy
    # from input/private/task outputs and no model/function parameter files.
    for source_dir in ['reports','evaluation','summary']:
        for p in sorted((root/source_dir).glob('*.csv')):
            target=result/p.name;assert not target.exists() or target.read_bytes()==p.read_bytes()
            shutil.copy2(p,target);copied.append(p)
    for name in ['independent_audit.json']:
        shutil.copy2(root/'evaluation'/name,result/name)
    for name in ['step0_receipt.json','step0_independent_R_check.json','exposure_2025.json']:
        shutil.copy2(root/'audit'/name,result/name)
    for name in ['full_tests_frozen_code.log','full_tests_delivery.log','full_tests_delivery_final.log']:
        shutil.copy2(root/'audit'/name,result/name)
    (public/'configs').mkdir(exist_ok=True)
    for year in protocol['outer_years']:
        shutil.copy2(root/'input/formal'/str(year)/'config.json',public/'configs'/f'outer_{year}.json')
    write(result/'selection_receipt.json',{k:v for k,v in selection.items() if k!='source_sha256'})
    sources={}
    for name in ['pre_fit_freeze.json','step0_receipt.json']:
        sources.update(json.loads((root/'audit'/name).read_text()).get('source_sha256',{}))
    sources.update(selection['source_sha256']);sources.update(audit['test_source_sha256'])
    for relative,h in json.loads((root/'audit/immutable_input_hashes.json').read_text()).items():sources[str(root/'input'/relative)]=h
    pd.DataFrame([dict(path=k,sha256=v) for k,v in sorted(sources.items())]).to_csv(result/'source_hashes.csv',index=False)
    # Capture actual interpreters and libraries used on each compute node.
    packages=['numpy','pandas','scikit-learn','scipy','pyarrow','pytest','matplotlib']
    local=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),packages={p:importlib.metadata.version(p) for p in packages})
    remote_code="import json,sys,platform,importlib.metadata as m; print(json.dumps(dict(python=sys.version,executable=sys.executable,platform=platform.platform(),packages={p:m.version(p) for p in "+repr(packages)+"})))"
    import shlex
    remote=json.loads(subprocess.check_output(['ssh','-o','BatchMode=yes','lemon','/home/lemon/.venvs/xai-baselines-py313-20261027/bin/python -c '+shlex.quote(remote_code)],text=True))
    write(public/'environment_versions.json',dict(oliver=local,lemon=remote,thread_limits=dict(OPENBLAS_NUM_THREADS=1,OMP_NUM_THREADS=1,MKL_NUM_THREADS=1),
        CRFOREST_RSCRIPT='/home/oliver/MainDiskData/financial_diagnosis/csf_joint_sensitivity_v1/environment/r/bin/Rscript',
        EW_RLIB='/home/oliver/MainDiskData/ernet_original_tau0_full_20261015/dynamic_competing_h24_20261015/rsf_er_competing_20261016/environment/rlib'))
    m=pd.read_csv(result/'termination_ensemble_summary.csv');m['scope']=m.scope.astype(str)
    def metric_table(h,scopes=('2023','2024'),selected=False):
        rows=[]
        for scope in scopes:
            arms=['baseline','selected_policy'] if selected else list(protocol['arms'])
            for arm in arms:
                r=m[(m.scope==scope)&(m.arm==arm)&(m.horizon_months==h)].iloc[0]
                rows.append([scope,LABEL[arm],f'{r.auc:.4f}',f'{r.ap:.4f}',f'{r.brier:.4f}',f'{r.risk_mean:.2%}',f'{r.ipcw_observed_incidence:.2%}',f'{r.observed_minus_predicted_pp:.2f}',f'{r.top20_captured}/{r.known_target_events}',f'{r.top25_captured}/{r.known_target_events}'])
        return table(['入组年','方案','AUC','AP','Brier↓','预测均值','观察发生率','低估差/百分点↓','Top20捕获','Top25捕获'],rows)
    def md(name,text): (public/name).write_text(text.strip()+'\n')

    md('README.md',f'''# 训练队列诊断、时间加权 AJ 与叶节点大小实验

**完成：时间加权只部分缓解低估，同时损失排序；增大叶节点没有一致预测收益。原深度4＋收缩结果保持不变，研究判断为 INCONCLUSIVE。**

本轮先冻结训练队列 AJ 诊断，再冻结有限比较和历史选择规则。只使用预测时点之前的结局选择权重；2023/2024 结果已多次暴露，均为描述性、PROVISIONAL。2025 以前已有实际评价，本轮封存，不能恢复为独立测试。

24个月的全部固定候选，均为三个种子预测平均后重新计算指标：

{metric_table(24)}

2024 年近期权重4把平均风险从11.58%提高到17.05%，Brier从0.3396降到0.3050，但AUC从0.7700降到0.7280、AP从0.7820降到0.7388；仍明显低于描述性观察发生率44.44%。Top20仍捕获8/20。2023年历史规则选择不加权，不能根据已暴露的2023结果改选权重。

![风险水平与排序权衡](figures/prediction_tradeoff.png)

- [数据对齐、训练 AJ 与暴露核查](DATA_ALIGNMENT_AUDIT_ZH.md)
- [冻结设计、实际任务与数值诊断](TRAINING_DIAGNOSTICS_ZH.md)
- [12/24个月、分箱、覆盖及历史选择](RESULTS_ZH.md)
- [完整解释与支持度](INTERPRETABILITY_COMPARISON_ZH.md)
- [逐步方案判断及研究结论](METHOD_INNOVATION_DECISION_ZH.md)
- [复现入口](RUN_ZH.md)、[冻结协议](frozen_protocol.json)、[执行回执](execution_receipt.json)

20/20调度任务成功；34次新森林拟合（4个小试、24个历史验证、6个正式min_leaf20），另外按固定树分区导出加权叶模型；正式产出24个种子模型和8套96树完整函数表。全量测试139项通过。企业明细、预测、模型参数、阈值及函数表保留于服务器：`{root}`。
''')

    aj=pd.read_csv(result/'cohort_aj.csv');rows=[]
    for year in [2023,2024]:
        for h in [12,24]:
            for cohort in ['pre_2021','post_2020']:
                r=aj[(aj.cutoff_year==year)&(aj.horizon_months==h)&(aj.cohort==cohort)].iloc[0]
                rows.append([f'{year}-01-01',cohort,h,r.companies,r.target_events_by_h,r.early_censored,r.risk_set_at_h,f'{r.aj_termination:.2%}' if r.horizon_supported else 'NA（无支持）'])
    md('DATA_ALIGNMENT_AUDIT_ZH.md',f'''# 数据对齐与第0步诊断

本轮复用旧C输入和现有首次财务/审计类ST入组队列，全部非金融行业，每家公司一个入组记录；预测起点仍为公告后的下一交易日开盘前，恢复/退市采用公告日。两个原因是首次恢复与最终退市的竞争结局；这不是再次预测首次ST。

训练截止2023-01-01前：309家公司、14例已观察退市、184例恢复；截止2024-01-01前：345家公司、26例退市、198例恢复。12个月内目标事件只有5/9例。两个训练集有重叠，合并独立公司为345家，不能把309+345称为654家独立企业，或把14+26称为40个独立退市事件。输入均为83列旧C变量，字段顺序、X、公司组、sample_id、时间和事件均与保存模型对应输入核对。未更换年报版本、可得日或随访定义。

真实披露版本仍有继承限制。当前口径沿用旧输入的保守版本选择，不等于通过原始PDF逐值核验或完整PIT认证；研究者此前接受新年度源的另一轮结果亦保留。本轮没有用报告期制造可得日，没有把未知标签视为正常。

训练元数据的stop严格早于对应截止；按实际日历月计算AJ诊断，提前删失留在风险集直到其删失日。未到目标时点且风险集为空、剩余存活未耗尽时记NA，禁止延长水平尾部制造24个月估计。6个月为补充，12/24个月为主要诊断时点。

{table(['信息截止（不含）','入组组','月','公司数','截至H退市','H前删失','H风险集','退市AJ'],rows)}

2024年截止下，post_2020组24个月AJ为21.98%（公司bootstrap95%区间13.63%—31.28%），旧组为3.45%；差18.53个百分点，差值区间9.51—28.31。近期训练信息中已经有风险上升。2023年截止下近期队列24个月无支持，12个月差仅0.50个百分点、区间跨零；不能据此断言真实风险没有上升，更不能断言所有方法都不可能修正。

逐年看，2020入组队列24个月AJ为11.11%，2021队列在2024截止下为15.56%；2022队列已有7例退市但24个月仍无完整支持。2024截止时2023训练入组队列的6个月AJ为11.43%（4例），12/24个月仍无支持。近期混合队列AJ依赖组内非信息删失；不同年份风险与删失时长相关可能破坏这个假设，必须同时看逐年表，不能把混合21.98%当作所有近期公司的已知真实风险。

pre_2021/post_2020按入组年份分组，依据2020-12-31新规公布日作描述性切分；不同板块适用和过渡规则不同，这不是政策因果识别。参见[上交所正式通知](https://www.sse.com.cn/aboutus/mediacenter/hotandd/c/c_20201231_5294528.shtml)、[深交所公告](https://www.szse.cn/aboutus/trends/news/t20201231_584060.html)。

![按入组年AJ](figures/training_cohort_aj.png)

500次公司bootstrap、种子20261028；零事件队列的百分位区间可能退化为0，不代表真实风险为0。使用R survival::survfit独立核对（无事件队列用解析零值），最大误差1.23e-15。完整年份、事件数、删失数、可支持重抽样次数和区间见[cohort_aj.csv](results/cohort_aj.csv)。森林预测仍使用实际天数并逐公司查询12/24日历月，故整体训练日历月AJ与按天数平均查询可能有微小差别。

测试为2023年36家、2024年45家：24个月退市12/20例，恢复15/16例，其余9/9家完整无事件随访。2024入组日期为4月24日至5月6日，最后24个月随访到2026-05-06；本次不是用不足两年的记录假定正常。12个月退市4/16例。两个时点未知结局均0，所以IPCW权重均1，Brier与普通已知二元端点Brier相同。训练中提前删失仍然存在，不能套用测试的完整随访结论。

2025暴露审计只读取以前的汇总回执：`post_st_train2024_h12_v1/execution_receipt.json`已记录COMPLETE、9次正式拟合和2025测试。当前不读取新的2025公司结局、不重新评价；协议状态为SEALED_NO_NEW_EVALUATION，不能写“2025从未执行”或“独立盲测”。

原始来源、训练与预测文件SHA256见[source_hashes.csv](results/source_hashes.csv)；每个正式模型/函数文件哈希见[model_hashes.csv](results/model_hashes.csv)。原结果文件的三个冻结哈希在执行回执中保留并重新核对一致。
''')

    gates={s:json.loads((root/'audit'/f'{s}_gate.json').read_text()) for s in ['pilot','validation','formal']}
    tasks=pd.concat([pd.read_csv(result/f'{s}_tasks.csv') for s in gates],ignore_index=True)
    fold=pd.read_csv(result/'rolling_fold_counts.csv');fold=fold[fold.outer_year==2023]
    md('TRAINING_DIAGNOSTICS_ZH.md',f'''# 冻结设计与执行诊断

拟合代码提交 `{identity['commit']}`；代码归档SHA256 `{identity['archive_sha256']}`。冻结前完成139项测试，包括独立R核验；最终交付仍139项通过，无跳过，1条既有EBM缺失可视化警告。新叶估计器为显式可选函数，旧ForestConfig与旧模型默认行为不变。

所有正式模型固定q2、depth4、32棵树、种子42/101/202、83列输入、原因分裂权重(1,1)、共享切点及原搜索预算。每年最终报告三种子预测平均（等价96树）并重新计算指标；另交付每个种子与范围，三种子范围不是置信区间。

|方案|改变|保持|
|---|---|---|
|baseline|无，复用已核验深度4＋收缩|原树分区、单位权重AJ、直接父节点强度10|
|time_w2 / time_w4|入组>=2021公司权重2/4，同时用于叶和直接父节点AJ|树分区、候选、bootstrap次数、两原因共同权重、收缩系数|
|leaf20|min_leaf从10增为20，重新拟合|单位权重AJ、父节点收缩10、其余搜索/训练预算|

没有“加权+leaf20”组合；没有新因果准则、分裂权重、深度、种子或额外搜索。min_leaf是bootstrap记录数限制，不等于每叶20家独立公司；min_groups仍为5，实际最小独立公司数从5增至9。

滚动验证依次以2019—2022年为入组验证组，拟合只取更早入组公司，结局在该验证年1月1日前截断；评价标签再次限制在外层2023/2024截止之前。两个外层快照的历史拟合/预测输入逐元素相等，故同一验证年/种子只拟合一次，再按两个可用结局快照评价。旧输入PIT局限仍适用。

{table(['验证入组年','训练公司','训练退市','训练恢复'],[[r.validation_entry_year,r.training_companies,r.training_termination,r.training_recovery] for r in fold.itertuples()])}

0事件历史折仍全部保留，未隐藏失败或另选成功种子。2019训练无退市，退市预测全零是该AJ估计器的有限支持结果。

实际Slurm任务：

{table(['阶段','作业号','成功任务','新森林拟合','模型输出','实际主机'],[[s,g['job'],g['tasks'],g['new_forest_fits'],g['model_artifacts'],json.dumps(g['actual_hosts'])] for s,g in gates.items()])}

总20/20任务，34次新森林拟合，80个阶段模型输出（小试8、验证48、正式24）。正式阶段中6个baseline复用、12个固定分区加权替换、6个leaf20新拟合，不能声称正式阶段新训练了24个森林。额外8套合并函数表是确定性导出。所有任务完成计算、NFS发布和SHA256核对；失败训练任务0，发布失败0。历史验证有8个任务在lemon、4个在oliver；小试/正式短任务均被oliver先领取，未虚报两机分担正式拟合。

每个任务单CPU、4GiB地址空间上限、数值库单线程，遵守Slurm分配的P/E核及SMT亲和性，不把单进程称作跨机加速。实际峰值RSS {tasks.peak_rss_MiB.min():.1f}—{tasks.peak_rss_MiB.max():.1f} MiB；每个任务（含导出）{tasks.seconds.min():.2f}—{tasks.seconds.max():.2f}秒。调度执行段小试2.14秒、验证14.28秒、正式6.60秒；含排队分别8.21/14.73/7.70秒，不含开发、数据诊断与报告耗时。这些小样本任务不需要GPU。

数值核验：单位权重替换与原收缩预测逐元素完全相等；6个baseline重载复现，48行原指标一致。全部24个正式种子模型逐叶用整数复制样本的原AJ独立验证加权结果，最大差{audit['maximum_independent_leaf_error']:.3g}。非整数权重另与R核验。所有12个加权模型树分区逐项相同。8个96树森林在训练/测试全部公司、全部事件时间及端点上核对两原因CIF非负、单调、总和不超过1，完整函数重构/保存重载最大误差{audit['maximum_all_time_reconstruction_error']:.3g}。未裁剪预测。森林没有迭代似然求解器，Hessian/gap不适用。

预检错误保留：最初R单元测试未带自定义survival库路径，7通过1失败；修正测试环境路径后8通过。第0步R核验第一次在稀疏队列的未观测factor状态上索引失败，改为实际观测状态并显式处理无事件队列；随后独立核验通过。没有放宽容差、删除测试或因此重训真实任务。原日志保留在私有audit目录。

报告刷新另有一次保护性停止：公共汇总JSON已存在，通用只允许新建文件的函数拒绝覆盖。报告生成器改为仅刷新本轮生成的公共报告JSON后重新成功生成；冻结输入、训练结果和旧实验文件未改动。日志`audit/report_final.log`保留；没有重新拟合。

完整资源、候选评估数、叶支持、阈值数量、实际树路径阶数、单模型复杂度及重构误差见[training_diagnostics.csv](results/training_diagnostics.csv)、[formal_tasks.csv](results/formal_tasks.csv)。
''')

    bins=pd.read_csv(result/'calibration_bins.csv');br=[]
    for year in ['2023','2024']:
        for arm in ['baseline','time_w4','leaf20']:
            d=bins[(bins.scope.astype(str)==year)&(bins.arm==arm)&(bins.cause=='termination')&(bins.horizon_months==24)]
            for r in d.itertuples():br.append([year,LABEL[arm],r.bin,r.n,f'{r.predicted:.2%}',f'{r.target_events}/{r.n}',f'{r.observed:.2%}'])
    recovery=pd.read_csv(result/'metrics.csv');rr=[]
    for year in ['2023','2024']:
        for arm in protocol['arms']:
            r=recovery[(recovery.scope.astype(str)==year)&(recovery.arm==arm)&(recovery.seed=='ensemble')&(recovery.cause=='recovery')&(recovery.horizon_months==24)].iloc[0]
            rr.append([year,LABEL[arm],f'{r.auc:.4f}',f'{r.ap:.4f}',f'{r.brier:.4f}',f'{r.risk_mean:.2%}',f'{r.ipcw_observed_incidence:.2%}'])
    scores=pd.read_csv(result/'weight_selection.csv');cal=pd.read_csv(result/'calibration_gate.csv')
    md('RESULTS_ZH.md',f'''# 预测、校准与事件覆盖

所有数字仅描述已暴露的2023/2024队列。AUC/AP越高通常越好，Brier越低越好；低估差为“观察发生率−平均预测”，单位为百分点。这是不同目的的指标，不能因Brier下降就说排序和覆盖都提高。

## 24个月全部固定候选

{metric_table(24)}

2024近期权重4的Brier下降0.03454（约10.17%），低估差减少5.47个百分点，但仍低估27.40个百分点；AUC下降0.0420、AP下降0.0432。Top20捕获数量相同不意味着企业名单相同：原模型提前3个月/6个月捕获6/2例，加权4为7/2例。

## 12个月全部固定候选

{metric_table(12)}

12个月未解决事件支持：训练目标只有5/9例，测试4/16例。2024权重4使平均风险仅从3.24%变为3.52%，观察为35.56%；AP从0.7184降到0.6109，Top20从6/16降到5/16。2023加权反而降低12个月风险并增加Brier。权重作用于全风险集和两个原因，改变各叶风险集构成，不是统一向上乘一个系数。

## 历史选择与合并策略

{table(['外层截止年','权重','平均历史Brier','支持单元数'],[[r.outer_year,r.recent_weight,f'{r.mean_brier:.9f}',r.evaluable_cells] for r in scores.itertuples()])}

每个验证队列×时点等权，比较权重1/2/4；已知全部端点或H仍有至少5家公司在风险集时才纳入。2023的5个单元预测完全相同，按预定平局规则选权重1。2024选权重4，但总平均Brier仅改善约0.0000394。

**2024的全部选择差异来自2022入组队列12个月零退市这一单元：Brier从0.0008847降到0.0006090，预测均值从2.797%降到2.310%。** 其他单元三个权重预测相同。故不能宣称滚动验证证实了对近期高退市风险的修复能力；它只按规则从很弱的可用信息里选出了一个候选。见[逐单元选择贡献](results/validation_selection_drivers.csv)。

实际历史选择策略为2023权重1、2024权重4。合并AP/AUC在81家公司的合并预测上重算，不平均两个年份指标：

24个月：

{metric_table(24,('pooled',),True)}

12个月：

{metric_table(12,('pooled',),True)}

分别采用相同固定权重的合并结果也全部保留在CSV，不能和逐年历史选择策略混淆。

## 分组校准

每个模型按各自风险分为四组，1最低、4最高；组成员可能随模型变化。每组仅9—12家，不能把组内比例直接当成稳定的个体概率。

{table(['年份','方案','风险组','公司数','预测均值','退市数','观察率'],br)}

2024最高风险四分位的权重4预测均值23.15%，观察为8/11=72.73%，仍严重低估；基线为16.20%、9/11=81.82%。这些已暴露组内结果用于描述，未用于拟合校准器。

## 校准未执行的具体依据

{table(['截止年','时点/月','最新完整历史队列','退市','其他结局','处理'],[[r.outer_year,r.horizon_months,r.calibration_entry_year,r.target_events,r.other_outcomes,'不足10例，跳过'] for r in cal.itertuples()])}

没有为通过门槛改用更老队列，没有估计斜率或强行放大概率。一般非线性logit校准即使单调保序，也不能自动保留最终CIF的纯q2加法函数分解；本轮完全未实施，不能宣称已验证其概率约束和精确解释兼容性。

## 竞争结局恢复也完整报告

{table(['年份','方案','AUC','AP','Brier','预测恢复均值','观察恢复率'],rr)}

两原因使用共同正权重，CIF总概率约束通过。见[全部指标与选择策略](results/metrics_with_selected_policy.csv)、[逐种子范围](results/seed_metric_ranges.csv)、[全部Top20/25及提前3/6个月覆盖](results/warnings_with_selected_policy.csv)。每家公司一个记录，记录等权与公司等权相同；未知结局0、重复预警0。Top20按年度公司数向上取整：2023核查8家、2024核查9家；Top25为9/12家。年度内排序是回顾性核查容量比较，不是假定年初已知道全年入组公司的在线策略。
''')

    complexity=pd.read_csv(result/'ensemble_complexity.csv');support=pd.read_csv(result/'leaf_support_summary.csv');rr=[]
    for r in complexity.itertuples():rr.append([r.year,LABEL[r.arm],r.main_functions,r.interaction_functions,r.spatial_table_cells,r.scalar_table_cells,f'{r.reconstruction_error:.2g}'])
    md('INTERPRETABILITY_COMPARISON_ZH.md',fr'''# 内生完整解释与支持度

当前模型仍为q2竞争风险森林，输出是恢复/退市的CIF。每条路径最多使用两个输入变量，整棵森林可按固定训练参考分布精确编译为：

$$\widehat F_k(t\mid x)=\mu_k(t)+\sum_j f_{{j,k}}(t,x_j)+\sum_{{j<l}}f_{{jl,k}}(t,x_j,x_l).$$

时间加权改变叶节点和直接父节点的AJ估计，不改变任何分裂、缺失路由、bootstrap样本或参考分布。只替换叶函数值，因此保留完整q2表示；函数的形状和幅度可以改变，不能说解释内容不变，也不能说所有贡献只是统一倍增。

加权AJ在树内bootstrap记录上用相同公司权重处理两个原因：

$$Y_w(u)=\sum_i w_i I(T_i\ge u),\quad dN_{{k,w}}(u)=\sum_i w_i I(T_i=u,\delta_i=k),$$
$$\widehat S_w(t)=\prod_{{u\le t}}\left[1-\frac{{\sum_k dN_{{k,w}}(u)}}{{Y_w(u)}}\right],\quad
\widehat F_{{k,w}}(t)=\sum_{{u\le t}}\widehat S_w(u-)\frac{{dN_{{k,w}}(u)}}{{Y_w(u)}}.$$

保留原父节点混合：$\widetilde F_{{\ell,k}}=\alpha_\ell F_{{\ell,k,w}}+(1-\alpha_\ell)F_{{parent(\ell),k,w}}$，$\alpha_\ell=n_\ell/(n_\ell+10)$，$n_\ell$为唯一入袋公司数。使用未经递归收缩的直接父节点。正权重同时作用在风险集和事件数，未制造额外独立事件。实现与[R survival加权AJ说明](https://stat.ethz.ch/R-manual/R-patched/library/survival/doc/compete.pdf)及整数复制法独立核对。

96树合并森林的完整两原因、全原生事件时间函数表：

{table(['年份','方案','主效应','二元交互','空间格数','全时间/原因标量格数','最大重构误差'],rr)}

时间加权的函数数、格数、低支持格比例完全不变。min_leaf20让空间格数减少11.1%/9.9%，但主效应和交互的总数不一定下降：2024变为67个主效应、148个交互；不能用平均叶更大推断解释一定更简单。

叶内独立目标事件支持（所有3×32树叶汇总，同一公司会在不同树重复出现，不能将叶支持加成新的事件数）：

{table(['年份','方案','叶数','零退市叶','零退市叶比例','有效公司支持*'],[[r.year,LABEL[r.arm],r.leaves,r.zero_target_leaves,f'{r.zero_target_fraction:.1%}',f'{r.mean_seed_median_effective_support:.2f}'] for r in support.itertuples()])}

*有效支持为按公司聚合bootstrap重复次数及时间权重后的Kish数，各种子叶中位数再平均，只作支持诊断，不用于改变收缩系数。权重4使有效支持2023从12.74降至9.50、2024从13.88降至10.80；加权不能解决独立事件不足，反而增加权重集中。零退市叶指原始叶内无目标事件，不代表收缩后的CIF必为零。

增大叶节点把零退市叶比例从57.9%/42.5%降到50.9%/36.9%，但2024的24个月Top20从8/20降到7/20，说明“零叶减少必然改善预测”不成立。跨种子二元交互集合Jaccard均值原为0.0727/0.0922，加权完全相同；leaf20为0.0636/0.0818，未改善选择稳定性。这个指标不等于函数形状稳定性。

全部8套完整函数重构与保存重载误差最大1.56e-15；每个种子模型也通过全函数求和、概率及重载核验。贡献是在CIF概率尺度上的可加项，不是对数风险率、风险率乘数或因果效应。没有证明统计可辨识性，没有新的Hessian惩罚，也没有因为精确重构而获得解释可靠性的保证。

模型、真实阈值、完整函数参数和公司级解释均保留私有服务器；Git仅含复杂度、支持、稳定性、误差和哈希。
''')

    md('METHOD_INNOVATION_DECISION_ZH.md','''# 方案判断与停止结论

**研究判断：INCONCLUSIVE。具体实验结论：时间加权可部分缓解24个月风险低估，但有排序代价；没有发现可以同时解决概率偏低、排序和事件稀缺的方案。原深度4＋收缩结果保留，不自动替换。**

|用户方案|本轮证据|判断|
|---|---|---|
|第0步：训练队列AJ|2024截止近期混合组24个月21.98%，旧组3.45%；2023近期24个月无支持|诊断有价值，2024并非完全没有近期信号；不能推断加权必有效|
|第1步：12个月时点|实际完成；训练只有5/9例退市，2024加权后12个月AP和Top20下降|标签更快成熟不等于目标事件数更多，未解决当前事件支持|
|第1步：非首次ST/辅助结局|未纳入；改变风险集、转移目标和解释，需要单独的首次/重复事件及可得日审计|尚未测试，不能报告为有效或无效；不能靠复制记录扩充独立事件|
|第2步：时间加权叶AJ|2024权重4平均风险11.58%→17.05%，Brier0.3396→0.3050；AUC0.770→0.728，AP0.782→0.739|部分缓解低估，但未修复校准，且损失排序；保留候选证据，不直接替代原模型|
|第3步：min_leaf20|零退市叶下降，格数减少约10%；2024的24个月Brier略差、Top20少1例|实现了支持与复杂度变化，未获得一致预测提升|
|第4步：概率映射|最新完整历史校准组只有1/6/0/7例目标事件，低于冻结10例门槛|未拟合；没有放宽门槛或用测试结局放大概率|
|第5步：2025独立评价|已有2025完成评价回执|本轮封存且未执行新评价，不能重新声称独立|

需要修正原建议中两种过强的推断。看到近期AJ上升，只说明有值得检验的信号，不保证旧树分区能利用它，也不保证风险变化就是政策所致。没看到上升，可能来自随访不成熟或事件过少，不能证明不存在变化，更不能证明所有合规预测方法都不可能有所改善。

这轮权重选择尤其脆弱：2024权重4的历史优势全部来自一个零退市的12个月组的预测降低，平均Brier只好约0.0000394；它不是一个证明概率应该上调的可靠校准样本。3个种子和多次暴露的2023/2024结果都不能替代新的独立证据。

解释方面，固定树分区的加权AJ保留精确q2分解，这是已由代码和数值测试支持的实现性质；函数数和交互选择稳定性没有改善，有效样本支持反而下降。不能把“还能精确解释”写成“解释已更可靠”。时间加权AJ本身也不能据此被宣称为独立方法创新。

当前可报告排序、Top20/25覆盖、各分档的历史实际发生率，并同时展示概率低估；描述性分档比例也不是已校准的未来个体概率。保留旧模型及全部候选，停止本轮有限实验，不追加权重、阈值、训练种子或非线性校准。未来任何独立检验需要真正未接触的结局来源；本轮不自行扩大研究任务。
''')

    md('RUN_ZH.md',f'''# 复现与交付位置

仓库分支：`research/xai-rsf-post-st-prognosis-v1`。已完成运行：`{root}`。原结果：`{protocol['reference_run']}`。本轮拟合代码SHA：`{identity['commit']}`；报告与只读聚合脚本可在后续交付提交中追加，不改变冻结训练快照。

完整基础森林配置见[2023](configs/outer_2023.json)、[2024](configs/outer_2024.json)；worker仅按冻结协议替换种子、小试树数，或在leaf20臂替换min_leaf。时间加权不更换这些配置。

执行顺序为`diagnose.py` → `check_step0.py` → `prepare.py --stage prepare` → 全量测试及冻结代码提交 → `prepare.py --stage freeze` → 小试 → 历史验证 → `stages.py --stage select` → 正式任务 → `evaluate.py` → `summarize.py` → `figures.py`/`report.py`。代码入口均在本目录，运行模块时工作目录为仓库根。

`prepare.freeze`生成独立任务JSONL及 `.py.hetero.json` sidecar；本次实际由 `/home/oliver/.local/bin/hetero-run RUN/pilot.py`、`RUN/validation.py`、`RUN/formal.py` 提交。任务明确从DATA_DIR读，只向OUTPUT_DIR写，由池管理计算与NFS发布，成功后再检查每个输出哈希。复现实验应创建新的唯一RUN，不能覆盖既有目录；模型已存在时只读复核，不为重复导出重新训练。

已完成的只读评价入口（输出目录必须新建；勿覆盖现有证据）：

```bash
python -m experiments.post_st_temporal_leaf_v1.evaluate --run RUN
python -m experiments.post_st_temporal_leaf_v1.summarize --run RUN
python -m experiments.post_st_temporal_leaf_v1.figures --run RUN --output FIGURES
python -m experiments.post_st_temporal_leaf_v1.report --run RUN --output PUBLIC_REPORT
```

`evaluate.py`先要求正式任务通过及历史选择冻结，再读取2023/2024标签；它不选择新参数。`summarize.py`增加按历史选择规则逐年组合的合并指标，合并AP重新计算。CSV含每个种子、三种子平均预测、全部候选和两原因/两时点。`report.py`仅显式复制reports/evaluation/summary的汇总表，不复制公司数据、函数参数或模型。

全量测试须设置 `CRFOREST_RSCRIPT` 与 `EW_RLIB`（路径见environment_versions.json），并限制OPENBLAS/OMP/MKL为1；最终139 passed、0 skipped。PNG/PDF为汇总研究图，已人工查看渲染；CSV可逐项复算。所有输入哈希、模型/预测/函数哈希及保存重载结果可核验，私有参数和企业明细需在服务器读取。
''')

    receipt=dict(status='COMPLETE',research_decision='INCONCLUSIVE',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        experiment='post_st_temporal_leaf_v1',private_run=str(root),training_commit=identity['commit'],code_archive_sha256=identity['archive_sha256'],
        branch='research/xai-rsf-post-st-prognosis-v1',repository='Egon749/XAI-RSF',protocol_sha256=sha(public/'frozen_protocol.json'),
        preserved_reference_commit=protocol['reference_commit'],preserved_public_sha256=preserved['preserved_public_sha256'],
        all_expected_tasks=20,tasks_succeeded=20,tasks_failed=0,new_forest_fits=34,pilot_fits=4,rolling_fits=24,formal_new_fits=6,
        formal_models=24,formal_reused_baselines=6,formal_weighted_leaf_replacements=12,formal_leaf20_fits=6,complete_ensemble_function_tables=8,
        training_by_outer_year={'2023':dict(companies=309,termination=14,recovery=184,termination_by12=5),'2024':dict(companies=345,termination=26,recovery=198,termination_by12=9)},
        unique_training_companies_union=345,training_fold_rows_with_overlap=654,
        test_by_entry_year={'2023':dict(companies=36,termination12=4,termination24=12,unknown12=0,unknown24=0),'2024':dict(companies=45,termination12=16,termination24=20,unknown12=0,unknown24=0)},
        historical_selected_weights={y:v['recent_weight'] for y,v in selection['chosen'].items()},selection_created_utc=selection['created_utc'],
        independent_validation=False,provisional_data=True,test_years_previously_exposed=[2023,2024,2025],new_2025_evaluation=False,
        calibration_fits=0,calibration_reason='all four latest-complete historical windows below10 target events',
        full_tests=dict(passed=139,skipped=0,warnings=1,log_sha256=sha(root/'audit/full_tests_delivery_final.log'),warning='existing EBM missing-value visualization warning'),
        preflight_failures_preserved=['R library path missing in first new unit test; corrected environment','unused factor-state indexing in first step0 R check; observed-state factor and analytic no-event handling'],
        reporting_failure_preserved='first report refresh stopped on exclusive JSON creation; public-report-only refresh corrected, no retraining',
        pool=gates,numerical_audit=audit,privacy='aggregate-only; no company rows/forecasts/thresholds/parameters/function values',
        no_automatic_followup_training=True)
    write(public/'execution_receipt.json',receipt)
    # The receipt cannot contain its own commit SHA. The delivery SHA is verified
    # after commit/push and recorded in a separate private sync receipt.
    inventory=[dict(file=str(p.relative_to(public)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(public.rglob('*')) if p.is_file() and p.name!='artifact_manifest.csv' and '__pycache__' not in p.parts]
    pd.DataFrame(inventory).to_csv(public/'artifact_manifest.csv',index=False)
    print(json.dumps(dict(report=str(public),artifact_count=len(inventory),status='COMPLETE'),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();report(a.run,a.output)
