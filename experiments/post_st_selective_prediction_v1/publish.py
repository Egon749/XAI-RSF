"""Publish aggregates only; private predictions, gate thresholds and IDs stay put."""
import shutil,subprocess,platform,importlib.metadata
import pandas as pd
from scipy.stats import spearmanr
from .common import *
from .gate_selection import load_diagnostics

PUB=REPO/f'experiments/{NAME}'

def table(frame):
    def fmt(v):
        if pd.isna(v):return 'NA'
        if isinstance(v,(float,np.floating)):return f'{v:.4f}'
        return str(v)
    return '| '+' | '.join(map(str,frame.columns))+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+''.join('| '+' | '.join(fmt(v) for v in row)+' |\n' for row in frame.itertuples(index=False,name=None))

def main():
    block_forest_fits();assert sha(REPO/'CURRENT_POST_ST_MODEL.json')==REFERENCE_SHA
    reg=load(RUN/'audit/registry.json');gate=load(RUN/'audit/gate.json');selection=load(RUN/'audit/selection_frozen.json');checks=load(RUN/'reports/metric_validation.json')
    assert gate['status']==checks['status']==selection['status']=='PASS'
    for ledger,root in [('input_hashes',RUN/'input'),('input_hashes_attempt2',RUN/'input_attempt2')]:
        for rel,h in load(RUN/f'audit/{ledger}.json').items():assert sha(root/rel)==h
    for p,h in load(RUN/'audit/source_hashes.json').items():assert sha(Path(p))==h
    for name in ['source_hashes','inherited_source_hashes']:
        for p,h in load(SOURCE/f'audit/{name}.json').items():assert sha(Path(p))==h
    d=load_diagnostics();audit=[];model_hash=[];correlations=[]
    for y in range(2019,2026):
        tr=arrays(RUN/f'input/folds/{y}/train_PRIVATE.npz');ev=arrays(RUN/f'input/folds/{y}/eval_PRIVATE.npz')
        assert tr['X'].shape[1]==ev['X'].shape[1]==83 and not set(tr['sample_ids'])&set(ev['sample_ids'])
        assert (tr['year']<y).all() and len(np.unique(tr['groups']))==len(tr['groups'])
        for h in [12,24]:
            t=tr['analysis_time'];e=tr['event'];known=(t>=h)|((e>0)&(t<=h))
            audit.append(dict(cutoff=f'{y}-01-01',months=h,training_companies=len(tr['X']),eval_companies=len(ev['X']),features=83,
                known_termination=int(((e==1)&(t<=h)).sum()),known_recovery=int(((e==2)&(t<=h)).sum()),known_endpoints=int(known.sum()),
                train_hash=sha(RUN/f'input/folds/{y}/train_PRIVATE.npz'),eval_metadata_hash=sha(RUN/f'input/folds/{y}/eval_PRIVATE.npz')))
            if y in YEARS:
                correlations.append(dict(year=y,months=h,cause=1,sd_risk_spearman=float(spearmanr(d[y][f'{h}_1_risk'],d[y][f'{h}_1_seed_sd']).statistic)))
        for seed in SEEDS:
            model_hash.append(dict(year=y,seed=seed,forest_sha256=sha(RUN/f'input/reference/{y}/{seed}/forest_PRIVATE.json'),
                original_prediction_sha256=sha(RUN/f'input/reference/{y}/{seed}/predictions_PRIVATE.npz'),reused=True,new_fit=False))
    pd.DataFrame(correlations).to_csv(RUN/'reports/SD_RISK_ASSOCIATION.csv',index=False)
    pd.DataFrame(audit).to_csv(PUB/'DATA_BOUNDARIES.csv',index=False);pd.DataFrame(model_hash).to_csv(PUB/'MODEL_REUSE.csv',index=False)
    pd.DataFrame([{k:v for k,v in r.items() if k!='path'} for r in reg]).sort_values('year').to_csv(PUB/'NUMERICAL_AND_COST_CHECKS.csv',index=False)
    failed=Path('/home/oliver/.local/share/heteropool-20261005/auto-runs/fb3c7653a6e6456a8a13358c601f02eb/tasks.jsonl.results-2620.json')
    ledger=[]
    for job,path in [('2620',failed),('2621',Path(gate['result']))]:
        for t in load(path)['completed']:
            ledger.append(dict(job=job,task_id=t['id'],year=2019+int(t['id']),host=t['host'],seconds=t['seconds'],
                returncode=t['returncode'],compute_completed=t['compute_completed'],publication_completed=t['publication_completed'],
                reason='standard_library_select_shadowed' if job=='2620' else 'verified_output_hashes'))
    pd.DataFrame(ledger).to_csv(PUB/'EXECUTION_LEDGER.csv',index=False)
    for p in (RUN/'reports').glob('*.csv'):shutil.copy2(p,PUB/p.name)
    shutil.copy2(RUN/'reports/metric_validation.json',PUB/'metric_validation.json')
    m=pd.read_csv(PUB/'METRICS.csv',dtype={'scope':str});mix=pd.read_csv(PUB/'BRIER_CASE_MIX.csv',dtype={'scope':str})
    def subset(c,arms=['all','combined','seed_sd']):
        return m[(m.cause==c)&m.supported&((m.arm=='all')|((m.target_coverage==.8)&m.arm.isin(arms)))]
    annual=subset(1);annual=annual[annual.scope!='pooled']
    baseline=annual[annual.arm=='all'][['scope','months','companies','all_events','auc','ap','brier','risk_mean','observed','original_budget_captured']]
    comparison=annual[annual.arm=='seed_sd'].merge(annual[annual.arm=='all'][['scope','months','brier']],on=['scope','months'],suffixes=('','_all'))
    compact=comparison[['scope','months','accepted','referred','accepted_events','referred_events','brier_all','brier','original_budget_captured']].copy()
    compact.columns=['年','月','接受公司','转交公司','接受退市事件','转交退市事件','原Brier','接受组Brier','原Top20名额下捕获']
    recovery=subset(2);recovery=recovery[(recovery.scope!='pooled')&recovery.arm.isin(['all','seed_sd'])]
    pooled=subset(1);pooled=pooled[(pooled.scope=='pooled')&pooled.arm.isin(['all','seed_sd'])]
    maxerr=max(r['full_component_error'] for r in reg)
    report=f'''# 固定森林的选择性预测实验

**结论：这套简单拒判规则未证明能稳定识别更可靠的预测，不建议接入当前参考模型。** 完成了真实保存模型上的诊断、历史门槛估计和2023—2025有支持窗口的评价；没有重训真实企业森林，没有改变概率。原模型与所有原预测保留。

## 实际执行与比较范围

复用2019—2025各3个种子的21个森林（每年96棵树）。7项诊断均成功，Slurm2621由oliver执行2项、lemon执行5项；含排队17.35秒。先前2620的7项在启动导入时失败，完整保留。另1次汇总因随机对照事件计数类型错误停止，修正并通过测试后重算；不涉及森林、门槛、预算或样本变化。见TEST_FIXES_ZH.md。

在各年1月1日前最近两个完全成熟的历史前向预测队列上，冻结144套点门槛，并以50次公司重抽样重算7200套门槛。四种规则为：五指标等权综合、仅四种支持度、仅三种子预测标准差、仅接受低风险公司的负对照。历史目标接受率80%为主，90%和60%为预定次要分析。目标比例只在历史参考集设定，外层实际接受比例不强行凑足。两个结局与两个时限分别设门槛，不能合称一套统一公司决策。

## 1. 综合支持度规则没有实际筛选作用

综合规则和仅支持度规则，在全部年度、两原因、两个时限和三档历史目标比例下，均接受100%的外层公司。2023/2024/2025分别36/45/53家，拒判数均为0。故其原预测、AUC、AP、Brier、Top20均完全不变。50次历史重抽样下，主分析综合规则也全部接受；这种稳定来自不触发门槛，不是可靠性已获证明。

原始退市结果如下；2025年24个月性能NA，不混入24个月pooled。风险与观察率为0—1数值。

{table(baseline)}

关键问题是**支持指标跨历史模型缺乏可比尺度**。早期森林训练样本较少，退市事件极少；后期森林的叶内公司数、事件数及随访支持自然增加。以2024年24个月退市为例，历史参考2020/2021模型的训练公司数为178/232，当前345。历史到当前：低支持函数贡献占比均值8.19%→2.21%；叶公司不足惩罚0.0669→0.0448；叶退市支持不足惩罚0.8831→0.4184；随访不足惩罚0.3638→0.1437。四种支持指标都明显降低，抵消了种子标准差0.00496→0.01351的上升。

这说明门槛迁移失败：数据量增长使后期企业在指标上普遍更“有支持”，但不代表制度环境变化下的概率误差小。上述数字是已观测分布变化；无法仅靠本实验将变化全部归因于样本量，树结构和公司组成也同时变化。

## 2. 仅用种子分歧：误差下降往往伴随退市企业被转交

下表均使用主分析历史80%门槛。接受表示继续提供自动判断，转交表示拒绝自动判断并等待人工复核；**转交不是正常、低风险或已成功识别**。原Top20名额分别为8、9、11家，应用到接受组内，最多用到其实际公司数。

{table(compact)}

2024年24个月，45家中只接受9家（20%），其余36家转交；20家实际退市中，仅3家留在接受组，17家转交。接受组Brier从全体0.3396降至0.2751，但不能称为整体预测器更准确，更不能把17家转交企业算作已经捕获。原Top20捕获8家，按相同原名额在接受组内只能捕获3家；若改为接受组自己的Top20（2个名额），结果另列完整CSV。

2023年12个月接受组18家中没有退市事件，Brier近0、AUC/AP为NA。这是最直观的“容易样本”效应。2025年12个月接受30家，5个退市事件全部留下，Brier反而从0.0847升至0.1464。种子一致并不等于预测正确，也不能检测三个种子共有的系统性低估。

## 3. 分解Brier变化，避免把事件比例变化当作可靠性改善

采用对称、精确的算术分解：接受组减全体的Brier变化＝事件发生比例变化项＋同结局类别内误差变化项。类别为“该原因在窗口内发生”与“未发生”；后者包括竞争事件与已知无事件。分解不是因果估计，一类缺失则保留NA。

{table(mix[(mix.cause==1)&(mix.arm=='seed_sd')&(mix.target_coverage==.8)][['scope','months','delta_brier','prevalence_mix_component','within_status_component']])}

2024年24个月总变化−0.06444，其中事件比例变化−0.08546，同类别内误差变化反而+0.02102。2023年24个月与2024年12个月也都是事件比例项改善、同类别误差项略恶化。2025年12个月的恶化几乎完全来自接受组退市比例提高。这些结果不支持把Brier下降解释为对不可靠预测的稳定识别。

标准差与预测风险的Spearman相关在2023/2024的12/24个月分别0.674、0.371、0.395、0.562，2025年12个月约−0.033。绝对概率分歧容易兼有风险水平信息，其筛选效果会随时期变化。该相关分析只用于事后诊断，不据此改变门槛。

低风险负对照进一步显示这一陷阱：2024年12个月只接受17家、其中仅1家退市，Brier降至0.0581；2024年24个月接受0家，性能NA。不能把拒绝几乎所有危险企业写成预测成功。

## 4. 随机对照与不确定性

每个年度/原因/时限用1000次共同随机排列，在相同接受公司数下比较随机拒判。2024年24个月种子规则的Brier0.2751位于随机对照95%范围[0.0940,0.5762]内；随机平均保留19.6%的退市事件，种子规则只保留15%。本实验没有清楚的稳定优势证据。随机分布用于描述，表中比例不是正式检验p值。

与综合规则匹配相同接受数的种子、支持及低风险对照全部退化为接受所有公司，因此不能从相同指标推论这些评分方法等效；这项比较本轮没有鉴别力。没有事后强制外层80%拒判来补造有利结果。

另做1000次按评价年度分层的配对公司bootstrap，森林和门槛均固定，统一抽样索引。2024年24个月“原全体−种子接受组”的Brier差0.06444，描述性95%区间[−0.19100,0.28932]；2025年12个月为−0.06174，[−0.13010,−0.01455]。这些比较的公司组成与接受比例不同，不能单凭区间判断拒判优劣。完整420个区间及不可估计次数全部保留，不作选择性显著性表述。

50次历史公司重抽样只重估可靠性经验分布及门槛，不重训森林。主分析种子规则的接受率标准差约3.85—7.93个百分点（五个支持窗口），体现门槛样本不确定性；它不包括完整森林训练及历次模型选择的不确定性。

## 5. 合并退市与恢复结果

{table(pooled[['months','arm','companies','accepted','all_events','accepted_events','event_retention','auc','ap','brier','risk_mean','observed']])}

合并指标在逐公司预测上重算：12个月包含134家，24个月81家。种子拒判后12个月仅留16/25退市事件、Brier0.1738相对原0.1724略差；24个月只留6/32退市事件，Brier由0.3157降至0.2115，同时丢失自动判断范围内的大多数目标事件。不同子集的AUC/AP不可直接当作同队列预测改进。

恢复结果也不一致：种子规则的接受组Brier在前四个窗口均上升，仅2025年12个月下降，并将3家恢复中的1家转交。完整表：

{table(recovery[['scope','months','arm','accepted','referred','accepted_events','referred_events','brier','risk_mean','observed']])}

## 6. 精确解释、失败记录与研究判断

21个森林原概率逐元素复现。7套合并预测器均在全部训练和评价样本、全部原生加展示时间点、两个原因上进行完整基线＋主效应＋二元函数求和；最大绝对误差{maxerr:.3e}，阈值1e−10，保存重载和CIF合法性均通过。原q2风险函数仍然精确，退市风险低估没有被修复。拒判是额外决策规则，不能声称“风险函数乘接受指示”仍必然是q2加性函数。

本轮支持：可从原生函数、叶节点和种子模型提取可审计诊断，保留所有企业预测，明确转交原因和工作量。**本轮不支持：综合评分已能识别可靠企业、拒判普遍降低预测错误、已获得错误率控制保证，或已解决低估/交互不稳定。** 未自动替换模型，也不继续搜索外层表现更好的拒判方案。

后续若单独授权研究，首先应解决历史支持指标随训练规模变化的可比性，并用历史前向损失检验评分是否有信息；同时预先约束复核量和真实事件转交代价。不能直接套用未经验证的叶样本数阈值。当前事件少、时期变化明显，不足以宣称个体可靠性保证。

选择性预测不是本项目首创，可参考[Selective Classification](https://proceedings.neurips.cc/paper/2017/file/4a8423d5e91fda00bb7e46540e2b0cf1-Paper.pdf)与[SelectiveNet](https://proceedings.mlr.press/v97/geifman19a.html)。本轮是固定预测器的简单历史规则实验，没有实现SelectiveNet，也没有证明其理论风险界适用于本数据。

2023—2025均为已暴露的探索队列，继承原始财报版本PROVISIONAL限制。2025年24个月无充分随访，全部性能NA。接受率是公司自动判断比例，事件保留率是全部该原因事件中留在接受组的比例，Top20覆盖另有分母；三者不能混用。
'''
    (PUB/'RESULTS_ZH.md').write_text(report)
    (PUB/'DATA_AND_PROTOCOL_AUDIT_ZH.md').write_text(f'''# 数据与协议审计

保留旧conservative C83列、q2、depth4、3×32树、balanced25、shared_quantile、直接raw父节点AJ10，以及原因权重(1,1)。起点为首次财务/审计ST公告后下一交易日开盘前；恢复2与最终退市1以公告日定义先发生结局，训练上限24日历月。沿用原始字段、缺失与时间定义，没有新财报版本认证。

当前参考指针SHA256：{REFERENCE_SHA}。数据、森林、原预测和所有继承来源哈希在汇总时全部重新核对；未改核心crforest代码。初始协议提交d81e63f9c5e5b5628d3424b5809a00079bb7adb6；成功诊断快照{load(RUN/'audit/code_snapshot_attempt2.json')['commit']}，只修复启动命名冲突，协议哈希不变。

DATA_BOUNDARIES.csv列出2019—2025各年训练人数、在当年截止前已知的12/24月事件、结局支持和输入哈希。2023/2024/2025训练309/345/390家公司，已知24月退市14/26/49家；这些数字不是成熟历史可靠性参考队列中的事件数。最近两个完整成熟OOF队列分别仅4/1/4例12月退市、7/13/16例24月退市，详见HISTORICAL_SUPPORT.csv。

每个历史森林的验证公司不在该森林训练集中；本轮再次核对sample_id不相交，继承原权威按公司groups及字段时钟核验。模型概率与原保存预测逐元素一致，训练企业每组1条。2023企业可在2025门槛的历史参考中出现，因其12月结局当时已成熟；不能进入2023/2024更早的成熟参考窗口。

诊断任务输入只含训练结局、验证X/ID/日历时限，不含验证结局。选择脚本只读取诊断值和按截止时间确定的历史成员，144套门槛及掩码保存后才由评价脚本加载外层标签；既不按外层损失调权重，也不调整门槛。准备阶段只从已有标签文件提取日历时限，不使用其结局决定门槛。

有支持的2023/2024两窗口及2025年12月全部端点已知，沿用原评价器并实测IPCW权重均为1。2025年24月尚有未知结局且无人达到完整时点，所有性能NA；仅保留模型数学验证和接受数量。未知不填为未退市。年度Top20向上取整，pooled先按年选名额再合计；AUC/AP/Brier在合并预测上重算。

两原因的拒判规则独立，不改变联合CIF；未声称其为一个已验证的统一临床/经营决策器。原生函数采用各年训练样本边际乘积参考，支持度则采用真实共同出现的联合格子计数，缺失单列状态；这不是因果识别或相关变量的唯一解释。

企业ID、概率、函数参数、门槛、接受掩码及随机抽样索引均留在{RUN}。公开仅代码、汇总、版本和哈希。当前参考不变，本次无新真实企业森林或概率校准器。单元测试中的合成小样本拟合仅用于验证实现，不计为企业实验。
''')
    (PUB/'README.md').write_text('''# 固定森林选择性预测实验

结果见RESULTS_ZH.md，冻结定义见frozen_protocol.json。依赖既有21个私有森林及历史快照；没有新训练入口。按顺序运行模块prepare、run freeze、run execute、gate_selection、analyze、publish，模块前缀为experiments.post_st_selective_prediction_v1。必须在common.py设置新的唯一RUN，保留SOURCE；prepare拒绝覆盖已有目录。冻结前提交全部代码并通过完整测试。失败启动用run的--attempt N保存独立快照与日志；先检查计算和发布状态，不能无条件重跑。

历史2620导入失败，2621为修复后的7项正式诊断。原规则不变，成功运行使用attempt2。只读重现本轮汇总可使用既有RUN内诊断和冻结掩码；不得重估门槛后覆盖原掩码或公司输出。没有私有数据时只能运行测试，不能宣称复现企业结果。

所有模型和公司级输出保留私有。METRICS含400行（包括2025H24的50行NA）；HISTORICAL_GATE_EVALUATION156行；PAIRED_BOOTSTRAP_95.csv含420行；完整90/80/60目标、四种规则与同接受数对照全部保留。combined匹配对照为100%接受时无鉴别力。MATCHED_RANDOM_REFERENCE的上下界为固定模型随机选公司产生的描述分布，不是正式误差控制保证。
''')
    (PUB/'METHOD_INNOVATION_DECISION_ZH.md').write_text('''# 研究判断

结论：NO_RELIABILITY_GAIN_DEMONSTRATED，仅针对本轮固定的简单拒判规则；不否定所有选择性预测方法。

原生函数和叶支持诊断可审计，风险函数精确性保留。综合规则在外层全接受，无实际拒判收益；仅种子分歧效果随年度变化，Brier下降主要伴随退市事件转交。不存在已验证的可靠性界，不能称为新成功模块，不能用接受组更低Brier替代全体预测表现。

风险预测器的低阶结构设计、固定函数精确解释，与额外选择性预测决策层是两项不同机制。后者不是已有内生可解释性的证明，也不修复原概率低估。继续保留原83/q2/depth4/parentAJ10，不自动部署，不扩展本轮搜索。
''')
    environment=dict(python=platform.python_version(),packages={n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','scikit-learn','matplotlib']},nodes=gate['by_node'],threads_per_task=1,memory_cap_GiB=4)
    (PUB/'environment.json').write_text(json.dumps(environment,indent=2)+'\n')
    assert not subprocess.check_output(['git','diff','e78d287987ed468065bfc84a37098e58b59692ab','--','crforest','CURRENT_POST_ST_MODEL.json'],cwd=REPO)
    tests=(PUB/'TESTS_final.log').read_text();assert '259 passed' in tests and ' failed' not in tests
    receipt=dict(status='COMPLETE_WITH_NEGATIVE_FINDINGS',tests_passed=259,tests_warnings=1,core_and_reference_unchanged=True,protocol_commit='d81e63f9c5e5b5628d3424b5809a00079bb7adb6',
        diagnostic_commit=load(RUN/'audit/code_snapshot_attempt2.json')['commit'],source_commit=load(PUB/'frozen_protocol.json')['source_commit'],
        reference_pointer_sha256=REFERENCE_SHA,protocol_sha256=sha(PUB/'frozen_protocol.json'),models_reused=21,new_forests=0,new_calibrators=0,
        successful_diagnostic_tasks=7,failed_startup_tasks=7,failed_aggregation_attempts=1,failed_job='2620',successful_job='2621',
        compute_and_publication_verified=True,hosts=gate['by_node'],execution_seconds=gate['execution_seconds'],elapsed_with_queue_seconds=gate['elapsed_including_queue'],
        point_thresholds=144,history_bootstraps=50,bootstrap_thresholds=7200,test_bootstraps=1000,random_reference_repetitions=1000,
        metric_rows=400,paired_intervals=420,full_function_max_error=maxerr,probabilities_unchanged=True,original_metrics_reproduced=True,
        unsupported='2025 H24 performance NA',all_source_hashes_unchanged=True,adopted=False,decision='NO_RELIABILITY_GAIN_DEMONSTRATED',
        private_run=str(RUN),code_delivery='this commit and branch; remote verification recorded separately after push')
    (PUB/'execution_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    # Hash every private diagnostic output without releasing its content.
    outputhash=[]
    for r in reg:
        for name,h in load(Path(r['path'])/'hashes.json').items():outputhash.append(dict(year=r['year'],artifact=name,sha256=h))
    pd.DataFrame(outputhash).to_csv(PUB/'PRIVATE_OUTPUT_HASHES.csv',index=False)
    figures(m,mix)
    print(json.dumps(receipt,ensure_ascii=False))

def figures(m,mix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'font.size':10,'pdf.fonttype':42})
    contexts=[('2023',12),('2023',24),('2024',12),('2024',24),('2025',12)]
    def rows(arm):
        return [m[(m.scope==y)&(m.months==h)&(m.cause==1)&(m.arm==arm)&((m.target_coverage==.8)|(m.arm=='all'))].iloc[0] for y,h in contexts]
    a,b=rows('all'),rows('seed_sd');x=np.arange(5);labels=[f'{y}\n{h}个月' for y,h in contexts]
    fig,ax=plt.subplots(2,2,figsize=(13,8.8),layout='constrained')
    ax[0,0].bar(x-.18,[r.brier for r in a],.36,color='#557ba8',label='全体＝综合门槛')
    ax[0,0].bar(x+.18,[r.brier for r in b],.36,color='#cc7858',label='仅种子分歧：接受组')
    ax[0,0].set(title='接受组Brier下降不等于全体预测改善',ylabel='Brier（越低越好）',xticks=x,xticklabels=labels)
    ax[0,0].legend(frameon=False,fontsize=9)
    ax[0,1].bar(x-.18,[r.coverage*100 for r in b],.36,color='#83b4ad',label='实际接受公司比例')
    bars=ax[0,1].bar(x+.18,[r.event_retention*100 for r in b],.36,color='#cc7858',label='退市事件留在接受组的比例')
    for bar,r in zip(bars,b):ax[0,1].annotate(f'{int(r.accepted_events)}/{int(r.all_events)}',(bar.get_x()+bar.get_width()/2,bar.get_height()),ha='center',va='bottom',fontsize=9)
    ax[0,1].set(title='仅种子分歧：转交了多少真正退市企业？',ylabel='比例（%）',ylim=(0,118),xticks=x,xticklabels=labels)
    ax[0,1].legend(frameon=False,fontsize=9,loc='upper left')
    sub=mix[(mix.cause==1)&(mix.arm=='seed_sd')&(mix.target_coverage==.8)&(mix.scope!='pooled')&mix.within_status_component.notna()]
    xx=np.arange(len(sub));ax[1,0].bar(xx-.18,sub.prevalence_mix_component,.36,color='#aaaeb2',label='事件比例变化项')
    ax[1,0].bar(xx+.18,sub.within_status_component,.36,color='#8165a3',label='同结局类别内误差变化项')
    ax[1,0].axhline(0,color='#555',lw=.7)
    ax[1,0].set(title='Brier变化的精确算术分解',ylabel='接受组 − 全体（负值表示下降）',xticks=xx,xticklabels=[f'{r.scope}\n{r.months}个月' for r in sub.itertuples()])
    ax[1,0].legend(frameon=False,fontsize=9,loc='lower right')
    for arm,label,color in [('seed_sd','种子分歧','#cc7858'),('combined','综合门槛','#557ba8')]:
        z=m[(m.scope=='2024')&(m.months==24)&(m.cause==1)&(m.arm==arm)].sort_values('coverage')
        ax[1,1].plot(z.coverage*100,z.brier,'o-',label=label,color=color)
        for r in z.itertuples():
            if arm=='seed_sd':ax[1,1].annotate(f'历史目标{r.target_coverage:.0%}',(r.coverage*100,r.brier),xytext=(3,8),textcoords='offset points',fontsize=8)
    ax[1,1].set(title='2024年24个月：实际接受率与误差',xlabel='实际接受公司比例（%）',ylabel='接受组Brier',xlim=(-2,107))
    ax[1,1].legend(frameon=False,loc='lower right',fontsize=9)
    for v in ax.flat:v.spines[['top','right']].set_visible(False);v.grid(axis='y',alpha=.18);v.set_axisbelow(True)
    fig.suptitle('固定XAI-RSF的拒判实验：尚未证明稳定的可靠性收益\n主表使用历史80%门槛；原概率和完整q2函数不变',fontsize=14)
    fig.savefig(PUB/'SELECTIVE_PREDICTION_SUMMARY.png',dpi=180);fig.savefig(PUB/'SELECTIVE_PREDICTION_SUMMARY.pdf');plt.close(fig)

if __name__=='__main__':main()
