"""Export aggregate 24-month experiment evidence; no private model parameters."""
from pathlib import Path
import argparse,datetime,json,shutil
import pandas as pd
from experiments.post_st_prognosis_v1.common import arrays,sha,write
from experiments.post_st_cause_weights_v1.deliver import table,LABELS


def deliver(run,dest):
    result=dest/'results';result.mkdir(exist_ok=False)
    for path in (run/'evaluation').iterdir():
        if path.suffix in ['.csv','.json']:shutil.copy2(path,result/path.name)
    shutil.copy2(run/'reports/cohort_counts.csv',result/'cohort_counts.csv')
    for name in ['data_alignment.json','pre_fit_freeze.json','pilot_gate.json','formal_gate.json','environment_oliver.json','environment_lemon.json','tests_before.log','tests_pre_freeze.log','tests_final.log']:
        shutil.copy2(run/'audit'/name,result/name)
    m=pd.read_csv(result/'metrics.csv',dtype={'scope':str,'seed':str});w=pd.read_csv(result/'warning_metrics.csv',dtype={'scope':str,'seed':str})
    c=pd.read_csv(result/'ensemble_complexity.csv');s=pd.read_csv(result/'training_diagnostics.csv')
    protocol=json.loads((run/'audit/protocol.json').read_text())
    primary=m[m.seed.isin(['ensemble','none'])&m.cause.eq('termination')]
    def metrics(frame):
        d=frame[['scope','model','auc','ap','brier','risk_mean','ipcw_observed_incidence']].copy();d['model']=d.model.map(LABELS)
        return table(d.rename(columns={'scope':'评价期','model':'模型','auc':'AUC','ap':'AP','brier':'IPCW Brier','risk_mean':'平均风险','ipcw_observed_incidence':'观察发生率'}))
    focus=primary[primary.model.isin(protocol['weights'])]
    ww=w[w.seed.eq('ensemble')&w.model.isin(protocol['weights'])][['scope','model','top_fraction','checked','captured','target_events','lead3','lead6']].copy();ww['model']=ww.model.map(LABELS)
    (dest/'RESULTS_ZH.md').write_text('''# 24个月完整历史实验：原因加权没有稳定改善

18/18正式模型、2/2试运行成功；108项完整测试通过。森林拟合使用309／345家、14／26个先退市事件，已经不再是12个月校准分割中的2／4例。三个权重匹配同一份数据、配置、随机种子与预算；未用Focal Loss、重新校准或新增滚动记录。

本轮结论 **INCONCLUSIVE（原因加权的可泛化增益未确立）**。在已观察的这组结果中，退市双权未改善主指标；只看退市在2024局部改善，但2023显著数值下降，合并AUC/AP及Top20覆盖低于默认。这里的“下降”是观测差异，不表示已完成统计显著性检验。没有选择新默认模型，不改变此前未加权24个月配置的负面研究记录。

## 主预测：三个固定种子的等权预测集成

'''+metrics(focus)+'''
AUC/AP越高越好，Brier越低越好；平均风险不是越高越好。合并指标在81家公司的拼接预测上重新计算，不是年度或种子指标均值。合并排序还含年份风险差异，必须同时查看分年结果。

2023只看退市：AUC 0.640000→0.443636、AP 0.546345→0.364569、Brier 0.270539→0.274384；三个种子的AUC都低于对应默认种子。2024只看退市：AUC 0.560000→0.592000、Brier 0.342960→0.328165，但集成AP 0.560032→0.539200。没有倒置2023分数、删种子或更换权重来制造改善。

## 固定预算的独立退市覆盖

'''+table(ww)+'''
主口径Top20%：两年合并默认捕获10/31，退市双权9/31，只看退市8/31。补充Top25%：默认10/31，双权10/31，只看退市11/31。2024只看退市的Top25从5/20升至8/20，不能忽略2023从5/11降至3/11的代价，也不能因为Top25更有利而把它替换为本轮主要目标。

至少提前6个月的合并捕获在只看退市下提高（Top20为2→4，Top25为2→6），这是局部收益；不能与其他配置最好的AUC拼成一个模型。年度核查数分别为8／9（2023）、9／12（2024），合并17／21。测试未知0、重复预警0；已知非目标计数完整保留，未捕获事件不视为正常。年度全队列回顾排名不称为在线策略。

风险水平仍严重低估：默认2023平均5.66%对观察30.56%，2024平均13.02%对观察44.44%；只看退市分别6.09%、14.68%，仍不足。24个月标签改变事件发生率，不能把本表与上一轮12个月指标直接作纯期限效果比较；历史范围也同时扩大。
''')
    (dest/'PREDICTION_COMPARISON_ZH.md').write_text('''# 同口径外部参照及恢复预测

标准RSF与同后端无阶数限制森林复用此前同训练／测试口径的24个月冻结预测，未重训。它们仍为原因等权；不能宣称加权q2优于同样加权的标准RSF，也不能把它们叫作原首次ST预警M2。六个本轮等权q2与历史树结构、训练和测试预测完全一致。

'''+metrics(primary)+'''
2023默认q2在本次样本中优于标准RSF；2024无阶数限制森林的AUC/AP更高。标准RSF自身也存在较弱的退市排序和概率低估，因此“接近标准RSF”不等于达到可用的预后预测水平。

## 恢复结局

'''+metrics(m[m.seed.isin(['ensemble','none'])&m.cause.eq('recovery')])+'''
只看退市仍使用同一叶内AJ预测恢复，两个原因CIF相容；但恢复表现并未因此改善。2024恢复AP由0.365614降至0.333880，Brier由0.260072增至0.270291；标准RSF恢复AUC为0.771552。只优化退市分裂并不免费保住恢复预测。

每家公司一行，记录等权与公司等权完全相同。24个月测试端点全部已知，既有日历月IPCW协议算得权重均为1；未知训练结局仍作为删失参与森林拟合。并列分数AP、AUC与Brier独立复核。所有个体预测固定，没有使用测试结局校准或挑种子。
''')
    cc=c[['test_year','variant','trees','main_functions','interaction_functions','spatial_table_cells','low_support_cell_fraction','reconstruction_error']].copy();cc['variant']=cc.variant.map(LABELS)
    stability=pd.read_csv(result/'seed_stability.csv');st=stability[stability.cause.eq('termination')].groupby(['test_year','variant'])[['main_jaccard','pair_jaccard','median_function_correlation']].mean().reset_index();st['variant']=st.variant.map(LABELS)
    (dest/'INTERPRETABILITY_COMPARISON_ZH.md').write_text('''# 精确解释保留，复杂度和稳定性不能混为一谈

下表为主预测实际使用的三个32树模型等权集成，即96树完整函数表；不是单种子复杂度均值。空间格包含精确切点分区及缺失状态，完整时间与两个原因的标量表规模见CSV。没有舍弃小贡献、合并阈值或把代理解释充当原模型。

'''+table(cc)+'''
只看退市使2023格数5034→2890（减少42.6%），2024格数4794→3430（减少28.5%）。但2023主效应60→62、交互152→154，函数数量反而略增；该年的格数下降主要表现为每个函数的分区减少，不能笼统写成“函数更少”。2024主效应61→58、交互138→126。

完整函数重构两个原因CIF的最大误差1.33e-15；贡献是概率贡献，不是风险率贡献或因果影响。没有加入newRSF2可辨识性惩罚，也没有证明统计可辨识性。

## 跨种子稳定性（同一折固定训练参考、730天）

'''+table(st)+'''
只看退市的交互Jaccard在2023从0.069降至0.049，2024从0.108略升至0.112。共同非恒定函数形状的相关性提高，只描述交集，不能解释成全部交互选择更稳定。2024主效应选择重叠也下降。三个种子的范围和相关性没有置信区间含义。

本轮全部bootstrap树均含退市事件且无树桩，与之前2／4例训练时的退化不同。尽管事件增多，只看退市仍有约44%／35%的叶节点无退市公司支持，低支持格比例约44%／45%；局部函数可靠性仍受限制。详细单种子、节点支持与完整集成表规模均保留。
''')
    counts=pd.read_csv(result/'cohort_counts.csv')
    (dest/'DATA_ALIGNMENT_AUDIT_ZH.md').write_text('''# 完整历史与24个月时钟对齐

'''+table(counts)+'''
2023训练为2012—2022入组，结局严格早于2023-01-01；2024训练为2012—2023入组，结局严格早于2024-01-01。两折分别使用此前已冻结审计的24个月输入，逐文件核验其公开回执SHA256，并从首次A/X公告和状态链重新验证所有训练标签、时间及资格。测试81条记录也重新验证首次A/X，日期采用公告日。

75列入组时特征及顺序不变，所有特征行与对应历史参照一致。训练／测试在各折内无公司交集；2023测试企业在2024折进入历史训练是预定年度推进，只能使用2024年初之前已知结局。两折训练共654条（有重复），去重345家公司、26个已知先退市事件；18次正式拟合共5886条重复计算记录，不称为新增样本。测试去重81家、31个先退市、31个先恢复。

训练49／44条提前删失并未当作正常。预测目标是首次ST后24个月内先恢复A还是先最终退市X；第12个月之后才发生的事件进入本轮目标，先恢复后再退市仍记为首次恢复。中间警示变更不算恢复。逐行验证h24实际730／731天和精确24日历月一致，不用730天统一截断闰年。

既有覆盖证据截止2026-09-03，不是当前日期。原始披露版本、行业历史版本及首次事件历史完整性限制全部继承，仍为PROVISIONAL。公告日起点未改，同日年报仍被排除，不能声称已修复PIT。测试结局此前已接触，不是盲测；不评价2025，不追加数据或阈值搜索。

本轮未再留出近两年作为校准集。相对上一轮12个月权重实验，训练历史与期限同时改变；本轮内部可以公平比较权重，但不能把跨轮差异全部归因于24个月。原12个月及更早24个月结果均保留。
''')
    audit=json.loads((result/'independent_audit.json').read_text());formal=json.loads((result/'formal_gate.json').read_text());pilot=json.loads((result/'pilot_gate.json').read_text())
    fitting=json.loads((run/'input/folds/2023/equal/plan.json').read_text())['code_commit']
    (dest/'TRAINING_DIAGNOSTICS_ZH.md').write_text(f'''# 实际执行与完整核验

冻结拟合代码：`{fitting}`。原106项测试通过；增加24个月事件、拟合截止与闰年端点检查后108项全部通过，独立R核查包含在内，0跳过。唯一警告为已有EBM缺失值可视化提示。核心原因权重实现未变，未放宽门槛。

2/2四树试运行成功，18/18正式训练成功，无失败或回传失败，无选择性重跑。Slurm作业{pilot['job']}／{formal['job']}均完成且退出码0；试运行oliver1、lemon1，正式任务oliver10、lemon8。物理核池独立单线程任务使用NFS及校验回传；遵守P/E核、SMT与cgroup限制，不声称48个物理工作核或单进程跨机器提速。没有GPU拟合。

正式任务池计算及发布约{formal['seconds']:.3f}秒，不代表整个开发、测试、审计时间。单任务墙钟{s.seconds.min():.3f}—{s.seconds.max():.3f}秒，峰值RSS最高{s.peak_rss_MiB.max():.2f}MiB，每任务地址空间上限4GiB。模型和函数文件哈希见results/model_hashes.csv，环境版本来自两机实际探测。

六个等权q2逐树和训练／测试预测精确复现历史模型。18个模型保存重载误差均0。独立审计重放{audit['nodes_checked']}个节点与公司bootstrap，检查{audit['selected_scores_checked']}个选中分裂和{audit['leaf_aj_checked']}个叶内AJ；最大分数差{audit['max_score_error']:.3e}，AJ差0。有限候选森林没有梯度求解器，因此收敛迭代gap不适用，不虚构此指标。

所有原生时间CIF满足非负、时间不减及双原因和不超过1，q2路径无违例。六个96树集成完整重构误差最大{audit['max_ensemble_all_native_time_reconstruction_error']:.3e}，未裁剪预测。126行AUC/AP/Brier经独立计算验证；合并AP重新计算。实际函数数量、格数、稀疏支持、单种子范围与全部不利结果公开保留。
''')
    (dest/'METHOD_INNOVATION_DECISION_ZH.md').write_text('''# 本轮判断：INCONCLUSIVE

判断范围是“24个月完整历史下，原因加权能否稳定改善q2森林”，不是重新判定整个XAI-RSF框架或此前首次ST预警任务。已观察结果不支持用加权方案替换默认，也不支持宣称加权解决了数据不足。样本仍有限且测试已暴露，不能由这轮证明任何数据场景下都不存在收益。

默认等权在两个年度集成AP和合并Top20%覆盖上优于两个加权方案。退市双权缺乏稳定收益；只看退市在2024的AUC、Brier、Top25及提前识别有所改善，但2023排序和覆盖下降，合并AUC/AP与Top20覆盖下降。不能事后改以Top25作为主目标，不能选择性报告2024。

解释保持精确，格数减少；但2023函数数量不降，交互选择重叠也下降，未证明解释整体更可靠。提高原因权重是已有方法，本轮是任务适配消融，不是独立方法创新或可辨识性惩罚验证。

实际平均退市风险仍明显低于观察发生率，恢复预测也没有稳定受益。此前未加权24个月实验的负面记录保持不变，本轮不以训练成功或个别指标改善宣布方法成功。没有运行Focal Loss，不能判断其效果。

本轮到此停止；保留所有种子、权重和两年度结果，不自动追加超参数、动态窗口、政策模型或2025测试。
''')
    groups=set();events=set()
    for year in [2023,2024]:
        a=arrays(run/'input/folds'/str(year)/'equal/train_PRIVATE.npz');groups.update(a['groups']);events.update(a['groups'][a['event']==1])
    assert len(groups)==345 and len(events)==26
    receipt=dict(status='COMPLETE',decision='INCONCLUSIVE',decision_scope='stable benefit of cause weighting under full-history 24-month prediction',
        provisional=True,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),private_run=str(run),fitting_commit=fitting,
        formal_fits=18,pilots=2,reused_comparator_fits=12,historical_q2_exact_reproduction=6,
        failed_fits=0,failed_publications=0,calibration_fits=0,focal_loss_fits=0,tests_passed=108,tests_skipped=0,
        training_company_union=345,training_termination_union=26,training_fold_rows=654,repeated_formal_fit_rows=5886,
        test_companies=81,test_terminations=31,test_recoveries=31,horizon_calendar_months=24,reference_runs=protocol['reference_runs'],
        slurm_jobs=[pilot['job'],formal['job']],formal_by_host=formal['by_host'],
        maximum_reconstruction_error=audit['max_ensemble_all_native_time_reconstruction_error'],
        result_sha256={str(p.relative_to(dest)):sha(p) for p in result.iterdir() if p.is_file()},
        code_sha256={p.name:sha(p) for p in dest.glob('*.py')},document_sha256={p.name:sha(p) for p in dest.glob('*.md')},
        protocol_sha256=sha(dest/'frozen_protocol.json'))
    write(dest/'execution_receipt.json',receipt);write(run/'execution_receipt.json',receipt)
    print(json.dumps({k:receipt[k] for k in ['status','decision','formal_fits','failed_fits','training_company_union','training_termination_union','test_companies','formal_by_host']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--dest',type=Path,required=True);a=p.parse_args();deliver(a.run,a.dest)
