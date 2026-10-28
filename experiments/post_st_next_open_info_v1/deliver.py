"""Publish aggregate evidence only; fitted functions and forecasts stay private."""
from pathlib import Path
import argparse,datetime,json,shutil
import numpy as np
import pandas as pd
from experiments.post_st_prognosis_v1.common import arrays,sha,write

LABELS={'market_clock':'A 起点/市场更新','financial_refresh':'B 再更新财务','announcement_fields':'C 再加入公告字段','common':'共同基线'}
MODELS={'xai_q2':'q2','standard_rsf':'标准RSF','training_AJ':'训练AJ'}


def table(frame):
    def fmt(x):
        if pd.isna(x):return '—'
        return f'{x:.6f}' if isinstance(x,(float,np.floating)) else str(x)
    return '| '+' | '.join(map(str,frame.columns))+' |\n|'+'|'.join(['---']*len(frame.columns))+'|\n'+''.join('| '+' | '.join(fmt(x) for x in row)+' |\n' for row in frame.itertuples(index=False,name=None))+'\n'


def metrics(frame):
    d=frame[['scope','variant','model','auc','ap','brier','risk_mean','ipcw_observed_incidence']].copy()
    d['variant']=d.variant.map(LABELS);d['model']=d.model.map(MODELS)
    d.columns=['年度','信息集','模型','AUC','AP','IPCW Brier','平均风险','观察发生率']
    return table(d)


def deliver(run,dest):
    out=dest/'results';out.mkdir(exist_ok=False)
    for p in (run/'evaluation').iterdir():
        assert p.suffix in ['.csv','.json'],p
        shutil.copy2(p,out/p.name)
    for p in (run/'reports').glob('*.csv'):shutil.copy2(p,out/p.name)
    for name in ['data_alignment.json','independent_data_gate.json','pre_fit_freeze.json','pilot_gate.json','formal_gate.json',
                 'environment_oliver.json','environment_lemon.json','tests_pre_freeze.log','tests_fitting_commit.log','tests_final.log']:
        shutil.copy2(run/'audit'/name,out/name)
    m=pd.read_csv(out/'metrics.csv',dtype={'seed':str,'scope':str});w=pd.read_csv(out/'warning_metrics.csv',dtype={'seed':str,'scope':str})
    c=pd.read_csv(out/'ensemble_complexity.csv');s=pd.read_csv(out/'training_diagnostics.csv')
    counts=pd.read_csv(out/'cohort_counts.csv');updates=pd.read_csv(out/'information_updates.csv')
    primary=m[(m.seed.isin(['ensemble','none']))&(m.horizon_months==24)&(m.cause=='termination')]
    q=primary[primary.model=='xai_q2'];qr=q.set_index(['scope','variant'])
    ww=w[(w.seed=='ensemble')&(w.model=='xai_q2')&(w.horizon_months==24)]
    wtab=ww[['scope','variant','top_fraction','checked','captured','target_events','lead3','lead6']].copy()
    wtab['variant']=wtab.variant.map(LABELS)
    delta=qr.loc[('2024','announcement_fields'),'auc']-qr.loc[('2024','market_clock'),'auc']
    assert abs(delta-.186)<1e-12
    (dest/'RESULTS_ZH.md').write_text('''# 结果：2024排序明显改善，概率低估仍未解决

36/36正式任务与4/4试运行已成功完成。本轮没有改模型算法，只移动预测起点、重算随访并有限更新信息集。主结果是三个固定种子的预测先平均，再计算指标；不是挑最佳种子。

A：新起点、更新市场、保留旧财务；B：A基础上更新为截至新起点合格的财务快照；C：B基础上加入警示类型、上一年度审计意见及持续经营非标字段。A/B为75列，C为83列。三组使用同一公司、同一新终点、相同树数/深度/搜索数值预算。

## q2：24个月先最终退市

'''+metrics(q)+'''
从A到C，2024 AUC 0.566→0.752（+0.186），AP 0.526182→0.758339，Brier 0.341580→0.335241。三个C种子的AUC为0.718—0.768，均高于三个A种子的0.536—0.608；这是种子间描述，未做训练不确定性的显著性检验。2023 AUC 0.701389→0.736111，但AP 0.650002→0.576727，不能只报提升。B在2023的AUC最高，却不是AP最高，整体排序与榜首精度不是同一指标。

合并A→C：AUC 0.630102→0.723852，AP 0.507276→0.669576，Brier 0.318542→0.313764。合并AP在81家拼接预测上重算，合并排序也包含年份风险水平差异，不替代分年结果。

**概率水平仍不可信。** C在2023平均退市风险6.27%，观察33.33%；2024平均11.93%，观察44.44%。2024排序更好但平均概率甚至低于A的13.24%。没有用测试发生率再校准，也没有调整输出以掩盖低估。

## 同一核查预算下的覆盖

'''+table(wtab)+'''
Top20主口径：2023 A→C捕获4/12→5/12；2024为4/20→7/20；合并17家核查、8/32→12/32（25.0%→37.5%）。Top25补充为11/32→15/32。C的Top20至少提前3个月捕获10例、至少6个月4例；B的至少6个月为5例，因此C也不是每个提前识别指标都更好。全部测试端点已知，无未知当正常、无重复公司；这是按完整入组年度回顾排序，不是在线核查政策。

**不能把本表直接与旧公告日起点相减来归因。** 起点移动让2023有1例退市进入新的24个月窗口，本轮目标为12/20、合计32例，旧表为11/20、合计31例。旧模型、旧报告全部保留。这里A/B/C间的比较才使用相同的新时钟和标签。

完整数据仍为PROVISIONAL。78/81家测试企业更新了财务报告，但多数是第一季报；25份存在未来版本的上一年年报未强行回填。不能称为原始年报恢复成功、正式验证通过，或政策漂移已经解决。
''')
    (dest/'PREDICTION_COMPARISON_ZH.md').write_text('''# q2与标准RSF的共同信息集比较

标准RSF使用randomForestSRC 3.9.0，全部18个参照模型也重新拟合，以适应新的起点和标签。它们不是旧首次ST预警M2。两模型采用相同公司bootstrap、32树/种子、深度6、mtry75、5个切点候选、原因等权；q2有二阶路径、共享切点和每树25输入限制，缺失原生路由；标准RSF不加阶数限制、训练中位数填补。不能称后端完全等价或把差异全部归因于二阶约束。

'''+metrics(primary)+'''
C信息集下2024 q2 AUC/AP 0.752/0.758339，高于标准RSF的0.536/0.531753；2023 q2 AUC较高，但AP 0.576727低于标准RSF的0.613934。合并q2 AUC/AP较高，而Brier 0.313764略差于标准RSF的0.311675。没有证明所有指标、年度或任务均优于普通RSF。

标准RSF加入公告字段后2023改善、2024下降；q2的年度模式不同。因此这些字段在这一q2配置下有价值的证据，并不意味着所有模型都能同样受益，也不能把某一个算法的较好结果当成普遍结论。

## 24个月首次恢复

'''+metrics(m[(m.seed.isin(['ensemble','none']))&(m.horizon_months==24)&(m.cause=='recovery')])+'''
q2 A→C在两个年度的恢复AUC/AP/Brier均改善，2024恢复AUC 0.502155→0.689655；但C的恢复表现仍不及相同信息集标准RSF（2024 AUC 0.730603、AP 0.601581、Brier 0.219556）。恢复风险仍偏高。两个结局共同满足CIF概率约束，不能据此保证经验校准正确。

## 补充12个月退市结果

'''+metrics(m[(m.seed.isin(['ensemble','none']))&(m.horizon_months==12)&(m.cause=='termination')])+'''
12个月结果预定为补充，不替代24个月主目标。日期移动也改变部分12个月标签，详见label_clock_changes.csv。所有300行指标均独立复核AUC、并列分数AP及Brier；测试随访完整，IPCW权重为1。每家公司一行，公司/记录等权相同。
''')
    changes=pd.read_csv(out/'label_clock_changes.csv')
    (dest/'DATA_ALIGNMENT_AUDIT_ZH.md').write_text('''# 数据对齐：起点已移动，原始财务版本仍未修复完整

预测时刻定义为入组公告后下一全市场交易日开盘前，使用已有交易日历。不是下一次公司复牌，不把日级时间伪装成小时。390家起点延后1—8日，无跨年、无同日起点A/X冲突或风险集排除；行业保留原入组口径，避免同时更换总体。

2023训练2012—2022入组，结局严格早于2023-01-01；2024推进至2012—2023入组，严格早于2024-01-01。由完整原始事件链重新选择首次A／最终X；训练和测试同时重算随访。先恢复后退市仍为先恢复；未知仍删失。覆盖截至2026-09-03，未扩展到今天。

'''+table(counts)+'''
两折共654条训练行，去重345家公司；26个独立先退市训练事件。36次正式拟合共11772条重复训练计算记录，不是新样本。测试81家、32个先退市、31个先恢复、18个两年内无首次事件。每折训练与测试无公司重叠，2023测试公司下一年进入历史训练时仅使用2024年初前已知结局。未评价2025入组测试集；2023/2024企业在2025发生的随访事件可依原协议使用。

## 信息更新量

'''+table(updates)+'''
A保留旧48项财务值，重新计算市场窗口与陈旧度；B选择最大报告期，但要求公告日与所有已记录更正日均早于新起点。增长/变化输入的可得日期取当前与同比组成数据的最晚日期，上一期未可得则保持缺失。旧390家公司全部48项财务、24项市场值先精确复现，独立核验735行折内风险集/终点，三组标签与顺序逐数组一致。实际逐字段时钟及每组财务/市场元数据仅在私有目录。

78/81测试报告期更新：2023最新为第一季报30份、上一年年报2份、其他4份；2024第一季报40份、上一年年报4份、其他1份。表中old_annual_selected统计任何年度的12月报告，不等于上一年度年报。不能把“78家更新”写成“78份原始年报恢复”。此前25份上一年年报仍被未来版本日期阻挡（2023为16、2024为9）；保留合格旧期或其他报告，未将更正后数值回填原公告日。当前财务快照没有独立原版数值证书，即使无更正标志也不升级为PIT认证。

上游版本审计1672行均无original_values_verified证据；另一个旧项目文档用研究者对来源语义的确认解除过阻断，并取消原报试点。这不是本轮独立版本证书，不能覆盖本任务的严格时钟要求。完整修复还需要：原始PDF/原始发布数据、报告版本键、版本发布日期及逐字段对账；变化指标同时需要同比原版。现有研究注册服务连接不可用，已查本地权威审计；没有虚构字段可得日。

C只从入组事件记录读取*ST、上一年度审计意见和持续经营非标，不读取后来恢复/退市文本。所有390家该元数据均有一致取值。六个审计意见哑变量存在冗余；“前一年审计意见”不是公告当天最新审计意见的额外认证，也不等于内控意见。没有加入扣除后营收、政策阈值、未来规则或未核实的规则距离。

## 相对旧公告日起点的端点变化

'''+table(changes)+'''
旧版本数值、历史结果和main均未覆盖；本轮仅部分解决信息时点问题，仍属PROVISIONAL、已暴露结局后的有限探索，不宣称独立盲测。
''')
    cc=c[['test_year','variant','main_functions','interaction_functions','spatial_table_cells','scalar_table_cells','low_support_cell_fraction','zero_support_cell_fraction','reconstruction_error']].copy();cc['variant']=cc.variant.map(LABELS)
    stability=pd.read_csv(out/'seed_stability.csv');st=stability[stability.cause=='termination'].groupby(['test_year','variant'])[['main_jaccard','pair_jaccard','median_function_correlation']].mean().reset_index();st['variant']=st.variant.map(LABELS)
    block=pd.read_csv(out/'announcement_block_contribution.csv');block=block[(block.variant=='announcement_fields')&(block.cause=='termination')]
    bs=block.groupby('test_year').added_field_term_absolute_share.agg(['mean','min','max']).reset_index()
    (dest/'INTERPRETABILITY_COMPARISON_ZH.md').write_text('''# 完整解释继续精确，稳定性仍有不足

每行对应主预测实际使用的三种子96树完整集成，不是单种子复杂度均值。含全部主效应、二元交互、缺失分区、原生事件时间及两个原因；没有剪掉小贡献。贡献在CIF概率尺度上精确相加，不是风险率，也不是因果作用。

'''+table(cc)+'''
A→C的空间格数2023为4902→3869（减少21.1%），2024为4907→4068（减少17.1%）。2023函数数66主效应/145交互→62/144；2024为63/143→65/146，所以不能笼统称函数更少。加入二元字段会改变分区几何，这不是新的复杂度惩罚效果。

最大完整重构误差1.5543e-15，模型重载一致。数学上的精确性仍成立，但低支持格约48%—51%，零支持格反而由19%左右升至20%/23%。审计意见互斥哑变量及变量相关性使乘积参考中的部分组合不可能或未出现；这些网格不提供经验解释可靠性。没有新增可辨识性准则，也没有证明统计可辨识性。

## 三种子稳定性

'''+table(st)+'''
2023交互Jaccard由0.0928降至0.0727，2024由0.0845升至0.0922；仍低。共同函数的形状相关性只针对共同且非恒定项，不代表全部交互稳定。未进行完整训练bootstrap置信推断。

## 新增字段确实是否被森林使用

'''+table(bs)+'''
表中为含新增字段函数的平均绝对贡献占比，分母是所有完整函数的绝对贡献质量；约39%/42%。它既不是CIF变化百分点，也不是预测改善的归因比例。参考样本固定为该折训练公司，各自精确24日历月。

被频繁使用的是上一年度标准无保留审计标记、持续经营非标及无法表示意见。*ST指示在两折各96棵树中均没有被用作分裂，因此不能把本轮收益说成“区分ST与*ST就解决了问题”。公告字段作为一组加入，而且75→83列改变候选覆盖；尚未独立消融每个字段，不能由平均绝对贡献证明某一字段的独立增量。逐变量汇总、种子范围见field_usage_seed_summary.csv；无企业明细或完整函数参数上传。

新信息没有消灭事件稀少：C中无退市公司支持的叶节点比例约2023年63.6%、2024年47.1%。预测提升与大量稀疏局部结构并存，不能据此称解释已经可靠。
''')
    audit=json.loads((out/'independent_audit.json').read_text());formal=json.loads((out/'formal_gate.json').read_text());pilot=json.loads((out/'pilot_gate.json').read_text())
    fitting=json.loads((run/'input/folds/2023/market_clock/plan.json').read_text())['code_commit']
    (dest/'TRAINING_DIAGNOSTICS_ZH.md').write_text(f'''# 训练、核验和资源回执

冻结拟合代码：`{fitting}`。冻结协议先于任何新拟合；116项完整测试通过，包含独立R检查，0跳过。唯一警告是原有EBM缺失值可视化提示，与本轮森林训练无关。没有改动核心森林实现、放宽断言或删失败测试。

4/4试运行、36/36正式任务成功，无训练/发布失败、无选择性重跑。Slurm作业{pilot['job']}／{formal['job']}。4个小试因动态取任务均在oliver完成；正式oliver28、lemon8。两机物理核布局遵守已有P/E与cgroup限制，单任务单线程，未虚称单进程跨机或48物理核。已提供hetero sidecar，使用现有physical/both调度以明确布局；没有声称它是经测量的最快布局。

正式任务池计算及回传{formal['seconds']:.3f}秒；不含准备、开发及审计时间。单任务{s.seconds.min():.3f}—{s.seconds.max():.3f}秒，Python父进程最高RSS {s.peak_rss_MiB.max():.3f}MiB，R子进程单独最高{s.child_peak_rss_MiB.max():.3f}MiB。两者不是进程树峰值之和测量，每任务继承4GiB地址空间限制。模型/预测/完整函数SHA256保留，源与冻结输入未变，任务ID0—35均成功且发布哈希核验通过。

独立重放18个q2森林的{audit['nodes_checked']}个节点、{audit['selected_scores_checked']}次已选分裂和{audit['leaf_aj_checked']}个叶内AJ。最大分数差{audit['max_score_error']:.3e}，AJ差0。另核验18个标准RSF的共同公司bootstrap及训练中位数。全部保存重载误差0，所有原生时刻CIF非负、随时间不减、原因和不超过1，q2路径无超阶。没有裁剪概率。

6个96树完整集成在训练/测试所有公司与全部原生时刻验证，最大误差{audit['max_ensemble_all_native_time_reconstruction_error']:.3e}。300行预测指标独立复算，合并AP重算。森林没有数值梯度优化器，迭代gap不适用，不制造“收敛”指标。所有种子、低支持叶节点和不利结果均保留。

准备阶段的注册服务查询因连接不可用而改查本地权威来源；hetero-submit的帮助探测不支持--help并报错，发生于作业提交之前，无数据/任务副作用。它们不是失败模型，未隐瞒为训练成功数。实验执行日志在私有audit目录。
''')
    (dest/'METHOD_INNOVATION_DECISION_ZH.md').write_text('''# 判断：INCONCLUSIVE；信息修复方向得到局部实测支持

本轮判断对象是“部分信息集修复能否改善24个月ST后预后”，不是新算法，也不是对整个论文框架作一次性成功裁决。

已经支持的结论：在固定q2预算下，公告字段组合提高本次2024样本的观测排序和Top20覆盖，改善两年度的恢复预测，且保留完整精确函数分解。这里“提高”指观测差值，未声称统计显著性。2024 AUC/AP由A的0.566/0.526升到C的0.752/0.758，不能再把该年度原先弱排序仅归因于模型完全学不到信息。

尚未支持的结论：没有完整恢复历史原始年报；概率仍严重低估；2023 AP下降；部分交互选择更不稳定；无退市支持叶节点仍多；新增字段和候选组成同时变化。因此不能称数据不足、政策漂移或解释可靠性已经解决，也不能证明相对标准RSF普遍占优。

机制解释要分开：本轮没有再校准，不存在近零斜率抹掉函数的问题；新增的审计信息确实进入函数，但*ST指示未分裂。结果说明先前信息集遗漏/陈旧是可以改善的方向，不能量化它与2024政策变化各自的因果贡献。普通RSF同字段的年度变化不同，也提示模型与样本的相互作用；没有单字段或政策反事实实验。

解释方法本身未改：bootstrap、生存分裂、叶内AJ、平均、共享切点和q2完整分解均保留。没有可辨识性惩罚的新增独立证据。新增字段属于数据/任务适配，不能包装为独立算法创新。

后续若继续，最优先仍是补齐原报版本证据并用未接触的新时间段验证；当前不根据这轮挑配置继续搜索。本轮到此停止：不追加叶收缩、校准、权重、Focal或2025训练/评价。历史负面实验记录不撤销，当前PROVISIONAL边界保留。
''')
    groups=set();events=set()
    for year in [2023,2024]:
        tr=arrays(run/'input/folds'/str(year)/'market_clock/train_PRIVATE.npz');groups.update(tr['groups']);events.update(tr['groups'][tr['event']==1])
    assert len(groups)==345 and len(events)==26
    receipt=dict(status='COMPLETE_WITH_UNRESOLVED_ORIGINAL_VINTAGE_LIMITATION',decision='INCONCLUSIVE',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),private_run=str(run),fitting_commit=fitting,
        repository='Egon749/XAI-RSF',branch='research/xai-rsf-post-st-prognosis-v1',formal_fits=36,pilots=4,
        failed_fits=0,failed_publications=0,reused_old_predictions=0,tests_passed=116,tests_skipped=0,
        training_fold_rows=654,training_company_union=345,training_termination_union=26,repeated_formal_fit_rows=11772,
        test_companies=81,test_terminations=32,test_recoveries=31,original_financial_vintages_restored=False,
        full_PIT_repair_completed=False,partial_information_experiment_completed=True,
        formal_by_host=formal['by_host'],slurm_jobs=[pilot['job'],formal['job']],
        main_horizon_months=24,secondary_horizon_months=12,algorithm_changed=False,
        calibration_fits=0,leaf_shrinkage_fits=0,new2025_evaluation=False,
        maximum_reconstruction_error=audit['max_ensemble_all_native_time_reconstruction_error'],
        result_sha256={str(p.relative_to(dest)):sha(p) for p in out.iterdir() if p.is_file()},
        code_sha256={p.name:sha(p) for p in dest.glob('*.py')},document_sha256={p.name:sha(p) for p in dest.glob('*.md')},
        protocol_sha256=sha(dest/'frozen_protocol.json'))
    write(dest/'execution_receipt.json',receipt);write(run/'execution_receipt.json',receipt)
    print(json.dumps({k:receipt[k] for k in ['status','decision','formal_fits','pilots','test_companies','test_terminations']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--dest',type=Path,required=True)
    a=p.parse_args();deliver(a.run,a.dest)
