"""Publish aggregate-only research evidence; private forecasts never leave the run."""
from pathlib import Path
import argparse,datetime,json,shutil
import numpy as np
import pandas as pd
from experiments.post_st_prognosis_v1.common import arrays,sha,write

LABELS={'equal':'默认等权(1,1)','double_termination':'退市双权(2,1)','termination_only':'只看退市(1,0)',
        'unrestricted':'同后端无阶数限制','standard_rsf':'标准RSF（复用）','training_AJ':'训练AJ基线'}


def table(df):
    def value(x):
        if pd.isna(x):return 'NA'
        return f'{x:.6f}' if isinstance(x,(float,np.floating)) else str(x)
    return '| '+' | '.join(map(str,df.columns))+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+''.join('| '+' | '.join(value(x) for x in row)+' |\n' for row in df.itertuples(index=False,name=None))


def deliver(run,dest):
    result=dest/'results';result.mkdir(exist_ok=False)
    for path in (run/'evaluation').iterdir():
        if path.suffix in ['.csv','.json']:shutil.copy2(path,result/path.name)
    for name in ['cohort_counts.csv']:shutil.copy2(run/'reports'/name,result/name)
    for name in ['data_alignment.json','pre_fit_freeze.json','pilot_gate.json','formal_gate.json','environment_oliver.json','environment_lemon.json','tests_before.log','tests_pre_freeze.log','tests_final.log']:
        shutil.copy2(run/'audit'/name,result/name)
    m=pd.read_csv(result/'metrics.csv',dtype={'scope':str,'seed':str});w=pd.read_csv(result/'warning_metrics.csv',dtype={'scope':str,'seed':str})
    c=pd.read_csv(result/'ensemble_complexity.csv');s=pd.read_csv(result/'training_diagnostics.csv')
    primary=m[m.seed.isin(['ensemble','none'])&m.cause.eq('termination')].copy()
    def display_metrics(frame):
        d=frame[['scope','model','auc','ap','brier','risk_mean','ipcw_observed_incidence']].copy();d['model']=d.model.map(LABELS)
        return table(d.rename(columns={'scope':'评价期','model':'模型','auc':'AUC','ap':'AP','brier':'IPCW Brier','risk_mean':'平均风险','ipcw_observed_incidence':'观察发生率'}))
    focus=primary[primary.model.isin(['equal','double_termination','termination_only'])]
    warning=w[w.seed.eq('ensemble')&w.model.isin(['equal','double_termination','termination_only'])].copy()
    warning['model']=warning.model.map(LABELS)
    warning=warning[['scope','model','top_fraction','checked','captured','target_events','lead3','lead6']]
    (dest/'RESULTS_ZH.md').write_text('''# 原因权重有限实验：局部收益与排序损失并存

18/18正式拟合和2/2试运行成功。仅改变竞争风险分裂的原因权重，未使用Focal Loss、过采样、样本类别加权、重新校准或新增滚动记录。六个默认等权q2模型与旧模型逐树及逐预测数组完全一致。

结论为 **INCONCLUSIVE**：只针对退市分裂改善合并Top20%覆盖和Brier，并缩小解释表；但2024年AUC/AP下降，跨种子证据不一致，训练独立退市支持只有2／4例，尚不能作为全面优于默认配置的证据。退市双权也未稳定改善结果。不能由本轮推断所有代价敏感方法无效，Focal Loss尚未实测。

## 三个匹配q2方案

主结果是三个种子预测先等权平均，再计算指标。AUC/AP越高越好，Brier越低越好；风险均值并非越高越好。合并指标在81家公司的拼接预测上重新计算，并非年度指标均值。

'''+display_metrics(focus)+'''
## 固定核查预算

Top20%为主，Top25%为补充。每年分别选取ceil(n×比例)，合并行为两年选择的并集。排序在完整年度队列上回顾计算，非实际可在线执行的策略。所有测试结局已完整，未知核查结果0；每家公司一条记录，重复预警0。未捕获事件不能视作正常。

'''+table(warning)+'''
只看退市时，合并Top20%从6/19升至8/19，Top25%从7/19升至8/19；至少提前3个月捕获由4升至6，提前6个月各方案均为0。2023只看退市的3个月提前捕获为1，双权为2，不能把不同权重的最好指标拼成一个模型。

2023只看退市的AUC由0.640625升至0.734375；2024则由0.742222降至0.642222，AP由0.623799降至0.596688。合并AP几乎持平（0.541449→0.537958），AUC略降（0.751273→0.735993），Brier下降（0.226307→0.213958）。同权重在不同年份的方向不一致。

2024只看退市的平均预测仍仅4.66%，观察退市率33.33%，严重低估并未解决。不能把风险升高本身称作校准成功，也不能借用上一轮仿射校准后的数值评价本轮未校准模型。

完整种子、恢复结局、校准分组、模型哈希和数学核查见results目录。仅描述种子差异，不报告显著性或完整训练不确定性。2023/2024此前已接触，原始披露版本仍有PIT限制，整体为PROVISIONAL探索性结果。
''')
    (dest/'PREDICTION_COMPARISON_ZH.md').write_text('''# 同记录预测比较

标准RSF与同后端无阶数限制森林复用上一轮冻结预测，未重训；它们的权重仍为(1,1)，所以只能作为外部参照，不能据此声称目标加权q2胜过同样目标加权的标准RSF。这里的标准RSF不是早期首次ST预警任务的M2。

'''+display_metrics(primary)+'''
## 恢复结局

'''+display_metrics(m[m.seed.isin(['ensemble','none'])&m.cause.eq('recovery')])+'''
2023没有恢复事件，AUC/AP未定义，不能填0或声称高准确度。只看退市仍由同一叶内AJ输出恢复CIF，未忽略竞争风险；2024恢复AP却由0.147834降至0.093056。它的恢复Brier有所下降，但不等于恢复排序也改善。

测试端点全部已知，按既有精确日历月删失协议计算的IPCW权重均为1。记录与公司等权相同。合并排序也受年份风险水平差异影响（训练AJ合并AUC高于0.5），应同时查看分年结果。Top20/25排序以sample_id固定打破平分，不能按标签打破平分。
''')
    cc=c[['test_year','variant','trees','main_functions','interaction_functions','spatial_table_cells','low_support_cell_fraction','reconstruction_error']].copy();cc['variant']=cc.variant.map(LABELS)
    st=pd.read_csv(result/'seed_stability.csv');ss=st[st.cause.eq('termination')].groupby(['test_year','variant'])[['main_jaccard','pair_jaccard','median_function_correlation']].mean().reset_index();ss['variant']=ss.variant.map(LABELS)
    (dest/'INTERPRETABILITY_COMPARISON_ZH.md').write_text('''# 完整解释仍精确；复杂度下降不等于解释可靠性已解决

下表对应主预测实际使用的三个32树模型等权集成，即96树完整函数表。不是把单种子函数数量取平均冒充集成模型。完整函数包括两个原因与全部原生时间；空间格数包含精确分区及缺失状态，未裁剪、合并切点或仅保留Top变量。

'''+table(cc)+'''
只看退市使2023的完整集成空间格数3577→839（减少76.5%），主效应63→48、交互168→71；2024为4077→1214（减少70.2%），主效应64→56、交互161→96。仍有数十个交互，不能称作已经简洁到无需审计。

完整分解在CIF概率尺度上重构两个结局，最大误差5.55e-16；不是风险率贡献，也不是因果效应。没有加入newRSF2的可辨识性惩罚，不能把本轮变化归为可辨识性机制的独立收益。

## 跨种子描述

'''+table(ss)+'''
表中的函数相关性是每对种子共同、非恒定函数在同一训练参考样本、固定365天上的相关性中位数，再对三对种子取均值。它只描述交集内形状，不能代替全部函数选择稳定性。只看退市的交互Jaccard仍仅约0.042／0.097，主效应选择重叠反而下降，不能说整体解释已经稳定。

复杂度缩小也与目标事件不足有关：2023每个32树森林有2—7棵bootstrap完全没有退市，目标单原因分裂时这些树不再分裂，而默认方案仍能依据恢复分裂。只看退市下，约59%—61%的叶节点没有退市公司支持；小函数表不代表其余函数得到充分估计。低支持格比例、无支持格比例、单种子复杂度完整保留在CSV。
''')
    counts=pd.read_csv(result/'cohort_counts.csv')
    (dest/'DATA_ALIGNMENT_AUDIT_ZH.md').write_text('''# 数据与时间对齐

所有输入逐文件复用此前冻结数据，特征顺序和75列内容未改变。训练／测试组无重叠，每家公司一行；标签按首次全恢复A与最终终止上市X公告日定义。中间警示变化不计恢复，未知结局不作正常。终点上限12个日历月；覆盖证据仍截止2026-09-03，并非运行当日。

'''+table(counts)+'''
2023训练：2012—2020入组，拟合信息严格早于2021-01-01。2024训练：2012—2021入组，信息严格早于2022-01-01。保留上一轮校准留出安排以便隔离权重效应；本轮不利用校准标签拟合任何参数。不是2012—2022/2023全历史扩样，也不是2023年6月30日截断的344家口径。

训练两折合计509条记录（重复公司未去重），去重277家公司；两折可见退市事件去重4例。18次正式拟合累计4581条训练记录仅是重复计算量，不是新增样本。测试去重81家、19退市、4恢复，所有12个月端点已知。两个测试年份此前已经接触，不称严格盲测。

预测原点仍是旧公告日口径；同日年报未纳入。原始披露版本、历史行业版本和部分首次事件历史完整性限制全部继承；不宣称PIT认证或上游问题已经修复。原始企业、预测、阈值和函数值留在私有运行目录，公开只有汇总和SHA256。
''')
    audit=json.loads((result/'independent_audit.json').read_text());formal=json.loads((result/'formal_gate.json').read_text())
    (dest/'TRAINING_DIAGNOSTICS_ZH.md').write_text(f'''# 执行与数值核查

完整测试：修改前91通过；修改后及交付前106通过，0跳过，唯一警告为既有EBM缺失值可视化提示。新增测试覆盖权重非法输入、独立风险集统计量、真实训练使用权重、默认／显式等权一致、叶内AJ不变、旧模型兼容、保存重载及精确重构，以及两组原因权重下与randomForestSRC独立R实现的单分裂核查。

冻结拟合代码：`4fc3681de82d563e134f8257109380fd14ccc13e`。同一代码完成2个4树试运行与18个32树正式拟合；无失败、无重训、无出版回传失败。2个试运行均在oliver完成；正式任务实际oliver12个、lemon6个。Slurm作业2542／2543均COMPLETED、ExitCode=0:0。

正式任务池执行及回传约{formal['seconds']:.3f}秒；不是整个代码开发／审计耗时。单个任务墙钟{float(s.seconds.min()):.3f}—{float(s.seconds.max()):.3f}秒，峰值RSS最高{float(s.peak_rss_MiB.max()):.2f}MiB。每任务单线程、地址空间上限4GiB；采用physical池，遵守P/E核与cgroup限制。实际没有GPU训练，也没有单个模型跨两台机器并行。

全部18份模型、预测及函数哈希核验；六个等权模型与历史树结构及训练／校准／测试预测逐元素完全一致。独立审计重放全部公司bootstrap、{audit['nodes_checked']}个节点支持，检查{audit['selected_scores_checked']}个实际选中分裂和{audit['leaf_aj_checked']}个叶内AJ；最大分裂分数差{audit['max_score_error']:.3e}，AJ差0。

全部原生时间曲线非负、时间不减、双原因和不超过1；q2路径无违例。18个保存重载误差均为0；六个96树完整集成全部原生时间重构最大误差{audit['max_ensemble_all_native_time_reconstruction_error']:.3e}。未裁剪预测或放宽测试阈值。

森林采用有限候选分裂及非参数AJ，没有梯度优化求解器；迭代收敛gap不适用，不伪造优化诊断。原因权重仅改变分裂分数，既有全部原因事件支持下限保持不变，并未偷偷加入目标原因最低事件门槛。

评价126行AUC/AP/Brier均经独立计算验证；全部种子保留，主结果为预测集成的指标。完整诊断、单种子范围、主／交互数量、分区规模、低支持比例、节点审计及环境版本见results目录。
''')
    (dest/'METHOD_INNOVATION_DECISION_ZH.md').write_text('''# 研究判断：INCONCLUSIVE

这轮实验成功执行，证明原因权重能在保留RSF机制和完整q2解释的条件下改变学习结果；不等于已经解决退市数据不足或得到全面更优模型。

- 只看退市：Top20%独立事件捕获6→8/19，Brier下降，解释表减少；2024排序及AP下降，恢复AP也下降，风险仍严重低估。
- 退市双权：2023部分改善，2024三种子AUC/AP均低于对应默认种子，合并主指标缺乏改善；不能假设更大的退市权重一定更好。
- 2023仅4个测试退市且训练仅2例，2024训练仅4例；三个种子不能提供充分的统计推断。纯目标分裂产生无目标支持的树桩，实际暴露了样本支持限制。
- 解释精确性维持，但共同函数的条件相关性上升不能替代选择稳定性；整体交互重叠仍低。此处没有可辨识性惩罚，不能推断其独立贡献。
- 原因加权是已有竞争风险森林功能，本轮贡献是任务适配实证与机制核查，不是新损失函数创新。未测试Focal Loss，不能宣称它成功或失败。

保留三个方案及负面结果。没有根据2023/2024选择新的默认模型、追加权重搜索、改动数据窗口或启动2025验证。后续若要更换主模型，需要额外的事前固定评价；本轮到此停止。
''')
    # Verify unique support counts rather than equating repeated fits with new data.
    groups=set();events=set()
    for year in [2023,2024]:
        a=arrays(run/'input/folds'/str(year)/'equal/train_PRIVATE.npz');groups.update(a['groups']);events.update(a['groups'][a['event']==1])
    assert len(groups)==277 and len(events)==4
    receipt=dict(status='COMPLETE',decision='INCONCLUSIVE',provisional=True,
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),private_run=str(run),
        fitting_commit='4fc3681de82d563e134f8257109380fd14ccc13e',formal_fits=18,pilots=2,
        reused_comparator_fits=12,failed_fits=0,failed_publications=0,calibration_fits=0,focal_loss_fits=0,
        tests_passed=106,tests_skipped=0,training_company_union=277,training_termination_union=4,
        training_fold_rows=509,repeated_formal_fit_rows=4581,test_companies=81,test_terminations=19,test_recoveries=4,
        source_run=json.loads((run/'audit/protocol.json').read_text())['source_run'],
        slurm_jobs=[2542,2543],formal_by_host=formal['by_host'],
        maximum_reconstruction_error=audit['max_ensemble_all_native_time_reconstruction_error'],
        result_sha256={str(p.relative_to(dest)):sha(p) for p in result.iterdir() if p.is_file()},
        code_sha256={p.name:sha(p) for p in dest.glob('*.py')},
        document_sha256={p.name:sha(p) for p in dest.glob('*.md')},
        protocol_sha256=sha(dest/'frozen_protocol.json'))
    write(dest/'execution_receipt.json',receipt);write(run/'execution_receipt.json',receipt)
    print(json.dumps({k:receipt[k] for k in ['status','decision','formal_fits','failed_fits','training_company_union','training_termination_union','test_companies','formal_by_host']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--dest',type=Path,required=True);a=p.parse_args();deliver(a.run,a.dest)
