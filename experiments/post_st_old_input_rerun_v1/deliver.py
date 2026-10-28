"""Publish only aggregate evidence from the freshly executed old-input rerun."""
from pathlib import Path
import argparse,datetime,json,shutil
import pandas as pd
from experiments.post_st_prognosis_v1.common import sha,write
from experiments.post_st_next_open_info_v1.deliver import table


def metrics(d):
    d=d[['scope','model','auc','ap','brier','risk_mean','ipcw_observed_incidence']].copy()
    d['model']=d.model.map({'xai_q2':'q2','standard_rsf':'标准RSF','training_AJ':'训练AJ'})
    d.columns=['年度','模型','AUC','AP','IPCW Brier','平均风险','观察发生率']
    return table(d)


def deliver(run,dest):
    a=json.loads((run/'audit/data_alignment.json').read_text())
    for name,h in a['unchanged_source_hashes'].items():assert sha(Path(name))==h
    for name,h in json.loads((run/'audit/immutable_input_hashes.json').read_text()).items():assert sha(run/'input'/name)==h
    rep=json.loads((run/'evaluation/reproduction_summary.json').read_text())
    audit=json.loads((run/'evaluation/independent_audit.json').read_text())
    gate=json.loads((run/'audit/formal_gate.json').read_text())
    val=json.loads((run/'evaluation/validation.json').read_text())
    assert rep['status']=='EXACT_REPRODUCTION' and audit['status']==gate['status']==val['status']=='PASS'
    rcmp=pd.read_csv(run/'evaluation/r_model_reproduction.csv')
    assert len(rcmp)==6 and rcmp.full_object_exact_excluding_timing_and_sampsize_closure.all()
    out=dest/'results';out.mkdir(exist_ok=False)
    for p in (run/'evaluation').iterdir():
        assert p.suffix in ['.csv','.json'] and 'PRIVATE' not in p.name
        shutil.copy2(p,out/p.name)
    for p in (run/'reports').glob('*.csv'):shutil.copy2(p,out/p.name)
    for name in ['data_alignment.json','pre_fit_freeze.json','formal_gate.json',
        'environment_oliver.json','environment_lemon.json','tests_pre_freeze.log','tests_final.log']:
        shutil.copy2(run/'audit'/name,out/name)
    plan=json.loads((run/'input/folds/2023/announcement_fields/plan.json').read_text())
    write(out/'frozen_model_configurations.json',dict(forest_common=plan['forest_common'],
        xai_q2=plan['forest_variants']['xai_q2'],standard_rsf=plan['standard_rsf'],
        feature_names=plan['feature_names'],seeds=[42,101,202]))
    m=pd.read_csv(out/'metrics.csv',dtype={'scope':str,'seed':str})
    w=pd.read_csv(out/'warning_metrics.csv',dtype={'scope':str,'seed':str})
    s=pd.read_csv(out/'training_diagnostics.csv');fresh=s[s.source_state=='rerun_old_C']
    main=m[(m.source_state=='rerun_old_C')&(m.seed=='ensemble')&(m.horizon_months==24)&(m.cause=='termination')].sort_values(['scope','model'])
    warn=w[(w.source_state=='rerun_old_C')&(w.seed=='ensemble')&(w.horizon_months==24)]
    wtab=warn[['scope','model','top_fraction','checked','captured','target_events','coverage','known_non_target','unknown','lead3','lead6']]
    c=pd.read_csv(out/'ensemble_complexity.csv')
    st=pd.read_csv(out/'seed_stability.csv')
    st=st[(st.source_state=='rerun_old_C')&(st.cause=='termination')].groupby('test_year')[['main_jaccard','pair_jaccard','median_function_correlation']].mean().reset_index()
    (dest/'RESULTS_ZH.md').write_text('''# 旧输入复跑：预测和指标完全复现

12/12正式模型已实际重新拟合，旧模型仅用于结果对照，未代替新训练。两台机器共用原83列C组输入，训练和评价均恢复年报确认前的冻结快照。三个固定种子的预测先平均，再计算主结果。

## 24个月先最终退市

'''+metrics(main)+'''
q2在2023的AUC/AP恢复为0.736111/0.576727，2024为0.752000/0.758339；合并为0.723852/0.669576。全部12个模型的每家公司预测与原C组逐项完全一致，最大差0；每个种子、两期限、两结局及合并指标也全部复现。不是挑选表现较好的种子，也没有为了复现而改变训练设置。

## 核查与事件覆盖

'''+table(wtab)+'''
q2 Top20：2023核查8家、捕获5/12，2024核查9家、捕获7/20，合计17家、12/32（37.5%）；至少提前3个月10例、6个月4例。Top25：合计21家、15/32（46.875%）；至少提前3个月12例、6个月6例。未知0、重复预警0。年度完整队列回顾排序不等于在线实时核查政策。

原有概率低估同样复现：q2平均预测2023为6.27%、2024为11.93%，对应观察先退市33.33%、44.44%。这次确认计算可复现，没有产生新的概率校准改进。

这是在看到接受年报口径结果后，按用户选择恢复旧输入的复跑。已完成的接受口径结果及其不利发现全部保留；不能把恢复旧输入后的结果当成新的独立验证，或据此证明旧财务时钟更真实。
''')
    secondary=m[(m.source_state=='rerun_old_C')&(m.seed=='ensemble')&(m.horizon_months==12)&(m.cause=='termination')]
    recovery=m[(m.source_state=='rerun_old_C')&(m.seed=='ensemble')&(m.horizon_months==24)&(m.cause=='recovery')]
    (dest/'PREDICTION_COMPARISON_ZH.md').write_text('''# q2与同输入标准RSF

标准RSF是randomForestSRC 3.9.0的本次ST后预后参照模型，不是原首次ST预警M2。两者使用相同公司bootstrap、32树/种子及原有数值预算；q2有二阶路径、共享切点、25输入/树与原生缺失路由，标准RSF不限交互阶数、用训练中位数填补。不能声称后端完全相同。

'''+metrics(main)+'''
2024 q2退市AUC/AP高于标准RSF；2023 q2 AUC较高但AP较低。q2的合并Brier 0.313764略差于标准RSF 0.311675。不能只凭AUC宣称整体概率质量更好。

## 24个月先恢复

'''+metrics(recovery.sort_values(['scope','model']))+'''
相同输入下标准RSF的恢复AUC/AP/Brier在两个年度均较好。这一不利结果保持原样。

## 补充12个月先最终退市

'''+metrics(secondary.sort_values(['scope','model']))+'''
12个月为预定补充，不替换24个月主目标。204行指标独立复算，合并AP在拼接预测上重新计算；测试端点全部已知，IPCW权重为1。每家公司一行，记录等权与公司等权相同。全种子结果和范围见metrics.csv、seed_summary.csv；三种子范围不代表置信区间。
''')
    (dest/'DATA_ALIGNMENT_AUDIT_ZH.md').write_text('''# 恢复原C组冻结输入，不重新生成财务值

本轮读取原next-open C训练及预测输入NPZ，逐字节复制并核验SHA256；评估结局和公司元数据另存，不进入训练任务的数据目录。sample_id、公司组、原点、事件、随访、标签可得性、全部83个输入与顺序均保持原样。没有调用年报接受口径适配器，没有混用新模型训练输入或新测试输入。

'''+table(pd.read_csv(out/'matched_data_audit.csv'))+'''
全非金融行业、观察历史中的首次财务/审计ST；预测起点仍为公告后下一交易日开盘前，先恢复A或最终终止上市X均使用公告日。没有恢复到更早的“公告日严格之前”预测起点。2023训练2012—2022入组、结局信息严格早于2023-01-01；2024训练2012—2023入组、结局信息严格早于2024-01-01。

训练两折309/345家、24个月先退市14/26例。两折共654条记录，去重345家、26个独立先退市事件；12次拟合共3924条重复训练计算记录，不是新增样本。测试36/45家，先退市12/20例；合计81家、32例先退市、31例先恢复、18例窗口内无首次事件。训练未知保持删失，测试12/24个月端点均完整。每折训练测试无公司重叠。

旧输入保留保守财务时钟：更正日期参与可得性，派生输入使用组成数据最晚可得日期。这是本轮明确恢复的已冻结数据版本，不声称完成独立原始PDF/PIT认证。此前用户确认接受年度来源的记录和接受口径实验仍在单独目录，不删除、不重写；恢复旧输入是最新的实验选择，不是对数据真实性作反向证明。

旧数据源覆盖仍截至2026-09-03，未延长至今日，没有评价2025入组队列。2023/2024已暴露且本次选择发生在看到结果之后，因此为POST_EXPOSURE_USER_SELECTED_REPRODUCTION。
''')
    ctab=c[['test_year','trees','main_functions','interaction_functions','spatial_table_cells','scalar_table_cells','low_support_cell_fraction','zero_support_cell_fraction']]
    (dest/'INTERPRETABILITY_COMPARISON_ZH.md').write_text('''# 完整解释复现，可靠性结论保持不变

各年度主结果为三种子96树集成。全部主效应、二元交互、缺失分区、两个原因和原生时间均保留，贡献在CIF概率尺度上相加；不是风险率贡献或因果影响。

'''+table(ctab)+f'''
6个q2森林文件与原模型逐字节一致，所有完整函数数组逐项一致；两个96树完整集成在训练及测试全部公司、全部原生时间的最大重构/重载误差为{audit['max_ensemble_all_native_time_reconstruction_error']:.3e}。

## 三种子解释稳定性

'''+table(st)+'''
2023/2024交互选择Jaccard仍约0.073/0.092。低支持格约50.8%/48.3%，零支持格约20.4%/22.9%，支持门槛10家公司；约63.6%/47.1%的单模型叶节点没有退市公司支持。这些限制没有因复跑消失。共同函数相关性仅描述共同非恒定项，不代表所有交互稳定。

函数数、真实分区、字段使用和贡献均保持原有结果；企业数据和完整函数参数仍在私有目录。共享切点及q2机制未改，没有新增可辨识性惩罚。精确性可复现，不等于解释已充分简单、可靠或统计可辨识。
''')
    (dest/'TRAINING_DIAGNOSTICS_ZH.md').write_text(f'''# 实际执行与复现核验

冻结拟合代码：`{plan['code_commit']}`。训练前和交付前完整120项测试通过，包含独立R检查，0跳过；唯一既有警告为EBM缺失值可视化提示。核心森林、原worker及R拟合实现与原C拟合提交dc8c7f49b62750b2ae87058e293b935b0b1fcd34逐文件相同。

12/12新训练成功，失败0、发布失败0、选择性重跑0。Slurm作业{gate['job']}，分配{gate['by_host']}，实际计算及校验回传{gate['seconds']:.3f}秒。未新增小试：此前相同数据/预算/实现已经通过原小试及正式执行，本轮没有新的资源假设。现有p/both管理oliver P核与lemon物理核，每任务单线程、4GiB地址空间限制；没有硬编码CPU或复制任务凑占用率。

单任务{fresh.seconds.min():.3f}—{fresh.seconds.max():.3f}秒，Python父进程最高RSS {fresh.peak_rss_MiB.max():.3f}MiB、R子进程单独最高{fresh.child_peak_rss_MiB.max():.3f}MiB。任务完成、输出发布及哈希全部核验，不用提交作业号代替成功证据。旧预测只在分析阶段加载作为对照。

12个模型的评估预测和全部原生时间曲线逐项一致，最大预测差0；6个q2模型文件哈希也与原模型相同。6个标准RSF的RDS文件哈希不同，独立R核验确认差别在运行计时和sampsize函数闭包序列化；去除这两类运行元数据后，森林参数、数据及对象内容严格identical，函数代码及形参相同。没有把不同RDS哈希隐瞒为逐字节一致。详见reproduction_models.csv与r_model_reproduction.csv。

独立重放6个q2模型的{audit['nodes_checked']}个节点、{audit['selected_scores_checked']}个已选分裂、{audit['leaf_aj_checked']}个AJ叶估计；最大分裂分数差{audit['max_score_error']:.3e}，AJ差0；6个标准RSF公司bootstrap和训练中位数核验通过。所有模型保存重载一致，概率约束和二阶路径约束有效，没有裁剪。

204行指标复算，96行新旧配对指标差全部为0。森林没有数值梯度求解器，Hessian秩或优化gap不适用。全部种子、低支持情况和不利结果均保留，未做额外搜索。
''')
    (dest/'METHOD_INNOVATION_DECISION_ZH.md').write_text('''# 计算结论EXACT_REPRODUCTION；研究判断INCONCLUSIVE保持不变

本轮成功完成用户要求的实际旧输入复跑。12个模型预测、指标及完整q2函数精确复现，证明这套固定代码、数据、种子和预算下的计算可重复。没有发现恢复旧输入后产生新的数值错误或解释误差。

这不是一次新样本检验，也没有新增算法或证明惩罚独立贡献。旧输入是在看到接受年报口径结果后选择恢复的，不能把这次复现当成额外一份独立性能证据。原接受口径实验及其2023下降、2024小幅提高等结果继续保留。

旧C的价值和限制均未变化：q2提供可用排序与完整精确函数，但2023 AP不及标准RSF，退市概率显著低估，恢复预测不及标准RSF，交互稳定性和事件支持不足。因此研究层面的INCONCLUSIVE不因复跑而升级为METHOD_PROMISING。

本轮按指定范围结束；没有校准、叶收缩、权重/阶数/超参数搜索或2025评价。恢复旧口径不证明其原始披露真实性更高，也不消除既有数据覆盖及历史暴露限制。
''')
    receipt=dict(status='COMPLETE',reproduction_status=rep['status'],research_decision='INCONCLUSIVE',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),private_run=str(run),
        repository='Egon749/XAI-RSF',branch='research/xai-rsf-post-st-prognosis-v1',fitting_commit=plan['code_commit'],
        source_status='RESTORED_FROZEN_CONSERVATIVE_C_INPUTS',evaluation_status='POST_EXPOSURE_USER_SELECTED_REPRODUCTION',
        formal_fits=12,new_pilots=0,reused_reference_models=12,failed_fits=0,failed_publications=0,
        tests_passed=120,tests_skipped=0,slurm_job=gate['job'],formal_by_host=gate['by_host'],
        training_fold_rows=654,training_company_union=345,training_termination_union=26,repeated_formal_fit_rows=3924,
        test_companies=81,test_terminations=32,test_recoveries=31,input_columns=83,
        input_byte_identical_to_original_C=True,changed_labels=0,exact_prediction_models=12,
        model_bytes_identical=6,standard_RSF_payload_exact_excluding_runtime_metadata=6,
        maximum_prediction_difference=rep['maximum_prediction_difference'],all_metrics_exact=True,
        maximum_reconstruction_error=audit['max_ensemble_all_native_time_reconstruction_error'],
        main_horizon_months=24,secondary_horizon_months=12,algorithm_changed=False,
        calibration_fits=0,leaf_shrinkage_fits=0,new2025_evaluation=False,
        result_sha256={str(p.relative_to(dest)):sha(p) for p in out.iterdir() if p.is_file()},
        code_sha256={p.name:sha(p) for p in dest.iterdir() if p.suffix in ['.py','.R']},
        document_sha256={p.name:sha(p) for p in dest.glob('*.md')},protocol_sha256=sha(dest/'frozen_protocol.json'))
    write(dest/'execution_receipt.json',receipt);write(run/'execution_receipt.json',receipt)
    print(json.dumps({k:receipt[k] for k in ['status','reproduction_status','formal_fits','new_pilots','test_companies','test_terminations']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--dest',type=Path,required=True)
    a=p.parse_args();deliver(a.run,a.dest)
