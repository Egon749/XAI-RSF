"""Publish aggregate reports only; private model parameters never enter Git."""
from pathlib import Path
import argparse,json,shutil,subprocess
import numpy as np
import pandas as pd
from crforest.transparent import file_sha256 as sha,write_json

def table(df):
    def fmt(v):
        if pd.isna(v):return '—'
        if isinstance(v,(float,np.floating)):return f'{v:.6g}'
        return str(v).replace('|','/')
    return '\n'.join(['| '+' | '.join(map(str,df.columns))+' |','| '+' | '.join(['---']*len(df.columns))+' |',*['| '+' | '.join(fmt(x) for x in row)+' |' for row in df.itertuples(index=False,name=None)]])

def main(a):
    run=a.run;repo=Path(__file__).resolve().parents[3];out=repo/'experiments/transparent_cif_v2/server/results_20261028';out.mkdir()
    for p in (run/'reports').glob('*.csv'):shutil.copy2(p,out/p.name)
    for p in (run/'evaluation').glob('*.csv'):shutil.copy2(p,out/p.name)
    for name in ['environment_oliver.txt','environment_lemon.txt','tests_authoritative.txt','tests_real_interpret.txt','tests_delivery.txt']:
        shutil.copy2(run/'audit'/name,out/name)
    for name in ['bootstrap_receipt.json','bootstrap_validation.json','input_alignment.json','publication_gate.json','standalone_runtime_hashes.json','diagnostic_correction.json']:
        shutil.copy2(run/'audit'/name,out/name)
    shutil.copy2(run/'standalone_deployment/deployment_receipt.json',out/'deployment_receipt.json')
    shutil.copy2(repo/'experiments/transparent_cif_v2/config.json',out/'frozen_protocol.json')
    # Explicit public field definitions; no raw dates, company rows or cutpoints.
    shutil.copy2(run/'input/features.json',out/'feature_contract.json')
    freeze=json.loads((run/'audit/freeze.json').read_text());align=json.loads((run/'audit/input_alignment.json').read_text())
    write_json(out/'frozen_input_code_hashes.json',freeze)
    versions={'oliver':(out/'environment_oliver.txt').read_text(),'lemon':(out/'environment_lemon.txt').read_text()}
    m=pd.read_csv(out/'metrics.csv');w=pd.read_csv(out/'warning_company_quarter.csv');c=pd.read_csv(out/'complexity.csv');b=pd.read_csv(out/'bootstrap_paired_differences.csv');st=pd.read_csv(out/'native_seed_stability.csv');ps=pd.read_csv(out/'pair_search_diagnostics.csv');wd=pd.read_csv(out/'weight_diagnostics.csv');sim=pd.read_csv(out/'synthetic_recovery.csv')
    ms=m[m.seed!=-1].groupby(['scope','model','weighting'],as_index=False)[['auc','ap','ipcw_brier','prediction_mean','observed_ipcw_incidence']].mean();ms.to_csv(out/'mean_seed_metrics.csv',index=False)
    cs=c.groupby('model',as_index=False)[['active_main_functions','active_pair_functions','retained_main_functions','retained_pair_functions','spatial_cells','native_time_scalar_cells','low_support_cell_fraction','zero_support_cell_fraction','effective_trees']].mean();cs.to_csv(out/'complexity_mean.csv',index=False)
    ss=sim.groupby(['bank','aggregation'],as_index=False)[['main_amplitude_ratio','pair_amplitude_ratio','uniform_grid_mse']].mean();ss.to_csv(out/'synthetic_recovery_mean.csv',index=False)
    dep=json.loads((out/'deployment_receipt.json').read_text());boot=json.loads((out/'bootstrap_receipt.json').read_text());gate=json.loads((out/'publication_gate.json').read_text())
    data=pd.read_parquet(run/'input/final_eval_metadata_PRIVATE.parquet');counts=[]
    for scope,dd in [('2023',data[data.year==2023]),('2024',data[data.year==2024]),('pooled',data)]:counts.append(dict(scope=scope,rows=len(dd),groups=dd.group.nunique(),target_records=int((dd.event==1).sum()),target_events=dd.loc[dd.event==1,'event_id'].nunique(),competing_records=int((dd.event==2).sum()),competing_events=dd.loc[dd.event==2,'event_id'].nunique(),unknown_records=int(((dd.event==0)&~dd.label_eligible.astype(bool)).sum())))
    pd.DataFrame(counts).to_csv(out/'evaluation_sample_counts.csv',index=False)
    def doc(name,s):
        with (out/name).open('x') as f:f.write(s.strip()+'\n')
    doc('DATA_ALIGNMENT_AUDIT_ZH.md',f'''# 数据对齐：PROVISIONAL，2023/2024后暴露探索性

本轮复用原制造业冻结输入，没有重建样本、改写时间或把未知结局设为正常。原始冻结数据、M2与q2参考预测、原统一评价代码逐文件SHA256核验；详见input_alignment.json与frozen_input_code_hashes.json。

原始输入位置：`/home/oliver/MainDiskData/xai_rsf_interpretable_baselines_v1_20261027_hw9gdrol/input`。本次唯一私有目录：`{run}`。旧工作区、旧模型、旧结果未改写。

训练原点2012—2020，信息截止2022-01-01。sample_id、group、origin、事件/随访及outcome_available_at与冻结NPZ/Parquet逐行对齐；期限是每条原点加12个日历月的实际365/366日。首次目标ST、首次竞争ST、删失与source eligibility沿用原合同。未知但名义达到12个月的记录仍由原endpoint_guard排除其结局权重，不能填正常。

{table(pd.DataFrame(align['counts']).T.reset_index(names='role'))}

公司组角色按固定哈希分割，与结局无关，组重叠0；三种子共用同一角色划分，不是三个独立开发集。内部留出只有33个独立目标事件与5个独立竞争事件（122/19条相关季度记录），必须区分记录与独立支持。

{table(pd.DataFrame(counts))}

年度目标事件数不能相加当合并事件数：同一首次ST可能被2023和2024的多个12月窗口同时覆盖，合并期重新按event_id去重为79。

75列为原72内容变量（NaN保留）加3质量字段：financial_period_age、financial_disclosure_age、market_missing_coverage。字段顺序与冻结model_matrix和既有适配代码核对；不是M2的72内容＋72缺失指示＋3质量的相同参数化。字段定义/转换及未核准单位见feature_contract.json。没有新增availability数组。

原始财务披露版本的PIT认证、派生字段全部组成信息的最晚可得日以及制造业历史分类时钟仍未补齐，本轮未从财报期伪造发布日期。冻结输入具有原审计可用时钟，但不能据此声称原始版本已经PIT认证。要解除PROVISIONAL，需要上游逐组成字段历史披露/修订版本与行业分类历史有效日证据。此次仅做明确探索性比较。

原全训练q2使用全部制造业训练记录；M2为原全行业训练模型在同一制造业验证记录上的冻结预测。本轮六方案与EBM仅用结构角色公司拟合，内部留出用于统一树权重或EBM轮数选择。前两者是描述性参照，不能作为公司角色匹配的机制消融。
''')
    doc('TRAINING_DIAGNOSTICS_ZH.md',f'''# 实际训练与数值诊断

权威提交fd4a87aee54c70d5e4d165b1e7fa7011fc43c5db；正式训练冻结提交{freeze['execution_commit']}。核心森林、分区、函数编译、成对候选与权重算法未改数学实现。本次增加服务器适配/独立核验和真实InterpretML测试；合成demo仅补记录主效应幅度与oracle收据。

旧＋新完整测试：原权威版本98 passed；补真实InterpretML/Parquet端到端测试后99 passed，0 failed、0 skipped。包含R survival独立AJ核验和randomForestSRC单分裂/叶AJ核验、Parquet与真实InterpretML 0.7.8。不能将这两项R检查理解成完整森林与randomForestSRC逐项等价。唯一警告为EBM默认图不显示缺失；数值缺失单元保留，不填极值、不裁剪预测。

固定42/101/202三种子，六匹配方案：base_equal、base_weighted、ordinary_extra_equal、ordinary_extra_weighted、pair_equal、pair_weighted。每种子1个32树基础拟合、增加16棵普通树及16棵联合候选树，形成3个树库，分别等权/学习权重。18个部署模型不是18个完全独立基础森林拟合。每种子2种EBM交互预算×3种轮数，共18次EBM候选拟合，保留6个选中模型；全部选择50轮，只用历史内部公司留出。全部轮数分数见ebm_selection.csv。

正式Slurm作业2546的4项任务全部计算成功且哈希发布成功，包含三种子train→predict与一个固定模拟任务。调度使用双节点池physical模式，实际这4项均被oliver领取；不冒称两机同时训练。bootstrap作业2547采用p模式，oliver的8个P物理核与lemon的物理核处理10批任务，实际8批oliver、2批lemon；没有使用P核第二线程或绕过cgroup。单任务BLAS1，正式训练8GiB地址空间上限，非GPU算法。

{table(pd.read_csv(out/'task_costs.csv')[['seed','training_seconds','seconds','peak_children_rss_MiB','host']])}

候选与支持：

{table(ps)}

每种子64个候选变量对、最多3个切点、最多16棵新树。联合评分四区域，每格必须满足原记录/公司/事件支持阈值；评分不是统计显著性。评估分区5078个，其中3518个支持合格。64是候选审计清单长度，不是实际森林交互数；实际交互42→42、36→38、38→39。原始逐候选日志含所有失败支持和分数，保留私有。一次汇总初稿误把coverage列表当计数，已纠正并保留原稿及diagnostic_correction.json；训练与预测没有改变。

{table(wd[['seed','bank','status','iterations','cleanup_iterations','frank_wolfe_gap','objective','uniform_objective','effective_tree_count','positive_weights','largest_weight','simplex_roundoff_adjustment']])}

9次QP均收敛且最终gap≤1e-8，没有未收敛后继续训练。正权重数与有效树数均记录；其中一次数值单纯形修正2.78e-17，原收据保留。这是参数浮点可行性修正，不是对预测裁剪。每企业、原因、时间使用同一套非负和为1的树权重，没有残差或门控。

损失是历史内部留出IPCW的K+1状态Brier＋0.001趋近等权正则；内部G最小约0.165，且竞争事件仅5个独立事件。内损失改善不等于2023/2024性能改善，也不能据本次结果认定稀少竞争事件是唯一原因。

实际任务失败0；没有选择性删种子、调预算、重训M2或追加搜索。报告修正与既有容器缺依赖记录分开保留。独立部署18/18成功，全体23799记录的两个原因CIF与原始加权树逐一核验；完整全时间格的概率/单调性额外检查64个固定记录与全缺失输入。不是把65条全时间检查冒充全部23799条全时间实测。
''')
    primary=ms[(ms.scope=='pooled')&(ms.weighting=='record_equal')]
    ensemble=m[(m.seed==-1)&(m.weighting=='record_equal')]
    contrasts=b[(b.scope=='pooled')&(b.seed=='mean_seed_metrics')&(b.weighting=='record_equal')&b.metric.isin(['ap','ipcw_brier','auc'])]
    alerts=w[(w.scope=='pooled')&(w.seed==-1)]
    doc('PREDICTION_COMPARISON_ZH.md',f'''# 预测、核查负担与条件区间

状态：PROVISIONAL；2023/2024后暴露探索性。没有用评价结局选择树权重、EBM轮数或追加配置。六个新森林及两种EBM公司角色匹配；原全训练q2与全行业M2仅作描述性参考。EBM是12个月二分类端点模型，不提供本轮原生全时间竞争CIF；数值解释为logit经logistic，而森林是CIF概率恒等链接。

主要配对统计为同种子比较，再对三个种子的指标求均值。合并期AP首先在合并记录上计算，绝非年度AP平均。下面是**三个单种子指标均值**，不是概率集成：

{table(primary[['model','auc','ap','ipcw_brier','prediction_mean','observed_ipcw_incidence']])}

下面是**先平均三种子概率、再计算指标**，与上表不同：

{table(ensemble[['scope','model','auc','ap','ipcw_brier']])}

记录等权与公司等权完整结果分别见metrics.csv、mean_seed_metrics.csv；季度警报/年度去重、所有种子及概率平均详见warning_company_quarter.csv。观察目标率的IPCW估计也保留；合并记录等权约0.950%，公司等权约1.685%，不能混用。校准十分位箱见calibration.csv，既含所有记录又列未知数量。

1000次原协议公司组有放回抽样使用seed20261025和spawn_key=(replicate)，同一公司抽样共同用于所有模型/种子/年度；预测、名单、原删失权重固定，不重训、不重排。2400个点估计与原评价直接核对，最大差约2.39e-13，并以非平凡重复公司抽样与sklearn核验排序指标。它是条件评价区间，不含训练不确定性；不作多重比较校正、不据此挑选优胜配置。

{table(contrasts[['model_a','model_b','metric','point','ci_low','ci_high']])}

联合候选＋权重相对基础等权，平均种子AP差−0.012636，区间跨零；Brier差+0.000067，区间[+0.000021,+0.000120]。相对同48树的普通扩展＋权重，AP差−0.010604，区间[−0.025223,−0.001185]；Brier亦较差。本轮证据不能支持联合候选或学习权重的独立预测增益。原M2/q2参照不角色匹配，差距不能全归于机制。

合并期**概率平均**的Top20/25核查结果：

{table(alerts[['model','top_fraction','alert_company_groups','hit_company_groups','company_hit_fraction','independent_events','detected_events','event_coverage','announcement_detected_3m','announcement_detected_6m']])}

企业级命中占比=至少一个被标记窗口命中目标事件的去重公司/去重预警公司。本队列每个目标事件是一家企业的首次事件，故本表事件/公司分子相同；实现仍分别计数。不能把未命中公司一律叫正常，也不能把这个含未知负担的比率叫成熟结局PPV。

{table(alerts[alerts.model=='pair_weighted'][['top_fraction','alert_records','repeated_alert_records','known_no_target_alerts','unknown_outcome_alerts','no_hit_with_unknown_groups','no_hit_all_alerts_known_non_target_groups','announcement_opportunity_3m','announcement_opportunity_6m']])}

联合＋权重Top25核查1399家公司，确认66/79独立事件，企业命中占比4.718%；107家无命中但存在未知预警窗口，1226家所有预警窗口均为已知非目标。后者仅说明这些窗口，不等于企业永远正常。5953条季度警报中4554条为去重公司数之外的重复警报。

公告前3/6个月用原日历日期而非90/180天近似，合并可识别机会分母各60；联合＋权重Top25分别49与46，不能以79直接解释为机会覆盖率。年度、季度及实施日前提前量也完整输出。AP/AUC只说明排序；Brier兼顾概率偏差；事件覆盖与核查负担须同时看，不能用覆盖单项宣布成功。
''')
    stability=st.groupby(['model','order'],as_index=False)[['structural_jaccard','anchor_active_jaccard','aligned_component_rmse','importance_spearman','sign_agreement_on_common_nonzero']].mean()
    doc('INTERPRETABILITY_COMPARISON_ZH.md',f'''# 完整透明预测器与实际解释规模

18个森林模型部署包全部通过独立核验。只携带functions.py、transparent.py、空__init__.py及完整函数包，在python -I独立进程执行；目录不含core.py和原树库，断言未导入森林。23799条记录的全体原点12个月CIF均与已冻结输出一致；所有模型展开重构误差最大{max(x['full_explanation_error'] for x in dep['models']):.3g}，未添加隐藏残差、代理解释或按企业变化权重。

`F_k(t|x)=mu_k(t)+sum_j f_jk(t,x_j)+sum_(j,l) g_jlk(t,x_j,x_l)`；全部时间和原因共享非负树权重。原公司bootstrap、竞争风险分裂思想和AJ叶估计保留，新增联合四区域候选搜索不是标准RSF原节点算法。函数是预测器自身的精确代数表示，原生尺度为CIF；带符号贡献不是风险率乘数，也不是因果影响。

{table(cs)}

上表为三个种子均值。活跃函数按全部精确空间格、全部原生时间、全部原因检查绝对值>1e-14；retained函数与所有单元仍完整存储，即使权重使函数为零也未删除。pair_weighted实际14—16个活跃主函数、16—18个活跃二元项，但保留23—26主函数、38—42二元项、1300—1502空间格、603200—696928个时间×原因标量。有效树数约8.62—13.08，不等于实际只保留这么多棵树。

同树库的等权/加权版本逻辑表格规模完全相同，不能把活跃函数减少宣传成完整导出表格已经缩小。完整空间单元中约34.95%没有目标独立事件、37.29%没有竞争独立事件支持（pair库；按单元计数，不是样本概率质量），详见cell_support_summary.csv。零支持空间格平均约19.7%，低于10条记录约24.5%；产品边际参考可产生现实联合样本稀少的组合，完整可见并不意味着这些函数有充分估计支持。

每个完整表含全部缺失行/列、零支持单元、真实联合记录支持、公司组支持、原因别记录和独立事件支持；全部不提交Git。6个固定匿名记录×18模型分别保存3/6/12个日历月的所有贡献、基线、重构及CIF。全时间函数存储为无损矩形因子化系数，另有完整365/366日可读表，未合并切点或TopK截断。部署包、支持表和个体完整解释路径见PRIVATE_ARTIFACT_INDEX_ZH.md。

EBM0为75主项/2249端点单元，EBM10为75主项＋10二元项/4809端点单元，保留缺失格。它们没有全时间×竞争原因轴；不能直接拿EBM端点标量数与森林全时间标量数作同任务复杂度优劣判定。

三种子固定原32个评价锚点的原生函数描述性稳定性（365日目标CIF）：

{table(stability)}

权重使主效应跨种子RMSE升高；pair_weighted二元RMSE高于pair_equal，活跃交互集合Jaccard略提高但仍约0.276，重要性相关均值为负。符号一致率仅在两个模型均非零的锚点项上计算，不能替代全函数方向稳定性。RMSE缩小也可能来自作用缩小；不同尺度统计一起保留。这不是重新完成50次训练bootstrap稳定性验证，不能套用旧模型的稳定性区间。

固定二值模拟的幅度恢复：

{table(ss)}

配对＋历史留出权重将真二元系数恢复比例从基础等权12.90%提高至21.31%，仍远低于100%；主效应从34.47%到63.56%。oracle真CIF选权重只给乐观容量诊断，配对库二元恢复也仅71.30%，不可当成实际可部署模型或真实企业效果。单一二值DGP、三个固定种子不能证明一般结构恢复优势。本轮完整保留27组结果。

已实现可执行、可逐项核算的模型透明性；尚未证明函数足够少、人员容易理解、相关变量下统计唯一或重训稳定，更不具有因果解释保证。
''')
    doc('METHOD_INNOVATION_DECISION_ZH.md','''# 本轮研究判断

**总体研究结论：INCONCLUSIVE。工程交付成功，但本轮联合候选＋全局权重更新未得到独立预测增益支持。**

可以确认：真实制造业样本上三种子、六匹配森林方案全部完成，完整主/二元CIF函数可直接独立部署；无需森林遍历、SHAP或隐藏残差，重构在浮点精度内一致。未知结局和竞争事件合同保持原定义，真实EBM内部轮数选择也已执行。

不能确认：联合搜索与凸权重能同时恢复交互幅度、提高预警命中率并增强解释稳定性。实际新增变量对仅0/2/1，模拟恢复仍严重不足；真实数据中相对匹配基础模型AP均值下降，Brier条件区间提示恶化。相对同树数普通扩展＋权重，AP与Brier亦不利。没有因为解释精确就把这轮方法更新判为成功。

活跃函数减少是可观察的工程性质，但保留的完整表格并未随权重减少，主效应重训波动更大、交互稳定性改善不一致。有限固定权重没有证明已消除作用稀释。EBM主效应对照在本轮AP/AUC更有竞争力，同时任务只涵盖12个月端点，不能冒充完整竞争CIF。

这些结果不否定q2森林能够精确透明表达的整体框架价值；它保留生存树/非参数叶估计并能输出全时间竞争CIF。它们也不支持把当前附加候选和权重宣布为已有独立方法创新。没有新可辨识性惩罚，本轮完全没有检验gamma的独立贡献，不能外推为证明或反驳另一项目的Hessian机制。

结论受PROVISIONAL数据时钟、后暴露2023/2024、唯一固定公司角色分割、仅5个内部独立竞争事件、固定预测条件区间等限制。虽有不利条件区间，也不等于未暴露外部验证下的普遍结论。原全训练q2和全行业M2范围不同，不能把差距全部归为机制。

本轮已按冻结预算停止。保留全部种子、候选、零权重、失败支持、未知结局和不利比较；不追加gamma、提高q、隐藏残差、动态企业权重或新一轮配置搜索。当前版本可用于透明性工程与负面消融的研究证据，不宜替代已有模型并宣称预警性能升级。
''')
    doc('PRIVATE_ARTIFACT_INDEX_ZH.md',f'''# 私有完整交付索引

根目录：`{run}`。

- `results/formal/2546/task-0/output`、task-1、task-2依次为42、101、202。各自fit下六个方案均含functions_PRIVATE.npz、manifest_PRIVATE.json、integrity.json、complete_tables_PRIVATE.json、complete_cases_PRIVATE.json、complexity_PRIVATE.json。三树库、完整候选日志、全局权重、EBM模型与每轮分数全部保留。
- `model_audit/*_all_cells_PRIVATE.json`：18份全函数表，含所有缺失格、记录/公司组/原因别独立事件支持；`*_complete_cases_PRIVATE.json`：18份、每份6个固定匿名案例，完整日历3/6/12个月解释。真实记录对应关系仅在私有文件中。
- `standalone_deployment/`：可单独执行的完整透明函数运行包；verify.py不依赖原树代码，deployment_receipt.json记录18模型全体23799记录的核验。主程序 `python -I verify.py`；如再次核验，先复制到新目录或指定新收据，不覆盖此次证明。
- `results/formal/2546/task-*/output/predictions`：全体私人预测，所有seed和model均保留；`evaluation`：统一结果。
- `results/formal/2546/task-3/output/demo`：27组模拟、9次可部署权重QP及9次oracle容量QP收据。
- `results/bootstrap/2547`：10批共同公司抽样；`bootstrap_replicates_PRIVATE.npz`：1000次所有模型/种子/年度结果。
- `audit/`：代码归档、输入哈希、测试日志、冻结与发布核验；formal.jsonl.results-2546.json及bootstrap.jsonl.results-2547.json为原调度运行收据。

Git仅提交代码、字段定义、冻结协议、汇总结果及模型文件SHA256。真实企业明细、预测、参数、原始切点与个体解释不上传。
''')
    doc('SERVER_EXECUTION_ZH.md',f'''# 服务器实际执行交付：2026-10-28

**运行已完成；完整透明部署通过，但本轮新增机制没有获得预测收益支持。** 数据继续PROVISIONAL，2023/2024继续后暴露探索性。

- 全部旧＋新测试99通过，0失败、0跳过，包含两项独立R、Parquet、真实InterpretML。
- 三种子42/101/202全部完成：18个透明CIF模型＋6个内部留出选中的EBM模型。EBM实际18次候选拟合，没有评价期选轮数。
- 4个正式任务＋10个公司bootstrap任务全部计算和发布成功；1000次共同公司抽样。训练/预测不利结果全部保留。
- 训练唯一记录56458/2577家公司/177个独立目标事件；结构训练45343/2061/144，内部留出11115/516/33。内部独立竞争事件只有5。
- 评价23799条记录/3138家公司/79个独立目标事件。年度重叠事件在合并时重新去重。
- 完整主函数、二元项、缺失格与联合支持均在私有服务器；独立部署不携带原森林。最大完整解释重构误差约2.22e-16。
- 六个森林版本、EBM、原q2/M2的年度与合并结果、两类等权口径、Top20/25、3/6月提前量、每种子与概率平均均已交付。

合并记录等权、三个单种子指标均值：

{table(primary[['model','auc','ap','ipcw_brier']])}

配对结论和概率平均表不同口径，见[PREDICTION_COMPARISON_ZH.md](PREDICTION_COMPARISON_ZH.md)。主要机制比较是本轮六方案；原全训练q2/M2为描述性参考。

[数据对齐](DATA_ALIGNMENT_AUDIT_ZH.md) · [训练诊断](TRAINING_DIAGNOSTICS_ZH.md) · [完整解释与模拟](INTERPRETABILITY_COMPARISON_ZH.md) · [研究判断](METHOD_INNOVATION_DECISION_ZH.md) · [私有交付索引](PRIVATE_ARTIFACT_INDEX_ZH.md) · [实际收据](execution_receipt.json)。

不追加训练或搜索。独立方法创新总体判断INCONCLUSIVE；透明性工程PASS不等于预测升级PASS。
''')
    receipt=dict(status='COMPLETED_PROVISIONAL_POST_EXPOSURE_EXPLORATORY',authoritative_commit=freeze['authoritative_commit'],training_execution_commit=freeze['execution_commit'],private_run=str(run),tests=dict(passed=99,failed=0,skipped=0,independent_R=True,real_InterpretML=True,parquet=True),formal_job=2546,bootstrap_job=2547,slurm_tasks_expected=14,slurm_tasks_completed=14,slurm_tasks_failed=0,all_publications_verified=True,seeds=[42,101,202],matched_native_models=18,base_forest_fits=3,additional_ordinary_banks=3,additional_pair_banks=3,tree_weight_QP_solves=9,ebm_candidate_fits=18,ebm_selected_models=6,synthetic_comparisons=27,bootstrap_replicates=1000,training_counts=align['counts'],evaluation_counts=counts,independent_native_deployments=18,max_complete_reconstruction_error=max(x['full_explanation_error'] for x in dep['models']),research_decision='INCONCLUSIVE',new_mechanism_predictive_gain_supported=False,hyperparameter_search_extended=False,test_outcome_configuration_selection=False,failed_tasks=[],report_corrections=['Coverage audit lists are not actual pair counts; corrected and original retained'],reference_scope=dict(q2='original full manufacturing train',M2='original all-sector train; same manufacturing evaluation rows'),bootstrap_scope=boot['scope'],slurm_receipt_hashes={p.name:sha(p) for p in [run/'formal.jsonl.results-2546.json',run/'bootstrap.jsonl.results-2547.json']},analysis_code_hashes={p.name:sha(p) for p in (repo/'experiments/transparent_cif_v2/server').glob('*.py')},privacy='company records, predictions, model parameters, cutpoints and complete individual functions remain private')
    write_json(out/'execution_receipt.json',receipt)
    write_json(out/'artifact_hashes.json',{p.name:sha(p) for p in out.iterdir() if p.is_file()})
    print('Published aggregate delivery directory',out)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);main(p.parse_args())
