"""Publish aggregate evidence only; preserve singleton and model privacy."""
from pathlib import Path
import argparse,datetime,json,shutil,re
import pandas as pd
from experiments.post_st_drop9_v1.deliver import table
from .common import *


def deliver(run,repo):
    pub=repo/'experiments'/NAME;reports=run/'reports'
    val=json.loads((run/'audit/validation.json').read_text());assert val['status']=='PASS_WITH_2025_EVALUATION_BLOCKER'
    gate=json.loads((run/'audit/formal_gate.json').read_text());frozen=json.loads((run/'audit/freeze.json').read_text())
    m=pd.read_csv(reports/'metrics.csv',dtype={'seed':str});w=pd.read_csv(reports/'warnings.csv',dtype={'seed':str})
    cv=pd.read_csv(reports/'function_validation.csv',dtype={'seed':str});diff=pd.read_csv(reports/'paired_differences.csv')
    fits=pd.read_csv(reports/'execution_tasks.csv');data=pd.read_csv(reports/'cohort_audit.csv')
    tests=(run/'audit/full_tests_pre_fit.log').read_text();assert 'failed' not in tests.lower()
    passed=int(re.search(r'(\d+) passed',tests).group(1))
    assert sha(repo/'CURRENT_POST_ST_MODEL.json')==REFERENCE_SHA
    sources=json.loads((run/'audit/source_hashes.json').read_text());preserved=[]
    for p,h in sources.items():
        assert sha(Path(p))==h
        if Path(p).name in ['forest_PRIVATE.json','predictions_PRIVATE.npz','functions_PRIVATE.npz']:
            preserved.append(dict(path=p,sha256=h,unchanged=True))
    assert len(preserved)==27
    pd.DataFrame(preserved).to_csv(reports/'reference_preservation.csv',index=False)
    ens=m[(m.seed=='ensemble')&(m.cause==1)&(m.status=='SUPPORTED')].merge(
        w[['year','arm','seed','checked','captured','target_events','coverage']],on=['year','arm','seed'])
    ens.to_csv(reports/'ensemble_comparison.csv',index=False)
    cc=['year','arm','version','seed','main_functions','interaction_functions','spatial_table_cells','scalar_table_cells','distinct_thresholds']
    cv[cc].to_csv(reports/'function_complexity.csv',index=False)
    cols=['training_cutoff','arm','companies','known_target_events','auc','ap','brier','risk_mean','ipcw_observed_incidence','checked','captured','coverage']
    statements=[]
    for year,g in ens.groupby('year',sort=True):
        a=g.set_index('arm').loc['reference83'];b=g.set_index('arm').loc['drop9']
        statements.append(f"截至{year}-06-30训练、评价{year}-07-01至{year+1}-06-30新入组企业："
            f"AUC {a.auc:.6f}→{b.auc:.6f}，AP {a.ap:.6f}→{b.ap:.6f}，Brier {a.brier:.6f}→{b.brier:.6f}；"
            f"Top20核查{int(a.checked)}家，捕获{int(a.captured)}/{int(a.target_events)}→{int(b.captured)}/{int(b.target_events)}。"
            f"平均风险{a.risk_mean:.2%}→{b.risk_mean:.2%}，观察发生率{a.ipcw_observed_incidence:.2%}。")
    paragraph='\n\n'.join(statements)
    result=f'''# 6月30日训练截点：12个月实验结果

18个新森林全部完成（3个截点×2套变量×3个种子），派生18个直接父节点AJ收缩版本。前两个未来入组窗口可完整评价；第三个窗口因缺少2026年审核冻结队列，评价尚未完成。原研究参考模型保留，没有新增调参、概率映射或选择新参考模型。

## 日期含义与样本

{table(data[['training_cutoff','test_entry_from','test_entry_through','training_companies','training_target_events','training_recovery_events','training_target_events_within12','evaluation_companies_available','evaluation_status']])}

6月30日是**训练信息截止日，含当日**。测试公司必须在此后12个月内新入组，且与训练公司互斥。每家公司从自身的入组公告后下一交易日开盘前，预测未来12个日历月的先最终退市；先恢复为竞争事件。它不是统一从6月30日开始跟踪所有已ST公司的landmark实验，也不是原来的自然年度测试队列。

训练随访上限沿用24个月，但只纳入截至训练截止日已经可得的结局；评价时点固定12个月。三个训练集嵌套，训练公司数和事件数不可相加当作独立样本。训练事件数包含24个月上限内已知退市，within12列另外给出12个月内事件数。

## 先最终退市预测

{table(ens[cols])}

reference83为原83变量设置，drop9为固定删除同九项后的74变量。表中指标先平均三个种子的逐公司预测，再重新计算。AUC/AP越高越好，Brier越低越好；平均风险与观察发生率接近表示总体水平更吻合。Top20按每个完整入组窗口公司数的20%向上取整，是窗口内回顾排序核查额度，不是前瞻在线阈值。

{paragraph}

两窗口不是原来的2023年和2024年自然年度样本，不能把指标变化全部归因于增加了半年的训练信息。它同时改变了训练截止和测试入组窗口。删除变量的同窗配对比较才使用相同样本与结局。

完整单种子、恢复原因、校准分箱与预测波动均保存在results目录；不能仅凭AUC或AP上升断言绝对概率已经校准。所有不利结果保留。

## 2025截点的限制

计划测试窗口为2025-07-01至2026-06-30。现有审核冻结的首次财务/审计ST队列只覆盖至2025年，该窗口仅有1家已冻结企业。2026年虽有原始事件记录，但没有同口径首次事件审核、非金融资格及与83输入对应的已核验入组数据。不能把原始事件记录直接当作完整合格队列。

2025截点的6个模型已实际训练并保存，该1家私有预测和完整函数重构也已完成；公开AUC、AP、Brier、平均风险、事件数与捕获率全部留空，防止将1家公司结果冒充队列验证或公开个体风险。补齐2026队列后仍需足够随访，才能评价完整窗口的12个月结局。当前不能回答这一窗口的预测性能。

## 配对差值与完整解释

{table(diff[['year','metric','difference','lower95','upper95','undefined_bootstraps']])}

差值为74变量减83变量；Brier负值更好，AUC/AP及覆盖正值更好。每个可评价窗口进行1000次公司配对bootstrap，所有模型复用相同抽样索引。区间仅反映固定模型条件下的测试样本波动，不包含完整训练、变量筛选或调参不确定性。不能选择性解释多个区间为确认性显著发现。

{table(cv[(cv.seed=='ensemble')&(cv.version=='shrunk')][cc])}

所有单种子raw/收缩模型及三种子平均的96树预测器，均在全部训练与评价样本、全部原生时点和两原因验证完整q2函数。最大绝对重构误差{val['max_explanation_error']:.3e}，低于1e-10；概率非负、时间单调、两原因之和不超过1及保存重载均通过，没有裁剪预测或删掉函数项。

空间格数含缺失状态，不乘时间或原因维；scalar_table_cells另外包含这两个维度。实际函数数是所有原生时点/原因上非零的项。三种子只是算法随机性，不是三个独立验证队列。函数形状比较使用对应训练集经验边际乘积参考分布，不表示因果效应或相关变量的统计可辨识性。

83→74保持保留列逐值及共享切点不变，但每树仍选25变量，均衡覆盖周期由4变3，具体可用变量组合也改变；因此结果衡量固定删变量策略及相伴搜索变化。继续保留原模型，本轮不自动追加实验。
'''
    (pub/'RESULTS_ZH.md').write_text(result)
    (pub/'DATA_ALIGNMENT_AUDIT_ZH.md').write_text(f'''# 截点与数据核验

{table(data)}

复用443家审核冻结企业的旧版conservative C83入组输入，未重新接受修订年报。使用真实financial_available、market_available及48项财务字段可得日数组核对不晚于预测原点；沿用历史保守版本及其PIT来源局限，这不是新增的逐份原始披露版本认证。保留变量逐值、缺失状态及训练共享切点一致。

结局从原始事件链、公告日、公司组映射及既有审核记录重建。训练结局截断到当年6月30日，不能沿用旧年初模型已经截断的标签；未发生/尚未知的未来结局不设为正常。测试标签单独存储在evaluation_input，正式训练DATA_DIR仅含输入X、sample_id与训练标签。

2023截点有1家在当日入组且截至当天随访长度为0，按既有label_record规则从两臂一致排除；有效训练343家。2024及2025训练分别390与442家。训练/测试公司不重叠，两个完整测试窗口也不重叠。公司bootstrap及重复次数逐树配对，共288对；固定九项名单及来源哈希见协议，保留其此前诊断使用截至2024年初信息的来源背景。

首次准备将2023训练数量预期写为344，审计拒绝后发现上述零随访边界；没有拟合。保留失败目录，在新的唯一目录修正预期为343并重新准备，未修改实际标签或强制正随访。详见audit/preparation_history.json。

2026审核队列缺失的上游来源详见audit/source_search.json。原始事件可见不等于已完成首次事件审核，不制造availability、资格或随访。所有测试结局此前已接触，本轮为固定设计比较。

旧27份模型、预测、函数工件以及CURRENT_POST_ST_MODEL.json哈希均保持不变。私有运行目录：`{run}`。公司身份、逐公司结局/预测、参数和切点保留服务器；公开汇总不含2025截点唯一公司的任何风险或结局摘要。
''')
    (pub/'TRAINING_DIAGNOSTICS_ZH.md').write_text(f'''# 运行与数值验证

冻结提交`{frozen['commit']}`，冻结时间{frozen['utc']}；拟合前{passed}项完整测试通过（含独立R检查）。实验未修改核心森林算法。

Slurm作业{gate['job']}，18任务计算、回传、SHA256验证全部成功；实际执行分布{json.dumps(gate['by_host'])}，池墙钟{gate['seconds']:.3f}秒。每任务独立单线程、4GiB地址空间上限，使用Slurm允许的亲和性，没有绕过P/E核或SMT约束。没有失败拟合、追加种子、校准或搜索；准备阶段1次边界数量审计失败保留。

{table(fits[['year','arm','seed','host','fit_seconds','shrink_seconds','seconds','peak_rss_MiB','actual_scored_candidates']])}

{table(pd.read_csv(reports/'leaf_support_summary.csv'))}

收缩来自各自raw森林的直接父节点AJ，alpha=n/(n+10)，n为独立in-bag公司数。所有时间和原因使用相同权重。重放验证原父节点AJ、叶支持、分裂得分及实际候选次数；借入正风险不增加真实退市事件数。

{table(cv[['year','arm','version','seed','verified_rows','verified_times','full_reconstruction','explicit_component_sum','function_reload','forest_reload']])}

前两个可评价窗口在12个月时所有结局状态已知，因此IPCW权重为1；AUC/AP/Brier另外用独立实现复核。2025窗口不计算公开性能。完整复杂度、跨种子集合/函数相关性和预测波动见results。
''')
    (pub/'METHODS_RESULTS_ZH.md').write_text(f'''# 方法与结果段落

本研究将2023、2024和2025年6月30日设为包含当日信息的模型拟合截点，以截点之后12个日历月内新入组的首次财务/审计ST非金融企业作为相应时间外队列。训练使用既有入组时输入并从事件链重建截至各截点可得的竞争结局，保持24个月训练随访上限；评价企业各自在公告后下一交易日开盘前起算12个日历月的先最终退市CIF，先恢复为竞争事件。比较原83输入与固定删除九项的74输入，保持q2、深度4、每树均衡选择25变量、共享分位切点、32树以及直接raw父节点AJ收缩10。每臂种子42、101、202，逐树公司bootstrap配对；先平均逐公司预测，再计算AUC、AP、IPCW Brier与Top20覆盖。

{paragraph}

共完成18次新森林拟合与18个收缩派生。2025截点因缺乏2026年审核冻结入组队列，尚不能评价计划窗口，未报告单一可用公司的性能。完整函数最大重构误差{val['max_explanation_error']:.3e}，全部CIF约束与保存重载检查通过。固定预测条件下完成每窗口1000次公司配对bootstrap；区间不包含完整训练和筛选不确定性。由于83→74改变每树均衡覆盖组合，本比较估计固定删除策略的整体效果，不单独识别变量噪声贡献。
''')
    dest=pub/'results';dest.mkdir(exist_ok=True)
    for p in reports.glob('*.csv'):shutil.copy2(p,dest/p.name)
    ap=pub/'audit';ap.mkdir(exist_ok=True)
    for name in ['freeze.json','source_hashes.json','data_alignment.json','source_search.json','formal_gate.json','validation.json','preparation_history.json','full_tests_pre_fit.log']:
        shutil.copy2(run/'audit'/name,ap/name)
    imm=json.loads((run/'audit/immutable_input_hashes.json').read_text())
    (ap/'input_hashes.json').write_text(json.dumps({p:h for p,h in imm.items() if not p.startswith('code/')},indent=2)+'\n')
    allhash=[]
    for i in range(18):
        folder=run/'results/formal'/str(gate['job'])/f'task-{i}/output'
        for n,h in json.loads((folder/'hashes.json').read_text()).items():
            assert sha(folder/n)==h;allhash.append(dict(year=YEARS[i//6],arm=ARMS[(i%6)//3],seed=SEEDS[i%3],artifact=n,sha256=h))
    (ap/'new_artifact_hashes.json').write_text(json.dumps(allhash,indent=2)+'\n')
    receipt=dict(status='COMPLETE_FITTING_PARTIAL_EVALUATION',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),private_run=str(run),
        source_commit=json.loads((run/'input/protocol.json').read_text())['source_commit'],fitting_commit=frozen['commit'],
        new_forest_fits=18,derived_shrunk_models=18,reused_reference_forests=0,pilot_fits=0,failed_fits=0,preparation_failures=1,
        calibration_fits=0,additional_search_trials=0,slurm_job=gate['job'],actual_execution_by_host=gate['by_host'],
        current_reference_sha256=REFERENCE_SHA,current_reference_replaced=False,preserved_original_artifacts=27,
        training_and_entry_windows=data.to_dict('records'),validation=val,tests_passed=passed,independent_R_checks=True,
        completed_evaluation_windows=['2023-07-01/2024-06-30','2024-07-01/2025-06-30'],
        blocked_evaluation_window='2025-07-01/2026-06-30: missing audited2026 entrants, matched inputs and complete follow-up',
        aggregate_hashes={str(p.relative_to(pub)):sha(p) for p in dest.glob('*.csv')})
    write(pub/'execution_receipt.json',receipt);shutil.copy2(pub/'execution_receipt.json',run/'audit/execution_receipt.json')
    print(table(ens[cols]))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    deliver(a.run,Path(__file__).resolve().parents[2])
