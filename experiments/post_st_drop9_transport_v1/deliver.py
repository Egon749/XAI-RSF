"""Generate aggregate reports from verified outputs, without fitting."""
from pathlib import Path
import argparse,datetime,json,shutil
import numpy as np
import pandas as pd
from experiments.post_st_drop9_v1.deliver import table
from experiments.post_st_leaf_shallow_v1.analyze import artifacts
from .common import *


def deliver(run,repo):
    pub=repo/'experiments'/NAME;reports=run/'reports'
    val=json.loads((run/'audit/validation.json').read_text());assert val['status']=='PASS'
    gate=json.loads((run/'audit/formal_gate.json').read_text());frozen=json.loads((run/'audit/freeze.json').read_text())
    m=pd.read_csv(reports/'metrics.csv',dtype={'seed':str});w=pd.read_csv(reports/'warnings.csv',dtype={'seed':str})
    cv=pd.read_csv(reports/'function_validation.csv',dtype={'seed':str});diff=pd.read_csv(reports/'paired_differences.csv')
    fits=pd.read_csv(reports/'execution_tasks.csv');support=pd.read_csv(reports/'horizon_support.csv');data=pd.read_csv(reports/'cohort_audit.csv')
    preserved=[];current=json.loads((repo/'CURRENT_POST_ST_MODEL.json').read_text())
    for year,arm,seed,folder in artifacts(HISTORIES[2023]):
        if arm!='shallow_parent':continue
        r=next(v for v in current['saved_models'] if v['test_year']==year and v['seed']==int(seed))
        for name,key in [('forest_PRIVATE.json','forest_sha256'),('predictions_PRIVATE.npz','predictions_sha256'),('functions_PRIVATE.npz','functions_sha256')]:
            assert sha(folder/name)==r[key];preserved.append(dict(year=year,seed=seed,artifact=name,sha256=r[key],unchanged=True))
    later=pd.read_csv(repo/'experiments/post_st_2025_shallow_parent_v1/results/model_hashes.csv')
    for seed,folder in references(2025).items():
        r=later[(later.model=='xai_q2')&(later.seed==seed)].iloc[0]
        for name,key in [('forest_PRIVATE.json','forest_sha256'),('predictions_PRIVATE.npz','prediction_sha256'),('functions_PRIVATE.npz','functions_sha256')]:
            assert sha(folder/name)==r[key];preserved.append(dict(year=2025,seed=seed,artifact=name,sha256=r[key],unchanged=True))
    assert len(preserved)==27
    pd.DataFrame(preserved).to_csv(reports/'reference_preservation.csv',index=False)
    ens=m[(m.seed=='ensemble')&(m.cause==1)&(m.status=='SUPPORTED')].merge(
        w[['year','arm','seed','horizon_months','checked','captured','target_events','coverage']],on=['year','arm','seed','horizon_months'])
    ens.to_csv(reports/'ensemble_comparison.csv',index=False)
    cc=['year','arm','seed','main_functions','interaction_functions','spatial_table_cells','scalar_table_cells','distinct_thresholds']
    cv[cc].to_csv(reports/'function_complexity.csv',index=False)
    columns=['year','horizon_months','arm','auc','ap','brier','risk_mean','captured','target_events','coverage']
    statements=[]
    for (year,h),g in ens.groupby(['year','horizon_months'],sort=True):
        a=g.set_index('arm').loc['reference83'];b=g.set_index('arm').loc['drop9']
        statements.append(f"{year}年{h}个月：AUC {a.auc:.6f}→{b.auc:.6f}，AP {a.ap:.6f}→{b.ap:.6f}，"
            f"Brier {a.brier:.6f}→{b.brier:.6f}，核查{int(a.checked)}家，捕获{int(a.captured)}/{int(a.target_events)}→{int(b.captured)}/{int(b.target_events)}。")
    result=f'''# 固定删除九项：2023与2025结果

六个新森林全部成功（两年×三个种子，每个32树），派生六个强度10父节点AJ模型；直接复用六个原83列模型。每年度先对逐公司预测取三个种子的平均，再计算指标。没有额外调参、校准或替换原模型。

## 先最终退市

{table(ens[columns])}

reference83为原83列模型，drop9为删除同九项后的74列模型。AUC/AP越高越好，Brier越低越好。Top20按当年公司数向上取整，2023核查8家、2025核查11家。

{chr(10).join(statements)}

三个可评价的年度/时点组合，其平均预测器的AUC、AP与Brier点估计均改善。2025三个单种子的这三项指标也都改善；2023的单种子结果则有升有降，见完整metrics.csv。2023的24个月AP提高，但前20%捕获没有增加，说明整体排序指标与固定核查额度的收益并不相同。

结合已完成的2024实验：12个月的AUC/AP提升出现在三个年度，24个月则2023改善、2024下降。该删变量策略有时点差异，本轮不据此统一替换原83列研究参考模型。2025的24个月尚无可评价结果。

原模型及2024已经完成的结果不覆盖。均值风险与观察发生率、校准分箱、恢复原因和全部单种子见results/metrics.csv及calibration_bins.csv；不能把排序变好直接解释为概率校准改善。

## 观察支持

{table(support)}

2025观察截止2026-06-30。12个月终点均已知，24个月无人达到时点且25家结局未知；因此24个月AUC、AP、Brier、校准与捕获率留空。已发生的6例先退市和22例先恢复，只是截至观察截止的累计事件，不能把其余25家设为正常。2025的24个月预测仍保留在私有文件。

## 配对差值

{table(diff[['year','horizon_months','metric','difference','lower95','upper95','undefined_bootstraps']])}

差值为删除后减原模型。各年1000次公司重抽样，两臂共用索引，同一年支持的时点也共用索引；独立随机种子2026110109与年份组成随机流。区间描述固定预测下的测试样本波动，不包含完整训练和变量选择不确定性。某次重抽样没有目标事件时，AUC/AP等无法计算，按上表计数，不改换抽样或补抽。

## 完整解释

{table(cv[cv.seed=='ensemble'][cc])}

主效应和交互计数为全部原生时点/原因上实际非零的函数。空间格数包含缺失状态，不乘时间与原因维；标量格数另乘对应的原生时点数和两个原因。两年所有单种子及96树平均预测器都验证完整q2表示，最大重构误差{val['max_explanation_error']:.3e}，小于1e-10。全部概率约束与模型保存重载通过，没有截断函数或裁剪CIF。

维数从83减为74后，保留变量值、缺失状态和共享切点不变，公司bootstrap逐棵配对；每树仍25变量，但均衡覆盖周期4→3，具体可用变量组合改变。因此本轮衡量固定删变量策略及其带来的搜索变化，不是单个变量效应的因果检验。跨种子预测波动及所有种子结果完整保存。

训练截止、名单来源及选用同一九项的决定见冻结协议。当前参考模型继续保留，本轮没有自动追加搜索。
'''
    (pub/'RESULTS_ZH.md').write_text(result)
    audit=f'''# 输入和工件核验

{table(data[['year','training_companies','training_target_events','training_recoveries','training_latest_entry_year','training_latest_stop','evaluation_companies','old_columns','new_columns']])}

2023使用2012—2022入组、信息严格早于2023-01-01；2025使用2012—2024入组、信息严格早于2025-01-01。两套训练集嵌套，不能将它们的记录/事件相加当作独立样本。原点、首次财务/审计ST非金融公司范围、竞争事件编码和24个月训练随访上限全部沿用原冻结C输入。

仅删除同九项字段，列表与2024实验完全一致；其余列保持原序和逐值一致，所有非X数组一致。各年公司标识和预测文件顺序一致，训练/评价公司不重叠。金融/市场可得日不晚于入组公告后下一交易日开盘前；沿用已有保守版本口径，没有另行接受修订年报值。

192棵树的公司bootstrap和重复次数哈希配对，新实际inbag逐棵核验；保存的旧模型原节点也按同一抽样重放。原模型的预测逐值复现；32行历史年度/种子/时点/原因指标重新计算一致，包括原来不支持的2025 24个月空值。

2023名单使用用户选定的固定九项，诊断来源截止2024年初；2023年前的诊断其中8项为疑似冗余。具体差异字段和来源哈希在协议selection_provenance中保存。2025沿用同一名单，未进行新筛选。测试结果此前已被查看，原始披露版本的独立PIT核验限制继承。

测试标签存于独立evaluation_input，不在训练作业DATA_DIR内；训练代码只读取评价X和sample_id。模型、函数参数、真实切点、逐公司预测和bootstrap索引保留于私有运行目录：`{run}`。公开目录只有汇总、代码、协议、环境和哈希。
'''
    (pub/'DATA_ALIGNMENT_AUDIT_ZH.md').write_text(audit)
    diagnosis=f'''# 运行和数值诊断

冻结代码提交`{frozen['commit']}`，冻结UTC时间{frozen['utc']}。配置差异表仅input_columns为83→74；主模型算法没有新增修改。mtry75继续沿用上轮已验证的有效候选上限语义，每树25变量，其他设置不变。

拟合前190项测试通过，包含独立R检查；1条既有EBM缺失值可视化警告。新增实质测试保护“不支持的24个月不能把删失当阴性”的评价分支。

Slurm作业{gate['job']}通过两节点池调度；本批实际执行主机为{json.dumps(gate['by_host'])}。六个任务均计算完成、回传完成并验证SHA256，没有重训或失败拟合；池内墙钟{gate['seconds']:.3f}秒。每项独立单线程，4GiB地址空间上限，按Slurm亲和性执行。

{table(fits[['year','seed','host','fit_seconds','shrink_seconds','seconds','peak_rss_MiB','actual_scored_candidates']])}

父节点收缩使用各自raw森林的直接父节点AJ，按叶内独立inbag公司数n计算n/(n+10)，各时间和原因使用相同权重。重放检查分裂得分、独立公司数及raw父节点估计均通过；收缩不增加真实事件支持。

{table(pd.read_csv(reports/'leaf_support_summary.csv'))}

{table(cv[['year','arm','seed','verified_rows','verified_times','full_reconstruction','explicit_component_sum','function_reload','forest_reload']])}

每个年度在全部训练和评价记录、全部原生时间点及两个原因验证。支持的评价时点标签均已知，IPCW权重均为1，AUC/AP/Brier与独立实现一致。

准备阶段曾发生1次记录工件元数据时seed重复键错误，尚未拟合；修正后使用新的唯一运行目录。旧失败目录保留，详见audit/preparation_history.json。此错误不是拟合失败，也没有删除失败种子。
'''
    (pub/'TRAINING_DIAGNOSTICS_ZH.md').write_text(diagnosis)
    paragraph='\n\n'.join(statements)
    (pub/'METHODS_RESULTS_ZH.md').write_text('''# 方法与结果段落

我们将此前冻结的九项变量删除策略应用于2023与2025年度队列，并与同年度保存的83变量q2森林配对比较。每年度训练截止、样本、随访、缺失处理和共享切点保持不变；保留q=2、最大深度4、每树均衡选择25变量、32树/种子及强度10的直接父节点AJ收缩。三个固定种子为42、101、202，公司bootstrap逐树配对。新增六个森林，原模型不重训。先平均逐公司预测后计算AUC、AP、IPCW Brier与年度Top20捕获。2023评价12和24个月；2025观察截至2026-06-30，仅12个月具备终点评价支持。变量维数减少改变了均衡调度的组合，故结果属于整体删变量策略的比较。

'''+paragraph+f'\n\n完整函数分解的最大绝对重构误差为{val["max_explanation_error"]:.3e}，全部原生时点、两原因的概率合法性与保存重载一致性通过。固定模型条件下进行1000次配对公司bootstrap，无法计算的重抽样次数完整保留；未自动变更研究参考模型。\n')
    dest=pub/'results';dest.mkdir(exist_ok=True)
    for p in reports.glob('*.csv'):shutil.copy2(p,dest/p.name)
    ap=pub/'audit';ap.mkdir(exist_ok=True)
    for name in ['freeze.json','source_hashes.json','data_alignment.json','formal_gate.json','validation.json','preparation_history.json','full_tests_pre_fit.log']:
        shutil.copy2(run/'audit'/name,ap/name)
    imm=json.loads((run/'audit/immutable_input_hashes.json').read_text())
    (ap/'input_hashes.json').write_text(json.dumps({p:h for p,h in imm.items() if not p.startswith('code/')},indent=2)+'\n')
    allhash=[]
    for i in range(6):
        folder=run/'results/formal'/str(gate['job'])/f'task-{i}/output'
        for n,h in json.loads((folder/'hashes.json').read_text()).items():
            assert sha(folder/n)==h;allhash.append(dict(year=YEARS[i//3],seed=SEEDS[i%3],artifact=n,sha256=h))
    (ap/'new_artifact_hashes.json').write_text(json.dumps(allhash,indent=2)+'\n')
    assert sha(repo/'CURRENT_POST_ST_MODEL.json')==REFERENCE_SHA
    receipt=dict(status='COMPLETE',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),private_run=str(run),
        source_commit='b997e734dd4fa900b8425735e175c4a9b843c8a4',fitting_commit=frozen['commit'],
        new_forest_fits=6,derived_shrunk_models=6,reused_reference_forests=6,pilot_fits=0,failed_fits=0,preparation_failures=1,
        calibration_fits=0,additional_search_trials=0,slurm_job=gate['job'],actual_execution_by_host=gate['by_host'],
        current_reference_sha256=REFERENCE_SHA,current_reference_replaced=False,preserved_original_artifacts=27,training_cohorts=data.to_dict('records'),
        horizon_support=support.to_dict('records'),validation=val,tests='190 passed, independent R enabled, one existing EBM warning',
        aggregate_hashes={str(p.relative_to(pub)):sha(p) for p in dest.glob('*.csv')})
    write(pub/'execution_receipt.json',receipt);shutil.copy2(pub/'execution_receipt.json',run/'audit/execution_receipt.json')
    print(table(ens[columns]))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    deliver(a.run,Path(__file__).resolve().parents[2])
