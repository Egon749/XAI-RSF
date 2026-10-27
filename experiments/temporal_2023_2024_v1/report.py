"""Render aggregate-only experiment reports; never publish private model outputs."""
from pathlib import Path
import argparse,json,shutil
import pandas as pd


def table(frame,columns):
    lines=['|'+'|'.join(columns.values())+'|','|'+'|'.join(['---']*len(columns))+'|']
    for _,r in frame.iterrows():
        values=[]
        for k in columns:
            x=r[k]
            values.append('—' if pd.isna(x) else f'{x:.6g}' if isinstance(x,float) else str(x))
        lines.append('|'+'|'.join(values)+'|')
    return '\n'.join(lines)


def main(run,dest):
    out=run/'reports';receipt=json.loads((out/'execution_receipt.json').read_text())
    assert receipt['formal_fits']==6 and not receipt['failed_tasks']
    for name in ['metrics.csv','metrics_seed_summary.csv','metrics_ensemble.csv','warning_metrics.csv',
                 'warning_seed_summary.csv','warning_ensemble.csv','calibration.csv','paired_contrasts.csv',
                 'training_diagnostics.csv','seed_stability.csv','execution_receipt.json','environment.json']:
        assert not (dest/name).exists(),f'Never overwrite completed delivery: {name}'
        shutil.copy2(out/name,dest/name)
    for name in ['M2_provenance.json','pilot_gate.json','training_gate.json','assembly_gate.json','tests_final.txt']:
        shutil.copy2(run/'audit'/name,dest/name)
    m=pd.read_csv(out/'metrics_seed_summary.csv');e=pd.read_csv(out/'metrics_ensemble.csv')
    w=pd.read_csv(out/'warning_seed_summary.csv');we=pd.read_csv(out/'warning_ensemble.csv')
    d=pd.read_csv(out/'training_diagnostics.csv');s=pd.read_csv(out/'seed_stability.csv')
    cal=pd.read_csv(out/'calibration.csv')
    support=cal[(cal.bin==10)&(cal.seed==-1)&(cal.weighting=='record_equal')].copy()
    support['zero_evaluation_weight']=support.n-support.known
    support['zero_weight_fraction']=1-support.known/support.n
    support=support[['scope','model','n','known','zero_evaluation_weight','zero_weight_fraction','prediction','observed']]
    support.to_csv(dest/'high_risk_followup_support.csv',index=False)
    record=m[m.weighting=='record_equal'];company=m[m.weighting=='company_equal']
    cols=dict(scope='评价原点',model='模型',auc_mean='AUC均值',auc_min='AUC最小',auc_max='AUC最大',
              ap_mean='AP均值',ap_min='AP最小',ap_max='AP最大',brier_mean='Brier均值',prediction_mean='风险均值')
    text='''# 2023与2024独立年度结果

**COMPLETED_PROVISIONAL_RETROSPECTIVE**。最终训练原点2012—2020、拟合截止2022-01-01；同一个固定最终模型用于2023和2024。6/6正式模型完成，每个配置的种子42/101/202全部保留。没有逐年增量训练、后期调参、再校准或测试选择。本轮并非严格盲测。

coverage是当前二阶可解释森林；unrestricted是同后端不限阶数参照；M2为原randomForestSRC冻结预测，不重训。三者表示、bootstrap或候选预算的差异见[数据审计](DATA_ALIGNMENT_AUDIT_ZH.md)。下表是各个种子先独立评价，再取指标均值及范围；范围不是统计置信区间。

## 记录等权：主要口径

'''+table(record,cols)+'''

## 公司等权

'''+table(company,cols)+'''

合并指标在2023/2024合并预测上重新计算，不能平均年度AP。IPCW AUC/AP将竞争事件作为已知非目标结局，未知终点权重为0；Brier越低越好。预测均值覆盖所有记录。实际事件/未知样本数见数据审计；不同年度事件基率不同，不能把年度AP差异全部解释为模型变化。

## 三种子预测平均后的集成：辅助口径

以下先平均三个种子的每条预测，再算指标，与上表的“指标均值”不同。固定三种子平均未使用测试表现挑选权重；M2也按同样方式展示。

'''+table(e[e.weighting=='record_equal'],dict(scope='评价原点',model='模型',auc='AUC',ap='AP',ipcw_brier='Brier',prediction_mean='风险均值',observed_ipcw_incidence='IPCW观察发生率',prediction_max='最大风险'))+'''

完整记录/公司等权集成指标见[metrics_ensemble.csv](metrics_ensemble.csv)，全部逐种子指标见[metrics.csv](metrics.csv)。校准十等分的预测均值与IPCW观察率见[calibration.csv](calibration.csv)。观察率仍依赖删失与源资格假设，不能把Brier较低单独解释为完美校准。

## 季度Top20/25预警

'''+table(w,dict(scope='评价原点',model='模型',top_fraction='核查比例',events_mean='捕获事件均值',events_min='最少',events_max='最多',independent_events='独立事件分母',alert_records='核查记录',companies_mean='去重公司均值',unknown_mean='未知结局均值',known_no_target_mean='已知非目标均值',repeats_mean='重复记录均值',lead3_mean='公告前3月均值',lead6_mean='公告前6月均值'))+'''

2023与2024的83、99个事件存在重叠窗口，合并只计137个独立事件。年度数字对应年度预测原点而非事件发生年份。公告前识别按日历月判断；机会分母、季度核查量及重复预警完整记录在[warning_metrics.csv](warning_metrics.csv)。未知结局不计为误报。

三种子集成预警如下，禁止与种子平均捕获数混用：

'''+table(we,dict(scope='评价原点',model='模型',top_fraction='核查比例',detected_events='捕获事件',independent_events='分母',event_coverage='覆盖率',announcement_detected_3m='公告前3月',announcement_detected_6m='公告前6月'))+'''

## 最高预测风险

'''+table(record,dict(scope='评价原点',model='模型',prediction_max_min='各种子最大值的最小值',prediction_max_max='各种子最大值的最大值'))+'''

这些是未来12个月目标ST的原始CIF概率，没有做min-max映射或人为拉伸到1。最高风险不等于模型能力上限；其可信度应结合校准、排序、AP和事件覆盖评价。这里的最大值针对本次固定最终模型和评价样本，不能沿用开发图里的23%概括全部模型。
'''
    text+='''

## 高风险区的随访支持限制

下表为三种子集成、年度/合并风险最高十分位（不是季度Top20）。zero evaluation weight包括源结果未知或未完成可评价随访的记录。它们不能被当作非事件。

'''+table(support,dict(scope='评价原点',model='模型',n='最高十分位记录',known='有正评价权重',zero_evaluation_weight='零评价权重',zero_weight_fraction='零权重比例',prediction='平均预测',observed='继承IPCW观察率'))+'''

coverage最高十分位中2023有69.9%、2024有75.9%的记录没有正评价权重，明显高于M2。未知结果集中在高预测风险区，使边际独立删失与正式校准解释尤其脆弱。IPCW和源资格保护层没有恢复这些真实未知结局；表中的观察率不是该组完整真实发生概率。不能把预测与观察均值之差全部归因于模型高估，也不能因记录等权Brier略低就认定概率更可信。公司等权Brier并未呈现同样优势。
'''
    (dest/'RESULTS_2023_2024_ZH.md').write_text(text)
    gates=receipt['publication'];pilots=[]
    for p in (run/'results/pilot').glob('*/task-*/output/summary.json'):
        x=json.loads(p.read_text());pilots.append({k:x[k] for k in ['variant','fit_seconds','peak_rss_MiB','reload_error','reconstruction_error']})
    train='''# 训练与计算诊断

2/2试运行、48/48独立树批次、6/6合并及验证成功。训练产生6个正式32树模型；不是对两个评价年份重复拟合12次。失败任务为0，未按结果删种子、改变预算或重训M2。所有pool计算和发布状态均成功，模型/输出SHA256经核验。

冻结训练代码：`'''+receipt['training_code']+'''`。训练/配置/数据快照位于私有run/input；评价和报告代码可位于之后的交付提交。模型及预测哈希见[execution_receipt.json](execution_receipt.json)。

## Slurm执行

'''+table(pd.DataFrame([dict(stage=x['stage'],job=x['job'],tasks=x['tasks'],seconds=x['seconds'],by_host=str(x['by_host'])) for x in gates]),dict(stage='阶段',job='作业',tasks='任务数',seconds='pool墙钟秒',by_host='节点完成数'))+'''

physical两节点池，oliver 8P+16E、lemon 8物理核。每项单线程，按独立全局树编号调度，候选搜索和随机种子语义不因节点变化而变。无GPU、无跨节点同步单进程。NFS输入与结果隔离，发布均校验成功。

## 小规模试运行

'''+table(pd.DataFrame(pilots),dict(variant='配置',fit_seconds='4树训练秒',peak_rss_MiB='含预测验证峰值MiB',reload_error='重载误差',reconstruction_error='重构误差'))+'''

## 正式模型

'''+table(d,dict(variant='配置',seed='种子',trees='树',leaves='叶',max_path_variables='最大路径变量数',candidate_evaluations='候选评价数',fit_seconds='各片段耗时和秒',fit_cpu_seconds='训练CPU秒',peak_rss_MiB='合并验证峰值MiB',reload_error='重载误差',reconstruction_error='重构误差',leaf_zero_target_company_support='零目标支持叶',leaf_zero_competing_company_support='零竞争支持叶'))+'''

正式fit_seconds为8个树片段耗时之和，不是并行端到端耗时；pool墙钟另列。每个训练进程地址空间上限6GiB。旧片段worker未单独记录训练RSS，Slurm accounting未启用；不能将合并验证RSS冒充训练峰值。两节点环境已核对，版本见[environment.json](environment.json)。

训练没有迭代数值优化器，因此Hessian、优化gap、系数触边并不适用；本轮也没有可辨识性惩罚。对应诊断是全部样本、全部原生事件时间、两个原因的CIF概率约束，路径阶数、源时钟、零事件叶支持、保存重载和完整函数重构。零竞争事件支持叶如上保留，不能因总样本充足就宣称局部竞争事件估计可靠。

完整单元及独立R检查：45 passed，无跳过，见[tests_final.txt](tests_final.txt)。试运行和正式输出全部保留。没有隐去失败种子或用人为裁剪预测通过验证。
'''
    (dest/'TRAINING_DIAGNOSTICS_ZH.md').write_text(train)
    q=d[d.variant=='coverage']
    interp='''# 最终模型的完整解释诊断

本轮只读编译和检查已拟合模型，没有训练解释代理、删掉交互或调整模型。完整解释位于CIF尺度：基线+全部主效应+全部双变量函数，精确重构原森林预测。少量匿名企业完整案例留在各模型私有cases文件；真实阈值、函数参数及公司明细不提交Git。

'''+table(q,dict(seed='种子',input_columns='输入列',main_functions='主效应函数',interaction_functions='二元函数',distinct_thresholds='不同分裂切点',spatial_table_cells='空间格',scalar_table_cells='全时间/原因标量格',low_support_cell_fraction='低支持格比例',zero_support_cell_fraction='零支持格比例',reconstruction_error='最大重构误差'))+'''

不同分裂切点按变量和切点组合去重，包含缺失专用分裂的负无穷哨兵，不全是有限数值切点。空间格与全时间/原因的标量格不是同一指标；低支持按既有实现每格少于10条训练记录统计。乘积参考下可能出现训练数据缺乏支持的组合，精确分解不能保证这些组合的科学可靠性。最多两变量的路径约束全部满足，最大深度仍可达到6；这是低阶交互限制，不是把森林限制为两个变量。

## 跨种子稳定性

在固定的512条训练参考记录上，按既有开发审计定义评价365日目标CIF。函数集合Jaccard基于所有入选项；相关性和符号一致性只对共同项计算，常数项不进入相关性。这里只评价算法随机性，不是企业重抽样或完整训练不确定性。

'''+table(s,dict(seed_a='种子A',seed_b='种子B',main_jaccard='主效应Jaccard',pair_jaccard='交互Jaccard',order1_common_functions='共同主效应',order2_common_functions='共同交互',order1_median_correlation='主效应相关中位',order2_median_correlation='交互相关中位',order2_median_sign_agreement='交互符号一致率中位'))+'''

2023和2024共用同一组模型，因此两年的函数结构相同；不能把结构相同当作跨年度重新训练的稳定性证据。函数相加给出CIF概率贡献，不是风险率乘数、因果效应或SHAP代理拟合。精确重构不等于解释简单、统计可辨识或所有交互稳定。

本轮无可辨识性惩罚，也未单独消融共享切点、树级变量资格与q2。不得据此宣称某项惩罚有效、预算有效，或已证明相对所有既有可解释森林的独立方法创新。
'''
    (dest/'INTERPRETABILITY_COMPARISON_ZH.md').write_text(interp)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--dest',type=Path,default=Path(__file__).parent)
    a=p.parse_args();main(a.run,a.dest)
