"""Aggregate-only delivery for the finite matched path-order experiment."""
from pathlib import Path
import argparse,json,shutil
import pandas as pd
from experiments.temporal_2023_2024_v1.report import table


def main(run,dest):
    out=run/'reports';receipt=json.loads((out/'execution_receipt.json').read_text())
    assert receipt['formal_fits']==6 and not receipt['failed_tasks']
    for name in ['metrics.csv','metrics_seed_summary.csv','metrics_ensemble.csv','warning_metrics.csv',
                 'warning_seed_summary.csv','warning_ensemble.csv','calibration.csv','paired_contrasts.csv',
                 'training_diagnostics.csv','seed_stability.csv','execution_receipt.json','environment.json']:
        assert not (dest/name).exists()
        shutil.copy2(out/name,dest/name)
    for name in ['M2_provenance.json','pilot_gate.json','training_gate.json','assembly_gate.json','tests_final.txt']:
        shutil.copy2(run/'audit'/name,dest/name)
    m=pd.read_csv(out/'metrics_seed_summary.csv');e=pd.read_csv(out/'metrics_ensemble.csv')
    w=pd.read_csv(out/'warning_seed_summary.csv');we=pd.read_csv(out/'warning_ensemble.csv')
    d=pd.read_csv(out/'training_diagnostics.csv');s=pd.read_csv(out/'seed_stability.csv')
    parent=Path(json.loads((run/'audit/reuse_receipt.json').read_text())['parent'])
    old=pd.read_csv(parent/'reports/training_diagnostics.csv');old=old[old.variant=='coverage'].copy();old['variant']='q2'
    for order in [1,2]:
        old[f'order{order}_functions']=old['main_functions' if order==1 else 'interaction_functions']
    for order in [3,4]:old[f'order{order}_functions']=0
    comp=pd.concat([old,d],ignore_index=True)
    cols=['variant','seed','main_functions','interaction_functions','order1_functions','order2_functions','order3_functions','order4_functions',
          'distinct_thresholds','spatial_table_cells','scalar_table_cells','low_support_cell_fraction','zero_support_cell_fraction','reconstruction_error']
    comp[cols].to_csv(dest/'complexity_comparison.csv',index=False)
    intro='''# q3/q4制造业匹配实验：2023和2024

本轮在已接触后期结果后，按用户要求比较q3、q4；属于探索性算法比较（POST_EXPOSURE_EXPLORATORY），不是独立测试或严格盲测。训练56,458条制造业公司季度记录、177个独立目标事件，原点2012—2020、拟合截止2022-01-01。两年使用同一批模型。数据和行业时钟限制仍为PROVISIONAL。

q2、q3、q4除了路径允许的不同变量数外，训练输入、3个种子、32棵树、深度6、共享切点、每树25个候选资格变量和分裂目标保持一致。q2复用已冻结模型。unrestricted为原制造业不限阶数参考，但候选/切点策略不匹配；M2_allsector为全行业训练的原M2，在同一制造业评价记录上计算，未重训。

## 记录等权：逐种子指标均值和范围

'''
    mc=dict(scope='评价原点',model='模型',auc_mean='AUC均值',auc_min='AUC最小',auc_max='AUC最大',ap_mean='AP均值',ap_min='AP最小',ap_max='AP最大',brier_mean='Brier均值',prediction_mean='风险均值')
    text=intro+table(m[m.weighting=='record_equal'],mc)+'\n\n## 公司等权\n\n'+table(m[m.weighting=='company_equal'],mc)+'''

AUC和AP使用继承的源一致性IPCW权重；Brier沿用原协议。未知结局不当作正常，保留在风险排名。合并AP从合并预测重新计算。种子范围不是置信区间，三种子不等于完整训练不确定性评估。

## Top20/25季度核查：逐种子均值

'''+table(w,dict(scope='评价原点',model='模型',top_fraction='核查比例',events_mean='捕获均值',events_min='最少',events_max='最多',independent_events='独立事件分母',alert_records='核查记录',companies_mean='去重公司均值',unknown_mean='未知结局均值',known_no_target_mean='已知非目标均值',repeats_mean='重复预警均值',lead3_mean='公告前3月',lead6_mean='公告前6月'))+'''

年度独立事件分母分别44和60，跨年窗口重叠后合并去重79；不能相加成104。核查每季度ceil(比例×记录数)，并列按sample_id排序。均值中的小数来自三个模型，不是“部分事件”。完整季度及机会分母见warning_metrics.csv。

## 三种子平均预测集成（辅助口径）

'''+table(e[e.weighting=='record_equal'],dict(scope='评价原点',model='模型',auc='AUC',ap='AP',ipcw_brier='Brier',prediction_mean='风险均值',observed_ipcw_incidence='IPCW观察率',prediction_max='最大风险'))+'\n\n'+table(we,dict(scope='评价原点',model='模型',top_fraction='核查比例',detected_events='捕获',independent_events='分母',event_coverage='覆盖率'))+'''

集成指标与“种子指标均值”不同，不能混用。预测为原始12个月目标CIF，未做min-max映射或后期校准；完整十等分校准见calibration.csv。观察率依赖继承的删失假设。
'''
    (dest/'RESULTS_2023_2024_ZH.md').write_text(text)
    train='''# 训练与数值诊断

2个四树试运行、48个四树训练片段和6个正式模型组装/完整解释核验全部成功。所有种子保留，未因效果调整预算；M2、q2和unrestricted参考没有重新训练。

训练提交：`'''+receipt['training_code']+'''`。每任务单CPU、BLAS线程1、6GiB地址空间上限；oliver使用8P+16E，lemon使用8物理核，现有Slurm池和NFS。单个Python任务不跨节点。GPU不参与此算法。

'''+table(pd.DataFrame([dict(stage=g['stage'],job=g['job'],tasks=g['tasks'],seconds=g['seconds'],hosts=str(g['by_host'])) for g in receipt['publication']]),dict(stage='阶段',job='作业',tasks='任务',seconds='墙钟秒',hosts='节点完成数'))+'\n\n'+table(d,dict(variant='配置',seed='种子',trees='树',leaves='叶',max_path_variables='最大路径变量数',candidate_evaluations='候选数',fit_cpu_seconds='训练CPU秒',training_shard_peak_rss_MiB='训练片段峰值MiB',peak_rss_MiB='组装验证峰值MiB',reload_error='重载误差',reconstruction_error='全记录重构误差',explicit_component_sum_error='逐函数求和误差',leaf_zero_target_company_support='零目标支持叶',leaf_zero_competing_company_support='零竞争支持叶'))+'''

CPU秒与片段耗时之和不能当作并行墙钟；RSS分训练片段/组装进程分别实测。全23,799条评价记录、全部原生时间和两个原因检查CIF概率及完整重构；另固定32条记录逐函数显式求和。没有裁剪预测或用二阶代理替代高阶解释。叶原因事件支持不足是统计可靠性限制，不能仅靠数值精确排除。

Hessian、优化gap不适用于此非参数叶估计森林。本轮无可辨识性惩罚，也不证明该惩罚有效。全部51项测试包含两项独立R检查；q2原模型回放的新旧解释器结果及复杂度一致。
'''
    (dest/'TRAINING_DIAGNOSTICS_ZH.md').write_text(train)
    interp='''# 完整解释与阶数代价

q3保留全部主效应、二元和三元函数，q4另含四元函数。函数依据固定训练边际乘积中心化，精确相加为原始CIF；不是SHAP代理、风险率贡献或因果影响。原森林保存和完整函数保存均可重载。

## 同预算q2/q3/q4复杂度

'''+table(comp,dict(variant='模型',seed='种子',order1_functions='主效应',order2_functions='二元',order3_functions='三元',order4_functions='四元',distinct_thresholds='不同切点',spatial_table_cells='空间格',scalar_table_cells='时间/原因标量格',low_support_cell_fraction='低支持格比例',zero_support_cell_fraction='零支持格比例',reconstruction_error='重构误差'))+'''

函数数按完整网格在任一原生时间/原因非零计数；空间格含缺失状态及逻辑笛卡尔积。低支持为少于10条训练记录，零支持为没有训练记录的组合。函数使用无损因子形式存储，逻辑格数不等于磁盘大小。切点按变量/值组合计，含缺失专用哨兵。共享切点并不保证高阶空间有足够样本。

## 高阶项不能省略

'''+table(d,dict(variant='模型',seed='种子',order1_mean_abs_combined_contribution='一阶合计绝对贡献均值',order2_mean_abs_combined_contribution='二阶合计绝对贡献均值',order3_mean_abs_combined_contribution='三阶合计绝对贡献均值',order4_mean_abs_combined_contribution='四阶合计绝对贡献均值',truncated_q2_max_error='截成二阶的最大遗漏误差'))+'''

贡献为每条评价记录实际12日历月的目标CIF概率单位：先将同阶函数相加，再对绝对值取记录均值，**不是所有单函数绝对贡献之和，也不是重要性占比**。截断误差在两个原因及所有评价记录上取最大值。展示为百分点需乘100。只展示主效应和二阶图时，必须同时交代高阶剩余贡献。

## 跨种子稳定性

'''+table(s,dict(model='模型',order='阶数',seed_a='种子A',seed_b='种子B',functions_a='A结构函数数',functions_b='B结构函数数',common='共同项',jaccard='Jaccard',nonconstant_common='共同非恒定项',median_correlation='相关中位',median_sign_agreement='符号一致率中位'))+'''

参考为固定512条制造业训练记录、365日目标CIF。集合是结构子集（可能包含恒零项）；函数数表是完整时间/原因非零项。相关仅共同非恒定函数，空集无值，不因共同项少而证明全结构稳定。乘积参考下中心化固定了表达约定，不保证相关变量下统计可辨识。三/四阶函数需高维切片，不能像q2一样只用一维/二维图覆盖全部函数。精确、简单、可靠应分开评价。低/零支持格比例是未加权的逻辑网格比例，含无样本的缺失状态；它不能直接等同于真实企业预测出错概率，高阶网格增多会机械增加稀疏组合。
'''
    (dest/'INTERPRETABILITY_COMPARISON_ZH.md').write_text(interp)
    audit=json.loads((run/'audit/data_alignment.json').read_text())
    (dest/'DATA_ALIGNMENT_AUDIT_ZH.md').write_text('''# 输入复用与证据限制

本轮逐文件复制已冻结制造业训练、评价、样本ID/公司组/时间/结局/特征顺序、原M2和参考预测；文件哈希复核通过，未重构样本或补造availability。详见reuse_receipt.json及上一阶段制造业数据审计。原始数据hash与行业证据沿用原审计，训练/预测不涉及重新提取。

'''+table(pd.DataFrame([dict(split=k,**v) for k,v in audit['source_counts'].items()]),dict(split='集合',records='记录',groups='公司组',target_records='目标记录',target_events='独立目标事件',competing_records='竞争记录',competing_events='独立竞争事件',unknown_endpoint_records='未知结局',origin_unresolved='原点资格未决',zero_followup='零随访'))+'''

制造业是与实际采用财报期关联的历史行业代理。原行业版本可得时刻、派生财务指标所有组成数据版本及部分随访/质量语义仍未正式认证，不能将报告期当作披露日。未知结局未改正常。2023/2024已有研究接触，本轮又新增阶数比较，明确属于后暴露探索性模型开发，不能声称独立盲测或确认性验证。上游阻塞来源与上一轮完全相同，不因q阶数改变而解决。
''')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);p.add_argument('--dest',type=Path,default=Path(__file__).parent)
    a=p.parse_args();main(a.run,a.dest)
