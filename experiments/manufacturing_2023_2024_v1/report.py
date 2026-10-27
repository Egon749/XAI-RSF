"""Aggregate-only manufacturing refit delivery, with same-cohort references."""
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
    for name in ['M2_provenance.json','pilot_gate.json','training_gate.json','assembly_gate.json']:
        shutil.copy2(run/'audit'/name,dest/name)
    m=pd.read_csv(out/'metrics_seed_summary.csv');e=pd.read_csv(out/'metrics_ensemble.csv')
    w=pd.read_csv(out/'warning_seed_summary.csv');we=pd.read_csv(out/'warning_ensemble.csv')
    diag=pd.read_csv(out/'training_diagnostics.csv');stable=pd.read_csv(out/'seed_stability.csv')
    c=pd.read_csv(out/'calibration.csv');support=c[(c.bin==10)&(c.seed==-1)&(c.weighting=='record_equal')].copy()
    support['zero_evaluation_weight']=support.n-support.known;support['zero_weight_fraction']=1-support.known/support.n
    support=support[['scope','model','n','known','zero_evaluation_weight','zero_weight_fraction','prediction','observed']]
    support.to_csv(dest/'high_risk_followup_support.csv',index=False)
    cols=dict(scope='评价原点',model='模型',auc_mean='AUC均值',auc_min='AUC最小',auc_max='AUC最大',ap_mean='AP均值',ap_min='AP最小',ap_max='AP最大',brier_mean='Brier均值',prediction_mean='风险均值')
    text='''# 制造业重训：2023与2024分别评价

状态COMPLETED_PROVISIONAL_RETROSPECTIVE_MANUFACTURING。训练原点2012—2020、拟合截止2022-01-01。两年共用相同最终模型，6个新拟合全部保留，无后期调参。制造业为历史财报关联行业代理口径，未认证行业PIT。

coverage/unrestricted：本轮制造业训练；coverage_allsector/unrestricted_allsector：上一轮全行业训练的固定模型；M2_allsector：原全行业M2冻结预测。**所有指标都在同一制造业评价子集上计算。** 后3者未重新训练，M2不能被称为制造业训练模型。

## 记录等权，种子指标均值与范围

'''+table(m[m.weighting=='record_equal'],cols)+'''

## 公司等权，种子指标均值与范围

'''+table(m[m.weighting=='company_equal'],cols)+'''

每个种子在年度/合并预测上独立计算IPCW AUC、AP和Brier后，再报告种子均值与范围。合并AP不是年度AP平均。范围不是置信区间；未知结果不算正常。公司等权沿用评价集合内公司记录数倒数，删失分布仍为原边际reverse-KM。

## 三种子预测平均后的集成（辅助口径）

'''+table(e[e.weighting=='record_equal'],dict(scope='评价原点',model='模型',auc='AUC',ap='AP',ipcw_brier='Brier',prediction_mean='风险均值',observed_ipcw_incidence='IPCW观察率',prediction_max='最大风险'))+'''

## 每季度Top20/25：种子均值

'''+table(w,dict(scope='评价原点',model='模型',top_fraction='核查比例',events_mean='捕获均值',events_min='最少',events_max='最多',independent_events='独立事件分母',alert_records='核查记录',companies_mean='去重公司均值',unknown_mean='未知结局均值',known_no_target_mean='已知非目标均值',repeats_mean='重复预警均值',lead3_mean='公告前3月均值',lead6_mean='公告前6月均值'))+'''

在制造业每季度风险排名内选取ceil(比例×记录数)，并列按sample_id排序。各模型具有相同核查容量，年度去重公司与独立事件。2023、2024事件分母44、60，跨年重叠窗口去重后合并79，不能把44+60用作合并分母。完整季度记录和机会分母见warning_metrics.csv。

## 集成Top20/25

'''+table(we,dict(scope='评价原点',model='模型',top_fraction='核查比例',detected_events='捕获',independent_events='分母',event_coverage='覆盖率',announcement_detected_3m='公告前3月',announcement_detected_6m='公告前6月'))+'''

## 最高风险与高风险区随访支持

'''+table(m[m.weighting=='record_equal'],dict(scope='评价原点',model='模型',prediction_max_min='各种子最大风险最小值',prediction_max_max='各种子最大风险最大值'))+'\n\n'+table(support,dict(scope='评价原点',model='模型',n='最高十分位记录',known='正评价权重',zero_evaluation_weight='零评价权重',zero_weight_fraction='零权重比例',prediction='平均预测',observed='继承IPCW观察率'))+'''

风险为原始12个月CIF，没有min-max拉伸。行业无法确认的记录已因预先冻结的行业规则排除，不能将其当作非制造业或已知正常。此处的未知比例与上一轮全体样本不同，不能将样本筛选导致的变化解释成模型修复。独立删失和源资格仍未经正式认证；完整校准十等分保留在calibration.csv。
'''
    (dest/'RESULTS_2023_2024_ZH.md').write_text(text)
    train='''# 制造业训练诊断

2个试运行、48个独立四树批次、6个正式森林合并/验证全部成功，失败0。未因指标差改变配置或筛掉种子；M2及全行业参考模型没有重新拟合。

冻结训练提交：`'''+receipt['training_code']+'''`。原训练引擎与候选语义不变，每个模型32树，复用现有temporal_2023_2024_v1运行入口。两节点physical Slurm池，每任务单线程，NFS输入/输出分离，6GiB地址空间上限。所有计算和发布哈希通过。

'''+table(pd.DataFrame([dict(stage=g['stage'],job=g['job'],tasks=g['tasks'],seconds=g['seconds'],hosts=str(g['by_host'])) for g in receipt['publication']]),dict(stage='阶段',job='作业',tasks='任务',seconds='墙钟秒',hosts='节点完成数'))+'\n\n'+table(diag,dict(variant='配置',seed='种子',trees='树',leaves='叶',max_path_variables='最大路径变量数',candidate_evaluations='候选评价数',fit_cpu_seconds='训练CPU秒',peak_rss_MiB='合并验证峰值MiB',reload_error='重载误差',reconstruction_error='重构误差',leaf_zero_target_company_support='零目标支持叶',leaf_zero_competing_company_support='零竞争支持叶'))+'''

fit_seconds是8个片段耗时之和，不能作为并行墙钟。正式片段没有独立RSS测量，表中为合并验证进程峰值；原Slurm accounting未启用。独立pilot输出保存其进程RSS。Hessian/优化gap不适用于非参数叶估计森林；本轮无可辨识性惩罚。未裁剪预测通过概率检查。

准备阶段系统Python环境没有改动训练环境。原表核对使用既有python_calamine；pandas关于fillna未来行为的警告保留在私有prepare.log，实际布尔筛选已经逐项核对。行业证据限制见数据审计。
'''
    (dest/'TRAINING_DIAGNOSTICS_ZH.md').write_text(train)
    interp='''# 制造业模型的完整解释诊断

完整解释是训练参考边际乘积下的CIF基线、主效应及二元函数，精确重构原森林；不是风险率贡献、因果效应或SHAP代理。私有模型及完整公司案例留在服务器，Git只提交汇总。

'''+table(diag[diag.variant=='coverage'],dict(seed='种子',main_functions='主效应',interaction_functions='二元函数',distinct_thresholds='不同分裂切点',spatial_table_cells='空间格',scalar_table_cells='全时间/原因标量格',low_support_cell_fraction='低支持格比例',zero_support_cell_fraction='零支持格比例',reconstruction_error='重构误差'))+'''

切点数按变量/切点组合去重，含缺失专用分裂哨兵。低支持格为少于10条训练记录。精确重构不证明未观测变量组合可靠。

'''+table(stable,dict(seed_a='种子A',seed_b='种子B',main_jaccard='主效应Jaccard',pair_jaccard='交互Jaccard',order1_median_correlation='共同主效应相关中位',order2_median_correlation='共同交互相关中位',order2_median_sign_agreement='共同交互符号一致率中位'))+'''

稳定性参考为固定512条制造业训练记录、365日目标CIF；集合Jaccard包括所有选择项，函数相关只看共同非恒定项。只刻画三个种子的算法随机性，不是公司bootstrap置信区间。两年共用模型，不能据此证明跨年度重训稳定。没有启用或验证可辨识性惩罚的独立贡献。
'''
    (dest/'INTERPRETABILITY_COMPARISON_ZH.md').write_text(interp)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);p.add_argument('--dest',type=Path,default=Path(__file__).parent)
    a=p.parse_args();main(a.run,a.dest)
