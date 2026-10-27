"""Write aggregate-only research reports; preserve limits and negative results."""
from pathlib import Path
import argparse,json,shutil
import pandas as pd
from experiments.temporal_2023_2024_v1.report import table


def main(run,dest):
    out=run/'reports';r=json.loads((out/'execution_receipt.json').read_text());assert r['formal_tasks']==261 and not r['failed_tasks']
    for p in out.iterdir():
        if p.suffix in ['.csv','.json']:
            assert not (dest/p.name).exists();shutil.copy2(p,dest/p.name)
    for name in ['pilot_gate.json','formal_gate.json','coverage_gate.json','input_provenance.json','tests_final.txt']:shutil.copy2(run/'audit'/name,dest/name)
    m=pd.read_csv(out/'metrics_summary.csv');c=pd.read_csv(out/'native_complexity.csv');w=pd.read_csv(out/'warning_metrics.csv');d=pd.read_csv(out/'task_diagnostics.csv');sim=pd.read_csv(out/'simulation_summary.csv');f=pd.read_csv(out/'shap_fidelity.csv');st=pd.read_csv(out/'paired_training_stability.csv');bg=pd.read_csv(out/'background_stability.csv');native=pd.read_csv(out/'native_function_stability.csv')
    ws=w[(w.aggregation=='single_seed')&w.scope.isin(['2023','2024','pooled'])].groupby(['scope','model','top_fraction']).agg(events_mean=('detected_events','mean'),events_min=('detected_events','min'),events_max=('detected_events','max'),independent_events=('independent_events','first'),unknown_mean=('unknown_outcome_alerts','mean'),known_non_target_mean=('known_no_target_alerts','mean'),repeated_mean=('repeated_alert_records','mean'),lead3_mean=('announcement_detected_3m','mean'),lead6_mean=('announcement_detected_6m','mean')).reset_index();ws.to_csv(dest/'warning_summary.csv',index=False)
    text='''# 制造业可解释基线与精度—复杂度前沿

本轮在此前已查看2023/2024结果后，经用户明确选择继续同一制造业口径，属于后暴露探索性比较。EBM-GAM和GA²M式EBM属于同一实现家族的不同交互配置，不能算作两个独立方法的重复支持。所有模型评价记录相同；森林与端点分类器训练损失和随访利用不同，见冻结协议。

![前沿](figures/accuracy_complexity_frontier.png)

横轴是365日目标输出的完整原生函数表空间格（对数轴），包含缺失/未见状态、未合并的冗余格；不是心理可读性或统计可靠性的统一量表。森林在CIF尺度分解，EBM在log-odds尺度分解后经logistic链接得到风险。固定2000轮的真实EBM出现严重过拟合与风险饱和，因而本图不能支持优于充分调优EBM的结论；训练诊断见EBM_BASELINE_VALIDITY_ZH.md。虚线连接只是展示配置内的描述性非支配点，不证明已找到理论全局前沿。q2/q3/q4、黑箱森林及M2预测复用，不重训。不限阶数同后端森林和M2只画预测参考线，不能把75个局部SHAP值当作75个全局原生函数。

## 记录等权，三个种子均值与范围

'''+table(m[m.weighting=='record_equal'],dict(scope='年份',model='模型',auc_mean='AUC均值',auc_min='AUC最小',auc_max='AUC最大',ap_mean='AP均值',ap_min='AP最小',ap_max='AP最大',ipcw_brier_mean='Brier均值',prediction_mean_mean='风险均值'))+'\n\n## 公司等权\n\n'+table(m[m.weighting=='company_equal'],dict(scope='年份',model='模型',auc_mean='AUC均值',ap_mean='AP均值',ipcw_brier_mean='Brier均值'))+'''

合并AP重新计算，不平均年度AP；种子范围不是置信区间。三模型预测平均的辅助集成结果见metrics.csv的seed_ensemble行，不能与种子指标均值混用。未知结局不作正常；完整校准十等分见calibration.csv。M2是原全行业训练参考，不是制造业重训。

## Top20/25季度核查

'''+table(ws,dict(scope='年份',model='模型',top_fraction='比例',events_mean='捕获均值',events_min='最少',events_max='最多',independent_events='独立事件分母',unknown_mean='未知结局均值',known_non_target_mean='已知非目标均值',repeated_mean='重复预警均值',lead3_mean='提前3月均值',lead6_mean='提前6月均值'))+'''

季度容量相同，年度独立目标事件分母44/60，跨年窗口去重合并79。小数来自三种子平均。全季度核查记录数、去重公司、重复预警及机会分母保存在warning_metrics.csv。

## 原生完整解释规模

'''+table(c,dict(model='模型',seed='种子',native_scale='原生尺度',link='链接',max_order='最高阶',spatial_cells='完整空间格',active_function_cells='非零函数所占格',active_functions='非零函数数',main_functions='主效应数',pair_functions='二元数',reconstruction_error='重构误差'))+'''

EBM没有提供完整时间/原因曲线，不能将它的端点表格与森林的所有时间×原因标量格相比较。本表只按目标365日输出。EBM原生项经过链接精确重构概率，不将log-odds贡献冒称风险百分点。保存/重载检查全部通过。固定训练轮数和分箱预算不是穷尽调优的EBM最优性能。
'''
    (dest/'PREDICTION_FRONTIER_ZH.md').write_text(text)
    text='''# 模拟中的结构恢复

![模拟](figures/simulation_structure_recovery.png)

七种已知DGP、五次共同数据重复、初轮每种五个模型，再补当前方法固定覆盖版本，共210拟合。8个二值输入允许在完整256点均匀网格上精确求全部阶数Walsh系数。初轮q2/q3为全部8候选的阶数原型；q2_coverage才对应当前每树约三分之一变量的覆盖版本（3/8）。补齐时间顺序及旧结果保留见SIMULATION_COVERAGE_AMENDMENT_ZH.md，不将它包装成初始预注册设计。真非零判据1e-10，估计结构选择阈值预先固定0.005概率单位，未根据结果调整。F1评价概率尺度结构；对数几率低阶经非线性链接会产生概率高阶，不能误解为GA²M实现错误。native_coefficient_rmse在各模型原生尺度分别计算，仅可在同尺度内解释。

'''+table(sim,dict(scenario='场景',model='模型',f1_mean='结构F1均值',f1_min='最小',f1_max='最大',pair_precision_mean='二阶precision',pair_recall_mean='二阶recall',probability_coefficient_rmse_mean='概率系数RMSE',native_coefficient_rmse_mean='原生系数RMSE',uniform_cube_probability_mse_mean='均匀网格风险MSE',observed_distribution_probability_mse_mean='观察分布风险MSE',true_attribution_rmse_mean='真概率归因RMSE'))+'''

二阶真项/选择项分母为0时recall/precision未定义，表中留空；原任务诊断里的零分母占位0已在评价阶段从计数恢复为缺失，不改拟合或预测。总体F1的真结构包括主效应，分母始终有定义。

相关场景保留均匀乘积参考和实际观察分布两个误差，不能将稀少组合上的恢复误差当成同等频率现实错误。竞争事件依赖额外变量，目标CIF真结构不依赖它；森林可能为联合事件建模使用该变量。五次Monte Carlo不足以声称稳定的显著差异。该二值实验不能回答连续财务切点定位精度。

真概率尺度Shapley来自完整Walsh系数的精确均匀乘积分配，用来衡量真信号恢复，不把SHAP单变量归因当作交互结构发现器。结构支持由完整函数展开单独评价。未选有利种子，所有场景和失败计数完整保留。
'''
    (dest/'SIMULATION_STRUCTURE_RECOVERY_ZH.md').write_text(text)
    fs=f.groupby(['model','permutations']).agg(attribution_rmse_mean=('attribution_rmse','mean'),attribution_rmse_min=('attribution_rmse','min'),attribution_rmse_max=('attribution_rmse','max'),attribution_mae_mean=('attribution_mae','mean'),attribution_max_error=('attribution_max_error','max'),additivity_max_error=('additivity_max_error','max'),seconds_mean=('shap_seconds','mean')).reset_index();fs.to_csv(dest/'shap_fidelity_summary.csv',index=False)
    ss=st.groupby('model').agg({k:['median','min','max'] for k in ['rmse','normalized_rmse','importance_spearman','top10_jaccard','sign_agreement']});ss.columns=['_'.join(k) for k in ss.columns];ss=ss.reset_index();ss.to_csv(dest/'training_stability_summary.csv',index=False)
    bs=bg.groupby('model').agg(rmse_median=('rmse','median'),rmse_min=('rmse','min'),rmse_max=('rmse','max'),top10_jaccard_median=('top10_jaccard','median'),importance_spearman_median=('importance_spearman','median')).reset_index()
    ns=native.groupby('model').agg(jaccard_median=('jaccard','median'),pair_jaccard_median=('pair_jaccard','median'),shape_correlation_median=('median_correlation','median')).reset_index()
    text='''# 黑箱SHAP的忠实性与稳定性实测

![SHAP](figures/shap_fidelity_stability.png)

固定同一365日目标CIF、背景和解释样本，q2及黑箱不限阶数森林均计算官方Permutation SHAP。有限背景精确参考经全子集穷举和官方ExactExplainer测试，不用近似的高预算SHAP冒充真值。此黑箱是同后端制造业森林，不是M2；未声称测过原M2模型的SHAP。

## 局部加和与归因估计误差分开

'''+table(fs,dict(model='模型',permutations='双向排列循环',attribution_rmse_mean='归因RMSE均值',attribution_rmse_min='最小',attribution_rmse_max='最大',attribution_mae_mean='归因MAE均值',attribution_max_error='最大单项误差',additivity_max_error='最大加和误差',seconds_mean='SHAP秒均值'))+'''

12个按公司抽取的固定评价锚点、16个训练背景公司各一条记录、3个解释器种子。背景不按结局选择。Permutation SHAP本来就满足局部加和；加和精确不能证明Shapley近似收敛，也不是本方法独有优势。q2函数在原始坐标下至多二阶，双向排列的低阶性质使其可能在一次循环已精确；这验证计算性质，不证明q2的真实预测或统计解释更正确。逐种子近似随机性见shap_computation_stability.csv。

## 同一8次训练公司组bootstrap

'''+table(ss,dict(model='模型',rmse_median='归因差RMSE中位',rmse_min='最小',rmse_max='最大',normalized_rmse_median='L1归一后RMSE中位',importance_spearman_median='重要性秩相关中位',top10_jaccard_median='Top10重合中位',sign_agreement_median='符号一致率中位'))+'''

这部分用精确CIF Shapley（共同16行经验联合背景、32个公司锚点），去掉解释器Monte Carlo噪声后比较训练样本扰动。固定模型随机种子42，三个模型每次读取同一组训练公司抽样；不同重复的样本多重性影响内部树bootstrap，这是训练不确定性的组成部分。28个两两组合有依赖，不能冒充28次独立重抽样或据此算过窄置信区间。原始RMSE受模型风险振幅影响，因此同时列每家公司归因L1归一后的差异。

## 固定模型，仅改变背景

'''+table(bs,dict(model='模型',rmse_median='背景变化RMSE中位',rmse_min='最小',rmse_max='最大',top10_jaccard_median='Top10重合中位',importance_spearman_median='重要性秩相关中位'))+'''

每次背景取自同一对应训练bootstrap，模型固定为原seed42。它与重训敏感性是两个实验，不相加，也不能把背景敏感性误报为训练不稳定。

## 原生函数的重抽样稳定性

'''+table(ns,dict(model='模型',jaccard_median='全部函数集合Jaccard中位',pair_jaccard_median='二元集合Jaccard中位',shape_correlation_median='共同非恒定函数相关中位'))+'''

q2原生函数在CIF尺度，EBM在log-odds尺度；不能逐项相减作忠实性排名。共同函数相关只在32个固定锚点测量，不能代表全域函数形状。EBM默认保留全部主效应会机械提高全部集合Jaccard，因此必须同时看二元集合与预测效果。q2的原生交互表与SHAP单变量分摊是不同对象；本实验没有把二者强行一一对应。

完整函数重构、SHAP局部加和、精确Shapley近似误差、真结构恢复以及抽样稳定性是五个不同问题。任何单项好都不自动证明统计可辨识或因果正确。
'''
    (dest/'SHAP_FIDELITY_STABILITY_ZH.md').write_text(text)
    text='''# 执行、数据与失败核查

9个真实EBM基线、24个配对公司bootstrap重训、210个模拟拟合及18个固定模型SHAP任务，共261正式任务全部完成；另3个试运行。模型拟合失败0，结果发布失败0。完整测试55项通过，含2项独立R检查。原森林模型代码未修改。

'''+table(pd.DataFrame(r['publication']),dict(stage='阶段',job='Slurm作业',tasks='任务数',seconds='墙钟秒',by_host='节点完成数'))+'''

正式训练提交：`'''+r['training_code']+'''`。环境及依赖锁定见requirements.lock、environment.json；模型文件哈希见execution_receipt.json。每任务单核、6GiB地址空间上限、NFS；任务时长/RSS/cpu_seconds保存在task_diagnostics.csv，不将CPU秒之和当墙钟，也不声称GPU参与。Python进程不会自动跨机并行；尾部独立森林任务不能自动扩展占用其他任务释放的核。

训练样本56,458条、2,577个公司组、177个独立目标事件、31个独立竞争事件。EBM训练正评价权重39,362条；其余17,096条结局未确定，不改成正常。2023评价11,686条/44个独立目标事件，2024为12,113条/60个，合并23,799条/79个。行业历史版本、财务数据可得日与删失假设限制沿用原制造业审计，仍为PROVISIONAL。训练标签信息截至2022-01-01之前，经实际字段核对。

准备时lemon默认python3指向缺少ensurepip的3.14，创建环境失败，改用其已有3.13解释器建立新的隔离环境后成功；安装日志保留，没有更改旧环境。EBM官方警告默认可视化不显示缺失值：本轮保留缺失分箱/参数并计入复杂度，没有把缺失填成任意极值。模型汇总过程曾在正式作业尚未结束时因缺少完成回执提前退出；保留日志，待正式完成后才进行评价，未重训或覆盖训练结果。

数据文件、样本身份及预测哈希全部复核；旧5类参考120行指标与上一轮一致，8次共同公司抽样哈希逐一核对。完整企业模型、参数、阈值和逐公司预测/解释留服务器，仅汇总及研究图提交Git。
'''
    (dest/'EXECUTION_AND_DATA_AUDIT_ZH.md').write_text(text)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--dest',type=Path,default=Path(__file__).parent);a=p.parse_args();main(a.run,a.dest)
