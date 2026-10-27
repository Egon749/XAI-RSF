"""Publish aggregate evidence only; fitted values/thresholds/identities stay private."""
from pathlib import Path
import argparse,json,shutil
import numpy as np
import pandas as pd


def table(d):
    def fmt(x):
        if isinstance(x,(float,np.floating)):return f'{x:.6g}'
        return str(x)
    return '\n'.join(['| '+' | '.join(map(str,d.columns))+' |','| '+' | '.join(['---']*len(d.columns))+' |']+['| '+' | '.join(fmt(x) for x in r)+' |' for r in d.itertuples(index=False,name=None)])
def put(p,s):
    with p.open('x') as f:f.write(s.rstrip()+'\n')
def main(run,repo):
    run,repo=Path(run),Path(repo);o=run/'reports';dest=repo/'experiments/search_coverage_v1'
    m=pd.read_csv(o/'metrics_summary.csv');d=pd.read_csv(o/'training_diagnostics.csv');w=pd.read_csv(o/'warning_summary.csv');c=pd.read_csv(o/'paired_contrasts.csv');g=pd.read_csv(o/'acceptance_gates.csv');s=pd.read_csv(o/'seed_stability.csv')
    plan=json.loads((run/'input/plan.json').read_text());rec=json.loads((o/'execution_receipt.json').read_text())
    validation='全部30个正式模型、5个试运行完成。全部模型概率约束、路径检查与保存重载通过；二阶模型对全部验证记录、全部原生时间和两个原因作精确函数核验。无近似解释器、无裁剪、无隐藏高阶残差。'
    proto='仅V19/V20开发。每配置3种子指标均值，不是种子融合；范围不是置信区间。M2复用冻结预测，输入表示和预算与当前后端不同。同后端unrestricted是主要匹配参照。数据PIT、质量、原点及随访证据仍PROVISIONAL，未使用2022—2024结局。'
    public_m=m[m.weighting=='record_equal'][['scope','model','auc_mean','auc_min','auc_max','ap_mean','brier_mean','prediction_mean']]
    public_w=w[w.top_fraction==.2][['scope','model','events_mean','independent_events','lead3_mean','lead6_mean']]
    summary=d[d.variant!='unrestricted'].groupby('variant',as_index=False).agg(main_mean=('main_functions','mean'),pair_mean=('interaction_functions','mean'),cells_mean=('spatial_table_cells','mean'),cells_max=('spatial_table_cells','max'),scalar_cells_max=('scalar_table_cells','max'),reconstruction_max=('reconstruction_error','max'),root_max_fraction_mean=('root_max_fraction','mean'),root_features_mean=('root_distinct_features','mean'))
    summary.to_csv(o/'complexity_summary.csv',index=False)
    put(o/'V19_V20_RESULTS_ZH.md','# V19/V20有限匹配消融结果\n\n'+proto+'\n\n'+table(public_m)+'\n\n公司等权结果：\n\n'+table(m[m.weighting=='company_equal'][['scope','model','auc_mean','ap_mean','brier_mean']]))
    put(o/'PREDICTION_COMPARISON_ZH.md','# 预测、事件覆盖与预设门槛\n\n'+proto+'\n\n季度Top20独立事件捕获及公告前3/6月识别：\n\n'+table(public_w)+'\n\n完整Top25、每季度核查记录、年度去重公司、已知非目标/未知、重复预警见warning_metrics.csv。独立事件跨年度去重，年度计数不能简单相加。\n\n配对差异（AUC/AP正值较好，Brier负值较好）：\n\n'+table(c[c.weighting=='record_equal'].groupby(['contrast','scope'],as_index=False)[['auc_difference','ap_difference','brier_difference']].mean())+'\n\n预设门槛：AUC差距<=0.02，AP保留>=95%，Brier比<=1.05，Top20平均少捕获<=3，分别对同后端不限阶数对照；非统计非劣效检验。\n\n'+table(g))
    put(o/'INTERPRETABILITY_COMPARISON_ZH.md','# 完整解释及搜索机制\n\n'+validation+'\n\n'+table(summary)+'\n\n空间格含缺失状态；完整时间/原因标量表另列。全部主效应和交互保留，不能只展示TopK并声称完整。所有贡献在CIF概率尺度，依赖训练边际乘积参考，不是风险率贡献或因果效应。\n\n覆盖开关限制树输入子集，每3树对全部75输入做一次分组覆盖，故单个输入32树中出现10或11次；不保证每个变量对覆盖，也不保证变量被实际选择。根节点集中度及实际结构变量对数见training_diagnostics.csv；公司明细、真实阈值及函数值未公开。\n\n细化字典最多27个分位位置，每节点每变量仍最多5个有限候选。它既改变切点位置，也改变深层节点候选可用性，不是单纯增加搜索次数。未使用森林硬预算；1000格仅是本轮预设可读性评价门槛。\n\n跨种子条件稳定性：\n\n'+table(s)+'\n\n共同函数相关性只覆盖共同选择项，集合稳定性另报。三个种子不证明重抽样稳定性、统计可辨识性或未来年份可靠性。')
    put(o/'TRAINING_DIAGNOSTICS_ZH.md','# 实际执行诊断\n\n'+validation+'\n\n正式训练代码：`'+plan['code_commit']+'`。240项4树训练片段与30项无拟合汇总，独立执行且按全局tree_id合并；树分片语义有测试，两个旧对照每折每种子须与历史树/预测完全一致。\n\n'+table(d[['fold','seed','variant','fit_seconds','fit_cpu_seconds','seconds','peak_rss_MiB','candidate_evaluations','root_distinct_features','root_max_fraction','structural_distinct_pairs','leaf_zero_target_company_support','leaf_zero_competing_company_support','reconstruction_error','reload_error']])+'\n\nfit_seconds为8个树片段耗时之和，不是模型墙钟关键路径；seconds为汇总与核验阶段墙钟，peak_rss_MiB为该阶段峰值，训练片段耗时见shard_resources.csv，未在片段内部记录逐进程最终RSS峰值；外部采样的内存高水位只作为观测下界，见resource_observations.json。节点/候选事件支持不能把重复季度当成独立事件。试运行、训练和汇总各阶段的全部任务、节点分工及发布哈希见execution_receipt.json。')
    old=(Path(plan['original_run'])/'reports/DATA_ALIGNMENT_AUDIT_ZH.md')
    if not old.exists():old=repo/'experiments/shared_partition_v1/DATA_ALIGNMENT_AUDIT_ZH.md'
    put(o/'DATA_ALIGNMENT_AUDIT_ZH.md','# 本轮数据继承与核验\n\n本轮逐文件核验上一轮'+str(len(plan['source_hashes']))+'个冻结数据/评价文件SHA256；未重建样本、未制造可得日期。训练折记录合计112,513，去重62,049，独立训练目标事件182；验证25,689记录、135独立目标事件。重复种子不增加样本数。75输入字段顺序继承input_fields.json。全部旧数据限制继续有效：\n\n'+old.read_text())
    # Copy only explicitly public aggregate artifacts; no recursive private copy.
    for p in o.iterdir():
        if p.suffix in ['.csv','.json','.md'] and '_PRIVATE' not in p.name:shutil.copy2(p,dest/p.name)
    for name in ['environment_oliver.txt','environment_lemon.txt','frozen_protocol.json','pilot_gate.json']:
        shutil.copy2(run/'audit'/name,dest/name)
    shutil.copy2(repo/'experiments/shared_partition_v1/input_fields.json',dest/'input_fields.json')
    print(table(public_m));print(table(summary))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--repo',required=True);a=p.parse_args();main(a.run,a.repo)
