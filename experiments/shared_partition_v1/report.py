"""Export only aggregate diagnostics and generate comparison tables."""
from pathlib import Path
import argparse,json,shutil
import numpy as np
import pandas as pd

def table(df):
    def fmt(v):
        if isinstance(v,(float,np.floating)):return '' if pd.isna(v) else f'{v:.6g}'
        return str(v)
    return '| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(v) for v in r)+' |' for r in df.itertuples(index=False,name=None))+'\n'

def main(run,repo):
    run,repo=Path(run),Path(repo);src=run/'reports';dest=repo/'experiments/shared_partition_v1'
    files=['metrics.csv','metrics_summary.csv','warning_metrics.csv','warning_summary.csv','calibration.csv','training_diagnostics.csv','budget_diagnostics.csv','mechanism_audit.csv','paired_contrasts.csv','acceptance_gates.csv','seed_stability.csv','execution_receipt.json','publication_verification.json','legacy_control_parity.json','legacy_metric_parity.json']
    for name in files:shutil.copy2(src/name,dest/name)
    for name in ['tests_full.txt','tests_delivery.txt','environment_oliver.txt','environment_lemon.txt','pilot_gate.json','training_publication_verification.json']:shutil.copy2(run/'audit'/name,dest/name)
    if (run/'audit/recovery_plan.json').exists():shutil.copy2(run/'audit/recovery_plan.json',dest/'recovery_plan.json')
    shutil.copy2(repo/'experiments/development_v1/input_fields.json',dest/'input_fields.json')
    def save(name,body):
        with (dest/name).open('x') as f:f.write(body)
    plan=json.loads((run/'input/plan.json').read_text())
    counts=pd.DataFrame([dict(split=k,**v) for k,v in plan['source_counts'].items()])
    save('DATA_ALIGNMENT_AUDIT_ZH.md','# 数据对齐审计\n\n**PROVISIONAL。** 本轮逐文件哈希验证上一轮冻结输入，未重建样本，未读取2022—2024结局。\n\n'+table(counts)+'''
训练两折合计112,513条、去重62,049条及182个独立目标事件；验证25,689条、135个独立目标事件。种子重复不增加独立样本数。
V19训练原点2012—2017、拟合截止2019-01-01；V20训练原点2012—2018、拟合截止2020-01-01。

历史披露版本、派生组成数据时钟、质量默认值、日状态随访和停牌/终止语义仍未完整认证。没有制造availability。训练沿用原删失；源结局不明的评价记录统一赋零结局权重，保留预警排名，单列未知数量。IPCW不能补证这些时钟。原审计见../development_v1/DATA_ALIGNMENT_AUDIT_ZH.md。

字段顺序见input_fields.json；哈希见frozen_protocol.json。M2为冻结预测，147列编码与本轮75列原生缺失表示不同。主要结构对照为同后端unrestricted。共享切点只读取训练值，12日历月按记录使用365/366日。

初次准备误复制生成的字节码缓存，因其不在冻结清单而中止，0次拟合启动。修复仅排除缓存，使用新唯一目录，原失败目录保留。
''')
    m=pd.read_csv(src/'metrics_summary.csv');d=pd.read_csv(src/'training_diagnostics.csv')
    save('V19_V20_RESULTS_ZH.md','# 有限匹配预测结果\n\n**PROVISIONAL；3个预先固定种子。** 各模型为指标均值，M2只有一套冻结预测。AP范围是种子范围，不是置信区间。\n\n记录等权：\n\n'+table(m[m.weighting=='record_equal'])+'\n公司等权：\n\n'+table(m[m.weighting=='company_equal'])+'''
合并AP每个种子基于合并预测重算，再平均指标；没有平均两折AP或融合三个种子的预测。AUC/AP采用同一IPCW及源结局门禁。公司等权按相应评价范围公司记录数的倒数加权。Brier分母为全体基准权重，未知结局权重0。预测均值覆盖所有记录。metrics.csv保留观察发生率；calibration.csv为十分组校准。
''')
    save('PREDICTION_COMPARISON_ZH.md','# 预测保留与预警\n\n固定每季度Top20%为主，Top25%为辅，同一sample_id并列规则。事件数为三种子均值/范围，小数不是部分捕获一个事件。\n\n'+table(pd.read_csv(src/'warning_summary.csv'))+'''
warning_metrics.csv含每季度核查量、年度去重公司、已知非目标、未知、重复预警、公告前3/6月识别及机会分母。合并事件跨年去重；未知不当正常/误报。

预先冻结门槛：每个范围AP≥不限阶数对照均值95%，Brier≤105%，Top20平均少捕获≤3个，空间表≤1000格。这是开发容忍度，不是统计非劣效证明。

'''+table(pd.read_csv(src/'acceptance_gates.csv'))+'\n所有配对开关差见paired_contrasts.csv；未挑选有利种子。\n')
    cols=['fold','seed','variant','leaves','distinct_thresholds','candidate_evaluations','fit_seconds','fit_cpu_seconds','peak_rss_MiB','reload_error','reconstruction_error','leaf_zero_competing_company_support']
    save('TRAINING_DIAGNOSTICS_ZH.md','# 训练与数值诊断\n\n训练前完整37项测试通过（含独立R参考），日志修复后38项通过。5项4树试运行后，配置不变完成30项32树拟合。\n\n'+table(d[cols])+'''
156项训练任务＝12个预算模型完整拟合＋144个4树片段；18项汇总完成其余18个森林。任务数不是模型数。publication_verification.json记录计算/回传核验、两节点任务数、整批实测时间。physical模式使用oliver 24物理核工作槽及lemon 8槽，未启用P核第二SMT线程。当前实现为CPU算法。

原轮部分模型在拟合后的严格JSON日志保存阶段失败：缺失单独分裂的-inf哨兵不能直接写入。原模型当时尚未保存，故只能按完全相同配置重做受影响拟合。恢复只把诊断哨兵编码为字符串，训练模块逐文件哈希相同；不是算法或预算修订。全部原失败和恢复映射见recovery_plan.json/execution_receipt.json，没有以指标选择重试或删除失败记录。

fit_seconds：预算模型是训练墙钟时间，分片模型是各片段墙钟时间之和，不是端到端时间。fit_cpu_seconds是对应CPU时间之和。peak_rss_MiB在预算模型覆盖拟合，在分片模型仅覆盖汇总/重构阶段；原片段训练峰值未单独记录。每项地址空间上限6GiB，pilot内存门禁通过。

所有验证记录及原生时间点通过概率和完整重构检查；保存重载一致。没有裁剪预测或预算后删函数。贪心森林无连续优化器，Hessian秩/gap/触边不适用。零竞争事件公司支持叶数单列，稀少原因估计限制仍存在。

预算拒绝诊断（树编号从0开始）：

'''+table(pd.read_csv(src/'budget_diagnostics.csv'))+'''
私有budget_log_PRIVATE.json保存每节点预算前后、拒绝数量、最好被拒分数/拟新增格数及实际选择分数；真实切点不上传。没有保存每个未选候选的完整得分。固定树序、左后右遍历，早期树有预算优先权。
''')
    cols=['fold','seed','variant','main_functions','interaction_functions','spatial_table_cells','scalar_table_cells','factored_leaf_coefficients','low_support_cell_fraction','zero_support_cell_fraction','reconstruction_error']
    save('INTERPRETABILITY_COMPARISON_ZH.md','# 完整解释复杂度与可靠性\n\n路径最多2个原始输入坐标，完整CIF函数直接重构森林。包含全部函数，不是TopK。\n\n'+table(d[d.variant!='unrestricted'][cols])+'''
空间格含缺失状态；标量表再乘原生时间和两个原因。无损因子存储不等于较少逻辑分区；不限阶数对照不导出伪完整二阶解释。预算保守结构计数不同于非零函数计数，没有借容差删零降格数。

跨种子稳定性在每折512个固定训练参考点计算。集合Jaccard与共同函数365日目标CIF贡献的相关/非零符号一致性中位数分别报告。共同函数高相关不能掩盖集合变化。3种子仅描述随机化稳定性，不代表样本重抽或时间迁移置信区间。

'''+table(pd.read_csv(src/'seed_stability.csv'))+'''
低支持格指训练记录少于10，不是独立企业/事件支持，格数比例不是公司比例。减少空格不自动证明可靠。乘积参考规范化约束代数表示，不保证相关数据上的统计可辨识。完整匿名案例保留cases_PRIVATE.json；贡献在CIF概率尺度、依赖参考分布，不是对数风险率或因果效应。

预算独立作用：

'''+table(pd.read_csv(src/'mechanism_audit.csv'))+'''
树相同且预测差0时，不能宣称预算有额外改善。共享切点改变候选位置和节点可用数量，可能降低尾部阈值分辨率；效果不等于单纯合并重复切点。本轮未引入可辨识性惩罚。
''')
    print('Aggregate reports written; reviewed decision document required.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--repo',required=True)
    a=p.parse_args();main(a.run,a.repo)

