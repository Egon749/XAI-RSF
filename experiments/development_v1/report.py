"""Publish aggregate evidence only; never copy fitted coefficients or company rows."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
import pandas as pd


def table(df):
    def val(v):
        if isinstance(v,(float,np.floating)):return '' if pd.isna(v) else f'{v:.6g}'
        return str(v)
    return '| '+' | '.join(map(str,df.columns))+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(val(v) for v in row)+' |' for row in df.itertuples(index=False,name=None))+'\n'
def main(run,repo):
    run,repo=Path(run),Path(repo);src=run/'reports';dest=repo/'experiments/development_v1'
    m=pd.read_csv(src/'metrics.csv');w=pd.read_csv(src/'warning_metrics.csv');d=pd.read_csv(src/'training_diagnostics.csv');s=pd.read_csv(src/'simulation_metrics.csv');sd=pd.read_csv(src/'simulation_diagnostics.csv');st=pd.read_csv(src/'function_stability.csv')
    receipt=json.loads((src/'execution_receipt.json').read_text());plan=json.loads((run/'input/real_config.json').read_text())
    for name in ['metrics.csv','calibration.csv','warning_metrics.csv','training_diagnostics.csv','function_stability.csv','component_agreement.csv','simulation_metrics.csv','simulation_diagnostics.csv','execution_receipt.json']:shutil.copy2(src/name,dest/name)
    for name in ['environment_oliver.txt','environment_lemon.txt','tests_final.txt','tests_parallel.txt','tests_delivery.txt']:shutil.copy2(run/'audit'/name,dest/name)
    def save(name,text):
        with (dest/name).open('x') as f:f.write(text)
    counts=pd.DataFrame([dict(split=k,**v) for k,v in plan['source_counts'].items()])
    counts['unknown_endpoint_records']=[int((~pd.read_parquet(run/'input'/k/'metadata_PRIVATE.parquet').label_eligible & (pd.read_parquet(run/'input'/k/'metadata_PRIVATE.parquet').event!=2)).sum()) for k in counts['split']]
    save('DATA_ALIGNMENT_AUDIT_ZH.md','# 企业数据对齐审计\n\n**PROVISIONAL**。复用冻结首次ST样本，未重建样本或读取2022—2024结局。\n\n'+table(counts)+'\n训练折合计112,513条，去重62,049条，独立目标事件182；验证25,689条，去重目标事件135。折内146/182及88/98不能直接相加当独立事件总数。\n\n核验了sample_id、group、origin、stop_date、event、训练信息截止、75字段名称/顺序、内容bundle与quality bundle哈希、M2预测ID对齐。实际365/366日对应12日历月。所有输入文件SHA256清单在服务器audit/input_hashes.json。\n\n历史原始披露版本、质量默认值时钟、日状态随访和停牌/终止语义仍未认证。训练保留历史删失截止，不能声称这些删失机制已被验证；评价对源结局不明记录统一赋零结局权重，仍保留其预警排名。公司组有跨年重叠，本研究是时间外推而非新公司验证。\n\n初次准备误将已经被上游置NaN的内容来源日期作为已观察值检查，发生240格来源日期断言。核对发现已观察内容违规为0，按原导出契约只对已观察内容核验可得日，保留缺失和来源日期并统计。未修改任何日期或数据值；保留失败目录，改用新的唯一运行目录。\n\nM2输入为147列，本轮75列含原生缺失；表示、预算与后端不同，公平结构比较以同后端unrestricted为主，M2为外部冻结参照。详见METHOD_AND_PROTOCOL_ZH.md。\n')
    dc=['fold','variant','trees','leaves','max_path_variables','max_depth','distinct_thresholds','candidate_evaluations','fit_seconds','peak_rss_MiB','reload_error','reconstruction_error','leaf_zero_competing_company_support']
    save('TRAINING_DIAGNOSTICS_ZH.md','# 训练与数值诊断\n\n训练前19项完整测试通过，包括独立R核对；树级并行交付版23项通过，且与训练快照及串行模型逐字节核验一致。2项真实试运行、90次合成拟合、8项正式企业拟合均完成并核对回传。\n\n'+table(d[dc])+'\n真实各模型按冻结32棵树执行。没有概率裁剪、验证集选配置或剔除不利种子。森林为有限候选贪心建树，无连续系数求解器，因此Hessian秩、优化gap与系数触边不适用，不能虚报为0或收敛。节点保留所选分裂得分和原因别支持；记录候选评估总数，未保存每个被淘汰候选的完整得分表。\n\n作业2508首次尝试8项均已启动，Slurm因节点启动后的batch状态缺失自动重排，0项完成。已保存原始任务日志及控制器原因；相同代码、种子、预算恢复，非按表现选择重试。首次数据准备断言也单独留档。训练参数和阈值只在私有模型文件中。\n')
    real=m.query("weighting=='record_equal'")[['scope','model','auc','ap','ipcw_brier','prediction_mean','observed_ipcw_incidence']]
    save('V19_V20_RESULTS_ZH.md','# V19/V20有限开发结果\n\n**PROVISIONAL；一个真实种子，无调参，非盲测成功声明。**\n\n记录等权：\n\n'+table(real)+'\n公司等权：\n\n'+table(m.query("weighting=='company_equal'")[['scope','model','auc','ap','ipcw_brier','prediction_mean']])+'\n合并AP基于合并预测重新计算。Brier使用同一源结局门禁和IPCW；未知结局没有被编码为普通负例。校准十分组见calibration.csv；模型之间可比较，原始数据缺失机制不能由IPCW补证。\n')
    warning=w[w.scope.isin(['V19','V20','pooled'])][['scope','model','top_fraction','alert_records','alert_company_groups','detected_events','independent_events','event_coverage','known_no_target_alerts','unknown_outcome_alerts','repeated_alert_records','announcement_detected_3m','announcement_detected_6m']]
    save('PREDICTION_COMPARISON_ZH.md','# 预测与预警比较\n\nM2为既有冻结预测，未重训。先看q2对同后端unrestricted的结构代价，再看与M2的外部差距。固定每季度ceil(20%/25%)核查预算和sample_id并列规则。\n\n'+table(warning)+'\n每季度及年度去重统计见warning_metrics.csv。pooled公司去重跨两年，不等于年度数量之和；事件跨年去重。未知结果单列，不能当作误报。3/6月捕获以公告时间为准，并同时保存机会分母。\n\n以下合成目标原因、t=1结果是每场景三个训练重复的均值，不用于配置筛选：\n\n'+table(s.query('horizon==1 and cause==1').groupby(['scenario','variant'],as_index=False)[['ap','ipcw_brier','cif_mse']].mean())+'\n合成A_rfsrc不是企业M2。该库与自定义森林只匹配主要预算和bootstrap样本，候选细节不保证相同。已知真CIF误差检验了精确概率目标，而不仅是排序指标。\n')
    cols=['fold','variant','main_functions','interaction_functions','spatial_table_cells','scalar_table_cells','factored_leaf_coefficients','low_support_cell_fraction','zero_support_cell_fraction','reconstruction_error']
    save('INTERPRETABILITY_COMPARISON_ZH.md','# 内生解释与可审计性\n\n低阶模型完整函数展开直接等于森林CIF，独立函数求值不遍历森林。以下包含全部函数，不是TopK：\n\n'+table(d[d.variant!='unrestricted'][cols])+'\n空间格数含缺失状态且不合并阈值；标量表规模再乘原生时间节点与两个原因。保存采用无损矩形因子形式，压缩存储不等于较少的逻辑分区。unrestricted可能高于二阶，拒绝导出伪完整二阶解释。\n\n'+table(st)+'\ncomponent_agreement.csv用共同V19训练参考的512个固定训练点描述两折函数值相关及非零符号一致性，不是完整分区的单调方向定理，也不是重复训练稳定性置信区间。跨年份样本不同，不能把差异全归因于模型不稳定。\n\n低支持/零支持格比例来自训练记录数，不是独立公司事件信息矩阵。因相关变量导致的无支持组合仍存在。函数规范化、精确重构、可读复杂度、统计可靠性是四个不同性质。全部真实匿名案例含所有主效应/交互及最终CIF，留在服务器cases_PRIVATE.json。贡献为概率尺度、依赖参考分布，非因果。\n')
    a=m.query("scope=='pooled' and weighting=='record_equal'").set_index('model')
    cap=w[(w.scope=='pooled')&(w.top_fraction==.2)].set_index('model')
    apdiff=float(a.loc['q2','ap']-a.loc['unrestricted','ap']);bdiff=float(a.loc['q2','ipcw_brier']-a.loc['unrestricted','ipcw_brier'])
    decision='PREDICTIVE_NO_GO'  # reviewed judgment for this frozen configuration, not a significance test
    save('METHOD_INNOVATION_DECISION_ZH.md',f'''# 方法判断

研究判断：**{decision}（仅限本轮冻结的q2实现与预算，PROVISIONAL）**。这是审阅后的研究判断，不是预注册统计显著性检验。整体框架的工程可行性已验证；更广泛的方法创新及惩罚贡献仍不能确定。

1. 能否训练预测：能，V19/V20四版本共8项完成；只在PROVISIONAL数据口径下成立。
2. 与M2差距：合并q2 AP={a.loc['q2','ap']:.6f}，M2={a.loc['M2','ap']:.6f}；Brier分别{a.loc['q2','ipcw_brier']:.6f}/{a.loc['M2','ipcw_brier']:.6f}；Top20独立事件分别{int(cap.loc['q2','detected_events'])}/135与{int(cap.loc['M2','detected_events'])}/135。两者预算及表示不同。
3. 相对同后端不限阶数：q2的合并AP差={apdiff:+.6f}，Brier差={bdiff:+.6f}（Brier正差为变差）；Top20捕获{int(cap.loc['q2','detected_events'])}对{int(cap.loc['unrestricted','detected_events'])}个。两折分别报告，不只看合并。
4. 可辨识性惩罚是否独立有效：未测试。本轮所有版本无该惩罚，不能把结构限制的效果归功于惩罚。
5. 原因别风险信息是否优于Gram：未测试。没有给森林移植不适用的风险率Hessian。
6. 是否获得内生解释：获得精确完整的低阶CIF函数表达；但函数数、分区数和无支持组合决定实际可读性，不能仅凭重构误差断言解释简单、稳定或统计可辨识。
7. 是否值得作为论文方向：本轮提供了可复核的RSF方法原型，但q2在两折均损失AP、Brier及Top20事件覆盖，且完整函数仍较大。不建议直接扩大当前配置；这不等于证明所有低阶RSF方案都不可行，也尚不足以确认独立方法创新。已知低阶树集成与函数ANOVA先例必须承认；需要的独立贡献不能由“几种已有技术组合”或单次开发结果替代。

解释框架的工程可行性、预测竞争力、惩罚独立贡献分别判断。这一负面判断基于相同输入和后端下的匹配对照，不用数据PROVISIONAL标签掩盖预测损失；单个真实种子、有限森林预算、未认证PIT/随访限制了外推范围。本轮到此停止，没有自动扩大树数、搜索配置或另换模型。
''')
    print(json.dumps(dict(decision=decision,pooled_ap_delta_q2_vs_unrestricted=apdiff,pooled_brier_delta=bdiff)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--repo',required=True);a=p.parse_args();main(a.run,a.repo)
