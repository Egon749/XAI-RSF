"""Aggregate-only report; no author-PROBoost or complete54 efficacy claim."""
from pathlib import Path
import argparse,datetime,json,shutil,re
import pandas as pd
from experiments.post_st_drop9_v1.deliver import table
from .common import *

SUBSET_ZH='''# 本轮“论文变量子集”具体指什么

论文表2列出54项指标。本轮只使用能从现有冻结输入中找到对应列、并核对既有计算口径和日期的20列。选择依据是数据可用性与口径可核对程度，没有根据本轮测试结果挑选。

| 类别 | 本轮输入（括号内为论文表2编号） | 列数 |
|---|---|---:|
| 盈利 | 总资产收益率ROA（X1）；净利润/总资产（X2）；财务费用/营业收入（X6） | 3 |
| 偿债 | 流动比率（X8）；速动比率（X9）；现金比率（X10）；利息保障倍数（X11）；经营现金流/流动负债（X12）；资产负债率（X13） | 6 |
| 增长 | 总资产增长率（X20）；收入增长率（X21） | 2 |
| 周转 | 应收账款周转率（X24）；存货周转率（X25）；总资产周转率（X30） | 3 |
| 资产结构 | 现金/总资产（X33）；营运资本/总资产（X35） | 2 |
| 现金流 | 经营现金流/净利润（X39）；经营现金流/营业收入（X40）；经营现金流/总资产（X41） | 3 |
| 市场 | 总市值（X53） | 1 |

20列有三个重要限制：

1. **列名可对应，公式未必完全一致。** 论文没有给出全部逐项公式。本轮保留既有明确口径，例如周转率以期末余额为分母，利息保障倍数使用既有代理公式。完整差异逐项保存在paper_variable_mapping.csv。
2. **20列不等于20种独立信息。** 冻结数据中的X1与X2逐值相同；本轮保留这两列以显式对应论文两个名称，并未声称独立识别了两种效应。市值从原自然对数值反变换，市场日期不变。
3. **另外34项尚未完成适配。** 例如ROE、ROIC、每股指标及估值比率，需要补齐定义、构造和组成数据的历史可得日核验。这不等于原始库一定没有这些数据，也不能用报告期充当发布日期。现有20列仍继承旧conservative输入的原始披露版本限制。

因此，本轮检验的是“这20列适配输入用于当前可解释森林的效果”，尚不能据此评价论文完整54变量方案。我们的目标仍是ST后先最终退市/先恢复；论文的ST分类目标与其不同。
'''


def deliver(run,repo):
    pub=repo/'experiments'/NAME;reports=run/'reports'
    v=json_load(run/'audit/validation.json');assert v['status']=='PASS_PROVISIONAL_ADAPTED_SUBSET'
    gate=json_load(run/'audit/formal_gate.json');freeze=json_load(run/'audit/freeze.json');protocol=json_load(run/'input/protocol.json')
    ens=pd.read_csv(reports/'ensemble_comparison.csv');cx=pd.read_csv(reports/'function_complexity.csv');dif=pd.read_csv(reports/'paired_differences.csv')
    tasks=pd.read_csv(reports/'execution_tasks.csv');data=pd.read_csv(reports/'cohort_audit.csv');mapping=pd.read_csv(reports/'paper_variable_mapping.csv')
    tests=(run/'audit/full_tests_pre_fit.log').read_text();assert 'failed' not in tests.lower();passed=int(re.search(r'(\d+) passed',tests).group(1))
    (pub/'VARIABLE_SUBSET_ZH.md').write_text(SUBSET_ZH)
    statements=[]
    for year,g in ens.groupby('year'):
        a=g.set_index('arm').loc['reference83'];b=g.set_index('arm').loc['drop9'];p=g.set_index('arm').loc['paper20']
        statements.append(f"{year}年：论文可映射20列AUC={p.auc:.6f}、AP={p.ap:.6f}、Brier={p.brier:.6f}，Top20核查{int(p.checked)}家，捕获{int(p.captured)}/{int(p.target_events)}。"
            f"相对83列，AUC变化{p.auc-a.auc:+.6f}、AP变化{p.ap-a.ap:+.6f}、Brier变化{p.brier-a.brier:+.6f}；"
            f"相对74列，AUC变化{p.auc-b.auc:+.6f}、AP变化{p.ap-b.ap:+.6f}、Brier变化{p.brier-b.brier:+.6f}。"
            f"平均风险{p.risk_mean:.2%}，观察先退市率{p.ipcw_observed_incidence:.2%}。")
    result=f'''# 论文可映射20变量：12个月实验结果

本轮新增9个森林全部成功（三年×三个种子），派生9个直接父节点AJ收缩模型；原83列及74列的18个参考森林直接复用，没有重训。三个种子先平均逐公司CIF再评价。

**这是论文表2中可映射20列的适配实验，尚不是完整54变量实验，也不是PROBoost算法复现。** 用户明确选择先运行可核验子集并记录缺项及定义差异。原研究模型继续保留。

## 变量范围

来源：Liu等，Progressive gradient boosted trees for imbalanced financial distress prediction，Expert Systems With Applications321(2026)132187，DOI10.1016/j.eswa.2026.132187，PDF第9页表2。原论文54项指标分8类，用于ST分类；本轮沿用ST后先最终退市/先恢复竞争风险任务，不能将两者AUC直接比较。

{table(mapping[mapping.selected][['paper_id','paper_name','existing_name','operational_definition']])}

这些字段按数据/定义可对应程度确定，没有按测试成绩筛选。论文没有提供逐项公式，故匹配是显式既有口径映射，不是作者原代码逐值复现。X1与X2在冻结输入中逐值相同，20列不能视为20种独立信息；保留两列并记录重复，未临时去冗余。周转率使用期末余额且不年化、收入优先营业收入、净利润优先归母、利息保障倍数为已有代理公式、X41采用经营现金流/总资产，均有对应说明。X53从原自然对数总市值反变换为CSMAR千元单位，日期仍为原可得市场日期。

{table(mapping[~mapping.selected][['paper_id','paper_name','status']])}

上述34项未进入本轮冻结输入，原因是尚未完成新字段构造、确切定义及历史版本/各组成数据可得日核验；不等于所有原始科目都不存在。盈利能力FI_T5等预计算表的Accper是报表期，不是公告日，不能直接将其视为PIT合格数据。未制造可得日、空白占位54列或未经说明的替代变量。

## 统一预测比较

{table(ens[['year','arm','companies','known_target_events','auc','ap','brier','risk_mean','ipcw_observed_incidence','checked','captured','coverage']])}

reference83=原83变量；drop9=固定删9后的74变量；paper20=论文可映射20列。每年1月1日前可得历史信息训练，评价当年新入组企业各自12个日历月内先最终退市。Brier越低越好，其余排序/覆盖指标越高越好。Top20按当年公司数的20%向上取整，分别核查8、9、11家公司，保留已知竞争事件与未知删失区别。三个年度12个月终点均可评价，IPCW权重为1。

{chr(10).join(statements)}

综合判断：这20列适配方案没有表现出稳定的预测收益。2023年相对83列的AP和Top20有所改善，与74列的AP和捕获数基本相同；2024年AUC、AP和Top20捕获下降，2025年这三项均低于两个参考方案。2024年Brier仅略有降低，不能据此忽略排序和事件捕获的损失。20列模型的平均风险仍低于各年观察发生率，特别是2024年仍为3.54%对35.56%，风险低估没有解决。

完整表示的函数数和空间格数减少，是可观察到的表示收益；它伴随部分年度预测损失。本文不以这些已查看队列上的结果作确认性方法选择，也不将全部下降归因于某个被删字段：变量内容、每树覆盖以及市场变量尺度共同变化，尚未被单独分离。

这种结果只评价当前适配输入集。不能将paper20的表现外推为完整论文54变量的优劣，也不能把年度AP简单平均当作合并AP。当前研究参考不因某个指标变化自动替换。

## 计算预算与解释

三套方案都使用q2、深度4、32树/种子、共享分位切点、mtry75、min_leaf10/min_groups5/min_events3、原因权重(1,1)、直接raw父节点AJ强度10及原24个月训练随访上限。由于只有20列，paper20每树使用全部20列；原83/74方案每树使用25列，因此输入内容、覆盖与市值变换共同改变，不能声称实际候选搜索量完全匹配。保留所有阈值候选与实际评分次数。

{table(cx[['year','arm','main_functions','interaction_functions','spatial_table_cells','scalar_table_cells','distinct_thresholds']])}

新9个raw与9个收缩版本，以及3个96树平均预测器，在全部训练/评价公司、所有原生时点及两原因上验证完整q2重构、保存重载和CIF合法性；最大误差{v['max_explanation_error']:.3e}，容差1e-10。旧参考模型哈希不变、逐公司预测复现，并重新算出48行原12个月指标一致。

空间格数包含缺失状态，不乘时间/原因维；标量格数另包含这些维度。数学上精确表示不等于真实作用恢复，X1/X2重复仍带来解释归属歧义；此实验不声称改善可辨识性。

## 配对差值

{table(dif[['year','reference_arm','metric','difference','lower95','upper95','undefined_bootstraps']])}

差值为paper20减对应参考。每年1000次公司配对bootstrap，三模型使用同一抽样矩阵；无事件导致指标无法计算的次数完整记录。区间仅反映固定预测条件下测试样本波动，不包含训练或变量方案选择不确定性。所有单种子、恢复原因、校准分箱、风险水平与不利结果完整保留。
'''
    (pub/'RESULTS_ZH.md').write_text(result)
    (pub/'DATA_ALIGNMENT_AUDIT_ZH.md').write_text(f'''# 输入、时间与来源核验

{table(data)}

冻结旧版conservative C83输入逐值与原443家入组向量核对。20列除X53市值反变换外保持原值和缺失状态，其余time/event/group/sample_id等数组逐值不变。金融48项真实可得日、增长变量既有双期可得日数组、市场日期均早于各自公告后下一交易日开盘前；继承原始披露版本PIT局限，不构成新版本认证。训练截止严格早于当年1月1日，24个月训练随访上限保持。测试标签位于独立evaluation_input，正式训练仅读取测试X及标识。

三个年度训练/测试公司互斥，测试为36/45/53家；训练公司数为309/345/390，训练先退市14/26/49例。训练集嵌套，不能相加当作独立企业数。每年96棵树公司bootstrap与重复次数逐树配对，共288对。保留列的共享切点不变，X53因变换重新由训练集构造切点。

可映射字段、20列输入质量和全部34项缺项见paper_variable_mapping.csv及results/input_quality.csv。完整54字段构造仍未完成，用户确认接受此子集范围；标签和目标没有改成论文的ST分类。

原83/74共18个模型及其预测/函数哈希保持，CURRENT_POST_ST_MODEL.json不变。私有运行：`{run}`。论文原文、企业记录、模型参数、切点、逐公司预测和解释未提交Git；公开仅代码、变量映射、配置、环境和汇总证据。
''')
    (pub/'TRAINING_DIAGNOSTICS_ZH.md').write_text(f'''# 执行与验证

冻结提交`{freeze['commit']}`，UTC{freeze['utc']}。拟合前{passed}项测试通过，含独立R检查。没有修改核心森林算法。

Slurm作业{gate['job']}，9项计算及回传均通过，实际分布{json.dumps(gate['by_host'])}，池墙钟{gate['seconds']:.3f}秒。每项单线程、4GiB地址空间上限；没有额外拟合或失败种子丢弃。

{table(tasks[['year','seed','host','fit_seconds','shrink_seconds','seconds','peak_rss_MiB','actual_scored_candidates']])}

{table(pd.read_csv(reports/'leaf_support_summary.csv'))}

收缩使用各自raw直接父节点AJ，alpha=n/(n+10)，n为独立in-bag公司数，各原因/时间同权。叶真实支持、父节点AJ和候选评分重放通过，收缩不增加真实事件数。

{table(pd.read_csv(reports/'function_validation.csv')[['year','seed','version','verified_rows','verified_times','full_reconstruction','explicit_component_sum','function_reload','forest_reload']])}

概率非负、单调、原因和不超过1，以及保存重载均通过；无裁剪或删函数掩盖误差。新预测AUC/AP/Brier由独立实现复核，参考指标48行重算一致。
''')
    (pub/'METHODS_RESULTS_ZH.md').write_text('''# 方法与结果段落

依据指定论文表2的54项指标，我们对照原有conservative C输入，构建20列可映射变量适配方案，并明确记录34项未覆盖指标及已有定义与论文未明确口径的差异。保持原ST后先最终退市与先恢复竞争风险任务，以2023、2024、2025年1月1日前可得信息训练，评价对应年度新入组企业各自12个月结局。新增九个q2深度4森林，采用32树/种子及强度10的raw直接父节点AJ收缩，种子42、101、202，公司bootstrap逐树配对。由于输入只有20列，每树候选上限为20；原83及74变量参考每树25列。三种子逐公司预测平均后计算指标。该实验衡量可获得变量方案与相应覆盖变化的整体效果，不代表完整54变量或PROBoost复现。

'''+ '\n\n'.join(statements)+f'\n\n完整函数最大重构误差为{v["max_explanation_error"]:.3e}。每年度1000次固定预测条件下配对公司bootstrap保留全部可计算与不可计算次数；未变更研究参考模型。\n')
    dest=pub/'results';dest.mkdir(exist_ok=True)
    for p in reports.glob('*.csv'):shutil.copy2(p,dest/p.name)
    aud=pub/'audit';aud.mkdir(exist_ok=True)
    for name in ['freeze.json','source_hashes.json','data_alignment.json','formal_gate.json','validation.json','full_tests_pre_fit.log']:shutil.copy2(run/'audit'/name,aud/name)
    (aud/'input_hashes.json').write_text(json.dumps({p:h for p,h in json_load(run/'audit/immutable_input_hashes.json').items() if not p.startswith('code/')},indent=2)+'\n')
    hashes=[]
    for i in range(9):
        p=run/'results/formal'/str(gate['job'])/f'task-{i}/output'
        for name,h in json_load(p/'hashes.json').items():assert sha(p/name)==h;hashes.append(dict(year=YEARS[i//3],seed=SEEDS[i%3],artifact=name,sha256=h))
    (aud/'new_artifact_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
    assert sha(repo/'CURRENT_POST_ST_MODEL.json')==REFERENCE_SHA
    receipt=dict(status='COMPLETE_PROVISIONAL_AVAILABLE_SUBSET',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),private_run=str(run),
        source_commit=protocol['source_commit'],fitting_commit=freeze['commit'],new_forest_fits=9,derived_models=9,reused_forests=18,failed_fits=0,pilot_fits=0,
        paper_variables_listed=54,paper_columns_used=20,unmapped_paper_variables=34,duplicate_pair='X1/X2',full54_experiment_completed=False,PROBoost_reproduced=False,
        slurm_job=gate['job'],actual_execution_by_host=gate['by_host'],current_reference_replaced=False,current_reference_sha256=REFERENCE_SHA,
        calibration_fits=0,additional_search_trials=0,training_cohorts=data.to_dict('records'),validation=v,tests_passed=passed,independent_R=True,
        aggregate_hashes={str(p.relative_to(pub)):sha(p) for p in dest.glob('*.csv')})
    write(pub/'execution_receipt.json',receipt);shutil.copy2(pub/'execution_receipt.json',run/'audit/execution_receipt.json')
    print(ens[['year','arm','auc','ap','brier','captured','target_events']].to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();deliver(a.run,Path(__file__).resolve().parents[2])
