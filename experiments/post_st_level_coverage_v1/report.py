"""Publish aggregate evidence and Chinese reports, keeping all models private."""
from pathlib import Path
import argparse,datetime,json,shutil,subprocess
import numpy as np
import pandas as pd
from .common import *
from experiments.post_st_structure_recovery_v1.plots import LABELS

NAMES={'reference':'当前25列','affine':'仅仿射校准','wide83':'每树83列','wide83_raw':'83列原始AJ'}
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(map(str,headers))+' |','|'+'---|'*len(headers)]+['| '+' | '.join(map(str,row))+' |' for row in rows])+'\n'
def number(v,d=4):return '—' if pd.isna(v) else f'{v:.{d}f}'

def report(run,repo):
    root=repo/'experiments'/NAME;out=root/'results';out.mkdir(exist_ok=True)
    src=run/'reports'
    for p in src.iterdir():
        if p.is_file() and (p.suffix in ['.csv','.json','.png','.pdf'] or p.name.endswith('.csv.gz')):shutil.copy2(p,out/p.name)
    # This public task ledger omits a fitted-parameter-distance diagnostic.
    ledger=pd.read_csv(out/'task_manifest.csv');ledger.drop(columns=['identity_change_max'],errors='ignore').to_csv(out/'task_manifest.csv',index=False)
    audit_names=['pre_fit_freeze.json','analysis_freeze.json','analysis_pool_freeze.json','analysis_summation_note.json',
        'pilot_gate.json','formal_gate.json','analysis_gate.json','environment.json','environment_lemon.json','storage_preflight.json','supplement_failure_01.json','report_failure_02.json','verification_failure_03.json','reuse.json','prefit_tests.log','analysis_tests.log','delivery_tests.log']
    for n in audit_names:shutil.copy2(run/'audit'/n,out/n)
    for stage in ['pilot','formal','analysis']:shutil.copy2(run/f'{stage}.jsonl',out/f'{stage}_manifest.jsonl')
    hashes=[]
    for name,h in json.loads((run/'audit/immutable_input_hashes.json').read_text()).items():
        if not name.startswith('code/'):hashes.append(dict(relative_input=name,sha256=h))
    pd.DataFrame(hashes).to_csv(out/'frozen_input_hashes.csv',index=False)
    metrics=pd.read_csv(src/'enterprise_metrics.csv',dtype={'seed':str,'scope':str})
    warnings=pd.read_csv(src/'enterprise_warnings.csv',dtype={'seed':str,'scope':str})
    def met_table(h,cause='termination'):
        rows=[]
        for scope in ['2023','2024','pooled']:
            for arm in ['reference','affine','wide83']:
                r=metrics[(metrics.scope==scope)&(metrics.arm==arm)&(metrics.seed=='ensemble')&(metrics.horizon_months==h)&(metrics.cause==cause)].iloc[0]
                rr=[scope,NAMES[arm],number(r.auc),number(r.ap),number(r.brier),f'{r.risk_mean*100:.2f}%',f'{r.ipcw_observed_incidence*100:.2f}%']
                if cause=='termination':
                    w=warnings[(warnings.scope==scope)&(warnings.arm==arm)&(warnings.seed=='ensemble')&(warnings.horizon_months==h)&(warnings.top_fraction==.2)].iloc[0]
                    rr+=[f'{int(w.captured)}/{int(w.target_events)}',int(w.checked)]
                rows.append(rr)
        return table(['年度','模型','AUC','AP','Brier↓','平均风险','观察发生率']+(['Top20捕获','核查数'] if cause=='termination' else []),rows)
    main=pd.read_csv(src/'simulation_main.csv')
    simrows=[]
    for scenario in LABELS:
        a=main[(main.scenario==scenario)&(main.arm=='reference')].iloc[0];b=main[(main.scenario==scenario)&(main.arm=='wide83')].iloc[0]
        simrows.append([LABELS[scenario],number(100*a.pair_amplitude_ratio,3)+'%' if pd.notna(a.pair_amplitude_ratio) else '不适用',
            number(100*b.pair_amplitude_ratio,3)+'%' if pd.notna(b.pair_amplitude_ratio) else '不适用',
            f'{int(round(50*a.pair_detected))}/50 → {int(round(50*b.pair_detected))}/50' if pd.notna(a.pair_detected) else '不适用',
            number(a.joint_risk_mse,6)+' → '+number(b.joint_risk_mse,6)])
    simtable=table(['场景 · 训练N','25列幅度恢复','83列幅度恢复','真实交互检出','真实CIF MSE↓'],simrows)
    geom=pd.read_csv(src/'enterprise_export_geometry.csv',dtype={'seed':str})
    grow=[]
    for y in YEARS:
        for a in ['reference','wide83']:
            r=geom[(geom.year==y)&(geom.arm==a)&(geom.seed=='ensemble')].iloc[0]
            grow.append([y,NAMES[a],int(r.nonzero_main_functions),int(r.nonzero_pair_functions),int(r.spatial_cells),int(r.scalar_function_cells),f'{r.function_bytes/1024:.1f}'])
    complexity=table(['年度','模型','非零主效应','非零交互','空间单元','函数标量数','导出KiB'],grow)
    support=pd.read_csv(src/'calibration_support.csv');srows=[]
    for r in support.itertuples():srows.append([r.outer_year,r.horizon_months,r.entry_years,r.companies,r.termination,r.recovery,'固定1' if r.termination<10 else '允许估计'])
    supporttable=table(['预测年','期限/月','历史入组年','公司','退市','恢复','退市斜率'],srows)
    validation=json.loads((src/'independent_validation.json').read_text());gate=json.loads((run/'audit/formal_gate.json').read_text())
    (root/'RESULTS_ZH.md').write_text(f'''# 概率水平与候选覆盖：两组分开的实验

**24个月历史仿射校准缓解了风险低估，且保留逐年度排序与精确解释，但没有消除低估。扩大候选资格能改善部分大样本交互恢复，却损害了企业2024年的排序。两组不组合，当前参考模型继续保留。**

本轮是已暴露2023/2024及既有模拟数据上的有限机制分析，不是独立确认性验证。企业输入仍有继承的PROVISIONAL/PIT限制。没有新增2025访问。

## 1. 实际执行

两台机器完成460/460正式任务：1350次模拟森林拟合、6次企业森林拟合、4次仿射校准优化；另外2次数值小试森林拟合。正式任务oliver343项、lemon117项，队列含等待约317.81秒。10项并行分析另用约27.89秒，均成功。没有重训历史对照，也没有更换当前模型、追加配置或挑选成功种子。

完整测试180项通过，包含独立R检查。企业最大重构误差{validation['maximum_enterprise_reconstruction_error']:.3e}，模拟最大误差{validation['maximum_simulation_reconstruction_error']:.3e}，均远低于1e-10。全部保存/重载、概率和配对检查通过。详细范围、存储计数和两类已修复的汇总脚本问题见[数值报告](NUMERICAL_VALIDATION_ZH.md)。

## 2. 主要企业结果：24个日历月先最终退市

{met_table(24)}

逐公司先平均三个种子再计算指标。2023为36家、12例退市；2024为45家、20例。Top20名额按年向上取整，为8家和9家；合并捕获按年度名额合计。pooled AUC、AP、Brier均重新计算。全部终点已知，本次IPCW权重实际均为1；这不表示训练样本没有删失。

校准把2023平均概率6.47%提高到12.01%，2024从11.58%提高到27.25%；对应观察发生率33.33%和44.44%，分别仍低估21.32和17.19个百分点。合并Brier从0.31575降到0.25531，配对差值−0.06044，描述性95%区间[−0.08753,−0.03290]。

各年校准斜率为正，所以年内AUC、AP和Top20不变。**不同年份使用不同映射，pooled排序可改变**：24个月合并AUC从0.72959降至0.69196，AP从0.68172降至0.66263。不能将“逐年排序不变”扩大为“跨年排序不变”。

扩大候选资格使2023 AP略升、Top20多捕获1例，但2024 AUC下降0.094，AP下降0.128，Top20少捕获1例。2024 AUC配对差值区间[−0.17990,−0.01213]；AP区间[−0.26343,0.00780]。合并AUC下降0.06569、AP下降0.09615，Brier改善0.01066。**概率水平、排序和事件捕获不是同一指标。**

这些区间来自固定预测下1000次按年度分层公司配对bootstrap；主要24个月退市指标均1000次可算。它们不包含重新训练、模型选择及此前多轮探索的不确定性，不作为选择性的确认性显著结论。

![平均概率](results/enterprise_probability_levels.png)

## 3. 12个月与恢复结果也全部保留

{met_table(12)}

2024/12个月校准窗口只有1例退市、9例恢复，两个斜率均固定1；历史目标风险本就不低于该窗口发生率，非负截距约束下得到恒等映射，未缓解2024测试队列的低估。2023/12个月提高了平均退市概率，但pooled AUC从0.79918降到0.63525，AP从0.63407降到0.32224。这是年际映射错位的实际不利结果，必须保留。

恢复原因24个月结果：

{met_table(24,'recovery')}

2023/12个月没有观察到先恢复事件，AUC和AP不定义，CSV保留NaN而非填0。恢复的12个月、各个种子、raw版本、Top25及分箱结果均在[完整指标](results/enterprise_metrics.csv)、[核查结果](results/enterprise_warnings.csv)、[分组校准](results/enterprise_calibration_bins.csv)。

## 4. 扩大候选范围是否恢复重要交互

{simtable}

幅度100%表示沿已知真实交互函数方向完全恢复，不是AUC、概率或事件捕获率。检出门槛为函数RMS≥0.005；50份数据重复均先合并3个种子，原始AJ及其他期限/原因完整保存。纯加性场景没有真实交互，召回和幅度不定义。

大样本普通二阶场景的幅度恢复从4.883%升至22.188%，配对增幅17.305个恢复比例百分点，95%均值区间[12.875,21.734]；仍距100%很远。该场景真实作用为±5个百分点，平均学到约±1.109个百分点，原来约±0.244个百分点。函数L2误差从0.047558降至0.038906。

真实变量对同时有资格的树由平均8.76/96变为96/96，形成非零交互的树由5.68/96增至26.10/96。按树加权的分解是9.125%×64.840%×82.532%=4.883%，以及100%×27.188%×81.610%=22.188%。扩大资格缓解了覆盖瓶颈，但多数树仍未形成该交互；不能解释成已经解决搜索和有限样本问题。

纯交互场景反而由0.244%降至0.084%；稀少事件仍约0.05%，连续场景仅0.558%。全部场景无关函数检出数的均值均上升；纯加性场景由0.54升至1.38。稀少事件场景真实CIF MSE反而增加约0.000023。更多候选同时增加竞争和噪声选择，不能把扩大范围称为普遍改进。

![幅度恢复](results/interaction_amplitude_recovery.png)

## 5. 完整解释是否保住

{complexity}

以上为96树完整预测器，缺失状态单独计格，空间单元不含时间/原因乘数；标量数包含全部原生时间和两个原因，但不含基线。校准保持变量、交互、阈值和空间单元不变，只变基线与原因内统一幅度。实际导出增加原点时间，校准标量数见[导出几何](results/enterprise_export_geometry.csv)，不能照搬原生时间数。

扩大资格后两年空间单元分别减少54.80%、65.22%，交互函数144→58和146→47。但因子化文件反而略增，说明“空间表更小”不等于“压缩文件更小”。2023/2024交互集合Jaccard均值由0.072/0.055升至0.284/0.396；2023主效应集合重合度下降，预测随机波动也未两年一致改善。集合更稳定不证明函数更真实，更不能抵消2024预测退化。

## 6. 本轮研究判断

1. 受约束仿射更新与完整q2函数表示可兼容，且本轮24个月数据中改善了概率水平与Brier；这是明确的工程/数值证据。尚未证明个体概率已经准确、跨期映射稳定或未来制度下仍有效。
2. 每树25列造成的交互资格不足是幅度衰减的一个实际来源。保留q2扩大资格可部分恢复含主效应的大样本交互，并不需要先放开交互阶数。
3. 资格扩大不是充分解决方案：纯交互、连续和稀少事件恢复仍很差，企业2024排序变差。它没有证明整体优于当前森林。
4. 校准改的是风险水平及统一幅度；覆盖实验改的是森林学到的结构。前者不能修复遗漏交互，后者不能自动解决概率漂移。两组未组合，不能声称联合方案成功。

**保留当前参考版本。本轮不自动推广wide83，也不把校准版替换为新的默认模型。** 证据用于说明设计取舍与方法边界；没有启动下一轮搜索。完整方法可见[论文文本](PAPER_TEXT_ZH.md)，执行与工件见[运行说明](RUN_ZH.md)。
''',encoding='utf-8')
    (root/'DATA_ALIGNMENT_AUDIT_ZH.md').write_text(f'''# 数据、时间与历史工件核验

本轮输入没有重建：直接复制当前模型保存的旧保守C、83列训练和评价快照，逐文件及逐数组核对哈希。[冻结输入哈希](results/frozen_input_hashes.csv)、[数组哈希](results/data_array_hashes.csv)、[历史模型复用](results/reuse.json)提供来源索引。模型参数、公司记录、原始切点、校准参数、逐公司预测与函数始终在私有运行目录。

| 测试年 | 训练公司 | 训练退市 | 训练恢复 | 训练24月未知 | 测试公司 | 测试24月退市 | 测试24月恢复 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2023 | 309 | 14 | 184 | 49 | 36 | 12 | 15 |
| 2024 | 345 | 26 | 198 | 44 | 45 | 20 | 16 |

两折训练记录合计654，去重345家公司、26个退市事件；六次企业新拟合的重复计算记录为1962，不是新增1962家公司。企业测试合计81家、32例24个月退市、31例先恢复、18例窗口内无首次事件。两期限测试端点均完整；训练未知仍按删失处理。模型起点为入组ST公告后下一交易日开盘前；先最终退市X与先恢复A竞争，均按公告日，先恢复再退市不改记为首次退市。2023拟合信息截止严格早于2023-01-01，2024严格早于2024-01-01。

旧C保留了年报版本保守处理，不等于通过原始PDF/PIT认证；继承原研究的PROVISIONAL字段时钟限制。没有将研究者接受年度版本后的新输入混入本轮，也没有重新使用2025结局。2025此前已研究接触，不将其重称严格盲测。

## 校准只用预测时点之前的样本外预测

{supporttable}

先依据入组日期加评价期限选择最近两个完整成熟年度，再读取结局；没有根据“已发生事件/结局已知”筛选个体。被选窗口恰好全部有完整终点，未知数为0。每个窗口样本ID、时间与截止当年的训练元数据逐行对齐，全部随访停止早于该外层截止。若端点不完整，冻结方案要求停止而非当正常处理。

复用2019—2022各3种子共12个历史滚动样本外模型，树配置与当前模型逐参数相同、父节点收缩10；原模型以更早年份训练。12份模型配置与哈希检查见[滚动配置核验](results/rolling_config_validation.csv)。它们来自后来撤回的时间加权实验中的**单位权重baseline工件**；复用这些历史预测不表示重新采用其时间加权策略或恢复被取消实验。

校准器没有读取2023/2024测试标签。六次wide83拟合仅接收训练标签及测试X/sample_id；测试结局独立存放，到分析阶段才读取。双方相同缺失语义、字段顺序和公司bootstrap重复数；共享切点、函数参考分布均由相应训练集构造。

## 模拟复用

沿用此前9场景×50份已暴露模拟数据及1350个q2对照森林，本轮只新拟合1350个wide83森林。独立重新生成全部450份数据，训练与测试输入哈希均一致；公司bootstrap乘数逐树一致。50份数据重复用于模拟变异评价，3个种子首先合并；不能把3种子当独立数据队列。另有2份专用小试，不纳入正式效应汇总。

私有运行目录：`{run}`。原始资料和旧运行均保留。
''',encoding='utf-8')
    (root/'CALIBRATION_ZH.md').write_text(f'''# 受约束仿射更新的结果与边界

映射为 F′ₖ(t,x)=cₖg_H(t)+bₖFₖ(t,x)。c≥0，0.5≤b≤4，斜率不随时间变化；某原因历史事件不足10例时固定其b=1。g_H采用外层训练集所有首次事件的AJ，除以365/730日累计值，在该日及以后保持1、原点为0；只声明12/24个月端点校准，中间时间形状是模型假定。

拟合联合两原因平方概率损失，并以20/n乘平均预测变化平方向恒等映射正则化。20个等效公司的强度和斜率范围均在结果前冻结，不宣称最优。四个校准器使用三个种子的平均预测，不另拟合12个单种子校准器。

## 联合概率约束

对每棵树终端叶CIF取二维凸包，再求平均Minkowski和。这个多边形覆盖任意输入可能达到的终端森林CIF；忽略树间路径兼容性使界限保守。加入所有历史样本外预测点，对每个顶点v约束 c₁+c₂+b₁v₁+b₂v₂≤1。正斜率、非负单调偏移使所有时间非负、单调，终端界保证所有原因总和≤1。未对输出做裁剪。

采用凸四变量QP、SLSQP和活跃约束KKT复核，并用独立线性规划计算最优性gap。四次gap绝对值最大约9.1e-16，原始约束违反为0。2024/24个月总概率边界激活；其余三个没有激活。实际参数留私有目录，公开[历史诊断](results/calibration_historical_diagnostics.csv)、[独立全域证书](results/calibration_domain_validation.csv)。没有偷偷放松约束换取更高概率。

## 结果解读

{supporttable}

2023/24个月退市斜率固定1，主要上移基线；恢复斜率降低。2024/24个月退市斜率上升且基线提高，恢复斜率下降，总概率约束起作用。目标概率提升并非无约束放大所有原因。2024/12个月两个原因均固定斜率，历史发生率不足以支持非负上移，得到恒等映射。

历史校准样本的24个月退市发生率分别只有7.78%与13.13%，测试队列却是33.33%与44.44%。使用预测前信息能缓解风险水平偏差，无法据此保证完全追上后来队列的发生率。校准后的样本平均值也不要求精确匹配历史比例，因为正则与联合可行域同时约束。

24个月两个年度Brier均改善，但逐年校准改变跨年可比性；12个月合并排序明显下降，所有不利结果保留。没有独立证据证明每家企业的真实概率已准确，不能把整体均值更接近解释成逐企业校准通过。

## 完整解释为何仍成立

原模型 Fₖ=μₖ+Σⱼfⱼₖ+Σⱼ<ₗfⱼₗₖ，则新基线 μ′ₖ=cₖg_H+bₖμₖ，所有主效应和二元交互分别乘相同bₖ。因此变量、切点、函数阶数与空间分区均不变，解释可精确重构新CIF。正斜率保留每个原因内函数升降方向；原来为零的交互仍为零，校准不能补回遗漏结构。

导出器实际保存变换后的基线、因子系数及时间网格，加载该函数文件即可预测，不依赖事后SHAP。检查覆盖两年所有训练/测试公司、全部原生时间以及偏移网格和日历端点、两个原因。更新仅提升数学/数值可审计性，不等于证明作用为因果效应或可统计辨识。
''',encoding='utf-8')
    (root/'COVERAGE_RECOVERY_ZH.md').write_text(f'''# 保留q2扩大每树候选资格

唯一训练配置变化为tree_feature_count25→83，保留depth4、path_budget2、mtry75、32树/种子、共享分位数5切点、原因权重(1,1)、最小叶10/公司5/事件3、直接父节点收缩10。节点尚未达到q上限时实际候选由25增加到75。不是相同计算预算；每树83列意味着均衡资格全部相同，但不变为新的随机覆盖算法或两层前瞻搜索。

{simtable}

大样本、存在可供首层使用的主效应时改善明显；纯交互无相应边际主效应，增加资格依然不能有效触发逐节点贪心搜索。该解释是与结果相符的机制判断，本轮没有新增前瞻搜索对照来独立证明全部原因。相关变量场景的28.07%恢复也仍远小于100%；在乘积边际参考下比较函数，并不代表相关特征实际联合分布中的独立因果作用。

无关函数检出增加、稀少事件MSE变差。两个小样本有真实交互的场景幅度差区间包含0；连续场景虽然均值略升，幅度仍接近0。真实三阶结构在q2下仍不可表示；该场景表格报告的是其中的二阶项，不是三阶恢复。

![预测与复杂度](results/simulation_prediction_complexity.png)

## 企业解释复杂度与支持

{complexity}

全部企业模型实际最大路径变量数为2、最大深度4。主效应/交互数为全部空间格和原生时间上数值非零的函数数，而非仅“路径出现过”。本轮结构数与非零数恰相同，均在[导出表](results/enterprise_export_geometry.csv)单独保存。低支持和零支持比例仍未消失：2024 wide83零训练支持空间格比例约30.31%，当前版25.13%；有NaN专门格与未观察组合，不能把这些格全解释成现实公司关系。

2024零退市叶比例由当前约42.49%升至wide83约54.85%，不是降低了零事件叶。收缩会借入父节点正风险，但不创造新的观察事件或公司。2024借入正退市风险的零事件叶由186个降为45个；解释空间更集中并不自动增加事件支持。逐种子数值见[叶支持](results/enterprise_leaf_support.csv)。

2023/2024交互集合重合度从0.072/0.055升至0.284/0.396，共同函数相关性的中位数均值从0.663/0.643升至0.747/0.772；2023主效应重合度反而下降。训练联合样本用于比较函数形状，中心化仍用同年度训练乘积边际；固定730天的稳定性展示不应误称每家不同日历终点。三种子属于算法随机性。完整数据见[函数稳定性](results/enterprise_function_stability.csv)、[预测波动](results/enterprise_prediction_variation.csv)。

## 计算代价

wide83两个年度候选路由尝试合计分别157291和157756；原25列为53613和53541，约2.93—2.95倍。真正通过叶支持筛选并打分的候选数分别148480和149607，原25列为49367和49699；有限切点数、缺失路由与小于5个有效切点的情况分别保存在[搜索成本](results/enterprise_search_cost.csv)。候选尝试不等于实际打分次数。

6个新企业森林单次拟合约6.41—12.81秒，实际峰值RSS约274—284MiB，收缩约0.20—0.65秒。不同机器P/E核、并发与历史负载不同，不用历史墙钟时间做公平算法速度结论；没有为了计时重训当前模型。模拟逐种子耗时、候选和模型存储见simulation_models.csv.gz及simulation_model_summary.csv。

结论是“覆盖瓶颈得到部分验证，放宽覆盖不是已验证的企业预测改进”。本轮不做结果驱动的新范围/深度/切点搜索。
''',encoding='utf-8')
    (root/'NUMERICAL_VALIDATION_ZH.md').write_text(f'''# 数值、复现与执行验证

- 全部180项测试通过，包含独立R验证；仅1条既有EBM缺失值展示警告。没有放宽测试、删除失败测试或裁剪概率。
- 小试2/2、正式460/460、分析10/10成功，存储发布和输出SHA256均核验。正式新增1356个森林、1356个收缩派生版本、4个校准器；另外小试2个森林及派生版本。参照森林和滚动工件均复用，不计新拟合。
- 不可变输入1801份、原始来源、正式输出哈希、当前模型引用文件均重新核验。完整日志见[独立核验](results/independent_validation.json)、[工件哈希](results/artifact_hashes.csv.gz)。
- 最大企业误差{validation['maximum_enterprise_reconstruction_error']:.3e}；最大模拟误差{validation['maximum_simulation_reconstruction_error']:.3e}。容差1e-10，CIF非负、单调、原因和≤1，模型/函数保存重载一致。
- 企业：6个参考单种子、12个新raw/收缩单种子、6个96树合并预测器、4个校准预测器，共28次函数工件验证；全部训练/评价输入×全部原生时间与补充时间×2原因，分批显式加和基线+所有主效应+所有交互。校准验证包括完整偏移网格。
- 模拟：450份数据×2臂×2叶版本×(3种子+合并)共7200个表示检查，其中对照来自经哈希核实的既有输出。全部训练/测试输入的两个期限，另32固定探针的全部原生时间、4个探针的显式函数和；不伪称全部样本×全部原生时间穷举。
- 校准独立验证直接求每棵树终端叶线性值最大值再平均，与凸包证书一致；不只在已观察公司上测概率界。四个QP最优gap、原始约束违反、稀少事件斜率门槛均通过。模型域证书是保守外包络。
- 原始直接父节点AJ、独立公司计数权重、bootstrap乘数及候选优胜分裂按节点重放，新森林均通过。未重复收缩。
- 参考预测48个指标行逐项复现原报告；AUC/AP/Brier用独立公式复核。模拟中配对期望Brier差与真实CIF MSE差相等，独立恒等式检查通过。

## 实际导出存储口径

原worker的scalar_table_cells按原始森林原生网格计数。校准导出额外包含原点，2023的123个时间点变124，2024的140变141；已补充实际exported_grid_size、scalar_function_cells和baseline_scalars到[enterprise_export_geometry.csv](results/enterprise_export_geometry.csv)。前者保留为原始诊断，后者用于实际校准文件存储解释。空间格数量不变。因子化文件保存系数而非展开所有函数格，压缩字节数不必随理论空间格数同向变化。

## 保留的故障与修正

1. 存储准备时本地新目录创建被NFS目录租约阻塞，通过lemon的授权NFS客户端创建唯一新空目录后继续。仅终止本次阻塞的mkdir进程，未重启服务或触碰旧工件，详见storage_preflight.json。
2. 汇总脚本补充数据数组哈希时，误用了只适用bootstrap整数的哈希函数；字符串ID转换被拒绝，未写出该汇总表。已改为包含dtype、shape和原始字节的哈希，重跑只读补充成功，详见supplement_failure_01.json。没有训练失败或重训。
3. 首次分析结果检查前，针对“3种子先平均”和“96棵树一次累加”的浮点加法顺序差异，把该辅助核对改为1e-13绝对容差；仍比正式重构1e-10严格。不是看到失败结果后放宽门槛，预结果时间记录在analysis_summation_note.json。
4. 交付报告第二次生成时，公共receipt使用了只允许新建文件的协议写入函数而被拒绝。已让公共生成报告支持重新渲染，私有冻结输入和计算工件仍不可覆盖；记录于report_failure_02.json，没有重训或改指标。
5. 最终重复核验在全部数值检查通过后，也因同一只新建写入规则而拒绝现有验证回执。已改为重新计算后严格核对既有回执内容相等，保留原回执；修复后重复核验通过。记录于verification_failure_03.json。上述为两类汇总问题、三次失败调用，均不涉及森林/校准拟合失败。

全部不利结果保留。分析任务使用6GiB地址上限，训练任务按冻结4GiB上限，实际RSS远小于上限；CPU单线程独立任务，不假装单进程跨两机，也没有调用无关GPU或绕过P/E核调度。
''',encoding='utf-8')
    (root/'PAPER_TEXT_ZH.md').write_text('''# 可用于论文的方法与结果文本

## 方法

为区分概率水平偏差与交互幅度衰减，我们对当前二阶可解释竞争风险森林进行了两项独立实验。参考森林保留公司组bootstrap、竞争风险分裂、叶内Aalen–Johansen估计和固定平均，并设置路径最多两个不同变量、深度4、每树25个均衡候选变量、共享分位数切点，以及基于独立公司数的直接父节点AJ收缩。每个预测器由三个固定种子的96棵树组成。第一项在预测截止之前的滚动样本外预测上估计联合两原因仿射映射，保持正的时间常数斜率、非负单调基准偏移，并用终端叶凸包Minkowski和对所有输入建立联合CIF约束。每个原因少于10例事件时固定斜率为1，并预先固定向恒等映射的正则化。映射对基线、主效应和二元交互作代数更新，保持完整预测表示。

第二项只将每树候选资格从25列扩大至全部83列，保留q2及其余训练参数；节点mtry仍为75，因此实际候选评分计算量增加。该对照在2023/2024企业队列及9个已知CIF场景各50份匹配模拟数据上执行，复用既有对照并逐树匹配公司bootstrap。模拟以真实函数方向的幅度恢复、完整L2误差、预设RMS检出阈值和真实CIF均方误差评价学习质量；企业以24日历月先最终退市为主要终点，另报告12个月及恢复。企业各年先合并三种子预测再计算AUC、AP、IPCW Brier、平均概率与按年度名额的Top20捕获。对固定预测实施1000次按年度分层公司配对bootstrap。两项改变不组合，当前模型保持不变。

## 结果

1356次正式森林拟合及4次校准优化全部成功。历史校准使2023与2024的24个月平均退市概率分别由6.47%和11.58%升至12.01%和27.25%，Brier由0.28599和0.33956降至0.25927和0.25214，但平均概率仍低于33.33%和44.44%的观察发生率。各年度AUC、AP和Top20保持不变，而年度特异映射改变了合并排序，合并AUC由0.72959降至0.69196。12个月2024校准得到恒等映射；12个月合并排序亦出现不利变化。

扩大候选资格使普通二阶大样本场景的交互幅度恢复从4.883%提高到22.188%，真实变量对检出由1/50增至34/50，真实CIF MSE从0.003949降至0.002583；然而纯交互、连续与稀少事件场景仍存在严重幅度衰减，无关函数检出增加。企业2024的AUC由0.770降至0.676，AP由0.78203降至0.65406，Top20捕获从8/20降至7/20。解释空间单元2023与2024分别由3146和3263降至1422和1135，交互集合跨种子重合度提高，但并非所有稳定性指标改善。两项实验均保持二阶完整表示，企业最大重构误差为2.67×10⁻¹⁵。

## 解释边界

结果支持历史受约束概率更新与精确低阶解释兼容，并支持每树资格不足是交互幅度衰减的部分来源。结果不支持将更宽候选视为普遍有效的企业预测改进，也不支持把精确重构等同于真实机制恢复。当前模型能够忠实展示其学到的函数，但恢复质量仍取决于样本事件支持、候选竞争和逐节点搜索。上述企业队列和模拟设定已经研究暴露，分析属于探索性机制验证；条件bootstrap区间未包含全部训练、选择和研究迭代不确定性。数据仍继承原披露版本审计限制，没有新增独立外部验证。
''',encoding='utf-8')
    (root/'RUN_ZH.md').write_text(f'''# 复现与工件

服务器本次独立运行：`{run}`。所有旧模型、当前引用和旧结果保持原样。公开results仅汇总、哈希、冻结配置、执行日志及匿名总体图；公司资料、逐公司预测、真实阈值、森林/校准参数和完整企业函数导出保留私有目录。

拟合代码冻结提交：`c82f303f164bd9013be0bc720ceb4494fad6ce01`；首次分析提交：`3cda259`；并行分析快照：`df9d839f6110a4968739c61d4dd6adbd3ec975ed`。后续改动只涉及汇总、真实导出网格计数、图表和报告，没有修改已拟合森林或校准器。

## 新运行的入口

先创建唯一私有目录，不覆盖本次目录。安装环境见results/environment.json；两机均使用已核验Python3.13环境。下列RUN需替换为新的唯一绝对路径：

```bash
python -m experiments.post_st_level_coverage_v1.prepare --stage prepare --run RUN
python -m experiments.post_st_level_coverage_v1.prepare --stage freeze --run RUN
/home/oliver/.local/bin/hetero-run RUN/pilot.py
```

以`experiments.post_st_ablation_v1.stages.gate`核验pilot任务、签名、SHA256与发布成功后，再运行`hetero-run RUN/formal.py`并核验formal gate。gate需要实际job编号和工具返回的结果JSON路径；不可仅凭提交成功继续。拟合快照与协议必须在第一项新结果前冻结。原prepare输入依赖18个历史企业/滚动森林以及450份旧模拟工件，缺失时不得静默重训或替代。

```bash
python -m experiments.post_st_level_coverage_v1.analysis_pool --stage prepare --run RUN
/home/oliver/.local/bin/hetero-run RUN/analysis.py
```

核验analysis gate后：

```bash
python -m experiments.post_st_level_coverage_v1.analysis_pool --stage collect --run RUN
python -m experiments.post_st_level_coverage_v1.validate --run RUN
python -m experiments.post_st_level_coverage_v1.supplements --run RUN
python -m experiments.post_st_level_coverage_v1.plots --run RUN
python -m experiments.post_st_level_coverage_v1.report --run RUN
```

分析读取已固定森林，不重新拟合。pool输入与结果分目录，DATA_DIR/OUTPUT_DIR由调度器提供；任务独立输出，NFS发布核对SHA256。只在未运行的唯一目录执行冻结准备，不把这段复现命令误用于再次启动已完成实验。报告脚本中的结论针对本次冻结结果，其他研究设置须重新审阅文字。

## 表格索引

- enterprise_metrics / warnings / calibration_bins：年度、合并、各种子及集成，12/24个月两原因，raw诊断完整保留。
- enterprise_paired_intervals：1000次共同重采样的24个月主要性能差；正AUC/AP/捕获有利，负Brier有利；calibration_gap=预测减观察，正差表示概率上移而非必然更好。
- calibration_support / historical_diagnostics / domain_validation：数据支持、训练前校准效果、优化与域证书，不公开拟合参数。
- enterprise_function_validation / export_geometry / function_stability / leaf_support / search_cost：完整解释、实际网格、稳定性、真实事件支持与计算量。
- simulation_main / metric_summary / paired_changes / signal_summary / tree_weighted_identity：模拟主表、全部主次指标、配对差、真实信号恢复与逐树机制；csv.gz保存逐重复及各种子结果。
- frozen_input_hashes / data_array_hashes / artifact_hashes / reuse：来源、输入与模型哈希。simulation_threshold_sensitivity完整保存0.0025/0.005/0.01三个门槛。
- execution_receipt.json：拟合数量、复用、失败、数值状态与提交来源。Git最终提交与远端同步在交付时另外核验。
''',encoding='utf-8')
    receipt=dict(status='COMPLETE',experiment=NAME,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),private_run=str(run),
        repository='Egon749/XAI-RSF',branch='research/xai-rsf-post-st-prognosis-v1',
        design_source_commit='0a9e0dc3824ec239eaf372bcb05a84a6a4c23a29',fit_commit='c82f303f164bd9013be0bc720ceb4494fad6ce01',
        analysis_commit='df9d839f6110a4968739c61d4dd6adbd3ec975ed',reference_sha256=REFERENCE_SHA,reference_unchanged=True,
        pilot_tasks=2,formal_tasks=460,analysis_tasks=10,formal_tasks_succeeded=460,failed_fits=0,failed_pool_tasks=0,
        new_formal_forests=1356,new_simulation_forests=1350,new_enterprise_forests=6,new_pilot_forests=2,new_calibrators=4,
        reused_current_forests=6,reused_rolling_forests=12,reused_simulation_reference_forests=1350,baseline_reproduction_fits=0,
        training_rows_sum_folds=654,training_unique_companies=345,training_unique_termination_events=26,
        enterprise_test_companies=81,enterprise_24m_termination_events=32,
        exploratory=True,inherited_data_status='PROVISIONAL',new_2025_access=False,combined_calibration_wide_arm=False,
        failed_descriptive_postprocess_attempts=3,postprocess_root_causes=2,postprocess_resolved_without_refit=True,
        validation=validation,formal_pool=gate,
        decisions=dict(calibration='24m level/Brier improvement, remaining underestimation and pooled-ranking harm; not promoted',
            coverage='Partial large-sample interaction recovery; rare/pure/continuous limitations and2024 predictive harm; not promoted'),
        publication='Aggregate/code only. Final repository commit and origin equality verified separately at delivery.')
    (root/'execution_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (out/'public_artifact_hashes.json').write_text(json.dumps({str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file() and '__pycache__' not in str(p) and p.name!='public_artifact_hashes.json'},indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[2]);a=p.parse_args();report(a.run,a.repo)
