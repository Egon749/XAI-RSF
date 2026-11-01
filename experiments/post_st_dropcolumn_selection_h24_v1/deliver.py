"""Scientific figures and Chinese aggregate-only H24 follow-up report."""
from pathlib import Path
import argparse,datetime,json,shutil,re,subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from experiments.post_st_drop9_v1.deliver import table
from .fitting import *

def figures(run,parent):
    reports=run/'reports';out=reports/'figures';out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc').get_name(),'axes.unicode_minus':False,'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        fig.savefig(out/(name+'.png'),dpi=160,bbox_inches='tight');fig.savefig(out/(name+'.svg'),bbox_inches='tight');plt.close(fig)
    d=pd.read_csv(parent/'reports/DROP_COLUMN_ALL_83.csv');features=load(parent/'input/features.json')
    x=d[(d.horizon==24)&(d.cause==1)&d.mature&d.supported&(d.known_termination_events>=1)&(d.validation_target_events>=3)]
    rank=x.groupby(['outer_cutoff','feature_id']).delta_ap.mean().unstack(0);order=rank[2025].sort_values().index;yy=np.arange(83)
    fig,axes=plt.subplots(1,3,figsize=(16,24),sharey=True)
    for ax,year in zip(axes,[2023,2024,2025]):
        values=rank.loc[order,year].to_numpy();ax.barh(yy,values,color=np.where(values>=0,'#c66346','#3274a1'));ax.axvline(0,c='#777',lw=.8);ax.set_title(f'{year}截止前历史验证');ax.set_xlabel('删除后AP − 全83项AP');ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    axes[0].set_yticks(yy,[f'{j:02d} {features[j]["label"]}' for j in order],fontsize=8)
    fig.suptitle('24个月先退市：全83项删除后完整重训\n复用同一批真实森林；有效历史年度AP差等权均值，单项排名不等于最终选择',y=.998);fig.tight_layout(rect=(0,0,1,.98));save(fig,'all83_h24_rank')
    m=pd.read_csv(reports/'MODEL_COMPARISON.csv',dtype={'year':str,'seed':str});m=m[(m.seed=='ensemble')&(m.cause==1)&m.year.isin(['2023','2024','2025'])]
    fig,axes=plt.subplots(2,4,figsize=(16,7));colors=['#3274a1','#d58b3c','#498265']
    for row,h in enumerate([12,24]):
        for col,k in enumerate(['auc','ap','brier','captured']):
            ax=axes[row,col]
            for i,(label,name,color) in enumerate(zip(['full83','legacy74','selected_h24'],['83项','既有74项','24个月历史筛选'],colors)):
                z=m[(m.horizon==h)&(m.label==label)].set_index('year').reindex(['2023','2024','2025']);ax.bar(np.arange(3)+(i-1)*.24,z[k],width=.23,color=color,label=name)
            ax.set_xticks(range(3),['2023','2024','2025']);ax.set_title(f'{h}个月 '+{'auc':'AUC','ap':'AP','brier':'Brier（越低越好）','captured':'Top20捕获事件数'}[k]);ax.grid(axis='y',alpha=.2)
            if h==24:ax.text(.98,.98,'2025：NA',transform=ax.transAxes,ha='right',va='top',fontsize=8)
    axes[0,0].legend(fontsize=8);fig.suptitle('已暴露外层队列：选择完成后统一评价，未据此改变变量集合');fig.tight_layout();save(fig,'annual_comparison')
    st=pd.read_csv(reports/'FEATURE_SELECTION_STABILITY.csv');z=st.pivot(index='feature_id',columns='outer',values='conditional_inclusion_given_scan').reindex(range(83))
    fig,ax=plt.subplots(figsize=(8,22));cmap=plt.get_cmap('Blues').copy();cmap.set_bad('#dddddd');im=ax.imshow(z,cmap=cmap,vmin=0,vmax=1,aspect='auto')
    ax.set_xticks(range(3),['2023','2024','2025']);ax.set_yticks(range(83),[f'{j:02d} {features[j]["label"]}' for j in range(83)],fontsize=8);ax.set_title('条件保留频率：仅计实际执行完整筛选的重抽样\n灰色为无可估计重复；高频保留不等于不可替代')
    fig.colorbar(im,ax=ax,fraction=.04,pad=.05,label='条件保留频率');fig.tight_layout();save(fig,'selection_stability')
    cx=pd.read_csv(reports/'EXPLANATION_COMPLEXITY.csv');fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,metric,title in zip(axes,['features','interaction_functions','spatial_table_cells'],['输入变量数','非零二元交互数','完整空间单元数']):
        for i,(label,name,color) in enumerate(zip(['full83','legacy74','selected_h24'],['83项','既有74项','24个月历史筛选'],colors)):
            g=cx[cx.model==label].set_index('year').reindex([2023,2024,2025]);ax.bar(np.arange(3)+(i-1)*.24,g[metric],width=.23,label=name,color=color)
        ax.set_xticks(range(3),['2023','2024','2025']);ax.set_title(title);ax.grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=8);fig.suptitle('完整96树预测器的表示复杂度；时间/原因标量存储另表报告');fig.tight_layout();save(fig,'explanation_complexity')

def deliver(run,repo):
    parent=Path(load(run/'input/protocol.json')['parent_private_run']);public=repo/'experiments/post_st_dropcolumn_selection_h24_v1';v=load(run/'audit/validation.json');assert v['status']=='PASS'
    reports=run/'reports';decisions=load(run/'audit/selection_decisions.json');features=load(parent/'input/features.json')
    test_logs=list((run/'audit').glob('tests*.log'));latest=max(test_logs,key=lambda p:p.stat().st_mtime);test_text=latest.read_text();match=re.search(r'(\d+) passed',test_text)
    assert match and not re.search(r'\d+ failed',test_text);test_count=int(match.group(1))
    m=pd.read_csv(reports/'MODEL_COMPARISON.csv',dtype={'year':str,'seed':str});main=m[(m.seed=='ensemble')&(m.cause==1)]
    selected=pd.read_csv(reports/'SELECTED_FEATURES.csv');boot=pd.read_csv(reports/'BOOTSTRAP_SUPPORT_AND_SELECTION.csv');freq=pd.read_csv(reports/'FEATURE_SELECTION_STABILITY.csv')
    choices=pd.DataFrame([dict(cutoff=y,selected_features=len(d['keep']),removed='、'.join(features[j]['label'] for j in sorted(set(range(83))-set(d['keep']))) or '未删除',historical24_events=d['gate']['target_events'],support_passed=d['gate']['passed']) for y,d in decisions.items()])
    bs=boot.groupby('year').agg(replicates=('repeat','size'),support_passes=('support_passed','sum'),actual_selection_scans=('selection_scan_executed','sum'),reduced_replicates=('selected_features',lambda x:int((x<83).sum())),min_features=('selected_features','min'),mean_features=('selected_features','mean'),new_forests=('new_forests','sum')).reset_index()
    named=freq.merge(selected[~selected.selected][['outer','feature_id']],on=['outer','feature_id'])
    chosen_freq=named[['outer','feature_name','actual_selection_scans','unconditional_inclusion','conditional_inclusion_given_scan']].copy()
    chosen_freq['conditional_deletion_frequency']=1-chosen_freq.conditional_inclusion_given_scan
    cx=pd.read_csv(reports/'EXPLANATION_COMPLEXITY.csv');history=pd.read_csv(reports/'HISTORICAL_SELECTED_COMPARISON.csv');delta=pd.read_csv(reports/'PAIRED_DIFFERENCES.csv')
    schedules=pd.read_csv(reports/'FEATURE_SCHEDULE_DIAGNOSTIC.csv');single_schedule=schedules[schedules.comparison=='single']
    eligible_overlap=float(single_schedule.eligible_common_count_mean.mean())
    drop=pd.read_csv(parent/'reports/DROP_COLUMN_ALL_83.csv')
    valid=drop[(drop.horizon==24)&(drop.cause==1)&drop.mature&drop.supported&(drop.known_termination_events>=1)&(drop.validation_target_events>=3)]
    rank=valid.groupby(['outer_cutoff','feature_id']).delta_ap.mean().reset_index()
    rank_status=rank.groupby('outer_cutoff').agg(variables=('feature_id','size'),positive_mean_AP=('delta_ap',lambda x:int((x>0).sum())),negative_mean_AP=('delta_ap',lambda x:int((x<0).sum()))).reset_index()
    deleted_context=pd.read_csv(reports/'REMOVED_FEATURE_CONTEXT.csv')
    function_comparison=pd.read_csv(reports/'FUNCTION_STABILITY_COMPARISON.csv')
    prediction_variation=pd.read_csv(reports/'PREDICTION_STABILITY_COMPARISON.csv')
    description=f'''# 24个月主要目标：追加自动Drop-column实验

本轮由研究者在12个月支持门槛说明后明确追加。保留原协议、原83列模型和既有74列模型，单独冻结24个月主要选择目标。它不是把原先失败的12个月门槛事后改小，也没有改变森林的24个月训练随访上限。

## 实际完成

复用前轮全83项单删和相关组真实模型，追加{v['actual_new_forests']}次森林拟合，{v['tasks']}个成功任务，失败{v['failed_fits']}。每个森林32树，42/101/202三种子平均形成完整96树预测器。新候选均完整重训；没有删除已经学好的非零函数充当重训。

其中2024最终{v['extra_outer_forests_reproducing_existing_subset']}个森林额外复现了已保存的同配置候选；12/24个月预测最大差异{v['existing_subset_forecast_reproduction_error']:.3e}。这部分已计入实际计算成本，未充当额外种子或独立证据。未达支持条件时直接引用上一轮相同公司抽样下的83列回退模型，未把复用记作新增拟合。

资源核验发现，本次oliver作业内管理进程的实际允许集合包含32个逻辑CPU，与此前“仅24个物理工作槽”的环境说明不同；没有据此宣称48个线程都在训练，也未修改Slurm或cgroup。部分任务完成后，仅在本作业内把已有任务动态迁到空闲物理P核，减少SMT争用与E核尾部等待，不增加任务、不重启拟合、不改数学计算。初始调度记录与后续核绑定分别保留在任务回执和affinity_adjustments.jsonl；耗时属于这些实际硬件调度条件，不能当成严格的算法速度对照。

{table(choices)}

2023的历史有效24个月验证只有2020一队列、6例退市，按冻结条件保留83列。2024、2025分别有13、22例，允许进入筛选，但这只是执行条件，不代表统计功效充分。所有选择使用各自年初前的历史前向预测。2023–2025已暴露的外层结果仅在选择决定固定后描述性评价。

## 内部选择证据

24个月主要目标为有效验证年度AP等权均值、各年度Top20捕获合计；置信下界、12个月表现、两原因Brier、种子方向、交互集合稳定性与空间格数共同保护。12个月守门指标仅使用可计算的历史队列，不要求其通过另一个事件数门槛；因此这些保护仍有小样本局限。

初始全83项和相关组候选均保留。最多12个删除单元进入递归池；每次删除后对剩余候选重新拟合。随后执行冻结的恢复/替换核查。完整接受、拒绝和停止记录见RECURSIVE_ELIMINATION_PATH.csv与SELECTION_CONTRASTS.csv。初始单项排名不直接作为连续删除顺序。

主样本在接受第一步后，两年各重新比较11个剩余删除候选，均未通过全部保护条件；恢复/替换候选也未通过。因此实际停在82/81列，并未继续追求65、55或更小规模。GUARD_REJECTION_SUMMARY.csv逐项列出停止原因；一个候选可以同时违反多条约束，不能把各行次数相加当成独立失败任务。

公开候选表分别列出指标约束通过与审计编码约束通过；两者均通过也不代表最终入选，还受有限候选池和递归顺序约束。审计7列在自动删除中作为完整块处理，单独删除其某一列只作诊断。最终变量集合以SELECTED_FEATURES.csv为准。

{table(history[(history.cause==1)][['outer','model','features','horizon','cohorts','auc','ap','brier','captured','spatial_cells']])}

这部分是模型选择用过的内部结果，不能当成独立性能提升。相同种子配对了公司bootstrap，但删列会改变均衡覆盖及分裂搜索；增量属于整个固定训练流程，不等于变量的因果贡献。

全83项单删的24个月年内AP均值方向如下。若多数甚至全部单删都上升，不能据此宣布多数甚至全部变量无用。

{table(rank_status)}

针对已有配置补做了不重训的候选调度核查：83种单删、3种子、32棵配对树中，每树25个可用变量平均只有{eligible_overlap:.3f}个与83列模型相同。删除1列会带来平均{25-eligible_overlap:.3f}个候选位置的变化。该诊断解释了比较中还包含候选调度变化，不能证明所有预测变化都由调度造成。详见FEATURE_SCHEDULE_DIAGNOSTIC.csv；它不参与筛选，也未改变冻结算法。

## 外层描述性比较

{table(main[['year','label','features','horizon','auc','ap','brier','risk_mean','observed','checked','captured','target_events','supported']])}

2025年观察截止2026-06-30，24个月随访不支持，结果仍为NA；合并24个月只包括2023/2024。各年度先平均三个种子的公司CIF；合并AUC/AP/Brier重新计算，Top20使用各年度向上取整的名额后合计。未知结局未当作正常。12个月与恢复原因、全部种子见MODEL_COMPARISON.csv。

以下为“24个月筛选模型减83列”的配对差值；AUC/AP正值为有利，Brier负值为有利。500次公司抽样按年度分层、两模型共用索引、预测与IPCW权重固定。这不是完整训练/筛选不确定性，多个区间不作选择性显著宣称。

{table(delta[delta.cause==1])}

## 真正重训的选择稳定性

{table(bs)}

与前轮50次回退模型的稳定性不同，本轮在支持和保护指标可计算时，重新执行了全量初始候选及相同递归算法；每个候选均重新拟合三种子森林。重抽样按历史入组年分层、公司为单位，重复公司的权重保留，门槛按独立公司计数，不把重复抽到的事件当作新增支持。相关组字典与三个森林种子固定，未覆盖重新估计组定义和种子分布的不确定性。

未达24个月支持、或必要12个月保护指标不可计算的重复按规则回退83列。无条件保留率包含这些回退，不能直接解释为变量不可替代；另外报告实际完成筛选条件下的保留频率。主样本删除项的重复表现如下。

{table(chosen_freq)}

所有83项频率、主效应/交互幅度与形状稳定性均保留。这里做的是选择稳定性评价，没有再根据外层结果或事后选定的频率阈值修改最终变量集合。固定50次，不额外扩展至100次。

此外，把自动筛选流程和原83列流程在同样50次公司抽样上配对比较。下表`component_spread_l2`为函数间方差之和在共同历史样本上取平均后开平方，缺失函数按零对齐，不除以会变化的函数数量。它衡量函数表示的整体重抽样波动，越低越稳定；`mean_anchor_distance_l2`则与各流程自己的主样本模型比较，两者含义不同。分位数只是50个扰动结果的经验范围，不是均值的95%置信区间。

{table(function_comparison[(function_comparison.horizon==24)&(function_comparison.cause==1)])}

{table(prediction_variation[(prediction_variation.horizon==24)&(prediction_variation.cause==1)])}

预测波动同时报告平均风险水平，避免把接近常数、幅度很小造成的低波动称为解释优势。完整逐重复配对值见PAIRED_FUNCTION_STABILITY.csv；2023两流程均回退83，应当一致。函数形状与幅度在相同历史参考记录上比较，不能据此宣称所有未观察到的变量组合都稳定。

## 内生解释及数值性质

{table(cx[['year','model','features','main_functions','interaction_functions','spatial_table_cells','scalar_table_cells','distinct_thresholds']])}

每个新raw/收缩森林及完整合并预测器都通过q≤2、两原因CIF合法性、保存/重载与完整函数重构核验。全训练/评价样本、所有原生时间点、两原因的最大重构误差为{v['max_reconstruction_error']:.3e}，门槛1e-10。没有clipping或事后代理模型。原模型和输入哈希未变。

完整测试{test_count}项通过（含独立R核查）。LOCAL_EXPLANATION_CHANGES.csv比较原83列和最终候选在同样8条历史训练记录上的完整贡献变化与CIF变化；完整个体向量保留私有目录。贡献变化可能相互抵消，所以预测变化小不等于解释变化小。

函数对比使用同一年度原训练集的经验边际乘积参考；删列后仍按原变量ID对齐，缺失项贡献作为零比较。函数出现率指该参考样本上的非零幅度，不等于所有空间格上的结构存在。主效应和二阶贡献是CIF贡献，不能解读成因果影响。精确重构证明表示忠实，不能证明模型学到真实作用或变量不可替代。

复杂度表的非零函数在两原因、全部原生时点检出；幅度与稳定性表按具体原因和时限计算，两种计数口径不混用。即使删除了披露陈旧度这一预测字段，原始可得日仍作为PIT审计元数据保留，不能因此忽略信息时间核验。

被删变量在原83列模型中的24个月贡献如下，数值单位为CIF概率（乘100才是百分点）。某个原因/时点上贡献为零，不意味着该变量在另一个原因、时点或另一个重训模型中也无用；完整两原因/两时点见REMOVED_FEATURE_CONTEXT.csv。

{table(deleted_context[(deleted_context.horizon==24)&(deleted_context.cause==1)])}

企业、预测、完整模型/函数、原始阈值及个体解释全部保留在私有目录`{run}`。森林使用无损gzip保存；完整函数文件仍可独立求值。哈希与逐模型数值核验表压缩为CSV.gz公开，未上传模型参数。沿用旧conservative C与原始披露版本PIT的PROVISIONAL限制。

## 研究判断

1. 找到了符合预设历史规则的82/81列候选，尚未证明其普遍优于83或74列；须同时看12/24个月、年内/合并排序、概率误差和稳定性。
2. 单项与相关组重训提供算法条件下的贡献证据，小样本和相关替代仍限制不可替代性判断。
3. “被本轮规则接受删除”不等于已证明安全删除。主样本选择与50次重抽样频率应一起解读。
4. 完整精确解释得以保留；复杂度与稳定性的增量按实际表格判断，不能从输入列变少直接推断。
5. Drop-column属于训练前选择，不是q2森林内部机制的新数学贡献。本轮不把它自动纳入默认训练流程。
6. 不替换当前模型。升级仍需要独立、未参与选择且随访充分的时间外证据及研究者批准。

![24个月全变量删除排名](figures/all83_h24_rank.png)
![年度性能](figures/annual_comparison.png)
![条件选择稳定性](figures/selection_stability.png)
![完整解释复杂度](figures/explanation_complexity.png)
'''
    (public/'RESULTS_ZH.md').write_text(description)
    (public/'METHOD_INNOVATION_DECISION_ZH.md').write_text(description[description.index('## 研究判断'):description.index('![24个月')])
    (public/'DATA_AND_PROTOCOL_AUDIT_ZH.md').write_text('''# 24个月追加实验的数据与协议核对

沿用[原实验完整数据审计](../post_st_dropcolumn_selection_v1/DATA_AND_PROTOCOL_AUDIT_ZH.md)，没有重建样本、扩大结局范围、移动起点或改用后来修订的年报值。旧conservative C的83列、首次财务/审计ST后下一交易日开盘前起点、先最终退市与先完整恢复的竞争风险定义、原始版本PIT的PROVISIONAL限制均保留。

三个年初训练集分别为309/345/390家公司，24个月训练随访内已知退市14/26/49例；对应评价公司36/45/53家，训练与同年度评价公司无重叠。这里的训练事件总数不同于筛选所用历史前向验证事件数。后者在24个月主要目标下为6/13/22例，2023只有一个有效队列，仍不允许自动删列。2024/2025的历史有效队列分别为2020–2021与2020–2022；2025截止可用的2023年12个月结局只作次要保护。

追加协议在24个月选择前冻结。原12个月优先协议及结果保留；本轮改变选择目标为24个月AP/Top20，并保留12个月、两原因Brier、解释复杂度和种子方向保护。森林本身原已使用24个月训练上限，未改训练标签、树结构设置或AJ估计方式。

训练进程的评价输入只含X和sample_id，不含外层结局。历史标签分别截止对应外层年初；自动选择与复核读取的是这些内部前向标签。外层2023–2025结局已暴露，本轮只在各截止选择决定固定后统一作描述性比较。2025截至2026-06-30不支持24个月指标，保留NA。

所有83项单删及相关组初始候选按哈希复用既有真实拟合；递归变化与支持条件内的公司重抽样候选实际重新拟合。重抽样按公司和历史入组年进行，重复抽取权重保留，事件支持按独立公司计数。相关组字典和三个森林种子固定，因此稳定性结论带有这些条件。未通过主要支持或必要次要指标不可计算时回退83列，回退保留率不能当作不可替代性证据。

每次候选删列保持25个变量/树，但会改变候选比例和均衡调度；不将Drop-column差值解释成纯粹的变量信息贡献。审计7列在自动删除中作为整块处理。原模型、所有源输入及冻结代码快照哈希均在最终validation.json中核验。私有模型和逐公司预测不上传；执行数量、来源复用及额外复现均见execution_receipt.json和EXECUTION_TASKS.csv。
''')
    d=pd.read_csv(parent/'reports/DROP_COLUMN_ALL_83.csv');d=d[d.horizon==24].rename(columns={'selection_status':'parent_h12_selection_status'})
    d['selection_status']=np.where(d.outer_cutoff.eq(2023),'H24_SUPPORT_GATE_CLOSED','H24_AGGREGATE_GUARDS_REPORTED_SEPARATELY')
    d.to_csv(reports/'DROP_COLUMN_ALL_83_H24.csv',index=False)
    for p in reports.iterdir():
        if p.is_file() and (p.suffix=='.csv' or p.name.endswith('.csv.gz')):
            if p.name in ['SELECTION_CONTRASTS.csv','INITIAL_H24_CONTRASTS.csv']:
                rows=pd.read_csv(p).rename(columns={'approved':'metric_guards_passed'})
                rows['audit_encoding_allowed']=rows.keep.apply(lambda s:len(set(json.loads(s))&set(range(76,83))) in [0,7])
                rows['passes_metric_and_encoding_guards']=rows.metric_guards_passed&rows.audit_encoding_allowed
                rows.to_csv(public/p.name,index=False)
            else:shutil.copy2(p,public/p.name)
    shutil.copytree(reports/'figures',public/'figures',dirs_exist_ok=True)
    aud=public/'audit';aud.mkdir(exist_ok=True)
    for p in (run/'audit').iterdir():
        if p.is_file() and (p.suffix in ['.json','.log'] or p.name=='affinity_adjustments.jsonl') and not p.name.startswith('registry_'):shutil.copy2(p,aud/p.name)
    receipt=dict(status='COMPLETED_EXPLORATORY_H24_SELECTION',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),private_run=str(run),parent_private_run=str(parent),
        original_h12_result_preserved=True,current_model_replaced=False,validation=v,selection_features={y:len(d['keep']) for y,d in decisions.items()},
        full_tests_passed=test_count,test_log=latest.name,test_log_sha256=sha(latest),independent_R_checks=True,
        analysis_source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        analysis_source_hashes={p.name:sha(p) for p in public.glob('*.py')},
        protocol_sha256=sha(run/'input/protocol.json'),code_snapshots={p.name:load(p)['commit'] for p in (run/'audit').glob('code_snapshot*.json')},
        public_output_hashes={p.name:sha(p) for p in public.iterdir() if p.name.endswith(('.csv','.csv.gz'))},all_original_vintage_limits_retained=True)
    write(public/'execution_receipt.json',receipt);write(run/'audit/execution_receipt.json',receipt)
    print(json.dumps(dict(status=receipt['status'],new_forests=v['actual_new_forests'],selected=receipt['selection_features'])),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--action',choices=['figures','report'],required=True);a=p.parse_args();parent=Path(load(a.run/'input/protocol.json')['parent_private_run'])
    figures(a.run,parent) if a.action=='figures' else deliver(a.run,Path(__file__).resolve().parents[2])
