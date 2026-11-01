"""Aggregate scientific figures, Chinese report and execution receipt."""
import argparse
import datetime
import json
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from experiments.post_st_dropcolumn_selection_v1.common import load, sha, write
from experiments.post_st_drop9_v1.deliver import table
from .search import FULL, LEGACY_REMOVED, anchors

NAME='post_st_feature_combination_search_v1'
LABELS={'full83':'原83项','legacy74':'既有74项','previous_auto':'上一轮自动','selected_search':'本轮组合搜索','historical75':'历史75项'}
COLORS={'full83':'#3176aa','legacy74':'#d28b34','previous_auto':'#999999','selected_search':'#378168'}


def figures(run):
    report=run/'reports';out=report/'figures';out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc').get_name(),
        'axes.unicode_minus':False,'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        fig.savefig(out/f'{name}.png',dpi=170,bbox_inches='tight');fig.savefig(out/f'{name}.svg',bbox_inches='tight');plt.close(fig)
        p=out/f'{name}.svg';p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
    m=pd.read_csv(report/'MODEL_COMPARISON.csv',dtype={'year':str,'seed':str});m=m[(m.seed=='ensemble')&(m.cause==1)]
    fig,axes=plt.subplots(2,4,figsize=(17,7))
    for row,h in enumerate([12,24]):
        for col,key in enumerate(['auc','ap','brier','captured']):
            ax=axes[row,col]
            for i,label in enumerate(COLORS):
                z=m[(m.horizon==h)&(m.model==label)].set_index('year').reindex(['2023','2024','2025'])
                ax.bar(np.arange(3)+(i-1.5)*.19,z[key],width=.18,label=LABELS[label],color=COLORS[label])
            ax.set_xticks(range(3),['2023','2024','2025']);ax.set_title(f'{h}个月 '+{'auc':'AUC','ap':'AP','brier':'Brier ↓','captured':'Top20捕获事件数'}[key]);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
            if h==24:ax.text(.98,.96,'2025：随访不足，NA',transform=ax.transAxes,ha='right',va='top',fontsize=8)
    axes[0,0].legend(fontsize=8);fig.suptitle('冻结历史组合搜索后的外层结果｜同一年度先平均三种子，再计算指标');fig.tight_layout();save(fig,'annual_12_24_comparison')
    c=pd.read_csv(report/'CANDIDATE_SCORES.csv');decisions=load(run/'audit/selection_decisions.json')
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    for col,year in enumerate([2023,2024,2025]):
        z=c[c.year==year]
        for row,h in enumerate([12,24]):
            ax=axes[row,col];ax.scatter(z.historical_cells,z[f'mean_AP{h}'],s=12,alpha=.35,c=np.where(z.approved,'#378168','#999999'))
            ordered=z.sort_values(['historical_cells',f'mean_AP{h}'],ascending=[True,False]);front=[];best=-np.inf
            for r in ordered.itertuples():
                value=getattr(r,f'mean_AP{h}')
                if np.isfinite(value) and value>best:front.append((r.historical_cells,value));best=value
            if front:ax.plot(*zip(*front),color='#ab6036',lw=1)
            marked=[('83',list(FULL),'#3176aa'),('74/75',list(anchors(year)[1]),'#d28b34'),('选定',decisions[str(year)]['keep'],'#8d4156')]
            for name,keep,color in marked:
                q=z[z.keep.apply(lambda x:json.loads(x)==keep)]
                if len(q):
                    r=q.iloc[0];ax.scatter([r.historical_cells],[r[f'mean_AP{h}']],marker='*',s=110,c=color,zorder=5,label=name)
            ax.set_title(f'{year}截止 / 历史{h}个月'+('（支持不足，仅诊断）' if year==2023 else ''));ax.set_xlabel('完整函数空间单元数：对应历史队列均值');ax.set_ylabel('年内AP等权均值');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);fig.suptitle('历史预测—表示复杂度前沿｜绿点通过指标保护，支持门槛另判；历史最优不等于未来最优');fig.tight_layout();save(fig,'historical_search_frontier')
    cx=pd.read_csv(report/'EXPLANATION_COMPLEXITY.csv');fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,key,title in zip(axes,['features','interaction_functions','spatial_table_cells'],['输入变量数','非零二元交互数','完整空间单元数']):
        for i,label in enumerate(COLORS):
            z=cx[cx.model==label].set_index('year').reindex([2023,2024,2025]);ax.bar(np.arange(3)+(i-1.5)*.19,z[key],width=.18,color=COLORS[label],label=LABELS[label])
        ax.set_xticks(range(3),['2023','2024','2025']);ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].legend(fontsize=8);fig.suptitle('完整96树预测器：输入数、函数数和空间复杂度分别计量');fig.tight_layout();save(fig,'explanation_complexity')
    st=pd.read_csv(report/'FIXED_SET_FUNCTION_STABILITY.csv');fig,axes=plt.subplots(2,2,figsize=(10,7))
    for row,h in enumerate([12,24]):
        for col,order in enumerate([1,2]):
            ax=axes[row,col]
            for i,label in enumerate(['full83','selected_search']):
                z=st[(st.horizon==h)&(st.cause==1)&(st.order==order)&(st.model==label)].set_index('year').reindex([2023,2024,2025])
                ax.bar(np.arange(3)+(i-.5)*.3,z.component_spread_l2,width=.28,label=LABELS[label],color=COLORS[label])
            ax.set_xticks(range(3),['2023','2024','2025']);ax.set_title(f'{h}个月 '+('主效应' if order==1 else '二元交互'));ax.set_ylabel('函数整体重抽样波动（CIF概率单位）↓');ax.grid(axis='y',alpha=.2)
    axes[0,0].legend();fig.suptitle('50次配对公司重训｜固定选定特征集，未重复整个组合搜索\n训练参考样本；2025的24个月仅为函数诊断，不代表结局评价');fig.tight_layout();save(fig,'fixed_set_function_stability')


def report(run):
    out=run/'reports';protocol=load(run/'input/protocol.json');validation=load(run/'audit/validation.json')
    assert validation['status']=='PASS'
    parent=Path(protocol['parent_private_run']);features=load(parent/'input/features.json');decisions=load(run/'audit/selection_decisions.json')
    c=pd.read_csv(out/'CANDIDATE_SCORES.csv');m=pd.read_csv(out/'MODEL_COMPARISON.csv',dtype={'year':str,'seed':str})
    main=m[(m.seed=='ensemble')&(m.cause==1)].copy();cx=pd.read_csv(out/'EXPLANATION_COMPLEXITY.csv')
    st=pd.read_csv(out/'FIXED_SET_FUNCTION_STABILITY.csv');diff=pd.read_csv(out/'PAIRED_DIFFERENCES.csv',dtype={'year':str})
    choices=[];reference_rows=[];coverage=[]
    for y,d in decisions.items():
        year=int(y);z=c[c.year==year]
        choices.append(dict(year=year,candidates=len(z),features=len(d['keep']),historical_events=d['gate']['target_events'],
            support_passed=d['gate']['passed'],removed='、'.join(features[j]['label'] for j in sorted(set(FULL)-set(d['keep']))) or '无',status=d['status']))
        coverage.append(dict(year=year,evaluated_candidates=len(z),minimum_features=int(z.features.min()),maximum_features=int(z.features.max()),
                             guard_passes=int(z.approved.sum()),initial=int((z.first_stage==0).sum()),round1=int((z.first_stage==1).sum()),round2=int((z.first_stage==2).sum())))
        for label,keep in [('full83',list(FULL)),('legacy74',[j for j in FULL if j not in LEGACY_REMOVED]),
                           ('historical_anchor',list(anchors(year)[1])),('selected_search',d['keep'])]:
            q=z[z.keep.apply(lambda x:json.loads(x)==keep)]
            if len(q):
                r=q.iloc[0];reference_rows.append(dict(year=year,candidate=label,features=len(keep),mean_AP12=r.mean_AP12,
                    mean_AP24=r.mean_AP24,delta_AP24=r.delta_ap24,lower90=r.ap24_lower90,
                    delta_capture24=r.delta_capture24,delta_cells=r.spatial_cell_change,
                    approved=r.approved,failed_guards=r.failed_guards,provenance=r.provenance))
    pd.DataFrame(reference_rows).to_csv(out/'ANCHOR_CANDIDATE_COMPARISON.csv',index=False)
    pd.DataFrame(coverage).to_csv(out/'SEARCH_COVERAGE.csv',index=False)
    choices=pd.DataFrame(choices)
    main['model']=main.model.map(LABELS)
    primary=main[['year','model','features','horizon','auc','ap','brier','risk_mean','observed','captured','target_events','supported']]
    paragraphs=[]
    for year in ['2023','2024','2025','pooled']:
        for h in [12,24]:
            z=m[(m.year==year)&(m.horizon==h)&(m.cause==1)&(m.seed=='ensemble')].set_index('model')
            a=z.loc['full83'];b=z.loc['selected_search'];old=z.loc['legacy74']
            if not b.supported:
                paragraphs.append(f'- {year}、{h}个月：随访支持不足，AUC/AP/Brier/完整窗口捕获保留NA。')
            else:
                paragraphs.append(f'- {year}、{h}个月：83→本轮，AUC {a.auc:.4f}→{b.auc:.4f}，AP {a.ap:.4f}→{b.ap:.4f}，Brier {a.brier:.6f}→{b.brier:.6f}，Top20 {int(a.captured)}/{int(a.target_events)}→{int(b.captured)}/{int(b.target_events)}。既有74为AUC {old.auc:.4f}、AP {old.ap:.4f}、Brier {old.brier:.6f}、捕获{int(old.captured)}/{int(old.target_events)}。')
    complexities=[]
    for year in [2023,2024,2025]:
        z=cx[cx.year==year].set_index('model');a=z.loc['full83'];b=z.loc['selected_search']
        complexities.append(f'- {year}：主效应{int(a.main_functions)}→{int(b.main_functions)}，二元交互{int(a.interaction_functions)}→{int(b.interaction_functions)}，完整空间单元{int(a.spatial_table_cells)}→{int(b.spatial_table_cells)}（{100*(b.spatial_table_cells/a.spatial_table_cells-1):+.2f}%）。')
    text=f'''# 显式83/74候选与多起点特征组合搜索

本轮完成真实重训与统一评价；当前83项研究参考保持。研究性质为已暴露2023–2025队列上的探索性机制比较，继承旧conservative C原始财务版本PIT的PROVISIONAL限制。

## 实际搜索与候选时间

83项与历史可用74/75项每轮都作为起点。2023的近3个月总市值变化诊断尚未成立，因此只用当时8项形成256种组合；2024/2025分别9项512种组合。完整74在2023没有被注入初始先验；对所有原变量对称枚举的通用邻域可以独立生成同一集合，其来源在候选表中保留。

随后两轮、每轮四个起点，枚举所有变量增删、审计整块和历史相关组代表替换。允许不合格中间结构继续探索，最终接受仍比较原83、使用同一保护条件。只在8/9字段子空间完整枚举，不能称作全83变量全局最优。没有依据外层结果改变搜索轮数、候选范围或最终集合。

{table(pd.DataFrame(coverage))}

{table(choices)}

2023支持不足时即使历史诊断找到高AP组合，正式选择仍回退83。2024/2025历史有效退市13/22例，达到操作门槛不代表充分功效；这些历史模型在拟合时只有2/3/6例退市（2025的12个月保护另含2023模型）。候选评分复用这批小样本，扩展搜索增加选择过拟合风险。

## 83与74确实参与了什么比较

{table(pd.DataFrame(reference_rows))}

83的增量保护未通过仅因为自身增量为0，它始终是合法的保守回退。74必须接受同样保护，并非凭外层表现强制获胜。历史AP用于选择，不当作独立验证增益；“已搜索可行候选中最高”也不表示外层最优。两种模型都可能包含相关可替代信息，删列同时改变均衡覆盖与每树候选比例，不能解释为变量的因果增量。

## 12/24个月统一结果

每年度先平均42/101/202三个种子的逐公司CIF，再算AUC/AP/IPCW Brier。合并指标重新计算；Top20按各年ceil(0.2N)名额后合计。12个月合并含2023–2025，24个月仅含2023/2024。观察截止2026-06-30，2025完整24个月不支持，未知结局没有当正常。

{table(primary)}

{chr(10).join(paragraphs)}

## 配对不确定性

{table(diff[(diff.cause==1)&(diff.comparison=='selected_search-minus-full83')][['year','cohorts','horizon','metric','difference','lower95','upper95','undefined']])}

1000次按评价年分层、公司为单位的共同重抽样，模型预测及IPCW固定；与74及上一轮自动的差值也完整保存。区间仅反映条件评价波动，不包含完整训练/组合搜索/模型开发不确定性，也没有多重比较确认性保证。跨零不等于性能等效。

## 内生解释与稳定性

{table(cx)}

{chr(10).join(complexities)}

{table(st[(st.horizon==24)&(st.cause==1)][['year','model','order','component_spread_l2','mean_anchor_distance_l2','mean_nonzero_jaccard_to_anchor']])}

这50次公司重抽样是在固定最终特征集合下重新训练，使用与原83相同抽样、同一原训练集的经验边际乘积参考，缺失函数以0对齐。它没有重新执行扩大后的特征搜索，不能作为变量选择频率或无条件算法稳定性证据。主效应/二阶函数的形状、幅度、预测波动与风险均值分别报告，低幅度不自动视为可靠解释。

相关变量的边际乘积参考会包含现实中稀少或未出现的组合；这是统一函数归一化的约定，不证明统计可辨识性或经济因果关系。历史复杂度保护也不保证随后重训的外层模型复杂度必然下降。

完整96树预测器与每个raw/收缩模型均验证q≤2、CIF合法、保存重载和全部训练/评价行、原生时间点、两个原因的显式完整函数重构；最大误差{validation['max_reconstruction_error']:.3e}，门槛1e-10。没有裁剪预测、代理模型或删掉已训练函数。CIF贡献不是因果影响。空间单元含缺失状态；乘时间和原因维度后的标量储量另列。

## 计算及可复现性

新增{validation['new_forest_fits']}个32树森林、{validation['bundles']}个三种子完整预测器、{validation['tasks']}个任务；拟合失败{validation['failed_fits']}，回传失败{validation['failed_publications']}。Slurm作业{','.join(validation['jobs'])}。另外复用{validation['reused_bundles']}个不同的历史三种子预测器、对应{validation['reused_seed_forests']}个种子森林，模型/函数/预测文件均按原清单核对哈希，见REUSED_MODELS。复用数量不计入新增拟合。原公司bootstrap哈希配对通过，特征覆盖随删列变化如实保留。

四个试运行包含12次真实拟合，正式搜索继续复用；每项最多4个完整预测器，避免上一轮巨大单任务回传。两节点单线程任务通过现有Slurm允许集合管理P/E核与SMT，没有修改cgroup或声称固定48训练线程。时间与内存见EXECUTION_TASKS/FIT_BUNDLES。

准备时修正了旧诊断CSV的字段名读取；最初独立R测试因R库路径设置错误失败，恢复既有正确路径后完整测试通过。原失败日志保留，未放宽测试或改森林公式。这些环境/准备失败不计为真实拟合失败。

私有模型、阈值、预测与个体解释保留在`{run}`。公开内容仅代码、协议、字段名单、汇总指标、哈希及诊断。

## 研究解释边界

扩大搜索解决了上一轮遗漏完整74组合及单步停止过早的限制；预测收益仍必须看实际年度、合并、校准、捕获与函数稳定性共同判断。训练前特征组合选择不是森林q2精确分解的新数学机制。当前参考模型不自动替换；新的未参与选择且随访充分的时间外证据仍是升级依据。

![年度12/24个月](figures/annual_12_24_comparison.png)
![历史搜索前沿](figures/historical_search_frontier.png)
![解释复杂度](figures/explanation_complexity.png)
![固定集合函数稳定性](figures/fixed_set_function_stability.png)
'''
    (out/'RESULTS_ZH.md').write_text(text)
    (out/'PAPER_METHODS_ZH.md').write_text('''# 可用于论文的方法描述

在保持二阶路径约束、深度4、共享分位数切点、均衡每树25输入以及强度10的直接父节点Aalen–Johansen收缩不变的条件下，采用受历史信息约束的多起点特征组合搜索。对各外层预测截止点，完整枚举当时已有冗余诊断字段的保留/删除组合，并从全变量与历史去冗余方案出发进行两轮邻域扩展；中间结构不要求单调改善。所有候选使用相同历史前向验证队列和三个配对随机种子，最终按24个月年内AP选择，并施加12个月排序、两个原因Brier、预警捕获及完整函数复杂度保护。信息支持不足时回退全变量模型。

变量集合固定后，评价2023–2025年12个月与有随访支持的24个月先退市/先恢复竞争结局。年度预测先平均三种子CIF，合并指标在合并预测上重算。以公司为单位、按年度分层的1000次固定预测重抽样估计描述性差值区间；另用50次配对公司重训评价固定特征集合下的函数和预测稳定性。每个模型均在全部训练/评价样本、原生时点与两个原因上核验完整二阶函数重构及竞争风险概率约束。研究队列已暴露，结果属于探索性比较，不代表未接触独立验证或全变量组合的全局最优。
''')
    test=run/'audit/tests_final_delivery.log'
    textlog=test.read_text();match=re.search(r'(\d+) passed',textlog)
    assert match and not re.search(r'\d+ (failed|skipped)',textlog)
    receipt=dict(status='COMPLETED_EXPLORATORY_COMBINATION_SEARCH',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        private_run=str(run),validation=validation,selected_counts={y:len(d['keep']) for y,d in decisions.items()},
        candidate_counts={str(y):int(len(g)) for y,g in c.groupby('year')},full_tests_passed=int(match.group(1)),
        test_log_sha256=sha(test),current_reference_replaced=False,full_search_bootstrap_performed=False,
        fixed_feature_set_bootstrap_repeats=50,protocol_sha256=sha(run/'input/protocol.json'),
        analysis_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[2],text=True).strip(),
        output_hashes={p.name:sha(p) for p in out.glob('*.csv*')},
        analysis_source_hashes={p.name:sha(p) for p in Path(__file__).resolve().parent.glob('*.py')})
    write(out/'execution_receipt.json',receipt)


def publish(run):
    public=Path(__file__).resolve().parent;report=run/'reports'
    assert load(report/'execution_receipt.json')['status']=='COMPLETED_EXPLORATORY_COMBINATION_SEARCH'
    for p in report.iterdir():
        if p.is_dir():shutil.copytree(p,public/p.name,dirs_exist_ok=True)
        else:shutil.copy2(p,public/p.name)
    audit=public/'audit';audit.mkdir(exist_ok=True)
    for pattern in ['code_snapshot.json','preparation.json','preparation_incidents.json','environment.json',
                    'validation.json','selection_decisions.json','centers_*.json','gate_*.json','tests*.log']:
        for p in (run/'audit').glob(pattern):shutil.copy2(p,audit/p.name)
    for p in public.rglob('*'):
        if p.is_file():
            assert 'PRIVATE' not in p.name and p.suffix not in ['.npz','.npy','.pkl'],p
    print(f'Aggregate results published into {public}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--action',choices=['figures','report','publish'],required=True)
    args=parser.parse_args();globals()[args.action](args.run)
