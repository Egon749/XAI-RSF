"""Publish aggregate seed results only; no model selection."""
import datetime,platform,subprocess,shutil
import pandas as pd
import scipy,sklearn,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.post_st_tree_count_v1.report import table
from .common import *

def main():
    r=run_path();pub=REPO/'experiments'/NAME
    for n in ['data','metric','function']:assert load(r/f'reports/{n}_validation.json')['status']=='PASS'
    for p in(r/'reports').iterdir():
        assert p.suffix in ['.csv','.json']and not(pub/p.name).exists();shutil.copy2(p,pub/p.name)
    d=pd.read_csv(pub/'METRICS_ALL.csv');d=d[d.version.eq('shrunk')].copy();d['seed']=d.seed.astype(str)
    d['capture']=d.apply(lambda x:f'{int(x.captured)}/{int(x.target_events)}'if pd.notna(x.captured)else'NA',axis=1)
    target=d[d.cause.eq(1)];frames={};sections=[]
    for y in YEARS:
        rows=[]
        for s in [*map(str,SEEDS),'reference96']:
            row=dict(seed=s,trees=96 if s=='reference96'else 32)
            for h in [12,24]:
                z=target[target.scope.eq(str(y))&target.seed.eq(s)&target.months.eq(h)].iloc[0]
                for m in ['auc','ap','brier','capture','risk_mean','observed','supported','reason']:row[f'{m}_h{h}']=z[m]
            rows.append(row)
        frame=pd.DataFrame(rows);frames[y]=frame;frame.to_csv(pub/f'TERMINATION_{y}_BY_SEED.csv',index=False)
        cols={'seed':'种子/参照','auc_h12':'12月AUC','ap_h12':'12月AP','brier_h12':'12月Brier','capture_h12':'12月Top20','auc_h24':'24月AUC','ap_h24':'24月AP','brier_h24':'24月Brier','capture_h24':'24月Top20'}
        sections.append(f'## {y}年：每个种子的退市结果\n\n'+table(frame,cols))
    stats=pd.read_csv(pub/'SEED_METRIC_SUMMARY.csv');st=stats[stats.version.eq('shrunk')&stats.cause.eq(1)&stats.scope.ne('pooled')&stats.metric.isin(['auc','ap','brier','captured'])]
    fn=load(pub/'function_validation.json');cost=pd.read_csv(pub/'COMPUTE_COST.csv');verdict=(pub/'INTERPRETATION_ZH.md').read_text()
    report=f'''# 20种子、每种子32树：完整结果

{verdict}

## 协议和执行

固定74输入、q≤2、深度4、min_leaf10/min_groups5/min_events3、共享5切点、均衡25变量池、mtry75、signed_sum原因权重(1,1)、退市支持保护0或≥2独立公司、直接父节点AJ收缩10。每种子32树，20个种子为{SEEDS}。每年使用自身1月1日前冻结信息，训练随访上限24个日历月，下一交易日开盘前为起点，先最终退市为原因1、先恢复为原因2。

60个年度×种子模型中51个新拟合、9个既有32树模型原样复用。首个完整32树数值试运行作为该格正式结果，没有再拟合一次；合计1632棵新增树。原3种子各32棵平均形成的96树模型单列reference96。各行20种子的预算仅为32树，不能把它们当成20份96树结果，也没有将20份预测平均成新的640树模型。既有96树模型和原3种子指标逐值复现。

主表为父节点收缩版本，预测时限以各公司的12/24个日历月计算。raw版本、恢复原因、单种子跨年合并结果均保留于METRICS_ALL.csv及YEAR_*_RESULTS.csv/POOLED_RESULTS.csv。合并AUC/AP/Brier重新计算；Top20先按年度ceil(0.2n)确定名额再相加。2025年24个月性能NA，未知结局不当正常。无事件的2023年12个月恢复AUC/AP为NA。

{chr(10).join(sections)}

## 种子间分布

{table(st,{'scope':'年度','months':'月数','metric':'指标','n_valid':'可计算种子数','mean':'均值','sd':'标准差','minimum':'最小','maximum':'最大'})}

这里的均值是20个单独模型指标的平均，不是平均预测后再计算的集成性能；2.5%/97.5%分位数也不是置信区间。20个种子共享相同训练/评价公司，反映算法随机性，不能视作20个独立时间外队列。种子清单包含原来的3个固定种子，其他17个在新结果前确定，未筛选有利种子，也不按不同年度/时限拼接赢家。

## 解释与数值验证

本轮120份raw/收缩表示及6份原96树参照均通过全部训练+评价样本、所有原生时间和两个原因的验证。最大完整重构误差{fn['max_full_reconstruction']:.3e}，独立逐项求和最大误差{fn['max_explicit_sum']:.3e}；最大路径变量数{fn['max_path_variables']}。CIF非负、随时间非减、两原因之和≤1、保存重载均通过，没有裁剪。每个森林重放候选分裂，并用独立AJ核对叶节点和raw直接父节点收缩。

NUMERICAL_COMPLEXITY.csv含各种子的主效应、二元函数、空间单元、标量规模、导出时间等；FUNCTION_STABILITY.csv含各年190个种子对的集合重合度、共同函数相关性和幅度差异。共同函数使用相同年度训练边际乘积参考，相关性仅基于共同项，不代表全部解释都稳定。LEAF_SUPPORT.csv反映真实叶支持，收缩不会增加真实观察事件。所有贡献均是CIF函数贡献，不是因果效应。

## 不确定性与局限

PAIRED_DIFFERENCES_95.csv采用原1000次公司级、按年分层的配对重抽样（seed2026110201），每个32树模型减固定96树参照；条件于已固定预测，不包含训练公司重抽样、此前设计选择或种子筛选不确定性。逐项记录不可计算次数。所有有利/不利结果均保留，不用某一个种子的高分升级模型。

2023–2025均已暴露，只作探索性随机稳定性分析。沿用原始披露版本/PIT和74变量删选时间的限制；没有重建输入或结局。此前每种子96树准备草案在用户明确32树后封存，草案无拟合/Slurm作业。本轮使用新私有目录。CURRENT_POST_ST_MODEL.json、主模型文件夹、核心实现与历史结果未改动。

## 计算与复现

试运行和正式任务均通过两节点Slurm提交并核对计算完成、发布完成及输出哈希。新拟合时间合计{cost.fit_seconds.sum():.2f}进程秒（不是墙钟耗时）；峰值RSS最大{cost.peak_rss_MiB.max():.1f}MiB。逐任务时间、节点及作业号见COMPUTE_COST.csv，试运行与正式作业端到端耗时见execution_receipt.json。入口见README.md；公司数据、预测、阈值及模型参数均在私有目录。
'''
    (pub/'RESULTS_ZH.md').write_text(report)
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':9})
    fig,axes=plt.subplots(2,3,figsize=(15,7),sharex=True)
    for col,y in enumerate(YEARS):
        f=frames[y].iloc[:-1];ref=frames[y].iloc[-1]
        for row,m in enumerate(['auc','ap']):
            ax=axes[row,col]
            for h,offset,color,marker in [(12,-.12,'#4477aa','o'),(24,.12,'#ee7733','s')]:
                if y==2025 and h==24:continue
                ax.scatter(np.arange(20)+offset,f[f'{m}_h{h}'],color=color,marker=marker,s=22,label=f'{h}月：单种子32树')
                ax.axhline(ref[f'{m}_h{h}'],color=color,ls='--',alpha=.7,label=f'{h}月：原96树')
            ax.set_title(f'{y}年 · {m.upper()}'+('（24月NA）'if y==2025 else''));ax.grid(alpha=.2);ax.set_xticks(range(20),SEEDS,rotation=90)
    axes[0,0].legend(fontsize=8);fig.suptitle('20个随机种子，每个32树｜退市预测｜虚线为原96树组合',fontsize=14);fig.supxlabel('固定种子编号（不代表时间或优劣顺序）');fig.tight_layout(rect=[0,.03,1,.96])
    for ext in ['png','pdf']:fig.savefig(pub/f'SEED_RESULTS_OVERVIEW.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    log=r/'audit/pytest_final.log';assert '280 passed'in log.read_text()and'skipped'not in log.read_text();shutil.copy2(log,pub/'pytest_final.log')
    write(pub/'execution_receipt.json',dict(status='COMPLETED_EXPLORATORY_SEED_TEST',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_commit='d2e0be83f4ddb7f9a33170808c032a842e3f6435',analysis_code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        worker_code_commit=load(r/'audit/code_snapshot.json')['commit'],branch='research/xai-rsf-seed-stability20-v1',private_run=str(r),seeds=SEEDS,trees_per_seed=32,
        models=60,forest_fits=51,reused_forests=9,new_trees=1632,calibration_fits=0,pilot_included_in_60=True,twenty_seed_ensemble_created=False,reference96_preserved=True,
        gates={s:load(r/f'audit/gate_{s}.json')for s in ['pilot','formal']},validations={s:load(pub/f'{s}_validation.json')for s in ['data','metric','function']},
        input_manifest_sha256=sha(r/'audit/input_hashes.json'),source_manifest_sha256=sha(r/'audit/source_hashes.json'),tests='280passed,independentR included,no skips;inheritedEBM warning',
        failures=[],environment=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__),
        current_pointer_sha256=sha(REPO/'CURRENT_POST_ST_MODEL.json'),public_output_hashes={p.name:sha(p)for p in pub.iterdir()if p.is_file()and p.name!='execution_receipt.json'}))
if __name__=='__main__':main()
