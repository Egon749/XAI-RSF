"""Aggregate report for eight user-requested refits."""
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
    order=[42,101,202,1009,503,17,419,300]
    for y in YEARS:
        rows=[]
        for s in [*map(str,order),'reference96']:
            row=dict(seed=s,trees=96 if s=='reference96'else 32)
            for h in [12,24]:
                z=target[target.scope.eq(str(y))&target.seed.eq(s)&target.months.eq(h)].iloc[0]
                for m in ['auc','ap','brier','capture','risk_mean','observed','supported','reason']:row[f'{m}_h{h}']=z[m]
            rows.append(row)
        frame=pd.DataFrame(rows);frames[y]=frame;frame.to_csv(pub/f'TERMINATION_{y}_BY_SEED.csv',index=False)
        cols={'seed':'种子/参照','auc_h12':'12月AUC','ap_h12':'12月AP','brier_h12':'12月Brier','capture_h12':'12月Top20','auc_h24':'24月AUC','ap_h24':'24月AP','brier_h24':'24月Brier','capture_h24':'24月Top20'}
        sections.append(f'## {y}年：逐种子先退市结果\n\n'+table(frame,cols))
    fn=load(pub/'function_validation.json');cost=pd.read_csv(pub/'COMPUTE_COST.csv');repro=pd.read_csv(pub/'HISTORICAL_REPRODUCTION.csv')
    verdict=(pub/'INTERPRETATION_ZH.md').read_text()
    report=f'''# 指定八种子实际重训结果

{verdict}

## 配置与执行口径

名单为{SEEDS}，每种子32树，每年各自重新训练，合计24个raw森林、768棵树；另派生24个直接父节点AJ收缩版本。没有把旧工件复用写成新训练。2024/seed300的32树试运行计入正式矩阵，另外23项另行完成。原96树组合（42/101/202各32树、预测等权平均）仅作固定参照，不改动。

固定74输入、q≤2、深度4、min_leaf10/min_groups5/min_events3、共享5切点、均衡25变量池、mtry75、signed_sum原因权重(1,1)、子节点退市支持0或≥2家、直接父节点AJ收缩强度10。每年自身1月1日前的冻结信息，训练随访最多24个日历月。下一交易日开盘前为起点，先最终退市原因1、先恢复原因2。

原来7个种子×3年×raw/收缩共{len(repro)}份重训表示与历史核对，树结构完全一致，预测最大差{repro.max_prediction_error.max():.3e}，所有既有指标也复现。seed300此前无对应模型，不宣称其为重现。

以下主表为收缩版。12/24个月使用公司自己的日历终点。Top20按每年ceil(0.2n)确定，2023/2024/2025名额分别8/9/11。2025年24个月随访不足，所有性能NA。原96树为更大预算的组合参照，不等同于某个32树种子。

{chr(10).join(sections)}

## 完整性与辅助交付

METRICS_ALL.csv和YEAR_*_RESULTS.csv保存两个原因、raw/收缩、全部种子和固定参照。POOLED_RESULTS.csv使用合并逐公司预测重新计算AUC/AP/Brier，Top20先按年筛查再合计。2025H24不进入合并24月；未知结局不作正常，2023年12月恢复无事件时AUC/AP保持NA。

所有{fn['representations']}份表示（本轮48份及6份原96树参照）在全部训练/评价样本、所有原生时间、两个原因上验证。最大完整重构误差{fn['max_full_reconstruction']:.3e}，独立逐项求和误差{fn['max_explicit_sum']:.3e}。q≤2、CIF非负/单调/两原因总和≤1和保存重载均通过，无裁剪。全森林重放分裂候选并独立核对叶AJ与raw直接父节点收缩。

NUMERICAL_COMPLEXITY.csv、FUNCTION_STABILITY.csv、PREDICTION_VARIABILITY.csv、LEAF_SUPPORT.csv记录解释复杂度、种子间函数/预测变化和真实叶支持。共同函数使用同年度训练边际乘积参考；精确分解不等于交互稳定、真实作用恢复或因果效应。

PAIRED_DIFFERENCES_95.csv采用原1000次公司级按年分层重抽样、analysis_seed2026110201，每种子减原96树，条件于固定预测。此名单在暴露20种子AUC排序后由用户指定，区间不校正选择过程，不能作为独立确认性优越证据。SEED_METRIC_SUMMARY.csv仅描述指定集合，不代表无偏随机种子总体。没有选取某年最优种子拼接表现，没有创建八种子组合或更换当前模型。

## 计算与审计

Slurm两节点完成并核验计算/发布与哈希。24次拟合累计{cost.fit_seconds.sum():.2f}进程秒，峰值RSS最大{cost.peak_rss_MiB.max():.1f}MiB；墙钟与排队时间单独记录在execution_receipt.json，不能把进程秒当墙钟。完整pytest含独立R通过。原模型、CURRENT_POST_ST_MODEL.json、主模型文件夹、核心实现和历史报告均不改动。企业输入/预测/阈值/参数保留私有，公开仅汇总。PIT与74变量选择的原有限制沿用。
'''
    (pub/'RESULTS_ZH.md').write_text(report)
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':10})
    fig,axes=plt.subplots(2,3,figsize=(12,6),sharex=True)
    for col,y in enumerate(YEARS):
        f=frames[y].iloc[:-1];ref=frames[y].iloc[-1]
        for row,m in enumerate(['auc','ap']):
            ax=axes[row,col]
            for h,off,color,marker in [(12,-.1,'#4477aa','o'),(24,.1,'#ee7733','s')]:
                if y==2025 and h==24:continue
                ax.scatter(np.arange(8)+off,f[f'{m}_h{h}'],color=color,marker=marker,s=28,label=f'{h}月：32树种子')
                ax.axhline(ref[f'{m}_h{h}'],color=color,ls='--',alpha=.7,label=f'{h}月：原96树')
            ax.set_title(f'{y}年 · {m.upper()}'+('（24月NA）'if y==2025 else''));ax.grid(alpha=.2);ax.set_xticks(range(8),order,rotation=45)
    axes[0,0].legend(fontsize=7);fig.suptitle('指定8个种子实际重训｜每种子32树｜先最终退市',fontsize=14);fig.tight_layout(rect=[0,0,1,.94])
    for ext in ['png','pdf']:fig.savefig(pub/f'REQUESTED_SEEDS_OVERVIEW.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    log=r/'audit/pytest_final.log';assert '281 passed'in log.read_text()and'skipped'not in log.read_text();shutil.copy2(log,pub/'pytest_final.log')
    write(pub/'execution_receipt.json',dict(status='COMPLETED_EXPLORATORY_REQUESTED_RERUN',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_commit='1b8f8fd28c07b0618dc945131e376a58256ddf55',analysis_code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),worker_code_commit=load(r/'audit/code_snapshot.json')['commit'],branch='research/xai-rsf-requested-seeds-v1',private_run=str(r),seeds=SEEDS,trees_per_seed=32,models=24,forest_fits=24,reused_forests=0,new_trees=768,calibration_fits=0,pilot_included_in_24=True,historical_reproduction_checks=42,eight_seed_ensemble_created=False,reference96_preserved=True,
        gates={s:load(r/f'audit/gate_{s}.json')for s in ['pilot','formal']},validations={s:load(pub/f'{s}_validation.json')for s in ['data','metric','function']},input_manifest_sha256=sha(r/'audit/input_hashes.json'),source_manifest_sha256=sha(r/'audit/source_hashes.json'),tests='281passed,independentR included,no skips;inheritedEBM warning',failures=[],environment=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__),current_pointer_sha256=sha(REPO/'CURRENT_POST_ST_MODEL.json'),public_output_hashes={p.name:sha(p)for p in pub.iterdir()if p.is_file()and p.name!='execution_receipt.json'}))
if __name__=='__main__':main()
