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
    verdict=(pub/'INTERPRETATION_ZH.md').read_text()
    for p in(r/'reports').iterdir():assert p.suffix in ['.csv','.json']and not(pub/p.name).exists();shutil.copy2(p,pub/p.name)
    d=pd.read_csv(pub/'METRICS_ALL.csv');d=d[d.version.eq('shrunk')].copy();d['seed']=d.seed.astype(str)
    d['capture']=d.apply(lambda x:f'{int(x.captured)}/{int(x.target_events)}'if pd.notna(x.captured)else'NA',axis=1)
    target=d[d.cause.eq(1)];sections=[];frames={};rankings=[]
    for y in YEARS:
        rows=[]
        for s in [*map(str,SEEDS),'ensemble512','reference96']:
            row=dict(seed=s,trees=512 if s=='ensemble512'else 96 if s=='reference96'else 32,reused=s.isdigit()and int(s)in REUSED_SEEDS)
            for h in [12,24]:
                z=target[target.scope.eq(str(y))&target.seed.eq(s)&target.months.eq(h)].iloc[0]
                for m in ['auc','ap','brier','capture','risk_mean','observed','supported','reason']:row[f'{m}_h{h}']=z[m]
            rows.append(row)
        frame=pd.DataFrame(rows);frames[y]=frame;frame.to_csv(pub/f'TERMINATION_{y}_BY_SEED.csv',index=False)
        cols={'seed':'种子/组合','auc_h12':'12月AUC','ap_h12':'12月AP','brier_h12':'12月Brier','capture_h12':'12月Top20','auc_h24':'24月AUC','ap_h24':'24月AP','brier_h24':'24月Brier','capture_h24':'24月Top20'}
        sections.append(f'## {y}年：逐种子及组合退市结果\n\n'+table(frame,cols))
        for h in [12,24]:
            if y==2025 and h==24:continue
            z=frame.iloc[:len(SEEDS)].copy();z['rank']=z[f'auc_h{h}'].rank(method='min',ascending=False).astype(int)
            for row in z.sort_values(['rank','seed']).to_dict('records'):
                rankings.append(dict(year=y,months=h,rank=row['rank'],seed=row['seed'],auc=row[f'auc_h{h}'],ap=row[f'ap_h{h}'],brier=row[f'brier_h{h}'],capture=row[f'capture_h{h}']))
    pd.DataFrame(rankings).to_csv(pub/'AUC_RANKINGS.csv',index=False)
    fn=load(pub/'function_validation.json');cost=pd.read_csv(pub/'COMPUTE_COST.csv')
    (pub/'RESULTS_ZH.md').write_text(f'''# 指定16种子及512树组合实验

{verdict}

## 训练和评价口径

种子{SEEDS}各32树，2023/2024/2025各自训练快照不变。复用{REUSED_SEEDS}的9个已核验raw森林及其收缩版本，新种子{NEW_SEEDS}实际拟合39个raw森林、1248棵新树；2024/2570完整32树试运行计入39次。另导出三个年度的16种子等权512树组合，raw/收缩分别合并，未额外拟合或再次收缩。原42/101/202的96树单独保留。复用与新训练在COMPUTE_COST.csv逐项区分，复用不虚报本轮拟合耗时。

固定74输入、q≤2、深度4、min_leaf10/min_groups5/min_events3、共享5分位切点、均衡25变量池、mtry75、signed_sum原因权重(1,1)、子节点退市支持0或≥2家、直接父节点AJ收缩10。入组公告下一交易日开盘前起点，先最终退市原因1、先恢复原因2，训练随访上限24个日历月，仅用自身年度1月1日前信息。

以下为收缩版本；每个公司的评价终点按12/24个日历月计算。Top20按年ceil(0.2n)确定，名额2023/2024/2025为8/9/11，捕获分母为该窗口全部目标事件数。2025H24随访不足，性能NA。组合是先平均各公司预测再计算指标，不是平均AUC/AP。不同树预算的结果不能仅归因于种子。

{chr(10).join(sections)}

## 完整数值、解释与不确定性

METRICS_ALL.csv包含全部种子、512及96组合、raw/收缩、两个原因、逐年度/合并结果。合并AUC/AP/Brier重新计算，Top20先按年选定再合计；2025H24不进入24月合并。2023H12恢复无事件时AUC/AP为NA。AUC_RANKINGS.csv按未四舍五入的AUC排序，真正并列共享名次，不据此更换模型。

共{fn['representations']}份表示核验完整重构：96份单种子表示（78份新、18份已核验复用）以及6份512组合、6份原96参照。新模型与新组合均对全部训练/评价样本、原生时间、两个原因做完整函数/独立逐项求和/保存重载核验；复用表示核对原验证及输入输出哈希。最大完整重构误差{fn['max_full_reconstruction']:.3e}，独立逐项求和误差{fn['max_explicit_sum']:.3e}。q≤2、CIF非负/单调/两原因和≤1通过，未裁剪。新增模型独立重放分裂候选、叶AJ及raw直接父节点收缩。三个复用种子和原96参照的指标全部复现。

NUMERICAL_COMPLEXITY.csv、FUNCTION_STABILITY.csv、PREDICTION_VARIABILITY.csv、LEAF_SUPPORT.csv记录函数个数、空间/标量单元、跨种子函数和预测变化、真实叶支持。共同函数使用同年度训练边际乘积参考。精确解释不等于解释简单、统计可辨识、稳定或因果作用；收缩不增加真实观察事件。

PAIRED_DIFFERENCES_95.csv保留1000次原公司级按年度分层配对重抽样，analysis_seed2026110201，各种子/512组合减96参照，记录无定义次数。条件区间固定预测，不包含训练与选择不确定性。名单在看过已暴露种子结果后指定，不能称无偏种子抽样或独立验证。本轮不按年拼接最佳种子、不搜索组合权重、不自动替换原模型。

## 执行与保存

39次新拟合累计{cost.fit_seconds.sum():.2f}任务计时秒，完整过程峰值RSS最大{cost.peak_rss_MiB.max():.1f}MiB；并行墙钟/排队/节点分配另列execution_receipt.json，累计任务计时不是总墙钟。现有Slurm池负责CPU与亲和性，输入和模型哈希核对通过。完整pytest含独立R通过。公开仅代码/配置/汇总/哈希，企业预测、函数参数、切点保留唯一私有目录。原输入时钟/PIT和变量选择限制沿用；原模型、CURRENT_POST_ST_MODEL.json、主模型文件夹、核心实现和旧报告均未改动。
''')
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':9})
    fig,axes=plt.subplots(2,3,figsize=(16,6),sharex=True)
    for col,y in enumerate(YEARS):
        f=frames[y].iloc[:len(SEEDS)];ens=frames[y].iloc[len(SEEDS)];ref=frames[y].iloc[len(SEEDS)+1]
        for row,m in enumerate(['auc','ap']):
            ax=axes[row,col]
            for h,off,color,marker in [(12,-.1,'#4477aa','o'),(24,.1,'#ee7733','s')]:
                if y==2025 and h==24:continue
                ax.scatter(np.arange(len(SEEDS))+off,f[f'{m}_h{h}'],color=color,marker=marker,s=22,label=f'{h}月：单种子32树')
                ax.axhline(ens[f'{m}_h{h}'],color=color,ls='-',alpha=.7,label=f'{h}月：512树组合')
                ax.axhline(ref[f'{m}_h{h}'],color=color,ls='--',alpha=.7,label=f'{h}月：原96树')
            ax.set_title(f'{y}年 · {m.upper()}'+('（24月NA）'if y==2025 else''));ax.grid(alpha=.2);ax.set_xticks(range(len(SEEDS)),SEEDS,rotation=60)
    axes[0,0].legend(fontsize=6);fig.suptitle('指定16种子｜每种子32树与等权512树组合｜先最终退市',fontsize=13);fig.tight_layout(rect=[0,0,1,.94])
    for ext in ['png','pdf']:fig.savefig(pub/f'SEEDS16_OVERVIEW.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    log=r/'audit/pytest_final.log';assert '289 passed'in log.read_text()and'skipped'not in log.read_text();shutil.copy2(log,pub/'pytest_final.log')
    write(pub/'execution_receipt.json',dict(status='COMPLETED_EXPLORATORY_SIXTEEN_SEEDS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_commit=SOURCE_COMMIT,
        analysis_code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),worker_code_commit=load(r/'audit/code_snapshot.json')['commit'],branch='research/xai-rsf-requested-seeds16-v1',
        private_run=str(r),seeds=SEEDS,new_seeds=NEW_SEEDS,reused_seeds=REUSED_SEEDS,trees_per_seed=32,individual_models=48,forest_fits=39,reused_forests=9,new_trees=1248,calibration_fits=0,ensemble_exports=6,ensemble_trees=512,reference96_preserved=True,
        gates={s:load(r/f'audit/gate_{s}.json')for s in ['pilot','formal','ensemble']},validations={s:load(pub/f'{s}_validation.json')for s in ['data','metric','function']},
        input_manifest_sha256=sha(r/'audit/input_hashes.json'),ensemble_input_manifest_sha256=sha(r/'audit/ensemble_input_hashes.json'),source_manifest_sha256=sha(r/'audit/source_hashes.json'),
        tests='289passed,independentR included,no skips;inheritedEBM visualization warning',failures=load(r/'reports/FAILURES_AND_RECOVERY.json'),environment=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__),
        public_output_hashes={p.name:sha(p)for p in pub.iterdir()if p.is_file()and p.name!='execution_receipt.json'}))
if __name__=='__main__':main()
